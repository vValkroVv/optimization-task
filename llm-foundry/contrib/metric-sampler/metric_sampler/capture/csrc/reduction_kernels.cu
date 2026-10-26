#include "cuda_host_functions.h"
#include "kernel_common.cuh"

namespace {

__device__ double load_double_words(const int32_t* words) {
    const unsigned long long low = static_cast<uint32_t>(words[0]);
    const unsigned long long high = static_cast<uint32_t>(words[1]);
    return __longlong_as_double(static_cast<long long>(low | (high << kWordBits)));
}

// The sum over a float32 population of -(x * log(x + 1e-12)), each term and the sum in fp64 like x.double() in
// torch; the block partials are summed in block order so the result is deterministic.
__global__ void entropy_partials_kernel(
    const float* __restrict__ x, int64_t element_count, double* __restrict__ partials) {
    __shared__ double warp_sums[kWarps];
    double sum = 0.0;
    const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < element_count; i += step) {
        const double value = static_cast<double>(x[i]);
        sum -= __dmul_rn(value, log(value + kEntropyEpsilon));
    }
    sum = warp_reduce_sum(sum);
    if (threadIdx.x % kWarpSize == 0) {
        warp_sums[threadIdx.x / kWarpSize] = sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double block_sum = 0.0;
        for (int warp_index = 0; warp_index < kWarps; ++warp_index) {
            block_sum += warp_sums[warp_index];
        }
        partials[blockIdx.x] = block_sum;
    }
}

// One warp: lane l sums partials l, l + 32, ... in order, then a fixed shuffle tree.
__global__ void sum_partials_kernel(const double* __restrict__ partials, int blocks, double* __restrict__ out) {
    double sum = 0.0;
    for (int i = threadIdx.x; i < blocks; i += kWarpSize) {
        sum += partials[i];
    }
    sum = warp_reduce_sum(sum);
    if (threadIdx.x == 0) {
        out[0] = sum;
    }
}

// Segments of rows given by host starts (row counts per local expert), so no copy of the counts
// reaches the device; at most kMaxSegments of them.
constexpr int kMaxSegments = 256;
struct SegmentStarts {
    int64_t start[kMaxSegments + 1];
};

// Sum of squares of every row of each segment in fp64: block (b, s) takes rows b, b + gridDim.x, ... of segment
// s, a warp per row (squares of 16-bit sources are exact in fp64), each lane summing its share and the lanes
// meeting once at the end, and writes its partial, which segment_finish_kernel sums in block order.
template <typename T>
__global__ void segment_sumsq_kernel(
    const T* __restrict__ x, int64_t row_stride, int64_t row_length, SegmentStarts segments,
    double* __restrict__ partials) {
    __shared__ double warp_sums[kWarps];
    constexpr int kVectorElements = kVectorBytes / sizeof(T);
    const int warp = static_cast<int>(threadIdx.x / kWarpSize);
    const int lane = static_cast<int>(threadIdx.x % kWarpSize);
    const int64_t end = segments.start[blockIdx.y + 1];
    double total = 0.0;
    for (int64_t row = segments.start[blockIdx.y] + static_cast<int64_t>(blockIdx.x) * kWarps + warp; row < end;
         row += static_cast<int64_t>(gridDim.x) * kWarps) {
        const T* row_ptr = x + row * row_stride;
        if (row_length % kVectorElements == 0 && reinterpret_cast<uintptr_t>(row_ptr) % kVectorBytes == 0) {
            for (int64_t vector_index = lane; vector_index < row_length / kVectorElements; vector_index += kWarpSize) {
                VectorLoad<T, kVectorElements> chunk =
                    reinterpret_cast<const VectorLoad<T, kVectorElements>*>(row_ptr)[vector_index];
#pragma unroll
                for (int j = 0; j < kVectorElements; ++j) {
                    double value = static_cast<double>(static_cast<float>(chunk.elements[j]));
                    total += value * value;
                }
            }
        } else {
            for (int64_t column = lane; column < row_length; column += kWarpSize) {
                double value = static_cast<double>(static_cast<float>(row_ptr[column]));
                total += value * value;
            }
        }
    }
    total = warp_reduce_sum(total);
    if (lane == 0) {
        warp_sums[warp] = total;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double block_total = 0.0;
        for (int warp_index = 0; warp_index < kWarps; ++warp_index) {
            block_total += warp_sums[warp_index];
        }
        partials[static_cast<int64_t>(blockIdx.y) * gridDim.x + blockIdx.x] = block_total;
    }
}

// One warp per segment: lane l sums the segment's partials l, l + 32, ... in order, then a fixed
// shuffle tree; the sum is stored in the segment's float64 words, or added to the value they hold.
__global__ void segment_finish_kernel(
    const double* __restrict__ partials, int blocks, int32_t* __restrict__ words, bool accumulate) {
    double sum = 0.0;
    for (int i = threadIdx.x; i < blocks; i += kWarpSize) {
        sum += partials[static_cast<int64_t>(blockIdx.x) * blocks + i];
    }
    sum = warp_reduce_sum(sum);
    if (threadIdx.x == 0) {
        int32_t* target = words + 2 * static_cast<int64_t>(blockIdx.x);
        store_double_words(target, accumulate ? load_double_words(target) + sum : sum);
    }
}

// Column sums of a float32 [rows, columns] tensor in fp64, like x.sum(0, dtype=torch.float64): block b sums its
// run of consecutive rows in order, and column_finish_kernel sums the block partials in a fixed order.
__global__ void column_partials_kernel(
    const float* __restrict__ x, int64_t row_stride, int64_t rows, int columns, int64_t rows_per_block,
    double* __restrict__ partials) {
    const int64_t first = static_cast<int64_t>(blockIdx.x) * rows_per_block;
    const int64_t last = min(rows, first + rows_per_block);
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        double sum = 0.0;
#pragma unroll 8
        for (int64_t row = first; row < last; ++row) {
            sum += static_cast<double>(x[row * row_stride + column]);
        }
        partials[static_cast<int64_t>(blockIdx.x) * columns + column] = sum;
    }
}

// One warp per column: lane l sums the column's partials l, l + 32, ... in order, then a fixed
// shuffle tree.
__global__ void column_finish_kernel(
    const double* __restrict__ partials, int blocks, int columns, double* __restrict__ out) {
    double sum = 0.0;
    for (int i = threadIdx.x; i < blocks; i += kWarpSize) {
        sum += partials[static_cast<int64_t>(i) * columns + blockIdx.x];
    }
    sum = warp_reduce_sum(sum);
    if (threadIdx.x == 0) {
        out[blockIdx.x] = sum;
    }
}

}  // namespace

void segment_sumsq_into(
    torch::Tensor matrix, std::vector<int64_t> counts, torch::Tensor words, int64_t offset, bool accumulate) {
    // Any number of rows, none included: segments without rows sum to zero.
    TORCH_CHECK(matrix.is_cuda() && matrix.dim() == 2, "matrix must be a 2D CUDA tensor");
    matrix = matrix.contiguous();
    TORCH_CHECK(!counts.empty() && counts.size() <= static_cast<size_t>(kMaxSegments),
                "segment_sumsq_into takes between one and ", kMaxSegments, " segments");
    SegmentStarts segments{};
    for (size_t i = 0; i < counts.size(); ++i) {
        TORCH_CHECK(counts[i] >= 0, "segment row counts must be nonnegative");
        segments.start[i + 1] = segments.start[i] + counts[i];
    }
    TORCH_CHECK(segments.start[counts.size()] == matrix.size(0), "segment row counts must cover every row");
    const int count = static_cast<int>(counts.size());
    TORCH_CHECK(words.device() == matrix.device() && words.scalar_type() == at::kInt && words.is_contiguous(),
                "words must be the recorder's contiguous int32 buffer on matrix's device");
    TORCH_CHECK(offset >= 0 && offset + 2 * static_cast<int64_t>(count) <= words.numel(),
                "the segments' words are outside the buffer");
    const at::cuda::OptionalCUDAGuard device_guard(matrix.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int blocks = static_cast<int>(std::max<int64_t>(
        1, static_cast<int64_t>(sm_count_for_device(matrix.get_device())) * kBlocksPerSm / count));
    auto scratch = torch::empty({static_cast<int64_t>(count) * blocks}, matrix.options().dtype(torch::kFloat64));
    double* partials = scratch.data_ptr<double>();
    const dim3 grid(blocks, count);
    const int64_t row_length = matrix.size(1), row_stride = matrix.stride(0);
    switch (matrix.scalar_type()) {
        case at::kBFloat16:
            segment_sumsq_kernel<at::BFloat16><<<grid, kBlockThreads, 0, stream>>>(
                matrix.data_ptr<at::BFloat16>(), row_stride, row_length, segments, partials);
            break;
        case at::kHalf:
            segment_sumsq_kernel<at::Half><<<grid, kBlockThreads, 0, stream>>>(
                matrix.data_ptr<at::Half>(), row_stride, row_length, segments, partials);
            break;
        case at::kFloat:
            segment_sumsq_kernel<float><<<grid, kBlockThreads, 0, stream>>>(
                matrix.data_ptr<float>(), row_stride, row_length, segments, partials);
            break;
        default:
            TORCH_CHECK(false, "activation_ext.segment_sumsq_into: unsupported dtype ", matrix.scalar_type());
    }
    segment_finish_kernel<<<count, kWarpSize, 0, stream>>>(partials, blocks, words.data_ptr<int32_t>() + offset,
                                                           accumulate);
}

torch::Tensor column_sums(torch::Tensor matrix) {
    check_rows(matrix);
    TORCH_CHECK(matrix.scalar_type() == at::kFloat, "column_sums takes float32 rows");
    const at::cuda::OptionalCUDAGuard device_guard(matrix.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int64_t rows = matrix.size(0), columns = matrix.size(1);
    const int64_t target_blocks = static_cast<int64_t>(sm_count_for_device(matrix.get_device())) * kBlocksPerSm;
    const int64_t rows_per_block = ceil_div(rows, target_blocks);
    const int64_t blocks = ceil_div(rows, rows_per_block);
    // Block partials, then one sum per column.
    auto scratch = torch::empty({blocks * columns + columns}, matrix.options().dtype(torch::kFloat64));
    double* partials = scratch.data_ptr<double>();
    column_partials_kernel<<<blocks, kBlockThreads, 0, stream>>>(
        matrix.data_ptr<float>(), matrix.stride(0), rows, static_cast<int>(columns), rows_per_block, partials);
    column_finish_kernel<<<columns, kWarpSize, 0, stream>>>(partials, static_cast<int>(blocks),
                                                            static_cast<int>(columns), partials + blocks * columns);
    return scratch.narrow(0, blocks * columns, columns);
}

torch::Tensor entropy_sum(torch::Tensor values) {
    TORCH_CHECK(values.is_cuda() && values.scalar_type() == at::kFloat && values.is_contiguous() && values.numel() > 0,
                "values must be a non-empty contiguous float32 CUDA tensor");
    const at::cuda::OptionalCUDAGuard device_guard(values.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int64_t element_count = values.numel();
    const int64_t grid = std::min<int64_t>(
        ceil_div(element_count, kBlockThreads),
        static_cast<int64_t>(sm_count_for_device(values.get_device())) * kBlocksPerSm);
    auto scratch = torch::empty({grid + 1}, values.options().dtype(torch::kFloat64));
    double* partials = scratch.data_ptr<double>();
    entropy_partials_kernel<<<grid, kBlockThreads, 0, stream>>>(values.data_ptr<float>(), element_count, partials);
    sum_partials_kernel<<<1, kWarpSize, 0, stream>>>(partials, static_cast<int>(grid), partials + grid);
    return scratch[grid];
}
