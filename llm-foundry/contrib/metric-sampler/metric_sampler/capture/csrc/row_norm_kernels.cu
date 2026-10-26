#include "cuda_host_functions.h"
#include "kernel_common.cuh"

namespace {

// Between these sums of squares of a row the fp32 square root of its norm is a normal number.
constexpr double kNormalSumOfSquaresMin = 1.2e-38;
constexpr double kNormalSumOfSquaresMax = 3.0e38;

__device__ __forceinline__ double nan_max(double left, double right) {
    return isnan(left) || isnan(right) ? NAN : fmax(left, right);
}

__device__ __forceinline__ float nan_max(float left, float right) {
    return isnan(left) || isnan(right) ? NAN : fmaxf(left, right);
}

// Takes one observation's [amax, sum of row norms] into an accumulator slot: the maximum propagates NaN like
// torch.maximum and the sum adds the observation first, as `new + previous`.
__device__ __forceinline__ void accumulate_observation(const double* stats, double* accumulator) {
    accumulator[kAmaxStat] = nan_max(accumulator[kAmaxStat], stats[kAmaxStat]);
    accumulator[kRowNormSumStat] = stats[kRowNormSumStat] + accumulator[kRowNormSumStat];
}

// The L2 norm of a row from its sum of squares, rounded to fp32: the fp32 square root of the sum rounded to fp32
// where that is a normal number, the fp64 square root elsewhere.
__device__ __forceinline__ float row_norm_of(double sum_of_squares) {
    return sum_of_squares >= kNormalSumOfSquaresMin && sum_of_squares < kNormalSumOfSquaresMax
               ? sqrtf(static_cast<float>(sum_of_squares))
               : static_cast<float>(sqrt(sum_of_squares));
}

// The rows of a [rows, row_length] tensor that the warps of block x take, one warp per row: rows x * kWarps +
// warp, then every `blocks` * kWarps rows. A 16-byte vector's squares add in fp32 and the row's in fp64, so a long
// row is as accurate as torch.linalg.vector_norm(dtype=float32); the lane sums meet in a shuffle-down tree and the
// norm is rounded to fp32 and added in fp64. `sum` is lane 0's sum of the norms of the warp's rows and `amax`
// the warp's absolute max, NaN if a row's sum of squares was (it is never negative, so exactly when a value is).
template <typename T>
__device__ __forceinline__ void row_norm_rows(
    const T* __restrict__ x, int64_t row_stride, int64_t rows, int64_t row_length, int blocks, int warp, int lane,
    double& sum, float& amax) {
    constexpr int kVectorElements = kVectorBytes / sizeof(T);
    float lane_amax = 0.f;
    bool saw_nan = false;
    sum = 0.0;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * kWarps + warp; row < rows;
         row += static_cast<int64_t>(blocks) * kWarps) {
        const T* row_ptr = x + row * row_stride;
        double sum_of_squares = 0.0;
        if (row_length % kVectorElements == 0 && reinterpret_cast<uintptr_t>(row_ptr) % kVectorBytes == 0) {
#pragma unroll 4
            for (int64_t vector_index = lane; vector_index < row_length / kVectorElements; vector_index += kWarpSize) {
                VectorLoad<T, kVectorElements> chunk =
                    reinterpret_cast<const VectorLoad<T, kVectorElements>*>(row_ptr)[vector_index];
                float partial = 0.f;
#pragma unroll
                for (int j = 0; j < kVectorElements; ++j) {
                    float value = static_cast<float>(chunk.elements[j]);
                    partial += value * value;
                    lane_amax = fmaxf(lane_amax, fabsf(value));
                }
                sum_of_squares += partial;
            }
        } else {
            for (int64_t column = lane; column < row_length; column += kWarpSize) {
                float value = static_cast<float>(row_ptr[column]);
                sum_of_squares += static_cast<double>(value) * value;
                lane_amax = fmaxf(lane_amax, fabsf(value));
            }
        }
        sum_of_squares = warp_reduce_sum(sum_of_squares);
        saw_nan |= isnan(sum_of_squares);
        sum += static_cast<double>(row_norm_of(sum_of_squares));
    }
    lane_amax = warp_reduce_max(lane_amax);
    amax = saw_nan ? NAN : lane_amax;
}

// Combines the blocks of a tensor: the warps of a block meet in the block's partials, and the last block to
// finish sums the others' partials in block order (thread t takes partials t, t + kBlockThreads, ...) and then the
// threads in a fixed tree, so the result is deterministic. Returns true in thread 0 of that block, with the
// tensor's [amax, sum of row norms] in `total_amax` and `total`. `blocks_done` starts at zero and the last
// block leaves it so for the next launch.
__device__ bool row_norm_reduce(
    double warp_sum, float warp_amax, int blocks, double* __restrict__ partial_sums, float* __restrict__ partial_amax,
    unsigned int* __restrict__ blocks_done, double& total, float& total_amax) {
    __shared__ double warp_sums[kWarps];
    __shared__ float warp_maxima[kWarps];
    __shared__ bool last_block;
    const int warp = static_cast<int>(threadIdx.x / kWarpSize);
    const int lane = static_cast<int>(threadIdx.x % kWarpSize);
    if (lane == 0) {
        warp_sums[warp] = warp_sum;
        warp_maxima[warp] = warp_amax;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double block_sum = 0.0;
        float block_amax = 0.f;
        for (int warp_index = 0; warp_index < kWarps; ++warp_index) {
            block_sum += warp_sums[warp_index];
            block_amax = nan_max(block_amax, warp_maxima[warp_index]);
        }
        partial_sums[blockIdx.x] = block_sum;
        partial_amax[blockIdx.x] = block_amax;
        // Publishes the partials before this block counts as done.
        __threadfence();
        last_block = atomicAdd(blocks_done, 1u) == static_cast<unsigned int>(blocks) - 1;
        if (last_block) {
            *blocks_done = 0;
        }
    }
    __syncthreads();
    if (!last_block) {
        return false;
    }
    double sum = 0.0;
    float amax = 0.f;
    for (int i = static_cast<int>(threadIdx.x); i < blocks; i += kBlockThreads) {
        sum += __ldcg(partial_sums + i);
        amax = nan_max(amax, __ldcg(partial_amax + i));
    }
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(kFullWarpMask, sum, offset);
        amax = nan_max(amax, __shfl_down_sync(kFullWarpMask, amax, offset));
    }
    if (lane == 0) {
        warp_sums[warp] = sum;
        warp_maxima[warp] = amax;
    }
    __syncthreads();
    if (threadIdx.x != 0) {
        return false;
    }
    total = 0.0;
    total_amax = 0.f;
    for (int warp_index = 0; warp_index < kWarps; ++warp_index) {
        total += warp_sums[warp_index];
        total_amax = nan_max(total_amax, warp_maxima[warp_index]);
    }
    return true;
}

// The accumulator slots a launch of row_norm_kernel takes the statistics into, in order: the offsets of their
// [amax, sum] in the accumulator.
constexpr int kMaxLaunchSlots = 8;
struct AccumulatorSlots {
    int count;
    int64_t offsets[kMaxLaunchSlots];
};

// The amax over |x| and the sum of the rows' L2 norms of a [rows, row_length] tensor (see row_norm_rows and
// row_norm_reduce); the `slots` of a non-null `accumulator` also take them, one after another.
template <typename T>
__global__ void __launch_bounds__(kBlockThreads, kBlocksPerSm) row_norm_kernel(
    const T* __restrict__ x,
    int64_t row_stride,
    int64_t rows,
    int64_t row_length,
    double* __restrict__ partial_sums,
    float* __restrict__ partial_amax,
    unsigned int* __restrict__ blocks_done,
    double* __restrict__ out,
    double* __restrict__ accumulator,
    AccumulatorSlots slots) {
    double sum;
    float amax;
    row_norm_rows(x, row_stride, rows, row_length, static_cast<int>(gridDim.x),
                  static_cast<int>(threadIdx.x / kWarpSize), static_cast<int>(threadIdx.x % kWarpSize), sum, amax);
    double total;
    float total_amax;
    if (row_norm_reduce(sum, amax, static_cast<int>(gridDim.x), partial_sums, partial_amax, blocks_done, total,
                        total_amax)) {
        out[kAmaxStat] = static_cast<double>(total_amax);
        out[kRowNormSumStat] = total;
        if (accumulator != nullptr) {
            for (int i = 0; i < slots.count; ++i) {
                accumulate_observation(out, accumulator + slots.offsets[i]);
            }
        }
    }
}

__global__ void accumulate_observation_kernel(const double* __restrict__ stats, double* __restrict__ accumulator) {
    accumulate_observation(stats, accumulator);
}

// Tensors of one dtype whose row-norm statistics one launch computes, and the accumulator slots each takes
// them into: tensor i into slots[first_slot[i]] .. slots[first_slot[i + 1] - 1], in that order.
constexpr int kMaxRowNorms = 32;
constexpr int kMaxRowNormSlots = 128;
// Blocks per SM the batch kernel is compiled for: its block needs 40 registers, and kBlocksPerSm blocks
// (32 registers) would make it spill.
constexpr int kRowNormBatchBlocksPerSm = 6;

struct RowNormList {
    const void* data[kMaxRowNorms];
    int64_t row_stride[kMaxRowNorms];
    int64_t rows[kMaxRowNorms];
    int64_t row_length[kMaxRowNorms];
    int blocks[kMaxRowNorms];  // the grid the tensor's own launch of row_norm_kernel has
    int first_slot[kMaxRowNorms + 1];
    int slots[kMaxRowNormSlots];
};

// row_norm_kernel of every tensor of a list in one launch: block x of tensor y is block x of the launch the
// tensor would have alone, whose grid has `blocks[y]` blocks (the blocks of this grid beyond them end at once),
// so a tensor's [amax, sum of row norms] does not depend on the other tensors and is bit for bit the one
// row_norm_accumulate computes. The last block of a tensor takes it into the tensor's accumulator slots, so no slot
// may belong to two tensors of a list. `blocks_done` starts at zero and each tensor's last block leaves it so.
template <typename T>
__global__ void __launch_bounds__(kBlockThreads, kRowNormBatchBlocksPerSm) row_norm_batch_kernel(
    RowNormList list,
    double* __restrict__ partial_sums,
    float* __restrict__ partial_amax,
    unsigned int* __restrict__ blocks_done,
    double* __restrict__ out,
    double* __restrict__ accumulator) {
    const int tensor = static_cast<int>(blockIdx.y);
    const int blocks = list.blocks[tensor];
    if (static_cast<int>(blockIdx.x) >= blocks) {
        return;
    }
    double sum;
    float amax;
    row_norm_rows(static_cast<const T*>(list.data[tensor]), list.row_stride[tensor], list.rows[tensor],
                  list.row_length[tensor], blocks, static_cast<int>(threadIdx.x / kWarpSize),
                  static_cast<int>(threadIdx.x % kWarpSize), sum, amax);
    double total;
    float total_amax;
    if (row_norm_reduce(sum, amax, blocks, partial_sums + static_cast<int64_t>(tensor) * gridDim.x,
                        partial_amax + static_cast<int64_t>(tensor) * gridDim.x, blocks_done + tensor, total,
                        total_amax)) {
        double* stats = out + kRowNormStats * tensor;
        stats[kAmaxStat] = static_cast<double>(total_amax);
        stats[kRowNormSumStat] = total;
        for (int slot = list.first_slot[tensor]; slot < list.first_slot[tensor + 1]; ++slot) {
            accumulate_observation(stats, accumulator + kRowNormStats * static_cast<int64_t>(list.slots[slot]));
        }
    }
}

constexpr int kDtypeKinds = kFloatKind + 1;

// Scratch of row_norm_kernel, one per stream of a device and kept for the life of the process, so a launch
// neither allocates nor clears it: block partial sums, block partial maxima as floats, room for the statistics of
// a launch whose caller keeps none, then the count of finished blocks, which each launch leaves at zero. Calls
// hold the GIL, so the map needs no lock; it is never destroyed, as its tensors would be freed after the CUDA
// context has gone.
struct RowNormScratch {
    torch::Tensor storage;
    double* partial_sums;
    float* partial_amax;
    double* out;
    unsigned int* blocks_done;
};

RowNormScratch& row_norm_scratch(const torch::Device& device, cudaStream_t stream) {
    static auto* scratches = new std::map<std::pair<int, cudaStream_t>, RowNormScratch>();
    const auto key = std::make_pair(static_cast<int>(device.index()), stream);
    auto found = scratches->find(key);
    if (found == scratches->end()) {
        const int64_t capacity = static_cast<int64_t>(sm_count_for_device(device.index())) * kBlocksPerSm;
        RowNormScratch scratch;
        scratch.storage = torch::zeros({capacity + (capacity + 1) / 2 + (kRowNormStats + 1)},
                                       torch::TensorOptions().dtype(torch::kFloat64).device(device));
        scratch.partial_sums = scratch.storage.data_ptr<double>();
        scratch.partial_amax = reinterpret_cast<float*>(scratch.partial_sums + capacity);
        scratch.out = scratch.partial_sums + capacity + (capacity + 1) / 2;
        scratch.blocks_done = reinterpret_cast<unsigned int*>(scratch.out + kRowNormStats);
        found = scratches->emplace(key, scratch).first;
    }
    return found->second;
}

// The [amax, sum of row norms] of the last-axis rows of `tensor` written to `out`, or to the stream's scratch
// when null; a non-null `accumulator` slot also takes them.
void row_norm_launch(const torch::Tensor& tensor, double* out, double* accumulator, const AccumulatorSlots& slots) {
    TORCH_CHECK(tensor.is_cuda() && tensor.dim() >= 1 && tensor.numel() > 0,
                "tensor must be a non-empty CUDA tensor with a last axis");
    torch::Tensor copy;
    const torch::Tensor& source = rows_to_read(tensor, copy);
    const int64_t row_length = source.size(-1);
    const int64_t rows = source.numel() / row_length;
    const int64_t row_stride = rows_stride(source);
    const at::cuda::OptionalCUDAGuard device_guard(source.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    RowNormScratch& scratch = row_norm_scratch(source.device(), stream);
    double* result = out == nullptr ? scratch.out : out;
    const int64_t grid = row_grid(rows, source.get_device());
    switch (source.scalar_type()) {
        case at::kBFloat16:
            row_norm_kernel<at::BFloat16><<<grid, kBlockThreads, 0, stream>>>(
                source.data_ptr<at::BFloat16>(), row_stride, rows, row_length, scratch.partial_sums,
                scratch.partial_amax, scratch.blocks_done, result, accumulator, slots);
            break;
        case at::kHalf:
            row_norm_kernel<at::Half><<<grid, kBlockThreads, 0, stream>>>(
                source.data_ptr<at::Half>(), row_stride, rows, row_length, scratch.partial_sums, scratch.partial_amax,
                scratch.blocks_done, result, accumulator, slots);
            break;
        case at::kFloat:
            row_norm_kernel<float><<<grid, kBlockThreads, 0, stream>>>(
                source.data_ptr<float>(), row_stride, rows, row_length, scratch.partial_sums, scratch.partial_amax,
                scratch.blocks_done, result, accumulator, slots);
            break;
        default:
            TORCH_CHECK(false, "activation_ext.row_norm_stats: unsupported dtype ", source.scalar_type());
    }
}

void check_accumulator(const torch::Tensor& accumulator, const torch::Tensor& like, int64_t slot) {
    TORCH_CHECK(accumulator.device() == like.device() && accumulator.scalar_type() == torch::kFloat64 &&
                    accumulator.is_contiguous() && accumulator.dim() == 2 &&
                    accumulator.size(1) == kRowNormStats && slot >= 0 && slot < accumulator.size(0),
                "accumulator must be a contiguous float64 [slots, 2] tensor on the observation's device with slot in "
                "range");
}

}  // namespace

torch::Tensor row_norm_stats(torch::Tensor tensor) {
    auto stats = torch::empty({kRowNormStats}, tensor.options().dtype(torch::kFloat64));
    row_norm_launch(tensor, stats.data_ptr<double>(), nullptr, AccumulatorSlots{0, {}});
    return stats;
}

c10::optional<torch::Tensor> row_norm_accumulate(
    torch::Tensor tensor, torch::Tensor accumulator, int64_t slot, bool keep_stats) {
    // `tensor` is an activation or its gradient: reading it must not join the autograd graph.
    at::NoGradGuard no_grad;
    check_accumulator(accumulator, tensor, slot);
    c10::optional<torch::Tensor> stats;
    if (keep_stats) {
        stats = torch::empty({kRowNormStats}, tensor.options().dtype(torch::kFloat64));
    }
    row_norm_launch(tensor, keep_stats ? stats->data_ptr<double>() : nullptr, accumulator.data_ptr<double>(),
                    AccumulatorSlots{1, {kRowNormStats * slot}});
    return stats;
}

void accumulate_row_norm_stats(torch::Tensor stats, torch::Tensor accumulator, int64_t slot) {
    check_accumulator(accumulator, stats, slot);
    TORCH_CHECK(stats.scalar_type() == torch::kFloat64 && stats.is_contiguous() && stats.numel() == kRowNormStats,
                "stats must be the two float64 statistics row_norm_accumulate returned");
    const at::cuda::OptionalCUDAGuard device_guard(stats.device());
    accumulate_observation_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        stats.data_ptr<double>(), accumulator.data_ptr<double>() + kRowNormStats * slot);
}

void row_norm_accumulate_slots(
    const torch::Tensor& tensor, const torch::Tensor& accumulator, const std::vector<int64_t>& slots) {
    at::NoGradGuard no_grad;
    TORCH_CHECK(!slots.empty(), "row_norm_accumulate_slots needs a slot");
    AccumulatorSlots launched{0, {}};
    for (const int64_t slot : slots) {
        check_accumulator(accumulator, tensor, slot);
        if (launched.count < kMaxLaunchSlots) {
            launched.offsets[launched.count++] = kRowNormStats * slot;
        }
    }
    c10::optional<torch::Tensor> stats;
    if (slots.size() > static_cast<size_t>(kMaxLaunchSlots)) {
        stats = torch::empty({kRowNormStats}, tensor.options().dtype(torch::kFloat64));
    }
    row_norm_launch(tensor, stats ? stats->data_ptr<double>() : nullptr, accumulator.data_ptr<double>(), launched);
    for (size_t i = kMaxLaunchSlots; i < slots.size(); ++i) {
        accumulate_row_norm_stats(*stats, accumulator, slots[i]);
    }
}

namespace {

// A scalar, one row of one element, taken into accumulator slots as amax_and_row_norm_sum and
// accumulate_row_norm_stats take it: its float32 magnitude, widened, is both statistics.
template <typename T>
__global__ void scalar_accumulate_kernel(
    const T* __restrict__ x, double* __restrict__ accumulator, AccumulatorSlots slots) {
    const double magnitude = static_cast<double>(fabsf(static_cast<float>(*x)));
    const double stats[kRowNormStats] = {magnitude, magnitude};
    for (int i = 0; i < slots.count; ++i) {
        accumulate_observation(stats, accumulator + slots.offsets[i]);
    }
}

}  // namespace

void scalar_accumulate_slots(
    const torch::Tensor& tensor, const torch::Tensor& accumulator, const std::vector<int64_t>& slots) {
    TORCH_CHECK(tensor.is_cuda() && tensor.dim() == 0 && tensor.is_floating_point(),
                "tensor must be a floating CUDA scalar");
    const at::cuda::OptionalCUDAGuard device_guard(tensor.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    for (size_t first = 0; first < slots.size(); first += kMaxLaunchSlots) {
        AccumulatorSlots launched{0, {}};
        for (size_t i = first; i < slots.size() && i < first + kMaxLaunchSlots; ++i) {
            check_accumulator(accumulator, tensor, slots[i]);
            launched.offsets[launched.count++] = kRowNormStats * slots[i];
        }
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, tensor.scalar_type(), "scalar_accumulate_slots", [&] {
            scalar_accumulate_kernel<scalar_t><<<1, 1, 0, stream>>>(
                tensor.data_ptr<scalar_t>(), accumulator.data_ptr<double>(), launched);
        });
    }
}

namespace {

// Scratch of row_norm_batch_kernel, one per stream of a device and kept for the life of the process like
// RowNormScratch: block partial sums of every tensor of a list, their partial maxima as floats, the statistics
// of each tensor, then each tensor's count of finished blocks, which every launch leaves at zero.
struct RowNormBatchScratch {
    torch::Tensor storage;
    double* partial_sums;
    float* partial_amax;
    double* out;
    unsigned int* blocks_done;
};

RowNormBatchScratch& row_norm_batch_scratch(const torch::Device& device, cudaStream_t stream) {
    static auto* scratches = new std::map<std::pair<int, cudaStream_t>, RowNormBatchScratch>();
    const auto key = std::make_pair(static_cast<int>(device.index()), stream);
    auto found = scratches->find(key);
    if (found == scratches->end()) {
        const int64_t capacity = static_cast<int64_t>(sm_count_for_device(device.index())) * kBlocksPerSm;
        const int64_t partials = kMaxRowNorms * capacity;
        RowNormBatchScratch scratch;
        scratch.storage = torch::zeros({partials + (partials + 1) / 2 + kRowNormStats * kMaxRowNorms +
                                        (kMaxRowNorms + 1) / 2},
                                       torch::TensorOptions().dtype(torch::kFloat64).device(device));
        scratch.partial_sums = scratch.storage.data_ptr<double>();
        scratch.partial_amax = reinterpret_cast<float*>(scratch.partial_sums + partials);
        scratch.out = scratch.partial_sums + partials + (partials + 1) / 2;
        scratch.blocks_done = reinterpret_cast<unsigned int*>(scratch.out + kRowNormStats * kMaxRowNorms);
        found = scratches->emplace(key, scratch).first;
    }
    return found->second;
}

}  // namespace

void row_norm_accumulate_batch(
    std::vector<torch::Tensor> tensors, torch::Tensor accumulator, std::vector<int64_t> slot_counts,
    std::vector<int64_t> slots) {
    at::NoGradGuard no_grad;
    TORCH_CHECK(!tensors.empty() && slot_counts.size() == tensors.size(),
                "row_norm_accumulate_batch needs one slot count per tensor");
    int64_t slot_total = 0;
    for (int64_t count : slot_counts) {
        TORCH_CHECK(count >= 1 && count <= kMaxRowNormSlots, "every tensor needs between one and ", kMaxRowNormSlots,
                    " slots");
        slot_total += count;
    }
    TORCH_CHECK(slot_total == static_cast<int64_t>(slots.size()), "row_norm_accumulate_batch needs every slot listed");
    for (int64_t slot : slots) {
        check_accumulator(accumulator, tensors[0], slot);
    }
    const at::cuda::OptionalCUDAGuard device_guard(tensors[0].device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    RowNormBatchScratch& scratch = row_norm_batch_scratch(tensors[0].device(), stream);
    const int sm_blocks = sm_count_for_device(tensors[0].get_device()) * kBlocksPerSm;

    // The kernels are per dtype, so there is a list per DtypeKind, in which the tensors of the dtype keep their
    // order. The copies of tensors that are not contiguous live until every launch is issued.
    struct Batch {
        RowNormList list{};
        int entries = 0, listed_slots = 0, max_blocks = 0;
    };
    Batch batches[kDtypeKinds];
    std::vector<torch::Tensor> copies;
    double* accumulator_ptr = accumulator.data_ptr<double>();
    auto launch = [&](int kind) {
        Batch& batch = batches[kind];
        if (batch.entries == 0) {
            return;
        }
        batch.list.first_slot[batch.entries] = batch.listed_slots;
        const dim3 grid(batch.max_blocks, batch.entries);
        switch (kind) {
            case kBFloat16Kind:
                row_norm_batch_kernel<at::BFloat16><<<grid, kBlockThreads, 0, stream>>>(
                    batch.list, scratch.partial_sums, scratch.partial_amax, scratch.blocks_done, scratch.out,
                    accumulator_ptr);
                break;
            case kHalfKind:
                row_norm_batch_kernel<at::Half><<<grid, kBlockThreads, 0, stream>>>(
                    batch.list, scratch.partial_sums, scratch.partial_amax, scratch.blocks_done, scratch.out,
                    accumulator_ptr);
                break;
            default:
                row_norm_batch_kernel<float><<<grid, kBlockThreads, 0, stream>>>(
                    batch.list, scratch.partial_sums, scratch.partial_amax, scratch.blocks_done, scratch.out,
                    accumulator_ptr);
        }
        batch = Batch{};
    };
    size_t next_slot = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        const torch::Tensor& x = tensors[i];
        TORCH_CHECK(x.is_cuda() && x.dim() >= 1 && x.numel() > 0 && x.device() == tensors[0].device(),
                    "every tensor must be a non-empty CUDA tensor with a last axis, on one device");
        torch::Tensor copy;
        const torch::Tensor& source = rows_to_read(x, copy);
        if (copy.defined()) {
            copies.push_back(copy);
        }
        const auto dtype = source.scalar_type();
        TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kHalf || dtype == at::kFloat,
                    "row_norm_accumulate_batch supports bfloat16, float16 and float32 tensors");
        const int kind = dtype == at::kBFloat16 ? kBFloat16Kind : (dtype == at::kHalf ? kHalfKind : kFloatKind);
        Batch& batch = batches[kind];
        if (batch.entries == kMaxRowNorms || batch.listed_slots + slot_counts[i] > kMaxRowNormSlots) {
            launch(kind);
        }
        const int entry = batch.entries++;
        batch.list.data[entry] = source.data_ptr();
        batch.list.row_length[entry] = source.size(-1);
        batch.list.rows[entry] = source.numel() / batch.list.row_length[entry];
        batch.list.row_stride[entry] = rows_stride(source);
        // The grid of the tensor's own launch, see row_grid.
        batch.list.blocks[entry] = static_cast<int>(
            std::min<int64_t>(ceil_div(batch.list.rows[entry], kWarps), static_cast<int64_t>(sm_blocks)));
        batch.max_blocks = std::max(batch.max_blocks, batch.list.blocks[entry]);
        batch.list.first_slot[entry] = batch.listed_slots;
        for (int64_t slot_index = 0; slot_index < slot_counts[i]; ++slot_index) {
            batch.list.slots[batch.listed_slots++] = static_cast<int>(slots[next_slot++]);
        }
    }
    for (int kind = 0; kind < kDtypeKinds; ++kind) {
        launch(kind);
    }
}
