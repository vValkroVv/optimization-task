#include "cuda_host_functions.h"
#include "kernel_common.cuh"

namespace {

// The norms of a weight are [amax, L2 norm, 1].
constexpr int kWeightNormWords = 3;
// A launch gives a weight one block per kBlockThreads * kWeightElementsPerThread elements, up to this many blocks.
constexpr int kWeightElementsPerThread = 64;
constexpr int kMaxWeightBlocks = 64;
// Weights whose norms one launch computes; a longer list is processed in chunks of this many.
constexpr int kMaxWeights = 64;

struct WeightList {
    const void* data[kMaxWeights];
    int64_t numel[kMaxWeights];
    int dtype[kMaxWeights];  // a DtypeKind
    int64_t out_offset[kMaxWeights];  // where each weight's [amax, L2 norm, 1] goes in the output
    int blocks[kMaxWeights];  // blocks that reduce each weight; the launch's grid has the most of them
};

__device__ __forceinline__ float weight_element(const void* data, int dtype, int64_t i) {
    if (dtype == kBFloat16Kind) {
        return static_cast<float>(static_cast<const at::BFloat16*>(data)[i]);
    }
    if (dtype == kHalfKind) {
        return static_cast<float>(static_cast<const at::Half*>(data)[i]);
    }
    return static_cast<const float*>(data)[i];
}

// Block x of weight y: its grid-stride share's sum of squares (fp32 per thread, fp64 across threads) and absolute
// max (NaN propagating). Weight y has weights.blocks[y] blocks and the blocks of the grid beyond them end at once,
// so a weight's reduction does not depend on the other weights of the launch. The last block of a weight sums its
// partials in block order and writes [amax, l2, 1], the L2 norm rounded to the weight's dtype as torch returns it.
// Each weight's `blocks_done` starts at zero.
__global__ void weight_norm_kernel(WeightList weights, double* __restrict__ partial_sums,
                                   float* __restrict__ partial_amax, unsigned int* __restrict__ blocks_done,
                                   float* __restrict__ out) {
    __shared__ double warp_sums[kWarps];
    __shared__ float warp_amax[kWarps];
    __shared__ bool last_block;
    const int weight = static_cast<int>(blockIdx.y);
    const int blocks = weights.blocks[weight];
    if (static_cast<int>(blockIdx.x) >= blocks) {
        return;
    }
    const void* data = weights.data[weight];
    const int dtype = weights.dtype[weight];
    const int64_t element_count = weights.numel[weight];
    float sum_of_squares = 0.f, amax = 0.f;
    bool nan = false;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < element_count;
         i += static_cast<int64_t>(blocks) * blockDim.x) {
        const float value = weight_element(data, dtype, i);
        sum_of_squares += value * value;
        amax = fmaxf(amax, fabsf(value));
        nan |= isnan(value);
    }
    double sum = static_cast<double>(sum_of_squares);
    sum = warp_reduce_sum(sum);
    amax = warp_reduce_max(amax);
    nan = __any_sync(kFullWarpMask, nan);
    if (threadIdx.x % kWarpSize == 0) {
        warp_sums[threadIdx.x / kWarpSize] = sum;
        warp_amax[threadIdx.x / kWarpSize] = nan ? NAN : amax;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double block_sum = 0.0;
        float block_amax = 0.f;
        for (int warp_index = 0; warp_index < kWarps; ++warp_index) {
            block_sum += warp_sums[warp_index];
            block_amax =
                isnan(block_amax) || isnan(warp_amax[warp_index]) ? NAN : fmaxf(block_amax, warp_amax[warp_index]);
        }
        partial_sums[weight * gridDim.x + blockIdx.x] = block_sum;
        partial_amax[weight * gridDim.x + blockIdx.x] = block_amax;
        // Publishes the partials before this block counts as done.
        __threadfence();
        last_block = atomicAdd(blocks_done + weight, 1u) == static_cast<unsigned int>(blocks) - 1;
    }
    __syncthreads();
    if (!last_block || threadIdx.x >= kWarpSize) {
        return;
    }
    // Lane l sums partials l, l + 32, ... in order, then a fixed shuffle tree.
    const int stride = static_cast<int>(gridDim.x);
    double total = 0.0;
    float total_amax = 0.f;
    for (int i = static_cast<int>(threadIdx.x); i < blocks; i += kWarpSize) {
        total += __ldcg(partial_sums + weight * stride + i);
        const float other = __ldcg(partial_amax + weight * stride + i);
        total_amax = isnan(total_amax) || isnan(other) ? NAN : fmaxf(total_amax, other);
    }
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        total += __shfl_down_sync(kFullWarpMask, total, offset);
        const float other = __shfl_down_sync(kFullWarpMask, total_amax, offset);
        total_amax = isnan(total_amax) || isnan(other) ? NAN : fmaxf(total_amax, other);
    }
    if (threadIdx.x == 0) {
        float l2_norm = static_cast<float>(sqrt(total));
        if (dtype == kBFloat16Kind) {
            l2_norm = static_cast<float>(at::BFloat16(l2_norm));
        } else if (dtype == kHalfKind) {
            l2_norm = static_cast<float>(at::Half(l2_norm));
        }
        out[weights.out_offset[weight] + 0] = total_amax;
        out[weights.out_offset[weight] + 1] = l2_norm;
        out[weights.out_offset[weight] + 2] = 1.f;
    }
}

// Enough blocks to keep the device busy for a launch whose largest weight has `largest` elements.
int weight_norm_blocks(int64_t largest) {
    constexpr int kBlockElements = kBlockThreads * kWeightElementsPerThread;
    return static_cast<int>(std::min<int64_t>(ceil_div(largest, kBlockElements), kMaxWeightBlocks));
}

// [amax, L2 norm, 1] of weights[i] written to out[offsets[i]], out[offsets[i] + 1] and out[offsets[i] + 2].
// launch_numels[i] is the element count of the largest weight of the launch that weight i is reduced like in: it
// sets the weight's block count, and so the order of its sums. Empty: every chunk of kMaxWeights weights is such
// a launch.
//
// One weight of a launch: its elements, their count and their dtype as WeightList numbers it.
struct WeightRef {
    const void* data;
    int64_t numel;
    int dtype;
};

// The weight dtype number of WeightList, or -1 for a dtype weight_norm_kernel does not read.
int weight_dtype(at::ScalarType dtype) {
    return dtype == at::kBFloat16 ? kBFloat16Kind
                                  : (dtype == at::kHalf ? kHalfKind : (dtype == at::kFloat ? kFloatKind : -1));
}

void launch_weight_norms(const std::vector<WeightRef>& weights, const torch::Device& device, float* out,
                         const std::vector<int64_t>& offsets, const std::vector<int64_t>& launch_numels) {
    TORCH_CHECK(!weights.empty(), "weight_norms needs at least one weight");
    const at::cuda::OptionalCUDAGuard device_guard(device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int64_t count = static_cast<int64_t>(weights.size());
    for (int64_t first = 0; first < count; first += kMaxWeights) {
        const int chunk = static_cast<int>(std::min<int64_t>(kMaxWeights, count - first));
        WeightList list{};
        int64_t largest = 1;
        for (int i = 0; i < chunk; ++i) {
            const WeightRef& weight = weights[first + i];
            list.data[i] = weight.data;
            list.numel[i] = weight.numel;
            list.dtype[i] = weight.dtype;
            list.out_offset[i] = offsets[first + i];
            largest = std::max(largest, weight.numel);
        }
        int blocks = 0;
        for (int i = 0; i < chunk; ++i) {
            list.blocks[i] = weight_norm_blocks(launch_numels.empty() ? largest : launch_numels[first + i]);
            blocks = std::max(blocks, list.blocks[i]);
        }
        // Block partial sums, then block partial maxima as floats, then each weight's count of
        // finished blocks.
        const int64_t partials = static_cast<int64_t>(chunk) * blocks;
        auto scratch = torch::empty({partials + (partials + 1) / 2 + (chunk + 1) / 2},
                                    torch::TensorOptions().device(device).dtype(torch::kFloat64));
        double* partial_sums = scratch.data_ptr<double>();
        float* partial_amax = reinterpret_cast<float*>(partial_sums + partials);
        unsigned int* blocks_done = reinterpret_cast<unsigned int*>(partial_sums + partials + (partials + 1) / 2);
        C10_CUDA_CHECK(cudaMemsetAsync(blocks_done, 0, chunk * sizeof(unsigned int), stream));
        weight_norm_kernel<<<dim3(blocks, chunk), kBlockThreads, 0, stream>>>(list, partial_sums, partial_amax,
                                                                            blocks_done, out);
    }
}

}  // namespace

torch::Tensor weight_norms(std::vector<torch::Tensor> weights) {
    auto out = torch::empty({static_cast<int64_t>(weights.size()), kWeightNormWords},
                            weights.empty() ? torch::TensorOptions() : weights[0].options().dtype(torch::kFloat32));
    std::vector<WeightRef> refs;
    std::vector<int64_t> offsets(weights.size());
    for (size_t i = 0; i < weights.size(); ++i) {
        const torch::Tensor& weight = weights[i];
        TORCH_CHECK(weight.device() == weights[0].device() && weight.is_contiguous() && weight.numel() > 0,
                    "weights must be non-empty contiguous tensors on one CUDA device");
        TORCH_CHECK(weight_dtype(weight.scalar_type()) >= 0,
                    "weight_norms supports bfloat16, float16 and float32 weights");
        refs.push_back({weight.data_ptr(), weight.numel(), weight_dtype(weight.scalar_type())});
        offsets[i] = static_cast<int64_t>(i) * kWeightNormWords;
    }
    launch_weight_norms(refs, weights.empty() ? torch::Device(torch::kCUDA) : weights[0].device(),
                        out.data_ptr<float>(), offsets, {});
    return out;
}

bool weight_norms_into(
    std::vector<torch::Tensor> weights, std::vector<int64_t> experts, torch::Tensor buffer,
    std::vector<int64_t> offsets, std::vector<int64_t> launch_numels) {
    TORCH_CHECK(
        buffer.is_cuda() && buffer.scalar_type() == torch::kFloat32 && buffer.is_contiguous() && buffer.dim() == 1,
        "buffer must be a contiguous 1D float32 CUDA tensor");
    TORCH_CHECK(experts.size() == weights.size(), "weight_norms_into needs one expert count per tensor");
    // Tensor i is one weight, or with experts[i] > 0 that many: the rows of its view as [experts[i], -1]. A reshape
    // that copies is kept until the launch is queued, so that no allocation takes its memory before.
    std::vector<WeightRef> refs;
    std::vector<torch::Tensor> stacks;
    for (size_t i = 0; i < weights.size(); ++i) {
        const torch::Tensor& weight = weights[i];
        const int dtype = weight_dtype(weight.scalar_type());
        if (!weight.is_cuda() || dtype < 0) {
            return false;
        }
        TORCH_CHECK(weight.device() == buffer.device(), "weights and buffer must share a device");
        TORCH_CHECK(experts[i] >= 0, "expert counts must be nonnegative");
        if (experts[i] == 0) {
            if (!weight.is_contiguous()) {
                return false;
            }
            TORCH_CHECK(weight.numel() > 0, "weights must be non-empty");
            refs.push_back({weight.data_ptr(), weight.numel(), dtype});
            continue;
        }
        const torch::Tensor& rows = stacks.emplace_back(weight.reshape({experts[i], -1}));
        const int64_t numel = rows.size(1);
        if (numel > 1 && rows.stride(1) != 1) {
            return false;
        }
        TORCH_CHECK(numel > 0, "weights must be non-empty");
        const char* data = static_cast<const char*>(rows.data_ptr());
        const int64_t row_bytes = rows.stride(0) * static_cast<int64_t>(rows.element_size());
        for (int64_t row = 0; row < experts[i]; ++row) {
            refs.push_back({data + row * row_bytes, numel, dtype});
        }
    }
    TORCH_CHECK_VALUE(offsets.size() == refs.size(), "weight_norms_into needs one offset per weight, got ",
                      offsets.size(), " for ", refs.size());
    for (int64_t offset : offsets) {
        TORCH_CHECK(offset >= 0 && offset + kWeightNormWords <= buffer.numel(), "weight norm offset ", offset,
                    " is outside buffer");
    }
    TORCH_CHECK(launch_numels.size() == refs.size(), "weight_norms_into needs one launch size per weight");
    for (int64_t launch_numel : launch_numels) {
        TORCH_CHECK(launch_numel > 0, "weight launch sizes must be positive");
    }
    if (!refs.empty()) {
        launch_weight_norms(refs, buffer.device(), buffer.data_ptr<float>(), offsets, launch_numels);
    }
    return true;
}
