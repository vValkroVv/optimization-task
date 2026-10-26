#include "cuda_host_functions.h"
#include "kernel_common.cuh"

namespace {

constexpr int kRouterFinishThreads = 1024;

// A router call has three top-k score tensors (normalized, raw and biased), and a population sums their values
// over its valid tokens, besides the count of the valid tokens.
constexpr int kTopkTensors = 3;
constexpr int kTopkSums = kTopkTensors + 1;
constexpr int kValidTokensSum = kTopkTensors;

// A router population's float64 sums, in the order of its `values`.
enum RouterSum { kNormalizedTopkSum, kRawTopkSum, kBiasedTopkSum, kEntropySum, kZLossSum, kSeqAuxLossSum, kAlphaSum };
constexpr int kRouterSums = kAlphaSum + 1;
// A router population's int32 counts, in the order of its recorder slot.
enum RouterCount { kTokensCount, kValidTokensCount, kCallsCount };
constexpr int kRouterCounts = kCallsCount + 1;

// The three top-k score tensors of a router call, float32 [tokens, columns] with a contiguous last axis.
struct RouterTopk {
    const float* values[kTopkTensors];
    int64_t stride[kTopkTensors];
    int64_t columns[kTopkTensors];
};

// A router call's validity mask, read in place whatever its dtype: a token is valid where its value is not zero,
// as torch's bool() takes it, and a value other than zero or one sets `nonbinary`.
struct TokenMask {
    const void* data;  // null: every token is valid
    int bytes;         // 1, 2, 4 or 8; a 1-byte mask is bool or uint8 when `is_signed` is false
    bool is_signed;
    int* nonbinary;    // null for a bool mask

    __device__ __forceinline__ int64_t value(int64_t token) const {
        switch (bytes) {
            case 1:
                return is_signed ? static_cast<const int8_t*>(data)[token] : static_cast<const uint8_t*>(data)[token];
            case 2:
                return static_cast<const int16_t*>(data)[token];
            case 4:
                return static_cast<const int32_t*>(data)[token];
            default:
                return static_cast<const int64_t*>(data)[token];
        }
    }

    __device__ __forceinline__ bool valid(int64_t token) const {
        return data == nullptr || value(token) != 0;
    }
};

// A router call's auxiliary loss: a float32 or float64 device scalar, or a host value.
struct RouterScalar {
    const void* device;
    bool is_double;
    double host;

    __device__ double value() const {
        if (device == nullptr) {
            return host;
        }
        return is_double ? *static_cast<const double*>(device)
                         : static_cast<double>(*static_cast<const float*>(device));
    }
};

// The partials of one router call, each in the blocks of the reduction it stands for: blocks below entropy_blocks
// take entropy_partials_kernel's share of the biased scores and blocks below column_blocks column_partials_kernel's
// run of score rows, so that router_finish_kernel's sums are bit for bit those of entropy_sum and column_sums;
// blocks below topk_blocks take a grid-stride share of the three top-k sums over the valid tokens and of the
// valid token count (whole numbers, exact).
__global__ void __launch_bounds__(kBlockThreads) router_partials_kernel(
    const float* __restrict__ scores, int64_t score_stride, const float* __restrict__ biased, int64_t tokens,
    int experts, int64_t entropy_blocks, int64_t rows_per_block, int64_t column_blocks, RouterTopk topk,
    TokenMask mask, int64_t topk_blocks, double* __restrict__ entropy_partials,
    double* __restrict__ column_partials, double* __restrict__ topk_partials) {
    __shared__ double entropy_warps[kWarps];
    __shared__ double topk_warps[kTopkSums][kWarps];
    const int64_t block = blockIdx.x;
    const int warp = static_cast<int>(threadIdx.x / kWarpSize);
    const int lane = static_cast<int>(threadIdx.x % kWarpSize);
    if (block < entropy_blocks) {
        double sum = 0.0;
        const int64_t element_count = tokens * experts;
        const int64_t step = entropy_blocks * blockDim.x;
        for (int64_t i = block * blockDim.x + threadIdx.x; i < element_count; i += step) {
            const double value = static_cast<double>(biased[i]);
            sum -= __dmul_rn(value, log(value + kEntropyEpsilon));
        }
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            entropy_warps[warp] = sum;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            double block_sum = 0.0;
            for (int warp_index = 0; warp_index < kWarps; ++warp_index) {
                block_sum += entropy_warps[warp_index];
            }
            entropy_partials[block] = block_sum;
        }
    }
    if (block < column_blocks) {
        const int64_t first = block * rows_per_block;
        const int64_t last = min(tokens, first + rows_per_block);
        for (int column = threadIdx.x; column < experts; column += blockDim.x) {
            double sum = 0.0;
#pragma unroll 8
            for (int64_t row = first; row < last; ++row) {
                sum += static_cast<double>(scores[row * score_stride + column]);
            }
            column_partials[block * experts + column] = sum;
        }
    }
    if (block < topk_blocks) {
        double sums[kTopkSums] = {0.0, 0.0, 0.0, 0.0};
        const int64_t step = topk_blocks * blockDim.x;
        for (int topk_index = 0; topk_index < kTopkTensors; ++topk_index) {
            const int64_t element_count = tokens * topk.columns[topk_index];
            for (int64_t i = block * blockDim.x + threadIdx.x; i < element_count; i += step) {
                const int64_t token = i / topk.columns[topk_index];
                if (mask.valid(token)) {
                    sums[topk_index] += static_cast<double>(
                        topk.values[topk_index][token * topk.stride[topk_index] + i % topk.columns[topk_index]]);
                }
            }
        }
        for (int64_t token = block * blockDim.x + threadIdx.x; token < tokens; token += step) {
            if (mask.data == nullptr) {
                sums[kValidTokensSum] += 1.0;
                continue;
            }
            const int64_t value = mask.value(token);
            sums[kValidTokensSum] += value != 0 ? 1.0 : 0.0;
            if (mask.nonbinary != nullptr && value != 0 && value != 1) {
                atomicOr(mask.nonbinary, 1);
            }
        }
        for (int sum_index = 0; sum_index < kTopkSums; ++sum_index) {
            sums[sum_index] = warp_reduce_sum(sums[sum_index]);
            if (lane == 0) {
                topk_warps[sum_index][warp] = sums[sum_index];
            }
        }
        __syncthreads();
        if (threadIdx.x < kTopkSums) {
            double block_sum = 0.0;
            for (int warp_index = 0; warp_index < kWarps; ++warp_index) {
                block_sum += topk_warps[threadIdx.x][warp_index];
            }
            topk_partials[threadIdx.x * topk_blocks + block] = block_sum;
        }
    }
}

// Where a router population's recorder slots are in the step buffer's words, and the population's token and call
// counts after this call; a null words pointer writes no slots.
struct RouterSlots {
    int32_t* words;
    int64_t mean;
    int64_t balance;
    int64_t counts;
    int64_t bias;
    int64_t tokens;
    int64_t calls;
    const float* expert_bias;
};

// Adds a router call's sums to its population: a warp per expert column in column_finish_kernel's order, warp 0
// the entropy in sum_partials_kernel's order, warp 1 the top-k sums and the valid token count, warp 2 the
// auxiliary scalars, each the same addition the population's += was; the first call adds to zero. Each writer also
// puts its result in the recorder slots: the score mean as float32 of a float64 product with the reciprocal of the
// tokens (NaN without tokens), the seven balance values as 32-bit word pairs, the counts as int32 and, on the
// first call, the expert bias (zero without one).
__global__ void router_finish_kernel(
    const double* __restrict__ entropy_partials, int entropy_blocks, const double* __restrict__ column_partials,
    int column_blocks, int experts, const double* __restrict__ topk_partials, int topk_blocks, RouterScalar z_loss,
    RouterScalar seq_aux_loss, double alpha, double* __restrict__ values, long long* __restrict__ valid_tokens,
    double* __restrict__ score_sums, bool first, RouterSlots slots) {
    const int warp = static_cast<int>(threadIdx.x / kWarpSize);
    const int lane = static_cast<int>(threadIdx.x % kWarpSize);
    const int warps = static_cast<int>(blockDim.x / kWarpSize);
    const double inverse_tokens = slots.tokens > 0 ? 1.0 / static_cast<double>(slots.tokens) : 0.0;
    // A warp per column across the blocks, each summed in the same order whichever warp has it; block 0 also
    // takes the population's other sums below.
    for (int column = static_cast<int>(blockIdx.x) * warps + warp; column < experts;
         column += static_cast<int>(gridDim.x) * warps) {
        double sum = 0.0;
        for (int i = lane; i < column_blocks; i += kWarpSize) {
            sum += column_partials[static_cast<int64_t>(i) * experts + column];
        }
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            const double total = (first ? 0.0 : score_sums[column]) + sum;
            score_sums[column] = total;
            if (slots.words != nullptr) {
                const float mean = slots.tokens > 0 ? __double2float_rn(__dmul_rn(total, inverse_tokens))
                                                    : __int_as_float(kQuietNanBits);
                slots.words[slots.mean + column] = __float_as_int(mean);
            }
        }
    }
    if (blockIdx.x != 0) {
        return;
    }
    if (slots.words != nullptr && first) {
        for (int column = threadIdx.x; column < experts; column += blockDim.x) {
            slots.words[slots.bias + column] =
                __float_as_int(slots.expert_bias == nullptr ? 0.0f : slots.expert_bias[column]);
        }
    }
    if (warp == 0) {
        double sum = 0.0;
        for (int i = lane; i < entropy_blocks; i += kWarpSize) {
            sum += entropy_partials[i];
        }
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            values[kEntropySum] = (first ? 0.0 : values[kEntropySum]) + sum;
            if (slots.words != nullptr) {
                store_double_words(slots.words + slots.balance + 2 * kEntropySum, values[kEntropySum]);
            }
        }
    } else if (warp == 1) {
        for (int sum_index = 0; sum_index < kTopkSums; ++sum_index) {
            double sum = 0.0;
            for (int i = lane; i < topk_blocks; i += kWarpSize) {
                sum += topk_partials[sum_index * topk_blocks + i];
            }
            sum = warp_reduce_sum(sum);
            if (lane == 0) {
                if (sum_index < kTopkTensors) {
                    values[sum_index] = (first ? 0.0 : values[sum_index]) + sum;
                    if (slots.words != nullptr) {
                        store_double_words(slots.words + slots.balance + 2 * sum_index, values[sum_index]);
                    }
                } else {
                    *valid_tokens = (first ? 0ll : *valid_tokens) + static_cast<long long>(sum);
                    if (slots.words != nullptr) {
                        slots.words[slots.counts + kValidTokensCount] = static_cast<int32_t>(*valid_tokens);
                    }
                }
            }
        }
    } else if (warp == 2 && lane == 0) {
        values[kZLossSum] = (first ? 0.0 : values[kZLossSum]) + z_loss.value();
        values[kSeqAuxLossSum] = (first ? 0.0 : values[kSeqAuxLossSum]) + seq_aux_loss.value();
        values[kAlphaSum] = (first ? 0.0 : values[kAlphaSum]) + alpha;
        if (slots.words != nullptr) {
            for (int sum_index = kZLossSum; sum_index < kRouterSums; ++sum_index) {
                store_double_words(slots.words + slots.balance + 2 * sum_index, values[sum_index]);
            }
            slots.words[slots.counts + kTokensCount] = static_cast<int32_t>(slots.tokens);
            slots.words[slots.counts + kCallsCount] = static_cast<int32_t>(slots.calls);
        }
    }
}

// A loss scalar as router_finish_kernel reads it: a float32 or float64 device scalar in place, another device
// dtype converted to float64 (kept alive in `kept`), a host scalar by value (a disabled z-loss is the integer
// host zero).
RouterScalar router_scalar(const torch::Tensor& value, const torch::Device& device, std::vector<torch::Tensor>& kept) {
    TORCH_CHECK(value.numel() == 1, "router losses must be scalars");
    if (!value.is_cuda()) {
        return RouterScalar{nullptr, false, value.item<double>()};
    }
    TORCH_CHECK(value.device() == device, "router losses must be on the scores' device or the host");
    torch::Tensor scalar = value;
    if (scalar.scalar_type() != at::kFloat && scalar.scalar_type() != at::kDouble) {
        scalar = scalar.to(at::kDouble);
        kept.push_back(scalar);
    }
    return RouterScalar{scalar.data_ptr(), scalar.scalar_type() == at::kDouble, 0.0};
}

}  // namespace

void router_accumulate(
    torch::Tensor scores, torch::Tensor biased_scores, torch::Tensor normalized_topk, torch::Tensor raw_topk,
    torch::Tensor biased_topk, c10::optional<torch::Tensor> mask, torch::Tensor z_loss,
    c10::optional<torch::Tensor> seq_aux_loss, double alpha, torch::Tensor values, torch::Tensor valid_tokens,
    torch::Tensor score_sums, bool first, c10::optional<torch::Tensor> words, int64_t mean_offset,
    int64_t balance_offset, int64_t counts_offset, int64_t bias_offset, int64_t tokens_total, int64_t calls_total,
    c10::optional<torch::Tensor> expert_bias, c10::optional<torch::Tensor> nonbinary) {
    // The operands are views of the router's outputs: reshaping them must not join the autograd graph.
    at::NoGradGuard no_grad;
    if (scores.dim() != 2) {
        scores = scores.reshape({-1, scores.size(-1)});
    }
    check_rows(scores);
    TORCH_CHECK(scores.scalar_type() == at::kFloat && biased_scores.scalar_type() == at::kFloat,
                "router_accumulate takes float32 scores");
    const torch::Device device = scores.device();
    const int64_t tokens = scores.size(0), experts = scores.size(1);
    TORCH_CHECK(biased_scores.device() == device && biased_scores.numel() == scores.numel(),
                "biased scores must match the scores");
    const torch::Tensor biased = biased_scores.contiguous();
    std::vector<torch::Tensor> kept;
    RouterTopk topk;
    const torch::Tensor* inputs[kTopkTensors] = {&normalized_topk, &raw_topk, &biased_topk};
    for (int topk_index = 0; topk_index < kTopkTensors; ++topk_index) {
        const torch::Tensor& input = *inputs[topk_index];
        TORCH_CHECK(input.device() == device && input.is_floating_point() && input.dim() >= 2 &&
                        input.size(-1) > 0 && input.numel() == tokens * input.size(-1),
                    "router top-k values must cover every token on the scores' device");
        // Exact: float32 holds every bfloat16 and float16 value.
        torch::Tensor rows = input.reshape({tokens, input.size(-1)}).to(at::kFloat);
        if (rows.stride(1) != 1) {
            rows = rows.contiguous();
        }
        kept.push_back(rows);
        topk.values[topk_index] = rows.data_ptr<float>();
        topk.stride[topk_index] = rows.stride(0);
        topk.columns[topk_index] = rows.size(1);
    }
    TokenMask token_mask{nullptr, 1, false, nullptr};
    if (mask.has_value()) {
        const auto kind = mask->scalar_type();
        TORCH_CHECK(mask->device() == device && mask->numel() == tokens &&
                        (kind == at::kBool || kind == at::kByte || kind == at::kChar || kind == at::kShort ||
                         kind == at::kInt || kind == at::kLong),
                    "the router mask must be a boolean or integer per token on the scores' device");
        kept.push_back(mask->contiguous());
        token_mask.data = kept.back().data_ptr();
        token_mask.bytes = static_cast<int>(mask->element_size());
        token_mask.is_signed = kind != at::kBool && kind != at::kByte;
        if (kind != at::kBool) {
            TORCH_CHECK(nonbinary.has_value() && nonbinary->device() == device &&
                            nonbinary->scalar_type() == at::kInt && nonbinary->numel() == 1,
                        "an integer router mask needs the int32 flag that takes its values other than zero and one");
            token_mask.nonbinary = nonbinary->data_ptr<int>();
        }
    }
    TORCH_CHECK(values.device() == device && values.scalar_type() == at::kDouble && values.is_contiguous() &&
                    values.numel() == kRouterSums,
                "values must be seven contiguous float64 sums on the scores' device");
    TORCH_CHECK(valid_tokens.device() == device && valid_tokens.scalar_type() == at::kLong && valid_tokens.numel() == 1,
                "valid_tokens must be one int64 count on the scores' device");
    TORCH_CHECK(score_sums.device() == device && score_sums.scalar_type() == at::kDouble &&
                    score_sums.is_contiguous() && score_sums.numel() == experts,
                "score_sums must be one contiguous float64 sum per expert on the scores' device");
    const RouterScalar z_loss_scalar = router_scalar(z_loss, device, kept);
    const RouterScalar seq_aux_loss_scalar = seq_aux_loss.has_value()
                                                 ? router_scalar(*seq_aux_loss, device, kept)
                                                 : RouterScalar{nullptr, false, std::nan("")};
    RouterSlots slots{nullptr, mean_offset, balance_offset, counts_offset, bias_offset, tokens_total, calls_total,
                      nullptr};
    if (words.has_value()) {
        TORCH_CHECK(words->device() == device && words->scalar_type() == at::kInt && words->is_contiguous(),
                    "words must be the recorder's contiguous int32 buffer on the scores' device");
        TORCH_CHECK(mean_offset >= 0 && mean_offset + experts <= words->numel() && balance_offset >= 0 &&
                        balance_offset + 2 * kRouterSums <= words->numel() && counts_offset >= 0 &&
                        counts_offset + kRouterCounts <= words->numel() && bias_offset >= 0 &&
                        bias_offset + experts <= words->numel(),
                    "router slot offsets are outside the buffer");
        TORCH_CHECK(tokens_total >= 0 && tokens_total <= std::numeric_limits<int32_t>::max() && calls_total > 0 &&
                        calls_total <= std::numeric_limits<int32_t>::max(),
                    "router token and call counts must fit int32");
        slots.words = words->data_ptr<int32_t>();
        if (expert_bias.has_value()) {
            TORCH_CHECK(expert_bias->device() == device && expert_bias->is_floating_point() &&
                            expert_bias->numel() == experts,
                        "expert_bias must be one floating value per expert on the scores' device");
            // Exact for the float32, bfloat16 and float16 a bias is kept in.
            kept.push_back(expert_bias->to(at::kFloat).contiguous());
            slots.expert_bias = kept.back().data_ptr<float>();
        }
    }
    const at::cuda::OptionalCUDAGuard device_guard(device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // The grids of entropy_sum and column_sums, so that both sums are theirs bit for bit.
    const int64_t sm_blocks = static_cast<int64_t>(sm_count_for_device(scores.get_device())) * kBlocksPerSm;
    const int64_t entropy_blocks = std::min<int64_t>(ceil_div(tokens * experts, kBlockThreads), sm_blocks);
    const int64_t rows_per_block = ceil_div(tokens, sm_blocks);
    const int64_t column_blocks = ceil_div(tokens, rows_per_block);
    const int64_t topk_work = std::max({tokens * topk.columns[0], tokens * topk.columns[1], tokens * topk.columns[2]});
    const int64_t topk_blocks = std::min<int64_t>(ceil_div(topk_work, kBlockThreads), sm_blocks);
    auto scratch = torch::empty({entropy_blocks + column_blocks * experts + kTopkSums * topk_blocks},
                                scores.options().dtype(at::kDouble));
    double* entropy_partials = scratch.data_ptr<double>();
    double* column_partials = entropy_partials + entropy_blocks;
    double* topk_partials = column_partials + column_blocks * experts;
    router_partials_kernel<<<std::max({entropy_blocks, column_blocks, topk_blocks}), kBlockThreads, 0, stream>>>(
        scores.data_ptr<float>(), scores.stride(0), biased.data_ptr<float>(), tokens, static_cast<int>(experts),
        entropy_blocks, rows_per_block, column_blocks, topk, token_mask, topk_blocks, entropy_partials, column_partials,
        topk_partials);
    router_finish_kernel<<<static_cast<unsigned int>(ceil_div(experts, kWarpSize)), kRouterFinishThreads, 0,
                           stream>>>(
        entropy_partials, static_cast<int>(entropy_blocks), column_partials, static_cast<int>(column_blocks),
        static_cast<int>(experts), topk_partials, static_cast<int>(topk_blocks), z_loss_scalar,
        seq_aux_loss_scalar, alpha, values.data_ptr<double>(),
        reinterpret_cast<long long*>(valid_tokens.data_ptr<int64_t>()), score_sums.data_ptr<double>(), first,
        slots);
}
