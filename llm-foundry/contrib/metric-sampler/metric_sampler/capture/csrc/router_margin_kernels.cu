#include "cuda_host_functions.h"
#include "radix_selection.cuh"

namespace {

// row_top_kernel keeps at most 32 columns of a row in each lane of a warp.
constexpr int kMaxTopColumns = 1024;

// The router margin statistics of the scores, then of the values: each source has a population of the gaps, of
// the near margins and of the margins, and these statistics. The q25 and q10 margins are interpolated among the
// margins of the unselected experts.
constexpr double kQuantile25 = 0.25;
constexpr double kQuantile10 = 0.10;
constexpr int kMarginSources = 2;
enum MarginKind { kGapKind, kNearKind, kMarginsKind };
constexpr int kMarginKinds = kMarginsKind + 1;
enum MarginStat { kGapMedian, kNearMedian, kQ25Margin, kQ10Margin };
constexpr int kMarginStats = kQ10Margin + 1;

// Float32 populations whose given ranks are selected together, each as order_statistics selects them: one launch
// per pass covers every population p (blockIdx.y of the counting and refining passes, blockIdx.x of the
// selections). Population p has its scratch words at p * kPopulationWords, its prefixes and ranks at
// p * kSelectionWords and its selected values at p * kSelectionTargets.
constexpr int kMaxPopulations = kMarginSources * kMarginKinds;
constexpr int64_t kPopulationWords =
    kScratchFirstDigit + kRadixBins + (kFloatPasses - 1) * kSelectionTargets * kRadixBins;

struct Populations {
    const float* values[kMaxPopulations];
    int64_t element_count[kMaxPopulations];
    GivenRanks given[kMaxPopulations];
};

__global__ void radix_count_populations_kernel(Populations populations, int32_t* __restrict__ words) {
    const int population_index = static_cast<int>(blockIdx.y);
    // A block past the population's elements has none to count: the grid is the largest population's.
    if (static_cast<int64_t>(blockIdx.x) * blockDim.x >= populations.element_count[population_index]) {
        return;
    }
    int32_t* scratch = words + population_index * kPopulationWords;
    radix_count<float, true>(populations.values[population_index], nullptr,
                             populations.element_count[population_index], scratch + kScratchCounts,
                             scratch + kScratchFirstDigit);
}

template <bool kFirstDigit>
__global__ void radix_refine_populations_kernel(
    Populations populations, int32_t* __restrict__ words, const int64_t* __restrict__ selection, int level) {
    const int population_index = static_cast<int>(blockIdx.y);
    if (static_cast<int64_t>(blockIdx.x) * blockDim.x >= populations.element_count[population_index]) {
        return;
    }
    int32_t* refined = words + population_index * kPopulationWords + kScratchFirstDigit + kRadixBins;
    radix_refine<float, true, kFirstDigit>(
        populations.values[population_index], nullptr, populations.element_count[population_index],
        selection + population_index * kSelectionWords, kKeyBits - kDigitBits * level,
        static_cast<uint32_t>((uint64_t{1} << (kDigitBits * level)) - 1), kFirstDigitShift - kDigitBits * level,
        refined + (level - 1) * kSelectionTargets * kRadixBins);
}

__global__ void radix_select_populations_kernel(
    Populations populations, const int32_t* __restrict__ words, int64_t* __restrict__ selection,
    float* __restrict__ selected, int pass) {
    const int population_index = static_cast<int>(blockIdx.x);
    const int32_t* scratch = words + population_index * kPopulationWords;
    const int32_t* histogram =
        pass == 0 ? scratch + kScratchFirstDigit
                  : scratch + kScratchFirstDigit + kRadixBins + (pass - 1) * kSelectionTargets * kRadixBins;
    int64_t* prefixes = selection + population_index * kSelectionWords;
    radix_select_targets(histogram, scratch + kScratchCounts, prefixes, prefixes + kSelectionTargets,
                         selected + population_index * kSelectionTargets, nullptr, pass, 0, pass == kFloatPasses - 1,
                         populations.given[population_index]);
}

// The k largest values of each row of a float32 [rows, columns] tensor, columns <= 32 * kPerLane, in descending
// order with NaN largest, the values torch.topk returns: one warp per row, lane l holding columns l, l + 32, ... as
// order keys sorted in descending order once; each of the k rounds takes the warp's largest first key, and its
// lane moves on to its next key.
template <int kPerLane, typename Visitor>
__device__ __forceinline__ void row_top(const float* __restrict__ row_ptr, int columns, int k, Visitor&& take) {
    const int lane = static_cast<int>(threadIdx.x % kWarpSize);
    // Key 0 belongs to no value (a negative NaN becomes the positive one), so it marks a column already taken or
    // past the end of the row.
    uint32_t keys[kPerLane];
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
        int column = lane + kWarpSize * i;
        keys[i] = column < columns ? population_key<true>(row_ptr[column]) : 0u;
    }
    // Odd-even transposition sort: kPerLane rounds of neighbor compare-exchanges.
#pragma unroll
    for (int round = 0; round < kPerLane; ++round) {
#pragma unroll
        for (int i = round % 2; i + 1 < kPerLane; i += 2) {
            const uint32_t larger = max(keys[i], keys[i + 1]);
            keys[i + 1] = min(keys[i], keys[i + 1]);
            keys[i] = larger;
        }
    }
    for (int j = 0; j < k; ++j) {
        uint32_t warp_best = keys[0];
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            warp_best = max(warp_best, __shfl_xor_sync(kFullWarpMask, warp_best, offset));
        }
        int winner = __ffs(__ballot_sync(kFullWarpMask, keys[0] == warp_best)) - 1;
        if (lane == winner) {
#pragma unroll
            for (int i = 0; i + 1 < kPerLane; ++i) {
                keys[i] = keys[i + 1];
            }
            keys[kPerLane - 1] = 0u;
        }
        take(j, from_ordered(static_cast<int32_t>(warp_best ^ kKeySignBit)));
    }
}

template <int kPerLane>
__global__ void row_top_kernel(
    const float* __restrict__ x, int64_t row_stride, int64_t rows, int columns, int k, float* __restrict__ out) {
    const int warp = static_cast<int>(threadIdx.x / kWarpSize);
    const int lane = static_cast<int>(threadIdx.x % kWarpSize);
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * kWarps + warp; row < rows;
         row += static_cast<int64_t>(gridDim.x) * kWarps) {
        row_top<kPerLane>(x + row * row_stride, columns, k, [&](int j, float value) {
            if (lane == 0) {
                out[row * k + j] = value;
            }
        });
    }
}

// The populations of the router margin statistics of one row of the router's float32 values or of their scores,
// written at the row's place in each: its gap, the top_k-th largest value (the boundary) less the next one; its
// near margins, the boundary less each of the next k - top_k values; and every value's margin, the boundary less
// the value, a NaN counting as +inf. The values in order are row_top_kernel's and the margins row_margins_kernel's.
struct MarginRows {
    float* gap;
    float* near;
    float* margins;
};

template <int kPerLane>
__device__ __forceinline__ void router_margin_row(
    const float* __restrict__ row_ptr, int64_t row, int columns, int top_k, int leading_count, MarginRows out) {
    const int lane = static_cast<int>(threadIdx.x % kWarpSize);
    float boundary = 0.0f;
    row_top<kPerLane>(row_ptr, columns, leading_count, [&](int j, float value) {
        if (j == top_k - 1) {
            boundary = value;
        }
        if (lane == 0 && j >= top_k) {
            out.near[row * (leading_count - top_k) + j - top_k] = boundary - value;
            if (j == top_k) {
                out.gap[row] = boundary - value;
            }
        }
    });
    for (int column = lane; column < columns; column += kWarpSize) {
        const float value = row_ptr[column];
        out.margins[row * columns + column] = boundary - (isnan(value) ? INFINITY : value);
    }
}

// One warp per row: the margin populations of the scores, then of the values.
template <int kPerLane>
__global__ void router_margin_rows_kernel(
    const float* __restrict__ scores, int64_t scores_stride, const float* __restrict__ values, int64_t values_stride,
    int64_t rows, int columns, int top_k, int leading_count, MarginRows score_rows, MarginRows value_rows) {
    const int warp = static_cast<int>(threadIdx.x / kWarpSize);
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * kWarps + warp; row < rows;
         row += static_cast<int64_t>(gridDim.x) * kWarps) {
        router_margin_row<kPerLane>(scores + row * scores_stride, row, columns, top_k, leading_count, score_rows);
        router_margin_row<kPerLane>(values + row * values_stride, row, columns, top_k, leading_count, value_rows);
    }
}

// torch.lerp of float32 values with a float32 weight (ATen's lerp).
__device__ __forceinline__ float torch_lerp(float self, float end, float weight) {
    return fabsf(weight) < 0.5f ? self + weight * (end - self) : end - (end - self) * (1.0f - weight);
}

// The eight statistics from the selected values of the six populations (gap, near margins and margins of the
// scores, then of the values): per source the median gap and near margin, the lower middle value unless the
// largest is NaN, as torch.median, and the q25 and q10 margins interpolated between their two ranks.
__global__ void router_margin_finish_kernel(
    const float* __restrict__ selected, float q25_weight, float q10_weight, float* __restrict__ out) {
    const int source = static_cast<int>(threadIdx.x);
    if (source < kMarginSources) {
        const float* gap = selected + (kMarginKinds * source) * kSelectionTargets;
        const float* near = gap + kSelectionTargets;
        const float* margins = near + kSelectionTargets;
        out[kMarginStats * source + kGapMedian] = isnan(gap[2]) ? gap[2] : gap[1];
        out[kMarginStats * source + kNearMedian] = isnan(near[2]) ? near[2] : near[1];
        out[kMarginStats * source + kQ25Margin] = torch_lerp(margins[1], margins[2], q25_weight);
        out[kMarginStats * source + kQ10Margin] = torch_lerp(margins[3], margins[4], q10_weight);
    }
}

// boundary[r] - x[r, c] in fp32 for every element of a float32 [rows, columns] tensor, a NaN of x counting as
// +inf, like boundary[:, None] - torch.where(x.isnan(), inf, x): one warp per row.
__global__ void row_margins_kernel(
    const float* __restrict__ x, int64_t row_stride, int64_t rows, int columns, const float* __restrict__ boundary,
    int64_t boundary_stride, float* __restrict__ out) {
    const int warp = static_cast<int>(threadIdx.x / kWarpSize);
    const int lane = static_cast<int>(threadIdx.x % kWarpSize);
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * kWarps + warp; row < rows;
         row += static_cast<int64_t>(gridDim.x) * kWarps) {
        const float* row_ptr = x + row * row_stride;
        const float row_boundary = boundary[row * boundary_stride];
        for (int column = lane; column < columns; column += kWarpSize) {
            const float value = row_ptr[column];
            out[row * columns + column] = row_boundary - (isnan(value) ? INFINITY : value);
        }
    }
}

}  // namespace

torch::Tensor row_top_values(torch::Tensor matrix, int64_t k) {
    check_rows(matrix);
    TORCH_CHECK(matrix.scalar_type() == at::kFloat, "row_top_values takes float32 rows");
    const int64_t column_count = matrix.size(1);
    TORCH_CHECK(column_count <= kMaxTopColumns && k >= 1 && k <= column_count,
                "row_top_values needs 1 <= k <= n <= 1024, got k=", k, " n=", column_count);
    const at::cuda::OptionalCUDAGuard device_guard(matrix.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto out = torch::empty({matrix.size(0), k}, matrix.options());
    const int64_t grid = row_grid(matrix);
    const int64_t per_lane = ceil_div(column_count, kWarpSize);
    const float* values = matrix.data_ptr<float>();
    const int64_t rows = matrix.size(0), row_stride = matrix.stride(0);
    float* out_ptr = out.data_ptr<float>();
    const int columns = static_cast<int>(column_count), selected = static_cast<int>(k);
    if (per_lane <= 1) {
        row_top_kernel<1><<<grid, kBlockThreads, 0, stream>>>(values, row_stride, rows, columns, selected, out_ptr);
    } else if (per_lane <= 2) {
        row_top_kernel<2><<<grid, kBlockThreads, 0, stream>>>(values, row_stride, rows, columns, selected, out_ptr);
    } else if (per_lane <= 4) {
        row_top_kernel<4><<<grid, kBlockThreads, 0, stream>>>(values, row_stride, rows, columns, selected, out_ptr);
    } else if (per_lane <= 8) {
        row_top_kernel<8><<<grid, kBlockThreads, 0, stream>>>(values, row_stride, rows, columns, selected, out_ptr);
    } else if (per_lane <= 16) {
        row_top_kernel<16><<<grid, kBlockThreads, 0, stream>>>(values, row_stride, rows, columns, selected, out_ptr);
    } else {
        row_top_kernel<32><<<grid, kBlockThreads, 0, stream>>>(values, row_stride, rows, columns, selected, out_ptr);
    }
    return out;
}

torch::Tensor row_margins(torch::Tensor matrix, torch::Tensor boundary) {
    check_rows(matrix);
    TORCH_CHECK(matrix.scalar_type() == at::kFloat, "row_margins takes float32 rows");
    TORCH_CHECK(boundary.is_cuda() && boundary.get_device() == matrix.get_device() &&
                    boundary.scalar_type() == at::kFloat && boundary.dim() == 1 && boundary.size(0) == matrix.size(0),
                "boundary must be one float32 value per row on the rows' device");
    const at::cuda::OptionalCUDAGuard device_guard(matrix.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto out = torch::empty({matrix.size(0), matrix.size(1)}, matrix.options());
    row_margins_kernel<<<row_grid(matrix), kBlockThreads, 0, stream>>>(
        matrix.data_ptr<float>(), matrix.stride(0), matrix.size(0), static_cast<int>(matrix.size(1)),
        boundary.data_ptr<float>(), boundary.stride(0), out.data_ptr<float>());
    return out;
}

bool router_margin_into(
    torch::Tensor logits, int64_t top_k, int64_t near_margin_count, torch::Tensor buffer, int64_t offset) {
    at::NoGradGuard no_grad;
    TORCH_CHECK(logits.is_cuda() && logits.dim() == 2 && logits.size(0) > 0, "logits must be non-empty 2D CUDA rows");
    const int64_t rows = logits.size(0), experts = logits.size(1);
    TORCH_CHECK(top_k >= 1 && top_k < experts && near_margin_count >= 1,
                "router margins need 1 <= top_k < experts and near_count >= 1");
    TORCH_CHECK(buffer.device() == logits.device() && buffer.scalar_type() == at::kFloat && buffer.is_contiguous() &&
                    buffer.dim() == 1 && offset >= 0 && offset + kMarginSources * kMarginStats <= buffer.numel(),
                "buffer must be the recorder's float32 buffer on the logits' device with room for eight values");
    // Beyond row_top_kernel's columns or int32 population counts the CUDA graph computes them.
    if (experts > kMaxTopColumns || rows * experts >= std::numeric_limits<int32_t>::max()) {
        return false;
    }
    torch::Tensor values = logits.to(at::kFloat);
    if (values.stride(1) != 1) {
        values = values.contiguous();
    }
    // torch.sigmoid's own kernel: the scores are the graph's bit for bit.
    torch::Tensor scores = at::sigmoid(values);
    if (scores.stride(1) != 1) {
        scores = scores.contiguous();
    }
    const int64_t leading_count = std::min(top_k + near_margin_count, experts);
    const int64_t sizes[kMarginKinds] = {rows, rows * (leading_count - top_k), rows * experts};
    const int64_t floats = kMarginSources * (sizes[kGapKind] + sizes[kNearKind] + sizes[kMarginsKind]) +
                           kMaxPopulations * kSelectionTargets;
    const int64_t words = kMaxPopulations * kPopulationWords;
    // One allocation in 8-byte units: the selections, the zeroed scratch words, then the populations and the
    // selected values.
    auto scratch = torch::empty({kMaxPopulations * kSelectionWords + (words + 1) / 2 + (floats + 1) / 2},
                                logits.options().dtype(at::kLong));
    int64_t* selection = scratch.data_ptr<int64_t>();
    int32_t* scratch_words = reinterpret_cast<int32_t*>(selection + kMaxPopulations * kSelectionWords);
    float* population_values = reinterpret_cast<float*>(scratch_words + 2 * ((words + 1) / 2));
    Populations populations{};
    MarginRows margin_rows[kMarginSources];
    float* next = population_values;
    for (int source = 0; source < kMarginSources; ++source) {
        float* starts[kMarginKinds];
        for (int kind = 0; kind < kMarginKinds; ++kind) {
            const int population_index = kMarginKinds * source + kind;
            starts[kind] = next;
            populations.values[population_index] = next;
            populations.element_count[population_index] = sizes[kind];
            next += sizes[kind];
        }
        margin_rows[source] = MarginRows{starts[kGapKind], starts[kNearKind], starts[kMarginsKind]};
    }
    float* selected = next;
    // The ranks the graph selects: medians as torch.median, the lower middle and the last value; the q25 and q10
    // margins between the floor and ceil of their positions among the unselected experts' margins, which follow
    // the rows * top_k selected ones.
    const double positions[2] = {static_cast<double>(rows * (experts - top_k) - 1) * kQuantile25,
                                 static_cast<double>(rows * (experts - top_k) - 1) * kQuantile10};
    for (int population_index = 0; population_index < kMaxPopulations; ++population_index) {
        GivenRanks& given = populations.given[population_index];
        if (population_index % kMarginKinds < kMarginsKind) {
            given.count = 2;
            given.rank[0] = (populations.element_count[population_index] - 1) / 2;
            given.rank[1] = populations.element_count[population_index] - 1;
        } else {
            given.count = 4;
            for (int quantile = 0; quantile < 2; ++quantile) {
                given.rank[2 * quantile] = rows * top_k + static_cast<int64_t>(std::floor(positions[quantile]));
                given.rank[2 * quantile + 1] = rows * top_k + static_cast<int64_t>(std::ceil(positions[quantile]));
            }
        }
    }
    const at::cuda::OptionalCUDAGuard device_guard(logits.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemsetAsync(scratch_words, 0, words * sizeof(int32_t), stream));
    const int64_t grid = row_grid(rows, logits.get_device());
    const int64_t per_lane = ceil_div(experts, kWarpSize);
    const float* scores_ptr = scores.data_ptr<float>();
    const float* values_ptr = values.data_ptr<float>();
#define ROUTER_MARGIN_ROWS(kPerLane)                                                                            \
    router_margin_rows_kernel<kPerLane><<<grid, kBlockThreads, 0, stream>>>(                                    \
        scores_ptr, scores.stride(0), values_ptr, values.stride(0), rows, static_cast<int>(experts),            \
        static_cast<int>(top_k), static_cast<int>(leading_count), margin_rows[0], margin_rows[1])
    if (per_lane <= 1) {
        ROUTER_MARGIN_ROWS(1);
    } else if (per_lane <= 2) {
        ROUTER_MARGIN_ROWS(2);
    } else if (per_lane <= 4) {
        ROUTER_MARGIN_ROWS(4);
    } else if (per_lane <= 8) {
        ROUTER_MARGIN_ROWS(8);
    } else if (per_lane <= 16) {
        ROUTER_MARGIN_ROWS(16);
    } else {
        ROUTER_MARGIN_ROWS(32);
    }
#undef ROUTER_MARGIN_ROWS
    const dim3 passes(static_cast<unsigned int>(std::min<int64_t>(
                          ceil_div(sizes[kMarginsKind], kBlockThreads),
                          static_cast<int64_t>(sm_count_for_device(logits.get_device())) * kBlocksPerSm)),
                      kMaxPopulations);
    radix_count_populations_kernel<<<passes, kBlockThreads, 0, stream>>>(populations, scratch_words);
    radix_select_populations_kernel<<<kMaxPopulations, kSelectionThreads, 0, stream>>>(
        populations, scratch_words, selection, selected, 0);
    for (int level = 1; level < kFloatPasses; ++level) {
        if (level == 1) {
            radix_refine_populations_kernel<true><<<passes, kBlockThreads, 0, stream>>>(
                populations, scratch_words, selection, level);
        } else {
            radix_refine_populations_kernel<false><<<passes, kBlockThreads, 0, stream>>>(
                populations, scratch_words, selection, level);
        }
        radix_select_populations_kernel<<<kMaxPopulations, kSelectionThreads, 0, stream>>>(
            populations, scratch_words, selection, selected, level);
    }
    // The weights torch.lerp is given: the positions' fractions, as float32.
    router_margin_finish_kernel<<<1, kWarpSize, 0, stream>>>(
        selected, static_cast<float>(positions[0] - std::floor(positions[0])),
        static_cast<float>(positions[1] - std::floor(positions[1])), buffer.data_ptr<float>() + offset);
    return true;
}
