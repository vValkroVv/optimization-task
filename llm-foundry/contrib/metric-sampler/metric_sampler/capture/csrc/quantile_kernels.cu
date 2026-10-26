#include "cuda_host_functions.h"
#include "key_histogram.cuh"

namespace {

constexpr int kKey64Bits = 64;
constexpr unsigned long long kKeySignBit64 = 0x8000000000000000ull;
constexpr long long kMagnitudeMask64 = 0x7FFFFFFFFFFFFFFFll;

constexpr int kUnmaskedCountKinds = kMaskedCount;

template <typename T>
__global__ void __launch_bounds__(kKeyThreads) key_finalize_kernel(
    unsigned long long* __restrict__ histogram,
    float* __restrict__ buffer,
    int64_t moments_offset,
    int64_t counts_offset,
    int64_t amax_offset,
    int64_t power_sums_offset,
    int bins_start,
    int bins_count,
    float* __restrict__ quantiles,
    int32_t* __restrict__ counts_out,
    bool clear) {
    extern __shared__ int exponent_histogram[];
    key_finalize_block<T>(histogram, buffer, moments_offset, counts_offset, amax_offset, power_sums_offset, bins_start,
                          bins_count, quantiles, counts_out, clear, exponent_histogram);
}

template <typename T, bool kAllValues>
__global__ void radix_count_kernel(
    const T* __restrict__ values,
    const bool* __restrict__ valid,
    int64_t element_count,
    int32_t* __restrict__ counts,
    int32_t* __restrict__ histogram) {
    radix_count<T, kAllValues>(values, valid, element_count, counts, histogram);
}

template <typename T, bool kAllValues, bool kFirstDigit>
__global__ void radix_refine_kernel(
    const T* __restrict__ values,
    const bool* __restrict__ valid,
    int64_t element_count,
    const int64_t* __restrict__ prefixes,
    int prefix_shift,
    uint32_t prefix_mask,
    int shift,
    int32_t* __restrict__ histogram) {
    radix_refine<T, kAllValues, kFirstDigit>(values, valid, element_count, prefixes, prefix_shift, prefix_mask, shift,
                                             histogram);
}

__global__ void radix_select_kernel(
    const int32_t* __restrict__ histogram,
    const int32_t* __restrict__ counts,
    int64_t* __restrict__ prefixes,
    int64_t* __restrict__ ranks,
    float* __restrict__ quantiles,
    int32_t* __restrict__ counts_out,
    int pass,
    int final_shift,
    bool final_pass,
    GivenRanks given) {
    radix_select_targets(histogram, counts, prefixes, ranks, quantiles, counts_out, pass, final_shift, final_pass,
                         given);
}

template <typename T, bool kAllValues>
void launch_radix_quantiles(
    const T* values,
    const bool* valid,
    int64_t element_count,
    int passes,
    int64_t grid,
    cudaStream_t stream,
    int32_t* counts,
    int32_t* first_digit_histogram,
    int32_t* refined,
    int64_t* prefixes,
    int64_t* ranks,
    float* quantiles,
    int32_t* counts_out,
    bool counted,
    GivenRanks given) {
    const int final_shift = kKeyBits - kDigitBits * passes;
    if (!counted) {
        radix_count_kernel<T, kAllValues><<<grid, kBlockThreads, 0, stream>>>(values, valid, element_count, counts,
                                                                             first_digit_histogram);
    }
    radix_select_kernel<<<1, kSelectionThreads, 0, stream>>>(
        first_digit_histogram, counts, prefixes, ranks, quantiles, counts_out, 0, final_shift, passes == 1, given);
    for (int level = 1; level < passes; ++level) {
        int32_t* histogram = refined + (level - 1) * kSelectionTargets * kRadixBins;
        uint32_t prefix_mask = static_cast<uint32_t>((uint64_t{1} << (kDigitBits * level)) - 1);
        if (level == 1) {
            radix_refine_kernel<T, kAllValues, true><<<grid, kBlockThreads, 0, stream>>>(
                values, valid, element_count, prefixes, kKeyBits - kDigitBits * level, prefix_mask,
                kFirstDigitShift - kDigitBits * level, histogram);
        } else {
            radix_refine_kernel<T, kAllValues, false><<<grid, kBlockThreads, 0, stream>>>(
                values, valid, element_count, prefixes, kKeyBits - kDigitBits * level, prefix_mask,
                kFirstDigitShift - kDigitBits * level, histogram);
        }
        radix_select_kernel<<<1, kSelectionThreads, 0, stream>>>(
            histogram, counts, prefixes, ranks, quantiles, counts_out, level, final_shift, level == passes - 1, given);
    }
}

constexpr int kSmallThreads = 1024;

// The same selection as launch_radix_quantiles, every pass in one block of kSmallThreads.
template <typename T, bool kAllValues>
__global__ void radix_select_small_kernel(
    const T* __restrict__ values,
    const bool* __restrict__ valid,
    int64_t element_count,
    int passes,
    GivenRanks given,
    float* __restrict__ quantiles,
    int32_t* __restrict__ counts_out) {
    __shared__ int32_t histogram[kQuantileTargets][kRadixBins];
    __shared__ int32_t shared_counts[kCountKinds];
    __shared__ int32_t shared_extremes[2];
    __shared__ int64_t prefixes[kQuantileTargets];
    __shared__ int64_t previous[kQuantileTargets];
    __shared__ int64_t ranks[kQuantileTargets];
    const int final_shift = kKeyBits - kDigitBits * passes;
    for (int i = threadIdx.x; i < kRadixBins; i += blockDim.x) {
        histogram[0][i] = 0;
    }
    if (threadIdx.x < kCountKinds) {
        shared_counts[threadIdx.x] = 0;
    }
    if (threadIdx.x == 0) {
        shared_extremes[0] = std::numeric_limits<int32_t>::max();
        shared_extremes[1] = std::numeric_limits<int32_t>::min();
    }
    __syncthreads();

    int finite = 0, nan = 0, positive = 0, negative = 0, masked = 0;
    int32_t smallest = std::numeric_limits<int32_t>::max();
    int32_t largest = std::numeric_limits<int32_t>::min();
    for_each_element(values, valid, element_count, [&](float value, bool is_valid) {
        if (!is_valid) {
            ++masked;
            return;
        }
        if (!kAllValues && isnan(value)) {
            ++nan;
            return;
        }
        if (!kAllValues && isinf(value)) {
            value > 0 ? ++positive : ++negative;
            return;
        }
        ++finite;
        uint32_t key = population_key<kAllValues>(value);
        int32_t ordered = static_cast<int32_t>(key ^ kKeySignBit);
        smallest = min(smallest, ordered);
        largest = max(largest, ordered);
        atomicAdd(&histogram[0][key >> kFirstDigitShift], 1);
    });
    finite = warp_reduce_sum(finite);
    nan = warp_reduce_sum(nan);
    positive = warp_reduce_sum(positive);
    negative = warp_reduce_sum(negative);
    masked = warp_reduce_sum(masked);
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        smallest = min(smallest, __shfl_down_sync(kFullWarpMask, smallest, offset));
        largest = max(largest, __shfl_down_sync(kFullWarpMask, largest, offset));
    }
    if ((threadIdx.x & (kWarpSize - 1)) == 0) {
        atomicAdd(&shared_counts[kFiniteCount], finite);
        atomicAdd(&shared_counts[kNanCount], nan);
        atomicAdd(&shared_counts[kPositiveInfinityCount], positive);
        atomicAdd(&shared_counts[kNegativeInfinityCount], negative);
        atomicAdd(&shared_counts[kMaskedCount], masked);
        atomicMin(&shared_extremes[0], smallest);
        atomicMax(&shared_extremes[1], largest);
    }
    __syncthreads();
    const int64_t population = shared_counts[kFiniteCount];

    for (int level = 0; level < passes; ++level) {
        if (level > 0) {
            for (int i = threadIdx.x; i < kQuantileTargets * kRadixBins; i += blockDim.x) {
                histogram[i / kRadixBins][i % kRadixBins] = 0;
            }
            if (threadIdx.x < kQuantileTargets) {
                previous[threadIdx.x] = prefixes[threadIdx.x];
            }
            __syncthreads();
            const int prefix_shift = kKeyBits - kDigitBits * level;
            const uint32_t prefix_mask = static_cast<uint32_t>((uint64_t{1} << (kDigitBits * level)) - 1);
            const int shift = kFirstDigitShift - kDigitBits * level;
            for_each_element(values, valid, element_count, [&](float value, bool is_valid) {
                if (!is_valid || (!kAllValues && (isnan(value) || isinf(value)))) {
                    return;
                }
                uint32_t key = population_key<kAllValues>(value);
                uint32_t high = (key >> prefix_shift) & prefix_mask;
                int row = -1;
#pragma unroll
                for (int target = kQuantileTargets - 1; target >= 0; --target) {
                    if (high == static_cast<uint32_t>(previous[target])) {
                        row = target;
                    }
                }
                if (row >= 0) {
                    atomicAdd(&histogram[row][(key >> shift) & (kRadixBins - 1)], 1);
                }
            });
            __syncthreads();
        }
        if (threadIdx.x < kWarpSize) {
            for (int target = 0; target < kQuantileTargets; ++target) {
                int64_t rank;
                int row = 0;
                if (level > 0) {
                    rank = ranks[target];
                    row = target;
                    for (int earlier = target - 1; earlier >= 0; --earlier) {
                        if (previous[earlier] == previous[target]) {
                            row = earlier;
                        }
                    }
                } else if (given.count > 0) {
                    rank = target < given.count ? given.rank[target] : -1;
                } else {
                    rank = max((population * kInnerPercentiles[target] + kPercentRoundUp) / kPercentScale - 1,
                               int64_t{0});
                }
                int64_t bin, below;
                warp_select_bin(histogram[row], rank, bin, below);
                if (threadIdx.x == 0) {
                    prefixes[target] = ((level == 0 ? int64_t{0} : previous[target]) << kDigitBits) | bin;
                    ranks[target] = rank - below;
                }
                __syncwarp();
            }
        }
        __syncthreads();
    }
    if (threadIdx.x < kQuantileTargets) {
        quantiles[1 + threadIdx.x] = population > 0 ? prefix_value(prefixes[threadIdx.x], final_shift) : NAN;
    }
    if (threadIdx.x == 0) {
        quantiles[0] = population > 0 ? from_ordered(shared_extremes[0]) : NAN;
        quantiles[kQuantileTargets + 1] = population > 0 ? from_ordered(shared_extremes[1]) : NAN;
    }
    if (counts_out != nullptr && threadIdx.x < kCountKinds) {
        counts_out[threadIdx.x] = shared_counts[threadIdx.x];
    }
}

// A float64 population this small is selected by exact_quantiles_float64_kernel; a larger one is sorted.
constexpr int64_t kFloat64SmallPopulation = int64_t{1} << 17;

__device__ __forceinline__ uint64_t float64_key(double value) {
    const int64_t bits = __double_as_longlong(value);
    return static_cast<uint64_t>(bits ^ ((bits >> (kKey64Bits - 1)) & kMagnitudeMask64)) ^ kKeySignBit64;
}

__device__ __forceinline__ double float64_from_key(uint64_t key) {
    const int64_t ordered = static_cast<int64_t>(key ^ kKeySignBit64);
    return __longlong_as_double(ordered ^ ((ordered >> (kKey64Bits - 1)) & kMagnitudeMask64));
}

// exact_quantiles of a float64 population in one block of kSmallThreads: the [finite, NaN, +Inf, -Inf, masked]
// counts and q0..q100 of the finite values, each the value at its rank in the sorted population, selected by
// eight passes over 8-bit key digits. The quantiles go out as the words of float64 values, so that they can be
// written into a recorder slot at any word offset.
__global__ void __launch_bounds__(kSmallThreads) exact_quantiles_float64_kernel(
    const double* __restrict__ values, int64_t element_count, int32_t* __restrict__ quantile_words,
    int32_t* __restrict__ counts_out) {
    __shared__ int32_t histogram[kSelectionTargets][kRadixBins];
    __shared__ int32_t shared_counts[kUnmaskedCountKinds];
    __shared__ uint64_t prefixes[kSelectionTargets];
    __shared__ uint64_t previous[kSelectionTargets];
    __shared__ int64_t ranks[kSelectionTargets];
    for (int i = threadIdx.x; i < kSelectionTargets * kRadixBins; i += blockDim.x) {
        histogram[i / kRadixBins][i % kRadixBins] = 0;
    }
    if (threadIdx.x < kUnmaskedCountKinds) {
        shared_counts[threadIdx.x] = 0;
    }
    __syncthreads();
    int finite = 0, nan = 0, positive = 0, negative = 0;
    for (int64_t i = threadIdx.x; i < element_count; i += blockDim.x) {
        const double value = values[i];
        if (isnan(value)) {
            ++nan;
        } else if (isinf(value)) {
            value > 0 ? ++positive : ++negative;
        } else {
            ++finite;
            atomicAdd(&histogram[0][float64_key(value) >> (kKey64Bits - kDigitBits)], 1);
        }
    }
    finite = warp_reduce_sum(finite);
    nan = warp_reduce_sum(nan);
    positive = warp_reduce_sum(positive);
    negative = warp_reduce_sum(negative);
    if ((threadIdx.x & (kWarpSize - 1)) == 0) {
        atomicAdd(&shared_counts[kFiniteCount], finite);
        atomicAdd(&shared_counts[kNanCount], nan);
        atomicAdd(&shared_counts[kPositiveInfinityCount], positive);
        atomicAdd(&shared_counts[kNegativeInfinityCount], negative);
    }
    __syncthreads();
    const int64_t population = shared_counts[kFiniteCount];
    for (int level = 0; level < kKey64Bits / kDigitBits; ++level) {
        if (level > 0) {
            for (int i = threadIdx.x; i < kSelectionTargets * kRadixBins; i += blockDim.x) {
                histogram[i / kRadixBins][i % kRadixBins] = 0;
            }
            if (threadIdx.x < kSelectionTargets) {
                previous[threadIdx.x] = prefixes[threadIdx.x];
            }
            __syncthreads();
            const int shift = kKey64Bits - kDigitBits - kDigitBits * level;
            for (int64_t i = threadIdx.x; i < element_count; i += blockDim.x) {
                const double value = values[i];
                if (isnan(value) || isinf(value)) {
                    continue;
                }
                const uint64_t key = float64_key(value);
                const uint64_t high = key >> (shift + kDigitBits);
                int row = -1;
#pragma unroll
                for (int target = kSelectionTargets - 1; target >= 0; --target) {
                    if (high == previous[target]) {
                        row = target;
                    }
                }
                if (row >= 0) {
                    atomicAdd(&histogram[row][(key >> shift) & (kRadixBins - 1)], 1);
                }
            }
            __syncthreads();
        }
        // One warp per target.
        const int target = static_cast<int>(threadIdx.x / kWarpSize);
        if (target < kSelectionTargets) {
            int64_t rank;
            int row = 0;
            if (level > 0) {
                rank = ranks[target];
                row = target;
                for (int earlier = target - 1; earlier >= 0; --earlier) {
                    if (previous[earlier] == previous[target]) {
                        row = earlier;
                    }
                }
            } else if (target == 0 || target == kSelectionTargets - 1) {
                rank = target == 0 ? 0 : population - 1;
            } else {
                rank = max((population * kInnerPercentiles[target - 1] + kPercentRoundUp) / kPercentScale - 1,
                           int64_t{0});
            }
            int64_t bin, below;
            warp_select_bin(histogram[row], rank, bin, below);
            if ((threadIdx.x & (kWarpSize - 1)) == 0) {
                // A rank outside the population selects bin kRadixBins, a prefix no key has.
                prefixes[target] =
                    ((level == 0 ? uint64_t{0} : previous[target]) << kDigitBits) | static_cast<uint64_t>(bin);
                ranks[target] = rank - below;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x < kSelectionTargets) {
        store_double_words(quantile_words + 2 * threadIdx.x,
                           population > 0 ? float64_from_key(prefixes[threadIdx.x]) : NAN);
    }
    if (threadIdx.x < kCountKinds) {
        counts_out[threadIdx.x] = threadIdx.x < kUnmaskedCountKinds ? shared_counts[threadIdx.x] : 0;
    }
}

// The entropy of each row of a float32 [rows, columns] tensor taken as weights, divided by log(columns), in fp64
// like -xlogy(p, p).sum(-1) / log(columns) with p = x.double() / x.double().sum(-1) in torch: one warp per
// row, lane l summing columns l, l + 32, ... and the lanes meeting in a fixed tree.
__global__ void row_entropy_kernel(
    const float* __restrict__ x, int64_t row_stride, int64_t rows, int columns, double log_columns,
    double* __restrict__ out) {
    const int warp = static_cast<int>(threadIdx.x / kWarpSize);
    const int lane = static_cast<int>(threadIdx.x % kWarpSize);
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * kWarps + warp; row < rows;
         row += static_cast<int64_t>(gridDim.x) * kWarps) {
        const float* row_ptr = x + row * row_stride;
        double total = 0.0;
        for (int column = lane; column < columns; column += kWarpSize) {
            total += static_cast<double>(row_ptr[column]);
        }
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            total += __shfl_xor_sync(kFullWarpMask, total, offset);
        }
        double sum = 0.0;
        for (int column = lane; column < columns; column += kWarpSize) {
            const double probability = static_cast<double>(row_ptr[column]) / total;
            sum += probability == 0.0 ? 0.0 : __dmul_rn(probability, log(probability));
        }
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(kFullWarpMask, sum, offset);
        }
        if (lane == 0) {
            out[row] = -sum / log_columns;
        }
    }
}

// Shared by exact_quantiles (no given ranks: the inner percentiles of the finite valid values) and
// order_statistics (given ranks of every value). Writes the nine selected values to `quantiles` and, unless
// `counts_out` is null, the five counts to `counts_out`. A non-null `first_pass` is a scratch whose counts and
// first digit a statistics pass already counted.
template <bool kAllValues>
void radix_select(
    torch::Tensor values, const bool* valid_ptr, GivenRanks given, float* quantiles, int32_t* counts_out,
    int32_t* first_pass) {
    const int64_t element_count = values.numel();
    const int passes = radix_passes(values.scalar_type());
    const at::cuda::OptionalCUDAGuard device_guard(values.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (element_count <= kSmallPopulation) {
        switch (values.scalar_type()) {
            case at::kBFloat16:
                radix_select_small_kernel<at::BFloat16, kAllValues><<<1, kSmallThreads, 0, stream>>>(
                    values.data_ptr<at::BFloat16>(), valid_ptr, element_count, passes, given, quantiles, counts_out);
                break;
            case at::kHalf:
                radix_select_small_kernel<at::Half, kAllValues><<<1, kSmallThreads, 0, stream>>>(
                    values.data_ptr<at::Half>(), valid_ptr, element_count, passes, given, quantiles, counts_out);
                break;
            case at::kFloat:
                radix_select_small_kernel<float, kAllValues><<<1, kSmallThreads, 0, stream>>>(
                    values.data_ptr<float>(), valid_ptr, element_count, passes, given, quantiles, counts_out);
                break;
            default:
                TORCH_CHECK(false, "activation_ext: unsupported radix selection dtype ", values.scalar_type());
        }
        return;
    }
    torch::Tensor scratch;
    if (first_pass == nullptr) {
        scratch = torch::zeros({radix_scratch_words(passes)}, values.options().dtype(torch::kInt32));
    }
    auto selection = torch::empty({kSelectionWords}, values.options().dtype(torch::kInt64));

    int32_t* scratch_ptr = first_pass == nullptr ? scratch.data_ptr<int32_t>() : first_pass;
    int32_t* counts = scratch_ptr + kScratchCounts;
    int32_t* first_digit_histogram = scratch_ptr + kScratchFirstDigit;
    int32_t* refined = first_digit_histogram + kRadixBins;
    const bool counted = first_pass != nullptr;
    int64_t* selection_ptr = selection.data_ptr<int64_t>();
    int64_t blocks = ceil_div(element_count, kBlockThreads);
    int64_t grid =
        std::min<int64_t>(blocks, static_cast<int64_t>(sm_count_for_device(values.get_device())) * kBlocksPerSm);
    switch (values.scalar_type()) {
        case at::kBFloat16:
            launch_radix_quantiles<at::BFloat16, kAllValues>(
                values.data_ptr<at::BFloat16>(), valid_ptr, element_count, passes, grid, stream, counts,
                first_digit_histogram, refined, selection_ptr, selection_ptr + kSelectionTargets, quantiles,
                counts_out, counted, given);
            break;
        case at::kHalf:
            launch_radix_quantiles<at::Half, kAllValues>(
                values.data_ptr<at::Half>(), valid_ptr, element_count, passes, grid, stream, counts,
                first_digit_histogram, refined, selection_ptr, selection_ptr + kSelectionTargets, quantiles,
                counts_out, counted, given);
            break;
        case at::kFloat:
            launch_radix_quantiles<float, kAllValues>(
                values.data_ptr<float>(), valid_ptr, element_count, passes, grid, stream, counts,
                first_digit_histogram, refined, selection_ptr, selection_ptr + kSelectionTargets, quantiles,
                counts_out, counted, given);
            break;
        default:
            TORCH_CHECK(false, "activation_ext: unsupported radix selection dtype ", values.scalar_type());
    }
}

void check_population(const torch::Tensor& values) {
    TORCH_CHECK(values.is_cuda() && values.is_contiguous(), "values must be a contiguous CUDA tensor");
    TORCH_CHECK(values.numel() > 0 && values.numel() <= std::numeric_limits<int32_t>::max(),
                "population size must fit int32 counts");
}

}  // namespace

void exact_quantiles(
    torch::Tensor population, c10::optional<torch::Tensor> mask, torch::Tensor quantiles, torch::Tensor counts,
    c10::optional<torch::Tensor> counted) {
    // Flattened here because each of these operations costs a dispatch in Python; a uniformly strided population
    // flattens to a strided view, hence the copy.
    const torch::Tensor values = population.detach().reshape({-1}).contiguous();
    const torch::Tensor valid = mask.has_value() ? mask->reshape({-1}).contiguous() : torch::Tensor();
    const torch::Tensor first_pass = counted.has_value() ? *counted : torch::Tensor();
    check_population(values);
    TORCH_CHECK(!valid.defined() || (valid.is_cuda() && valid.scalar_type() == torch::kBool &&
                                     valid.is_contiguous() && valid.numel() == values.numel()),
                "valid must be None or a contiguous boolean CUDA tensor shaped like values");
    TORCH_CHECK(quantiles.device() == values.device() && quantiles.scalar_type() == torch::kFloat32 &&
                    quantiles.is_contiguous() && quantiles.numel() == kQuantileTargets + 2,
                "quantiles must be nine contiguous float32 values on the population's device");
    TORCH_CHECK(counts.device() == values.device() && counts.scalar_type() == torch::kInt32 && counts.is_contiguous() &&
                    counts.numel() == kCountKinds,
                "counts must be five contiguous int32 values on the population's device");
    const bool key_counts = first_pass.defined() && first_pass.scalar_type() == torch::kInt64;
    TORCH_CHECK(!first_pass.defined() ||
                    (!valid.defined() && first_pass.device() == values.device() && first_pass.is_contiguous() &&
                     (key_counts ? has_16_bit_keys(values.scalar_type()) && first_pass.numel() == kKeyScratchWords
                                 : first_pass.scalar_type() == torch::kInt32 &&
                                       first_pass.numel() == radix_scratch_words(radix_passes(values.scalar_type())))),
                "first_pass must be None or observe's selection scratch for this unmasked population");
    if (key_counts) {
        const at::cuda::OptionalCUDAGuard device_guard(values.device());
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        auto* histogram = reinterpret_cast<unsigned long long*>(first_pass.data_ptr<int64_t>());
        if (values.scalar_type() == at::kBFloat16) {
            key_finalize_kernel<at::BFloat16><<<1, kKeyThreads, 0, stream>>>(
                histogram, nullptr, 0, 0, 0, 0, 0, 0, quantiles.data_ptr<float>(), counts.data_ptr<int32_t>(), false);
        } else {
            key_finalize_kernel<at::Half><<<1, kKeyThreads, 0, stream>>>(
                histogram, nullptr, 0, 0, 0, 0, 0, 0, quantiles.data_ptr<float>(), counts.data_ptr<int32_t>(), false);
        }
        return;
    }
    const bool* valid_ptr = valid.defined() ? valid.data_ptr<bool>() : nullptr;
    radix_select<false>(values, valid_ptr, GivenRanks{}, quantiles.data_ptr<float>(), counts.data_ptr<int32_t>(),
                        first_pass.defined() ? first_pass.data_ptr<int32_t>() : nullptr);
}

void observe_with_quantiles(
    torch::Tensor tensor,
    torch::Tensor buffer,
    int64_t moments_offset,
    int64_t counts_offset,
    int64_t amax_offset,
    int64_t power_sums_offset,
    int64_t bins_start,
    int64_t bins_count,
    int64_t quantiles_offset,
    int64_t quantile_counts_offset,
    bool key_histogram) {
    at::NoGradGuard no_grad;
    TORCH_CHECK(tensor.numel() > 0, "observe_with_quantiles needs a non-empty tensor");
    TORCH_CHECK(quantiles_offset >= 0 && quantiles_offset + kQuantileTargets + 2 <= buffer.numel() &&
                    quantile_counts_offset >= 0 && quantile_counts_offset + kCountKinds <= buffer.numel(),
                "quantile offsets are outside buffer");
    float* base = buffer.data_ptr<float>();
    int32_t* quantile_counts = reinterpret_cast<int32_t*>(base + quantile_counts_offset);
    const torch::Tensor first_pass = observe_impl(
        tensor, buffer, moments_offset, counts_offset, amax_offset, power_sums_offset, bins_start, bins_count, true,
        key_histogram, base + quantiles_offset, quantile_counts);
    if (!first_pass.defined()) {
        return;
    }
    // The population is the elements in memory order: a contiguous tensor is read in place.
    const torch::Tensor values = tensor.is_contiguous() ? tensor : tensor.reshape({-1}).contiguous();
    check_population(values);
    radix_select<false>(values, nullptr, GivenRanks{}, base + quantiles_offset, quantile_counts,
                        first_pass.data_ptr<int32_t>());
}

c10::optional<std::tuple<torch::Tensor, torch::Tensor>> exact_quantiles_float64(torch::Tensor values) {
    TORCH_CHECK(values.is_cuda() && values.scalar_type() == at::kDouble && values.dim() == 1 &&
                    values.is_contiguous() && values.numel() > 0,
                "values must be a non-empty contiguous 1D float64 CUDA tensor");
    if (values.numel() > kFloat64SmallPopulation) {
        return c10::nullopt;
    }
    const at::cuda::OptionalCUDAGuard device_guard(values.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto quantiles = torch::empty({kQuantileTargets + 2}, values.options());
    auto counts = torch::empty({kCountKinds}, values.options().dtype(torch::kInt32));
    exact_quantiles_float64_kernel<<<1, kSmallThreads, 0, stream>>>(
        values.data_ptr<double>(), values.numel(), reinterpret_cast<int32_t*>(quantiles.data_ptr<double>()),
        counts.data_ptr<int32_t>());
    return std::make_tuple(quantiles, counts);
}

torch::Tensor order_statistics(torch::Tensor values, std::vector<int64_t> ranks) {
    TORCH_CHECK(values.dim() == 1, "values must be a 1D tensor");
    check_population(values);
    TORCH_CHECK(!ranks.empty() && ranks.size() <= static_cast<size_t>(kQuantileTargets),
                "order_statistics selects between one and ", kQuantileTargets, " ranks");
    GivenRanks given{};
    given.count = static_cast<int>(ranks.size());
    for (size_t i = 0; i < ranks.size(); ++i) {
        TORCH_CHECK(ranks[i] >= 0 && ranks[i] < values.numel(), "rank ", ranks[i], " is outside a population of ",
                    values.numel());
        given.rank[i] = ranks[i];
    }
    auto selected = torch::empty({kQuantileTargets + 2}, values.options().dtype(torch::kFloat32));
    radix_select<true>(values, nullptr, given, selected.data_ptr<float>(), nullptr, nullptr);
    return selected.narrow(0, 1, given.count);
}

torch::Tensor row_entropy(torch::Tensor matrix) {
    check_rows(matrix);
    TORCH_CHECK(matrix.scalar_type() == at::kFloat, "row_entropy takes float32 rows");
    const at::cuda::OptionalCUDAGuard device_guard(matrix.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto out = torch::empty({matrix.size(0)}, matrix.options().dtype(torch::kFloat64));
    // The host's log, as math.log(n) divides in torch.
    row_entropy_kernel<<<row_grid(matrix), kBlockThreads, 0, stream>>>(
        matrix.data_ptr<float>(), matrix.stride(0), matrix.size(0), static_cast<int>(matrix.size(1)),
        std::log(static_cast<double>(matrix.size(1))), out.data_ptr<double>());
    return out;
}

bool entropy_quantiles_into(
    torch::Tensor matrix, torch::Tensor words, int64_t quantiles_offset, int64_t counts_offset) {
    check_rows(matrix);
    TORCH_CHECK(matrix.scalar_type() == at::kFloat, "entropy_quantiles_into takes float32 rows");
    TORCH_CHECK(words.device() == matrix.device() && words.scalar_type() == at::kInt && words.is_contiguous() &&
                    quantiles_offset >= 0 && quantiles_offset + 2 * kSelectionTargets <= words.numel() &&
                    counts_offset >= 0 && counts_offset + kCountKinds <= words.numel(),
                "the quantile and count slots must lie in the int32 words on the rows' device");
    if (matrix.size(0) > kFloat64SmallPopulation) {
        return false;
    }
    const at::cuda::OptionalCUDAGuard device_guard(matrix.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto entropy = torch::empty({matrix.size(0)}, matrix.options().dtype(torch::kFloat64));
    row_entropy_kernel<<<row_grid(matrix), kBlockThreads, 0, stream>>>(
        matrix.data_ptr<float>(), matrix.stride(0), matrix.size(0), static_cast<int>(matrix.size(1)),
        std::log(static_cast<double>(matrix.size(1))), entropy.data_ptr<double>());
    int32_t* base = words.data_ptr<int32_t>();
    exact_quantiles_float64_kernel<<<1, kSmallThreads, 0, stream>>>(
        entropy.data_ptr<double>(), entropy.numel(), base + quantiles_offset, base + counts_offset);
    return true;
}
