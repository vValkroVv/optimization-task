#pragma once

#include "kernel_common.cuh"

namespace {

// Order keys: the unsigned order of the key of a floating point value is the float order.
constexpr int kKeyBits = 32;
constexpr uint32_t kKeySignBit = 0x80000000u;
constexpr int kMagnitudeMask32 = 0x7FFFFFFF;

// Radix selection works on 8-bit digits of a key.
constexpr int kDigitBits = 8;
constexpr int kRadixBins = 1 << kDigitBits;
constexpr int kFirstDigitShift = kKeyBits - kDigitBits;
constexpr int kBinsPerLane = kRadixBins / kWarpSize;

// The counts of a population, in the order of its int32 `counts`.
enum Count { kFiniteCount, kNanCount, kPositiveInfinityCount, kNegativeInfinityCount, kMaskedCount };
constexpr int kCountKinds = kMaskedCount + 1;

// Exact selection: a pass selects the smallest value, the inner percentiles and the largest value; the scratch
// holds the counts, the first digit's histogram, then one target-major histogram per refining pass.
constexpr int kQuantileTargets = 7;
constexpr int kSelectionTargets = kQuantileTargets + 2;
constexpr int kSelectionThreads = kWarpSize * kSelectionTargets;
__constant__ int kInnerPercentiles[kQuantileTargets] = {1, 5, 25, 50, 75, 95, 99};
constexpr int kPercentScale = 100;
// Added before the division by kPercentScale to round it up.
constexpr int kPercentRoundUp = kPercentScale - 1;
constexpr int kScratchCounts = 0;
constexpr int kScratchFirstDigit = kScratchCounts + kCountKinds;

// Exact quantiles by radix selection, the values quantiles_kernel.py's Triton kernels select. A value widened
// exactly to fp32 maps to a key whose unsigned order is the float order; each pass counts one 8-bit key digit in
// shared memory for the nine targets (q0, q1..q99, q100) and narrows each to one bin. A bf16 key has 16
// significant bits (two passes), fp16 19 (three), fp32 32 (four). order_statistics selects given ranks of the
// whole population the same way, ordered like torch.sort: NaN after +inf.
//
// Refine histograms per block: the warps of a block spread their increments over this many copies.
constexpr int kRefineCopies = 2;

// The passes over 8-bit digits that select from a bf16, fp16 and fp32 key.
constexpr int kBFloat16Passes = 2;
constexpr int kHalfPasses = 3;
constexpr int kFloatPasses = 4;

// Ranks selected instead of the inner percentiles; count == 0 selects the percentiles.
struct GivenRanks {
    int64_t rank[kQuantileTargets];
    int count;
};

template <int kLength>
struct alignas(kLength) MaskLoad {
    bool elements[kLength];
};

// Signed order of the result is the float order.
__device__ __forceinline__ int32_t ordered_bits(float value) {
    int32_t bits = __float_as_int(value);
    return bits ^ ((bits >> (kKeyBits - 1)) & kMagnitudeMask32);
}

__device__ __forceinline__ float from_ordered(int32_t ordered) {
    return __int_as_float(ordered ^ ((ordered >> (kKeyBits - 1)) & kMagnitudeMask32));
}

__device__ __forceinline__ uint32_t radix_key(int32_t ordered) {
    return static_cast<uint32_t>(ordered) ^ kKeySignBit;
}

// The value of a selected key prefix: the key bits below a 16-bit source's precision are zeros for a positive
// value and ones for a negative one, whose key has a clear top bit.
__device__ __forceinline__ float prefix_value(int64_t prefix, int final_shift) {
    uint32_t key = static_cast<uint32_t>(prefix << final_shift);
    if ((key & kKeySignBit) == 0) {
        key |= (1u << final_shift) - 1u;
    }
    return from_ordered(static_cast<int32_t>(key ^ kKeySignBit));
}

// visit(value, valid) for every element of a 1D population, 16 bytes per load where alignment allows. Which
// thread sees which element does not matter: the passes only count.
template <typename T, typename Visitor>
__device__ __forceinline__ void for_each_element(
    const T* __restrict__ values, const bool* __restrict__ valid, int64_t element_count, Visitor&& visit) {
    constexpr int kVectorElements = kVectorBytes / sizeof(T);
    const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const bool aligned = reinterpret_cast<uintptr_t>(values) % kVectorBytes == 0 &&
                         (valid == nullptr || reinterpret_cast<uintptr_t>(valid) % kVectorElements == 0);
    const int64_t vectors = aligned ? element_count / kVectorElements : 0;
    for (int64_t vector = first; vector < vectors; vector += step) {
        VectorLoad<T, kVectorElements> chunk = reinterpret_cast<const VectorLoad<T, kVectorElements>*>(values)[vector];
        MaskLoad<kVectorElements> mask{};
        if (valid != nullptr) {
            mask = reinterpret_cast<const MaskLoad<kVectorElements>*>(valid)[vector];
        }
#pragma unroll
        for (int j = 0; j < kVectorElements; ++j) {
            visit(static_cast<float>(chunk.elements[j]), valid == nullptr || mask.elements[j]);
        }
    }
    for (int64_t i = vectors * kVectorElements + first; i < element_count; i += step) {
        visit(static_cast<float>(values[i]), valid == nullptr || valid[i]);
    }
}

// The key of a population element. With kAllValues every value has one, every NaN as +NaN, which orders after
// +inf; otherwise NaN and infinities have none, and each pass leaves them out.
template <bool kAllValues>
__device__ __forceinline__ uint32_t population_key(float value) {
    if (kAllValues && isnan(value)) {
        value = __int_as_float(kQuietNanBits);
    }
    return radix_key(ordered_bits(value));
}

template <typename T, bool kAllValues>
__device__ void radix_count(
    const T* __restrict__ values,
    const bool* __restrict__ valid,
    int64_t element_count,
    int32_t* __restrict__ counts,
    int32_t* __restrict__ histogram) {
    // One histogram per warp: values of one magnitude share a top digit, and a single histogram per block would
    // serialize its warps on a few bins.
    __shared__ int32_t shared_histogram[kWarps][kRadixBins];
    __shared__ int32_t shared_counts[kCountKinds];
    for (int i = threadIdx.x; i < kWarps * kRadixBins; i += blockDim.x) {
        shared_histogram[i / kRadixBins][i % kRadixBins] = 0;
    }
    if (threadIdx.x < kCountKinds) {
        shared_counts[threadIdx.x] = 0;
    }
    __syncthreads();

    int32_t* warp_histogram = shared_histogram[threadIdx.x / kWarpSize];
    int finite = 0, nan = 0, positive = 0, negative = 0, masked = 0;
    // One shared atomic per element, without aggregating a thread's runs of one digit.
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
        atomicAdd(&warp_histogram[population_key<kAllValues>(value) >> kFirstDigitShift], 1);
    });
    finite = warp_reduce_sum(finite);
    nan = warp_reduce_sum(nan);
    positive = warp_reduce_sum(positive);
    negative = warp_reduce_sum(negative);
    masked = warp_reduce_sum(masked);
    if ((threadIdx.x & (kWarpSize - 1)) == 0) {
        atomicAdd(&shared_counts[kFiniteCount], finite);
        atomicAdd(&shared_counts[kNanCount], nan);
        atomicAdd(&shared_counts[kPositiveInfinityCount], positive);
        atomicAdd(&shared_counts[kNegativeInfinityCount], negative);
        atomicAdd(&shared_counts[kMaskedCount], masked);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < kRadixBins; i += blockDim.x) {
        int32_t total = 0;
#pragma unroll
        for (int warp_index = 0; warp_index < kWarps; ++warp_index) {
            total += shared_histogram[warp_index][i];
        }
        if (total != 0) {
            atomicAdd(&histogram[i], total);
        }
    }
    if (threadIdx.x < kCountKinds) {
        atomicAdd(&counts[threadIdx.x], shared_counts[threadIdx.x]);
    }
}

// Row t of the histogram counts the digit at `shift` of the keys whose bits above `prefix_shift` equal target t's
// prefix; an element is counted once, in the first target with its prefix, and radix_select_kernel reads that row
// for the others. Every element takes one shared atomic and no branch: an element that matches no prefix, or is
// masked, is counted in a row that is never flushed. NaN and infinities are not tested per element: their keys
// fall in bins no finite key has, which the first refining pass (kFirstDigit: a one-digit prefix, mapped to its
// first target through a table) leaves out, and no later prefix can match them.
template <typename T, bool kAllValues, bool kFirstDigit>
__device__ void radix_refine(
    const T* __restrict__ values,
    const bool* __restrict__ valid,
    int64_t element_count,
    const int64_t* __restrict__ prefixes,
    int prefix_shift,
    uint32_t prefix_mask,
    int shift,
    int32_t* __restrict__ histogram) {
    constexpr int kUnmatched = kSelectionTargets;
    constexpr int kRows = kSelectionTargets + 1;
    __shared__ int32_t shared_histogram[kRefineCopies][kRows * kRadixBins];
    __shared__ uint32_t shared_prefixes[kSelectionTargets];
    __shared__ int8_t first_target[kRadixBins];
    for (int i = threadIdx.x; i < kRefineCopies * kRows * kRadixBins; i += blockDim.x) {
        shared_histogram[i / (kRows * kRadixBins)][i % (kRows * kRadixBins)] = 0;
    }
    if (threadIdx.x < kSelectionTargets) {
        // A target past the end of an empty population has a prefix no key can match.
        shared_prefixes[threadIdx.x] = static_cast<uint32_t>(prefixes[threadIdx.x]);
    }
    __syncthreads();

    uint32_t prefix[kSelectionTargets];
#pragma unroll
    for (int target = 0; target < kSelectionTargets; ++target) {
        prefix[target] = shared_prefixes[target];
    }
    if (kFirstDigit) {
        for (int high = threadIdx.x; high < kRadixBins; high += blockDim.x) {
            int row = kUnmatched;
#pragma unroll
            for (int target = kSelectionTargets - 1; target >= 0; --target) {
                if (static_cast<uint32_t>(high) == prefix[target]) {
                    row = target;
                }
            }
            first_target[high] = static_cast<int8_t>(row);
        }
        __syncthreads();
    }
    int32_t* copy = shared_histogram[(threadIdx.x / kWarpSize) % kRefineCopies];
    for_each_element(values, valid, element_count, [&](float value, bool is_valid) {
        uint32_t key = population_key<kAllValues>(value);
        int row = kUnmatched;
        if (kFirstDigit) {
            row = first_target[key >> kFirstDigitShift];
        } else {
            uint32_t high = (key >> prefix_shift) & prefix_mask;
#pragma unroll
            for (int target = kSelectionTargets - 1; target >= 0; --target) {
                if (high == prefix[target]) {
                    row = target;
                }
            }
        }
        row = is_valid ? row : kUnmatched;
        atomicAdd(&copy[row * kRadixBins + static_cast<int>((key >> shift) & (kRadixBins - 1))], 1);
    });
    __syncthreads();
    for (int i = threadIdx.x; i < kSelectionTargets * kRadixBins; i += blockDim.x) {
        const uint32_t row_prefix = shared_prefixes[i / kRadixBins];
        const int digit = i % kRadixBins;
        const bool nonfinite = (row_prefix == 0 && digit < kRadixBins / 2) ||
                               (row_prefix == kRadixBins - 1 && digit >= kRadixBins / 2);
        if (!kAllValues && kFirstDigit && nonfinite) {
            continue;
        }
        int32_t total = 0;
#pragma unroll
        for (int copy_index = 0; copy_index < kRefineCopies; ++copy_index) {
            total += shared_histogram[copy_index][i];
        }
        if (total != 0) {
            atomicAdd(&histogram[i], total);
        }
    }
}

// A warp finds the bin of `histogram_row` holding `rank` and the count below that bin; lane l owns the
// kBinsPerLane bins from l * kBinsPerLane. A rank outside the row selects bin kRadixBins.
__device__ __forceinline__ void warp_select_bin(
    const int32_t* histogram_row, int64_t rank, int64_t& bin, int64_t& below) {
    const int lane = static_cast<int>(threadIdx.x & (kWarpSize - 1));
    int32_t local[kBinsPerLane];
    int64_t local_sum = 0;
#pragma unroll
    for (int j = 0; j < kBinsPerLane; ++j) {
        local[j] = histogram_row[lane * kBinsPerLane + j];
        local_sum += local[j];
    }
    int64_t inclusive = local_sum;
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
        int64_t other = __shfl_up_sync(kFullWarpMask, inclusive, offset);
        if (lane >= offset) {
            inclusive += other;
        }
    }
    const int64_t exclusive = inclusive - local_sum;
    const unsigned owners = __ballot_sync(kFullWarpMask, exclusive <= rank && rank < inclusive);
    int64_t found_bin = kRadixBins, found_below = 0;
    if (owners != 0 && lane == __ffs(owners) - 1) {
        int64_t running = exclusive;
#pragma unroll
        for (int j = 0; j < kBinsPerLane; ++j) {
            if (found_bin == kRadixBins && rank < running + local[j]) {
                found_bin = lane * kBinsPerLane + j;
                found_below = running;
            }
            running += local[j];
        }
    }
    const int source = owners != 0 ? __ffs(owners) - 1 : 0;
    bin = __shfl_sync(kFullWarpMask, found_bin, source);
    below = __shfl_sync(kFullWarpMask, found_below, source);
}

// One warp per target; target t is quantiles[t]. The final pass also copies the counts out,
// unless counts_out is null.
__device__ void radix_select_targets(
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
    __shared__ int64_t previous[kSelectionTargets];
    const int64_t population = counts[kFiniteCount];
    const int target = static_cast<int>(threadIdx.x / kWarpSize);
    if (pass > 0 && threadIdx.x < kSelectionTargets) {
        previous[threadIdx.x] = prefixes[threadIdx.x];
    }
    __syncthreads();
    int64_t rank;
    int row = 0;
    if (pass > 0) {
        rank = ranks[target];
        row = target;
        for (int earlier = target - 1; earlier >= 0; --earlier) {
            if (previous[earlier] == previous[target]) {
                row = earlier;
            }
        }
    } else if (target == 0 || target == kSelectionTargets - 1) {
        // An empty population's largest value is rank -1, which selects no bin.
        rank = target == 0 ? 0 : population - 1;
    } else if (given.count > 0) {
        // An unused target selects no bin.
        rank = target <= given.count ? given.rank[target - 1] : -1;
    } else {
        rank = max((population * kInnerPercentiles[target - 1] + kPercentRoundUp) / kPercentScale - 1, int64_t{0});
    }
    int64_t bin, below;
    warp_select_bin(histogram + row * kRadixBins, rank, bin, below);
    if ((threadIdx.x & (kWarpSize - 1)) == 0) {
        const int64_t prefix = ((pass == 0 ? int64_t{0} : previous[target]) << kDigitBits) | bin;
        prefixes[target] = prefix;
        ranks[target] = rank - below;
        if (final_pass) {
            quantiles[target] = population > 0 ? prefix_value(prefix, final_shift) : NAN;
        }
    }
    if (final_pass && counts_out != nullptr && threadIdx.x < kCountKinds) {
        counts_out[threadIdx.x] = counts[threadIdx.x];
    }
}

// The prefixes, then the ranks, of the targets of a selection.
constexpr int kSelectionWords = 2 * kSelectionTargets;

// A population this small is selected by radix_select_small_kernel in one block and one launch; beyond it the
// multi-block passes are faster.
constexpr int64_t kSmallPopulation = int64_t{1} << 13;

inline int radix_passes(at::ScalarType dtype) {
    if (dtype == at::kBFloat16) {
        return kBFloat16Passes;
    }
    return dtype == at::kHalf ? kHalfPasses : kFloatPasses;
}

inline int64_t radix_scratch_words(int passes) {
    return kScratchFirstDigit + kRadixBins + (passes - 1) * kSelectionTargets * kRadixBins;
}

inline bool has_16_bit_keys(at::ScalarType dtype) {
    return dtype == at::kBFloat16 || dtype == at::kHalf;
}

}  // namespace
