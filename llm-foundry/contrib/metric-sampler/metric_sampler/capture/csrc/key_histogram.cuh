#pragma once

#include "radix_selection.cuh"

namespace {

// A bf16/fp16 element has 65536 raw values, counted in full in shared 16-bit counters. The global histogram has
// a 64-bit word per pair of raw values 2w and 2w + 1 (the count of 2w in the low half), then a 32-bit total per
// group of 32 words, the 64 keys of a slice.
constexpr int kKeyBins = 1 << kHalfWordBits;
constexpr int kKeyWords = kKeyBins / 2;
constexpr int kKeysPerSlice = 64;
constexpr int kKeySlices = kKeyBins / kKeysPerSlice;
constexpr int kKeySliceWords = kKeyWords / kKeySlices;
constexpr int kKeyScratchWords = kKeyWords + kKeySlices / 2;
constexpr int kKeyThreads = 1024;
static_assert(kKeySlices == kKeyThreads, "a thread of the finalize pass has a slice of keys");
constexpr int kKeyWarps = kKeyThreads / kWarpSize;

// floor(log2|value|) read from the fp32 exponent field and clamped into [0, bins_count); the same bit trick and
// clamp as the Triton kernel's _floor_log2_bin, so both kernels bin an element identically.
__device__ __forceinline__ int floor_log2_bin(float value, int bins_start, int bins_count) {
    int bits = __float_as_int(value);
    int exponent = ((bits >> kFloatMantissaBits) & kFloatExponentMask) - kFloatExponentBias;
    int bin = exponent - bins_start;
    bin = bin < 0 ? 0 : bin;
    bin = bin > bins_count - 1 ? bins_count - 1 : bin;
    return bin;
}

// The recorder's counts of a tensor: the non-finite and the finite count, then the exponent histogram of the
// finite values, then the root-mean-square histogram of its rows.
enum RecorderCount { kNonFiniteWord, kFiniteWord, kExponentBinsOffset };

template <typename T>
__device__ __forceinline__ float raw_value(uint32_t raw) {
    T element;
    element.x = static_cast<uint16_t>(raw);
    return static_cast<float>(element);
}

// A key orders as its value does, negative zero before positive zero: the unsigned order of the keys is the
// float order.
__device__ __forceinline__ uint32_t raw_of_key16(uint32_t key) {
    return key ^ (key >= kSignBit16 ? kSignBit16 : kLowHalfMask);
}

// The first of the 32 words of the histogram that hold slice `slice` of the keys, 64 keys of one sign and
// exponent. The slices are in key order: the negative values come first, in descending order of their raw values.
__device__ __forceinline__ int slice_first_word(int slice) {
    return slice < kKeySlices / 2 ? kKeyWords - kKeySliceWords * (slice + 1)
                                  : kKeySliceWords * (slice - kKeySlices / 2);
}

// The moments of a population: the sums of its values and of their squares, cubes and fourth powers.
enum Moment { kSumMoment, kSumOfSquaresMoment, kCubeMoment, kFourthPowerMoment };
constexpr int kMoments = kFourthPowerMoment + 1;
// The non-finite counts of a population: NaN and infinities together, then +Inf and -Inf alone.
enum NonFinite { kNonFiniteTotal, kPositiveInfinityTotal, kNegativeInfinityTotal };
constexpr int kNonFiniteKinds = kNegativeInfinityTotal + 1;

// The rank, among the finite values in ascending order, of selection target `target`: the smallest value, the
// nearest rank of each inner percentile, the largest value (rank -1 for an empty population, which no slice owns).
// The nearest rank ceil(population * q / 100) - 1 is computed as population / 100 * q +
// ceil(population % 100 * q / 100) - 1, which stays in 32 bits.
__device__ __forceinline__ int quantile_rank(int target, int population) {
    return target == 0                       ? 0
           : target == kSelectionTargets - 1 ? population - 1
                                             : max(population / kPercentScale * kInnerPercentiles[target - 1] +
                                                       (population % kPercentScale * kInnerPercentiles[target - 1] +
                                                        kPercentRoundUp) / kPercentScale - 1,
                                                   0);
}

// What the raw value counts of a bf16/fp16 population give, in one block: thread t owns slice t of the keys, and
// a scan of the slice totals gives the finite values before each slice. The moments, power sums, absolute max,
// counts and exponent histogram go to the recorder's words unless `buffer` is null; the nine quantiles and the
// five counts go to `quantiles` and `counts_out` unless they are null; `clear` leaves the histogram and the
// totals zeroed. Warp w takes the slices with counts w, w + 32, ... in key order, so the sums do not depend on
// the order in which the statistics blocks finished. `histogram` is read through L2 (`__ldcg`): the block runs
// as the statistics kernel's last block, and the other blocks' counts are in L2.
template <typename T>
__device__ __forceinline__ void key_finalize_block(
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
    bool clear,
    int* exponent_histogram) {
    // NaN and infinities have an all-ones exponent field, and infinity no mantissa.
    constexpr uint32_t kExponentMask = std::is_same<T, at::Half>::value ? kHalfExponentMask : kBFloat16ExponentMask;
    constexpr int kUnroll = 4;
    __shared__ double warp_moments[kMoments][kKeyWarps];
    __shared__ float warp_amax[kKeyWarps];
    __shared__ int warp_finite[kKeyWarps];
    __shared__ int warp_populated[kKeyWarps];
    __shared__ int populated[kKeyThreads];
    __shared__ int non_finite[kNonFiniteKinds];
    __shared__ int owner_starts[kSelectionTargets];
    __shared__ int owners[kSelectionTargets];

    const int warp = static_cast<int>(threadIdx.x / kWarpSize);
    const int lane = static_cast<int>(threadIdx.x % kWarpSize);
    int* group_totals = reinterpret_cast<int*>(histogram + kKeyWords);
    const int total = __ldcg(group_totals + slice_first_word(threadIdx.x) / kKeySliceWords);
    for (int i = threadIdx.x; i < bins_count; i += kKeyThreads) {
        exponent_histogram[i] = 0;
    }
    if (threadIdx.x < kSelectionTargets) {
        owners[threadIdx.x] = -1;
    }
    if (threadIdx.x < kNonFiniteKinds) {
        non_finite[threadIdx.x] = 0;
    }

    // The finite values of the slice, and the slices with counts of the warp.
    const uint32_t first_raw = 2u * slice_first_word(threadIdx.x);
    const int finite = (first_raw & kExponentMask) != kExponentMask ? total : 0;
    int inclusive = finite;
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
        const int other = __shfl_up_sync(kFullWarpMask, inclusive, offset);
        if (lane >= offset) {
            inclusive += other;
        }
    }
    const unsigned has_counts = __ballot_sync(kFullWarpMask, total > 0);
    if (lane == kWarpSize - 1) {
        warp_finite[warp] = inclusive;
    }
    if (lane == 0) {
        warp_populated[warp] = __popc(has_counts);
    }
    __syncthreads();

    // The finite values of a slice with a non-zero exponent field all have one exponent bin; those of the slices
    // with a zero exponent field, zero and the subnormal values, are binned one by one below.
    if (buffer != nullptr && finite > 0 && (first_raw & kExponentMask) != 0) {
        atomicAdd(&exponent_histogram[floor_log2_bin(raw_value<T>(first_raw), bins_start, bins_count)], finite);
    }

    // Each warp scans the totals of the warps for its own start and the block's sums.
    const int finite_of_lane = warp_finite[lane];
    const int populated_of_lane = warp_populated[lane];
    int finite_scan = finite_of_lane;
    int populated_scan = populated_of_lane;
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
        const int other_finite = __shfl_up_sync(kFullWarpMask, finite_scan, offset);
        const int other_populated = __shfl_up_sync(kFullWarpMask, populated_scan, offset);
        if (lane >= offset) {
            finite_scan += other_finite;
            populated_scan += other_populated;
        }
    }
    const int population = __shfl_sync(kFullWarpMask, finite_scan, kWarpSize - 1);
    const int populated_count = __shfl_sync(kFullWarpMask, populated_scan, kWarpSize - 1);
    const int start = __shfl_sync(kFullWarpMask, finite_scan - finite_of_lane, warp) + inclusive - finite;
    const int populated_before = __shfl_sync(kFullWarpMask, populated_scan - populated_of_lane, warp);
    if (total > 0) {
        populated[populated_before + __popc(has_counts & ((1u << lane) - 1))] = threadIdx.x;
    }
    if (quantiles != nullptr && finite > 0) {
        for (int target = 0; target < kSelectionTargets; ++target) {
            const int rank = quantile_rank(target, population);
            if (rank >= start && rank < start + finite) {
                owners[target] = static_cast<int>(threadIdx.x);
                owner_starts[target] = start;
            }
        }
    }
    __syncthreads();

    // Warp w reads slices w, w + 32, ... of those with counts, a few at a time to overlap the loads; lane l reads
    // word l of the slice. The warp that selects a quantile reads its owner's slice first.
    const int owner = warp < kSelectionTargets ? owners[warp] : -1;
    unsigned long long owner_pair = 0;
    if (owner >= 0) {
        owner_pair =
            __ldcg(histogram + slice_first_word(owner) + (owner < kKeySlices / 2 ? kKeySliceWords - 1 - lane : lane));
    }
    double moments[kMoments] = {0.0, 0.0, 0.0, 0.0};
    float amax = 0.f;
    for (int first = warp; first < populated_count; first += kUnroll * kKeyWarps) {
        int words[kUnroll];
        unsigned long long pairs[kUnroll];
#pragma unroll
        for (int unrolled_index = 0; unrolled_index < kUnroll; ++unrolled_index) {
            const int index = first + unrolled_index * kKeyWarps;
            words[unrolled_index] = index < populated_count ? slice_first_word(populated[index]) + lane : 0;
            pairs[unrolled_index] = index < populated_count ? __ldcg(histogram + words[unrolled_index]) : 0;
        }
#pragma unroll
        for (int unrolled_index = 0; unrolled_index < kUnroll; ++unrolled_index) {
            const unsigned long long pair = pairs[unrolled_index];
            if (pair == 0) {
                continue;
            }
            if ((2u * words[unrolled_index] & kExponentMask) != kExponentMask) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const uint32_t count =
                        half == 0 ? static_cast<uint32_t>(pair) : static_cast<uint32_t>(pair >> kWordBits);
                    if (count == 0) {
                        continue;
                    }
                    const float value = raw_value<T>(2 * words[unrolled_index] + half);
                    const float square = value * value;
                    const double weight = count;
                    moments[kSumMoment] += weight * value;
                    moments[kSumOfSquaresMoment] += weight * square;
                    moments[kCubeMoment] += weight * (square * value);
                    moments[kFourthPowerMoment] += weight * (square * square);
                    amax = fmaxf(amax, fabsf(value));
                    if (buffer != nullptr && (2u * words[unrolled_index] & kExponentMask) == 0) {
                        atomicAdd(&exponent_histogram[floor_log2_bin(value, bins_start, bins_count)],
                                  static_cast<int>(count));
                    }
                }
            } else {
                const int low = static_cast<int>(static_cast<uint32_t>(pair));
                atomicAdd(&non_finite[kNonFiniteTotal], low + static_cast<int>(pair >> kWordBits));
                if (words[unrolled_index] == kExponentMask / 2) {
                    atomicAdd(&non_finite[kPositiveInfinityTotal], low);
                }
                if (words[unrolled_index] == (kExponentMask | kSignBit16) / 2) {
                    atomicAdd(&non_finite[kNegativeInfinityTotal], low);
                }
            }
        }
    }

    if (quantiles != nullptr && warp < kSelectionTargets) {
        // Lane l of the warp has keys 2l and 2l + 1 of the owner's 64.
        float selected = NAN;
        if (owner >= 0) {
            const int low = static_cast<int>(static_cast<uint32_t>(owner_pair));
            const int high = static_cast<int>(owner_pair >> kWordBits);
            const int first = owner < kKeySlices / 2 ? high : low;
            const int both = low + high;
            int through = both;
            for (int offset = 1; offset < kWarpSize; offset <<= 1) {
                const int other = __shfl_up_sync(kFullWarpMask, through, offset);
                if (lane >= offset) {
                    through += other;
                }
            }
            const int rank = quantile_rank(warp, population) - owner_starts[warp];
            const unsigned holding = __ballot_sync(kFullWarpMask, through - both <= rank && rank < through);
            const int holder = __ffs(holding) - 1;
            const int before = __shfl_sync(kFullWarpMask, through - both, holder);
            const int holder_first = __shfl_sync(kFullWarpMask, first, holder);
            const uint32_t key = owner * kKeysPerSlice + 2 * holder + (rank - before >= holder_first ? 1 : 0);
            selected = raw_value<T>(raw_of_key16(key));
        }
        if (lane == 0) {
            quantiles[warp] = selected;
        }
    }
    if (warp < populated_count) {
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            for (int i = 0; i < kMoments; ++i) {
                moments[i] += __shfl_down_sync(kFullWarpMask, moments[i], offset);
            }
            amax = fmaxf(amax, __shfl_down_sync(kFullWarpMask, amax, offset));
        }
    }
    if (lane == 0) {
        for (int i = 0; i < kMoments; ++i) {
            warp_moments[i][warp] = moments[i];
        }
        warp_amax[warp] = amax;
    }
    __syncthreads();

    if (clear) {
        for (int index = warp; index < populated_count; index += kKeyWarps) {
            histogram[slice_first_word(populated[index]) + lane] = 0;
        }
        group_totals[threadIdx.x] = 0;
    }
    if (buffer != nullptr) {
        int* counts_words = reinterpret_cast<int*>(buffer + counts_offset);
        for (int i = threadIdx.x; i < bins_count; i += kKeyThreads) {
            if (exponent_histogram[i] != 0) {
                atomicAdd(counts_words + kExponentBinsOffset + i, exponent_histogram[i]);
            }
        }
    }
    if (warp == 0) {
        for (int i = 0; i < kMoments; ++i) {
            moments[i] = warp_moments[i][lane];
        }
        amax = warp_amax[lane];
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            for (int i = 0; i < kMoments; ++i) {
                moments[i] += __shfl_down_sync(kFullWarpMask, moments[i], offset);
            }
            amax = fmaxf(amax, __shfl_down_sync(kFullWarpMask, amax, offset));
        }
        if (lane == 0) {
            if (buffer != nullptr) {
                int* counts_words = reinterpret_cast<int*>(buffer + counts_offset);
                atomicAdd(buffer + moments_offset + 0, static_cast<float>(moments[kSumMoment]));
                atomicAdd(buffer + moments_offset + 1, static_cast<float>(moments[kSumOfSquaresMoment]));
                atomicAdd(buffer + power_sums_offset + 0, static_cast<float>(moments[kCubeMoment]));
                atomicAdd(buffer + power_sums_offset + 1, static_cast<float>(moments[kFourthPowerMoment]));
                atomicAdd(counts_words + kNonFiniteWord, non_finite[kNonFiniteTotal]);
                atomicAdd(counts_words + kFiniteWord, population);
                // amax is never negative, so its bit pattern as an unsigned int orders like its value.
                atomicMax(reinterpret_cast<unsigned int*>(buffer + amax_offset), __float_as_uint(amax));
            }
            if (counts_out != nullptr) {
                counts_out[kFiniteCount] = population;
                counts_out[kNanCount] = non_finite[kNonFiniteTotal] - non_finite[kPositiveInfinityTotal] -
                                        non_finite[kNegativeInfinityTotal];
                counts_out[kPositiveInfinityCount] = non_finite[kPositiveInfinityTotal];
                counts_out[kNegativeInfinityCount] = non_finite[kNegativeInfinityTotal];
                counts_out[kMaskedCount] = 0;
            }
        }
    }
}

}  // namespace

torch::Tensor observe_impl(
    torch::Tensor tensor,
    torch::Tensor buffer,
    int64_t moments_offset,
    int64_t counts_offset,
    int64_t amax_offset,
    int64_t power_sums_offset,
    int64_t bins_start,
    int64_t bins_count,
    bool quantile_first_pass,
    bool key_histogram,
    float* quantiles,
    int32_t* counts_out);
