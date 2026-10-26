#include "cuda_host_functions.h"
#include "key_histogram.cuh"

namespace {

// Rows at least this wide are reduced by a whole block each, narrower ones by a warp each.
constexpr int64_t kWideRow = 4096;

// A 16-bit counter wraps from kCounterMax to 0 and carries one count, kHighHalfUnit, into the half above it.
constexpr uint32_t kCounterMax = kLowHalfMask;
constexpr uint32_t kHighHalfUnit = 0x10000u;

constexpr int kKeyRowThreads = 16;
constexpr int kKeyBlockRowsPerSm = 8;
// The fixed shared memory the statistics and finalize passes use besides the table.
constexpr size_t kKeyStatsStaticSharedBytes = 8192;

// Elements are counted by (sign, fp32 exponent field), the top nine bits, with one atomic per element and no
// branch; the exponent histogram and the first radix digit both derive from it. NaN and infinities (exponent
// field 255) land in two bins no finite value shares, which the flush leaves out.
constexpr int kSignExponentBins = 512;

// Adds a block's per-warp (sign, exponent) histograms into the exponent bins and, unless `first_pass` is null,
// into the first-digit histogram of the exact selection's scratch, where a negative value's digit is complemented.
__device__ void flush_sign_exponent_histogram(
    int* histogram, int* exponent_counts, int bins_start, int bins_count, int32_t* first_pass) {
    if (threadIdx.x < kWarps) {
        histogram[threadIdx.x * kSignExponentBins + kSignExponentBins / 2 - 1] = 0;
        histogram[threadIdx.x * kSignExponentBins + kSignExponentBins - 1] = 0;
    }
    __syncthreads();
    for (int exponent_field = threadIdx.x; exponent_field < kSignExponentBins / 2; exponent_field += blockDim.x) {
        int total = 0;
        for (int warp_index = 0; warp_index < kWarps; ++warp_index) {
            total += histogram[warp_index * kSignExponentBins + exponent_field] +
                     histogram[warp_index * kSignExponentBins + kSignExponentBins / 2 + exponent_field];
        }
        if (total != 0) {
            const int bin =
                floor_log2_bin(__int_as_float(exponent_field << kFloatMantissaBits), bins_start, bins_count);
            atomicAdd(exponent_counts + bin, total);
        }
    }
    if (first_pass == nullptr) {
        return;
    }
    for (int digit = threadIdx.x; digit < kRadixBins; digit += blockDim.x) {
        const bool negative = digit < kRadixBins / 2;
        const int exponent_pair = negative ? kRadixBins / 2 - 1 - digit : digit - kRadixBins / 2;
        const int* sign_histogram = histogram + (negative ? kSignExponentBins / 2 : 0) + 2 * exponent_pair;
        int total = 0;
        for (int warp_index = 0; warp_index < kWarps; ++warp_index) {
            total += sign_histogram[warp_index * kSignExponentBins] +
                     sign_histogram[warp_index * kSignExponentBins + 1];
        }
        if (total != 0) {
            atomicAdd(first_pass + kScratchFirstDigit + digit, total);
        }
    }
}

// Adds the block totals to the recorder buffer and, unless `first_pass` is null, the selection counts
// [finite, NaN, +Inf, -Inf] to its scratch.
__device__ void flush_block_totals(
    float block_sum, float block_sum_of_squares, float block_cube, float block_fourth, int block_finite,
    int block_nonfinite, int block_posinf, int block_neginf, float block_amax, float* buffer, int64_t moments_offset,
    int64_t counts_offset, int64_t amax_offset, int64_t power_sums_offset, int32_t* first_pass) {
    atomicAdd(buffer + moments_offset + 0, block_sum);
    atomicAdd(buffer + moments_offset + 1, block_sum_of_squares);
    atomicAdd(buffer + power_sums_offset + 0, block_cube);
    atomicAdd(buffer + power_sums_offset + 1, block_fourth);
    int* counts = reinterpret_cast<int*>(buffer + counts_offset);
    atomicAdd(counts + kNonFiniteWord, block_nonfinite);
    atomicAdd(counts + kFiniteWord, block_finite);
    // block_amax is never negative, so its bit pattern as an unsigned int orders like its value and an integer
    // atomicMax is a lock-free float max (the trick tl.atomic_max relies on in the Triton kernel).
    atomicMax(reinterpret_cast<unsigned int*>(buffer + amax_offset), __float_as_uint(block_amax));
    if (first_pass != nullptr) {
        atomicAdd(first_pass + kScratchCounts + kFiniteCount, block_finite);
        atomicAdd(first_pass + kScratchCounts + kNanCount, block_nonfinite - block_posinf - block_neginf);
        atomicAdd(first_pass + kScratchCounts + kPositiveInfinityCount, block_posinf);
        atomicAdd(first_pass + kScratchCounts + kNegativeInfinityCount, block_neginf);
    }
}

// Resident blocks per SM of activation_stats_kernel: eight blocks would limit it to 32 registers and spill.
constexpr int kStatsResidentBlocks = 6;

template <typename T>
__global__ void __launch_bounds__(kBlockThreads, kStatsResidentBlocks) activation_stats_kernel(
    const T* __restrict__ x,
    int64_t row_stride,
    int64_t rows,
    int64_t row_length,
    float* __restrict__ buffer,
    int64_t moments_offset,
    int64_t counts_offset,
    int64_t amax_offset,
    int64_t power_sums_offset,
    int bins_start,
    int bins_count,
    int32_t* __restrict__ first_pass) {
    // Dynamic shared memory: one (sign, exponent) histogram per warp, since elements of one magnitude share a bin
    // and a single histogram would serialize the warps, then the root-mean-square histogram.
    extern __shared__ int shared_histogram[];
    const int warp_id = static_cast<int>(threadIdx.x / kWarpSize);
    const int lane_id = static_cast<int>(threadIdx.x % kWarpSize);
    int* warp_histogram = shared_histogram + warp_id * kSignExponentBins;
    int* root_mean_square_bins = shared_histogram + kWarps * kSignExponentBins;

    __shared__ float warp_sum[kWarps];
    __shared__ float warp_sum_of_squares[kWarps];
    __shared__ float warp_cube[kWarps];
    __shared__ float warp_fourth[kWarps];
    __shared__ int warp_finite[kWarps];
    __shared__ int warp_nonfinite[kWarps];
    __shared__ int warp_posinf[kWarps];
    __shared__ int warp_neginf[kWarps];
    __shared__ float warp_amax[kWarps];

    for (int i = threadIdx.x; i < kWarps * kSignExponentBins + bins_count; i += blockDim.x) {
        shared_histogram[i] = 0;
    }
    __syncthreads();

    float thread_sum = 0.f, thread_sum_of_squares = 0.f, thread_cube = 0.f, thread_fourth = 0.f, thread_amax = 0.f;
    int thread_finite = 0, thread_nonfinite = 0, thread_posinf = 0, thread_neginf = 0;

    constexpr int kVectorElements = kVectorBytes / sizeof(T);

    // One warp per row: no block-wide barrier per row.
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * kWarps + warp_id; row < rows;
         row += static_cast<int64_t>(gridDim.x) * kWarps) {
        const T* row_ptr = x + row * row_stride;
        float row_sum_of_squares = 0.f;
        int row_finite = 0;

        auto accumulate = [&](float value) {
            bool finite = isfinite(value);
            float zeroed = finite ? value : 0.f;
            const float square = zeroed * zeroed;
            thread_sum += zeroed;
            row_sum_of_squares += square;
            thread_cube += square * zeroed;
            thread_fourth += square * square;
            row_finite += finite ? 1 : 0;
            thread_posinf += value == INFINITY ? 1 : 0;
            thread_neginf += value == -INFINITY ? 1 : 0;
            thread_amax = fmaxf(thread_amax, fabsf(zeroed));
            atomicAdd(&warp_histogram[static_cast<uint32_t>(__float_as_int(value)) >> kFloatMantissaBits], 1);
        };

        bool vectorizable =
            (row_length % kVectorElements == 0) && (reinterpret_cast<uintptr_t>(row_ptr) % kVectorBytes == 0);
        if (vectorizable) {
            int64_t vector_count = row_length / kVectorElements;
            for (int64_t vector_index = lane_id; vector_index < vector_count; vector_index += kWarpSize) {
                VectorLoad<T, kVectorElements> chunk =
                    reinterpret_cast<const VectorLoad<T, kVectorElements>*>(row_ptr)[vector_index];
#pragma unroll
                for (int j = 0; j < kVectorElements; ++j) {
                    accumulate(static_cast<float>(chunk.elements[j]));
                }
            }
        } else {
            for (int64_t column = lane_id; column < row_length; column += kWarpSize) {
                accumulate(static_cast<float>(row_ptr[column]));
            }
        }

        thread_sum_of_squares += row_sum_of_squares;
        thread_finite += row_finite;
        row_sum_of_squares = warp_reduce_sum(row_sum_of_squares);
        row_finite = warp_reduce_sum(row_finite);
        if (lane_id == 0) {
            thread_nonfinite += static_cast<int>(row_length) - row_finite;
        }
        // A row without a finite element contributes nothing, as in the CPU path's row_has_finite mask.
        if (lane_id == 0 && row_finite > 0) {
            float row_root_mean_square = sqrtf(row_sum_of_squares / static_cast<float>(row_finite));
            atomicAdd(&root_mean_square_bins[floor_log2_bin(row_root_mean_square, bins_start, bins_count)], 1);
        }
    }

    thread_sum = warp_reduce_sum(thread_sum);
    thread_sum_of_squares = warp_reduce_sum(thread_sum_of_squares);
    thread_cube = warp_reduce_sum(thread_cube);
    thread_fourth = warp_reduce_sum(thread_fourth);
    thread_finite = warp_reduce_sum(thread_finite);
    thread_nonfinite = warp_reduce_sum(thread_nonfinite);
    thread_posinf = warp_reduce_sum(thread_posinf);
    thread_neginf = warp_reduce_sum(thread_neginf);
    thread_amax = warp_reduce_max(thread_amax);
    if (lane_id == 0) {
        warp_sum[warp_id] = thread_sum;
        warp_sum_of_squares[warp_id] = thread_sum_of_squares;
        warp_cube[warp_id] = thread_cube;
        warp_fourth[warp_id] = thread_fourth;
        warp_finite[warp_id] = thread_finite;
        warp_nonfinite[warp_id] = thread_nonfinite;
        warp_posinf[warp_id] = thread_posinf;
        warp_neginf[warp_id] = thread_neginf;
        warp_amax[warp_id] = thread_amax;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float block_sum = 0.f, block_sum_of_squares = 0.f, block_cube = 0.f, block_fourth = 0.f, block_amax = 0.f;
        int block_finite = 0, block_nonfinite = 0, block_posinf = 0, block_neginf = 0;
        for (int warp_index = 0; warp_index < kWarps; ++warp_index) {
            block_sum += warp_sum[warp_index];
            block_sum_of_squares += warp_sum_of_squares[warp_index];
            block_cube += warp_cube[warp_index];
            block_fourth += warp_fourth[warp_index];
            block_finite += warp_finite[warp_index];
            block_nonfinite += warp_nonfinite[warp_index];
            block_posinf += warp_posinf[warp_index];
            block_neginf += warp_neginf[warp_index];
            block_amax = fmaxf(block_amax, warp_amax[warp_index]);
        }
        flush_block_totals(block_sum, block_sum_of_squares, block_cube, block_fourth, block_finite, block_nonfinite,
                           block_posinf, block_neginf, block_amax, buffer, moments_offset, counts_offset, amax_offset,
                           power_sums_offset, first_pass);
    }

    int* counts = reinterpret_cast<int*>(buffer + counts_offset);
    flush_sign_exponent_histogram(shared_histogram, counts + kExponentBinsOffset, bins_start, bins_count, first_pass);
    for (int i = threadIdx.x; i < bins_count; i += blockDim.x) {
        if (root_mean_square_bins[i] != 0) {
            atomicAdd(counts + kExponentBinsOffset + bins_count + i, root_mean_square_bins[i]);
        }
    }
}

// The same statistics with one block per row, faster than a warp per row for rows of kWideRow elements or more.
template <typename T>
__global__ void __launch_bounds__(kBlockThreads, kStatsResidentBlocks) activation_stats_block_kernel(
    const T* __restrict__ x,
    int64_t row_stride,
    int64_t rows,
    int64_t row_length,
    float* __restrict__ buffer,
    int64_t moments_offset,
    int64_t counts_offset,
    int64_t amax_offset,
    int64_t power_sums_offset,
    int bins_start,
    int bins_count,
    int32_t* __restrict__ first_pass) {
    // Dynamic shared memory: one (sign, exponent) histogram per warp, then the root-mean-square histogram.
    extern __shared__ int shared_histogram[];
    const int warp_id = static_cast<int>(threadIdx.x / kWarpSize);
    const int lane_id = static_cast<int>(threadIdx.x % kWarpSize);
    int* warp_histogram = shared_histogram + warp_id * kSignExponentBins;
    int* root_mean_square_bins = shared_histogram + kWarps * kSignExponentBins;

    __shared__ float warp_sum[kWarps];
    __shared__ float warp_sum_of_squares[kWarps];
    __shared__ int warp_finite[kWarps];
    __shared__ float warp_amax[kWarps];

    for (int i = threadIdx.x; i < kWarps * kSignExponentBins + bins_count; i += blockDim.x) {
        shared_histogram[i] = 0;
    }
    __syncthreads();

    float block_sum = 0.f, block_sum_of_squares = 0.f, block_amax = 0.f;
    int block_finite = 0, block_nonfinite = 0;
    // Infinities and the higher power sums are accumulated per thread and reduced once at the end.
    int thread_posinf = 0, thread_neginf = 0;
    float thread_cube = 0.f, thread_fourth = 0.f;

    constexpr int kVectorElements = kVectorBytes / sizeof(T);

    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const T* row_ptr = x + row * row_stride;
        float thread_sum = 0.f, thread_sum_of_squares = 0.f, thread_amax = 0.f;
        int thread_finite = 0;

        auto accumulate = [&](float value) {
            bool finite = isfinite(value);
            float zeroed = finite ? value : 0.f;
            const float square = zeroed * zeroed;
            thread_sum += zeroed;
            thread_sum_of_squares += square;
            thread_cube += square * zeroed;
            thread_fourth += square * square;
            thread_finite += finite ? 1 : 0;
            thread_posinf += value == INFINITY ? 1 : 0;
            thread_neginf += value == -INFINITY ? 1 : 0;
            thread_amax = fmaxf(thread_amax, fabsf(zeroed));
            atomicAdd(&warp_histogram[static_cast<uint32_t>(__float_as_int(value)) >> kFloatMantissaBits], 1);
        };

        bool vectorizable =
            (row_length % kVectorElements == 0) && (reinterpret_cast<uintptr_t>(row_ptr) % kVectorBytes == 0);
        if (vectorizable) {
            int64_t vector_count = row_length / kVectorElements;
            for (int64_t vector_index = threadIdx.x; vector_index < vector_count; vector_index += blockDim.x) {
                VectorLoad<T, kVectorElements> chunk =
                    reinterpret_cast<const VectorLoad<T, kVectorElements>*>(row_ptr)[vector_index];
#pragma unroll
                for (int j = 0; j < kVectorElements; ++j) {
                    accumulate(static_cast<float>(chunk.elements[j]));
                }
            }
        } else {
            for (int64_t column = threadIdx.x; column < row_length; column += blockDim.x) {
                accumulate(static_cast<float>(row_ptr[column]));
            }
        }

        thread_sum = warp_reduce_sum(thread_sum);
        thread_sum_of_squares = warp_reduce_sum(thread_sum_of_squares);
        thread_finite = warp_reduce_sum(thread_finite);
        thread_amax = warp_reduce_max(thread_amax);
        if (lane_id == 0) {
            warp_sum[warp_id] = thread_sum;
            warp_sum_of_squares[warp_id] = thread_sum_of_squares;
            warp_finite[warp_id] = thread_finite;
            warp_amax[warp_id] = thread_amax;
        }
        __syncthreads();

        if (warp_id == 0) {
            float row_sum = lane_id < kWarps ? warp_sum[lane_id] : 0.f;
            float row_sum_of_squares = lane_id < kWarps ? warp_sum_of_squares[lane_id] : 0.f;
            int row_finite = lane_id < kWarps ? warp_finite[lane_id] : 0;
            float row_amax = lane_id < kWarps ? warp_amax[lane_id] : 0.f;
            row_sum = warp_reduce_sum(row_sum);
            row_sum_of_squares = warp_reduce_sum(row_sum_of_squares);
            row_finite = warp_reduce_sum(row_finite);
            row_amax = warp_reduce_max(row_amax);
            if (lane_id == 0) {
                block_sum += row_sum;
                block_sum_of_squares += row_sum_of_squares;
                block_finite += row_finite;
                block_nonfinite += static_cast<int>(row_length) - row_finite;
                block_amax = fmaxf(block_amax, row_amax);
                // A row without a finite element contributes nothing, as in the CPU path's row_has_finite mask.
                if (row_finite > 0) {
                    float row_root_mean_square = sqrtf(row_sum_of_squares / static_cast<float>(row_finite));
                    int bin = floor_log2_bin(row_root_mean_square, bins_start, bins_count);
                    root_mean_square_bins[bin] += 1;
                }
            }
        }
        __syncthreads();  // warp_*[] is reused by the next row
    }

    thread_posinf = warp_reduce_sum(thread_posinf);
    thread_neginf = warp_reduce_sum(thread_neginf);
    thread_cube = warp_reduce_sum(thread_cube);
    thread_fourth = warp_reduce_sum(thread_fourth);
    __shared__ int warp_posinf[kWarps];
    __shared__ int warp_neginf[kWarps];
    __shared__ float warp_cube[kWarps];
    __shared__ float warp_fourth[kWarps];
    if (lane_id == 0) {
        warp_posinf[warp_id] = thread_posinf;
        warp_neginf[warp_id] = thread_neginf;
        warp_cube[warp_id] = thread_cube;
        warp_fourth[warp_id] = thread_fourth;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int block_posinf = 0, block_neginf = 0;
        float block_cube = 0.f, block_fourth = 0.f;
        for (int warp_index = 0; warp_index < kWarps; ++warp_index) {
            block_posinf += warp_posinf[warp_index];
            block_neginf += warp_neginf[warp_index];
            block_cube += warp_cube[warp_index];
            block_fourth += warp_fourth[warp_index];
        }
        flush_block_totals(block_sum, block_sum_of_squares, block_cube, block_fourth, block_finite, block_nonfinite,
                           block_posinf, block_neginf, block_amax, buffer, moments_offset, counts_offset, amax_offset,
                           power_sums_offset, first_pass);
    }

    int* counts = reinterpret_cast<int*>(buffer + counts_offset);
    flush_sign_exponent_histogram(shared_histogram, counts + kExponentBinsOffset, bins_start, bins_count, first_pass);
    for (int i = threadIdx.x; i < bins_count; i += blockDim.x) {
        if (root_mean_square_bins[i] != 0) {
            atomicAdd(counts + kExponentBinsOffset + bins_count + i, root_mean_square_bins[i]);
        }
    }
}

// The statistics of a bf16/fp16 tensor whose exact quantiles are wanted come from the counts of its 65536 raw
// values: a block counts every element in a shared table of 16-bit counters (two per 32-bit word), and a counter
// that wraps gives its 65536 counts to the global histogram, as does each word of the table at the end.
// key_finalize_kernel derives the moments, power sums, absolute maximum, counts, exponent histogram and the
// nine exact quantiles from that histogram; each sum adds fp32 powers in fp64 and rounds once, so it does not
// depend on the order in which the blocks finish.

__device__ __forceinline__ uint32_t raw_bits(at::BFloat16 value) { return value.x; }
__device__ __forceinline__ uint32_t raw_bits(at::Half value) { return value.x; }

// The increment of the counter of `raw` in its 32-bit word of the table, which holds two 16-bit counters, the
// even raw value in the low half.
__device__ __forceinline__ uint32_t counter_increment(uint32_t raw) {
    return 1u + (raw & 1u) * kCounterMax;
}

// Counts an element by its raw bits in `table` and returns the word as it was before.
__device__ __forceinline__ uint32_t count_raw(uint32_t* table, uint32_t raw) {
    return atomicAdd(table + (raw >> 1), counter_increment(raw));
}

// Whether the increment of the counter of `raw` that returned `before` wrapped it from 0xFFFF to 0.
__device__ __forceinline__ bool counter_wrapped(uint32_t before, uint32_t raw) {
    const uint32_t counter = counter_increment(raw) * kCounterMax;
    return (before & counter) == counter;
}

// Gives the 65536 counts of a wrapped counter to the histogram and to the total of its group of words. The even
// value's counter wraps into the odd value's, which then holds one count too many that the odd value's count in
// the histogram makes up for. The word is only ever written by increments, so no thread's wrap waits on
// another's. A call, not inline code: the loops have no registers to spare for the rare wrap.
__device__ __noinline__ void give_away(unsigned long long* histogram, uint32_t raw, uint32_t before) {
    const uint32_t carried = (raw & 1u) ^ 1u;
    const uint32_t to_low = carried << kHalfWordBits;
    const uint32_t to_high = ((before >> kHalfWordBits) == kCounterMax ? kHighHalfUnit : 0u) - carried;
    const uint32_t word = raw >> 1;
    atomicAdd(histogram + word, (static_cast<unsigned long long>(to_high) << kWordBits) | to_low);
    atomicAdd(reinterpret_cast<int*>(histogram + kKeyWords) + word / kKeySliceWords,
              static_cast<int>(to_low + to_high));
}

// The 16-bit raw values of a 16-byte chunk.
constexpr int kRawsPerChunk = kVectorBytes / sizeof(uint16_t);

// Counts the elements of a 16-byte chunk of a bf16/fp16 row, the earliest in the low half of the first word, and
// adds their squares to `sum_of_squares` in that order, each rounded as a product and as a sum like the statistics
// kernels do. A counter wraps rarely, so the elements are counted first and tested together.
template <typename T>
__device__ __forceinline__ float count_chunk(
    const uint4& chunk, uint32_t* table, unsigned long long* histogram, float sum_of_squares) {
    const uint32_t words[4] = {chunk.x, chunk.y, chunk.z, chunk.w};
    uint32_t raw[kRawsPerChunk];
    uint32_t before[kRawsPerChunk];
#pragma unroll
    for (int j = 0; j < kRawsPerChunk; ++j) {
        raw[j] = j % 2 == 0 ? words[j / 2] & kLowHalfMask : words[j / 2] >> kHalfWordBits;
        const float value = raw_value<T>(raw[j]);
        sum_of_squares = __fadd_rn(sum_of_squares, __fmul_rn(value, value));
        before[j] = count_raw(table, raw[j]);
    }
    bool wrapped = false;
#pragma unroll
    for (int j = 0; j < kRawsPerChunk; ++j) {
        wrapped |= counter_wrapped(before[j], raw[j]);
    }
    if (wrapped) {
#pragma unroll
        for (int j = 0; j < kRawsPerChunk; ++j) {
            if (counter_wrapped(before[j], raw[j])) {
                give_away(histogram, raw[j], before[j]);
            }
        }
    }
    return sum_of_squares;
}

// A row belongs to the threads of a half warp or to the block. Its sum of squares is added as the statistics
// kernels add it, by the 32 lanes of a warp: lane l sums chunks l, l + 32, ... in element order and the tree of
// row_sum adds the lanes. A half-warp lane stands for lanes l and l + 16 and keeps their `kSets` sums apart,
// `kStride` chunks apart.
template <int kRowThreads>
constexpr int kRowSets = kRowThreads < kWarpSize ? kWarpSize / kRowThreads : 1;

template <int kRowThreads>
constexpr int kChunkStride = kRowThreads < kWarpSize ? kWarpSize : kRowThreads;

// The sum over the threads that share a row, returned to every one of them: a shuffle-down tree over the half
// warp, or over a warp's lanes and then over the warps' sums for a block.
template <int kRowThreads, typename Number>
__device__ __forceinline__ Number row_sum(Number value, Number* partials) {
    constexpr int kWidth = kRowThreads < kWarpSize ? kRowThreads : kWarpSize;
    const unsigned mask = kRowThreads < kWarpSize
                              ? kHalfWarpMask << (threadIdx.x % kWarpSize / kRowThreads * kRowThreads)
                              : kFullWarpMask;
    for (int offset = kWidth / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(mask, value, offset, kWidth);
    }
    if constexpr (kRowThreads > kWarpSize) {
        if (threadIdx.x % kWarpSize == 0) {
            partials[threadIdx.x / kWarpSize] = value;
        }
        __syncthreads();
        if (threadIdx.x < kWarpSize) {
            value = partials[threadIdx.x];
            for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
                value += __shfl_down_sync(kFullWarpMask, value, offset);
            }
            partials[0] = value;
        }
        __syncthreads();
        value = partials[0];
        __syncthreads();
        return value;
    }
    return __shfl_sync(mask, value, 0, kWidth);
}

// The sum of the sets of a lane in order: lanes l and l + 16 of a warp meet first in the tree.
template <int kSets>
__device__ __forceinline__ float sets_sum(const float (&sums)[kSets]) {
    float total = sums[0];
    for (int set = 1; set < kSets; ++set) {
        total += sums[set];
    }
    return total;
}

// Counts a row in the root-mean-square histogram by its sum of squares and its finite count; a row without a
// finite element contributes nothing, as in the CPU path's row_has_finite mask.
__device__ __forceinline__ void count_row_root_mean_square(
    int* root_mean_square_bins, float sum_of_squares, int finite, int bins_start, int bins_count) {
    if (finite > 0) {
        const float row_root_mean_square = sqrtf(sum_of_squares / static_cast<float>(finite));
        atomicAdd(&root_mean_square_bins[floor_log2_bin(row_root_mean_square, bins_start, bins_count)], 1);
    }
}

// The sums of squares of a row's finite elements, one per set of thread `group_lane`, and the count of its
// finite elements, in the order in which activation_key_stats_kernel adds up all its squares.
template <typename T, int kRowThreads>
__device__ __forceinline__ void finite_squares(
    const T* __restrict__ row_ptr, int64_t row_length, int group_lane, float (&sums)[kRowSets<kRowThreads>],
    int& finite) {
    constexpr int kVectorElements = kVectorBytes / sizeof(T);
    constexpr int kSets = kRowSets<kRowThreads>;
    constexpr int kStride = kChunkStride<kRowThreads>;
    finite = 0;
    float sum_of_squares = 0.f;
    const auto add = [&](float value) {
        const bool is_finite = isfinite(value);
        const float zeroed = is_finite ? value : 0.f;
        sum_of_squares = __fadd_rn(sum_of_squares, __fmul_rn(zeroed, zeroed));
        finite += is_finite ? 1 : 0;
    };
    const bool vectors = row_length % kVectorElements == 0 && reinterpret_cast<uintptr_t>(row_ptr) % kVectorBytes == 0;
#pragma unroll
    for (int set = 0; set < kSets; ++set) {
        sum_of_squares = 0.f;
        if (vectors) {
            const uint4* chunks = reinterpret_cast<const uint4*>(row_ptr);
            for (int64_t vector_index = group_lane + set * kRowThreads; vector_index < row_length / kVectorElements;
                 vector_index += kStride) {
                const uint4 chunk = chunks[vector_index];
                const uint32_t words[4] = {chunk.x, chunk.y, chunk.z, chunk.w};
#pragma unroll
                for (int j = 0; j < kVectorElements; ++j) {
                    add(raw_value<T>(j % 2 == 0 ? words[j / 2] & kLowHalfMask : words[j / 2] >> kHalfWordBits));
                }
            }
        } else {
            for (int64_t column = group_lane + set * kRowThreads; column < row_length; column += kStride) {
                add(static_cast<float>(row_ptr[column]));
            }
        }
        sums[set] = sum_of_squares;
    }
}

// Counts the elements of a [rows, row_length] tensor of 16-bit values and finds the root-mean-square bin of each
// row. A row belongs to a half warp, or to the block when few rows are wide (see kRowSets); the warps of a block
// take consecutive rows, unless there are fewer rows than half warps, when the rows are spread over the blocks.
// A row whose sum of squares is not finite is summed again over its finite elements.
template <typename T, int kRowThreads>
__global__ void __launch_bounds__(kKeyThreads, 1) activation_key_stats_kernel(
    const T* __restrict__ x,
    int64_t row_stride,
    int64_t rows,
    int64_t row_length,
    float* __restrict__ buffer,
    int64_t moments_offset,
    int64_t counts_offset,
    int64_t amax_offset,
    int64_t power_sums_offset,
    int bins_start,
    int bins_count,
    unsigned long long* __restrict__ histogram,
    float* __restrict__ quantiles,
    int32_t* __restrict__ counts_out,
    bool clear,
    unsigned int* __restrict__ blocks_done) {
    static_assert(sizeof(T) == 2, "raw values are counted for 16-bit elements");
    static_assert(kRowThreads == kKeyRowThreads || kRowThreads == kKeyThreads,
                  "a row belongs to a half warp or to the block");
    constexpr int kRowGroups = kKeyThreads / kRowThreads;
    constexpr int kVectorElements = kVectorBytes / sizeof(T);
    constexpr int kSets = kRowSets<kRowThreads>;
    constexpr int kStride = kChunkStride<kRowThreads>;
    constexpr int kChunksInFlight = 4;
    constexpr int kStepsInFlight = kChunksInFlight / kSets;
    extern __shared__ uint32_t key_table[];
    int* root_mean_square_bins = reinterpret_cast<int*>(key_table + kKeyWords);
    __shared__ float float_partials[kKeyWarps];
    __shared__ int int_partials[kKeyWarps];

    for (int i = threadIdx.x; i < kKeyWords + bins_count; i += kKeyThreads) {
        key_table[i] = 0;
    }
    __syncthreads();

    const int group_lane = static_cast<int>(threadIdx.x % kRowThreads);
    const int64_t groups = static_cast<int64_t>(gridDim.x) * kRowGroups;
    const int group = static_cast<int>(threadIdx.x / kRowThreads);
    // A half warp counts the root-mean-square bins of its rows kRowThreads at a time, one row per lane, which is
    // cheaper than one lane counting each row while the others wait.
    float staged_sum_of_squares = 0.f;
    int staged_finite = 0;
    int staged = 0;
    for (int64_t row = rows >= groups ? static_cast<int64_t>(blockIdx.x) * kRowGroups + group
                                      : blockIdx.x + static_cast<int64_t>(group) * gridDim.x;
         row < rows; row += groups) {
        const T* row_ptr = x + row * row_stride;
        float sums[kSets] = {};
        if (row_length % kVectorElements == 0 && reinterpret_cast<uintptr_t>(row_ptr) % kVectorBytes == 0) {
            const uint4* vectors = reinterpret_cast<const uint4*>(row_ptr);
            // The loads of a few chunks are issued before the first is counted, to keep them in flight.
            for (int64_t step = 0; group_lane + step * kStride < row_length / kVectorElements; step += kStepsInFlight) {
                uint4 chunks[kChunksInFlight];
#pragma unroll
                for (int chunk_index = 0; chunk_index < kChunksInFlight; ++chunk_index) {
                    const int64_t vector_index =
                        group_lane + (chunk_index % kSets) * kRowThreads + (step + chunk_index / kSets) * kStride;
                    if (vector_index < row_length / kVectorElements) {
                        chunks[chunk_index] = vectors[vector_index];
                    }
                }
#pragma unroll
                for (int chunk_index = 0; chunk_index < kChunksInFlight; ++chunk_index) {
                    const int64_t vector_index =
                        group_lane + (chunk_index % kSets) * kRowThreads + (step + chunk_index / kSets) * kStride;
                    if (vector_index < row_length / kVectorElements) {
                        sums[chunk_index % kSets] =
                            count_chunk<T>(chunks[chunk_index], key_table, histogram, sums[chunk_index % kSets]);
                    }
                }
            }
        } else {
#pragma unroll
            for (int set = 0; set < kSets; ++set) {
                for (int64_t column = group_lane + set * kRowThreads; column < row_length; column += kStride) {
                    const T element = row_ptr[column];
                    const float value = static_cast<float>(element);
                    sums[set] = __fadd_rn(sums[set], __fmul_rn(value, value));
                    const uint32_t before = count_raw(key_table, raw_bits(element));
                    if (counter_wrapped(before, raw_bits(element))) {
                        give_away(histogram, raw_bits(element), before);
                    }
                }
            }
        }

        float sum_of_squares = row_sum<kRowThreads>(sets_sum(sums), float_partials);
        int finite = static_cast<int>(row_length);
        if (!isfinite(sum_of_squares)) {
            int counted;
            finite_squares<T, kRowThreads>(row_ptr, row_length, group_lane, sums, counted);
            sum_of_squares = row_sum<kRowThreads>(sets_sum(sums), float_partials);
            finite = row_sum<kRowThreads>(counted, int_partials);
        }
        if constexpr (kRowThreads == kKeyRowThreads) {
            if (group_lane == staged) {
                staged_sum_of_squares = sum_of_squares;
                staged_finite = finite;
            }
            if (++staged == kRowThreads) {
                count_row_root_mean_square(root_mean_square_bins, staged_sum_of_squares, staged_finite, bins_start,
                                           bins_count);
                staged = 0;
            }
        } else if (group_lane == 0) {
            count_row_root_mean_square(root_mean_square_bins, sum_of_squares, finite, bins_start, bins_count);
        }
    }
    if (kRowThreads == kKeyRowThreads && group_lane < staged) {
        count_row_root_mean_square(root_mean_square_bins, staged_sum_of_squares, staged_finite, bins_start,
                                   bins_count);
    }
    __syncthreads();

    // The words of the table go to the histogram, a group of 32 per warp, which also adds the group's counts to its
    // total. The blocks start at different groups so that they do not all add to the same ones together.
    int* group_totals = reinterpret_cast<int*>(histogram + kKeyWords);
    const int first_group = static_cast<int>(blockIdx.x * static_cast<int64_t>(kKeySlices) / gridDim.x);
    for (int i = 0; i < kKeySlices / kKeyWarps; ++i) {
        const int group = (first_group + i * kKeyWarps + static_cast<int>(threadIdx.x / kWarpSize)) % kKeySlices;
        const int word = group * kKeySliceWords + static_cast<int>(threadIdx.x % kWarpSize);
        const uint32_t pair = key_table[word];
        if (__any_sync(kFullWarpMask, pair != 0)) {
            if (pair != 0) {
                atomicAdd(histogram + word,
                          (static_cast<unsigned long long>(pair >> kHalfWordBits) << kWordBits) |
                              (pair & kLowHalfMask));
            }
            int counted = static_cast<int>((pair & kLowHalfMask) + (pair >> kHalfWordBits));
            counted = warp_reduce_sum(counted);
            if (threadIdx.x % kWarpSize == 0) {
                atomicAdd(group_totals + group, counted);
            }
        }
    }
    int* counts = reinterpret_cast<int*>(buffer + counts_offset);
    for (int i = threadIdx.x; i < bins_count; i += kKeyThreads) {
        if (root_mean_square_bins[i] != 0) {
            atomicAdd(counts + kExponentBinsOffset + bins_count + i, root_mean_square_bins[i]);
        }
    }

    // The last block to finish runs the finalize pass over every block's counts in `histogram`; its table is free
    // for the exponent histogram.
    __threadfence();
    __syncthreads();
    __shared__ bool last_block;
    if (threadIdx.x == 0) {
        last_block = atomicAdd(blocks_done, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last_block) {
        return;
    }
    __threadfence();
    if (threadIdx.x == 0) {
        *blocks_done = 0;
    }
    key_finalize_block<T>(histogram, buffer, moments_offset, counts_offset, amax_offset, power_sums_offset, bins_start,
                          bins_count, quantiles, counts_out, clear, reinterpret_cast<int*>(key_table));
}

// The dynamic shared memory of activation_key_stats_kernel: the table and the root-mean-square histogram.
size_t key_stats_shared_bytes(int bins_count) {
    return (static_cast<size_t>(kKeyWords) + bins_count) * sizeof(uint32_t);
}

// Lets `kernel` use `bytes` of dynamic shared memory on the current device. The limit lives in the device's
// context, so it is raised once per kernel and device. Calls hold the GIL, so the map needs no lock.
void allow_dynamic_shared(const void* kernel, size_t bytes) {
    static auto* allowed = new std::map<std::pair<const void*, int>, size_t>();
    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    auto found = allowed->find({kernel, device});
    if (found != allowed->end() && found->second >= bytes) {
        return;
    }
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
    (*allowed)[{kernel, device}] = bytes;
}

// The statistics of a bf16/fp16 tensor and the nine exact quantiles of its elements, from the counts of its raw
// values in `histogram`, which is zero before and, with `clear`, after. The shared memory of a block is beyond
// the default limit and a block is resident per SM, so one wave of sm_count blocks runs. A row of kWideRow
// elements or more gets a block only when there are fewer than kKeyBlockRowsPerSm rows per SM; otherwise half a
// warp per row keeps the block's warps busy and skips two barriers per row.
template <typename T>
void launch_key_stats(
    const T* x,
    int64_t row_stride,
    int64_t rows,
    int64_t row_length,
    float* buffer,
    int64_t moments_offset,
    int64_t counts_offset,
    int64_t amax_offset,
    int64_t power_sums_offset,
    int bins_start,
    int bins_count,
    unsigned long long* histogram,
    float* quantiles,
    int32_t* counts_out,
    bool clear,
    unsigned int* blocks_done,
    int sm_count,
    cudaStream_t stream) {
    const size_t shared_bytes = key_stats_shared_bytes(bins_count);
    const int64_t grid = std::min<int64_t>(rows, sm_count);
    if (row_length >= kWideRow && rows < static_cast<int64_t>(kKeyBlockRowsPerSm) * sm_count) {
        allow_dynamic_shared(reinterpret_cast<const void*>(activation_key_stats_kernel<T, kKeyThreads>), shared_bytes);
        activation_key_stats_kernel<T, kKeyThreads><<<grid, kKeyThreads, shared_bytes, stream>>>(
            x, row_stride, rows, row_length, buffer, moments_offset, counts_offset, amax_offset, power_sums_offset,
            bins_start, bins_count, histogram, quantiles, counts_out, clear, blocks_done);
    } else {
        allow_dynamic_shared(reinterpret_cast<const void*>(activation_key_stats_kernel<T, kKeyRowThreads>),
                             shared_bytes);
        activation_key_stats_kernel<T, kKeyRowThreads><<<grid, kKeyThreads, shared_bytes, stream>>>(
            x, row_stride, rows, row_length, buffer, moments_offset, counts_offset, amax_offset, power_sums_offset,
            bins_start, bins_count, histogram, quantiles, counts_out, clear, blocks_done);
    }
}

template <typename T>
void launch_activation_stats(
    const T* x,
    int64_t row_stride,
    int64_t rows,
    int64_t row_length,
    float* buffer,
    int64_t moments_offset,
    int64_t counts_offset,
    int64_t amax_offset,
    int64_t power_sums_offset,
    int bins_start,
    int bins_count,
    int32_t* first_pass,
    int sm_count,
    cudaStream_t stream) {
    const int64_t capacity = static_cast<int64_t>(sm_count) * kBlocksPerSm;
    size_t shared_bytes = (static_cast<size_t>(kWarps) * kSignExponentBins + bins_count) * sizeof(int);
    if (row_length >= kWideRow) {
        int64_t grid = std::min<int64_t>(rows, capacity);
        activation_stats_block_kernel<T><<<grid, kBlockThreads, shared_bytes, stream>>>(
            x, row_stride, rows, row_length, buffer, moments_offset, counts_offset, amax_offset, power_sums_offset,
            bins_start, bins_count, first_pass);
    } else {
        int64_t grid = std::min<int64_t>(ceil_div(rows, kWarps), capacity);
        activation_stats_kernel<T><<<grid, kBlockThreads, shared_bytes, stream>>>(
            x, row_stride, rows, row_length, buffer, moments_offset, counts_offset, amax_offset, power_sums_offset,
            bins_start, bins_count, first_pass);
    }
}

// Whether a block of the device may have the shared memory of activation_key_stats_kernel, which differs between
// GPU generations: the table and the root-mean-square histogram, and the static arrays of the statistics and
// finalize passes.
bool key_stats_supported(int device, int bins_count) {
    static std::vector<int> optin_bytes;
    if (device >= static_cast<int>(optin_bytes.size())) {
        optin_bytes.resize(device + 1, -1);
    }
    if (optin_bytes[device] < 0) {
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin_bytes[device], cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    }
    return static_cast<size_t>(optin_bytes[device]) >=
           key_stats_shared_bytes(bins_count) + kKeyStatsStaticSharedBytes;
}

// The counts of the raw values of the observations of a stream whose statistics are being taken, for
// key_finalize_kernel, which leaves them zero: one per stream of a device, kept for the life of the process.
unsigned long long* key_counts_scratch(const torch::Device& device, cudaStream_t stream) {
    static auto* scratches = new std::map<std::pair<int, cudaStream_t>, torch::Tensor>();
    const auto key = std::make_pair(static_cast<int>(device.index()), stream);
    auto found = scratches->find(key);
    if (found == scratches->end()) {
        found = scratches->emplace(
            key, torch::zeros({kKeyScratchWords}, torch::TensorOptions().dtype(torch::kInt64).device(device))).first;
    }
    return reinterpret_cast<unsigned long long*>(found->second.data_ptr<int64_t>());
}

// The count of finished blocks of activation_key_stats_kernel, one per stream of a device and kept for the life
// of the process like key_counts_scratch: the last block of each launch leaves it at zero.
unsigned int* key_blocks_done(const torch::Device& device, cudaStream_t stream) {
    static auto* counters = new std::map<std::pair<int, cudaStream_t>, torch::Tensor>();
    const auto key = std::make_pair(static_cast<int>(device.index()), stream);
    auto found = counters->find(key);
    if (found == counters->end()) {
        found = counters->emplace(
            key, torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(device))).first;
    }
    return reinterpret_cast<unsigned int*>(found->second.data_ptr<int32_t>());
}

}  // namespace

// `observe`; given `quantiles` and `counts_out`, a tensor whose raw values are counted in full also gets its
// exact quantiles and counts there, and the result is undefined as nothing is left to select from.
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
    int32_t* counts_out) {
    TORCH_CHECK(tensor.is_cuda(), "tensor must be a CUDA tensor");
    TORCH_CHECK(buffer.is_cuda() && buffer.scalar_type() == torch::kFloat32, "buffer must be a CUDA float32 tensor");

    if (tensor.numel() == 0) {
        return torch::empty({0}, tensor.options().dtype(torch::kInt32));
    }
    torch::Tensor copy;
    const torch::Tensor& matrix = rows_to_read(tensor, copy);
    const int64_t row_length = matrix.size(-1);
    const int64_t rows = matrix.numel() / row_length;
    const int64_t row_stride = rows_stride(matrix);

    const at::cuda::OptionalCUDAGuard device_guard(matrix.device());
    const int sm_count = sm_count_for_device(matrix.get_device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    float* buffer_ptr = buffer.data_ptr<float>();
    int start = static_cast<int>(bins_start);
    int count = static_cast<int>(bins_count);
    // A population selected in one block (kSmallPopulation) is counted again there, whatever it has here.
    if (quantile_first_pass && key_histogram && has_16_bit_keys(matrix.scalar_type()) &&
        matrix.numel() > kSmallPopulation && key_stats_supported(matrix.get_device(), count)) {
        TORCH_CHECK(matrix.numel() <= std::numeric_limits<int32_t>::max(), "population size must fit int32 counts");
        const bool select = quantiles != nullptr;
        torch::Tensor counts_scratch;
        unsigned long long* histogram;
        if (select) {
            histogram = key_counts_scratch(matrix.device(), stream);
        } else {
            counts_scratch = torch::zeros({kKeyScratchWords}, matrix.options().dtype(torch::kInt64));
            histogram = reinterpret_cast<unsigned long long*>(counts_scratch.data_ptr<int64_t>());
        }
        unsigned int* blocks_done = key_blocks_done(matrix.device(), stream);
        if (matrix.scalar_type() == at::kBFloat16) {
            launch_key_stats<at::BFloat16>(matrix.data_ptr<at::BFloat16>(), row_stride, rows, row_length, buffer_ptr,
                                           moments_offset, counts_offset, amax_offset, power_sums_offset, start,
                                           count, histogram, quantiles, counts_out, select, blocks_done, sm_count,
                                           stream);
        } else {
            launch_key_stats<at::Half>(matrix.data_ptr<at::Half>(), row_stride, rows, row_length, buffer_ptr,
                                       moments_offset, counts_offset, amax_offset, power_sums_offset, start, count,
                                       histogram, quantiles, counts_out, select, blocks_done, sm_count, stream);
        }
        return counts_scratch;
    }
    // An exact selection's scratch with its counts and first digit already counted here.
    auto first_pass = torch::zeros(
        {quantile_first_pass ? radix_scratch_words(radix_passes(matrix.scalar_type())) : 0},
        matrix.options().dtype(torch::kInt32));
    int32_t* first_pass_ptr = quantile_first_pass ? first_pass.data_ptr<int32_t>() : nullptr;

    switch (matrix.scalar_type()) {
        case at::kBFloat16:
            launch_activation_stats<at::BFloat16>(matrix.data_ptr<at::BFloat16>(), row_stride, rows, row_length,
                                                  buffer_ptr, moments_offset, counts_offset, amax_offset,
                                                  power_sums_offset, start, count, first_pass_ptr, sm_count, stream);
            break;
        case at::kHalf:
            launch_activation_stats<at::Half>(matrix.data_ptr<at::Half>(), row_stride, rows, row_length, buffer_ptr,
                                              moments_offset, counts_offset, amax_offset, power_sums_offset,
                                              start, count, first_pass_ptr, sm_count, stream);
            break;
        case at::kFloat:
            launch_activation_stats<float>(matrix.data_ptr<float>(), row_stride, rows, row_length, buffer_ptr,
                                           moments_offset, counts_offset, amax_offset, power_sums_offset,
                                           start, count, first_pass_ptr, sm_count, stream);
            break;
        default:
            TORCH_CHECK(false, "activation_ext.observe: unsupported dtype ", matrix.scalar_type());
    }
    return first_pass;
}

torch::Tensor observe(
    torch::Tensor tensor,
    torch::Tensor buffer,
    int64_t moments_offset,
    int64_t counts_offset,
    int64_t amax_offset,
    int64_t power_sums_offset,
    int64_t bins_start,
    int64_t bins_count,
    bool quantile_first_pass,
    bool key_histogram) {
    at::NoGradGuard no_grad;
    return observe_impl(tensor, buffer, moments_offset, counts_offset, amax_offset, power_sums_offset, bins_start,
                        bins_count, quantile_first_pass, key_histogram, nullptr, nullptr);
}
