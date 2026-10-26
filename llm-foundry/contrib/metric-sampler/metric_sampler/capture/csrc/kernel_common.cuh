#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <limits>
#include <map>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// The helpers of the .cuh files are in anonymous namespaces: each .cu file compiles its own copy of the ones
// it uses, and a kernel is defined in exactly one .cu file.
namespace {

// A warp has 32 lanes, kHalfWarpMask selects its lower 16, and a vector load reads 16 bytes.
constexpr int kWarpSize = 32;
constexpr unsigned int kFullWarpMask = 0xffffffffu;
constexpr unsigned int kHalfWarpMask = 0xFFFFu;
constexpr int kVectorBytes = 16;

constexpr int kBlockThreads = 256;
constexpr int kWarps = kBlockThreads / kWarpSize;
// Blocks per SM of a grid-stride launch: enough to keep the device busy whatever the row count; the kernel's
// grid-stride loop covers the rest.
constexpr int kBlocksPerSm = 8;

// IEEE 754 binary32: the exponent field is read from the bit pattern of a float.
constexpr int kFloatMantissaBits = 23;
constexpr int kFloatExponentMask = 0xFF;
constexpr int kFloatExponentBias = 127;
// The bit patterns of a positive quiet NaN of float32 and of float64.
constexpr int kQuietNanBits = 0x7FC00000;
constexpr long long kQuietNanBits64 = 0x7ff8000000000000LL;

// The halves of a 32-bit word and the 32-bit words of a 64-bit one, as the counters and the histograms split them.
constexpr int kHalfWordBits = 16;
constexpr int kWordBits = 32;
constexpr unsigned long long kLowWordMask = 0xffffffffull;
constexpr uint32_t kLowHalfMask = 0xFFFFu;
constexpr uint32_t kSignBit16 = 0x8000u;
// The exponent fields of fp16 and bf16 values.
constexpr uint32_t kHalfExponentMask = 0x7C00u;
constexpr uint32_t kBFloat16ExponentMask = 0x7F80u;

// Added to a value inside the logarithm of an entropy sum.
constexpr double kEntropyEpsilon = 1e-12;

inline int64_t ceil_div(int64_t numerator, int64_t denominator) { return (numerator + denominator - 1) / denominator; }

// A 16-byte aligned array that is read or written as one vector load whatever the element size.
template <typename T, int kLength>
struct alignas(kVectorBytes) VectorLoad {
    T elements[kLength];
};

__device__ __forceinline__ float warp_reduce_sum(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kFullWarpMask, value, offset);
    }
    return value;
}

__device__ __forceinline__ int warp_reduce_sum(int value) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kFullWarpMask, value, offset);
    }
    return value;
}

__device__ __forceinline__ double warp_reduce_sum(double value) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kFullWarpMask, value, offset);
    }
    return value;
}

__device__ __forceinline__ float warp_reduce_max(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(kFullWarpMask, value, offset));
    }
    return value;
}

// getDeviceProperties caches internally, but this runs on the training hot path, so it has a tiny cache.
inline int sm_count_for_device(int device) {
    static std::vector<int> cache;
    if (device >= static_cast<int>(cache.size())) {
        cache.resize(device + 1, -1);
    }
    if (cache[device] < 0) {
        cache[device] = at::cuda::getDeviceProperties(device)->multiProcessorCount;
    }
    return cache[device];
}

// A float64 value as its little-endian pair of 32-bit words: the bytes a double store would write, at any word
// offset.
__device__ void store_double_words(int32_t* words, double value) {
    const long long bits = __double_as_longlong(value);
    words[0] = static_cast<int32_t>(static_cast<unsigned long long>(bits) & kLowWordMask);
    words[1] = static_cast<int32_t>(static_cast<unsigned long long>(bits) >> kWordBits);
}

// The statistics of a tensor's row norms, in the order of an accumulator slot: the absolute maximum, then the
// sum of the norms of its rows.
enum RowNormStat { kAmaxStat, kRowNormSumStat };
constexpr int kRowNormStats = kRowNormSumStat + 1;

// The dtypes whose tensors the batched row norm and weight norm launches group, in the order of their lists.
enum DtypeKind { kBFloat16Kind, kHalfKind, kFloatKind };

// The last-axis rows of a tensor a hook observed, detached: a view, or a copy for a strided last axis. Done here
// because each of these operations costs a dispatch in Python.
inline torch::Tensor last_axis_rows(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.dim() >= 1, "the observed tensor must have a last axis");
    auto matrix = tensor.detach().reshape({-1, tensor.size(-1)});
    return matrix.stride(1) == 1 ? matrix : matrix.contiguous();
}

// The last-axis rows of `tensor` a kernel reads: `tensor` itself when it is contiguous, so that no op is dispatched
// for it, otherwise `last_axis_rows(tensor)`, which `copy` holds.
inline const torch::Tensor& rows_to_read(const torch::Tensor& tensor, torch::Tensor& copy) {
    if (tensor.dim() >= 1 && tensor.is_contiguous()) {
        return tensor;
    }
    copy = last_axis_rows(tensor);
    return copy;
}

// The distance between the rows `rows_to_read` returned.
inline int64_t rows_stride(const torch::Tensor& rows) {
    return rows.dim() == 2 ? rows.stride(0) : rows.size(-1);
}

inline void check_rows(const torch::Tensor& matrix) {
    TORCH_CHECK(matrix.is_cuda() && matrix.dim() == 2 && matrix.stride(1) == 1 && matrix.size(0) > 0
                    && matrix.size(1) > 0,
                "matrix must be a non-empty 2D CUDA tensor with a contiguous last axis");
}

inline int64_t row_grid(int64_t rows, int device) {
    int64_t blocks = ceil_div(rows, kWarps);
    return std::min<int64_t>(blocks, static_cast<int64_t>(sm_count_for_device(device)) * kBlocksPerSm);
}

inline int64_t row_grid(const torch::Tensor& matrix) {
    return row_grid(matrix.size(0), matrix.get_device());
}

}  // namespace
