#include <assert.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace cg = cooperative_groups;

constexpr int vocab = 128256;
constexpr int padded_vocab = 131072;
constexpr int vector_width = 8;

__device__ __forceinline__ float centered_exp(float input) {
    float output;
    float base2 = input * 0x1.715476p+0f;
    asm("ex2.approx.f32 %0, %1;" : "=f"(output) : "f"(base2));
    return output;
}

__device__ __forceinline__ float centered_log(float input) {
    float output;
    asm("lg2.approx.f32 %0, %1;" : "=f"(output) : "f"(input));
    return output * 0x1.62e43p-1f;
}

__device__ __forceinline__ float unpack(unsigned int pair, int index) {
    return __uint_as_float(index == 0 ? pair << 16 : pair & 0xffff0000U);
}

__device__ __forceinline__ unsigned int pack(float low, float high) {
    unsigned int pair;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(pair) : "f"(high), "f"(low));
    return pair;
}

template <bool maximum>
__device__ __forceinline__ float reduce_pair(float left, float right) {
    if constexpr (maximum) return fmaxf(left, right);
    return left + right;
}

template <int threads, bool maximum>
__device__ __forceinline__ float reduce_cta(float item, float* warps) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    #pragma unroll
    for (int step = 16; step > 0; step /= 2) {
        item = reduce_pair<maximum>(item, __shfl_xor_sync(0xffffffff, item, step));
    }
    if (lane == 0) warps[warp] = item;
    __syncthreads();
    if (warp == 0) {
        item = lane < threads / 32 ? warps[lane] : (maximum ? -CUDART_INF_F : 0.0f);
        #pragma unroll
        for (int step = threads / 64; step > 0; step /= 2) {
            item = reduce_pair<maximum>(item, __shfl_xor_sync(0xffffffff, item, step));
        }
    }
    return item;
}

template <int ctas, int threads, bool shared_input>
__global__ __launch_bounds__(threads) void centered_ce(
    uint4* logits, const int64_t* target, float* loss,
    const float* scales, int rows_per_head
) {
    constexpr int part = padded_vocab / ctas;
    constexpr int thread_stride = threads * vector_width;
    constexpr int items = part / thread_stride;
    constexpr int width = vocab / vector_width;
    const auto cluster = cg::this_cluster();
    const int cluster_rank = cluster.block_rank();
    const int row = blockIdx.x / ctas;
    const int column = cluster_rank * part + threadIdx.x * vector_width;
    auto* vectors = logits + static_cast<size_t>(row) * width + column / vector_width;
    const int64_t label = target[row];
    if (label == -100) {
        #pragma unroll
        for (int i = 0; i < items; ++i) {
            if (column + i * thread_stride < vocab) vectors[i * threads] = {};
        }
        if (cluster_rank == 0 && threadIdx.x == 0) loss[row] = 0.0f;
        return;
    }
    assert(label >= 0 && label < vocab);

    extern __shared__ uint4 staged[];
    __shared__ float warps[threads / 32];
    __shared__ float local_maximum, local_sum, local_target;
    uint4 retained[shared_input ? 1 : items];
    float maximum = -CUDART_INF_F;
    float chosen = 0.0f;
    const int local_vector = label / vector_width - column / vector_width;
    const int first_pair = label % vector_width / 2;
    auto* scratch = staged + threadIdx.x;
    #pragma unroll
    for (int i = 0; i < items; ++i) {
        const bool valid = column + i * thread_stride < vocab;
        uint4 values = valid ? vectors[i * threads] : make_uint4(
            0xff80ff80U, 0xff80ff80U, 0xff80ff80U, 0xff80ff80U
        );
        if constexpr (shared_input) scratch[i * threads] = values;
        else retained[i] = values;
        const unsigned int pairs[] = {values.x, values.y, values.z, values.w};
        #pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            float low = unpack(pairs[pair], 0);
            float high = unpack(pairs[pair], 1);
            maximum = fmaxf(maximum, fmaxf(low, high));
            if (local_vector == i * threads && pair == first_pair) {
                chosen = (label & 1) ? high : low;
                local_target = chosen;
            }
        }
    }
    maximum = reduce_cta<threads, true>(maximum, warps);
    if (threadIdx.x == 0) local_maximum = maximum;
    cluster.sync();

    maximum = -CUDART_INF_F;
    #pragma unroll
    for (int rank = 0; rank < ctas; ++rank) {
        maximum = fmaxf(maximum, *cluster.map_shared_rank(&local_maximum, rank));
    }

    float sums[4] = {};
    #pragma unroll
    for (int i = 0; i < items; ++i) {
        uint4 values = shared_input ? scratch[i * threads] : retained[i];
        const unsigned int pairs[] = {values.x, values.y, values.z, values.w};
        #pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float low = centered_exp(unpack(pairs[pair], 0) - maximum);
            const float high = centered_exp(unpack(pairs[pair], 1) - maximum);
            sums[pair] += low + high;
        }
    }
    float sum = reduce_cta<threads, false>((sums[0] + sums[1]) + (sums[2] + sums[3]), warps);
    if (threadIdx.x == 0) local_sum = sum;
    cluster.sync();

    sum = 0.0f;
    #pragma unroll
    for (int rank = 0; rank < ctas; ++rank) {
        sum += *cluster.map_shared_rank(&local_sum, rank);
    }
    const float log_denominator = centered_log(sum);
    if (cluster_rank == label / part && threadIdx.x == 0) {
        loss[row] = -((local_target - maximum) - log_denominator);
    }
    const float scale = scales[row >= rows_per_head];
    #pragma unroll
    for (int i = 0; i < items; ++i) {
        const int start = column + i * thread_stride;
        if (start < vocab) {
            uint4 values = shared_input ? scratch[i * threads] : retained[i];
            const unsigned int pairs[] = {values.x, values.y, values.z, values.w};
            unsigned int output[4];
            #pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const int index = start + pair * 2;
                const float low = centered_exp((unpack(pairs[pair], 0) - maximum) - log_denominator)
                    - static_cast<float>(label == index);
                const float high = centered_exp((unpack(pairs[pair], 1) - maximum) - log_denominator)
                    - static_cast<float>(label == index + 1);
                output[pair] = pack(low * scale, high * scale);
            }
            vectors[i * threads] = make_uint4(output[0], output[1], output[2], output[3]);
        }
    }
    // Distributed shared-memory readers must finish before any CTA exits.
    cluster.sync();
}

#include "exact.cuh"

template <int ctas, int threads, bool shared_input, bool exact = false, int optimization = 0>
cudaError_t launch(
    void* logits, const void* labels, void* loss, const void* scales,
    int rows, cudaStream_t stream
) {


    constexpr auto kernel = []{
        if constexpr (exact) return exact_ce<ctas, threads, optimization>;
        else return centered_ce<ctas, threads, shared_input>;
    }();
    constexpr int shared_bytes = shared_input ? padded_vocab / ctas * 2 : 0;
    if constexpr (shared_bytes > 48 * 1024) {
        auto status = cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes
        );
        if (status != cudaSuccess) return status;
    }
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = ctas;
    attribute.val.clusterDim.y = 1;
    attribute.val.clusterDim.z = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(rows * ctas);
    config.blockDim = dim3(threads);
    config.dynamicSmemBytes = shared_bytes;
    config.stream = stream;
    config.attrs = &attribute;
    config.numAttrs = 1;
    return cudaLaunchKernelEx(
        &config, kernel,
        static_cast<uint4*>(logits), static_cast<const int64_t*>(labels),
        static_cast<float*>(loss), static_cast<const float*>(scales), rows / 2
    );
}

extern "C" int centered_ce_launch(
    void* logits, const void* labels, void* loss, const void* scales,
    int rows, void* stream, int config
) {
    auto cuda_stream = static_cast<cudaStream_t>(stream);
    switch (config) {
        case 0: return launch<2, 512, false>(logits, labels, loss, scales, rows, cuda_stream);
        case 1: return launch<4, 256, false>(logits, labels, loss, scales, rows, cuda_stream);
        case 2: return launch<8, 128, false>(logits, labels, loss, scales, rows, cuda_stream);
        case 3: return launch<4, 512, false>(logits, labels, loss, scales, rows, cuda_stream);
        case 4: return launch<4, 256, true>(logits, labels, loss, scales, rows, cuda_stream);
        case 5: return launch<8, 128, true>(logits, labels, loss, scales, rows, cuda_stream);
        case 6: return launch<2, 256, true>(logits, labels, loss, scales, rows, cuda_stream);
        case 7: return launch<2, 512, true>(logits, labels, loss, scales, rows, cuda_stream);
        case 8: return launch<4, 128, false>(logits, labels, loss, scales, rows, cuda_stream);
        case 9: return launch<8, 256, false>(logits, labels, loss, scales, rows, cuda_stream);
        case 10: return launch<4, 256, true, true>(logits, labels, loss, scales, rows, cuda_stream);
        case 11: return launch<4, 512, true, true>(logits, labels, loss, scales, rows, cuda_stream);
        case 12: return launch<8, 128, true, true>(logits, labels, loss, scales, rows, cuda_stream);
        case 13: return launch<8, 256, true, true>(logits, labels, loss, scales, rows, cuda_stream);
        case 14: return launch<8, 512, true, true>(logits, labels, loss, scales, rows, cuda_stream);
        case 15: return launch<2, 512, true, true>(logits, labels, loss, scales, rows, cuda_stream);
        case 16: return launch<8, 256, true, true, 1>(logits, labels, loss, scales, rows, cuda_stream);
        case 17: return launch<8, 256, true, true, 2>(logits, labels, loss, scales, rows, cuda_stream);
        case 18: return launch<4, 512, true, true, 1>(logits, labels, loss, scales, rows, cuda_stream);
        case 19: return launch<4, 512, true, true, 2>(logits, labels, loss, scales, rows, cuda_stream);
        default: return cudaErrorInvalidValue;
    }
}
