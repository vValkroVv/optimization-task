// The native kernel reduces 4096 adjacent vocabulary entries at a time.
// Each 128-thread logical tile uses 8 adjacent columns per lane, then 4
// striped vectors per lane. Sum within the lane, then within the warp,
// then combine warp 0 with 2 and warp 1 with 3.
constexpr int tile_width = 4096;
constexpr int tiles_per_row = padded_vocab / tile_width;

__device__ __forceinline__ unsigned int bf16_pair_max(unsigned int a, unsigned int b) {
    unsigned int output;
    asm("max.bf16x2 %0, %1, %2;" : "=r"(output) : "r"(a), "r"(b));
    return output;
}

__device__ __forceinline__ unsigned int replace_bf16(unsigned int pair, float value, int position) {
    unsigned short adjusted;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(adjusted) : "f"(value));
    const unsigned int replacement = adjusted;
    return position == 0 ? (pair & 0xffff0000U) | replacement
                         : (pair & 0x0000ffffU) | (replacement << 16);
}

template <int ctas, int threads, int optimization = 0>
__global__ __launch_bounds__(threads) void exact_ce(
    uint4* logits, const int64_t* target, float* loss,
    const float* scales, int rows_per_head
) {
    constexpr int tiles = tiles_per_row / ctas;
    constexpr int part = tiles * tile_width;
    constexpr int vector_tile = tile_width / vector_width;
    constexpr int warps_per_tile = (threads / 32) / tiles;
    static_assert(4 % warps_per_tile == 0);
    const auto cluster = cg::this_cluster();
    const int rank = cluster.block_rank();
    const int row = blockIdx.x / ctas;
    const int lane = threadIdx.x % 32;
    const int thread_warp = threadIdx.x / 32;
    const int tile = thread_warp / warps_per_tile;
    const int warp = thread_warp % warps_per_tile;
    const int base = rank * part;
    const int width = vocab / vector_width;
    auto* vectors = logits + static_cast<size_t>(row) * width + base / vector_width;
    const int64_t source_label = target[row];
    if (source_label == -100) {
        for (int i = threadIdx.x; i < part / vector_width; i += threads) {
            if (base + i * vector_width < vocab) vectors[i] = {};
        }
        if (rank == 0 && threadIdx.x == 0) loss[row] = 0.0f;
        return;
    }
    assert(source_label >= 0 && source_label < vocab);
    const int label = static_cast<int>(source_label);

    extern __shared__ uint4 staged[];
    __shared__ float tile_max[tiles], tile_sum[tiles];
    __shared__ float warp_data[tiles][4];
    __shared__ float prefix[tiles_per_row];
    __shared__ float log_denominator;
    auto* tile_vectors = vectors + tile * vector_tile;
    auto* tile_staged = staged + tile * vector_tile;
    const int tile_base = base + tile * tile_width;

    // Keep the whole BF16 row in cluster-local shared memory through the
    // prefix maximum, native sum, and gradient phases.
    float local_max = -CUDART_INF_F;
    unsigned int packed_max = 0xff80ff80U;
    #pragma unroll
    for (int i = 0; i < 16 / warps_per_tile; ++i) {
        const int vector = threadIdx.x % (32 * warps_per_tile) + i * 32 * warps_per_tile;
        const bool valid = tile_base + vector * vector_width < vocab;
        uint4 values = valid ? tile_vectors[vector] : make_uint4(
            0xff80ff80U, 0xff80ff80U, 0xff80ff80U, 0xff80ff80U
        );
        tile_staged[vector] = values;
        const unsigned int pairs[] = {values.x, values.y, values.z, values.w};
        #pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            if constexpr (optimization > 0) packed_max = bf16_pair_max(packed_max, pairs[pair]);
            else local_max = fmaxf(local_max, fmaxf(unpack(pairs[pair], 0), unpack(pairs[pair], 1)));
        }
    }
    if constexpr (optimization > 0) {
        local_max = fmaxf(unpack(packed_max, 0), unpack(packed_max, 1));
    }
    #pragma unroll
    for (int step = 16; step > 0; step /= 2) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, step));
    }
    if (lane == 0) warp_data[tile][warp] = local_max;
    __syncthreads();
    if (warp == 0 && lane == 0) {
        local_max = warp_data[tile][0];
        #pragma unroll
        for (int i = 1; i < warps_per_tile; ++i) {
            local_max = fmaxf(local_max, warp_data[tile][i]);
        }
        tile_max[tile] = local_max;
    }
    cluster.sync();

    if (thread_warp == 0) {
        float value = cluster.map_shared_rank(tile_max, lane / tiles)[lane % tiles];
        #pragma unroll
        for (int step = 1; step < tiles_per_row; step *= 2) {
            const float earlier = __shfl_up_sync(0xffffffff, value, step);
            if (lane >= step) value = fmaxf(value, earlier);
        }
        prefix[lane] = value;
    }
    __syncthreads();

    const float center = prefix[rank * tiles + tile];
    #pragma unroll
    for (int logical_warp = warp; logical_warp < 4; logical_warp += warps_per_tile) {
        float stripe_sum[4];
        #pragma unroll
        for (int stripe = 0; stripe < 4; ++stripe) {
            const auto values = tile_staged[(stripe * 4 + logical_warp) * 32 + lane];
            const unsigned int pairs[] = {values.x, values.y, values.z, values.w};
            float pair_sum[4];
            #pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const float low = centered_exp(unpack(pairs[pair], 0) - center);
                const float high = centered_exp(unpack(pairs[pair], 1) - center);
                pair_sum[pair] = low + high;
            }
            stripe_sum[stripe] = (pair_sum[0] + pair_sum[1]) + (pair_sum[2] + pair_sum[3]);
        }
        float subtotal = (stripe_sum[0] + stripe_sum[1]) + (stripe_sum[2] + stripe_sum[3]);
        #pragma unroll
        for (int step = 16; step > 0; step /= 2) {
            subtotal += __shfl_xor_sync(0xffffffff, subtotal, step);
        }
        if (lane == 0) warp_data[tile][logical_warp] = subtotal;
    }
    __syncthreads();
    if (warp == 0 && lane == 0) {
        const auto* partial = warp_data[tile];
        tile_sum[tile] = (partial[0] + partial[2]) + (partial[1] + partial[3]);
    }
    cluster.sync();

    if (thread_warp == 0) {
        const float subtotal = cluster.map_shared_rank(tile_sum, lane / tiles)[lane % tiles];
        const float lane_center = prefix[lane];
        float denominator = 0.0f;
        float last = -CUDART_INF_F;
        for (int i = 0; i < tiles_per_row; ++i) {
            const float next = __shfl_sync(0xffffffff, lane_center, i);
            const float add = __shfl_sync(0xffffffff, subtotal, i);
            denominator = denominator * centered_exp(last - next) + add;
            last = next;
        }
        if (lane == 0) log_denominator = logf(denominator);
    }
    __syncthreads();
    const float maximum = prefix[tiles_per_row - 1];
    const float logsum = log_denominator;
    if (rank == label / part && threadIdx.x == 0) {
        uint4 values = staged[(label - base) / vector_width];
        const unsigned int pairs[] = {values.x, values.y, values.z, values.w};
        const float chosen = unpack(pairs[(label / 2) % 4], label % 2);
        loss[row] = -((chosen - maximum) - logsum);
    }
    const float scale = scales[row >= rows_per_head];
    #pragma unroll
    for (int i = 0; i < part / (threads * vector_width); ++i) {
        const int vector = threadIdx.x + i * threads;
        const int start = base + vector * vector_width;
        if (start < vocab) {
            uint4 values = staged[vector];
            const unsigned int pairs[] = {values.x, values.y, values.z, values.w};
            unsigned int output[4];
            #pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const int index = start + pair * 2;
                const float low = centered_exp((unpack(pairs[pair], 0) - maximum) - logsum);
                const float high = centered_exp((unpack(pairs[pair], 1) - maximum) - logsum);
                if constexpr (optimization == 2) {
                    output[pair] = pack(low * scale, high * scale);
                } else {
                    output[pair] = pack(
                        (low - static_cast<float>(label == index)) * scale,
                        (high - static_cast<float>(label == index + 1)) * scale
                    );
                }
            }
            if constexpr (optimization == 2) {
                if (start / vector_width == label / vector_width) {
                    const int pair = (label / 2) % 4;
                    const int position = label % 2;
                    const float item = unpack(pairs[pair], position);
                    const float grad = (centered_exp((item - maximum) - logsum) - 1.0f) * scale;
                    output[pair] = replace_bf16(output[pair], grad, position);
                }
            }
            vectors[vector] = make_uint4(output[0], output[1], output[2], output[3]);
        }
    }
    cluster.sync();
}
