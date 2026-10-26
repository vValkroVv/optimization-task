#include "common.h"
#include "gigachat35_gated_norm_reduce.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace sgl_kernel_ext {
namespace {

using bf16 = __nv_bfloat16;

// GigaChat3.5 gated-norm bottleneck (gate_up: H -> PROJ_DIM, gate_down back).
constexpr int PROJ_DIM = 16;
// Each thread loads a float4 (4 channels), so PROJ_DIM / 4 threads cover one
// split-row of PROJ_DIM channels.
constexpr int THREADS_PER_ROW = PROJ_DIM / 4;  // 4
// A 32-thread warp thus reads 32 * 4 = 128 channels per iteration, i.e.
// 128 / PROJ_DIM == 32 / THREADS_PER_ROW split-rows.
constexpr int SPLITS_PER_WARP = 32 / THREADS_PER_ROW;  // 8

__device__ __forceinline__ float silu(float x) {
  return x * (1.0f / (1.0f + __expf(-x)));
}

// "warpf4N": N = ceil(NSPLIT / SPLITS_PER_WARP) warps per token, float4 loads
// over pA [NSPLIT, B, PROJ_DIM]. Each (warp, group = lane / THREADS_PER_ROW)
// reads split warp*SPLITS_PER_WARP + group as a float4 covering channels
// [pos*4, pos*4+4) with pos = lane % THREADS_PER_ROW; the SPLITS_PER_WARP groups
// fold via a stride-THREADS_PER_ROW warp shuffle into lanes 0..3, and each warp
// publishes its PROJ_DIM channel-partials to its own shared slot (no atomics).
// S is reduced by warp 0 alone (stride-32 read of pS [NSPLIT, B] + butterfly
// all-reduce, kept in registers). Finally PROJ_DIM threads sum A across the N
// warps and emit g = silu(rms * A) and rms.
__global__ void gated_norm_reduce_kernel(const float* __restrict__ pA, const float* __restrict__ pS,
                                         bf16* __restrict__ g, float* __restrict__ rms_out, int B, int NSPLIT, int H,
                                         float eps) {
  extern __shared__ float smem[];
  const int N = blockDim.x >> 5;  // warps per token
  float* sA = smem;               // N * PROJ_DIM (only A crosses warps; S stays in regs)
  const int b = blockIdx.x, t = threadIdx.x;
  const int warp = t >> 5, lane = t & 31;
  // PDL (sm_90+ only): wait for the producer (gate_up) to finish writing
  // pA/pS before reading them. Compiled out on pre-Hopper (A100 = sm_80).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();
#endif
  const int pos = lane % THREADS_PER_ROW, group = lane / THREADS_PER_ROW;

  const auto* pA4 = reinterpret_cast<const float4*>(pA);
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 1
  for (int sp = warp * SPLITS_PER_WARP + group; sp < NSPLIT; sp += N * SPLITS_PER_WARP) {
    // element (sp*B+b)*PROJ_DIM + pos*4  ==  float4 index (sp*B+b)*4 + pos
    const float4 v = pA4[static_cast<std::size_t>(sp * B + b) * 4 + pos];
    acc.x += v.x;
    acc.y += v.y;
    acc.z += v.z;
    acc.w += v.w;
  }
  // fold the SPLITS_PER_WARP groups (lanes pos, pos+THREADS_PER_ROW, ...) down
  // to lanes 0..3, a stride-THREADS_PER_ROW tree over the top 16 lanes.
#pragma unroll
  for (int off = 16; off >= THREADS_PER_ROW; off >>= 1) {
    acc.x += __shfl_down_sync(0xffffffffu, acc.x, off, 32);
    acc.y += __shfl_down_sync(0xffffffffu, acc.y, off, 32);
    acc.z += __shfl_down_sync(0xffffffffu, acc.z, off, 32);
    acc.w += __shfl_down_sync(0xffffffffu, acc.w, off, 32);
  }
  if (lane < 4) {  // lanes 0..3 hold this warp's A for channels [pos*4, pos*4+4)
    float* dst = sA + warp * PROJ_DIM + pos * 4;
    dst[0] = acc.x;
    dst[1] = acc.y;
    dst[2] = acc.z;
    dst[3] = acc.w;
  }

  // Only warp 0 reduces S: the g-writer threads below are `t < PROJ_DIM`, i.e.
  // lanes 0..PROJ_DIM-1 of warp 0, so they already hold `sum_sq` in registers
  // (no shared staging needed). This holds only while every writer lives in
  // warp 0 -- i.e. PROJ_DIM <= 32.
  static_assert(PROJ_DIM <= 32, "final g-writers (t < PROJ_DIM) must all lie in warp 0");
  float sum_sq = 0.0f;
  if (warp == 0) {
    for (int sp = lane; sp < NSPLIT; sp += 32)
      sum_sq += pS[static_cast<std::size_t>(sp) * B + b];
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, offset, 32);
  }
  __syncthreads();

  if (t < PROJ_DIM) {
    float proj_sum = 0.0f;
    for (int warp_idx = 0; warp_idx < N; ++warp_idx)
      proj_sum += sA[warp_idx * PROJ_DIM + t];
    const float rms = rsqrtf(sum_sq / static_cast<float>(H) + eps);
    g[static_cast<std::size_t>(b) * PROJ_DIM + t] = __float2bfloat16(silu(rms * proj_sum));
    if (t == 0)
      rms_out[b] = rms;
  }
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> gigachat35_gated_norm_reduce(at::Tensor& pA, at::Tensor& pS, int64_t H, double eps) {
  TORCH_CHECK(pA.is_cuda() && pS.is_cuda(), "pA/pS must be CUDA tensors");
  TORCH_CHECK(pA.dim() == 3 && pA.size(2) == PROJ_DIM, "pA must be [NSPLIT, B, PROJ_DIM]");
  TORCH_CHECK(pA.scalar_type() == at::kFloat && pS.scalar_type() == at::kFloat, "pA/pS must be float32");
  TORCH_CHECK(pA.is_contiguous() && pS.is_contiguous(), "pA/pS must be contiguous");
  const int NSPLIT = static_cast<int>(pA.size(0));
  const int B = static_cast<int>(pA.size(1));
  TORCH_CHECK(pS.size(0) == NSPLIT && pS.size(1) == B, "pS must be [NSPLIT, B]");
  // float4 loads over pA require a 16-byte aligned base (torch allocations are).
  TORCH_CHECK((reinterpret_cast<std::uintptr_t>(pA.data_ptr()) & 15) == 0,
              "pA base must be 16-byte aligned for float4 loads");
  // ceil(NSPLIT / SPLITS_PER_WARP): each warp covers SPLITS_PER_WARP splits.
  const int N = (NSPLIT + SPLITS_PER_WARP - 1) / SPLITS_PER_WARP;
  const int block = N * 32;
  TORCH_CHECK(block <= 1024, "ceil(NSPLIT/8)*32 must be <= 1024");

  auto g = at::empty({B, PROJ_DIM}, pA.options().dtype(at::kBFloat16));
  auto rms = at::empty({B}, pA.options());
  const std::size_t smem = static_cast<std::size_t>(N) * PROJ_DIM * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();
  // PDL launch on sm_90+ (Hopper/Blackwell), plain launch on older GPUs (A100).
  launch_maybe_pdl(gated_norm_reduce_kernel, dim3(B), dim3(block), smem, stream, pA.data_ptr<float>(),
                   pS.data_ptr<float>(), reinterpret_cast<bf16*>(g.data_ptr<at::BFloat16>()), rms.data_ptr<float>(), B,
                   NSPLIT, static_cast<int>(H), static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {g, rms};
}

}  // namespace sgl_kernel_ext
