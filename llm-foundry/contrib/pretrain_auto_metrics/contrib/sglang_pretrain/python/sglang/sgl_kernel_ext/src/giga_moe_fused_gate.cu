// GigaChat 3.5 fused MoE top-K gate kernel. Adapted from sgl-kernel's
// kimi_k2_moe_fused_gate.cu (NUM_EXPERTS=384), specialized for NUM_EXPERTS=256
// and templated on input/bias dtype so the caller can pass bf16 directly —
// the cast to fp32 happens inside the kernel on the load, with no extra
// elementwise cast kernel in the trace.
#include "giga_moe_fused_gate.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <cfloat>

namespace sgl_kernel_ext {

// ────────────────────────────────────────────────────────────────────────────
// Compile-time constants. Tuned for NUM_EXPERTS=256 (GigaChat 3.5 MoE).
// ────────────────────────────────────────────────────────────────────────────
static constexpr int WARP_SIZE = 32;
static constexpr int WARPS_PER_CTA = 6;
static constexpr int NUM_EXPERTS = 256;
static constexpr int VPT = NUM_EXPERTS / WARP_SIZE;  // 8 (experts per warp lane)

static constexpr int SMALL_TOKEN_THRESHOLD = 512;
static constexpr int WARPS_PER_TOKEN_SMALL = VPT;                                  // 8
static constexpr int THREADS_PER_BLOCK_SMALL = WARPS_PER_TOKEN_SMALL * WARP_SIZE;  // 256

// Upper bound for selected_experts shared buffer (covers topk ≤ 16).
static constexpr int SELECTED_BUF = 16;

// Scalar load helper. Works for float, at::BFloat16, at::Half (PyTorch types
// expose `operator float()` in device code).
template <typename T>
__device__ __forceinline__ float load_as_float(const T* ptr, int idx) {
  return static_cast<float>(ptr[idx]);
}

// ────────────────────────────────────────────────────────────────────────────
// Small token kernel (num_rows ≤ SMALL_TOKEN_THRESHOLD).
// One block per token, NUM_EXPERTS=256 threads = one thread per expert.
// ────────────────────────────────────────────────────────────────────────────
template <typename InputT, typename BiasT>
__global__ void giga_moe_fused_gate_kernel_small_token(const InputT* __restrict__ input, const BiasT* __restrict__ bias,
                                                       float* output_ptr, int32_t* indices_ptr, int64_t num_rows,
                                                       int64_t topk, int64_t num_fused_shared, bool renormalize,
                                                       float routed_scaling_factor,
                                                       bool apply_routed_scaling_factor_on_output) {
  int64_t row_idx = blockIdx.x;
  if (row_idx >= num_rows)
    return;

  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  __shared__ float shared_scores[NUM_EXPERTS];
  __shared__ float shared_original_scores[NUM_EXPERTS];
  __shared__ int selected_experts[SELECTED_BUF];
  __shared__ float warp_maxs[WARPS_PER_TOKEN_SMALL];
  __shared__ int warp_experts[WARPS_PER_TOKEN_SMALL];

  // Load + sigmoid + bias in fp32, regardless of input dtype.
  if (tid < NUM_EXPERTS) {
    float input_val = load_as_float<InputT>(input, row_idx * NUM_EXPERTS + tid);
    float bias_val = load_as_float<BiasT>(bias, tid);
    float sigmoid_val = 1.0f / (1.0f + expf(-input_val));
    float biased_val = sigmoid_val + bias_val;
    shared_scores[tid] = biased_val;
    shared_original_scores[tid] = sigmoid_val;
  }

  __syncthreads();

  for (int k = 0; k < topk; k++) {
    float my_val = (tid < NUM_EXPERTS) ? shared_scores[tid] : -FLT_MAX;
    int my_expert = tid;

    float warp_max_val = my_val;
    int warp_max_expert = my_expert;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      float other_val = __shfl_down_sync(0xFFFFFFFF, warp_max_val, offset);
      int other_expert = __shfl_down_sync(0xFFFFFFFF, warp_max_expert, offset);
      if (other_val > warp_max_val) {
        warp_max_val = other_val;
        warp_max_expert = other_expert;
      }
    }

    if (lane_id == 0) {
      warp_maxs[warp_id] = warp_max_val;
      warp_experts[warp_id] = warp_max_expert;
    }

    __syncthreads();

    if (warp_id == 0) {
      float final_max = (lane_id < WARPS_PER_TOKEN_SMALL) ? warp_maxs[lane_id] : -FLT_MAX;
      int final_expert = (lane_id < WARPS_PER_TOKEN_SMALL) ? warp_experts[lane_id] : -1;
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xFFFFFFFF, final_max, offset);
        int other_expert = __shfl_down_sync(0xFFFFFFFF, final_expert, offset);
        if (other_val > final_max) {
          final_max = other_val;
          final_expert = other_expert;
        }
      }
      if (lane_id == 0) {
        selected_experts[k] = final_expert;
      }
    }

    __syncthreads();

    int selected = selected_experts[k];
    if (tid == selected) {
      shared_scores[tid] = -FLT_MAX;
    }

    __syncthreads();
  }

  if (tid == 0) {
    for (int k = 0; k < topk; k++) {
      int expert_id = selected_experts[k];
      if (expert_id >= 0 && expert_id < NUM_EXPERTS) {
        output_ptr[row_idx * topk + k] = shared_original_scores[expert_id];
        indices_ptr[row_idx * topk + k] = expert_id;
      } else {
        output_ptr[row_idx * topk + k] = 0.0f;
        indices_ptr[row_idx * topk + k] = 0;
      }
    }

    // Phantom-shared fusion: last slot becomes a phantom expert
    // (id = NUM_EXPERTS, weight = sum_real / routed_scaling_factor). After the
    // split renormalize below, phantom weight ends up at 1/routed_scaling_factor.
    int real_topk = topk - num_fused_shared;
    if (num_fused_shared > 0) {
      float sum_real = 0.0f;
      for (int k = 0; k < real_topk; k++) {
        sum_real += output_ptr[row_idx * topk + k];
      }
      int last = topk - 1;
      output_ptr[row_idx * topk + last] = sum_real / routed_scaling_factor;
      indices_ptr[row_idx * topk + last] = NUM_EXPERTS;
    }

    if (renormalize) {
      int sum_count = (num_fused_shared > 0) ? real_topk : topk;
      float sum = 0.0f;
      for (int k = 0; k < sum_count; k++) {
        sum += output_ptr[row_idx * topk + k];
      }
      if (sum > 0.0f) {
        for (int k = 0; k < topk; k++) {
          int64_t idx = row_idx * topk + k;
          output_ptr[idx] /= sum;
          if (apply_routed_scaling_factor_on_output) {
            output_ptr[idx] *= routed_scaling_factor;
          }
        }
      }
    }
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Large token kernel (num_rows > SMALL_TOKEN_THRESHOLD). One warp per row.
// Scalar loads (vectorized variant dropped for dtype-agnostic clarity).
// ────────────────────────────────────────────────────────────────────────────
template <typename InputT, typename BiasT>
__global__ void giga_moe_fused_gate_kernel(const InputT* __restrict__ input, const BiasT* __restrict__ bias,
                                           float* output_ptr, int32_t* indices_ptr, int64_t num_rows, int64_t topk,
                                           int64_t num_fused_shared, bool renormalize, float routed_scaling_factor,
                                           bool apply_routed_scaling_factor_on_output) {
  int64_t row_idx = blockIdx.x * WARPS_PER_CTA + threadIdx.y;
  if (row_idx >= num_rows)
    return;

  int lane_id = threadIdx.x;
  int warp_id = threadIdx.y;

  __shared__ float shared_scores[NUM_EXPERTS * WARPS_PER_CTA];
  __shared__ float shared_original_scores[NUM_EXPERTS * WARPS_PER_CTA];

  float* warp_scores = shared_scores + warp_id * NUM_EXPERTS;
  float* warp_original_scores = shared_original_scores + warp_id * NUM_EXPERTS;

  // Each lane handles VPT=8 experts.
#pragma unroll
  for (int i = 0; i < VPT; i++) {
    int expert = lane_id * VPT + i;
    float inp = load_as_float<InputT>(input, row_idx * NUM_EXPERTS + expert);
    float b = load_as_float<BiasT>(bias, expert);
    float sigmoid_val = 1.0f / (1.0f + expf(-inp));
    warp_scores[expert] = sigmoid_val + b;
    warp_original_scores[expert] = sigmoid_val;
  }

  __syncthreads();

  for (int k = 0; k < topk; k++) {
    float max_val = -FLT_MAX;
    int max_expert = -1;

    for (int expert = lane_id; expert < NUM_EXPERTS; expert += WARP_SIZE) {
      if (warp_scores[expert] > max_val) {
        max_val = warp_scores[expert];
        max_expert = expert;
      }
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      float other_val = __shfl_down_sync(0xFFFFFFFF, max_val, offset);
      int other_expert = __shfl_down_sync(0xFFFFFFFF, max_expert, offset);
      if (other_val > max_val || (other_val == max_val && other_expert < max_expert)) {
        max_val = other_val;
        max_expert = other_expert;
      }
    }

    if (lane_id == 0) {
      int64_t output_idx = row_idx * topk + k;
      if (max_expert != -1) {
        output_ptr[output_idx] = warp_original_scores[max_expert];
        indices_ptr[output_idx] = max_expert;
        warp_scores[max_expert] = -FLT_MAX;
      } else {
        output_ptr[output_idx] = 0.0f;
        indices_ptr[output_idx] = 0;
      }
    }

    __syncwarp();
  }

  __syncthreads();

  if (lane_id == 0) {
    // Phantom-shared fusion — see small-token kernel for semantics.
    int real_topk = topk - num_fused_shared;
    if (num_fused_shared > 0) {
      float sum_real = 0.0f;
      for (int k = 0; k < real_topk; k++) {
        sum_real += output_ptr[row_idx * topk + k];
      }
      int last = topk - 1;
      output_ptr[row_idx * topk + last] = sum_real / routed_scaling_factor;
      indices_ptr[row_idx * topk + last] = NUM_EXPERTS;
    }

    if (renormalize) {
      int sum_count = (num_fused_shared > 0) ? real_topk : topk;
      float sum = 0.0f;
      for (int k = 0; k < sum_count; k++) {
        sum += output_ptr[row_idx * topk + k];
      }
      if (sum > 0.0f) {
        for (int k = 0; k < topk; k++) {
          int64_t idx = row_idx * topk + k;
          output_ptr[idx] /= sum;
          if (apply_routed_scaling_factor_on_output) {
            output_ptr[idx] *= routed_scaling_factor;
          }
        }
      }
    }
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Host launcher: dispatches by input dtype.
// ────────────────────────────────────────────────────────────────────────────
template <typename InputT, typename BiasT>
static void launch_kernels(const InputT* input_ptr, const BiasT* bias_ptr, float* output_ptr, int32_t* indices_ptr,
                           int64_t num_rows, int64_t topk, int64_t num_fused_shared, bool renormalize,
                           float routed_scaling_factor, bool apply_routed_scaling_factor_on_output,
                           cudaStream_t stream) {
  const bool use_small = num_rows <= SMALL_TOKEN_THRESHOLD;
  if (use_small) {
    dim3 block(THREADS_PER_BLOCK_SMALL);
    giga_moe_fused_gate_kernel_small_token<InputT, BiasT>
        <<<num_rows, block, 0, stream>>>(input_ptr, bias_ptr, output_ptr, indices_ptr, num_rows, topk, num_fused_shared,
                                         renormalize, routed_scaling_factor, apply_routed_scaling_factor_on_output);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    int64_t num_blocks = (num_rows + WARPS_PER_CTA - 1) / WARPS_PER_CTA;
    dim3 block(WARP_SIZE, WARPS_PER_CTA);
    giga_moe_fused_gate_kernel<InputT, BiasT><<<num_blocks, block, 0, stream>>>(
        input_ptr, bias_ptr, output_ptr, indices_ptr, num_rows, topk, num_fused_shared, renormalize,
        routed_scaling_factor, apply_routed_scaling_factor_on_output);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

std::tuple<at::Tensor, at::Tensor> giga_moe_fused_gate(at::Tensor& input, at::Tensor& bias, int64_t topk,
                                                       int64_t num_fused_shared, bool renormalize,
                                                       double routed_scaling_factor,
                                                       bool apply_routed_scaling_factor_on_output) {
  TORCH_CHECK(input.is_cuda() && bias.is_cuda(), "inputs must be on CUDA");
  TORCH_CHECK(input.dim() == 2, "input must be 2D [num_rows, num_experts], got dim=", input.dim());
  TORCH_CHECK(bias.dim() == 1, "bias must be 1D [num_experts], got dim=", bias.dim());
  // Kernel indexes via row_idx * NUM_EXPERTS + expert and bias[expert] — both
  // tensors must be physically contiguous; a transpose/slice would silently
  // read the wrong memory.
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

  int64_t num_rows = input.size(0);
  int32_t num_experts = input.size(1);

  TORCH_CHECK(num_experts == NUM_EXPERTS, "giga_moe_fused_gate is specialized for NUM_EXPERTS=", NUM_EXPERTS,
              ", got num_experts=", num_experts);
  TORCH_CHECK(bias.numel() == NUM_EXPERTS, "bias must have ", NUM_EXPERTS, " elements, got ", bias.numel());
  TORCH_CHECK(topk > 0 && topk <= SELECTED_BUF, "giga_moe_fused_gate supports topk in (0, ", SELECTED_BUF, "], got ",
              topk);
  // Only N ∈ {0, 1} is implemented. N=1 matches biased_grouped_topk_impl exactly
  // (one phantom slot at the end). For N>1 both that reference and a kernel
  // implementation would be semantically ambiguous (Python picks the phantom id
  // randomly from N options; the renorm divisor would also need a non-obvious
  // interpretation) — left unimplemented until a model actually needs it.
  TORCH_CHECK(num_fused_shared == 0 || num_fused_shared == 1,
              "giga_moe_fused_gate only supports num_fused_shared in {0, 1}, got ", num_fused_shared);
  TORCH_CHECK(num_fused_shared < topk, "num_fused_shared (", num_fused_shared, ") must be < topk (", topk, ")");
  // Phantom weight is sum_real / routed_scaling_factor when fusion is on; even
  // without fusion a zero factor would zero out the output via
  // apply_routed_scaling_factor_on_output. Either way it is a misuse.
  TORCH_CHECK(routed_scaling_factor != 0.0, "routed_scaling_factor must be non-zero");

  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto output = torch::empty({num_rows, topk}, options);
  auto indices = torch::empty({num_rows, topk}, options.dtype(torch::kInt32));

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const float scaling = static_cast<float>(routed_scaling_factor);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "giga_moe_fused_gate.input", [&] {
        using input_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16, bias.scalar_type(), "giga_moe_fused_gate.bias", [&] {
              using bias_t = scalar_t;
              launch_kernels<input_t, bias_t>(input.data_ptr<input_t>(), bias.data_ptr<bias_t>(),
                                              output.data_ptr<float>(), indices.data_ptr<int32_t>(), num_rows, topk,
                                              num_fused_shared, renormalize, scaling,
                                              apply_routed_scaling_factor_on_output, stream);
            });
      });

  return std::make_tuple(output, indices);
}

}  // namespace sgl_kernel_ext
