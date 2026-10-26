#pragma once

#include <ATen/ATen.h>

namespace sgl_kernel_ext {

/// GigaChat 3.5 variant of kimi_k2_moe_fused_gate: flat top-K
/// (num_expert_group=1) over `NUM_EXPERTS=256` experts with sigmoid +
/// correction-bias scoring, optional renormalize, optional output scaling.
/// Single CUDA kernel — fuses the sigmoid, biased top-K selection, and
/// renormalize-by-sum into one pass per token.
///
/// Tensors:
///   input  : [num_rows, 256]   gating logits (router output)
///   bias   : [256]             per-expert correction bias
///   output : [num_rows, topk]  float32 — top-K gate weights (post-renorm,
///                                        post-scale if requested)
///   indices: [num_rows, topk]  int32   — selected expert ids
///
/// Dtypes (independent per tensor — input and bias may differ):
///   input.dtype  ∈ {float32, bfloat16, float16}
///   bias.dtype   ∈ {float32, bfloat16, float16}
///
/// The kernel casts both to fp32 on load and runs sigmoid / top-K / renorm
/// in fp32. Output is always float32 (weights) + int32 (indices), regardless
/// of input dtype. The independent dispatch matches `MoEGate.__init__` in
/// `models/deepseek_v2.py`, where `e_score_correction_bias` defaults to fp32
/// for router stability even when the gate output is bf16.
///
/// Returns:
///   std::tuple{output, indices}
///     output  : [num_rows, topk] float32
///     indices : [num_rows, topk] int32
/// num_fused_shared: number of phantom "shared" experts folded into the
/// top-K. For N > 0, the kernel mirrors biased_grouped_topk_impl with
/// num_fused_shared_experts=N: real-routed top-K is selected over the 256
/// experts, then the last slot is overwritten — id = NUM_EXPERTS,
/// weight = sum(first topk-N) / routed_scaling_factor. On renormalize the
/// divisor is the same sum, so the phantom weight ends up at exactly
/// 1/routed_scaling_factor and the real-routed weights sum to 1.
std::tuple<at::Tensor, at::Tensor> giga_moe_fused_gate(at::Tensor& input, at::Tensor& bias, int64_t topk,
                                                       int64_t num_fused_shared, bool renormalize,
                                                       double routed_scaling_factor,
                                                       bool apply_routed_scaling_factor_on_output);

}  // namespace sgl_kernel_ext
