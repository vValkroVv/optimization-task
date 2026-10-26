#pragma once

#include <ATen/ATen.h>

#include <tuple>

namespace sgl_kernel_ext {

/// GigaChat 3.5 ZeroCenteredGatedNorm split-K reduction.
///
/// The Triton gate_up kernel splits the hidden dim H into NSPLIT chunks and
/// writes per-chunk partials:
///   pA : [NSPLIT, B, 16] float32  -- partial  A_k = sum_h W_up[k,h] * u_h
///   pS : [NSPLIT, B]     float32  -- partial  sum_h x_h^2
/// (u_h = x_h * (1 + weight_h), which is rms-independent, so rms factors out
///  of the gate_up GEMM.)
///
/// This op reduces along the NSPLIT axis (one CUDA block per token), then
/// applies the norm scalar and the gate prologue:
///   rms  = rsqrt(sum(pS)/H + eps)
///   g    = silu(rms * sum(pA))            -- [B, 16] bf16
/// and returns rms so the gate_down kernel can rebuild the normalized y.
///
/// Returns:
///   std::tuple{g, rms}
///     g   : [B, 16] bfloat16  -- silu(rms * A), input to gate_down GEMM
///     rms : [B]     float32   -- rsqrt(mean(x^2)+eps) per token
std::tuple<at::Tensor, at::Tensor> gigachat35_gated_norm_reduce(at::Tensor& pA, at::Tensor& pS, int64_t H, double eps);

}  // namespace sgl_kernel_ext
