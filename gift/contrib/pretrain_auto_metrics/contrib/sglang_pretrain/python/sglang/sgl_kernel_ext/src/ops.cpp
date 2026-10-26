#include "giga_moe_fused_gate.h"
#include "gigachat35_gated_norm_reduce.h"

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch/library.h>

// Minimal sgl_kernel_ext: only the GigaChat 3.5 fused MoE gate is shipped here.
// (The upstream ultra-3-5 module also carries CUDA logit-processor kernels;
// those are intentionally not ported in this MR.)
TORCH_LIBRARY(sgl_kernel_ext, m) {
  m.def("giga_moe_fused_gate("
        "Tensor input, Tensor bias, int topk, int num_fused_shared, bool renormalize, "
        "float routed_scaling_factor, bool apply_routed_scaling_factor_on_output) -> (Tensor, Tensor)");
  m.impl("giga_moe_fused_gate", torch::kCUDA, &sgl_kernel_ext::giga_moe_fused_gate);
  m.def("gigachat35_gated_norm_reduce(Tensor pA, Tensor pS, int H, float eps) -> (Tensor, Tensor)");
  m.impl("gigachat35_gated_norm_reduce", torch::kCUDA, &sgl_kernel_ext::gigachat35_gated_norm_reduce);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "sgl_kernel_ext JIT (CUDA ops + torch.library registration)";
}
