import torch

try:
    from . import _C  # ahead-of-time build
except ImportError:
    from ._jit_fallback import _C  # fallback

# expose torch ops
from torch.ops import sgl_kernel_ext as ops


@torch.library.register_fake("sgl_kernel_ext::giga_moe_fused_gate")
def _giga_moe_fused_gate_fake(
    input,
    bias,
    topk,
    num_fused_shared,
    renormalize,
    routed_scaling_factor,
    apply_routed_scaling_factor_on_output,
):
    # Mirrors the CUDA impl's allocation: fp32 weights and int32 indices,
    # both [num_rows, topk]. Required so the op is traceable by
    # torch.compile / the tc_piecewise prefill cuda-graph backend.
    num_rows = input.shape[0]
    output = input.new_empty((num_rows, topk), dtype=torch.float32)
    indices = input.new_empty((num_rows, topk), dtype=torch.int32)
    return output, indices


@torch.library.register_fake("sgl_kernel_ext::gigachat35_gated_norm_reduce")
def _gigachat35_gated_norm_reduce_fake(pA, pS, H, eps):
    B = pA.shape[1]
    g = pA.new_empty((B, 16), dtype=torch.bfloat16)
    rms = pA.new_empty((B,), dtype=torch.float32)
    return g, rms


__all__ = ["ops"]