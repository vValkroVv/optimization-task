"""GatedNorm (weight applied as w) on CuTe DSL for Hopper (SM90).

    rho = rsqrt(mean(x^2) + eps)                        per row
    P   = rho * (x @ W_eff^T),  W_eff = W_up * w
    S   = silu(P)
    y   = lgw * rho * x * w * sigmoid(S @ W_down^T)

common.py    tile constants, device helpers, compile / plan / workspace caches
forward.py   prep, F1 (split-K stats), F2 (gate + output) kernels; gated_norm_fwd
backward.py  B1, B1f, B2, Wfin kernels; gated_norm_bwd
op.py        autograd.Function, gated_norm(), GatedNorm module, gated_norm_precompile()

Requirements: Hopper GPU and nvidia-cutlass-dsl 4.2.1. x bf16 (..., H) with H % 128 == 0,
gate rank 16.
"""

from .backward import gated_norm_bwd
from .common import RANK, get_sm_budget, set_sm_budget
from .forward import gated_norm_fwd
from .op import GnCfg, GatedNorm, gated_norm, gated_norm_precompile

__all__ = [
    "GatedNorm",
    "GnCfg",
    "RANK",
    "gated_norm",
    "gated_norm_bwd",
    "gated_norm_fwd",
    "gated_norm_precompile",
    "get_sm_budget",
    "set_sm_budget",
]
