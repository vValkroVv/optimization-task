"""Shared constants and shape conventions for the MoE package.

Shape suffix legend
(https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
    B = batch, L = sequence length, D = model dimension,
    F = hidden (FFN intermediate) dimension, E = num experts,
    e = num local experts (E / EP, used in token dispatcher for
        per-local-expert token counts after EP dispatch /_permute),
    K = top-k, T = num tokens (B*L flattened),
    N = routed tokens (T*K), R = routed tokens assigned to local experts
"""

from torchtitan.config.schema.models.common.moe.const import ScoreFunc

__all__ = ["ScoreFunc"]
