"""``ScaledDotProductAttention`` -- inner kernel over ``F.scaled_dot_product_attention``."""

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from typing import TypeAlias

from torchtitan.config.schema.models.common.attention.softmax.kernels.sdpa import (
    ScaledDotProductAttentionConfig,
)
from torchtitan.protocols.module import Module


__all__ = ["ScaledDotProductAttention"]

_BACKENDS = [
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.MATH,
]


class ScaledDotProductAttention(Module):
    """Inner attention using ``F.scaled_dot_product_attention`` with CP support.

    Each backend handles its own layout transpose: ``forward()`` transposes from
    ``(bs, seq, heads, dim)`` to ``(bs, heads, seq, dim)`` before calling
    ``scaled_dot_product_attention``, and transposes back before returning.

    Note:
        The forward function must have q, k, v as the first three arguments to be
        compatible with _ContextParallel.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = ScaledDotProductAttentionConfig

    def __init__(self, config: Config) -> None:
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
        enable_gqa: bool = False,
        is_causal: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        # Transpose to (bs, heads, seq, dim) for SDPA
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        with sdpa_kernel(_BACKENDS, set_priority=True):
            out = F.scaled_dot_product_attention(
                q, k, v, scale=scale, is_causal=is_causal, enable_gqa=enable_gqa
            )
        # Transpose back to (bs, seq, heads, dim)
        return out.transpose(1, 2)
