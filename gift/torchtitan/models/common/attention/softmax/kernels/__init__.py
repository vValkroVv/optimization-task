"""Softmax inner-attention kernels.

These are the scaled-dot-product (softmax) kernels that the softmax-family outer
modules (GQA, MLA) plug into via their ``inner_attention`` config field:

- ``ScaledDotProductAttention`` -- ``F.scaled_dot_product_attention``.
- ``FlexAttention`` -- ``flex_attention`` with block masks / score mods.
- ``VarlenAttention`` -- FlashAttention ``varlen_attn`` over packed documents;
  consumes the family-agnostic varlen contract from ``attention/varlen_meta.py``.
- ``FlashAttention{2,3,4}`` / ``FlashVarlenAttention{2,3,4}`` -- Dao ``flash_attn``
  builds (optional dependency, imported lazily on build).

A kernel is selected by naming its config: ``inner_attention=FlexAttention.Config(
block_size=(256, 128))``. Each config declares what it needs from the decoder
(``mask_kind`` / ``supported_mask_types``, see ``base.py``), so kernel knobs such
as ``window_size`` stay reachable and masks have a single source of truth.
"""

from .base import InnerAttentionConfig
from .flash_fa2 import FlashAttention2, FlashVarlenAttention2
from .flash_fa3 import FlashAttention3, FlashVarlenAttention3
from .flash_fa4 import FlashAttention4, FlashVarlenAttention4
from .flex import FlexAttention
from .sdpa import ScaledDotProductAttention
from .varlen import VarlenAttention

__all__ = [
    "FlexAttention",
    "InnerAttentionConfig",
    "ScaledDotProductAttention",
    "VarlenAttention",
    "FlashAttention2",
    "FlashAttention3",
    "FlashAttention4",
    "FlashVarlenAttention2",
    "FlashVarlenAttention3",
    "FlashVarlenAttention4",
]
