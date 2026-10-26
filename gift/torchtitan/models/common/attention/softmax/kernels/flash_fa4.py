"""FlashAttention-4 (Blackwell/Hopper) kernels over the CuTe-DSL ``flash_attn.cute``.

FA4 ships as a separate ``flash-attn-4`` package (pure-Python on
``nvidia-cutlass-dsl``). Its ``flash_attn.cute`` interface differs from FA2/FA3
in two spots, handled via the base's knobs: it spells the LSE flag ``return_lse``
(not ``return_attn_probs``) and an unbounded window side as ``None`` (not ``-1``).
"""

from .flash_base import FlashAttention, FlashVarlenAttention
from typing import TypeAlias

from torchtitan.config.schema.models.common.attention.softmax.kernels.flash_fa4 import (
    FlashAttention4Config,
    FlashVarlenAttention4Config,
)

__all__ = ["FlashAttention4", "FlashVarlenAttention4"]


class FlashAttention4(FlashAttention):
    _PACKAGE = "flash_attn.cute"
    _LABEL = "fa4"
    _LSE_KWARG = "return_lse"
    _WINDOW_UNLIMITED_IS_NONE = True

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = FlashAttention4Config


class FlashVarlenAttention4(FlashVarlenAttention):
    _PACKAGE = "flash_attn.cute"
    _LABEL = "fa4_varlen"
    _WINDOW_UNLIMITED_IS_NONE = True

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = FlashVarlenAttention4Config
