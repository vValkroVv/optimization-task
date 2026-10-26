"""FlashAttention-3 kernels backed by the Hopper ``flash_attn_interface``."""

from .flash_base import FlashAttention, FlashVarlenAttention
from typing import TypeAlias

from torchtitan.config.schema.models.common.attention.softmax.kernels.flash_fa3 import (
    FlashAttention3Config,
    FlashVarlenAttention3Config,
)

__all__ = ["FlashAttention3", "FlashVarlenAttention3"]


class FlashAttention3(FlashAttention):
    _PACKAGE = "flash_attn_interface"
    _LABEL = "fa3"

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = FlashAttention3Config


class FlashVarlenAttention3(FlashVarlenAttention):
    _PACKAGE = "flash_attn_interface"
    _LABEL = "fa3_varlen"

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = FlashVarlenAttention3Config
