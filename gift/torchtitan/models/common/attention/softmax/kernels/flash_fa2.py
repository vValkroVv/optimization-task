"""FlashAttention-2 kernels backed by the ``flash_attn`` package."""

from .flash_base import FlashAttention, FlashVarlenAttention
from typing import TypeAlias

from torchtitan.config.schema.models.common.attention.softmax.kernels.flash_fa2 import (
    FlashAttention2Config,
    FlashVarlenAttention2Config,
)

__all__ = ["FlashAttention2", "FlashVarlenAttention2"]


class FlashAttention2(FlashAttention):
    _PACKAGE = "flash_attn"
    _LABEL = "fa2"

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = FlashAttention2Config


class FlashVarlenAttention2(FlashVarlenAttention):
    _PACKAGE = "flash_attn"
    _LABEL = "fa2_varlen"

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = FlashVarlenAttention2Config
