"""Declarative base for inner softmax-attention kernel configs.

Each kernel config states what it needs from the decoder — which mask
representation it consumes (``mask_kind``) and which mask types it accepts —
so the outer attention families never enumerate kernel classes.
"""

from torchtitan.config.schema.models.common.attention.softmax.kernels.base import (
    InnerAttentionConfig,
)

__all__ = ["InnerAttentionConfig"]
