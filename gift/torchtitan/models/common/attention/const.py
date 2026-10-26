"""Attention mask-mod primitives and shared types (family-agnostic).

The inner softmax kernels live in ``softmax/kernels/``; the varlen contract in ``varlen_meta.py``. Attention
families are selected by naming their config (``MLA.Config``, ``GQAttention.Config``,
...), not by a string, so there is no family enum here.
"""

import functools

import torch
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    BlockMask,
    create_block_mask,
)

from torchtitan.config.schema.models.common.attention.const import (
    AttentionMaskKind,
    MaskType,
)
from torchtitan.models.common.attention.varlen_meta import VarlenMetadata

__all__ = [
    "AttentionMaskKind",
    "AttentionMasksType",
    "MaskType",
    "create_attention_mask",
    "get_causal_mask_mod",
    "get_document_mask_mod",
    "get_fixed_block_mask_mod",
    "get_sliding_window_mask_mod",
]


AttentionMasksType = dict[str, BlockMask] | BlockMask | VarlenMetadata


def get_causal_mask_mod() -> _mask_mod_signature:
    """Causal mask modifier for flex attention."""

    def _causal_mask(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        return q_idx >= kv_idx

    return _causal_mask


def get_document_mask_mod(positions: torch.Tensor) -> _mask_mod_signature:
    """Creates a document mask that prevents attention across document boundaries.

    Document boundaries are detected where ``positions`` resets to 0, which
    marks the start of a new packed document.

    Args:
        positions: Per-token position tensor with shape ``[b, s]``. Positions
            reset to 0 at each document start.

    Returns:
        A mask modifier function that implements document-level masking.
    """
    doc_ids = torch.cumsum((positions == 0).int(), dim=1) - 1

    def document_mask(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        return doc_ids[b, q_idx] == doc_ids[b, kv_idx]

    return document_mask


def get_fixed_block_mask_mod(fixed_block_size: int) -> _mask_mod_signature:
    """Mask modifier that only allows attention within fixed-size blocks."""

    # Credit to @drisspg.
    def blocked_mask_mod(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        q_block = q_idx // fixed_block_size
        kv_block = kv_idx // fixed_block_size
        return q_block == kv_block

    blocked_mask_mod.__name__ = f"blocked_mask_mod_fixed_block_size_{fixed_block_size}"

    return blocked_mask_mod


def get_sliding_window_mask_mod(window_size: int) -> _mask_mod_signature:
    """Causal sliding-window mask: attend to self and up to ``window_size - 1``
    previous tokens (``window_size >= 1``)."""

    if window_size < 1:
        raise ValueError(
            f"window_size must be >= 1 for sliding window attention mask, got {window_size}"
        )

    def sliding_window_mod(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        # Window mask: can only attend within the window
        # q_idx - kv_idx < window_size ensures we look at most window_size-1 tokens back
        return (kv_idx <= q_idx) & (q_idx - kv_idx < window_size)

    sliding_window_mod.__name__ = f"sliding_window_mod_window_size_{window_size}"

    return sliding_window_mod


@functools.cache
def _compiled_create_block_mask():
    return torch.compile(create_block_mask)


def create_attention_mask(*args, **kwargs):
    """Create an attention mask using compiled create_block_mask."""
    return _compiled_create_block_mask()(*args, **kwargs)
