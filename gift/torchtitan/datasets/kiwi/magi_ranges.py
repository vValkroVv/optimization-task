"""Magi flex attention ranges — production wire format for kiwi attention.

Encoding (matches ``magi_attention``):

- ``attn_type_map == 0`` → full (bidirectional) rectangle
- ``attn_type_map == 1`` → causal triangle

Adjacency encoding used as input: ``0`` blocked, ``1`` causal, ``2`` full.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor


__all__ = [
    "MagiRanges",
    "build_magi_ranges",
    "make_sdpa_mask",
]


@dataclass(frozen=True, slots=True)
class MagiRanges:
    """Query/key segment ranges for Magi flex attention (and SDPA via dense)."""

    seq_len: int
    """Packed sequence length (token count); set by the dataloader / builder."""
    q_ranges: Tensor
    """``[N, 2]`` int32 — query ``[start, end)`` per segment."""
    k_ranges: Tensor
    """``[N, 2]`` int32 — key ``[start, end)`` per segment."""
    attn_type_map: Tensor
    """``[N]`` int32 — ``0`` full, ``1`` causal."""
    dist_attn_key: Any | None = None
    """Optional Magi ``DistAttnRuntimeKey`` from ``build_dist_attn_key`` (batch prep)."""

    def __post_init__(self) -> None:
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}.")
        if self.q_ranges.ndim != 2 or self.q_ranges.shape[-1] != 2:
            raise ValueError(
                f"q_ranges must be [N, 2], got shape {tuple(self.q_ranges.shape)}."
            )
        if self.k_ranges.shape != self.q_ranges.shape:
            raise ValueError(
                f"k_ranges shape {tuple(self.k_ranges.shape)} must match "
                f"q_ranges {tuple(self.q_ranges.shape)}."
            )
        if self.attn_type_map.shape != (self.q_ranges.shape[0],):
            raise ValueError(
                f"attn_type_map must be [N]={self.q_ranges.shape[0]}, "
                f"got shape {tuple(self.attn_type_map.shape)}."
            )
        for name, tensor in (
            ("q_ranges", self.q_ranges),
            ("k_ranges", self.k_ranges),
            ("attn_type_map", self.attn_type_map),
        ):
            if tensor.dtype != torch.int32:
                raise ValueError(f"{name} must be int32, got {tensor.dtype}.")

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> MagiRanges:
        return MagiRanges(
            seq_len=self.seq_len,
            q_ranges=self.q_ranges.to(device=device, non_blocking=non_blocking),
            k_ranges=self.k_ranges.to(device=device, non_blocking=non_blocking),
            attn_type_map=self.attn_type_map.to(
                device=device, non_blocking=non_blocking
            ),
            dist_attn_key=self.dist_attn_key,
        )


def _qk_segments(
    reordered_seq: list[int], mat: np.ndarray
) -> tuple[Tensor, Tensor, Tensor]:
    """QK segments description for SDPA, Flex or MagiAttention masks."""
    num_sub_seq = len(reordered_seq)
    if num_sub_seq < 1:
        raise ValueError("reordered_seq must be non-empty.")
    mat = np.asarray(mat, dtype=np.uint8)
    if mat.shape != (num_sub_seq, num_sub_seq):
        raise ValueError(
            f"adjacency shape {mat.shape} does not match num_segments={num_sub_seq}."
        )
    offsets = np.cumsum([0] + list(reordered_seq)).tolist()

    if num_sub_seq == 1:
        diag = int(mat[0, 0])
        if diag == 0:
            raise ValueError(
                "Single-segment adjacency diagonal must be causal or full."
            )
        attn_type = 1 if diag == 1 else 0
        ranges = torch.tensor([[0, offsets[1]]], dtype=torch.int32)
        return ranges, ranges.clone(), torch.tensor([attn_type], dtype=torch.int32)

    q_segments: list[list[int]] = []
    k_segments: list[list[int]] = []
    seg_types: list[int] = []

    start_seg = 0
    rect_start = num_sub_seq
    for i in range(num_sub_seq - 1):
        next_diag = int(mat[i + 1, i + 1])
        # Close a causal run on a document boundary, before padding (diag 0),
        # or at the first full (visual) segment. Padding must not sit inside
        # a causal triangle: grouped order is text..., pad, visual....
        should_close = (mat[i, i] == 1 and mat[i + 1, i] == 0) or next_diag in (
            0,
            2,
        )
        if should_close and int(mat[i, i]) == 1 and start_seg < offsets[i + 1]:
            q_segments.append([start_seg, offsets[i + 1]])
            k_segments.append([start_seg, offsets[i + 1]])
            seg_types.append(1)
            start_seg = offsets[i + 1]
        if next_diag == 0:
            start_seg = offsets[i + 2]
        if next_diag == 2:
            rect_start = i + 1
            break

    for i in range(num_sub_seq):
        start_seg = -1
        js = rect_start if i < rect_start else 0
        for j in range(js, num_sub_seq):
            if mat[i, j] == 2 and start_seg == -1:
                start_seg = offsets[j]
            if mat[i, j] == 0 and start_seg >= 0:
                q_segments.append([offsets[i], offsets[i + 1]])
                k_segments.append([start_seg, offsets[j]])
                seg_types.append(0)
                start_seg = -1
        if start_seg >= 0:
            q_segments.append([offsets[i], offsets[i + 1]])
            k_segments.append([start_seg, offsets[num_sub_seq]])
            seg_types.append(0)

    if not q_segments:
        raise ValueError("Adjacency produced no magi attention segments.")

    return (
        torch.tensor(q_segments, dtype=torch.int32),
        torch.tensor(k_segments, dtype=torch.int32),
        torch.tensor(seg_types, dtype=torch.int32),
    )


def build_magi_ranges(
    item_lengths: list[int],
    adjacency: Tensor | np.ndarray,
) -> MagiRanges:
    """Build Magi Q/K ranges from packed item lengths and the item adjacency."""
    if isinstance(adjacency, Tensor):
        mat = adjacency.detach().cpu().numpy()
    else:
        mat = np.asarray(adjacency)
    q_ranges, k_ranges, attn_type_map = _qk_segments(item_lengths, mat)
    return MagiRanges(
        seq_len=sum(item_lengths),
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=attn_type_map,
    )


def make_sdpa_mask(
    q_segments: Tensor, k_segments: Tensor, seg_types: Tensor, bsize: int = 64
) -> Tensor:
    """Make a SDPA mask from QK segments."""
    size_tensor = torch.maximum(q_segments.max(), k_segments.max())
    size_tensor = (
        ((size_tensor // bsize) + 1) * bsize if size_tensor % bsize > 0 else size_tensor
    )
    size: int = int(size_tensor.item())
    mask = torch.zeros((size, size), dtype=torch.bool)
    for i in range(q_segments.shape[0]):
        if seg_types[i] == 0:
            mask[
                q_segments[i, 0] : q_segments[i, 1], k_segments[i, 0] : k_segments[i, 1]
            ] = True
        else:
            row_size = int((q_segments[i, 1] - q_segments[i, 0]).item())
            col_size = int((k_segments[i, 1] - k_segments[i, 0]).item())
            mask[
                q_segments[i, 0] : q_segments[i, 1], k_segments[i, 0] : k_segments[i, 1]
            ] = torch.tril(torch.ones((row_size, col_size), dtype=torch.bool))
    return mask
