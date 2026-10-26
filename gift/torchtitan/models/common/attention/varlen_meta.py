"""Family-agnostic variable-length (varlen) attention core.

Varlen packs several documents into one sequence and resets attention at document
boundaries. What every attention family shares is not the kernel but the *contract*
— the :class:`VarlenMetadata` ``cu_seqlens`` descriptor and the packing helpers
below; each family supplies its own kernel that consumes it.
"""

from typing import NamedTuple

import torch
from torch.distributed.tensor import DTensor

__all__ = [
    "VarlenMetadata",
    "create_varlen_metadata_for_document",
    "flatten_varlen_batch",
    "pack_varlen_qkv",
    "unflatten_varlen_batch",
    "unpack_varlen_out",
]


def flatten_varlen_batch(x: torch.Tensor) -> torch.Tensor:
    """Collapse the batch into a single packed sequence: ``[B, S, …] -> [1, B*S, …]``.

    ``fla``'s ``cu_seqlens`` kernels require batch size 1 (one flat token stream).
    Dense document-packing (no padding) makes this a free reshape; the row-major
    flatten matches the ``offset = b*seq_len`` convention in
    :func:`create_varlen_metadata_for_document`, so tokens align with ``cu_seq_q``.
    """
    return x.reshape(1, x.shape[0] * x.shape[1], *x.shape[2:])


def unflatten_varlen_batch(
    x: torch.Tensor, batch_size: int, seq_len: int
) -> torch.Tensor:
    """Inverse of :func:`flatten_varlen_batch`: ``[1, B*S, …] -> [B, S, …]``."""
    return x.reshape(batch_size, seq_len, *x.shape[2:])


class VarlenMetadata(NamedTuple):
    """Cumulative sequence positions (``cu_seqlens``) for queries and keys/values,
    plus the per-side maxima."""

    cu_seq_q: torch.Tensor
    cu_seq_k: torch.Tensor
    max_q: int
    max_k: int


def pack_varlen_qkv(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten the batch and sequence axes for varlen kernels.

    Converts each of ``q`` / ``k`` / ``v`` from the padded
    ``(bs, seq, heads, dim)`` layout to the flat ``(bs * seq, heads, dim)``
    layout that varlen kernels index with ``cu_seqlens``. The head count may
    differ between q and k/v (GQA); only the trailing ``dim`` is assumed shared.
    """

    def _pack(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0] * x.shape[1], -1, x.shape[-1])

    return _pack(q), _pack(k), _pack(v)


def unpack_varlen_out(out: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
    """Inverse of :func:`pack_varlen_qkv` for the attention output.

    Reshapes ``(bs * seq, heads, dim)`` back to ``(bs, seq, heads, dim)``.
    """
    return out.view(batch_size, seq_len, -1, out.shape[-1])


def create_varlen_metadata_for_document(
    positions: torch.Tensor,
) -> VarlenMetadata:
    """Creates cumulative sequence length indices needed for variable length attention.

    Document boundaries are detected where ``positions`` resets to 0 (same
    convention as :func:`get_document_mask_mod`).

    Args:
        positions: Per-token position tensor with shape ``[b, s]``. Positions
            reset to 0 at each document start. A DTensor is read as its local shard.

    Returns:
        VarlenMetadata containing cumulative sequence length indices for q, k, and
        max_seq_len. The index tensors are plain tensors, as the kernels expect.
    """
    if isinstance(positions, DTensor):
        positions = positions.to_local()
    batch_size, seq_len = positions.shape
    device = positions.device
    cu_seqlens_list, all_seq_lengths = [], []
    offset = 0

    for b in range(batch_size):
        doc_starts = (positions[b] == 0).nonzero(as_tuple=True)[0].to(torch.int32)
        sample_cu_seqlens = torch.cat(
            [doc_starts, torch.tensor([seq_len], dtype=torch.int32, device=device)]
        )

        seq_lengths = torch.diff(sample_cu_seqlens)
        all_seq_lengths.append(seq_lengths)

        cu_seqlens_adjusted = sample_cu_seqlens[:-1] + offset
        cu_seqlens_list.append(cu_seqlens_adjusted)

        offset += seq_len

    packed_cu_seqlens = torch.cat(
        cu_seqlens_list + [torch.tensor([offset], dtype=torch.int32, device=device)]
    )

    max_seqlen: int = 0
    if len(all_seq_lengths) > 0:
        all_seq_lengths = torch.cat(all_seq_lengths)
        # device to host sync but only done once per model forward
        # pyrefly: ignore[bad-assignment]
        max_seqlen = all_seq_lengths.max().item()

    return VarlenMetadata(
        cu_seq_q=packed_cu_seqlens,
        cu_seq_k=packed_cu_seqlens,
        max_q=max_seqlen,
        max_k=max_seqlen,
    )
