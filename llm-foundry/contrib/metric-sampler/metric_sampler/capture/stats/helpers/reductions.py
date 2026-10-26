from __future__ import annotations

import torch

from metric_sampler.capture.kernels import activation_ext
from metric_sampler.capture.stats.helpers.norms import MATRIX_DIMENSIONS, SUPPORTED_DTYPES
from metric_sampler.contract.metrics import WORDS_PER_FLOAT64

_ENTROPY_EPSILON = 1e-12
# The most segments one extension launch takes: their row counts are kernel arguments.
_MAX_SEGMENTS = 256


def float64_scalar(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    # A scalar on another device (a disabled router z-loss is a host torch.tensor(0)) is written by
    # a fill instead of a pageable host-to-device copy, which would wait for the queued forward.
    # Called under no_grad: a scalar already float64 and 0-dim takes no op at all.
    if value.device != device:
        return torch.full((), value.item(), device=device, dtype=torch.float64)
    scalar = value.to(dtype=torch.float64)
    return scalar.reshape(()) if scalar.ndim else scalar


def entropy_sum(scores: torch.Tensor) -> torch.Tensor:
    # Each term in float64, without float64 copies of the whole score matrix on CUDA.
    if scores.is_cuda and scores.dtype == torch.float32 and scores.numel():  # pragma: no cover
        extension = activation_ext.prebuilt()
        if extension:
            return extension.entropy_sum(scores.contiguous())
    doubled = scores.double()
    return -(doubled * (doubled + _ENTROPY_EPSILON).log()).sum()


def column_sums(scores: torch.Tensor) -> torch.Tensor:
    # In float64, without a float64 copy of the whole score matrix on CUDA.
    if (scores.is_cuda and scores.dtype == torch.float32 and scores.ndim == MATRIX_DIMENSIONS and scores.numel()
            and scores.stride(-1) == 1):  # pragma: no cover
        extension = activation_ext.prebuilt()
        if extension:
            return extension.column_sums(scores)
    return scores.sum(dim=0, dtype=torch.float64)


def add_segment_sums_of_squares(
    rows: torch.Tensor, counts: list[int], words: torch.Tensor, offset: int, first_call: bool
) -> None:
    """Adds the float64 sum of squares of each consecutive segment of `counts` rows to the float64 values held
    as two int32 words each in `words` from `offset`, or writes them there on the `first_call`."""
    if rows.is_cuda and rows.dtype in SUPPORTED_DTYPES and len(counts) <= _MAX_SEGMENTS:  # pragma: no cover
        extension = activation_ext.prebuilt()
        if extension:
            # The counts are kernel arguments: no host-to-device copy waits for the queued forward.
            extension.segment_sumsq_into(rows, counts, words, offset, not first_call)
            return
    lengths = torch.tensor(counts, dtype=torch.int64, device=rows.device)
    sums_of_squares = torch.segment_reduce(
        rows.detach().double().square().sum(dim=1), "sum", lengths=lengths, unsafe=True
    )
    target = words[offset:offset + WORDS_PER_FLOAT64 * len(counts)]
    if not first_call:
        # A copy: the words of a slot need not be aligned to float64.
        sums_of_squares = target.clone().view(torch.float64) + sums_of_squares
    target.copy_(sums_of_squares.view(torch.int32))
