from __future__ import annotations

import math

import torch

from metric_sampler.capture.kernels import activation_ext
from metric_sampler.capture.recorder import RankRecorder, ResolvedRows

SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
# The rank of a [rows, columns] tensor.
MATRIX_DIMENSIONS = 2
# A weight's row: its absolute max, its L2 norm and a count of one.
_WEIGHT_ROW_WIDTH = 3


def _is_cuda_rows(value: torch.Tensor) -> bool:
    """Whether the extension reduces the last-axis rows of `value`: a non-scalar, non-empty CUDA tensor of a
    supported dtype."""
    return value.ndim > 0 and value.is_cuda and value.dtype in SUPPORTED_DTYPES and value.numel() > 0


def amax_and_row_norm_sum(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """The float32 absolute max of `value`, the float64 sum of the float32 L2 norms of its last-axis
    rows, and the row count; NaN propagates to both sums. A scalar is one row."""
    detached = value.detach()
    if detached.ndim == 0:
        magnitude = detached.float().abs()
        return magnitude, magnitude.double(), 1
    row_matrix = detached.reshape(-1, detached.shape[-1])
    if row_matrix.is_cuda and row_matrix.dtype in SUPPORTED_DTYPES and row_matrix.numel():  # pragma: no cover
        extension = activation_ext.prebuilt()
        if extension:
            stats = extension.row_norm_stats(row_matrix)
            return stats[0].float(), stats[1], row_matrix.shape[0]
    # Half and single precision reduce in float32 straight from the source, without a float32 copy.
    if detached.dtype not in SUPPORTED_DTYPES:
        detached = detached.float()
    row_norms = torch.linalg.vector_norm(detached, dim=-1, dtype=torch.float32)
    return torch.linalg.vector_norm(detached, ord=math.inf).float(), row_norms.double().sum(), row_matrix.shape[0]


def accumulate_row_norms(
    value: torch.Tensor, accumulator: torch.Tensor, slot: int, keep_stats: bool = True
) -> torch.Tensor | None:
    """Takes `value`'s `amax_and_row_norm_sum` statistics into float64 `accumulator[slot]`: the
    NaN-propagating maximum of the absolute max and the sum of the row-norm sums. Returns them as
    two float64 values, which `accumulate_row_norm_stats` takes into further slots, or None when
    `keep_stats` is false."""
    if _is_cuda_rows(value):  # pragma: no cover
        extension = activation_ext.prebuilt()
        if extension:
            return extension.row_norm_accumulate(value, accumulator, slot, keep_stats)
    absolute_max, norm_sum, _ = amax_and_row_norm_sum(value)
    stats = torch.stack((absolute_max.double(), norm_sum))
    accumulate_row_norm_stats(stats, accumulator, slot)
    return stats if keep_stats else None


def accumulate_row_norm_stats(stats: torch.Tensor, accumulator: torch.Tensor, slot: int) -> None:
    """Takes statistics `accumulate_row_norms` returned into another slot."""
    if stats.is_cuda:  # pragma: no cover
        extension = activation_ext.prebuilt()
        if extension:
            extension.accumulate_row_norm_stats(stats, accumulator, slot)
            return
    accumulator[slot, 0] = torch.maximum(accumulator[slot, 0], stats[0])
    accumulator[slot, 1] = stats[1] + accumulator[slot, 1]


def accumulate_row_norms_batch(
    values: list[torch.Tensor], accumulator: torch.Tensor, slots: list[list[int]]
) -> None:
    """`accumulate_row_norms` of each of `values` into each of its `slots` of float64 `accumulator`: the
    tensor is reduced once, into the first slot, and the others take its statistics, as
    `accumulate_row_norm_stats` does. No slot is listed twice. On CUDA, one launch covers up to 32 tensors,
    each reduced as it is alone."""
    if all(_is_cuda_rows(value) for value in values):  # pragma: no cover
        extension = activation_ext.prebuilt()
        if extension:
            extension.row_norm_accumulate_batch(
                values, accumulator, [len(value_slots) for value_slots in slots],
                [slot for value_slots in slots for slot in value_slots],
            )
            return
    for value, value_slots in zip(values, slots):
        stats = accumulate_row_norms(value, accumulator, value_slots[0], keep_stats=len(value_slots) > 1)
        if stats is not None:
            for slot in value_slots[1:]:
                accumulate_row_norm_stats(stats, accumulator, slot)


def weight_norm_rows(weights: list[torch.Tensor]) -> torch.Tensor:
    """float32 [absolute max, L2 norm, 1] of each non-empty weight, each norm in the weight's dtype as torch
    computes it; on CUDA one extension call covers every weight."""
    if all(weight.is_cuda and weight.dtype in SUPPORTED_DTYPES and weight.is_contiguous()
           for weight in weights):  # pragma: no cover
        extension = activation_ext.prebuilt()
        if extension:
            return extension.weight_norms(weights)
    absolute_max = torch.stack(torch._foreach_norm(weights, ord=math.inf)).float()
    l2_norm = torch.stack(torch._foreach_norm(weights, ord=2)).float()
    return torch.stack((absolute_max, l2_norm, torch.ones_like(absolute_max)), dim=1)


def record_weight_norm_rows(
    recorder: RankRecorder, rows: ResolvedRows, weights: list[torch.Tensor], experts: list[int],
    launch_numels: list[int],
) -> None:
    """Records `weight_norm_rows` of the weights into float32 `rows` of width 3; on CUDA the extension
    writes them into the recorder's buffer, without a tensor of them or an indexed copy. `weights[i]` is
    one weight, or with `experts[i]` > 0 a stack of that many experts' weights, each a row of its view as
    [experts[i], -1]. `launch_numels[i]` is the element count of the largest weight in the batch that weight
    i would be reduced in alone, which sets the order of its sums on CUDA: weights of several batches can be
    recorded together, each reduced as in its own batch."""
    if rows.width == _WEIGHT_ROW_WIDTH and weights[0].is_cuda:  # pragma: no cover
        extension = activation_ext.prebuilt()
        if extension:
            recorder.mark_rows_recorded(rows)
            if extension.weight_norms_into(
                    weights, experts, recorder.float_buffer(), rows.offsets.tolist(), launch_numels):
                return
    unstacked = [row for weight, expert_count in zip(weights, experts)
                 for row in (weight.reshape(expert_count, -1).unbind(0) if expert_count else (weight,))]
    recorder.record_rows(rows, weight_norm_rows(unstacked))
