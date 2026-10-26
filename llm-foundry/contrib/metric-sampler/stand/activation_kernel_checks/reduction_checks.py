from __future__ import annotations

import math

import torch

from metric_sampler.capture.stats.helpers.quantiles import row_entropy, row_margins
from metric_sampler.capture.stats.helpers.reductions import add_segment_sums_of_squares, column_sums, entropy_sum
from metric_sampler.contract.metrics import WORDS_PER_FLOAT64
from stand.activation_kernel_checks.norm_checks import (
    check_accumulation,
    check_batched_accumulation,
    check_layout_row_norms,
    check_nan_propagation,
    check_unkept_accumulation,
)
from stand.activation_kernel_checks.statistics_oracle import equal_with_nans
from stand.activation_kernel_checks.weight_norm_checks import check_weight_norms

FLOAT64_RELATIVE_TOLERANCE = 1e-13
ENTROPY_EPSILON = 1e-12
ROW_REDUCTION_SHAPES = (
    (1, 1, torch.float32), (5, 7, torch.bfloat16), (49152, 896, torch.bfloat16), (1000, 2304, torch.bfloat16),
    (777, 256, torch.float16), (3000, 96, torch.float32),
)
STRIDED_EXTRA_COLUMNS = 3
SEGMENT_ROWS = 3000
SEGMENT_WIDTH = 96
SEGMENT_COUNTS = (256, 257)
ODD_WORD_OFFSET = 1
ROUTER_ROWS = 49152
ROUTER_EXPERTS = 256
ROUTER_TOP_K = 8
MARGIN_ROWS = 3000
UNALIGNED_COLUMN_OFFSET = 100
ALIGNED_COLUMN_OFFSET = 128


def check_row_reductions(device: torch.device) -> list[str]:
    """The extension's per-row reductions against torch: amax and router row margins exactly, the
    sum of fp32 row norms within fp32 rounding, fp64 row sums of squares, column sums and router
    row entropies within fp64 rounding, NaN propagating."""
    problems = []
    for rows, width, dtype in ROW_REDUCTION_SHAPES:
        for name, activations in _row_layouts(rows, width, dtype, device):
            label = f"{name} {rows}x{width} {dtype}"
            problems.extend(check_layout_row_norms(label, activations, rows, device))
            problems.extend(_check_layout_segment_sums(label, activations, rows, device))
    problems.extend(_check_segment_counts(device))
    problems.extend(_check_router_scores(device))
    problems.extend(_check_row_margins(device))
    problems.extend(check_nan_propagation(device))
    problems.extend(check_accumulation(device))
    problems.extend(check_unkept_accumulation(device))
    problems.extend(check_batched_accumulation(device))
    problems.extend(check_weight_norms(device))
    return problems


def _row_layouts(
    rows: int, width: int, dtype: torch.dtype, device: torch.device
) -> tuple[tuple[str, torch.Tensor], ...]:
    return (("contiguous", torch.randn(rows, width, device=device).to(dtype)),
            ("strided",
             torch.randn(rows, width + STRIDED_EXTRA_COLUMNS, device=device).to(dtype)[:, 1:width + 1]),
            ("transposed", torch.randn(width, rows, device=device).to(dtype).t()))


def _strided_columns(values: torch.Tensor, offset: int) -> torch.Tensor:
    return torch.cat((values, values), dim=1)[:, offset:offset + ROUTER_EXPERTS]


def _segment_sums_of_squares(activations: torch.Tensor, counts: list[int]) -> torch.Tensor:
    """The float64 sums of squares add_segment_sums_of_squares writes, from an odd word offset, and adds to them."""
    words = torch.zeros(ODD_WORD_OFFSET + WORDS_PER_FLOAT64 * len(counts), dtype=torch.int32, device=activations.device)
    add_segment_sums_of_squares(activations, counts, words, ODD_WORD_OFFSET, True)
    written = words[ODD_WORD_OFFSET:].clone().view(torch.float64)
    add_segment_sums_of_squares(activations, counts, words, ODD_WORD_OFFSET, False)
    if not torch.equal(words[ODD_WORD_OFFSET:].clone().view(torch.float64), written + written):
        raise AssertionError("segment sums of squares were not added to the words they were written to")
    return written


def _check_layout_segment_sums(label: str, activations: torch.Tensor, rows: int, device: torch.device) -> list[str]:
    problems = []
    counts = [rows // 3, 0, rows - rows // 3]
    sum_of_squares = _segment_sums_of_squares(activations, counts)
    expected = torch.segment_reduce(activations.double().square().sum(dim=1), "sum",
                                    lengths=torch.tensor(counts, device=device))
    if not torch.allclose(sum_of_squares, expected, rtol=FLOAT64_RELATIVE_TOLERANCE, atol=0):
        problems.append(f"segment sums of squares {label} differ beyond fp64 rounding")
    return problems


def _check_segment_counts(device: torch.device) -> list[str]:
    problems = []
    activations = torch.randn(SEGMENT_ROWS, SEGMENT_WIDTH, device=device).to(torch.bfloat16)
    for segments in SEGMENT_COUNTS:
        counts = [SEGMENT_ROWS // segments] * (segments - 1)
        counts.append(SEGMENT_ROWS - sum(counts))
        expected = torch.segment_reduce(
            activations.double().square().sum(dim=1), "sum", lengths=torch.tensor(counts, device=device)
        )
        if not torch.allclose(
            _segment_sums_of_squares(activations, counts), expected, rtol=FLOAT64_RELATIVE_TOLERANCE, atol=0
        ):
            problems.append(f"segment sums of squares over {segments} segments differ beyond fp64 rounding")
    return problems


def _check_router_scores(device: torch.device) -> list[str]:
    problems = []
    scores = torch.softmax(torch.randn(ROUTER_ROWS, ROUTER_EXPERTS, device=device), dim=-1)
    biased = scores.double()
    expected = -(biased * (biased + ENTROPY_EPSILON).log()).sum()
    if abs(entropy_sum(scores).item() - expected.item()) > FLOAT64_RELATIVE_TOLERANCE * abs(expected.item()):
        problems.append("router entropy sum differs beyond fp64 rounding")
    for name, rows in (("contiguous", scores), ("strided", _strided_columns(scores, UNALIGNED_COLUMN_OFFSET)),
                       ("one row", scores[:1]), ("narrow", scores[:5000, :3])):
        if not torch.allclose(
            column_sums(rows), rows.sum(dim=0, dtype=torch.float64), rtol=FLOAT64_RELATIVE_TOLERANCE, atol=0
        ):
            problems.append(f"router per-expert score sums {name} differ beyond fp64 rounding")
    scores[1] = 0.0
    scores[2, 0] = float("nan")
    scores[3] = 0.0
    scores[3, 7] = 1.0
    for name, rows in (("contiguous", scores), ("strided", _strided_columns(scores, ALIGNED_COLUMN_OFFSET)),
                       ("narrow", scores[:5, :3])):
        probabilities = rows.double() / rows.double().sum(dim=-1, keepdim=True)
        expected = -torch.special.xlogy(probabilities, probabilities).sum(dim=-1) / math.log(rows.shape[-1])
        if not torch.allclose(row_entropy(rows), expected, rtol=FLOAT64_RELATIVE_TOLERANCE, atol=0, equal_nan=True):
            problems.append(f"router row entropy {name} differs beyond fp64 rounding")
    return problems


def _check_row_margins(device: torch.device) -> list[str]:
    problems = []
    logits = torch.randn(MARGIN_ROWS, ROUTER_EXPERTS, device=device)
    logits[0, 5] = float("nan")
    logits[1, 7] = float("inf")
    logits[2, 9] = -float("inf")
    boundary = logits.topk(ROUTER_TOP_K + 1, dim=-1).values[:, ROUTER_TOP_K - 1]
    odd_nan = torch.where(torch.arange(MARGIN_ROWS, device=device) % 2 == 1, float("nan"), boundary)
    for name, rows, row_boundary in (("contiguous", logits, boundary), ("NaN boundaries", logits, odd_nan),
                                     ("strided", _strided_columns(logits, UNALIGNED_COLUMN_OFFSET), boundary)):
        expected = row_boundary[:, None] - torch.where(rows.isnan(), math.inf, rows)
        got = row_margins(rows, row_boundary)
        if not (torch.equal(got.isnan(), expected.isnan()) and equal_with_nans(got, expected)):
            problems.append(f"router row margins {name} differ from torch")
    return problems
