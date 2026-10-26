from __future__ import annotations

import math

import torch

from metric_sampler.capture.kernels import activation_ext
from metric_sampler.capture.stats.helpers.quantiles import order_statistics, row_margins, row_top_values

# The length of the result of router_margin_stats.
ROUTER_MARGIN_WIDTH = 8
# The quantiles of the unselected experts' margins: q25 and q10.
_UNSELECTED_MARGIN_QUANTILES = (0.25, 0.10)


def router_margin_stats(scoring_function: str, top_k: int, near_count: int, logits: torch.Tensor) -> torch.Tensor:
    """[median gap, median near, q25 and q10 unselected margin] of the scores, then of the logits."""
    values = logits.float()
    scores = torch.softmax(values, dim=-1) if scoring_function == "softmax" else torch.sigmoid(values)
    rows, experts = values.shape
    stats: list[torch.Tensor] = []
    selected_count = rows * top_k
    positions = [(rows * (experts - top_k) - 1) * fraction for fraction in _UNSELECTED_MARGIN_QUANTILES]
    ranks = [selected_count + bound(position) for position in positions for bound in (math.floor, math.ceil)]
    for population in (scores, values):
        top = row_top_values(population, min(top_k + near_count, experts))
        boundary = top[:, top_k - 1:top_k]
        gap = (boundary - top[:, top_k:top_k + 1]).reshape(-1)
        near = (boundary - top[:, top_k:]).reshape(-1)
        # A selected expert's margin is at most 0 and an unselected one's at least 0, so
        # past the first selected_count ranks of all margins come the unselected ones in order.
        # Top-k ranks NaN above +inf, so a NaN score counts as +inf.
        margins = order_statistics(row_margins(population, top[:, top_k - 1]).reshape(-1), ranks)
        quantiles = [torch.lerp(margins[2 * index], margins[2 * index + 1], position - math.floor(position))
                     for index, position in enumerate(positions)]
        stats.extend((_median(gap), _median(near), *quantiles))
    return torch.stack(stats)


def router_margin_graph_stats(
    scoring_function: str, top_k: int, near_count: int, logits: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """`router_margin_stats` as the graph computes it: a sigmoid router's from the extension's few kernels, the
    same values."""
    extension = activation_ext.prebuilt()
    if extension and scoring_function == "sigmoid":
        stats = torch.empty(ROUTER_MARGIN_WIDTH, dtype=torch.float32, device=logits.device)
        if extension.router_margin_into(logits, top_k, near_count, stats, offset=0):
            return stats
    return router_margin_stats(scoring_function, top_k, near_count, logits)


def _median(values: torch.Tensor) -> torch.Tensor:
    # torch.median: the lower middle value, or NaN when there is one (NaN sorts last).
    middle, last = order_statistics(values, [(values.numel() - 1) // 2, values.numel() - 1])
    return torch.where(last.isnan(), last, middle)
