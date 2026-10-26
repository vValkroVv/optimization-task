from __future__ import annotations

import math

import pytest
import torch

from metric_sampler.capture.stats.router_margin import ROUTER_MARGIN_WIDTH, router_margin_stats

pytestmark = pytest.mark.unit


def _sorted_margin_stats(scoring_function: str, top_k: int, near_count: int, logits: torch.Tensor) -> torch.Tensor:
    values = logits.float()
    scores = torch.softmax(values, dim=-1) if scoring_function == "softmax" else torch.sigmoid(values)
    selected_count = values.shape[0] * top_k
    stats = []
    for population in (scores, values):
        top = population.topk(min(top_k + near_count, values.shape[1]), dim=-1)
        boundary = top.values[:, top_k - 1:top_k]
        gap = (boundary - top.values[:, top_k:top_k + 1]).reshape(-1)
        near = (boundary - top.values[:, top_k:]).reshape(-1)
        # Selected experts become -inf margins, sorted before the unselected ones.
        excluded = population.scatter(-1, top.indices[:, :top_k], float("inf"))
        margins = (boundary - excluded).reshape(-1).sort().values
        quantiles = []
        for fraction in (0.25, 0.10):
            position = (margins.numel() - selected_count - 1) * fraction
            lower, upper = math.floor(position), math.ceil(position)
            quantiles.append(torch.lerp(margins[selected_count + lower], margins[selected_count + upper],
                                        position - lower))
        stats.extend((gap.median(), near.median(), *quantiles))
    return torch.stack(stats)


def _logits(case: str) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(64, 16, generator=generator)
    if case == "ties":
        logits = logits.round()
    elif case == "nan":
        logits[3, 5] = float("nan")
        logits[9, 0] = float("nan")
    elif case == "inf":
        logits[3, 5] = float("inf")
        logits[9, :2] = float("inf")
    return logits


@pytest.mark.parametrize("case", ["random", "ties", "nan", "inf"])
@pytest.mark.parametrize("scoring_function", ["softmax", "sigmoid"])
@pytest.mark.parametrize(("top_k", "near_count"), [(2, 3), (4, 12)])
def test_margins_equal_the_sorted_formula(case, scoring_function, top_k, near_count):
    logits = _logits(case)

    stats = router_margin_stats(scoring_function, top_k, near_count, logits)

    assert stats.shape == (ROUTER_MARGIN_WIDTH,)
    torch.testing.assert_close(stats, _sorted_margin_stats(scoring_function, top_k, near_count, logits),
                               rtol=0, atol=0, equal_nan=True)
