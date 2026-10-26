from __future__ import annotations

import pytest
import torch

from metric_sampler.capture.stats.helpers.histograms import bin_indices
from metric_sampler.contract.metrics import HistogramBins

pytestmark = pytest.mark.unit


def test_bin_indices_floor_into_the_linear_grid_and_clamp_both_ends() -> None:
    bins = HistogramBins(start=-1.0, width=0.5, count=4)
    values = torch.tensor([-5.0, -1.0, -0.51, -0.5, 0.0, 0.49, 0.5, 0.99, 1.0, 7.0])

    indices = bin_indices(values, bins)

    assert indices.dtype == torch.int64
    assert indices.tolist() == [0, 0, 0, 1, 2, 2, 3, 3, 3, 3]
