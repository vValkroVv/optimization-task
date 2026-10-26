from __future__ import annotations

import torch

from metric_sampler.contract.metrics import HistogramBins


def bin_indices(values: torch.Tensor, bins: HistogramBins) -> torch.Tensor:
    # Not torch.bucketize: the grid is linear, so a closed-form index avoids a sorted-search op.
    return ((values - bins.start) / bins.width).floor().clamp_(0, bins.count - 1).long()
