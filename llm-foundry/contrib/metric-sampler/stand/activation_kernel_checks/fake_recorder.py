from __future__ import annotations

from dataclasses import dataclass

import torch

from metric_sampler.capture.recorder import ResolvedSlot
from metric_sampler.capture.stats.helpers.quantiles import QUANTILE_PERCENTILES
from metric_sampler.contract.metrics import (
    ACTIVATION_COUNTS_WIDTH,
    ACTIVATION_LOG2_BINS,
    QUANTILE_COUNTS_WIDTH,
    ValueType,
)

HISTOGRAM_BINS = ACTIVATION_LOG2_BINS.count
# The count slot aliases int32 words inside the recorder's float32 view.
MOMENTS_OFFSET = 0
MOMENTS_WIDTH = 4
COUNTS_OFFSET = 4
COUNT_TOTALS_WIDTH = 2
EXPONENT_HISTOGRAM_OFFSET = COUNTS_OFFSET + COUNT_TOTALS_WIDTH
TOKEN_RMS_HISTOGRAM_OFFSET = EXPONENT_HISTOGRAM_OFFSET + HISTOGRAM_BINS
AMAX_OFFSET = COUNTS_OFFSET + COUNT_TOTALS_WIDTH + 2 * HISTOGRAM_BINS
AMAX_WIDTH = 1
POWER_SUMS_OFFSET = AMAX_OFFSET + AMAX_WIDTH
POWER_SUMS_WIDTH = 2
BUFFER_WORDS = POWER_SUMS_OFFSET + POWER_SUMS_WIDTH
SLOTS_PER_MICROBATCH = 4
# The float sums, [sum, sumsq] of the moments and [sum_cube, sum_fourth] of the power sums, which differ by
# rounding between two calls even on one tensor: every other word of the statistics is an exact count or maximum.
FLOAT_SUM_WORDS = (MOMENTS_OFFSET, MOMENTS_OFFSET + 1, POWER_SUMS_OFFSET, POWER_SUMS_OFFSET + 1)
# The extension's call with exact quantiles writes them, then their counts, right after the statistics words.
QUANTILE_WORDS = len(QUANTILE_PERCENTILES)
QUANTILES_OFFSET = BUFFER_WORDS
QUANTILE_COUNTS_OFFSET = QUANTILES_OFFSET + QUANTILE_WORDS
WORDS_WITH_QUANTILES = QUANTILE_COUNTS_OFFSET + QUANTILE_COUNTS_WIDTH


@dataclass
class FakeRecorder:
    float_words: torch.Tensor
    sum_reference: tuple[float, float] | None = None
    sum_bounds: tuple[float, float] | None = None
    power_reference: tuple[float, float] | None = None
    power_bounds: tuple[float, float] | None = None
    # Whether an element's cube or fourth power is beyond the float32 range.
    power_overflow: tuple[bool, bool] = (False, False)

    @property
    def buffer(self) -> torch.Tensor:
        return self.float_words.view(torch.int32)

    def float_buffer(self) -> torch.Tensor:
        return self.float_words

    def resolve(self, metric_name: str, layer: int, microbatch: int) -> ResolvedSlot:
        suffix = metric_name.rsplit("/", 1)[-1]
        offset, width, slot_index = {
            "moments": (MOMENTS_OFFSET, MOMENTS_WIDTH, 0),
            "counts": (COUNTS_OFFSET, ACTIVATION_COUNTS_WIDTH, 1),
            "amax": (AMAX_OFFSET, AMAX_WIDTH, 2),
            "power_sums": (POWER_SUMS_OFFSET, POWER_SUMS_WIDTH, 3),
        }[suffix]
        return ResolvedSlot(
            offset=offset + microbatch * BUFFER_WORDS, width=width,
            slot_index=slot_index + microbatch * SLOTS_PER_MICROBATCH, detailed=False,
            value_type=ValueType.INT32 if suffix == "counts" else ValueType.FLOAT32
        )

    def mark_recorded(self, slot_indices) -> None:
        pass

    def tensor(self, suffix: str) -> torch.Tensor:
        offset, width = {
            "moments": (MOMENTS_OFFSET, MOMENTS_WIDTH),
            "counts": (COUNTS_OFFSET, ACTIVATION_COUNTS_WIDTH),
            "exp_hist": (EXPONENT_HISTOGRAM_OFFSET, HISTOGRAM_BINS),
            "token_rms_hist": (TOKEN_RMS_HISTOGRAM_OFFSET, HISTOGRAM_BINS),
            "amax": (AMAX_OFFSET, AMAX_WIDTH),
            "power_sums": (POWER_SUMS_OFFSET, POWER_SUMS_WIDTH),
        }[suffix]
        buffer = self.buffer if suffix in ("counts", "exp_hist", "token_rms_hist") else self.float_words
        return buffer[offset : offset + width]


def make_recorder(device: torch.device) -> FakeRecorder:
    return FakeRecorder(torch.zeros(2 * BUFFER_WORDS, dtype=torch.float32, device=device))
