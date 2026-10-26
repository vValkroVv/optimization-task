from __future__ import annotations

import torch

from metric_sampler.capture.recorder import CaptureCapacityError, RankRecorder
from metric_sampler.capture.stats.helpers.quantiles import exact_quantiles, exact_quantiles_into, validate_population


class ExactQuantileStats:
    """Exact quantiles of each rank/layer/microbatch population, computed as it is observed.

    Every point is observed once per microbatch, so no population is retained. Quantiles of a
    union cannot be built from the quantiles of its parts: a second observation of one key within
    a microbatch is rejected rather than approximated.
    """

    def __init__(self, recorder: RankRecorder) -> None:
        self._recorder = recorder
        self._microbatch: int | None = None
        self._observed: set[tuple[str, int]] = set()

    def observe(
        self,
        metric: str,
        counts_metric: str,
        layer: int,
        microbatch: int,
        values: torch.Tensor,
        valid: torch.Tensor | None = None,
        *,
        pack_float64: bool = False,
        first_pass: torch.Tensor | None = None,
    ) -> None:
        validate_population(values, valid)
        if (values.dtype == torch.float64) != pack_float64:
            raise ValueError("float64 populations require explicit lossless word packing; other dtypes must not pack")
        if pack_float64:
            self.record_float64(metric, counts_metric, layer, microbatch, *exact_quantiles(values, valid))
            return
        self._check_unobserved(metric, layer, microbatch)
        quantiles = self._recorder.accumulator(metric, layer, microbatch)
        counts = self._recorder.accumulator(counts_metric, layer, microbatch)
        exact_quantiles_into(values, valid, quantiles, counts, first_pass)
        self._mark_observed(metric, layer, microbatch)

    def record_float64(
        self, metric: str, counts_metric: str, layer: int, microbatch: int, quantiles: torch.Tensor,
        counts: torch.Tensor,
    ) -> None:
        """Records `exact_quantiles` of a float64 population computed by the caller (for example
        replayed from a CUDA graph), the quantiles losslessly packed into integer words."""
        self._check_unobserved(metric, layer, microbatch)
        self._recorder.accumulator(metric, layer, microbatch).copy_(quantiles.contiguous().view(torch.int32))
        self._recorder.accumulator(counts_metric, layer, microbatch).copy_(counts)
        self._mark_observed(metric, layer, microbatch)

    def claim(self, metric: str, layer: int, microbatch: int) -> None:
        """Registers a population whose quantiles the caller writes into the metric's slot itself
        (the extension selects activation quantiles in the statistics call)."""
        self._check_unobserved(metric, layer, microbatch)
        self._mark_observed(metric, layer, microbatch)

    def _check_unobserved(self, metric: str, layer: int, microbatch: int) -> None:
        if self._microbatch is not None and microbatch != self._microbatch:
            raise CaptureCapacityError("quantile populations must be flushed before the next microbatch")
        if (metric, layer) in self._observed:
            raise ValueError(f"{metric} layer {layer} was observed twice in microbatch {microbatch}; "
                             "exact quantiles of a repeated population are not supported")

    def _mark_observed(self, metric: str, layer: int, microbatch: int) -> None:
        self._observed.add((metric, layer))
        self._microbatch = microbatch

    def flush(self) -> None:
        self._observed.clear()
        self._microbatch = None
