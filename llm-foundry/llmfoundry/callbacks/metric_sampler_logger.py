from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from composer.core import State
from composer.loggers import Logger, TensorboardLogger
from composer.loggers.logger_destination import LoggerDestination
from composer.utils import dist

from llmfoundry.callbacks.metric_sampler import attempt_id
from metric_sampler.logged import LoggedWriter

__all__ = ["MetricSamplerLogger"]

log = logging.getLogger(__name__)


class MetricSamplerLogger(LoggerDestination):
    """Writes every scalar Composer logs into `metric_sampler`'s `logged` family, on rank 0 only.

    Args:
        output_dir (str): Root directory for windows (`attempt=<id>/logged/w<a>-<b>/`), shared
            with `MetricSampler`'s node windows.
        window_steps (int): Number of distinct steps per written window. Defaults to 10.
    """

    def __init__(self, output_dir: str, window_steps: int = 10):
        self.output_dir = Path(output_dir)
        self.window_steps = window_steps
        self.writer: Optional[LoggedWriter] = None
        self._tensor_logging_safe = False
        self._warned_unsynced_tensor = False
        self._training_step: Optional[int] = None

    def fit_start(self, state: State, logger: Logger) -> None:
        # `Logger.log_metrics` calls every destination on every rank, so the rank filtering happens here, as in
        # `TensorboardLogger`.
        if dist.get_global_rank() != 0:
            return
        attempt_dir = self.output_dir / f"attempt={attempt_id()}"
        self.writer = LoggedWriter(attempt_dir, self.window_steps)
        # Safe to read a CUDA tensor without an extra host sync only if something ahead of us in
        # the destination list already synced it, e.g. a rank-zero-only TensorboardLogger.
        destinations = list(logger.destinations)
        self._tensor_logging_safe = any(
            isinstance(destination, TensorboardLogger) and destination.rank_zero_only
            for destination in destinations[: destinations.index(self)]
        )

    def batch_start(self, state: State, logger: Logger) -> None:
        self._training_step = state.timestamp.batch.value + 1

    def batch_end(self, state: State, logger: Logger) -> None:
        self._training_step = None

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.writer is None:
            return
        step_int = int(step) if step is not None else 0
        source_step = int(step) if step is not None else None
        # Composer advances its timestamp between loss logging and BATCH_END callbacks.
        if self._training_step is not None and step_int == self._training_step - 1:
            step_int = self._training_step
        for name, value in metrics.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                self.writer.add(step_int, name, float(value), source_step=source_step)
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                if value.is_cpu or self._tensor_logging_safe:
                    self.writer.add(step_int, name, float(value), source_step=source_step)
                elif not self._warned_unsynced_tensor:
                    self._warned_unsynced_tensor = True
                    log.warning(
                        "metric_sampler_logger: skipping unsynchronized CUDA tensor metrics "
                        "(no rank-zero-only TensorboardLogger ahead of this destination)"
                    )

    def fit_end(self, state: State, logger: Logger) -> None:
        self._close()

    def close(self, state: State, logger: Logger) -> None:
        self._close()

    def _close(self) -> None:
        if self.writer is not None:
            try:
                self.writer.close()
            except BufferError:
                raise
            except Exception:
                # Metrics must never break training: a failed write must not fail fit_end or close.
                log.exception("metric_sampler_logger: LoggedWriter.close() failed")
            self.writer = None
