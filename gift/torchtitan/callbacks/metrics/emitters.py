from collections.abc import Callable, Mapping

import torch

from torchtitan.callbacks.metrics.reducers import (
    gather_metrics_over_pp,
    PPGatherCache,
    STEP_SEP,
)
from torchtitan.callbacks.metrics.sample import MetricRecord
from torchtitan.config.schema.callbacks.metrics.sample import ReduceMode
from torchtitan.tools.logging import logger


class MetricEmitter:
    def __init__(self) -> None:
        self._pp_cache = PPGatherCache()
        self._warned_non_scalar: set[str] = set()

    def emit(
        self,
        records: list[MetricRecord],
        *,
        parallel_dims,
        publish: Callable[[int, Mapping[str, float]], None],
    ) -> None:
        """Publish the records' scalars through ``publish``."""

        metrics: dict[str, float] = {}
        tensor_records: dict[torch.device, list[tuple[str, torch.Tensor]]] = {}
        for record in records:
            if record.reduce.mode != ReduceMode.online:
                continue
            key = f"{record.step}{STEP_SEP}{record.name}"
            value = record.value
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    if record.name not in self._warned_non_scalar:
                        self._warned_non_scalar.add(record.name)
                        logger.warning(
                            "Monitoring: tag '%s' has shape %s and cannot be "
                            "emitted to the metrics logger, which takes "
                            "scalars; add a stat step that reduces it, or "
                            "enable the metric store to keep raw vectors.",
                            record.name,
                            tuple(value.shape),
                        )
                    continue
                tensor_records.setdefault(value.device, []).append((key, value))
            else:
                metrics[key] = float(value)
        for device_records in tensor_records.values():
            # FP32 packing would round integer counters above 2**24.
            stacked = torch.stack(
                [value.detach().double().reshape(()) for _, value in device_records]
            ).cpu()
            for (key, _), scalar in zip(device_records, stacked.tolist()):
                metrics[key] = scalar
        metrics = gather_metrics_over_pp(metrics, parallel_dims, self._pp_cache)
        by_step: dict[int, dict[str, float]] = {}
        for key, value in metrics.items():
            step_text, _, name = key.partition(STEP_SEP)
            by_step.setdefault(int(step_text), {})[name] = value
        for step in sorted(by_step):
            publish(step, by_step[step])
