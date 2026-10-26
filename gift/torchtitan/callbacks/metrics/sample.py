from dataclasses import dataclass, field

import torch

from torchtitan.config.schema.callbacks.metrics.sample import (
    AxisSpec,
    DTensorMode,
    MetricConfig,
    MetricStepConfig,
    ReduceConfig,
    ReduceStageConfig,
    StatOpConfig,
)

__all__ = [
    "AxisSpec",
    "DTensorMode",
    "MetricConfig",
    "MetricRecord",
    "MetricStepConfig",
    "MetricValue",
    "ReduceConfig",
    "ReduceStageConfig",
    "Sample",
    "StatOpConfig",
]

MetricValue = torch.Tensor | int | float


@dataclass(kw_only=True, slots=True)
class Sample:
    """Mutable in-flight payload: a value plus the metric plans to run on it."""

    name: str
    value: MetricValue
    axes: AxisSpec = field(default_factory=AxisSpec)
    metrics: list[MetricConfig] = field(default_factory=list)


@dataclass(frozen=True, kw_only=True, slots=True)
class MetricRecord:
    """Immutable computed metric awaiting reduction and logging.

    ``pending_stages`` holds reductions deferred to
    ``finalize_pending_reductions``. ``post_stats`` holds the stat steps of a
    ``[reduce..., stat...]`` metric, which by definition run on the REDUCED
    value: they are applied by ``apply_post_stats`` right after the pending
    reductions resolve, on ``post_axes``. Deferring the whole chain is what
    lets the reductions of many such records fuse into one collective instead
    of one inline collective per metric per layer.
    """

    name: str
    value: MetricValue
    step: int
    reduce: ReduceConfig
    pending_stages: tuple[ReduceStageConfig, ...] = ()
    post_stats: tuple[MetricStepConfig, ...] = ()
    post_axes: AxisSpec = AxisSpec()
