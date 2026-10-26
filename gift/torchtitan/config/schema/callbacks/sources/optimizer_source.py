from dataclasses import field
from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.callbacks.metrics.sample import MetricConfig, StatOpConfig
from torchtitan.config.schema.callbacks.sources.base import FilteredSourceConfig


def _default_optimizer_metrics() -> list[MetricConfig]:
    return [
        MetricConfig(name="mean", pipeline=[StatOpConfig(op="mean")]),
        MetricConfig(name="std", pipeline=[StatOpConfig(op="std")]),
        MetricConfig(name="abs_max", pipeline=[StatOpConfig(op="abs_max")]),
        MetricConfig(name="finite_ratio", pipeline=[StatOpConfig(op="finite_ratio")]),
    ]


@config(type="optimizer")
class OptimizerSourceConfig(FilteredSourceConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.callbacks.sources.optimizer_source:OptimizerSource"
    )

    gradients: bool = True
    parameters: bool = True
    optimizer_states: bool = True
    per_expert_metrics: bool = False
    metrics: list[MetricConfig] = field(default_factory=_default_optimizer_metrics)
