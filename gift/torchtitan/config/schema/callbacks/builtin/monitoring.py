from dataclasses import field
from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.callbacks.base import CallbackConfig
from torchtitan.config.schema.callbacks.metrics.stores import MetricStoreConfig
from torchtitan.config.schema.callbacks.sources.base import MonitorSourceConfig


@config(type="monitoring")
class MonitoringCallbackConfig(CallbackConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.callbacks.builtin.monitoring:MonitoringCallback"
    )
    requires: ClassVar[tuple[str, ...]] = ("sharding_backend",)

    log_interval: int = 100
    debug_full_tensor: bool = False
    empty_cuda_cache_before_optimizer: bool = True
    sources: dict[str, MonitorSourceConfig] = field(default_factory=dict)
    store: MetricStoreConfig = field(default_factory=MetricStoreConfig)

    def __post_init__(self) -> None:
        if self.log_interval <= 0:
            raise ValueError("MonitoringCallback log_interval must be positive")
