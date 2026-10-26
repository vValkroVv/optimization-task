from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.callbacks.base import CallbackConfig


@config(type="system_monitor")
class SystemMetricsMonitorConfig(CallbackConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.callbacks.builtin.system_monitor:SystemMetricsMonitor"
    )

    host_log_interval: int = 50
    memory_log_interval: int = 50

    def __post_init__(self) -> None:
        for name in ("host_log_interval", "memory_log_interval"):
            interval = getattr(self, name)
            if interval < 0:
                raise ValueError(
                    f"SystemMetricsMonitor '{name}' must be >= 0, got {interval}."
                )
        if not (self.host_log_interval or self.memory_log_interval):
            raise ValueError(
                "SystemMetricsMonitor with both intervals at 0 would log "
                "nothing. Set host_log_interval and/or memory_log_interval, "
                "or leave the callback out."
            )
