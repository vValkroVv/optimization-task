from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.callbacks.base import CallbackConfig


@config(type="timers_monitor")
class TimersMonitorConfig(CallbackConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.callbacks.builtin.timers_monitor:TimersMonitor"
    )

    interval: int = 10
    timers_threshold: int = 1

    def __post_init__(self) -> None:
        if self.interval < 1:
            raise ValueError(f"'interval' must be >= 1, got {self.interval}.")
        if self.timers_threshold not in (0, 1, 2):
            raise ValueError(
                f"`timers_threshold` must be one of {{0, 1, 2}}, "
                f"got {self.timers_threshold}."
            )
