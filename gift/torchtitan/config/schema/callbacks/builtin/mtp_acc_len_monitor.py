from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.callbacks.base import CallbackConfig


@config(type="mtp_acc_len_monitor")
class MtpAccLenMonitorConfig(CallbackConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.callbacks.builtin.mtp_acc_len_monitor:MtpAccLenMonitor"
    )

    log_interval: int = 10

    def __post_init__(self) -> None:
        if self.log_interval <= 0:
            raise ValueError("MtpAccLenMonitor log_interval must be positive")
