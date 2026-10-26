from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.callbacks.base import CallbackConfig


@config(type="prores_warmup")
class ProResWarmupConfig(CallbackConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.callbacks.builtin.prores_warmup:ProResWarmup"
    )

    emit_metrics: bool = False
    log_interval: int = 10

    def __post_init__(self) -> None:
        if self.log_interval <= 0:
            raise ValueError("ProResWarmup log_interval must be positive")
