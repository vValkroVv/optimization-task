from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.callbacks.base import CallbackConfig


@config(type="ema")
class EMAConfig(CallbackConfig):
    _owner_spec: ClassVar[str] = "torchtitan.callbacks.builtin.ema:EMA"

    requires: ClassVar[tuple[str, ...]] = ("model_parts", "sharding_backend")
    use_ema: bool = True
    decay: float = 0.999
    update_period: int = 1
    start_step: int = 0

    def __post_init__(self) -> None:
        if self.update_period < 1:
            raise ValueError(
                f"EMA 'update_period' must be >= 1, got {self.update_period}."
            )
        if not 0.0 < self.decay < 1.0:
            raise ValueError(f"EMA 'decay' must be in (0, 1), got {self.decay}.")
        if self.start_step < 0:
            raise ValueError(f"EMA 'start_step' must be >= 0, got {self.start_step}.")
