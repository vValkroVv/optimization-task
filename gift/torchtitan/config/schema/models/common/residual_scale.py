from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.protocols.module import ModuleConfig
from torchtitan.config.schema.protocols.types import StrEnum


class ResidualScaleSchedule(StrEnum):
    """Warmup shape ``f(p)`` for progress ``p`` in [0, 1]."""

    linear = "linear"  # p
    sqrt = "sqrt"  # sqrt(p)
    cosine = "cosine"  # 1 - cos(pi * p / 2)
    power = "power"  # p ** power


@config
class ResidualScaleConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.common.residual_scale:ResidualScale"

    # This layer's warmup; fractional under depth scaling.
    warmup_steps: float
    start_value: float
    schedule: ResidualScaleSchedule = ResidualScaleSchedule.linear
    # Only for ``schedule=power``.
    power: float = 1.0

    def __post_init__(self) -> None:
        if self.warmup_steps <= 0:
            raise ValueError(
                f"ResidualScale warmup_steps must be > 0, got {self.warmup_steps}"
            )
        if not 0.0 <= self.start_value <= 1.0:
            raise ValueError(
                f"ResidualScale start_value must be in [0, 1], got {self.start_value}"
            )
        if self.power <= 0:
            raise ValueError(f"ResidualScale power must be > 0, got {self.power}")
        if self.schedule != ResidualScaleSchedule.power and self.power != 1.0:
            raise ValueError(
                f"ResidualScale power={self.power} has no effect with "
                f"schedule={self.schedule!r}; use schedule='power'."
            )

    def fwd_flops_per_token(self, seq_len: int) -> int:
        return 0  # elementwise
