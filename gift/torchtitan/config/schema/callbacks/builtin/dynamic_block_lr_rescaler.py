from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.callbacks.base import CallbackConfig
from torchtitan.config.schema.callbacks.builtin.monitoring import (
    MonitoringCallbackConfig,
)
from torchtitan.config.validation import Issue


@config(type="dynamic_block_lr_rescaler")
class DynamicBlockLRRescalerConfig(CallbackConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.callbacks.builtin.dynamic_block_lr_rescaler:DynamicBlockLRRescaler"
    )

    requires: ClassVar[tuple[str, ...]] = ("monitoring",)
    # Which DispatcherSource max_vio series drives the decision.
    max_vio_series: str = "experts_max_vio_batch_avg"
    # Multiplicative LR factor applied to a block when the heuristic fires.
    reduce_coef: float = 0.99
    # Rolling max-violation samples the slope is fit to; one sample per
    # monitoring log_interval. Checkpointed.
    window_size: int = 100
    # Minimum samples between consecutive fires of the same block.
    cooldown: int = 20
    # Relative-slope dead-band: fire when theil_sen_slope / median exceeds
    # this. Median-normalization makes the test scale-free across layers;
    # 0.0 reproduces a bare ``slope > 0`` test.
    slope_threshold: float = 0.05
    # Hard lower bound on the cumulative dynamic scale.
    lr_scaling_floor: float = 0.3
    # No rescaling before this step; must exceed window_size so the first
    # decision never sees pre-warmup chaos.
    warmup_steps: int = 2000
    # Couple the leading dense layers to the first MoE block's scale.
    mirror_dense_prefix: bool = False

    def validate_dependencies(self, previous: list[CallbackConfig]) -> list[Issue]:
        count = sum(isinstance(c, MonitoringCallbackConfig) for c in previous)
        if count == 1:
            return []
        return [
            Issue(
                path="",
                message=(
                    "dynamic_block_lr_rescaler requires exactly one monitoring callback "
                    f"before it in callbacks; found {count}."
                ),
            )
        ]

    def __post_init__(self) -> None:
        if not 0.0 < self.reduce_coef <= 1.0:
            raise ValueError(f"reduce_coef must be in (0, 1]; got {self.reduce_coef}")
        if not 0.0 < self.lr_scaling_floor <= 1.0:
            raise ValueError(
                f"lr_scaling_floor must be in (0, 1]; got {self.lr_scaling_floor}"
            )
        if self.slope_threshold < 0.0:
            raise ValueError(
                f"slope_threshold must be >= 0; got {self.slope_threshold}"
            )
        if self.window_size < 2:
            raise ValueError(f"window_size must be >= 2; got {self.window_size}")
        if self.cooldown < 1:
            raise ValueError(f"cooldown must be >= 1; got {self.cooldown}")
        if self.warmup_steps <= self.window_size:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be greater than "
                f"window_size ({self.window_size})"
            )
