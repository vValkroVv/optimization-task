import math
import re
from dataclasses import dataclass, field
from typing import ClassVar, Literal

from torchtitan.config.configurable import config, Configurable
from torchtitan.config.schema.protocols.types import StrEnum


class OptimizerName(StrEnum):
    ADAM = "Adam"
    ADAMW = "AdamW"


@dataclass(kw_only=True, slots=True)
class ParamGroupConfig:
    """Configuration for a parameter group with custom optimizer settings.

    Parameters matching the regex pattern will use lr and weight_decay values
    derived by multiplying the global optimizer values with the specified multipliers.
    """

    pattern: str
    """Regex pattern matched against parameter fully qualified names (FQNs).
    E.g. '.*bias$', '.*norm.*', '.*\\.embed_tokens\\..*'"""

    lr_multiplier: float = 1.0
    """Multiplied with the global optimizer lr to get this group's lr."""

    weight_decay_multiplier: float = 1.0
    """Multiplied with the global optimizer weight_decay to get this group's weight_decay."""

    beta1: float | None = None
    beta2: float | None = None
    """Override betas for this group. None means use the global optimizer betas.
    Each can be overridden independently."""

    def __post_init__(self) -> None:
        try:
            re.compile(self.pattern)
        except re.error as exc:
            raise ValueError(f"invalid parameter-group pattern: {exc}") from exc
        for name in ("lr_multiplier", "weight_decay_multiplier"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("beta1", "beta2"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or not 0 <= value < 1):
                raise ValueError(f"{name} must be finite and in [0, 1)")


@config(family="optimizer", default="default", type="default")
class OptimizersContainerConfig(Configurable.Config):
    _owner_spec: ClassVar[str] = "torchtitan.components.optimizer:OptimizersContainer"

    name: str = "AdamW"

    lr: float = 8e-4

    beta1: float = 0.9
    beta2: float = 0.95

    eps: float = 1e-8

    weight_decay: float = 0.1

    implementation: Literal["for-loop", "foreach", "fused", "fused_opt_states_bf16"] = (
        "fused"
    )
    """
    Specify which optimizer implementation to use:
    - 'fused': Use fused implementation (CUDA only) for best performance.
    - 'foreach': Use some horizontal fusion of tensors for better performance.
    - 'for-loop': Use the default implementation for the optimizer (slowest).
    - 'fused_opt_states_bf16': Like 'fused', but initialize Adam/AdamW
      momentum and variance in bfloat16 via a step pre-hook so the fused
      CUDA kernel uses its mixed-precision path (fp32 params + bf16 states).
      Only supported for Adam/AdamW with OptimizersContainer (not
      OptimizersInBackwardContainer). See docs/bf16_optimizer_states.md.
    - more info: https://pytorch.org/docs/stable/optim.html
    """

    param_groups: list[ParamGroupConfig] = field(default_factory=list)
    """Per-parameter-group overrides; the first matching pattern wins and
    unmatched parameters use the global values."""

    def __post_init__(self) -> None:
        OptimizerName(self.name)
        for name in ("lr", "eps", "weight_decay"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("beta1", "beta2"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} must be finite and in [0, 1)")
        if self.implementation == "fused_opt_states_bf16":
            if self.name not in ("Adam", "AdamW"):
                raise ValueError(
                    "implementation='fused_opt_states_bf16' is only supported "
                    f"for Adam/AdamW, got optimizer '{self.name}'"
                )


@config(type="in_backward")
class OptimizersInBackwardContainerConfig(OptimizersContainerConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.components.optimizer:OptimizersInBackwardContainer"
    )

    def __post_init__(self) -> None:
        if self.implementation == "fused_opt_states_bf16":
            raise ValueError(
                "implementation='fused_opt_states_bf16' is not supported with "
                "OptimizersInBackwardContainer"
            )
        OptimizersContainerConfig.__post_init__(self)
