import math
from typing import ClassVar

from torchtitan.config.errors import ConfigFieldError
from torchtitan.config.configurable import config, Derived
from torchtitan.config.schema.models.common.moe.const import ScoreFunc
from torchtitan.config.schema.models.common.nn_modules import LinearConfig
from torchtitan.config.schema.protocols.module import ModuleConfig


@config
class TokenChoiceTopKRouterConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.models.common.moe.router:TokenChoiceTopKRouter"
    )

    # None: the owning MoE sizes it (dim -> num_experts) with the gate child's initializer.
    gate: Derived[LinearConfig] = None
    num_expert_groups: int | None = None  # must be a divisor of num_experts
    num_limited_groups: int | None = None
    top_k: int = 1
    score_func: ScoreFunc = ScoreFunc.SIGMOID
    route_norm: bool = False
    route_scale: float = 1.0
    _debug_force_load_balance: bool = False
    _debug_force_first_group: bool = False

    @property
    def num_experts(self) -> int:
        """One logit per expert, so the gate width says how many there are."""
        assert self.gate is not None
        return self.gate.require_features()[1]

    def __post_init__(self) -> None:
        if self._debug_force_load_balance and self._debug_force_first_group:
            raise ValueError(
                "_debug_force_load_balance and _debug_force_first_group are mutually exclusive"
            )
        if not math.isfinite(self.route_scale):
            raise ConfigFieldError("route_scale", "must be finite")
        if type(self.top_k) is not int or self.top_k <= 0:
            raise ConfigFieldError("top_k", "must be a positive integer")
        for name in ("num_expert_groups", "num_limited_groups"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ConfigFieldError(name, "must be a positive integer when set")
        if self.num_limited_groups is not None:
            if self.num_expert_groups is None:
                raise ConfigFieldError(
                    "num_limited_groups", "requires num_expert_groups"
                )
            if self.num_limited_groups > self.num_expert_groups:
                raise ConfigFieldError(
                    "num_limited_groups", "must not exceed num_expert_groups"
                )
        if self.gate is not None and self.gate.out_features is not None:
            if self.top_k > self.gate.out_features:
                raise ConfigFieldError(
                    "top_k", f"must not exceed num_experts ({self.gate.out_features})"
                )

        if (
            self.num_expert_groups is None
            or self.gate is None
            or self.gate.in_features is None
            or self.gate.out_features is None
        ):
            return  # node-limited routing off, or partial: the MoE sizes the gate
        if self.num_limited_groups is None:
            raise ValueError(
                "num_limited_groups must be set when num_expert_groups is set"
            )
        if self.num_experts % self.num_expert_groups != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by "
                f"num_expert_groups ({self.num_expert_groups})"
            )
        experts_per_group = self.num_experts // self.num_expert_groups
        if experts_per_group < 2:
            raise ValueError(f"experts_per_group ({experts_per_group}) must be >= 2")
        # After group masking only num_limited_groups groups survive; top_k must
        # fit within the experts they hold or torch.topk would select -inf entries.
        surviving_experts = self.num_limited_groups * experts_per_group
        if surviving_experts < self.top_k:
            raise ValueError(
                f"node-limited routing keeps num_limited_groups "
                f"({self.num_limited_groups}) * experts_per_group "
                f"({experts_per_group}) = {surviving_experts} experts, which is "
                f"fewer than top_k ({self.top_k})"
            )
