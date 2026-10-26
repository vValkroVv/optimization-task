from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.errors import ConfigFieldError
from torchtitan.config.schema.models.common.regularization import validate_z_loss_coef
from torchtitan.config.schema.models.common.ffn import FeedForwardConfig
from torchtitan.config.schema.models.common.moe.experts.gmm import GroupedExpertsConfig
from torchtitan.config.schema.models.common.moe.router import (
    TokenChoiceTopKRouterConfig,
)
from torchtitan.config.schema.models.common.nn_modules import LinearConfig
from torchtitan.config.schema.models.common.validation import (
    resolve_child,
    validate_values,
)
from torchtitan.config.schema.protocols.module import ModuleConfig


@config(family="moe", default="token_choice", type="token_choice")
class MoEConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.common.moe:MoE"

    experts: GroupedExpertsConfig
    router: TokenChoiceTopKRouterConfig
    # 0 disables the expert-bias load balancing (and its bias buffer).
    load_balance_coeff: float = 1e-3
    gate_bias: ClassVar[bool | None] = None
    """Gate bias fixed by the family; ``None`` leaves it to the gate config."""

    @property
    def num_experts(self) -> int:
        return self.experts.num_experts

    shared_experts: FeedForwardConfig | None = None
    router_aux_loss_coef: float = 0.0
    router_z_loss_coef: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "load_balance_coeff",
            "router_aux_loss_coef",
            "router_z_loss_coef",
        ):
            validate_z_loss_coef(getattr(self, name), name)
        if self.shared_experts is not None:
            if (
                self.shared_experts.hidden_dim is not None
                and self.shared_experts.hidden_dim <= 0
            ):
                raise ConfigFieldError("shared_experts.hidden_dim", "must be positive")
            if (
                self.shared_experts.dim is not None
                and self.experts.dim is not None
                and self.shared_experts.dim != self.experts.dim
            ):
                raise ConfigFieldError(
                    "shared_experts.dim",
                    f"shared_experts.dim must match experts.dim ({self.experts.dim}), got {self.shared_experts.dim}",
                )

        dim = self.experts.dim
        if dim is None:
            return  # partial: the owner fills the experts' dim, then this runs again
        # The gate is dim -> num_experts; shared experts take the routed dim.
        gate = resolve_child(
            self.router.gate,
            LinearConfig,
            name="MoE.router.gate",
            dims=dict(in_features=dim, out_features=self.num_experts),
            require=None if self.gate_bias is None else dict(bias=self.gate_bias),
            extra=dict(init_role="router_gate"),
        )
        if gate is not self.router.gate:
            self.router = self.router._copy(gate=gate)
        if self.shared_experts is not None:
            self.shared_experts = resolve_child(
                self.shared_experts,
                FeedForwardConfig,
                name="MoE.shared_experts",
                dims=dict(dim=dim),
            )
        validate_values(
            "MoE", ("num_experts", self.router.num_experts, self.num_experts)
        )

    def fwd_flops_per_token(self, seq_len: int) -> int:
        # Router gate matmul; scoring and top-k count as 0.
        assert self.router.gate is not None
        flops = self.router.gate.fwd_flops_per_token(seq_len)
        # top_k routed experts per token, each a SwiGLU: 6·dim·hidden_dim.
        assert self.experts.dim is not None
        flops += self.router.top_k * 6 * self.experts.dim * self.experts.hidden_dim
        # Shared experts run for every token.
        if self.shared_experts is not None:
            flops += self.shared_experts.fwd_flops_per_token(seq_len)
        return flops
