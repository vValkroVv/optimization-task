from dataclasses import field
from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.models.common.moe.dispatch.base import (
    BaseTokenDispatcherConfig,
)
from torchtitan.config.schema.models.common.moe.dispatch.local import (
    LocalTokenDispatcherConfig,
)
from torchtitan.config.schema.protocols.module import ModuleConfig


@config
class GroupedExpertsConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.models.common.moe.experts.gmm:GroupedExperts"
    )
    default_init_role: ClassVar[str] = "experts"

    # None: the owner fills it.
    dim: int | None = None
    hidden_dim: int
    num_experts: int
    # None resolves to the local dispatcher; ``parallelize`` swaps in the EP
    # one. Never None after __post_init__.
    token_dispatcher: BaseTokenDispatcherConfig | None = None
    # True: scores scale the expert input; False: its output.
    score_before_experts: bool = False
    # Derived by FP8Config at runtime; not a second YAML settings surface.
    use_float8_grouped_gemm: bool = field(
        default=False,
        init=False,
        metadata={"config_source": "fp8.sparse_use_grouped_gemm"},
    )
    deep_gemm_num_sms: int = field(
        default=116,
        init=False,
        metadata={"config_source": "fp8.sparse_deep_gemm_num_sms"},
    )
    sparse_fused_swiglu_quant: bool = field(
        default=True,
        init=False,
        metadata={"config_source": "fp8.sparse_fused_swiglu_quant"},
    )
    # Recompute local BF16 expert intermediates; dispatch/combine run once.
    recompute: bool = False
    swiglu_limit: float = 0.0
    # Hidden z-loss on the gate (w1) and up (w3) halves of the fused w13.
    w1_hidden_z_loss_coef: float = 0.0
    w3_hidden_z_loss_coef: float = 0.0
    hidden_z_loss_projections: ClassVar[dict[str, str]] = {
        "w1": "w1_hidden_z_loss_coef",
        "w3": "w3_hidden_z_loss_coef",
    }

    def __post_init__(self) -> None:
        if type(self.recompute) is not bool:
            raise ValueError("GroupedExperts.recompute must be a bool")
        if self.token_dispatcher is None:
            self.token_dispatcher = LocalTokenDispatcherConfig()
        if self.swiglu_limit < 0.0:
            raise ValueError(
                f"GroupedExperts.swiglu_limit must be >= 0, got {self.swiglu_limit}."
            )
