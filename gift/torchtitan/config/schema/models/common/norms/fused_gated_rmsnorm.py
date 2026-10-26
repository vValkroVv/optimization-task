from typing import ClassVar

from torchtitan.config.configurable import config, OwnerFilled
from torchtitan.config.schema.models.common.norms.base import validate_norm_eps
from torchtitan.config.schema.models.common.norms.const import (
    DEFAULT_NORM_EPS,
    GatedNormActivation,
)
from torchtitan.config.schema.protocols.module import ModuleConfig


@config
class FusedRMSNormGatedConfig(ModuleConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.models.common.norms.fused_gated_rmsnorm:FusedRMSNormGated"
    )
    default_init_role: ClassVar[str] = "o_norm"

    # None: the owner fills it (a GDN's ``head_v_dim``); set: the owner checks it.
    hidden_size: OwnerFilled[int] = None
    activation: GatedNormActivation = GatedNormActivation.SWISH
    eps: float = DEFAULT_NORM_EPS
    zero_centered: bool = False

    def __post_init__(self) -> None:
        validate_norm_eps(self.eps)

    def fwd_flops_per_token(self, seq_len: int) -> int:
        # Elementwise, no matmul.
        return 0
