from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.models.common.norms.const import DEFAULT_NORM_EPS
from torchtitan.config.schema.models.common.norms.base import (
    BaseNormConfig,
    validate_norm_eps,
)


@config(type="rmsnorm")
class RMSNormConfig(BaseNormConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.common.norms.rmsnorm:RMSNorm"
    default_init_role: ClassVar[str] = "norm"

    eps: float = DEFAULT_NORM_EPS
    elementwise_affine: bool = True

    def __post_init__(self) -> None:
        validate_norm_eps(self.eps)

    def fwd_flops_per_token(self, seq_len: int) -> int:
        return 0  # elementwise, no matmul
