from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.models.common.norms.const import DEFAULT_NORM_EPS
from torchtitan.config.schema.models.common.norms.base import (
    BaseNormConfig,
    validate_norm_eps,
)


@config(type="groupnorm")
class GroupNormConfig(BaseNormConfig):
    _owner_spec: ClassVar[str] = "torchtitan.models.common.norms.groupnorm:GroupNorm"
    default_init_role: ClassVar[str] = "norm"

    num_groups: int
    num_channels: int
    eps: float = DEFAULT_NORM_EPS

    def __post_init__(self) -> None:
        validate_norm_eps(self.eps)
        if self.normalized_shape is None:
            self.normalized_shape = self.num_channels
        elif self.normalized_shape != self.num_channels:
            raise ValueError(
                f"GroupNormConfig: normalized_shape ({self.normalized_shape}) "
                f"must equal num_channels ({self.num_channels})"
            )

    def fwd_flops_per_token(self, seq_len: int) -> int:
        return 0  # elementwise, no matmul
