from typing import ClassVar, Literal

from torchtitan.config.configurable import config
from torchtitan.config.errors import ConfigFieldError
from torchtitan.config.schema.models.common.norms.const import DEFAULT_NORM_EPS
from torchtitan.config.schema.models.common.norms.base import (
    BaseNormConfig,
    validate_norm_eps,
)


@config(type="gated_rmsnorm")
class GatedRMSNormConfig(BaseNormConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.models.common.norms.gated_norm:GatedRMSNorm"
    )
    default_init_role: ClassVar[str] = "gated_norm"

    eps: float = DEFAULT_NORM_EPS
    gate_rank: int = 16
    gating_weight: float = 2.0
    # False: scale by ``weight``; True: scale by ``1 + weight``.
    zero_centered: bool = False
    norm_backend: Literal["compile", "cute"] = "compile"

    def __post_init__(self) -> None:
        validate_norm_eps(self.eps)
        if self.norm_backend not in ("compile", "cute"):
            raise ConfigFieldError("norm_backend", "must be 'compile' or 'cute'")
        if self.norm_backend == "cute":
            if self.normalized_shape is not None and (
                self.normalized_shape <= 0 or self.normalized_shape % 128
            ):
                raise ConfigFieldError(
                    "normalized_shape",
                    "norm_backend='cute' requires a positive multiple of 128",
                )
            if self.gate_rank != 16:
                raise ConfigFieldError(
                    "gate_rank", "norm_backend='cute' requires gate_rank=16"
                )

    def fwd_flops_per_token(self, seq_len: int) -> int:
        # Gate MLP matmuls: up (2*hidden*r) + down (2*r*hidden). The RMS
        # reduction / elementwise ops are memory-bound (0), like RMSNorm.
        assert self.normalized_shape is not None
        return 4 * self.normalized_shape * self.gate_rank
