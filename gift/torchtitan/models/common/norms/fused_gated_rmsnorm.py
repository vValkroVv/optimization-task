"""``fla``'s fused gated RMSNorm (``rmsnorm(x) * act(gate)``), wrapped as a
torchtitan ``Module``. Output norm of the gated-delta linear-attention families
(GDN, KDA). Imports ``fla`` at top level (an optional dependency), so callers must
import it lazily.
"""

import torch
from fla.modules import FusedRMSNormGated as _FusedRMSNormGated

from typing import TypeAlias

from torchtitan.config.schema.models.common.norms.fused_gated_rmsnorm import (
    FusedRMSNormGatedConfig,
)
from torchtitan.models.common.attention.linear.fla_custom_ops import gated_rmsnorm_op
from torchtitan.protocols.module import Module

__all__ = ["FusedRMSNormGated"]


class FusedRMSNormGated(_FusedRMSNormGated, Module):
    """Configurable fla ``FusedRMSNormGated`` as a torchtitan ``Module``."""

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = FusedRMSNormGatedConfig

    def __init__(self, config: Config):
        assert config.hidden_size is not None, "partial FusedRMSNormGated.Config"
        super().__init__(
            hidden_size=config.hidden_size,
            eps=config.eps,
            activation=config.activation,
            zero_centered=config.zero_centered,
        )

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # Through the custom op rather than fla's forward: that one is
        # ``torch.compiler.disable``d, which blocks ``fullgraph=True``.
        return gated_rmsnorm_op(
            x, g, self.weight, self.eps, self.activation, self.zero_centered
        )
