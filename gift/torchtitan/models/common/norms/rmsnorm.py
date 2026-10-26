import torch.nn as nn

from typing import TypeAlias

from torchtitan.config.schema.models.common.norms.rmsnorm import RMSNormConfig
from torchtitan.protocols.module import Module

__all__ = ["RMSNorm"]


class RMSNorm(nn.RMSNorm, Module):
    """Configurable nn.RMSNorm."""

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = RMSNormConfig

    def __init__(self, config: Config):
        assert config.normalized_shape is not None, "partial RMSNorm.Config"
        super().__init__(
            config.normalized_shape,
            eps=config.eps,
            elementwise_affine=config.elementwise_affine,
        )
