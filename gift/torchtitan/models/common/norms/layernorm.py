import torch.nn as nn

from typing import TypeAlias

from torchtitan.config.schema.models.common.norms.layernorm import LayerNormConfig
from torchtitan.protocols.module import Module

__all__ = ["LayerNorm"]


class LayerNorm(nn.LayerNorm, Module):
    """Configurable nn.LayerNorm."""

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = LayerNormConfig

    def __init__(self, config: Config):
        assert config.normalized_shape is not None, "partial LayerNorm.Config"
        super().__init__(
            config.normalized_shape,
            eps=config.eps,
            elementwise_affine=config.elementwise_affine,
        )
