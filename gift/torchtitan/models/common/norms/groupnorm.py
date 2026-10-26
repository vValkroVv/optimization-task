import torch.nn as nn

from typing import TypeAlias

from torchtitan.config.schema.models.common.norms.groupnorm import GroupNormConfig
from torchtitan.protocols.module import Module

__all__ = ["GroupNorm"]


class GroupNorm(nn.GroupNorm, Module):
    """Configurable nn.GroupNorm."""

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = GroupNormConfig

    def __init__(self, config: Config):
        super().__init__(
            config.num_groups,
            config.num_channels,
            eps=config.eps,
        )
