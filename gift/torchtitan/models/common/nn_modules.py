# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurable wrappers around standard ``torch.nn`` modules.

Each class uses diamond inheritance (``nn.X`` + ``Module``) so the ``nn`` logic is
reused as-is while satisfying the ``Module`` protocol; each ``Config`` exposes only
the fields current callsites set.

Norms live in :mod:`torchtitan.models.common.norms` and embeddings in
:mod:`torchtitan.models.common.embeddings`; import them from there.
"""

import torch.nn as nn

from typing import TypeAlias

from torchtitan.config.schema.models.common.nn_modules import (
    Conv1dConfig,
    Conv2dConfig,
    GELUConfig,
    IdentityConfig,
    LinearConfig,
    SiLUConfig,
)
from torchtitan.protocols.module import Module


class Conv1d(nn.Conv1d, Module):
    """Configurable nn.Conv1d."""

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = Conv1dConfig

    def __init__(self, config: Config):
        super().__init__(
            config.in_channels,
            config.out_channels,
            config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            groups=config.groups,
            bias=config.bias,
        )


class Conv2d(nn.Conv2d, Module):
    """Configurable nn.Conv2d."""

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = Conv2dConfig

    def __init__(self, config: Config):
        super().__init__(
            config.in_channels,
            config.out_channels,
            config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            bias=config.bias,
        )


class GELU(nn.GELU, Module):
    """Configurable nn.GELU."""

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = GELUConfig

    def __init__(self, config: Config):
        super().__init__(approximate=config.approximate)


class Identity(nn.Identity, Module):
    """Configurable nn.Identity."""

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = IdentityConfig

    def __init__(self, config: Config):
        super().__init__()


class Linear(nn.Linear, Module):
    """Configurable nn.Linear."""

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = LinearConfig

    def __init__(self, config: Config):
        in_features, out_features = config.require_features()
        super().__init__(in_features, out_features, bias=config.bias)


class SiLU(nn.SiLU, Module):
    """Configurable nn.SiLU."""

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = SiLUConfig

    def __init__(self, config: Config):
        super().__init__()


__all__ = [
    "Conv1d",
    "Conv2d",
    "GELU",
    "Identity",
    "Linear",
    "SiLU",
]
