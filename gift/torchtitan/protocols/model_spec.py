# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch.nn as nn
from torch.distributed.pipelining.schedules import _PipelineSchedule

from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter

# Type aliases for ModelSpec callables
ParallelizeFunction: TypeAlias = Callable[..., nn.Module]
PipeliningFunction: TypeAlias = Callable[
    ..., tuple[_PipelineSchedule, list[nn.Module], bool, bool]
]
FragmentFunction: TypeAlias = Callable[..., list[nn.Module]]
PostOptimizerBuildFn: TypeAlias = Callable[..., None]


@dataclass
class ModelSpec:
    """Per-model bundle. Contains already-selected arch config + callables."""

    name: str
    flavor: str
    model: BaseModel.Config
    parallelize_fn: Callable
    pipelining_fn: Callable | None
    post_optimizer_build_fn: Callable | None
    state_dict_adapter: type[BaseStateDictAdapter] | None

    def to_dict(self) -> dict[str, Any]:
        """Diagnostic identity and complete runtime architecture, excluding hooks.

        Internal/converted model trees are inspection data, not config YAML.
        """
        from torchtitan.config.configurable import to_plain

        return {
            "name": self.name,
            "flavor": self.flavor,
            "model": to_plain(self.model, require_reloadable=False),
        }
