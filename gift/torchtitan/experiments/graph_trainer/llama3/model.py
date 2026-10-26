# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch

from typing import TypeAlias

from torchtitan.config.schema.experiments.graph_trainer.llama3.model import (
    GraphTrainerLlama3ModelConfig,
)
from torchtitan.models.llama3 import Llama3Model

from ..simple_fsdp import disable_active_parametrization


class GraphTrainerLlama3Model(Llama3Model):
    # pyrefly: ignore [bad-override]
    Config: TypeAlias = GraphTrainerLlama3ModelConfig

    def __init__(self, config: Config):
        super().__init__(config)

    def init_states(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        with disable_active_parametrization():
            super().init_states(buffer_device=buffer_device)
