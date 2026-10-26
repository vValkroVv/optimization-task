# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


from typing import TypeAlias, cast

import torch
import torch.nn as nn

from torchtitan.config.schema.models.llama3.model import (
    Llama3ModelConfig,
)
from torchtitan.config.schema.protocols.model import BaseModelConfig
from torchtitan.models.common.decoder_blocks import Decoder
from torchtitan.protocols.model import ModelRuntimeConfig


class Llama3Model(Decoder):
    """
    Llama3Model Module

    Args:
        config (Llama3Model.Config): Model configuration.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = Llama3ModelConfig

    def __init__(self, config: Config):
        super().__init__(config)
        self.enable_weight_tying = config.enable_weight_tying

        if self.enable_weight_tying:
            self.tok_embeddings.weight = self.lm_head.weight

    def init_states(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        if self.enable_weight_tying:
            # Re-tie weights before parameter init so that tok_embeddings.weight
            # (left as built by ``Skip()``) and output.weight point to the same
            # tensor after output is initialized.
            assert self.tok_embeddings is not None and self.lm_head is not None
            self.tok_embeddings.weight = self.lm_head.weight

        super().init_states(buffer_device=buffer_device)

    @classmethod
    def prepare_config(
        cls,
        model_config: BaseModelConfig,
        *,
        config: ModelRuntimeConfig,
        **kwargs,
    ) -> None:
        model_config = cast(Llama3ModelConfig, model_config)
        Decoder.prepare_config(model_config, config=config, **kwargs)
        parallelism = config.parallelism
        from torchtitan.models.llama3.sharding import set_llama3_sharding_config

        set_llama3_sharding_config(
            model_config,
            loss_parallel=not parallelism.disable_loss_parallel,
            enable_sp=parallelism.enable_sequence_parallel,
        )

    @classmethod
    def get_nparams_and_flops(
        cls, model_config: BaseModelConfig, model: "nn.Module", seq_len: int
    ) -> tuple[int, int]:
        model_config = cast(Llama3ModelConfig, model_config)
        from torchtitan.models.utils import get_blockwise_nparams_and_flops

        return get_blockwise_nparams_and_flops(model_config, model, seq_len)
