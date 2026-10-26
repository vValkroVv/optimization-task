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

from torchtitan.config.schema.models.common.attention.softmax.gqa import (
    GQAttentionConfig,
)
from torchtitan.config.schema.models.qwen3.model import (
    Qwen3ModelConfig,
)
from torchtitan.config.schema.protocols.model import BaseModelConfig
from torchtitan.models.common.decoder_blocks import Decoder
from torchtitan.protocols.model import ModelRuntimeConfig


class Qwen3Model(Decoder):
    """
    Qwen3Model Module

    Args:
        config (Qwen3Model.Config): Model configuration.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = Qwen3ModelConfig

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
            # Re-tie before init: on meta device the __init__ tying may
            # not have worked correctly.
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
        model_config = cast(Qwen3ModelConfig, model_config)
        Decoder.prepare_config(model_config, config=config, **kwargs)
        parallelism = config.parallelism
        from torchtitan.models.qwen3.sharding import set_qwen3_sharding_config

        set_qwen3_sharding_config(
            model_config,
            loss_parallel=not parallelism.disable_loss_parallel,
            enable_sp=parallelism.enable_sequence_parallel,
            enable_ep=parallelism.expert_parallel_degree > 1,
        )

    @classmethod
    def get_nparams_and_flops(
        cls, model_config: BaseModelConfig, model: "nn.Module", seq_len: int
    ) -> tuple[int, int]:
        model_config = cast(Qwen3ModelConfig, model_config)
        from torchtitan.models.utils import get_blockwise_nparams_and_flops

        attention = model_config.placed_layers()[0].attention
        assert isinstance(attention, GQAttentionConfig)
        assert attention.head_dim is not None
        return get_blockwise_nparams_and_flops(model_config, model, seq_len)
