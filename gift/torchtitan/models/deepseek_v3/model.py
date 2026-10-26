# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import TypeAlias, cast

import torch.nn as nn

from torchtitan.config.schema.models.common.attention.softmax.mla import MLAConfig
from torchtitan.config.schema.models.deepseek_v3.model import (
    DeepSeekV3ModelConfig,
)
from torchtitan.config.schema.protocols.model import BaseModelConfig
from torchtitan.models.common.decoder_blocks import Decoder
from torchtitan.protocols.model import ModelRuntimeConfig


class DeepSeekV3Model(Decoder):
    """
    DeepSeek-V3 Transformer model with attention and feed-forward layers.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = DeepSeekV3ModelConfig

    @classmethod
    def prepare_config(
        cls,
        model_config: BaseModelConfig,
        *,
        config: ModelRuntimeConfig,
        **kwargs,
    ) -> None:
        model_config = cast(DeepSeekV3ModelConfig, model_config)
        Decoder.prepare_config(model_config, config=config, **kwargs)
        parallelism = config.parallelism
        from torchtitan.models.deepseek_v3.sharding import (
            set_deepseek_v3_sharding_config,
        )

        set_deepseek_v3_sharding_config(
            model_config,
            loss_parallel=not parallelism.disable_loss_parallel,
            enable_sp=parallelism.enable_sequence_parallel,
            enable_ep=parallelism.expert_parallel_degree > 1,
        )

    @classmethod
    def get_nparams_and_flops(
        cls, model_config: BaseModelConfig, model: "nn.Module", seq_len: int
    ) -> tuple[int, int]:
        model_config = cast(DeepSeekV3ModelConfig, model_config)
        from torchtitan.models.utils import get_blockwise_nparams_and_flops

        assert isinstance(model_config.placed_layers()[0].attention, MLAConfig)
        return get_blockwise_nparams_and_flops(model_config, model, seq_len)
