# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import TypeAlias, cast

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import and_masks

from torchtitan.config.schema.components.lora import unwrap_frozen
from torchtitan.config.schema.models.common.attention.softmax.gqa import (
    GQAttentionConfig,
)

from torchtitan.config.schema.models.llama4.model import (
    Llama4ModelConfig,
    Llama4TransformerBlockConfig,
)
from torchtitan.config.schema.protocols.model import BaseModelConfig
from torchtitan.models.common.attention import (
    AttentionMasksType,
    MaskType,
    create_attention_mask,
    get_causal_mask_mod,
    get_document_mask_mod,
    get_fixed_block_mask_mod,
)
from torchtitan.models.common.decoder_blocks import Decoder, ResidualTransformerBlock
from torchtitan.protocols.model import ModelRuntimeConfig


class Llama4TransformerBlock(ResidualTransformerBlock):
    """
    Llama4 TransformerBlock Module

    Args:
        config (Llama4TransformerBlock.Config): Block configuration.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = Llama4TransformerBlockConfig

    def __init__(self, config: Config):
        super().__init__(config)
        self.fixed_attn_block_size = config.fixed_attn_block_size
        attention = unwrap_frozen(config.attention)
        assert isinstance(attention, GQAttentionConfig)
        self.use_rope = attention.use_rope

    def forward(
        self, x, rope_cache, attention_masks=None, positions=None, token_mask=None
    ):
        if isinstance(attention_masks, dict):
            key = f"rope_{self.fixed_attn_block_size}" if self.use_rope else "nope"
            attention_masks = (
                attention_masks.get(key, attention_masks["rope"])
                if self.use_rope
                else attention_masks[key]
            )
        return super().forward(x, rope_cache, attention_masks, positions, token_mask)


class Llama4Model(Decoder):
    """
    Llama4Model Module

    Args:
        config (Llama4Model.Config): Model configuration.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = Llama4ModelConfig

    def get_attention_masks(
        self,
        positions: torch.Tensor,
    ) -> AttentionMasksType:
        mask_mods = [get_causal_mask_mod()]
        attn_config = self.config.attention
        match attn_config.mask_type:
            case MaskType.CAUSAL:
                B = 1
            case MaskType.BLOCK_CAUSAL:
                mask_mods.append(get_document_mask_mod(positions))
                B = positions.shape[0]
            case _:
                raise ValueError(
                    f"Unknown attention mask type: {attn_config.mask_type}"
                )

        nope_mask_mod = and_masks(*mask_mods)

        seqlen = positions.shape[1]
        masks = {
            "nope": create_attention_mask(nope_mask_mod, B, None, seqlen, seqlen),
        }
        sizes = dict.fromkeys(
            block.fixed_attn_block_size
            for _, block, _, _ in self.config.traverse(Llama4TransformerBlock.Config)
            if cast(GQAttentionConfig, unwrap_frozen(block.attention)).use_rope
        )
        for size in sizes:
            mask_mod = and_masks(*mask_mods, get_fixed_block_mask_mod(size))
            masks[f"rope_{size}"] = create_attention_mask(
                mask_mod, B, None, seqlen, seqlen
            )
        masks["rope"] = masks[f"rope_{next(iter(sizes))}"] if sizes else masks["nope"]
        if len(sizes) == 1:
            del masks[f"rope_{next(iter(sizes))}"]
        return masks

    @classmethod
    def prepare_config(
        cls,
        model_config: BaseModelConfig,
        *,
        config: ModelRuntimeConfig,
        **kwargs,
    ) -> None:
        model_config = cast(Llama4ModelConfig, model_config)
        Decoder.prepare_config(model_config, config=config, **kwargs)
        parallelism = config.parallelism
        from torchtitan.models.llama4.sharding import set_llama4_sharding_config

        set_llama4_sharding_config(
            model_config,
            loss_parallel=not parallelism.disable_loss_parallel,
            enable_sp=parallelism.enable_sequence_parallel,
            enable_ep=parallelism.expert_parallel_degree > 1,
        )

    @classmethod
    def get_nparams_and_flops(
        cls, model_config: BaseModelConfig, model: "nn.Module", seq_len: int
    ) -> tuple[int, int]:
        model_config = cast(Llama4ModelConfig, model_config)
        from torchtitan.models.utils import get_blockwise_nparams_and_flops

        return get_blockwise_nparams_and_flops(model_config, model, seq_len)
