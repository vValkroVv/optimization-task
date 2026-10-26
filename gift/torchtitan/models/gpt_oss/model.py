# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import TypeAlias, cast

from torchtitan.config.schema.components.lora import unwrap_frozen

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import BlockMask, and_masks

from torchtitan.config.schema.models.gpt_oss.model import (
    AttentionConfig,
    GptOssModelConfig,
    GptOssTransformerBlockConfig,
)
from torchtitan.config.schema.protocols.model import BaseModelConfig
from torchtitan.models.common.attention import (
    AttentionMasksType,
    FlexAttention,
    MaskType,
    SoftmaxAttention,
    create_attention_mask,
    get_causal_mask_mod,
    get_document_mask_mod,
    get_sliding_window_mask_mod,
)
from torchtitan.models.common.decoder_blocks import Decoder, ResidualTransformerBlock
from torchtitan.models.common.embeddings.rope import apply_rotary_emb_cos_sin
from torchtitan.protocols.model import ModelRuntimeConfig


class Attention(SoftmaxAttention):
    """
    Multi-head attention (MLA) module with sink attention.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = AttentionConfig

    def __init__(self, config: Config):
        super().__init__()
        self.head_dim = config.head_dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.enable_gqa = self.n_heads > self.n_kv_heads

        self.n_rep = self.n_heads // self.n_kv_heads

        # Standard attention softmax scale (1/sqrt(head_dim))
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)

        assert config.qkv_linear is not None and config.wo is not None, (
            "partial Attention.Config"
        )
        self.qkv_linear = config.qkv_linear.build()
        self.wo = config.wo.build()
        self.sinks = nn.Parameter(torch.empty(config.n_heads))
        assert isinstance(
            unwrap_frozen(config.inner_attention), FlexAttention.Config
        ), "gpt-oss only supports FlexAttention"
        self.inner_attention = config.inner_attention.build()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies for rope embedding.
            attention_masks: Attention mask (BlockMask).
            positions: Optional position indices (unused, for API compatibility).

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()

        q, k, v = self.qkv_linear(x)

        q, k = apply_rotary_emb_cos_sin(q, k, freqs_cis, positions)

        assert isinstance(attention_masks, BlockMask), attention_masks
        # FlexAttention handles transpose internally; returns (bs, seq, heads, dim)
        # and lse as (bs, seq, heads)
        output, lse = self.inner_attention(
            q,
            k,
            v,
            attention_masks=attention_masks,
            scale=self.softmax_scale,
            return_lse=True,
            enable_gqa=self.enable_gqa,
        )

        # Apply attention sink rescaling: rescale by sigma(lse - w[h])
        # output: (bs, seq, heads, dim), lse: (bs, seq, heads)
        sink_scale = torch.sigmoid(lse - self.sinks.view(1, 1, -1)).unsqueeze(-1)
        output = output * sink_scale.to(output.dtype)

        # Reshape and project output
        output = output.reshape(
            bsz, seqlen, -1
        ).contiguous()  # (bsz, seqlen, n_heads * v_head_dim)
        output = self.wo(output)  # (bsz, seqlen, dim)
        return output


class GptOssTransformerBlock(ResidualTransformerBlock):
    """
    GptOss Transformer block with sliding window attention support.

    Inherits the standard residual ``__init__`` from ``ResidualTransformerBlock``
    and only overrides ``forward`` to pick this layer's attention mask
    (sliding-window vs full) before delegating to the usual residual wiring.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = GptOssTransformerBlockConfig

    def __init__(self, config: Config):
        super().__init__(config)
        self.use_sliding_attention = config.use_sliding_attention
        attention = unwrap_frozen(config.attention)
        assert isinstance(attention, Attention.Config)
        self.sliding_window_size = attention.sliding_window_size

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Extract the appropriate mask for this layer
        assert isinstance(attention_masks, dict)
        if self.use_sliding_attention:
            layer_mask = attention_masks.get(
                f"sliding_window_{self.sliding_window_size}",
                attention_masks["sliding_window_mask"],
            )
        else:
            layer_mask = attention_masks.get("basic_mask", None)
        assert layer_mask is not None

        return super().forward(x, rope_cache, layer_mask, positions, token_mask)


class GptOssModel(Decoder):
    """
    GPT-OSS Transformer model with attention and feed-forward layers.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = GptOssModelConfig

    def __init__(self, config: Config):
        super().__init__(config)

    def get_attention_masks(
        self,
        positions: torch.Tensor,
    ) -> AttentionMasksType:
        basic_mask_mods = []
        attn_cfg = self.config.attention
        assert isinstance(attn_cfg, Attention.Config)
        seq_len = positions.shape[1]
        match attn_cfg.mask_type:
            case MaskType.CAUSAL:
                B = 1
                basic_mask_mods.append(get_causal_mask_mod())
            case MaskType.BLOCK_CAUSAL:
                B = positions.shape[0]
                basic_mask_mods.append(get_document_mask_mod(positions))
            case _:
                raise ValueError(f"Unknown attention mask type: {attn_cfg.mask_type}")

        # create basic attention mask: causal or block_causal
        basic_mask = create_attention_mask(
            and_masks(*basic_mask_mods),
            B,
            None,
            seq_len,
            seq_len,
        )

        masks = {"basic_mask": basic_mask}
        sizes = dict.fromkeys(
            cast(Attention.Config, unwrap_frozen(block.attention)).sliding_window_size
            for _, block, _, _ in self.config.traverse(GptOssTransformerBlock.Config)
            if block.use_sliding_attention
        )
        for size in sizes:
            mask_mod = and_masks(*basic_mask_mods, get_sliding_window_mask_mod(size))
            masks[f"sliding_window_{size}"] = create_attention_mask(
                mask_mod, B, None, seq_len, seq_len
            )
        masks["sliding_window_mask"] = (
            masks[f"sliding_window_{next(iter(sizes))}"] if sizes else basic_mask
        )
        if len(sizes) == 1:
            del masks[f"sliding_window_{next(iter(sizes))}"]
        return masks

    @classmethod
    def prepare_config(
        cls,
        model_config: BaseModelConfig,
        *,
        config: ModelRuntimeConfig,
        **kwargs,
    ) -> None:
        model_config = cast(GptOssModelConfig, model_config)
        Decoder.prepare_config(model_config, config=config, **kwargs)
        parallelism = config.parallelism
        from torchtitan.models.gpt_oss.sharding import set_gpt_oss_sharding_config

        set_gpt_oss_sharding_config(
            model_config,
            loss_parallel=not parallelism.disable_loss_parallel,
            enable_sp=parallelism.enable_sequence_parallel,
            enable_ep=parallelism.expert_parallel_degree > 1,
        )

    @classmethod
    def get_nparams_and_flops(
        cls, model_config: BaseModelConfig, model: "nn.Module", seq_len: int
    ) -> tuple[int, int]:
        model_config = cast(GptOssModelConfig, model_config)
        from torchtitan.models.utils import get_blockwise_nparams_and_flops

        assert isinstance(model_config.placed_layers()[0].attention, AttentionConfig)
        return get_blockwise_nparams_and_flops(model_config, model, seq_len)
