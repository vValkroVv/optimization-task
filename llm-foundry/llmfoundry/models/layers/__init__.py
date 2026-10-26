# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from llmfoundry.models.layers.attention import (
    ATTN_CLASS_REGISTRY,
    RING_ATTN_CLASSES,
    MultiheadAttention,
    MultiQueryAttention,
    attn_bias_shape,
    build_alibi_bias,
    build_attn_bias,
    flash_attn_fn,
    scaled_multihead_dot_product_attention,
    triton_flash_attn_fn,
)
from llmfoundry.models.layers.blocks import BLOCK_CLASS_REGISTRY
from llmfoundry.models.layers.custom_embedding import (
    SharedEmbedding,
    VocabParallelEmbedding,
    EmbeddingParallelEmbedding,
    PARALLEL_EMBEDDING_REGISTRY,
    ScaledEmbedding
)
from llmfoundry.models.layers.fc import FC_CLASS_REGISTRY, ColumnParallelLinear, RowParallelLinear
from llmfoundry.models.layers.ffn import FFN_CLASS_REGISTRY, build_ffn
from llmfoundry.models.layers.norm import (
    NORM_CLASS_REGISTRY,
    LlamaRMSNorm,
    resolve_norm_class,
)
from llmfoundry.models.layers.mtp import MTP_CLASS_REGISTRY
from llmfoundry.models.layers.moe.moe_layers import DeepseekGMMMoeBlock


MOE_CLASS_REGISTRY = {
    "DeepseekGMMoeBlock": DeepseekGMMMoeBlock,
}


__all__ = [
    "scaled_multihead_dot_product_attention",
    "flash_attn_fn",
    "triton_flash_attn_fn",
    "MultiheadAttention",
    "MultiQueryAttention",
    "attn_bias_shape",
    "build_attn_bias",
    "build_alibi_bias",
    "ATTN_CLASS_REGISTRY",
    "BLOCK_CLASS_REGISTRY",
    "NORM_CLASS_REGISTRY",
    "resolve_norm_class",
    "FC_CLASS_REGISTRY",
    "SharedEmbedding",
    "FFN_CLASS_REGISTRY",
    "build_ffn",
    "ScaledEmbedding",
    "VocabParallelEmbedding",
    "EmbeddingParallelEmbedding",
    "PARALLEL_EMBEDDING_REGISTRY",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "LlamaRMSNorm",
    "MTP_CLASS_REGISTRY",
    "RING_ATTN_CLASSES",
]
