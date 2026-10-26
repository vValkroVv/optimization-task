"""Runtime boundaries require the dimensions and children supplied by owners."""

import pytest
import torch

from torchtitan.config.schema.models.common.embeddings.token_embeddings import (
    EmbeddingConfig,
)
from torchtitan.config.schema.models.flux.model.layers import (
    DoubleStreamBlockConfig,
    SingleStreamBlockConfig,
)
from torchtitan.config.schema.models.qwen3_vl.vision_encoder import (
    PatchMergerConfig,
    VisionAttentionConfig,
    VisionMLPConfig,
    VisionRotaryEmbeddingConfig,
)


@pytest.mark.parametrize(
    ("config", "field"),
    [
        (EmbeddingConfig(), "Embedding.num_embeddings"),
        (DoubleStreamBlockConfig(), "DoubleStreamBlock.num_heads"),
        (SingleStreamBlockConfig(), "SingleStreamBlock.hidden_size"),
        (PatchMergerConfig(), "PatchMerger.spatial_merge_size"),
        (VisionAttentionConfig(), "VisionAttention.dim"),
        (VisionMLPConfig(hidden_dim=16), "VisionMLP.fc1"),
        (VisionRotaryEmbeddingConfig(), "VisionRotaryEmbedding.dim"),
    ],
)
def test_partial_child_build_reports_owner_completion(config, field):
    with pytest.raises(ValueError, match=field + " must be completed"):
        config.build()


@pytest.mark.parametrize("block", [DoubleStreamBlockConfig, SingleStreamBlockConfig])
def test_completed_flux_blocks_build_the_declared_dimensions(block):
    config = block(hidden_size=32, num_heads=4)
    with torch.device("meta"):
        built = config.build()
    assert built.hidden_size == 32
    assert built.num_heads == 4


def test_completed_embedding_preserves_dimensions():
    with torch.device("meta"):
        built = EmbeddingConfig(num_embeddings=17, embedding_dim=8).build()
    assert built.weight.shape == (17, 8)
