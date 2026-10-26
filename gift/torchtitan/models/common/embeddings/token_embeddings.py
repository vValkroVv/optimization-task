import torch.nn as nn

from typing import TypeAlias

from torchtitan.config.schema.models.common.embeddings.token_embeddings import (
    EmbeddingConfig,
)
from torchtitan.protocols.module import Module

__all__ = ["Embedding"]


class Embedding(nn.Embedding, Module):
    """Configurable nn.Embedding (token -> hidden state lookup table)."""

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = EmbeddingConfig

    def __init__(self, config: Config):
        num_embeddings, embedding_dim = config.require_dimensions()
        super().__init__(num_embeddings, embedding_dim)
