"""Token de-embedding (the LM head / output projection): final hidden
``(B, L, D)`` -> vocab logits ``(B, L, vocab_size)``.

A :class:`...nn_modules.Linear` with optional tanh soft-capping of the logits.
"""

import torch

from typing import TypeAlias

from torchtitan.config.schema.models.common.embeddings.token_debeddings import (
    UnembeddingConfig,
)
from torchtitan.models.common.nn_modules import Linear
from torchtitan.models.common.regularization import apply_softcap

__all__ = ["Unembedding"]


class Unembedding(Linear):
    """Output projection from hidden dim to vocab logits.

    ``logit_softcap > 0`` bounds the logits to ``[-cap, cap]`` via
    ``cap * tanh(logits / cap)``, keeping late-training logit growth from
    saturating the softmax. Capping here rather than in ``Decoder.forward`` is
    deliberate: ``ChunkedCELoss`` applies this module itself, per sequence chunk,
    so the cap has to travel with the module to reach the training path.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = UnembeddingConfig

    def __init__(self, config: Config):
        if config.logit_softcap < 0.0:
            raise ValueError(f"logit_softcap must be >= 0, got {config.logit_softcap}.")
        super().__init__(config)
        self.logit_softcap = config.logit_softcap

    # The argument must stay named ``input``: the lm_head ShardingConfig keys its
    # in_src / in_dst placements on that name (decoder_blocks/sharding.py).
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return apply_softcap(super().forward(input), self.logit_softcap)
