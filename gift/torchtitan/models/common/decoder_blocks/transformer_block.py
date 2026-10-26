"""Transformer decoder block: the shared base contract and the concrete
residual realization used by every language model.

``TransformerBlock`` is the abstract contract (sublayer configs + FLOP
accounting). ``ResidualTransformerBlock`` is the concrete pre-norm /
sandwich-norm block that all current decoder models build on.
"""

import torch

from typing import TypeAlias

from torchtitan.config.schema.models.common.decoder_blocks.transformer_block import (
    ResidualTransformerBlockConfig,
    TransformerBlockConfig,
)
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.protocols.blocks import Block

__all__ = ["TransformerBlock", "ResidualTransformerBlock"]


# TODO: we can unify the TransformerBlock impl across all models when
# there is no special logic for each model, including
# ffn vs. moe naming and creation, rope vs. nope, etc.
class TransformerBlock(Block):
    """Abstract base for transformer blocks: sublayer configs (attention, FFN/MoE,
    norms) + FLOP accounting. Children implement ``__init__`` / ``forward``.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = TransformerBlockConfig


class ResidualTransformerBlock(TransformerBlock):
    """Concrete pre-norm / sandwich-norm residual transformer block.

    Pre-norm is the default; setting ``post_attn_norm`` / ``post_ffn_norm`` upgrades
    it to sandwich norm (an extra norm on each sublayer output). Setting
    ``attn_residual_scale`` / ``ffn_residual_scale`` scales that output by a
    step-scheduled ``alpha`` before the residual add. Subclass and override
    ``forward`` for bespoke wiring (e.g. per-layer mask selection).
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = ResidualTransformerBlockConfig

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.post_attn_norm = (
            config.post_attn_norm.build() if config.post_attn_norm is not None else None
        )
        self.post_ffn_norm = (
            config.post_ffn_norm.build() if config.post_ffn_norm is not None else None
        )
        self.attn_residual_scale = (
            config.attn_residual_scale.build()
            if config.attn_residual_scale is not None
            else None
        )
        self.ffn_residual_scale = (
            config.ffn_residual_scale.build()
            if config.ffn_residual_scale is not None
            else None
        )

        # ``moe`` and ``feed_forward`` are distinct checkpoint keys, so keep
        # both attribute names and select at forward time. ``moe_enabled`` is
        # read by the optimizer's MoE load-balancing hook and by parallelize.
        self.moe_enabled = config.moe is not None
        if self.moe_enabled:
            assert config.moe is not None
            self.moe = config.moe.build()
            self.feed_forward = None
        else:
            assert config.feed_forward is not None, (
                "ResidualTransformerBlock requires exactly one of feed_forward / moe."
            )
            self.feed_forward = config.feed_forward.build()
            self.moe = None

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_out = self.attention(
            self.attention_norm(x), rope_cache, attention_masks, positions
        )
        if self.post_attn_norm is not None:
            attn_out = self.post_attn_norm(attn_out)
        if self.attn_residual_scale is not None:
            attn_out = self.attn_residual_scale(attn_out)
        x = x + attn_out

        ffn_in = self.ffn_norm(x)
        ffn_out = (
            self.moe(ffn_in, token_mask)
            if self.moe_enabled
            else self.feed_forward(ffn_in)
        )
        if self.post_ffn_norm is not None:
            ffn_out = self.post_ffn_norm(ffn_out)
        if self.ffn_residual_scale is not None:
            ffn_out = self.ffn_residual_scale(ffn_out)
        return x + ffn_out
