"""Multi-Token Prediction (MTP) heads.

Embeddings and the output head are shared with the main model, so each head
owns only its two input norms, a ``2*dim -> dim`` projection, a configurable
:class:`ResidualTransformerBlock`, and a final norm. Heads return hidden
states (not logits); the shared ``lm_head`` is applied per head in ``MTPLoss``.
"""

import torch

from typing import TypeAlias

from torchtitan.config.schema.models.common.decoder_blocks.mtp import (
    MTPBlockConfig,
    MTPConfig,
)
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.protocols.blocks import Block
from torchtitan.protocols.module import Module, ModuleDict


__all__ = ["MTPBlock", "MTP"]


class MTPBlock(Block):
    """One sequential multi-token-prediction head.

    ``forward`` consumes the previous head's hidden state and the (shifted)
    input-token embedding, both ``[B, S, dim]``, and returns the normalized
    hidden state ``[B, S, dim]`` for the shared output head.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = MTPBlockConfig

    def __init__(self, config: Config):
        if (
            config.token_norm is None
            or config.hidden_norm is None
            or config.input_proj is None
            or config.final_norm is None
        ):
            raise ValueError("MTPBlock needs completed norms and input projection")
        super().__init__()
        self.token_norm = config.token_norm.build()
        self.hidden_norm = config.hidden_norm.build()
        self.input_proj = config.input_proj.build()
        self.block = config.block.build()
        self.final_norm = config.final_norm.build()

    def forward(
        self,
        prev_hidden: torch.Tensor,
        token_emb: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Concat order [hidden, embedding]; the state-dict adapter
        # assumes this column layout for ``input_proj``.
        proj_in = torch.cat(
            [self.hidden_norm(prev_hidden), self.token_norm(token_emb)], dim=-1
        )
        h = self.input_proj(proj_in)
        h = self.block(h, rope_cache, attention_masks, positions, token_mask)
        return self.final_norm(h)


class MTP(Module):
    """Container of ``D`` sequential :class:`MTPBlock` heads.

    ``forward`` takes the main trunk hidden state (taken *before* the main
    final norm) and the shared input-token embeddings, and returns stacked
    hidden states ``[D, B, S, dim]`` (one per head). Head ``k`` predicts token
    ``t+1+k``: its embedding input is the input embeddings rolled left by
    ``k+1`` (tail zeroed), and its hidden input chains from head ``k-1``.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = MTPConfig

    def __init__(self, config: Config):
        super().__init__()
        self.layers = ModuleDict()
        for i, module_config in enumerate(config.placed_layers()):
            self.layers[str(i)] = module_config.build()

    def forward(
        self,
        trunk_hidden: torch.Tensor,
        input_embeds: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = []
        prev_hidden = trunk_hidden
        emb = input_embeds
        for module in self.layers.values():
            # Shift left by one, zero-filling the tail: head k sees the embeddings
            # shifted by k+1, whose last k+1 positions have no future token. Built
            # with ``cat`` rather than ``roll`` + an in-place zero of the wrapped
            # tail because ``fill_`` has no DTensor sharding rule, which breaks TP.
            emb = torch.cat([emb[:, 1:], torch.zeros_like(emb[:, :1])], dim=1)
            prev_hidden = module(
                prev_hidden, emb, rope_cache, attention_masks, positions, token_mask
            )
            outputs.append(prev_hidden)
        return torch.stack(outputs, dim=0)
