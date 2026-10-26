"""Sigmoid output gate for softmax attention.

Projects a per-(head·head_dim) gate from the attention layer input and multiplies
it into the attention output (in fp32) before the output projection:
``out = attn_output * sigmoid(gate_proj(x))``.
"""

import torch

from typing import TypeAlias

from torchtitan.config.schema.models.common.attention.softmax.gate import (
    SigmoidGateConfig,
)
from torchtitan.protocols.module import Module

__all__ = ["SigmoidGate"]


class SigmoidGate(Module):
    # pyrefly: ignore [bad-override]
    Config: TypeAlias = SigmoidGateConfig

    def __init__(self, config: Config):
        super().__init__()
        assert config.gate_proj is not None, "partial SigmoidGate.Config"
        self.gate_proj = config.gate_proj.build()

    def forward(self, x: torch.Tensor, attn_output: torch.Tensor) -> torch.Tensor:
        # Gate in fp32, cast back to the attention output dtype.
        gate = torch.sigmoid(self.gate_proj(x).float())
        return (attn_output.float() * gate).to(attn_output.dtype)
