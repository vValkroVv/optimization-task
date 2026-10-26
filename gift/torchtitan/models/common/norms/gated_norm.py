"""Self-gated RMSNorm.

RMS-normalizes the input, then multiplies by a learned low-rank self-gate computed
from the normalized output: ``out = norm(x) * gating_weight * sigmoid(down(silu(up(norm(x)))))``.
``zero_centered`` switches the affine weight from ``weight`` to ``1 + weight`` (so a
zeros-initialized weight starts as identity).
"""

from typing import TypeAlias

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.config.schema.models.common.norms.gated_norm import GatedRMSNormConfig
from torchtitan.protocols.module import Module

__all__ = ["GatedRMSNorm"]


def _gated_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_down_weight: torch.Tensor,
    *,
    eps: float,
    gating_weight: float,
    zero_centered: bool,
) -> torch.Tensor:
    n = x.float()
    n = n * torch.rsqrt(n.pow(2).mean(-1, keepdim=True) + eps)
    weight = weight.float()
    n = n * (1.0 + weight if zero_centered else weight)
    gate = F.silu(F.linear(n, gate_up_weight.float()))
    gate = torch.sigmoid(F.linear(gate, gate_down_weight.float()))
    return (n * gating_weight * gate).to(x.dtype)


class GatedRMSNorm(Module):
    """RMSNorm with a learned low-rank self-gate (standard or zero-centered)."""

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = GatedRMSNormConfig

    def __init__(self, config: Config):
        super().__init__()
        assert config.normalized_shape is not None, "partial GatedRMSNorm.Config"
        self.eps = config.eps
        self.gating_weight = config.gating_weight
        self.zero_centered = config.zero_centered
        self.norm_backend = config.norm_backend
        if self.norm_backend == "cute":
            from .cute_gated_norm import gated_norm

            self._cute_gated_norm = gated_norm
        self.weight = nn.Parameter(torch.empty(config.normalized_shape))
        self.gate_up_weight = nn.Parameter(
            torch.empty(config.gate_rank, config.normalized_shape)
        )
        self.gate_down_weight = nn.Parameter(
            torch.empty(config.normalized_shape, config.gate_rank)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_backend == "cute":
            # Form 1+w in fp32, before the kernel rounds tensor-core operands.
            weight = 1.0 + self.weight.float() if self.zero_centered else self.weight
            return self._cute_gated_norm(
                x,
                weight,
                self.gate_up_weight,
                self.gate_down_weight,
                eps=self.eps,
                gating_weight=self.gating_weight,
            )
        return _gated_rms_norm(
            x,
            self.weight,
            self.gate_up_weight,
            self.gate_down_weight,
            eps=self.eps,
            gating_weight=self.gating_weight,
            zero_centered=self.zero_centered,
        )
