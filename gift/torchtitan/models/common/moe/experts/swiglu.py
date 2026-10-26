# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SwiGLU used by BF16 grouped experts."""

import torch
import torch.nn.functional as F

__all__ = ["fused_swiglu"]


def fused_swiglu(
    x: torch.Tensor,
    probs: torch.Tensor | None = None,
    swiglu_limit: float = 0.0,
    clip_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply SwiGLU, optional routing scores, and DeepSeek-style clipping.

    When ``swiglu_limit > 0``, gate is upper-clipped and up is symmetrically
    clipped. If supplied, ``clip_counts`` accumulates the corresponding gate
    and up clipping counts in-place.

    ``probs`` must be broadcastable to the output, normally ``(tokens, 1)``.
    Passing ``None`` leaves routing scores to the token dispatcher.
    """
    gate, up = torch.chunk(x, 2, dim=-1)
    if swiglu_limit > 0:
        if clip_counts is not None:
            clip_counts[0].add_(gate.ge(swiglu_limit).sum(dtype=torch.int64))
            clip_counts[1].add_(
                up.le(-swiglu_limit)
                .logical_or(up.ge(swiglu_limit))
                .sum(dtype=torch.int64)
            )
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(-swiglu_limit, swiglu_limit)
    output = F.silu(gate) * up
    if probs is not None:
        output = output * probs
    return output.to(x.dtype)
