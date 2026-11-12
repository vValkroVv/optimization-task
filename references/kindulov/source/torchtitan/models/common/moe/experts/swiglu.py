# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SwiGLU used by BF16 grouped experts."""

import torch
import torch.nn.functional as F

__all__ = ["fused_swiglu"]


def _swiglu(x: torch.Tensor, probs: torch.Tensor | None, limit: float) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    if limit > 0:
        gate = gate.clamp(max=limit)
        up = up.clamp(-limit, limit)
    output = F.silu(gate) * up
    if probs is not None:
        output = output * probs
    return output.to(x.dtype)


class _SwiGLU(torch.autograd.Function):
    """Recompute pointwise intermediates in backward; retain only projections."""

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx, x: torch.Tensor, probs: torch.Tensor | None, limit: float
    ) -> torch.Tensor:
        ctx.save_for_backward(x, probs)
        ctx.limit = limit
        return _swiglu(x, probs, limit)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output: torch.Tensor):
        x, probs = ctx.saved_tensors
        create_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            # Use separate graph targets even when probs was derived from x.
            x = x.view_as(x)
            probs = None if probs is None else probs.view_as(probs)
            output = _swiglu(x, probs, ctx.limit)
            inputs = [v for v in (x, probs) if v is not None and v.requires_grad]
            gradients = iter(
                torch.autograd.grad(
                    output, inputs, grad_output, create_graph=create_graph
                )
            )
        return (
            *(
                next(gradients) if v is not None and v.requires_grad else None
                for v in (x, probs)
            ),
            None,
        )


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

    Eager backward recomputes the pointwise expression from the original
    projections instead of retaining clamped and activated intermediates.
    Compiled execution leaves buffer planning and fusion to the compiler.
    Neither path changes model parameters or checkpoint state.
    """
    if swiglu_limit > 0 and clip_counts is not None:
        # Counters belong to forward only, never to backward recomputation.
        with torch.no_grad():
            gate, up = x.chunk(2, dim=-1)
            clip_counts[0].add_(gate.ge(swiglu_limit).sum(dtype=torch.int64))
            clip_counts[1].add_(
                up.le(-swiglu_limit)
                .logical_or(up.ge(swiglu_limit))
                .sum(dtype=torch.int64)
            )
    if torch.compiler.is_compiling():
        # Keep compiled expert graphs free of nested autograd execution.
        return _swiglu(x, probs, swiglu_limit)
    return _SwiGLU.apply(x, probs, swiglu_limit)
