"""Pointwise-fused softcap without logits-sized FP32 saved tensors."""

from collections.abc import Callable
from typing import cast

import torch
from torch.distributed.tensor import DTensor


_tanh_backward = cast(
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor], torch.ops.aten.tanh_backward
)


def _forward(x: torch.Tensor, cap: float) -> torch.Tensor:
    return (cap * torch.tanh(x.float() / cap)).to(x.dtype)


def _backward(x: torch.Tensor, grad: torch.Tensor, cap: float) -> torch.Tensor:
    # Preserve the eager backward's operation order, including both scalar ops.
    tanh = torch.tanh(x.float() / cap)
    result = _tanh_backward(grad.float() * cap, tanh) / cap
    return result.to(x.dtype)


# Compile forward and backward separately so AOT autograd does not retain a
# full FP32 tanh output. Only the original logits survive until backward.
_fused_forward = torch.compile(_forward, fullgraph=True, dynamic=True)
_fused_backward = torch.compile(_backward, fullgraph=True, dynamic=True)


class _Softcap(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, x: torch.Tensor, cap: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.cap = cap
        return _fused_forward(x, cap)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad: torch.Tensor):
        (x,) = ctx.saved_tensors
        # Higher derivatives need a visible differentiable expression.
        fn = _backward if torch.is_grad_enabled() else _fused_backward
        return fn(x, grad, ctx.cap), None


def softcap(x: torch.Tensor, cap: float) -> torch.Tensor:
    """Fuse local CUDA pointwise work; preserve eager CPU/distributed behavior."""
    if cap == 0.0:
        return x
    if (
        x.device.type == "cuda"
        and not isinstance(x, DTensor)
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
    ):
        return _Softcap.apply(x, cap)
    return _forward(x, cap)
