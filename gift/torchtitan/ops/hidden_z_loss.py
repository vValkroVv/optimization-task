"""FP32 magnitude logsumexp for contiguous BF16 hidden states of width 4096."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _shifted_exp_kernel(Hidden, Exponentials, Maxima, WIDTH: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, WIDTH)
    offsets = row.to(tl.int64) * WIDTH + cols
    magnitude = tl.abs(tl.load(Hidden + offsets).to(tl.float32))
    # A NaN exponential propagates through ATen's sum even if max ignores NaNs.
    maximum = tl.max(magnitude, axis=0)
    # ATen uses zero as the shift for an infinite row maximum.
    shift = tl.where(maximum == float("inf"), 0.0, maximum)
    tl.store(Exponentials + offsets, libdevice.exp(magnitude - shift))
    tl.store(Maxima + row, shift)


@triton.jit
def _magnitude_grad_kernel(Hidden, LogSumExp, Upstream, Grad, WIDTH: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, WIDTH)
    offsets = row.to(tl.int64) * WIDTH + cols
    hidden = tl.load(Hidden + offsets).to(tl.float32)
    lse = tl.load(LogSumExp + row)
    upstream = tl.load(Upstream + row)
    sign = (hidden > 0).to(tl.float32) - (hidden < 0).to(tl.float32)
    magnitude_grad = upstream * libdevice.exp(tl.abs(hidden) - lse)
    tl.store(Grad + offsets, magnitude_grad * sign)


@torch.library.custom_op(
    "torchtitan::magnitude_logsumexp", mutates_args=(), device_types="cuda"
)
def magnitude_logsumexp(hidden: torch.Tensor) -> torch.Tensor:
    width = hidden.shape[-1]
    exp = torch.empty_like(hidden, dtype=torch.float32)
    shift = exp.new_empty(hidden.shape[:-1])
    _shifted_exp_kernel[(shift.numel(),)](hidden, exp, shift, width, num_warps=4)
    # Keep ATen's FP32 summation tree. A Triton sum can change the rounded LSE,
    # which propagates to every activation gradient in the row.
    return exp.sum(dim=-1).log_().add_(shift)


@magnitude_logsumexp.register_fake
def _(hidden):
    return hidden.new_empty(hidden.shape[:-1], dtype=torch.float32)


@torch.library.custom_op(
    "torchtitan::magnitude_logsumexp_backward", mutates_args=(), device_types="cuda"
)
def _magnitude_logsumexp_backward(
    hidden: torch.Tensor, lse: torch.Tensor, upstream: torch.Tensor
) -> torch.Tensor:
    grad = torch.empty_like(hidden)
    _magnitude_grad_kernel[(lse.numel(),)](
        hidden, lse, upstream.contiguous(), grad, hidden.shape[-1], num_warps=4
    )
    return grad


@_magnitude_logsumexp_backward.register_fake
def _(hidden, lse, upstream):
    return torch.empty_like(hidden)


def _setup_context(ctx, inputs, output):
    (hidden,) = inputs
    ctx.save_for_backward(hidden, output)


def _backward(ctx, upstream):
    hidden, lse = ctx.saved_tensors
    if torch.is_grad_enabled():
        # Keep the differentiable ATen expression for higher-order gradients.
        value = hidden.float()
        magnitude_grad = (
            upstream.unsqueeze(-1) * (value.abs() - lse.unsqueeze(-1)).exp()
        )
        return (magnitude_grad * value.sign()).to(hidden.dtype)
    return _magnitude_logsumexp_backward(hidden, lse, upstream)


magnitude_logsumexp.register_autograd(_backward, setup_context=_setup_context)
