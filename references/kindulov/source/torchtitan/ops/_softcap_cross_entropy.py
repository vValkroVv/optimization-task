"""CUDA row kernels for softcap + CE; imported only by the CUDA runtime path."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _capped(raw, CAP: tl.constexpr):
    value = raw.to(tl.float32)
    if CAP > 0:
        value = CAP * libdevice.tanh(value / CAP)
        # Match Unembedding -> CE: softcap rounds to the logits dtype first.
        value = value.to(raw.dtype).to(tl.float32)
    return value


@triton.jit
def forward_kernel(
    X, Y, LOSS, LSE, V: tl.constexpr, CAP: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    target = tl.load(Y + row)
    if target == -100:
        tl.store(LOSS + row, 0.0)
        tl.store(LSE + row, 0.0)
        return
    tl.device_assert((target >= 0) & (target < V), "CE target outside vocabulary")
    maximum = float("-inf")
    denominator = 0.0
    for start in range(0, V, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        raw = tl.load(X + row * V + cols, cols < V, other=0.0)
        values = tl.where(cols < V, _capped(raw, CAP), float("-inf"))
        next_maximum = tl.maximum(maximum, tl.max(values, 0))
        denominator = denominator * tl.exp(maximum - next_maximum) + tl.sum(
            tl.exp(values - next_maximum), 0
        )
        maximum = next_maximum
    lse = maximum + tl.log(denominator)
    chosen = tl.load(X + row * V + target, (target >= 0) & (target < V), other=0.0)
    tl.store(LOSS + row, lse - _capped(chosen, CAP))
    tl.store(LSE + row, lse)


@triton.jit
def backward_kernel(
    X, Y, LSE, SCALE, DX, V: tl.constexpr, CAP: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    target = tl.load(Y + row)
    lse = tl.load(LSE + row)
    scale = tl.load(SCALE).to(tl.float32)
    for start in range(0, V, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        raw = tl.load(X + row * V + cols, cols < V, other=0.0)
        values = _capped(raw, CAP)
        grad = (tl.exp(values - lse) - tl.where(cols == target, 1.0, 0.0)) * scale
        if CAP > 0:
            # Preserve the gradient cast at the eager BF16 softcap/CE boundary.
            grad = grad.to(raw.dtype).to(tl.float32)
            tangent = libdevice.tanh(raw.to(tl.float32) / CAP)
            grad = ((grad * CAP) * (1.0 - tangent * tangent)) / CAP
        grad = tl.where(target == -100, 0.0, grad)
        tl.store(DX + row * V + cols, grad, cols < V)
