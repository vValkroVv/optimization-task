# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from Kindulov's BSD-licensed cap-zero row CE kernels at
# b983803337dfcc34fdd571b38aaf5e01df583bbe (see root LICENSE).
"""Stable first-order SUM CE; keep the row maximum separate from log(sum(exp))."""

from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.distributed.tensor import DTensor


@triton.jit
def forward_kernel(
    X, Y, LOSS, MAXIMUM, LOG_DENOMINATOR, V: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    target = tl.load(Y + row)
    if target == -100:
        tl.store(LOSS + row, 0.0)
        tl.store(MAXIMUM + row, 0.0)
        tl.store(LOG_DENOMINATOR + row, 0.0)
        return
    tl.device_assert((target >= 0) & (target < V), "CE target outside vocabulary")
    maximum = float("-inf")
    denominator = 0.0
    for start in range(0, V, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        values = tl.load(X + row * V + cols, cols < V, other=float("-inf")).to(
            tl.float32
        )
        next_maximum = tl.maximum(maximum, tl.max(values, 0))
        denominator = denominator * tl.exp(maximum - next_maximum) + tl.sum(
            tl.exp(values - next_maximum), 0
        )
        maximum = next_maximum
    log_denominator = tl.log(denominator)
    chosen = tl.load(X + row * V + target, (target >= 0) & (target < V), other=0.0).to(
        tl.float32
    )
    tl.store(LOSS + row, -((chosen - maximum) - log_denominator))
    tl.store(MAXIMUM + row, maximum)
    tl.store(LOG_DENOMINATOR + row, log_denominator)


@triton.jit
def backward_kernel(
    X,
    Y,
    MAXIMUM,
    LOG_DENOMINATOR,
    SCALE,
    DX,
    V: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS_PER_HEAD: tl.constexpr = 0,
    ROW_OFFSET: tl.constexpr = 0,
    HEAD_WEIGHTS: tl.constexpr = None,
    HOST_RECIPROCAL: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    target = tl.load(Y + row)
    maximum = tl.load(MAXIMUM + row)
    log_denominator = tl.load(LOG_DENOMINATOR + row)
    if HOST_RECIPROCAL:
        scale = SCALE.to(tl.float32)
    else:
        scale = tl.load(SCALE).to(tl.float32)
    if HEAD_WEIGHTS is not None:
        head = (row + ROW_OFFSET) // ROWS_PER_HEAD
        weight = 0.0
        for index in tl.static_range(len(HEAD_WEIGHTS)):
            weight = tl.where(head == index, HEAD_WEIGHTS[index], weight)
        if HOST_RECIPROCAL:
            scale = weight * scale
        else:
            scale = tl.div_rn(weight, scale)
    for start in range(0, V, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        values = tl.load(X + row * V + cols, cols < V, other=0.0).to(tl.float32)
        # Subtract the target and apply incoming normalization before one dtype cast.
        grad = (
            tl.exp((values - maximum) - log_denominator)
            - tl.where(cols == target, 1.0, 0.0)
        ) * scale
        grad = tl.where(target == -100, 0.0, grad)
        tl.store(DX + row * V + cols, grad, cols < V)


@torch.library.custom_op("four_fusion::stable_ce_forward", mutates_args=())
def _forward(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    losses = torch.empty(logits.shape[0], device=logits.device, dtype=torch.float32)
    maximum = torch.empty_like(losses)
    log_denominator = torch.empty_like(losses)
    forward_kernel[(logits.shape[0],)](
        logits,
        labels,
        losses,
        maximum,
        log_denominator,
        logits.shape[1],
        4096,
        num_warps=4,
        enable_fp_fusion=False,
        debug=True,
    )
    return losses.sum(), maximum, log_denominator


@_forward.register_fake
def _forward_fake(logits, labels):
    return (
        logits.new_empty((), dtype=torch.float32),
        logits.new_empty((logits.shape[0],), dtype=torch.float32),
        logits.new_empty((logits.shape[0],), dtype=torch.float32),
    )


def _launch_backward(
    logits: torch.Tensor,
    labels: torch.Tensor,
    maximum: torch.Tensor,
    log_denominator: torch.Tensor,
    scale: torch.Tensor,
    grad: torch.Tensor,
) -> None:
    backward_kernel[(logits.shape[0],)](
        logits,
        labels,
        maximum,
        log_denominator,
        scale,
        grad,
        logits.shape[1],
        4096,
        num_warps=4,
        enable_fp_fusion=False,
    )


@torch.library.custom_op("four_fusion::stable_ce_backward", mutates_args=())
def _backward(
    logits: torch.Tensor,
    labels: torch.Tensor,
    maximum: torch.Tensor,
    log_denominator: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    grad = torch.empty_like(logits)
    _launch_backward(logits, labels, maximum, log_denominator, scale, grad)
    return grad


@_backward.register_fake
def _backward_fake(logits, labels, maximum, log_denominator, scale):
    return torch.empty_like(logits)


def _setup_context(ctx, inputs, output):
    logits, labels = inputs
    _, maximum, log_denominator = output
    ctx.save_for_backward(logits, labels, maximum, log_denominator)
    ctx.mark_non_differentiable(maximum, log_denominator)


def _autograd_backward(ctx, scale, _maximum_grad, _log_denominator_grad):
    if torch.is_grad_enabled():
        raise ValueError("Stable CE supports first-order gradients only")
    return _backward(*ctx.saved_tensors, scale), None


_forward.register_autograd(_autograd_backward, setup_context=_setup_context)


@torch.library.custom_op("four_fusion::stable_ce_folded_backward", mutates_args=())
def _folded_backward(
    logits: torch.Tensor,
    labels: torch.Tensor,
    maxima: Sequence[torch.Tensor],
    log_denominators: Sequence[torch.Tensor],
    scales: Sequence[torch.Tensor],
) -> torch.Tensor:
    # Each launch receives an output view whose pointer starts at its head's
    # first row. The LM head consumes the full allocation directly.
    grad = torch.empty_like(logits)
    _write_folded_backward(logits, labels, maxima, log_denominators, scales, grad)
    return grad


def _write_folded_backward(logits, labels, maxima, log_denominators, scales, grad):
    num_heads = len(scales)
    for inputs in zip(
        logits.chunk(num_heads),
        labels.chunk(num_heads),
        maxima,
        log_denominators,
        scales,
        grad.chunk(num_heads),
        strict=True,
    ):
        _launch_backward(*inputs)


@_folded_backward.register_fake
def _folded_backward_fake(logits, labels, maxima, log_denominators, scales):
    return torch.empty_like(logits)


class _FoldedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, num_heads):
        return _save_folded_statistics(ctx, logits, labels, num_heads, _forward)

    @staticmethod
    def backward(ctx, *scales):
        if torch.is_grad_enabled():
            raise ValueError("Stable CE supports first-order gradients only")
        logits, labels, *stats = ctx.saved_tensors
        num_heads = len(scales)
        grad = _folded_backward(
            logits, labels, stats[:num_heads], stats[num_heads:], list(scales)
        )
        return grad, None, None


def _save_folded_statistics(ctx, logits, labels, num_heads, forward):
    # Keep the existing per-head launch shapes and FP32 sum reductions.
    losses, maxima, log_denominators = zip(
        *(
            forward(head_logits, head_labels)
            for head_logits, head_labels in zip(
                logits.chunk(num_heads), labels.chunk(num_heads), strict=True
            )
        ),
        strict=True,
    )
    ctx.save_for_backward(logits, labels, *maxima, *log_denominators)
    return losses


class _ConsumeFoldedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, num_heads, forward):
        return _save_folded_statistics(ctx, logits, labels, num_heads, forward)

    @staticmethod
    def backward(ctx, *scales):
        if torch.is_grad_enabled():
            raise ValueError("Stable CE supports first-order gradients only")
        logits, labels, *stats = ctx.saved_tensors
        num_heads = len(scales)
        # Each program reads and overwrites one row. The staged Linear saved only
        # its input and weight, and the loss reads argmax before calling backward.
        _write_folded_backward(
            logits, labels, stats[:num_heads], stats[num_heads:], scales, logits
        )
        torch.autograd.graph.increment_version(logits)
        return logits, None, None, None


def supports_folded_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> bool:
    return (
        not isinstance(logits, DTensor)
        and not isinstance(labels, DTensor)
        and logits.is_cuda
        and logits.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and logits.numel() > 0
        and logits.is_contiguous()
        and labels.is_contiguous()
    )


def _flatten_folded_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, num_heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if labels.dtype != torch.int64:
        raise ValueError("Cross-entropy targets must have dtype int64")
    if logits.shape[:-1] != labels.shape or logits.device != labels.device:
        raise ValueError(
            "Cross-entropy logits and targets must have compatible shapes and device"
        )
    if num_heads < 1 or logits.shape[0] % num_heads:
        raise ValueError("Folded CE requires equal batch rows per head")
    return logits.view(-1, logits.shape[-1]), labels.view(-1)


def folded_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, num_heads: int
) -> tuple[torch.Tensor, ...]:
    """Compute a sum per head and return one contiguous logit gradient."""
    flat, labels = _flatten_folded_cross_entropy(logits, labels, num_heads)
    return _FoldedCrossEntropy.apply(flat, labels, num_heads)


def make_consuming_folded_cross_entropy(
    backend: str | Callable | None,
) -> Callable[[torch.Tensor, torch.Tensor, int], tuple[torch.Tensor, ...]]:
    # Compile the statistics boundary and keep the one-shot storage mutation
    # in eager backward, outside the compiler's functionalization.
    forward = _forward
    if backend is not None:
        forward = torch.compile(_forward, backend=backend, fullgraph=True)

    def consume(logits, labels, num_heads):
        if logits.dtype != torch.bfloat16:
            raise ValueError("Consuming Folded CE requires BF16 logits")
        flat, labels = _flatten_folded_cross_entropy(logits, labels, num_heads)
        return _ConsumeFoldedCrossEntropy.apply(flat, labels, num_heads, forward)

    return consume


def cross_entropy_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Local cap-zero CE; DTensor retains the native distributed implementation."""
    if isinstance(pred, DTensor) or isinstance(labels, DTensor):
        from torchtitan.components.loss import cross_entropy_loss as native_ce

        return native_ce(pred, labels)
    if labels.dtype != torch.int64:
        raise ValueError("Cross-entropy targets must have dtype int64")
    if pred.shape[:-1] != labels.shape or pred.device != labels.device:
        raise ValueError(
            "Cross-entropy logits and targets must have compatible shapes and device"
        )
    flat = pred.reshape(-1, pred.shape[-1])
    targets = labels.reshape(-1)
    if (
        pred.is_cuda
        and flat.numel()
        and pred.dtype in (torch.bfloat16, torch.float16, torch.float32)
    ):
        return _forward(flat.contiguous(), targets.contiguous())[0]
    return F.cross_entropy(flat.float(), targets, reduction="sum", ignore_index=-100)
