"""Replay Apple CCE with the native GFSDP FP32 weight-gradient slot."""

from types import MethodType

import torch
import triton
from cut_cross_entropy.cce_backward import _cce_backward_kernel
from cut_cross_entropy.cce_lse_forward import cce_lse_forward_kernel
from cut_cross_entropy.tl_utils import b_bin_fn
from torchtitan.distributed.gfsdp._common import TrainingState
from torchtitan.distributed.gfsdp._staged_head_dw import StagedHeadDw


def _accumulate_backward(lease, hidden, weight, targets, lse, dloss):
    """Use Apple's unfiltered backward on the GFSDP-owned output buffer."""
    group, slot = lease.group, lease.slot
    index = lease.writes
    stream = group.device_handle.current_stream()
    if (
        group._staged_head_dw is not lease
        or group._training_state != TrainingState.PRE_BACKWARD
        or index >= lease.num_chunks
        or index != lease.completed_callbacks
        or group.reduce_grads != (index == lease.num_chunks - 1)
        or lease.param.unsharded_param.grad is not None
        or lease.param.unsharded_accumulated_grad is not None
        or stream.device != slot.device
    ):
        raise RuntimeError("Apple CCE did not receive the active GFSDP gradient slot")
    if index == 0:
        if lease.context.owner is not None or lease.context.conflicting_owner(group) is not None:
            raise RuntimeError("GFSDP gradient slot already has an owner")
        lease.context.wait_for_release(stream)
        lease.context.owner = group
        lease.stream = stream
        slot.zero_()
    elif lease.context.owner is not group or lease.stream != stream:
        raise RuntimeError("GFSDP gradient slot changed between Apple CCE chunks")

    rows, width = hidden.shape
    vocab = weight.size(0)
    dx = torch.zeros_like(hidden, dtype=torch.float32)
    lock_columns = triton.cdiv(width, 64)
    dx_locks = hidden.new_zeros((triton.cdiv(rows, 128), lock_columns), dtype=torch.int32)
    dw_locks = weight.new_zeros((triton.cdiv(vocab, 128), lock_columns), dtype=torch.int32)

    def grid(meta):
        return (triton.cdiv(rows, meta["BLOCK_B"]) * triton.cdiv(vocab, meta["BLOCK_V"]),)

    # Apple's exact preset consumes BF16 dZ and accumulates both products in
    # FP32. Here dC aliases the GFSDP slot and retains FP32 across the chunks.
    _cce_backward_kernel[grid](
        hidden,
        weight,
        None,  # bias
        lse,
        dloss,
        1.0,  # normalization is already represented in dloss
        None,  # gradient of the exposed LSE output
        None,  # valid indices, since ignored rows have zero dloss
        None,  # vocabulary ordering
        None,  # softcap
        targets,
        dx,
        None,  # Kahan compensation for dE
        dx_locks,
        slot,
        None,  # Kahan compensation for dC
        dw_locks,
        None,  # bias gradient
        rows,
        width,
        vocab,
        rows,
        *dx_locks.shape,
        *dw_locks.shape,
        *hidden.stride(),
        *weight.stride(),
        1,  # bias stride
        1,  # valid-index stride
        None,  # filter epsilon
        shift=0,
        B_BIN=b_bin_fn(rows),
        FILTER_E_GRAD=False,
        FILTER_C_GRAD=False,
    )
    lease.writes += 1
    return dx.to(hidden.dtype)


class _AppleDirect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, lease):
        if (
            weight is not lease.param.unsharded_param
            or weight.dtype != hidden.dtype
            or weight.dtype != torch.bfloat16
            or not weight.is_contiguous()
            or not hidden.is_contiguous()
            or targets.shape != hidden.shape[:-1]
            or hidden.size(0) % 2
            or lease.slot.shape != weight.shape
            or lease.slot.stride() != weight.stride()
            or lease.slot.dtype != torch.float32
        ):
            raise RuntimeError("Apple CCE requires two folded BF16 heads and an FP32 slot")
        x, labels = hidden.flatten(0, -2), targets.flatten()
        result = cce_lse_forward_kernel(e=x, c=weight, targets=labels)
        valid = labels.ne(-100)
        nll = torch.where(valid, result.neg_correct_logit + result.lse, 0.0)
        ctx.save_for_backward(x, weight, labels, result.lse, valid)
        ctx.hidden_shape, ctx.lease = hidden.shape, lease
        return nll.view(2, -1).sum(dim=1)

    @staticmethod
    def backward(ctx, grad_terms):
        hidden, weight, targets, lse, valid = ctx.saved_tensors
        dloss = torch.where(valid, grad_terms.repeat_interleave(targets.numel() // 2), 0.0)
        dx = _accumulate_backward(ctx.lease, hidden, weight, targets, lse, dloss)
        return dx.view(ctx.hidden_shape), None, None, None


class _FoldedTerms:
    @staticmethod
    def consume_if_supported(terms, labels, weights, denominator):
        if terms.shape != (2,) or len(weights) != 2:
            raise RuntimeError("Apple CCE did not produce both folded MTP terms")
        return terms.unbind()


def install_apple_direct(loss):
    if isinstance(_cce_backward_kernel, triton.runtime.Autotuner):
        raise RuntimeError("Use Apple's fixed launch configuration for the leased dW slot")
    original_loop = loss._chunked_loss_and_grad
    loss._exact_folded_ce = _FoldedTerms()

    def loop(self, hidden, targets, denominator, **kwargs):
        label_chunks = iter(targets.chunk(self.num_chunks, dim=1))
        original_linear = StagedHeadDw.linear

        def linear(lease, x, weight):
            return _AppleDirect.apply(x, weight, next(label_chunks).contiguous(), lease)

        # The head keeps its actual Linear.forward and the native GFSDP hooks.
        StagedHeadDw.linear = linear
        try:
            result = original_loop(hidden, targets, denominator, **kwargs)
            if next(label_chunks, None) is not None:
                raise RuntimeError("Apple CCE did not consume all GFSDP chunks")
            return result
        finally:
            StagedHeadDw.linear = original_linear

    loss._chunked_loss_and_grad = MethodType(loop, loss)
