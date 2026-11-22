"""Lease the native GFSDP FP32 LM-head gradient slot across loss chunks."""

import logging
import os
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from ._common import TrainingState
from ._staged_head_dw_writer import make_head_dw_writer

if TYPE_CHECKING:
    from ._param import GFSDPParam
    from ._param_group import GFSDPBufferContext, GFSDPParamGroup


logger = logging.getLogger("gfsdp")
_MODE_ENV = "GIFT_EXPERIMENT_LM_HEAD_MATRIX"
_CHUNKS_ENV = "GIFT_EXPERIMENT_LM_HEAD_MATRIX_CHUNKS"
_HEAD_SHAPE = (128256, 4096)
_ROWS_PER_BATCH = 16384


class StagedHeadDw:
    def __init__(
        self,
        group: "GFSDPParamGroup",
        param: "GFSDPParam",
        context: "GFSDPBufferContext",
        slot: torch.Tensor,
        mode: str,
        num_chunks: int,
    ) -> None:
        self.group = group
        self.param = param
        self.context = context
        self.slot = slot
        self.mode = mode
        self.num_chunks = num_chunks
        self.writes = 0
        self.completed_callbacks = 0
        self.stream: torch.Stream | None = None

    def begin(self) -> None:
        if self.group._staged_head_dw is not None:
            raise RuntimeError("The GFSDP LM head already has a direct dW lease")
        self.group._staged_head_dw = self

    def close(self) -> None:
        # Once handed off, the native reduction owns the slot and its event.
        if self.group._staged_head_dw is self:
            self.group._staged_head_dw = None
            if self.stream is not None and self.context.owner is self.group:
                self.context.release_event = self.stream.record_event()
                self.context.owner = None

    def check_reduced(self) -> None:
        group = self.group
        if (
            self.completed_callbacks != self.num_chunks
            or group._staged_head_dw is not None
            or self.context.owner is group
            or group._post_reduce_event is None
            or self.param.sharded_param.grad is not self.param.sharded_param_grad
        ):
            raise RuntimeError("GFSDP did not reduce the complete direct LM-head dW")
        if not group._staged_head_dw_reduced:
            group._staged_head_dw_reduced = True
            if dist.get_rank() == 0:
                logger.info(
                    "LM_HEAD_MATRIX_DW_REDUCED rank=0 mode=%s chunks=%d "
                    "slot=GFSDP.reduce_scatter_input dtype=%s shape=%s",
                    self.mode,
                    self.num_chunks,
                    self.slot.dtype,
                    tuple(self.slot.shape),
                )

    def write(self, grad: torch.Tensor, hidden: torch.Tensor) -> None:
        group = self.group
        index = self.writes
        if (
            group._staged_head_dw is not self
            or group._training_state != TrainingState.PRE_BACKWARD
            or index >= self.num_chunks
            or index != self.completed_callbacks
            or group.reduce_grads != (index == self.num_chunks - 1)
            or self.param.unsharded_param.grad is not None
            or self.param.unsharded_accumulated_grad is not None
        ):
            raise RuntimeError("Unexpected GFSDP callback state for direct LM-head dW")
        if (
            grad.dtype != torch.bfloat16
            or hidden.dtype != torch.bfloat16
            or grad.device != self.slot.device
            or hidden.device != self.slot.device
            or not grad.is_contiguous()
            or not hidden.is_contiguous()
            or grad.size(0) != hidden.size(0)
            or (grad.size(1), hidden.size(1)) != self.slot.shape
        ):
            raise ValueError(
                "Direct LM-head dW requires contiguous BF16 operands matching its FP32 slot"
            )

        stream = group.device_handle.current_stream()
        if stream.device != self.slot.device:
            raise RuntimeError("Direct LM-head dW must run on the slot's CUDA device")
        if index == 0:
            if (
                self.context.owner is not None
                or self.context.conflicting_owner(group) is not None
            ):
                raise RuntimeError(
                    "The direct LM-head gradient buffer already has an owner"
                )
            self.context.wait_for_release(stream)
            self.context.owner = group
            self.stream = stream
        elif self.context.owner is not group or self.stream != stream:
            raise RuntimeError(
                "The direct LM-head gradient lease changed between chunks"
            )

        writer = group._staged_head_dw_writer
        if writer is None:
            vocab, width = self.slot.shape
            writer = make_head_dw_writer(
                self.mode, self.num_chunks, hidden.size(0), vocab, width, stream
            )
            group._staged_head_dw_writer = writer
            logger.info(
                "LM_HEAD_MATRIX_DW_WRITER rank=%d mode=%s chunks=%d writer=%s "
                "pair=%s version=%s slot=GFSDP.reduce_scatter_input dtype=%s shape=%s",
                dist.get_rank(),
                self.mode,
                self.num_chunks,
                writer.name,
                getattr(writer, "pair", None),
                getattr(writer, "version", None),
                self.slot.dtype,
                tuple(self.slot.shape),
            )
        writer(grad, hidden, self.slot, index != 0, stream)
        self.writes += 1

    def post_backward(self, was_in_backward: bool) -> bool:
        """Return whether the no-sync callback keeps the slot leased."""
        group = self.group
        if (
            not was_in_backward
            or self.writes != self.completed_callbacks + 1
            or self.context.owner is not group
            or self.stream != group.device_handle.current_stream()
            or group.reshard_after_backward
        ):
            raise RuntimeError("GFSDP did not complete the direct LM-head dW write")
        self.completed_callbacks += 1
        if not group.reduce_grads:
            return True
        if self.writes != self.num_chunks:
            raise RuntimeError("Every direct LM-head write must precede reduction")

        # The optimizer-facing sharded gradient can already hold earlier
        # microbatches. Native reduce_scatter adds this microbatch to that buffer.
        self.param.unsharded_accumulated_grad = self.slot
        group._staged_head_dw = None
        return False

    def linear(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return _StagedHeadLinear.apply(input, weight, self)


class _StagedHeadLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, lease):
        if (
            weight is not lease.param.unsharded_param
            or weight.dtype != torch.bfloat16
            or not weight.is_contiguous()
        ):
            raise RuntimeError("The direct LM head did not receive its BF16 weight")
        ctx.save_for_backward(input, weight)
        ctx.lease = lease
        return torch.nn.functional.linear(input, weight)

    @staticmethod
    def backward(ctx, grad):
        input, weight = ctx.saved_tensors
        grad = grad.flatten(0, -2)
        dx = torch.mm(grad, weight).view_as(input)
        ctx.lease.write(grad, input.flatten(0, -2))
        return dx, None, None


def select_staged_head_dw(
    head: torch.nn.Module, chunks: list[torch.Tensor]
) -> StagedHeadDw | None:
    from torchtitan.models.common.nn_modules import Linear

    from .fully_shard import GFSDPModule

    mode = os.environ.get(_MODE_ENV)
    if not mode:
        return None
    if mode not in ("previous", "new"):
        raise ValueError(f"{_MODE_ENV} must be previous or new")
    expected_chunks = os.environ.get(_CHUNKS_ENV)
    if expected_chunks not in ("2", "8") or len(chunks) != int(expected_chunks):
        raise ValueError(f"{_CHUNKS_ENV} must match the actual 2 or 8 LM-head chunks")

    rows_per_chunk = _ROWS_PER_BATCH // len(chunks)
    if (
        not isinstance(head, GFSDPModule)
        or getattr(type(head), "_gfsdp_orig_cls", None) is not Linear
        or head.bias is not None
        or head._compiled_call_impl is not None
        or not head.training
        or not torch.is_grad_enabled()
        or any(
            not x.is_cuda
            or isinstance(x, DTensor)
            or x.dtype != torch.bfloat16
            or x.ndim != 3
            or x.size(0) * x.size(1) != rows_per_chunk
            or not x.requires_grad
            or not x.is_contiguous()
            for x in chunks
        )
        or (
            torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") != torch.bfloat16
        )
    ):
        raise RuntimeError(
            "Direct LM-head dW requires the targeted eager GFSDP Linear and BF16 chunks"
        )
    if "forward" in head.__dict__:
        contract = head._parallelized_forward_contract
        if contract is None:
            raise RuntimeError(
                "Direct LM-head dW requires the original Linear forward wrapper"
            )
        wrapped, original, parallel_dims = contract
        if (
            head.forward is not wrapped
            or getattr(original, "__func__", None) is not Linear.forward
            or parallel_dims.tp_enabled
            or parallel_dims.cp_enabled
            or parallel_dims.pp_enabled
        ):
            raise RuntimeError(
                "Direct LM-head dW requires the original Linear forward with TP/CP/PP disabled"
            )
    state = head._get_fsdp_state()
    groups = state._fsdp_param_groups
    if (
        state._is_root is None
        or state._modules != (head,)
        or not state._own_grad_buffer
        or len(groups) != 1
    ):
        raise RuntimeError(
            "Direct LM-head dW requires one initialized GFSDP head with its own gradient buffer"
        )
    group = groups[0]
    context = group.grad_buffer_ctx
    if (
        len(group.fsdp_params) != 1
        or not group.reduce_grads
        or group.orig_dtype != torch.float32
        or group.param_dtype != torch.bfloat16
        or group.reduce_dtype != torch.float32
        or group.mesh_info.mesh.ndim != 1
        or context is None
        or context.yccl_handles.get(None) is not None
    ):
        raise RuntimeError(
            "Direct LM-head dW requires synchronized one-dimensional NCCL GFSDP with an FP32 gradient slot"
        )
    param = group.fsdp_params[0]
    info = param._module_info
    slot = param._unsharded_accumulated_grad
    reduction_slot = group._reduce_scatter_input.get(None)
    weight = param.unsharded_param
    if (
        info.module is not head
        or info.param_name != "weight"
        or info.shared_modules
        or param.is_dtensor
        or slot is None
        or slot.shape != _HEAD_SHAPE
        or slot.dtype != torch.float32
        or not slot.is_contiguous()
        or reduction_slot is None
        or slot.numel() != reduction_slot.numel()
        or slot.data_ptr() != reduction_slot.data_ptr()
        or any(x.size(-1) != slot.size(1) or x.device != slot.device for x in chunks)
        or not weight.requires_grad
        or weight._backward_hooks
        or weight._post_accumulate_grad_hooks
        or weight.grad is not None
        or param.unsharded_accumulated_grad is not None
    ):
        raise RuntimeError(
            "Direct LM-head dW requires the targeted weight and its actual reduce-scatter input"
        )

    return StagedHeadDw(group, param, context, slot, mode, len(chunks))
