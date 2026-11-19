"""Two BF16 LM-head gradients staged in their existing FP32 GFSDP slot."""

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl
from torch.distributed.tensor import DTensor

from ._common import TrainingState

if TYPE_CHECKING:
    from ._param import GFSDPParam
    from ._param_group import GFSDPBufferContext, GFSDPParamGroup


@triton.jit
def merge_head_dw(staged, output, width: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, width)
    base = row.to(tl.int64) * width
    first = tl.load(staged + 2 * base + column).to(tl.float32)
    second = tl.load(staged + 2 * base + width + column).to(tl.float32)
    # All warps must read both halves before any FP32 store aliases their inputs.
    tl.debug_barrier()
    tl.store(output + base + column, first + second)


class StagedHeadDw:
    """Own the gradient slot from the first chunk write to the final reduction."""

    def __init__(
        self,
        group: "GFSDPParamGroup",
        param: "GFSDPParam",
        context: "GFSDPBufferContext",
        slot: torch.Tensor,
    ) -> None:
        self.group = group
        self.param = param
        self.context = context
        self.slot = slot
        self.staged = slot.view(torch.bfloat16)
        width = slot.size(1)
        self.halves = self.staged[:, :width], self.staged[:, width:]
        self.writes = 0
        self.completed_callbacks = 0
        self.stream: torch.Stream | None = None

    def begin(self) -> None:
        if self.group._staged_head_dw is not None:
            raise RuntimeError("The GFSDP LM head already has a staged dW lease")
        self.group._staged_head_dw = self

    def close(self) -> None:
        # Only cancel before handoff. If reduction itself fails, GFSDP retains
        # ownership for the failed run because a collective may still read it.
        if self.group._staged_head_dw is self:
            self.group._staged_head_dw = None
            if self.stream is not None:
                self.context.release(self.group, self.stream)

    def check_reduced(self) -> None:
        if (
            self.completed_callbacks != len(self.halves)
            or self.context.owner is self.group
        ):
            raise RuntimeError("The GFSDP LM head did not reduce both staged gradients")

    def write(self, grad: torch.Tensor, input: torch.Tensor) -> None:
        group = self.group
        index = self.writes
        if (
            group._staged_head_dw is not self
            or group._training_state != TrainingState.PRE_BACKWARD
            or index >= len(self.halves)
            or index != self.completed_callbacks
            or group.reduce_grads != (index == len(self.halves) - 1)
            or self.param.unsharded_param.grad is not None
            or self.param.unsharded_accumulated_grad is not None
        ):
            raise RuntimeError("Unexpected GFSDP callback state for staged LM-head dW")
        stream = group.device_handle.current_stream()
        if index == 0:
            if (
                self.context.conflicting_owner(group) is not None
                or self.context.owner is not None
            ):
                raise RuntimeError(
                    "The staged LM-head gradient buffer already has an owner"
                )
            self.context.wait_for_release(stream)
            self.context.owner = group
            self.stream = stream
        elif self.context.owner is not group or self.stream != stream:
            raise RuntimeError(
                "The staged LM-head gradient lease changed between chunks"
            )
        torch.mm(grad.t(), input, out=self.halves[index])
        self.writes += 1

    def post_backward(self, was_in_backward: bool) -> bool:
        """Return whether the no-sync callback must retain the staged slot."""
        group = self.group
        if (
            not was_in_backward
            or self.writes != self.completed_callbacks + 1
            or self.context.owner is not group
            or self.stream != group.device_handle.current_stream()
            or group.reshard_after_backward
        ):
            raise RuntimeError("The GFSDP LM head did not finish its staged gradient")
        if not group.reduce_grads:
            self.completed_callbacks += 1
            return True

        if self.writes != len(self.halves):
            raise RuntimeError("Both staged LM-head gradients must precede reduction")
        rows, width = self.slot.shape
        merge_head_dw[(rows,)](self.staged, self.slot, width, num_warps=4)
        self.param.unsharded_accumulated_grad = self.slot
        self.completed_callbacks += 1
        # post_backward now owns the FP32 slot through the existing reduce-scatter.
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
            raise RuntimeError("The staged LM head did not receive its BF16 weight")
        ctx.save_for_backward(input, weight)
        ctx.lease = lease
        return torch.nn.functional.linear(input, weight)

    @staticmethod
    def backward(ctx, grad):
        input, weight = ctx.saved_tensors
        # Match eager Linear's two native mm operations and output dtypes.
        grad = grad.flatten(0, -2)
        dx = torch.mm(grad, weight).view_as(input)
        ctx.lease.write(grad, input.flatten(0, -2))
        return dx, None, None


def select_staged_head_dw(
    head: torch.nn.Module, chunks: list[torch.Tensor]
) -> StagedHeadDw | None:
    from torchtitan.models.common.nn_modules import Linear

    from .fully_shard import GFSDPModule

    if (
        not isinstance(head, GFSDPModule)
        or getattr(type(head), "_gfsdp_orig_cls", None) is not Linear
        or head.bias is not None
        or head._compiled_call_impl is not None
        or not head.training
        or len(chunks) != 2
        or any(
            not x.is_cuda
            or isinstance(x, DTensor)
            or x.dtype != torch.bfloat16
            or x.ndim != 3
            or x.numel() == 0
            or not x.is_contiguous()
            for x in chunks
        )
        or (
            torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") != torch.bfloat16
        )
    ):
        return None
    if "forward" in head.__dict__:
        contract = head._parallelized_forward_contract
        if contract is None:
            return None
        wrapped, original, parallel_dims = contract
        if (
            head.forward is not wrapped
            or getattr(original, "__func__", None) is not Linear.forward
            or parallel_dims.tp_enabled
            or parallel_dims.cp_enabled
            or parallel_dims.pp_enabled
        ):
            return None
    state = head._get_fsdp_state()
    groups = state._fsdp_param_groups
    if (
        state._is_root is None
        or state._modules != (head,)
        or not state._own_grad_buffer
        or len(groups) != 1
    ):
        return None
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
        return None
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
        or slot.ndim != 2
        or slot.numel() == 0
        or slot.dtype != torch.float32
        or not slot.is_contiguous()
        or reduction_slot is None
        or slot.numel() != reduction_slot.numel()
        or slot.data_ptr() != reduction_slot.data_ptr()
        or not weight.requires_grad
        or weight._backward_hooks
        or weight._post_accumulate_grad_hooks
        or weight.grad is not None
        or param.unsharded_accumulated_grad is not None
    ):
        return None
    width = slot.size(1)
    # One four-warp CTA holds one complete row across the in-place barrier.
    # Bound its register footprint to the measured production row width.
    if not 128 <= width <= 4096 or width & (width - 1):
        return None
    if any(x.size(-1) != width or x.device != slot.device for x in chunks):
        return None

    return StagedHeadDw(group, param, context, slot)
