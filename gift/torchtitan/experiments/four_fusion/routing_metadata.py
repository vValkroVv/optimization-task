"""Explicit CUDA routing metadata kernel for the four-fusion experiment."""

import torch
import triton
import triton.language as tl

from torchtitan.distributed.deepep.routing import (
    gather_route_gradients,
    indices_to_map as _reference_indices_to_map,
)


_MAX_BLOCK_EXPERTS = 1024
_MAX_STATIC_TOP_K = 64
_MAX_GRID_X = (1 << 31) - 1
_MAX_INT64 = (1 << 63) - 1


def _metadata_views(
    routing_storage: torch.Tensor,
    probability_storage: torch.Tensor,
    num_tokens: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_entries = num_tokens * num_experts
    return (
        routing_storage.narrow(0, 0, num_entries).view(num_tokens, num_experts),
        probability_storage.narrow(0, 0, num_entries).view(num_tokens, num_experts),
    )


@triton.jit
def _routing_metadata_kernel(
    expert_ids,
    weights,
    routing_storage,
    probability_storage,
    num_tokens: tl.constexpr,
    top_k: tl.constexpr,
    num_experts: tl.constexpr,
    ids_stride_0: tl.constexpr,
    ids_stride_1: tl.constexpr,
    weights_stride_0: tl.constexpr,
    weights_stride_1: tl.constexpr,
    block_experts: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    experts = tl.arange(0, block_experts).to(tl.int64)
    active_expert = experts < tl.full((), num_experts, tl.int64)

    if num_tokens > 0:
        route = tl.full((block_experts,), False, tl.int1)
        probability = tl.full((block_experts,), 0.0, tl.float32)
        if top_k > 0:
            ids_row_offset = token * tl.full((), ids_stride_0, tl.int64)
            weights_row_offset = token * tl.full((), weights_stride_0, tl.int64)
            for slot in tl.static_range(0, top_k):
                slot_offset = tl.full((), slot, tl.int64)
                expert_id = tl.load(
                    expert_ids
                    + ids_row_offset
                    + slot_offset * tl.full((), ids_stride_1, tl.int64)
                ).to(tl.int64)
                score = tl.load(
                    weights
                    + weights_row_offset
                    + slot_offset * tl.full((), weights_stride_1, tl.int64)
                ).to(tl.float32)
                match = (expert_id >= 0) & (expert_id == experts) & active_expert
                route = route | match
                probability = tl.where(match, score, probability)

        output_offset = token * tl.full((), num_experts, tl.int64) + experts
        tl.store(routing_storage + output_offset, route, mask=active_expert)
        tl.store(
            probability_storage + output_offset,
            probability,
            mask=active_expert,
        )

    if token == 0:
        sink_offset = tl.full((), num_tokens, tl.int64) * tl.full(
            (), num_experts, tl.int64
        )
        tl.store(routing_storage + sink_offset, False)
        tl.store(probability_storage + sink_offset, 0.0)


@torch.library.custom_op(
    "four_fusion::routing_metadata_cuda", mutates_args=(), device_types="cuda"
)
def _routing_metadata_cuda(
    expert_ids: torch.Tensor, weights: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, top_k = expert_ids.shape
    num_entries = num_tokens * num_experts
    routing_storage = torch.empty(
        (num_entries + 1,), dtype=torch.bool, device=expert_ids.device
    )
    probability_storage = torch.empty(
        (num_entries + 1,), dtype=torch.float32, device=expert_ids.device
    )
    block_experts = triton.next_power_of_2(max(num_experts, 1))
    _routing_metadata_kernel[(max(num_tokens, 1),)](
        expert_ids,
        weights,
        routing_storage,
        probability_storage,
        num_tokens,
        top_k,
        num_experts,
        expert_ids.stride(0),
        expert_ids.stride(1),
        weights.stride(0),
        weights.stride(1),
        block_experts,
        num_warps=4,
    )
    return _metadata_views(
        routing_storage, probability_storage, num_tokens, num_experts
    )


@_routing_metadata_cuda.register_fake
def _routing_metadata_cuda_fake(
    expert_ids: torch.Tensor, weights: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    num_entries = expert_ids.shape[0] * num_experts
    routing_storage = torch.empty(
        (num_entries + 1,), dtype=torch.bool, device=expert_ids.device
    )
    probability_storage = torch.empty(
        (num_entries + 1,), dtype=torch.float32, device=expert_ids.device
    )
    return _metadata_views(
        routing_storage, probability_storage, expert_ids.shape[0], num_experts
    )


def _routing_metadata_setup_context(ctx, inputs, output) -> None:
    expert_ids, weights, _ = inputs
    ctx.save_for_backward(expert_ids)
    ctx.weight_dtype = weights.dtype


def _routing_metadata_backward(ctx, grad_routing, grad_probabilities):
    if grad_probabilities is None:
        grad_weights = None
    else:
        (expert_ids,) = ctx.saved_tensors
        grad_weights = gather_route_gradients(grad_probabilities, expert_ids).to(
            ctx.weight_dtype
        )
    return None, grad_weights, None


_routing_metadata_cuda.register_autograd(
    _routing_metadata_backward, setup_context=_routing_metadata_setup_context
)


def _strided_offset_fits_int64(
    num_tokens: int, top_k: int, stride_0: int, stride_1: int
) -> bool:
    if num_tokens == 0 or top_k == 0:
        return True
    max_offset = (num_tokens - 1) * stride_0 + (top_k - 1) * stride_1
    return 0 <= max_offset <= _MAX_INT64


def _can_use_cuda_kernel(
    expert_ids: torch.Tensor, weights: torch.Tensor, num_experts: int
) -> bool:
    if expert_ids.device.type != "cuda" or expert_ids.device != weights.device:
        return False
    if expert_ids.ndim != 2 or weights.ndim != 2 or expert_ids.shape != weights.shape:
        return False
    if expert_ids.dtype not in (torch.int32, torch.int64):
        return False
    if weights.dtype not in (torch.float32, torch.bfloat16):
        return False

    num_tokens, top_k = expert_ids.shape
    if (
        num_experts <= 0
        or num_experts > _MAX_BLOCK_EXPERTS
        or num_tokens > _MAX_GRID_X
        or top_k > _MAX_STATIC_TOP_K
    ):
        return False
    block_experts = triton.next_power_of_2(num_experts)
    num_entries = num_tokens * num_experts
    if block_experts > _MAX_BLOCK_EXPERTS or num_entries >= _MAX_INT64:
        return False
    return _strided_offset_fits_int64(
        num_tokens, top_k, expert_ids.stride(0), expert_ids.stride(1)
    ) and _strided_offset_fits_int64(
        num_tokens, top_k, weights.stride(0), weights.stride(1)
    )


def indices_to_map(
    expert_ids: torch.Tensor, weights: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build routing metadata, using the eager reference outside the CUDA crop.

    CUDA inputs must have matching ``[T, K]`` shapes, signed int32/int64 IDs,
    and FP32/BF16 weights. Every retained ID is required to be in
    ``[0, num_experts)``; retained IDs are distinct per token as produced by
    top-k routing. These data-dependent producer invariants are not checked on
    the host. CUDA execution is bounded to ``num_experts <= 1024``,
    ``top_k <= 64``, the signed CUDA CTA grid limit, and signed int64 input and
    output offsets; CPU and unsupported CUDA shape/dtype cases use the eager
    reference.
    """
    if not _can_use_cuda_kernel(expert_ids, weights, num_experts):
        return _reference_indices_to_map(expert_ids, weights, num_experts)
    return _routing_metadata_cuda(expert_ids, weights, num_experts)
