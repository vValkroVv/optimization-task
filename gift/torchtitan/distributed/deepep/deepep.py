# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
DeepEP primitives for MoE Expert Parallel.

Provides low-level functions and autograd wrappers for DeepEP communication.
Used by DeepEPTokenDispatcher in token_dispatcher.py.
"""

from dataclasses import dataclass

import torch
from torch.distributed import ProcessGroup
from torch.utils._python_dispatch import _disable_current_modes

from torchtitan.distributed.deepep.fp8_state import (
    FP8DispatchState as BaseFP8DispatchState,
)
from torchtitan.ops.float8.fp8_tensor import Float8BlockwiseQTensor, InputLayoutType

try:
    # DeepEP HEAD re-exports all three at the top level; older releases
    # (e.g. v1.2.1) keep EventHandle/EventOverlap under deep_ep.utils.
    try:
        from deep_ep import Buffer, EventHandle, EventOverlap
    except ImportError:
        from deep_ep import Buffer
        from deep_ep.utils import EventHandle, EventOverlap
except ImportError as e:
    raise ImportError(
        "DeepEP is required for this module. "
        "Install from: https://github.com/deepseek-ai/deepep"
    ) from e

# Global buffer (single buffer per process, recreated if group changes)
_buffer: Buffer = None

# Global cache for dispatch handles, keyed by handle_id
# SAC saves the handle_id tensor; we use it to retrieve the non-tensor handle
_handle_cache: dict = {}
_handle_counter: int = 0

# Pending combine event for deferred synchronization.
# Stores the EventOverlap from buffer.combine() to allow overlapping
# shared_experts computation with combine communication.
# This is process-local state (each GPU process has its own Python interpreter),
# and execution is single-threaded, so a simple module variable suffices.
_pending_combine_event: EventOverlap | None = None


_ASYNC_FINISH = True


def _record_stream(*tensors: object) -> None:
    """Tie DeepEP outputs to the stream that consumes them."""
    stream = torch.cuda.current_stream()
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            tensor.record_stream(stream)
        elif isinstance(tensor, (tuple, list)):
            _record_stream(*tensor)


def _get_next_handle_id() -> torch.Tensor:
    """Generate a unique handle_id tensor on CPU to avoid GPU-CPU sync."""
    global _handle_counter
    _handle_counter += 1
    return torch.tensor([_handle_counter], dtype=torch.int64, device="cpu")


# ============================================================================
# Custom Op Registration for SAC Integration
# ============================================================================

_lib = torch.library.Library("deepep", "DEF")

_lib.define(
    "dispatch(Tensor x, Tensor topk_idx, Tensor topk_weights, "
    "Tensor num_tokens_per_rank, Tensor num_tokens_per_rdma_rank, "
    "Tensor is_token_in_rank, Tensor num_tokens_per_expert, "
    "bool is_float8_training) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor)"
)

# combine returns: combined_x
_lib.define(
    "combine(Tensor x, Tensor handle_id, bool is_float8_training=False) -> Tensor"
)


@torch.library.impl(_lib, "dispatch", "CUDA")
def _dispatch_op_impl(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_tokens_per_rank: torch.Tensor,
    num_tokens_per_rdma_rank: torch.Tensor,
    is_token_in_rank: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    use_fp8_dispatch: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Execute DeepEP dispatch."""
    global _buffer
    buffer = _buffer
    assert buffer is not None, "Buffer must be initialized before dispatch"
    if use_fp8_dispatch:
        from torchtitan.ops.float8.triton_kernels.row_quantization import (
            row_quantization_1x128_fn,
        )

        x = row_quantization_1x128_fn(x)
    previous_event = EventOverlap(EventHandle()) if _ASYNC_FINISH else None
    (
        recv_x,
        recv_indices,
        recv_probs,
        recv_num_tokens_per_expert_list,
        handle,
        after_event,
    ) = buffer.dispatch(
        x=x,
        topk_idx=topk_idx,
        topk_weights=topk_weights.to(torch.float32),
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        previous_event=previous_event,
        async_finish=_ASYNC_FINISH,
        allocate_on_comm_stream=_ASYNC_FINISH,
    )
    if _ASYNC_FINISH:
        after_event.current_stream_wait()
    _record_stream(recv_x, recv_indices, recv_probs)

    handle_id = _get_next_handle_id()
    _handle_cache[handle_id.item()] = handle

    recv_num_tokens_per_expert = torch.tensor(
        recv_num_tokens_per_expert_list, dtype=torch.int32, device="cpu"
    )
    if use_fp8_dispatch:
        recv_x = Float8BlockwiseQTensor(
            data=recv_x[0],
            scale=recv_x[1],
            orig_dtype=torch.bfloat16,
            input_layout_type=InputLayoutType.ROWWISE,
            block_scaling_dim=1,
            scale_block_first=False,
        )
    return (
        recv_x,
        recv_indices,
        recv_probs,
        recv_num_tokens_per_expert,
        handle_id,
    )


def _dispatch_setup_context(ctx, inputs, output):
    x = inputs[0]
    handle_id = output[4]
    ctx.input_dtype = x.dtype
    ctx.saved_handle = _handle_cache.get(handle_id.item())


def _dispatch_backward(
    ctx,
    grad_recv_x,
    grad_recv_indices,
    grad_recv_probs,
    grad_recv_num_tokens_per_expert,
    grad_handle_id,
):
    """Backward for dispatch: performs combine on gradients."""
    global _buffer
    if grad_recv_x is None:
        return None, None, None, None, None, None, None, None

    # Use handle saved in setup_context instead of from cache
    handle = ctx.saved_handle
    assert handle is not None
    grad_recv_x = grad_recv_x.contiguous()
    topk_weights = (
        grad_recv_probs.float().contiguous() if grad_recv_probs is not None else None
    )
    previous_event = EventOverlap(EventHandle()) if _ASYNC_FINISH else None
    grad_x, grad_probs, after_event = _buffer.combine(
        x=grad_recv_x,
        handle=handle,
        topk_weights=topk_weights,
        previous_event=previous_event,
        async_finish=_ASYNC_FINISH,
        allocate_on_comm_stream=_ASYNC_FINISH,
    )
    if _ASYNC_FINISH:
        after_event.current_stream_wait()
    _record_stream(grad_x, grad_probs)
    # combine op involves weighted sum on float() so need to convert back
    grad_x = grad_x.to(ctx.input_dtype)
    grad_topk_weights = (
        grad_probs.to(ctx.input_dtype) if grad_probs is not None else None
    )
    return (
        grad_x,  # x
        None,  # topk_idx
        grad_topk_weights,  # topk_weights
        None,  # num_tokens_per_rank
        None,  # num_tokens_per_rdma_rank
        None,  # is_token_in_rank
        None,  # num_tokens_per_expert
        None,  # is_float8_training
    )


@torch.library.impl(_lib, "combine", "CUDA")
def _combine_op_impl(
    x: torch.Tensor,
    handle_id: torch.Tensor,
    use_fp8_dispatch: bool = False,
) -> torch.Tensor:
    """Execute DeepEP combine."""
    global _buffer, _pending_combine_event

    buffer = _buffer
    assert buffer is not None, "Buffer must be initialized before combine"

    # In inference mode, setup_context doesn't run, so we clean up handle_cache here.
    # NOTE: For inference, use torch.inference_mode() instead of torch.no_grad()
    if torch.is_inference_mode_enabled():
        handle = _handle_cache.pop(handle_id.item(), None)
    else:
        handle = _handle_cache.get(handle_id.item())
    assert handle is not None, f"Handle not found for handle_id={handle_id.item()}"

    previous_event = EventOverlap(EventHandle()) if _ASYNC_FINISH else None

    combined, _, after_event = buffer.combine(
        x=x,
        handle=handle,
        previous_event=previous_event,
        async_finish=_ASYNC_FINISH,
        allocate_on_comm_stream=_ASYNC_FINISH,
    )

    # Not ordering-sensitive: this registers the consuming stream on the block,
    # and the allocator inserts the guarding event when the block is freed.
    _record_stream(combined)

    # Store event for deferred sync instead of syncing immediately.
    # This enables overlapping shared_experts computation with combine communication.
    # The caller MUST call sync_combine() before using the returned tensor.
    _pending_combine_event = after_event

    return combined


def _combine_setup_context(ctx, inputs, output):
    _, handle_id, use_fp8_dispatch = inputs
    # Pop handle from cache and save it for backward
    ctx.saved_handle = _handle_cache.pop(handle_id.item(), None)
    ctx.use_fp8_dispatch = use_fp8_dispatch


def _combine_backward(ctx, grad_combined):
    """Backward for combine: performs dispatch on gradients."""
    global _buffer
    handle = ctx.saved_handle
    assert handle is not None, "Handle not found in combine backward"
    if ctx.use_fp8_dispatch:
        from torchtitan.ops.float8.triton_kernels.row_quantization import (
            row_quantization_1x128_fn,
        )

        grad_combined = row_quantization_1x128_fn(grad_combined.contiguous())
    else:
        grad_combined = grad_combined.contiguous()
    previous_event = EventOverlap(EventHandle()) if _ASYNC_FINISH else None
    grad_x, _, _, _, _, after_event = _buffer.dispatch(
        x=grad_combined,
        topk_idx=None,
        topk_weights=None,
        num_tokens_per_rank=None,
        num_tokens_per_rdma_rank=None,
        is_token_in_rank=None,
        num_tokens_per_expert=None,
        handle=handle,
        previous_event=previous_event,
        async_finish=_ASYNC_FINISH,
        allocate_on_comm_stream=_ASYNC_FINISH,
    )
    if _ASYNC_FINISH:
        after_event.current_stream_wait()
    _record_stream(grad_x)
    if ctx.use_fp8_dispatch:
        assert isinstance(grad_x, (tuple, list)), (
            f"fp8 dispatch must return a (data, scale) pair; got {type(grad_x)}."
        )
        grad_x = Float8BlockwiseQTensor(
            data=grad_x[0],
            scale=grad_x[1],
            orig_dtype=torch.bfloat16,
            input_layout_type=InputLayoutType.ROWWISE,
            block_scaling_dim=1,
            scale_block_first=False,
        )
    return grad_x, None, None


torch.library.register_autograd(
    "deepep::dispatch", _dispatch_backward, setup_context=_dispatch_setup_context
)
torch.library.register_autograd(
    "deepep::combine", _combine_backward, setup_context=_combine_setup_context
)


def sync_combine() -> None:
    """Synchronize the current CUDA stream with the pending combine operation.

    This function MUST be called before using the result of combine_tokens()
    to ensure the async combine has completed. It inserts a wait operation
    on the current CUDA stream, making subsequent CUDA kernels wait for
    the combine to finish.

    torch.compile Compatibility:
        Guarded with is_compiling() to skip in compile mode.
        This avoids issues with CUDA event operations not being traceable.

    Process Isolation:
        Each GPU process has its own Python interpreter, so this module-level
        variable is inherently process-local. No cross-process interference.

    Single-Threaded Execution:
        PyTorch training is single-threaded per process, so no thread safety
        concerns. Sequential execution guarantees correct event ordering.

    Activation Checkpointing Compatibility:
        - During forward: combine stores event, sync_combine() waits on it
        - During AC recomputation: combine runs again, stores NEW event,
          sync_combine() waits on the new event
        - Sequential execution ensures each forward/recompute uses its own event

    Multiple MoE Layers:
        Each layer's combine overwrites the pending event. Since sync_combine()
        is called before using each layer's output (and before the next layer's
        combine), this is safe. The sync clears the event to prevent double-sync.

    Safe to call multiple times - subsequent calls are no-ops if the event
    was already synced or if no combine operation is pending.
    """
    global _pending_combine_event
    # is_compiling() is True under Dynamo (aot mode) — skip to avoid tracing
    # through global state Dynamo can't handle. Under make_fx (aot_fx_trace),
    # is_compiling() is False so this guard doesn't fire, but the function is
    # still safe: _pending_combine_event is None during tracing (no real
    # combine ran), so the body below is a no-op.
    if torch.compiler.is_compiling():
        return

    if _pending_combine_event is not None:
        if _ASYNC_FINISH:
            _pending_combine_event.current_stream_wait()
        _pending_combine_event = None


def get_hidden_bytes(x: torch.Tensor) -> int:
    """Calculate the number of hidden bytes for one token."""
    if isinstance(x, Float8BlockwiseQTensor):
        t = x._data
    elif isinstance(x, tuple):
        t = x[0]
    else:
        t = x
    return t.size(1) * max(t.element_size(), 2)


def set_num_sms(num_sms: int) -> None:
    """Set the SM budget for DeepEP's dispatch / combine kernels."""
    Buffer.set_num_sms(num_sms)


def get_buffer(group: ProcessGroup, hidden_bytes: int) -> Buffer:
    """Get or create a buffer for all-to-all communication."""
    global _buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes
        )
    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


def _permute_tokens(
    hidden_states: torch.Tensor,
    dispatched_indices: torch.Tensor,
    dispatched_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert dispatch output to grouped_mm format with permutation and token expansion.

    Each token may be routed to multiple experts (top-k), so tokens are expanded and sorted
    by expert ID for efficient grouped matrix multiplication. num_recv_tokens is the number
    of unique tokens received, while num_all_tokens is the expanded count after replication.

    Args:
        hidden_states: Received tokens [num_recv_tokens, hidden_dim]
        dispatched_indices: Expert indices for each token [num_recv_tokens, topk], -1 means masked
        dispatched_probs: Routing probs [num_recv_tokens, topk]

    Returns:
        permuted_hidden_states: Tokens sorted by expert [num_all_tokens, hidden_dim]
        permuted_probs: Routing probs in same order [num_all_tokens]
        permuted_indices: Original token indices for unpermutation [num_all_tokens]
    """
    mask = dispatched_indices != -1
    valid_expert_ids = dispatched_indices[mask]  # 1d tensor
    valid_probs = dispatched_probs[mask]
    # Repeat each token by its valid count and select tokens in expert order
    sort_order = torch.argsort(valid_expert_ids, stable=True)
    permuted_indices = torch.arange(
        len(hidden_states), device=hidden_states.device
    ).repeat_interleave(mask.sum(dim=1))[sort_order]
    permuted_hidden_states = hidden_states.index_select(0, permuted_indices)
    permuted_probs = valid_probs[sort_order]
    return permuted_hidden_states, permuted_probs, permuted_indices


def _unpermute_tokens(
    permuted_hidden_states: torch.Tensor,
    permuted_indices: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Reverse permutation applied by _permute_tokens.

    Args:
        permuted_hidden_states: The permuted token tensor [num_all_tokens, hidden_dim]
        permuted_indices: The indices used to permute the tokens [num_all_tokens]
        num_tokens: Number of unique tokens received by the current rank

    Returns:
        Tokens aggregated and restored to their original order [num_tokens, hidden_dim]
    """
    hidden_dim = permuted_hidden_states.shape[1]
    output_hidden_states = torch.zeros(
        (num_tokens, hidden_dim),
        dtype=permuted_hidden_states.dtype,
        device=permuted_hidden_states.device,
    )
    output_hidden_states.scatter_add_(
        0, permuted_indices.unsqueeze(1).expand(-1, hidden_dim), permuted_hidden_states
    )
    return output_hidden_states


@dataclass(kw_only=True)
class FP8DispatchState(BaseFP8DispatchState):
    """State for the padded expert-major FP8 dispatch layout."""

    row_id_map: torch.Tensor
    row_to_padded_row_id: torch.Tensor
    pad_starts_t: torch.Tensor
    pad_counts_t: torch.Tensor


@dataclass
class DispatchState:
    """State from dispatch needed for combine."""

    handle_id: torch.Tensor  # CPU tensor used to retrieve cached handle
    num_recv_tokens: int
    permuted_indices: torch.Tensor | None = None
    permuted_probs: torch.Tensor | None = None
    fp8: FP8DispatchState | None = None


# TODO (fedorovgv): fused triton kernel?
def _dispatched_indices_to_local_map(
    dispatched_indices: torch.Tensor,
    dispatched_probs: torch.Tensor,
    num_local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert DeepEP's per-token expert INDEX list to a per-expert MASK map."""
    device = dispatched_indices.device
    num_recv_tokens = dispatched_indices.shape[0]
    safe_indices = torch.where(
        dispatched_indices == -1,
        torch.tensor(num_local_experts, device=device, dtype=dispatched_indices.dtype),
        dispatched_indices,
    )
    padded_map = torch.zeros(
        (num_recv_tokens, num_local_experts + 1), dtype=torch.bool, device=device
    )
    padded_probs = torch.zeros(
        (num_recv_tokens, num_local_experts + 1), dtype=torch.float32, device=device
    )
    padded_map.scatter_(dim=1, index=safe_indices, value=True)
    padded_probs.scatter_(dim=1, index=safe_indices, src=dispatched_probs.float())
    # Drop the throwaway masked column.
    return padded_map[:, :-1], padded_probs[:, :-1]


def dispatch_tokens(
    hidden_states: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    top_probs: torch.Tensor,
    num_local_experts: int,
    num_experts: int,
    group: ProcessGroup,
    score_before_experts: bool = True,
    use_fp8_dispatch: bool = False,
    expert_weights: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, DispatchState]:
    """Dispatch tokens to experts via DeepEP.

    Args:
        hidden_states: Input tokens [num_tokens, hidden_dim]
        selected_experts_indices: Expert indices for each token [num_tokens, top_k]
        top_probs: Routing probs for each token [num_tokens, top_k]
        num_local_experts: Number of experts on this rank
        num_experts: Total number of experts across all ranks
        group: EP process group
        score_before_experts: If True, apply routing probs before expert computation.
        use_fp8_dispatch: If True, produce the padded expert-major fp8 layout the
            fp8 grouped GEMMs consume.
        expert_weights: This rank's expert weights, quantized here rather than in
            the experts. The result reaches them as
            ``FP8DispatchState.fp8_expert_weights``; None means they quantize the
            weights themselves. Note that this backend does not yet overlap the
            quantization with its all-to-all -- see the TODO at the call site.

    Returns:
        (permuted_tokens, tokens_per_expert, state_for_combine)
    """
    selected_experts_indices = selected_experts_indices.contiguous()
    top_probs = top_probs.contiguous()

    # Mask out zero-score tokens
    selected_experts_indices = selected_experts_indices.masked_fill(top_probs == 0, -1)

    # Ensure float32 probs (DeepEP requirement)
    if top_probs.dtype != torch.float32:
        top_probs = top_probs.float()

    # Hide buffer setup from SAC's __torch_dispatch__ via _disable_current_modes().
    # Buffer.__init__ calls all_gather_object() which triggers aten._to_copy
    # (CUDA→CPU), a MUST_SAVE op in our SAC policy. These are infrastructure
    # ops, not model compute, and must not enter SAC's FIFO cache.
    with _disable_current_modes():
        buffer = get_buffer(group, get_hidden_bytes(hidden_states))

        # Calculate dispatch layout before actual dispatch
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert_dispatch,
            is_token_in_rank,
            _,
        ) = buffer.get_dispatch_layout(
            topk_idx=selected_experts_indices, num_experts=num_experts
        )

    # TODO (veveselov): need to overlap quantize weights with comm
    fp8_expert_weights = None
    if expert_weights and use_fp8_dispatch:
        from torchtitan.ops.float8.quantize_expert_weights import (
            quantize_expert_weight,
        )

        fp8_expert_weights = [quantize_expert_weight(w) for w in expert_weights]

    # Dispatch tokens to experts
    (
        hidden_states,
        dispatched_indices,
        dispatched_expert_probs,
        num_tokens_per_expert,
        handle_id,
    ) = torch.ops.deepep.dispatch(
        hidden_states,
        selected_experts_indices,
        top_probs,
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        is_token_in_rank,
        num_tokens_per_expert_dispatch,
        use_fp8_dispatch,
    )
    num_recv_tokens = hidden_states.shape[0]

    if use_fp8_dispatch:
        from torchtitan.ops.float8.triton_kernels.padded_permutation import (
            permute_and_pad_fn,
        )

        local_map, local_probs = _dispatched_indices_to_local_map(
            dispatched_indices, dispatched_expert_probs, num_local_experts
        )
        tokens_per_expert_e = num_tokens_per_expert.to(hidden_states.device)
        (
            padded_tokens,
            permuted_probs,
            row_id_map,
            row_to_padded_row_id,
            pad_starts_t,
            pad_counts_t,
        ) = permute_and_pad_fn(
            hidden_states, local_map, local_probs, tokens_per_expert_e
        )
        padded_group_sizes_list = [
            (m + 127) // 128 * 128 for m in num_tokens_per_expert.tolist()
        ]
        padded_group_sizes_t = torch.tensor(
            padded_group_sizes_list, dtype=torch.int32, device=hidden_states.device
        )
        state = DispatchState(
            handle_id=handle_id,
            num_recv_tokens=num_recv_tokens,
            fp8=FP8DispatchState(
                permuted_probs=permuted_probs,
                padded_group_sizes_t=padded_group_sizes_t,
                padded_group_sizes_list=padded_group_sizes_list,
                row_id_map=row_id_map,
                row_to_padded_row_id=row_to_padded_row_id,
                pad_starts_t=pad_starts_t,
                pad_counts_t=pad_counts_t,
                fp8_expert_weights=fp8_expert_weights,
            ),
        )
        return padded_tokens, tokens_per_expert_e, state

    (permuted_hidden_states, permuted_probs, permuted_indices) = _permute_tokens(
        hidden_states, dispatched_indices, dispatched_expert_probs
    )
    num_tokens_per_expert = num_tokens_per_expert.to(permuted_hidden_states.device)
    if score_before_experts and permuted_probs is not None:
        # Avoid float32 conversion to save memory
        permuted_hidden_states = permuted_hidden_states * permuted_probs.to(
            permuted_hidden_states.dtype
        ).reshape(-1, 1)
        permuted_probs_for_state = None
    else:
        permuted_probs_for_state = permuted_probs
    state = DispatchState(
        handle_id=handle_id,
        num_recv_tokens=num_recv_tokens,
        permuted_indices=permuted_indices,
        permuted_probs=permuted_probs_for_state,
    )
    return permuted_hidden_states, num_tokens_per_expert, state


def combine_tokens(
    hidden_states: torch.Tensor,
    state: DispatchState,
) -> torch.Tensor:
    """Combine tokens from experts via DeepEP."""
    if state.fp8 is None and state.permuted_probs is not None:
        hidden_states = hidden_states * state.permuted_probs.to(
            hidden_states.dtype
        ).reshape(-1, 1)
    if state.fp8 is not None:
        from torchtitan.ops.float8.triton_kernels.padded_permutation import (
            unpad_and_unpermute_fn,
        )

        hidden_states = unpad_and_unpermute_fn(
            hidden_states,
            state.fp8.row_id_map,
            state.fp8.row_to_padded_row_id,
            (state.num_recv_tokens, hidden_states.shape[-1]),
            state.fp8.pad_starts_t,
            state.fp8.pad_counts_t,
        )
    else:
        assert state.permuted_indices is not None
        hidden_states = _unpermute_tokens(
            hidden_states, state.permuted_indices, state.num_recv_tokens
        )
    return torch.ops.deepep.combine(
        hidden_states, state.handle_id, state.fp8 is not None
    )
