# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
HybridEP: Expert Parallel Communication for GB200 NVLink72 Systems.

Provides efficient token dispatch/combine for MoE training via TMA-optimized all-to-all.

Configuration (via HybridEPTokenDispatcher.Config):
    non_blocking_capacity_factor: float | None
        None = blocking mode (default).  HybridEP calls cudaStreamSynchronize
        after dispatch and computes the exact num_permuted_tokens on the host.
        float in (0, 1] = non-blocking mode; num_permuted_tokens is estimated as
        num_tokens × ep_size × min(num_local_experts, top_k) × cf, aligned for
        MXFP8.  See _num_permuted_tokens_for_non_blocking().
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch._library.opaque_object import CustomClassBase, register_opaque_type
from torch.fx.experimental.symbolic_shapes import guard_or_true

from torchtitan.distributed.deepep.capacity import non_blocking_buffer_capacity
from torchtitan.distributed.deepep.fp8_state import FP8DispatchState
from torchtitan.distributed.deepep.routing import gather_route_gradients, indices_to_map
from torchtitan.ops.float8.fp8_tensor import (
    bf16_grad_edge,
    Float8BlockwiseQTensor,
    InputLayoutType,
)
from torchtitan.tools.logging import logger

try:
    from deep_ep import HybridEPBuffer
except ImportError as exc:
    raise ImportError(
        "HybridEP requires the deep_ep library. "
        "Install from: https://github.com/deepseek-ai/DeepEP, branch: hybrid-ep"
    ) from exc


_buffer: HybridEPBuffer | None = None  # Global buffer instance
# Tokens per rank the buffer is sized for at least; see reserve_tokens_per_rank.
_reserved_tokens_per_rank: int = 0
_buffer_use_fp8: bool = False  # Wire dtype the global buffer was built for

# Side stream for the all-to-all, so the weight quantization can overlap it.
_comm_stream: torch.cuda.Stream | None = None

# Set from HybridEPTokenDispatcher.Config by set_kernel_options().
_COMM_STREAM_ENABLED: bool = False
_FUSE_PERMUTE: bool = True


def set_kernel_options(*, comm_stream: bool, fuse_permute: bool) -> None:
    """Set the dispatch/combine kernel options from the dispatcher config."""
    global _COMM_STREAM_ENABLED, _FUSE_PERMUTE
    _COMM_STREAM_ENABLED, _FUSE_PERMUTE = comm_stream, fuse_permute


def _comm_stream_enabled() -> bool:
    return _COMM_STREAM_ENABLED


def _fuse_permute_enabled() -> bool:
    """Fold permute/unpermute into the dispatch/combine kernels."""
    return _FUSE_PERMUTE


def _iter_tensors(obj: Any) -> Iterator[torch.Tensor]:
    """Yield the CUDA tensors inside op inputs/outputs, for ``record_stream``."""
    if isinstance(obj, Float8BlockwiseQTensor):
        yield from _iter_tensors((obj._data, obj._scale))
    elif isinstance(obj, torch.Tensor):
        if obj.is_cuda:
            yield obj
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            yield from _iter_tensors(item)


def _launch_on_comm_stream(
    fn: Callable[[], Any],
    *produced_on_compute: Any,
    overlap_on_compute: Callable[[], Any] | None = None,
) -> tuple[Any, Any, Callable[[], None]]:
    global _comm_stream
    if not _comm_stream_enabled() or not torch.cuda.is_available():
        overlapped = overlap_on_compute() if overlap_on_compute is not None else None
        return fn(), overlapped, lambda: None

    if _comm_stream is None:
        _comm_stream = torch.cuda.Stream()

    compute = torch.cuda.current_stream()
    _comm_stream.wait_stream(compute)
    for arg in produced_on_compute:
        for t in _iter_tensors(arg):
            t.record_stream(_comm_stream)

    overlapped = overlap_on_compute() if overlap_on_compute is not None else None

    with torch.cuda.stream(_comm_stream):
        out = fn()

    def _join() -> None:
        compute.wait_stream(_comm_stream)
        for t in _iter_tensors(out):
            t.record_stream(compute)

    return out, overlapped, _join


def _run_on_comm_stream(fn: Callable[[], Any], *produced_on_compute: Any) -> Any:
    """``_launch_on_comm_stream`` with nothing to overlap: join immediately."""
    out, _, join = _launch_on_comm_stream(fn, *produced_on_compute)
    join()
    return out


class DispatchHandle(CustomClassBase):
    """Opaque wrapper for HybridEP dispatch handle.

    Wraps the deep_ep dispatch handle as an opaque type so it can be returned
    from custom ops and flow through the torch.compile graph, eliminating
    the need for a global handle cache.
    """

    def __init__(self, value=None):
        self.value = value

    def __eq__(self, other):
        if not isinstance(other, DispatchHandle):
            return False
        return self.value is other.value

    def __hash__(self):
        if self.value is None:
            return 0
        try:
            return hash(self.value)
        except TypeError:
            return id(self.value)

    def __fx_repr__(self):
        return "DispatchHandle()", {"DispatchHandle": DispatchHandle}


# make unvisible code inside handle for torch.compile
register_opaque_type(DispatchHandle, typ="reference")


@dataclass
class DispatchState:
    """State from dispatch needed for combine.

    Attributes:
        handle: Opaque dispatch handle wrapping the deep_ep handle.
        permuted_probs: probs for score_before_experts=False mode.
        num_tokens: Original input token count (for combine fake shape inference).
        non_blocking: whether the dispatch ran CPU-free. ``combine``'s backward
            is another ``dispatch_with_permute``, so it has to be told the same
            or it reintroduces the host sync this mode exists to avoid.
    """

    handle: DispatchHandle
    permuted_probs: torch.Tensor | None = None
    num_tokens: int = 0
    use_fp8_dispatch: bool = False
    non_blocking: bool = False
    fp8: FP8DispatchState | None = None


def _apply_probs(
    hidden: torch.Tensor,
    probs: torch.Tensor,
    apply_now: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply routing probs to hidden states if apply_now, else defer."""
    if apply_now and probs is not None and guard_or_true(probs.numel() > 0):
        return hidden * probs.to(hidden.dtype).reshape(-1, 1), None
    return hidden, probs


def _num_permuted_tokens_for_non_blocking(
    num_tokens: int,
    ep_size: int,
    num_local_experts: int,
    top_k: int,
    moe_expert_capacity_factor: float,
    pad_multiple: int | None = None,
) -> int:
    """Pre-allocated output buffer size for non-blocking dispatch.

    Formula: num_tokens × ep_size × min(num_local_experts, top_k) × cf,
    aligned to pad_multiple if set.

    capacity_factor=1.0 covers the unpadded worst case for equal-sized sources.
    Per-expert padding can require more than aligning the aggregate total;
    capacity-enabled callers validate that separately before dispatch.
    Values < 1.0 reduce memory at the cost of potentially dropping tokens
    when the permuted offset exceeds the buffer capacity. A token-drop policy
    must bound the padded receiver load before selecting a smaller allocation.
    """
    return non_blocking_buffer_capacity(
        num_tokens,
        ep_size,
        num_local_experts,
        top_k,
        moe_expert_capacity_factor,
        pad_multiple,
    )


def _quantize_expert_weights(
    expert_weights: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Blockwise-quantize the fused w13 / w2 expert weights, flattened."""
    if not expert_weights:
        return []
    assert len(expert_weights) == 2, (
        f"Expected fused w13 and w2 expert weights, got {len(expert_weights)}"
    )
    from torchtitan.ops.float8.triton_kernels import block_quant_fn

    w13, w2 = expert_weights
    with torch.no_grad():
        w13_data, w13_scale = block_quant_fn(w13)
        w2_data, w2_scale = block_quant_fn(w2)
    return [w13_data, w13_scale, w2_data, w2_scale]


# Custom op registration for torch.compile and SAC compatibility


@torch.library.custom_op("hybridep::dispatch", mutates_args=(), device_types="cuda")
def _dispatch_impl(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    ep_size: int,
    group_name: str,
    expert_weights: list[torch.Tensor],
    non_blocking: bool = False,
    moe_expert_capacity_factor: float | None = None,
    pad_multiple: int | None = None,
    use_fp8_dispatch: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    DispatchHandle,
    list[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    DeepEP's dispatch_with_permute needs to know the output buffer size
    (num_permuted_tokens) for the fused permute kernel.

    * **non_blocking=True** — no D2H sync is allowed, so num_permuted_tokens
      must be supplied upfront via moe_expert_capacity_factor.
    * **non_blocking=False (blocking)** — HybridEP does cudaStreamSynchronize,
      then reads tokens_per_expert from pinned CPU memory to compute the
      exact num_permuted_tokens on the host.
    """
    num_local_experts = num_experts // ep_size

    group = dist.distributed_c10d._resolve_process_group(group_name)
    get_buffer(
        group=group,
        hidden_dim=x.shape[1],
        num_tokens=x.shape[0],
        num_local_experts=num_local_experts,
        use_fp8_dispatch=use_fp8_dispatch,
    )

    routing_map, probs = indices_to_map(topk_idx, topk_weights, num_experts)

    num_permuted_tokens = None
    if non_blocking:
        assert moe_expert_capacity_factor is not None, (
            "moe_expert_capacity_factor is required for non_blocking dispatch"
        )
        num_permuted_tokens = _num_permuted_tokens_for_non_blocking(
            x.shape[0],
            ep_size,
            num_local_experts,
            topk_idx.shape[1],
            moe_expert_capacity_factor,
            pad_multiple=pad_multiple,
        )
    device = x.device
    x = x if x.is_contiguous() else x.contiguous()
    if use_fp8_dispatch:
        from torchtitan.ops.float8.triton_kernels import row_quant_fn

        hidden_in, scaling_factor = row_quant_fn(x)
    else:
        hidden_in, scaling_factor = x, None

    (
        (
            hidden,
            probs,
            recv_scale,
            tokens_per_expert,
            handle,
        ),
        weights_quantized,
        _join_dispatch,
    ) = _launch_on_comm_stream(
        lambda: _buffer.dispatch_with_permute(
            hidden=hidden_in,
            routing_map=routing_map,
            probs=probs,
            scaling_factor=scaling_factor,
            use_fp8=use_fp8_dispatch,
            num_of_experts_per_rank=num_local_experts,
            pad_multiple=pad_multiple,
            num_permuted_tokens=num_permuted_tokens,
            non_blocking=non_blocking,
            fuse_permute_dispatch=_fuse_permute_enabled(),
        ),
        hidden_in,
        scaling_factor,
        routing_map,
        probs,
        overlap_on_compute=lambda: _quantize_expert_weights(expert_weights),
    )
    _join_dispatch()

    # In non_blocking mode nothing here can detect a drop: HybridEP's overflow
    # signal stays on GPU and reading it would force the sync this mode exists to
    # avoid. Drops are prevented by sizing, not detection -- see
    # _num_permuted_tokens_for_non_blocking (1.0 covers unpadded routes;
    # per-expert padding needs additional space). The dispatcher validates this
    # bound when token_drop is enabled.
    # TODO: accumulate the flag into a GPU counter the metrics processor can read
    # at its own (already synchronizing) cadence.

    dispatch_handle = DispatchHandle(value=handle)

    # csrc/hybrid_ep/executor/executor.cu:165
    # // Both paths: handle.tokens_per_expert is raw int32 on GPU.
    # // Produce padded_tokens_per_expert (pinned or device int64) via pad kernel.
    # if (non_blocking) {
    #     handle.padded_tokens_per_expert = torch::empty({E}, dtype(kInt64).device(kCUDA));
    # } else {
    #     handle.padded_tokens_per_expert = torch::empty({E}, dtype(kInt64).pinned_memory(true));
    # }
    padded_group_sizes_cpu = torch.empty(0, dtype=torch.int32, device="cpu")
    if non_blocking:
        # Nothing may read the counts on the host in this mode, and the fake
        # declares a device tensor.
        assert tokens_per_expert.device == device, (
            "non-blocking HybridEP dispatch must leave the per-expert counts on "
            f"the device, got {tokens_per_expert.device}"
        )
    else:
        assert tokens_per_expert.device.type == "cpu", (
            "blocking HybridEP dispatch must hand the per-expert counts back in "
            f"pinned host memory, got {tokens_per_expert.device}"
        )
        # Read them once here, so no later reader needs a D2H of its own. Only
        # the int64 -> int32 narrowing happens, on the host.
        padded_group_sizes_cpu = tokens_per_expert.to(torch.int32)
        tokens_per_expert = tokens_per_expert.to(device, non_blocking=True)

    if use_fp8_dispatch:
        # make fake bf16 hidden, for bypass torch.ops
        hidden_grad_edge = bf16_grad_edge(
            hidden.shape[0], hidden.shape[1], device=device
        )
    else:
        recv_scale = hidden.new_empty(0, dtype=torch.float32)
        hidden_grad_edge = hidden.new_empty(0)

    # handle[7] of the fused layout is the scan kernel's tokens_per_expert, the
    # counts HybridEP then pads in place -- the original, not something read
    # back from the rounded numbers. The standalone layout is a different tuple
    # with no unpadded counts at all. deep_ep/hybrid_ep_buffer.py
    # (returned_handle), csrc/hybrid_ep/executor/executor.cu:127,150.
    unpadded = handle[7] if _fuse_permute_enabled() else tokens_per_expert
    # Copied because the handle keeps reading this buffer -- the fused dispatch
    # kernel takes it as an input (executor.cu:279) and combine's backward
    # replays that dispatch. Nothing writes to it today, so this holds an
    # invariant the compiler cannot see: inside the graph the handle is opaque.
    unpadded_tokens_per_expert = unpadded.to(
        device=device, dtype=torch.int32, copy=True
    )

    return (
        hidden,
        recv_scale,
        probs,
        tokens_per_expert,
        dispatch_handle,
        weights_quantized,
        hidden_grad_edge,
        padded_group_sizes_cpu,
        unpadded_tokens_per_expert,
    )


@_dispatch_impl.register_fake
def _dispatch_fake(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    ep_size: int,
    group_name: str,
    expert_weights: list[torch.Tensor],
    non_blocking: bool = False,
    moe_expert_capacity_factor: float | None = None,
    pad_multiple: int | None = None,
    use_fp8_dispatch: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    DispatchHandle,
    list[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Fake dispatch for torch.compile tracing."""
    num_local_experts = num_experts // ep_size
    if non_blocking:
        out_tokens = _num_permuted_tokens_for_non_blocking(
            x.shape[0],
            ep_size,
            num_local_experts,
            topk_idx.shape[1],
            moe_expert_capacity_factor,  # pyrefly: ignore [bad-argument-type]
            pad_multiple=pad_multiple,
        )
    else:
        out_tokens = torch.library.get_ctx().new_dynamic_size()
        # Groups are padded to pad_multiple, so the total is too. Downstream
        # kernels assert their row block divides it -- 128 in the quantizations
        # and row2col, 64 in the fused SwiGLU -- and on an unbacked size that is
        # a data-dependent guard. Both are stated because the solver does not
        # factor a modulus: "% 128" leaves "% 64" undecidable.
        for row_block in (128, 64):
            if pad_multiple and pad_multiple % row_block == 0:
                torch._check(out_tokens % row_block == 0)

    # Matches the impl: under fp8 the op returns the raw e4m3 payload, and the
    # Float8BlockwiseQTensor is rebuilt outside it.
    hidden = x.new_empty(
        out_tokens,
        x.shape[1],
        dtype=torch.float8_e4m3fn if use_fp8_dispatch else x.dtype,
    )
    probs = x.new_empty(out_tokens, dtype=torch.float32)
    tpe = x.new_empty(num_local_experts, dtype=torch.int64)
    scale = (
        x.new_empty(out_tokens, x.shape[1] // 128, dtype=torch.float32)
        if use_fp8_dispatch
        else x.new_empty(0, dtype=torch.float32)
    )
    # The impl returns four tensors per weight pair when expert_weights is
    # non-empty (w13 data/scale, w2 data/scale); an empty list here would make
    # the traced arity disagree with the runtime one.
    quantized: list[torch.Tensor] = []
    for w in expert_weights:
        quantized.append(w.new_empty(w.shape, dtype=torch.float8_e4m3fn))
        quantized.append(
            w.new_empty(
                (w.shape[0], w.shape[1] // 128, w.shape[2] // 128),
                dtype=torch.float32,
            )
        )
    if use_fp8_dispatch:
        # make fake bf16 hidden, for bypass torch.ops
        hidden_grad_edge = bf16_grad_edge(out_tokens, x.shape[1], device=x.device)
    else:
        hidden_grad_edge = x.new_empty(0)
    # Blocking leaves the counts on the host, non-blocking never reads them.
    # The branch is on a static argument, so the fake and the impl agree.
    padded_group_sizes_cpu = torch.empty(
        0 if non_blocking else num_local_experts, dtype=torch.int32, device="cpu"
    )
    unpadded_tokens_per_expert = x.new_empty(num_local_experts, dtype=torch.int32)
    return (
        hidden,
        scale,
        probs,
        tpe,
        DispatchHandle(),
        quantized,
        hidden_grad_edge,
        padded_group_sizes_cpu,
        unpadded_tokens_per_expert,
    )


@torch.library.custom_op("hybridep::combine", mutates_args=(), device_types="cuda")
def _combine_impl(
    x: torch.Tensor,
    handle: DispatchHandle,
    num_tokens: int,
    pad_multiple: int | None = None,
    use_fp8_dispatch: bool = False,
    non_blocking: bool = False,
) -> torch.Tensor:
    """CUDA combine: reverse dispatch permutation via opaque handle."""
    global _buffer
    if _buffer is None:
        raise RuntimeError("HybridEP buffer not initialized.")

    x = x if x.is_contiguous() else x.contiguous()
    combined, _ = _run_on_comm_stream(
        lambda: _buffer.combine_with_unpermute(
            hidden=x,
            handle=handle.value,
            fuse_unpermute_combine=_fuse_permute_enabled(),
        ),
        x,
    )
    return combined


@_combine_impl.register_fake
def _combine_fake(
    x: torch.Tensor,
    handle: DispatchHandle,
    num_tokens: int,
    pad_multiple: int | None = None,
    use_fp8_dispatch: bool = False,
    non_blocking: bool = False,
) -> torch.Tensor:
    """Fake combine for torch.compile tracing."""
    return x.new_empty(num_tokens, x.shape[1])


# ============================================================================
# Backward op implementations
# ============================================================================


@torch.library.custom_op("hybridep::dispatch_bwd", mutates_args=(), device_types="cuda")
def _dispatch_bwd_impl(
    grad_combined: torch.Tensor,
    handle: DispatchHandle,
    num_permuted_tokens: int,
    pad_multiple: int | None = None,
    use_fp8_dispatch: bool = False,
    non_blocking: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward of combine: scatter gradients via dispatch_with_permute."""
    global _buffer
    if _buffer is None:
        raise RuntimeError("HybridEP buffer not initialized.")

    # dispatch_with_permute asserts hidden.is_contiguous(); autograd hands back
    # non-contiguous grads often enough that deepep guards the same spots.
    grad_combined = grad_combined.contiguous()
    if use_fp8_dispatch:
        from torchtitan.ops.float8.triton_kernels import row_quant_fn

        g_data, g_scale = row_quant_fn(grad_combined)
    else:
        g_data, g_scale = grad_combined, None

    grad_x, _, grad_scale, _, _ = _run_on_comm_stream(
        lambda: _buffer.dispatch_with_permute(
            hidden=g_data,
            scaling_factor=g_scale,
            handle=handle.value,
            num_permuted_tokens=num_permuted_tokens,
            pad_multiple=pad_multiple,
            non_blocking=non_blocking,
            fuse_permute_dispatch=_fuse_permute_enabled(),
        ),
        g_data,
        g_scale,
    )

    # Plain pair out of the op; _Fp8CombineGrad repacks it. See the note on
    # hybridep::dispatch's return -- functionalization drops the wrapper.
    if not use_fp8_dispatch:
        grad_scale = grad_x.new_empty(0, dtype=torch.float32)
    return grad_x, grad_scale


@_dispatch_bwd_impl.register_fake
def _dispatch_bwd_fake(
    grad_combined: torch.Tensor,
    handle: DispatchHandle,
    num_permuted_tokens: int,
    pad_multiple: int | None = None,
    use_fp8_dispatch: bool = False,
    non_blocking: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake dispatch_bwd for torch.compile tracing."""
    # Under fp8 the impl row-quantizes the gradient before scattering it, so
    # what comes back is e4m3 and not the bf16 it was handed -- ``new_empty``
    # would otherwise inherit the input's dtype and inductor would catch the
    # mismatch only at runtime.
    grad_x = grad_combined.new_empty(
        num_permuted_tokens,
        grad_combined.shape[1],
        dtype=torch.float8_e4m3fn if use_fp8_dispatch else grad_combined.dtype,
    )
    scale = (
        grad_combined.new_empty(
            num_permuted_tokens, grad_combined.shape[1] // 128, dtype=torch.float32
        )
        if use_fp8_dispatch
        else grad_combined.new_empty(0, dtype=torch.float32)
    )
    return grad_x, scale


@torch.library.custom_op("hybridep::combine_bwd", mutates_args=(), device_types="cuda")
def _combine_bwd_impl(
    grad_hidden: torch.Tensor,
    grad_probs: torch.Tensor,
    handle: DispatchHandle,
    num_tokens: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward of dispatch: gather gradients via combine_with_unpermute."""
    global _buffer
    if _buffer is None:
        raise RuntimeError("HybridEP buffer not initialized.")

    # combine_with_unpermute asserts hidden.is_contiguous(), and for probs both
    # is_contiguous() and dtype == kFloat32.
    grad_hidden = grad_hidden.contiguous()
    # An empty receiver can still own source tokens sent to another rank.
    # Keep the probs channel enabled to receive their routing-score gradients.
    probs_in = grad_probs.float().contiguous()

    grad_x, grad_probs_dense = _run_on_comm_stream(
        lambda: _buffer.combine_with_unpermute(
            hidden=grad_hidden,
            probs=probs_in,
            handle=handle.value,
            fuse_unpermute_combine=_fuse_permute_enabled(),
        ),
        grad_hidden,
        probs_in,
    )
    if grad_probs_dense is None:
        grad_probs_dense = torch.zeros(
            num_tokens, num_experts, device=grad_hidden.device, dtype=torch.float32
        )
    return grad_x, grad_probs_dense


@_combine_bwd_impl.register_fake
def _combine_bwd_fake(
    grad_hidden: torch.Tensor,
    grad_probs: torch.Tensor,
    handle: DispatchHandle,
    num_tokens: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake combine_bwd for torch.compile tracing."""
    del grad_probs  # its emptiness does not reach the output shapes
    hidden_dim = grad_hidden.shape[1]
    grad_x = grad_hidden.new_empty(num_tokens, hidden_dim)
    grad_probs_dense = grad_hidden.new_empty(
        num_tokens, num_experts, dtype=torch.float32
    )
    return grad_x, grad_probs_dense


def _dispatch_backward(
    ctx,
    grad_hidden,
    grad_scale,
    grad_probs,
    grad_tpe,
    grad_handle,
    grad_weights_quantized,
    grad_edge,
    grad_group_sizes_cpu,
    grad_unpadded_tokens_per_expert,
):
    """Backward: gather gradients via combine_bwd op."""
    del grad_scale
    grad_hidden = grad_edge if ctx.use_fp8_dispatch else grad_hidden
    # `expert_weights` is a Tensor[] input, so autograd wants a list of grads back
    # for it, not a bare None -- returning None fails the structure check.
    grad_expert_weights = [None] * ctx.num_expert_weights
    if grad_hidden is None:
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            grad_expert_weights,
            None,
            None,
            None,
            None,
        )

    (topk_idx,) = ctx.saved_tensors
    num_tokens = topk_idx.shape[0]

    grad_x, grad_probs_dense = torch.ops.hybridep.combine_bwd(
        grad_hidden, grad_probs, ctx.dispatch_handle, num_tokens, ctx.num_experts
    )
    grad_x = grad_x.to(ctx.input_dtype)

    # Dense unconditionally, so no emptiness test here either.
    grad_weights = gather_route_gradients(grad_probs_dense, topk_idx)
    # Gradients, in input order: x, topk_idx, topk_weights, num_experts, ep_size,
    # group_name, expert_weights, non_blocking, moe_expert_capacity_factor,
    # pad_multiple, use_fp8_dispatch
    return (
        grad_x,
        None,
        grad_weights,
        None,
        None,
        None,
        grad_expert_weights,
        None,
        None,
        None,
        None,
    )


def _dispatch_setup_context(ctx, inputs, output):
    """Save context for dispatch backward."""
    x, topk_idx, _, num_experts, _, _, expert_weights, _, _, _, use_fp8 = inputs
    (
        hidden,
        recv_scale,
        _,
        _,
        dispatch_handle,
        quantized,
        hidden_grad_edge,
        _,
        _,
    ) = output
    ctx.mark_non_differentiable(recv_scale, *quantized)
    if use_fp8:
        ctx.mark_non_differentiable(hidden)
    else:
        ctx.mark_non_differentiable(hidden_grad_edge)
    ctx.use_fp8_dispatch = use_fp8
    ctx.dispatch_handle = dispatch_handle
    ctx.input_dtype = x.dtype
    ctx.num_experts = num_experts
    ctx.num_expert_weights = len(expert_weights)
    ctx.save_for_backward(topk_idx)


def _combine_backward(ctx, grad_combined):
    """Backward: scatter gradients via dispatch_bwd op.

    The bf16 path only. Under fp8 the gradient is packed, so combine runs
    inside ``_Fp8CombineGrad``, which owns the edge it goes back on.
    """
    grad_x, grad_scale = torch.ops.hybridep.dispatch_bwd(
        grad_combined,
        ctx.dispatch_handle,
        ctx.num_permuted_tokens,
        ctx.pad_multiple,
        ctx.use_fp8_dispatch,
        ctx.non_blocking,
    )
    del grad_scale  # empty on this path: fp8 goes through _Fp8CombineGrad
    return grad_x, None, None, None, None, None


def _combine_setup_context(ctx, inputs, output):
    """Save context for combine backward."""
    x, dispatch_handle, _num_tokens, pad_multiple, use_fp8_dispatch, non_blocking = (
        inputs
    )
    ctx.dispatch_handle = dispatch_handle
    ctx.num_permuted_tokens = x.shape[0]
    ctx.pad_multiple = pad_multiple
    ctx.use_fp8_dispatch = use_fp8_dispatch  # always False here; see the docstring
    ctx.non_blocking = non_blocking


_dispatch_impl.register_autograd(
    _dispatch_backward, setup_context=_dispatch_setup_context
)
_combine_impl.register_autograd(_combine_backward, setup_context=_combine_setup_context)


class _Fp8CombineGrad(torch.autograd.Function):
    """Owns the autograd edge of an fp8 combine.

    The combine moves bf16 expert outputs, but its backward is an fp8 dispatch,
    so the gradient it produces is packed -- and a packed gradient cannot come
    out of ``hybridep::combine`` itself, whose returns lose the wrapper to
    functionalization. It goes up through a packed edge instead, while the bf16
    value stays off the graph: the mirror of the dispatch side.
    """

    @staticmethod
    def forward(ctx, fp8_edge, x, handle, num_tokens, pad_multiple, non_blocking):
        ctx.handle = handle
        ctx.pad_multiple = pad_multiple
        ctx.non_blocking = non_blocking
        # The permuted row count is data-dependent, and an unbacked SymInt
        # cannot travel on ``ctx``: the autograd.Function HOP hands it back as
        # a node of the outer graph. The edge carries the same count in its
        # shape, so save it and read the size in backward.
        ctx.save_for_backward(fp8_edge._data)
        with torch.no_grad():
            return torch.ops.hybridep.combine(
                x, handle, num_tokens, pad_multiple, True, non_blocking
            )

    @staticmethod
    def backward(ctx, grad_combined):
        (edge_data,) = ctx.saved_tensors
        grad_data, grad_scale = torch.ops.hybridep.dispatch_bwd(
            grad_combined,
            ctx.handle,
            edge_data.shape[0],
            ctx.pad_multiple,
            True,
            ctx.non_blocking,
        )
        grad_payload = Float8BlockwiseQTensor(
            data=grad_data,
            scale=grad_scale,
            orig_dtype=grad_combined.dtype,
            input_layout_type=InputLayoutType.ROWWISE,
            block_scaling_dim=1,
            scale_block_first=False,
        )
        # Gradients for: fp8_edge, x, handle, num_tokens, pad_multiple,
        # non_blocking
        return grad_payload, None, None, None, None, None


# HybridEP's own defaults are 24 dispatch / 24 combine intranode (8/8 multinode).
# We pin 16/16: NVIDIA's Hybrid-EP README reports HybridEP at 16 SMs matching DeepEP
# at 32, so this leaves more SMs for compute.  ``None`` delegates to HybridEP.
_NUM_SMS_DISPATCH: int = 16
_NUM_SMS_COMBINE: int = 16

# SM budget the live buffer was built with, so a later change forces a reinit.
_buffer_num_sms: tuple[int | None, int | None] = (None, None)


def set_num_sms(*, dispatch: int, combine: int) -> None:
    """Set the SM budget for HybridEP's dispatch / combine kernels."""
    global _NUM_SMS_DISPATCH, _NUM_SMS_COMBINE
    _NUM_SMS_DISPATCH, _NUM_SMS_COMBINE = dispatch, combine


def reserve_tokens_per_rank(num_tokens: int) -> None:
    """Size the buffer for at least ``num_tokens`` per rank from its first allocation."""
    global _reserved_tokens_per_rank
    _reserved_tokens_per_rank = max(_reserved_tokens_per_rank, num_tokens)


def get_buffer(
    group: dist.ProcessGroup,
    hidden_dim: int,
    num_tokens: int,
    num_local_experts: int,
    use_fp8_dispatch: bool = False,
) -> None:
    """Ensure the global HybridEP buffer is initialized, reinitializing if config changed.

    Allocates the all-to-all communication buffers (RDMA inter-node + NVLink IPC
    intra-node), sized by num_tokens as max_num_of_tokens_per_rank.  No capacity
    factor is applied here — these buffers hold the per-rank input tokens, not the
    fan-out permuted output.  HybridEP auto-grows via update_template_config if a
    later dispatch has more tokens than the initial allocation.
    """
    global _buffer, _buffer_use_fp8, _buffer_num_sms

    # The JIT template's static assert needs capacity % 64 == 0; unrounded
    # lazy growth lands on 16-aligned counts and fails nvcc.
    max_tokens_per_rank = (max(num_tokens, _reserved_tokens_per_rank) + 63) // 64 * 64

    needs_reinit = (
        _buffer is None
        or _buffer_use_fp8 != use_fp8_dispatch
        or _buffer.group != group
        or _buffer.configurer.buffer_config.hidden_dim < hidden_dim
        or _buffer.configurer.buffer_config.max_num_of_tokens_per_rank
        < max_tokens_per_rank
        or _buffer.configurer.buffer_config.num_of_experts_per_rank < num_local_experts
        or _buffer_num_sms != (_NUM_SMS_DISPATCH, _NUM_SMS_COMBINE)
    )

    if needs_reinit:
        logger.info(
            "Initializing HybridEP buffer: hidden_dim=%d, max_tokens_per_rank=%d, "
            "num_local_experts=%d, ep_size=%d, num_sms=(%s, %s)",
            hidden_dim,
            max_tokens_per_rank,
            num_local_experts,
            group.size(),
            _NUM_SMS_DISPATCH,
            _NUM_SMS_COMBINE,
        )
        _buffer = HybridEPBuffer(
            group=group,
            hidden_dim=hidden_dim,
            max_num_of_tokens_per_rank=max_tokens_per_rank,
            num_local_experts=num_local_experts,
            use_fp8=use_fp8_dispatch,
            num_sms_dispatch_api=_NUM_SMS_DISPATCH,
            num_sms_combine_api=_NUM_SMS_COMBINE,
            load_cached_kernels=True,
            use_shared_buffer=True,
            enable_custom_allgather=True,
        )
        _buffer_use_fp8 = use_fp8_dispatch
        _buffer_num_sms = (_NUM_SMS_DISPATCH, _NUM_SMS_COMBINE)


def dispatch_tokens(
    hidden_states: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    top_probs: torch.Tensor,
    num_local_experts: int,
    num_experts: int,
    group: dist.ProcessGroup,
    score_before_experts: bool = True,
    non_blocking_expert_capacity_factor: float | None = None,
    pad_multiple: int | None = None,
    use_fp8_dispatch: bool = False,
    expert_weights: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, DispatchState]:
    """Dispatch tokens to experts via HybridEP all-to-all.

    Args:
        hidden_states: [num_tokens, hidden_dim]
        selected_experts_indices: [num_tokens, top_k]
        top_probs: [num_tokens, top_k]
        num_local_experts: Experts on this EP rank
        num_experts: Total experts across all ranks
        group: EP ProcessGroup
        score_before_experts: Apply probs before expert computation
        non_blocking_expert_capacity_factor: None = blocking mode (default).
            float in (0, 1] = non-blocking mode; pre-sizes the permute output
            tensor as num_tokens × ep_size × min(num_local_experts, top_k) × cf,
            aligned to pad_multiple.
        pad_multiple: Pad per-expert token groups to this multiple (e.g. 32 for
            MXFP8). None means no padding.

    Returns:
        (permuted_hidden, tokens_per_expert, state)
    """
    non_blocking = non_blocking_expert_capacity_factor is not None

    selected_experts_indices = selected_experts_indices.contiguous()
    top_probs = top_probs.contiguous()

    ep_size = group.size()
    group_name = group.group_name

    (
        hidden,
        recv_scale,
        permuted_probs,
        tokens_per_expert,
        dispatch_handle,
        weights_quantized,
        hidden_grad_edge,
        padded_group_sizes_cpu,
        unpadded_tokens_per_expert,
    ) = torch.ops.hybridep.dispatch(
        hidden_states,
        selected_experts_indices,
        top_probs,
        num_experts,
        ep_size,
        group_name,
        # Only when the fp8 experts will actually read them; otherwise the op
        # would quantize the weights and the result would be thrown away.
        list(expert_weights or []) if use_fp8_dispatch else [],
        non_blocking,
        non_blocking_expert_capacity_factor,
        pad_multiple,
        use_fp8_dispatch,
    )

    if use_fp8_dispatch:
        from torchtitan.ops.float8.quantize_expert_weights import (
            quantize_expert_weight,
        )

        assert not score_before_experts, (
            "fp8 grouped-GEMM experts require score_before_experts=False: probs "
            "are applied in the experts, not before dispatch."
        )
        # Repacked here rather than inside the op; see the note on its return.
        padded_tokens = Float8BlockwiseQTensor(
            data=hidden,
            scale=recv_scale,
            orig_dtype=hidden_states.dtype,
            input_layout_type=InputLayoutType.ROWWISE,
            block_scaling_dim=1,
            scale_block_first=False,
        )
        # make restore data, scales in Float8BlockwiseQTensor (if weights also quantized)
        fp8_expert_weights = (
            [
                quantize_expert_weight(
                    w, (weights_quantized[2 * i], weights_quantized[2 * i + 1])
                )
                for i, w in enumerate(expert_weights or [])
            ]
            if weights_quantized
            else None
        )
        # The host copy comes off the op's own CPU output -- one channel for the
        # counts, not a Python attribute smuggled on the opaque handle, which
        # Dynamo could not trace anyway. Empty in non-blocking mode and under
        # tracing, where they never reach the host: everything but DeepGEMM's
        # k-grouped wgrad reads the device tensor instead, and that one takes
        # its own D2H in the backward -- see _GroupedGemm.backward.
        padded_group_sizes_list = (
            None
            if torch.compiler.is_compiling() or padded_group_sizes_cpu.numel() == 0
            else padded_group_sizes_cpu.tolist()
        )
        state = DispatchState(
            handle=dispatch_handle,
            num_tokens=hidden_states.shape[0],
            use_fp8_dispatch=True,
            non_blocking=non_blocking,
            fp8=FP8DispatchState(
                permuted_probs=permuted_probs,
                padded_tokens=padded_tokens,
                uses_gradient_edges=True,
                padded_group_sizes_t=tokens_per_expert.to(torch.int32),
                padded_group_sizes_list=padded_group_sizes_list,
                padded_group_sizes_cpu=padded_group_sizes_cpu,
                fp8_expert_weights=fp8_expert_weights,
            ),
        )
        # The payload travels in the state; the experts get the edge that
        # carries its gradient.
        return hidden_grad_edge, unpadded_tokens_per_expert, state

    hidden, permuted_probs = _apply_probs(hidden, permuted_probs, score_before_experts)
    if permuted_probs is not None and permuted_probs.dtype != hidden.dtype:
        permuted_probs = permuted_probs.to(hidden.dtype)

    state = DispatchState(
        handle=dispatch_handle,
        permuted_probs=permuted_probs,
        num_tokens=hidden_states.shape[0],
        non_blocking=non_blocking,
    )
    return hidden, tokens_per_expert, state


def combine_tokens(
    hidden_states: torch.Tensor,
    state: DispatchState,
    pad_multiple: int | None = None,
) -> torch.Tensor:
    """Combine expert outputs back to original token order.

    Applies deferred probs (if any), then unpermutes via the opaque dispatch handle.
    """
    if state.use_fp8_dispatch:
        assert state.fp8 is not None and state.fp8.output_fp8_edge is not None, (
            "the fp8 experts must hand back the packed edge for their output"
        )
        return _Fp8CombineGrad.apply(
            state.fp8.output_fp8_edge,
            hidden_states,
            state.handle,
            state.num_tokens,
            pad_multiple,
            state.non_blocking,
        )

    if state.permuted_probs is not None:
        hidden_states = hidden_states * state.permuted_probs.reshape(-1, 1)

    return torch.ops.hybridep.combine(
        hidden_states,
        state.handle,
        state.num_tokens,
        pad_multiple,
        False,
        state.non_blocking,
    )


__all__ = [
    "dispatch_tokens",
    "combine_tokens",
    "DispatchState",
    "DispatchHandle",
]
