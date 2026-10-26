# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""In-backward gradient reduce-scatter under pipeline parallelism.

Forces gradient sync on every microbatch backward (sharded accumulation
instead of unsharded fp32 until REDUCE_GRAD); replicate-axis all-reduces are
deferred to the chunk's last backward. The scope selects which param groups
participate ("full", or only "dense"/"moe" groups by their
``grad_reduce_kind`` stamped at wrap time; the rest keep stock behavior).
Loss-internal backwards keep their own synchronization. Pure-SUM grads make
this equivalent to stock, not bitwise.
"""

from typing import cast

import torch
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_common import DDPMeshInfo

# Semi-private torch internals, pinned by test_pipeline_grad_reduce.py;
# sibling pins: loss.py (state._modules), pipeline_parallel.py
# (_maybe_compute_loss, _internal_losses/_n_microbatches).
from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    _PipelineScheduleRuntime,
    BACKWARD_WEIGHT,
    OVERLAP_F_B,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
)
from torch.distributed.pipelining.stage import _PipelineStageBase

from torchtitan.tools.logging import logger

__all__ = ["install_per_microbatch_grad_reduce"]


def _release_reduce_scatter_keepalives(fsdp_root: FSDPModule) -> None:
    """Wait on (stream-side) and drop FSDP's reduce-scatter input keepalives
    so the allocator reuses them before the next microbatch forward."""
    # pyrefly: ignore[missing-attribute]  # fully_shard.state exists at runtime
    state = fully_shard.state(fsdp_root)
    comm_ctx = state._comm_ctx
    current_stream = state._device_handle.current_stream()
    for rs_state in comm_ctx.reduce_scatter_states:
        if rs_state.event is not None:
            current_stream.wait_event(rs_state.event)
    comm_ctx.reduce_scatter_states.clear()


def _reject_gfsdp_stage(stage: _PipelineStageBase) -> None:
    """GFSDPModule subclasses FSDPModule on purpose, so the isinstance gate
    below would wrap it; the wrapper drives torch-FSDP comm internals GFSDP
    does not expose. The gfsdp2 backend validation rejects the option
    earlier; this guards direct schedule-building paths."""
    from torchtitan.distributed.gfsdp import GFSDPModule

    if isinstance(stage.submod, GFSDPModule):
        raise ValueError(
            "parallelism.pipeline_parallel_grad_reduce does not support "
            f'gfsdp stages (stage {stage.stage_index}); set it to "off" '
            "or run on fsdp_backend='fsdp2'."
        )


def _iter_param_groups(fsdp_root: FSDPModule):
    for module in cast(nn.Module, fsdp_root).modules():
        if isinstance(module, FSDPModule):
            yield from module._get_fsdp_state()._fsdp_param_groups


def _group_selected(group, scope: str) -> bool:
    """Whether ``scope`` puts this param group on per-microbatch reduce."""
    if scope == "full":
        return True
    kind = getattr(group, "grad_reduce_kind", None)
    if kind == "mixed":
        raise ValueError(
            f'parallelism.pipeline_parallel_grad_reduce="{scope}" cannot '
            f"split param group {group._module_fqn}: expert and dense params "
            "share one reduction buffer (expert_parallel_degree == 1)."
        )
    if kind is None:
        if not getattr(group, "_grad_reduce_kind_warned", False):
            logger.warning(
                f"param group {group._module_fqn} has no grad_reduce_kind "
                "(wrapped outside apply_fsdp); keeping its stock chunk-final "
                "reduction."
            )
            group._grad_reduce_kind_warned = True
        return False
    return kind == scope


def _assert_sync_took_effect(fsdp_root: FSDPModule, scope: str) -> None:
    """Fail loudly if the setter shadow stopped forcing sync (torch toggling
    it another way would silently degrade the feature to stock)."""
    for group in _iter_param_groups(fsdp_root):
        assert group.reduce_grads == _group_selected(group, scope), (
            "per-microbatch backward finished with unexpected gradient-sync"
            f" state for {group._module_fqn}; torch no longer routes stage"
            " sync through set_requires_gradient_sync"
        )


def _is_loss_managed(group) -> bool:
    """The chunked loss reduces the head's group itself (deferred across
    accumulation on purpose); the backward wrapper leaves it alone."""
    return any(
        getattr(module, "_loss_managed_grad_reduce", False) for module in group.modules
    )


def _needs_fp32_accumulator(group, scope: str) -> bool:
    """FSDP casts every reduce-scatter output to the param dtype before adding
    it to the sharded grad. With a replicate axis the cross-microbatch sum
    stays in the reduce dtype (deferred all-reduce partial); without one,
    low-precision params (``training.dtype="bfloat16"``) would accumulate the
    microbatches in bf16, where stock accumulates the unsharded grads in the
    reduce dtype."""
    reduce_dtype, orig_dtype = group._reduce_dtype, group._orig_dtype
    return (
        reduce_dtype is not None
        and orig_dtype is not None
        and reduce_dtype.itemsize > orig_dtype.itemsize
        and not isinstance(group.mesh_info, DDPMeshInfo)
        and not _is_loss_managed(group)
        and _group_selected(group, scope)
    )


@torch.no_grad()
def _fold_into_fp32_accumulators(
    fsdp_root: FSDPModule,
    scope: str,
    accumulators: dict,
    last_backward: bool,
) -> None:
    """Move each microbatch's sharded grad into a reduce-dtype accumulator and
    zero it, so the next reduce-scatter adds onto zeros (exact) instead of the
    running sum; the chunk's last backward writes the total back with a single
    cast. Each contribution is still rounded once to the param dtype by FSDP's
    cast; removing that needs the cast skipped in torch's ``foreach_reduce``."""
    for group in _iter_param_groups(fsdp_root):
        if not _needs_fp32_accumulator(group, scope):
            continue
        # Under PP the stage backward never finalizes FSDP (REDUCE_GRAD does),
        # so the compute stream is not yet ordered after the grad accumulate.
        if group._post_reduce_event is not None:
            group.device_handle.current_stream().wait_event(group._post_reduce_event)
        for fsdp_param in group.fsdp_params:
            grad = fsdp_param.sharded_param.grad
            if grad is None:
                continue
            if fsdp_param.grad_offload_event is not None:
                fsdp_param.grad_offload_event.synchronize()
            local_grad = grad.to_local()
            accumulator = accumulators.get(fsdp_param)
            if accumulator is None:
                accumulators[fsdp_param] = local_grad.to(group._reduce_dtype)
            else:
                accumulator.add_(local_grad)
            if last_backward:
                local_grad.copy_(accumulators.pop(fsdp_param))
            else:
                local_grad.zero_()


def _arm_unused_param_reduces(fsdp_root: FSDPModule, scope: str) -> list:
    """Make every selected group reduce in every backward, grads or not.

    A group with no grads in a microbatch (an expert routed zero tokens)
    would skip its post-backward reduce-scatter -- and, in the chunk's last
    backward, its replica all-reduce -- on that rank only, so peers would
    pair collectives of different groups. Each group registers its own
    post-backward hook on the module inputs, so it fires at the same point
    on every rank; with ``reduce_scatter_unused_params`` it reduces zeros
    there instead of skipping. Armed only for the stage backward: REDUCE_GRAD
    calls ``post_backward`` on every group by hand and would otherwise
    reduce zeros for all of them. The loss-managed head is left alone."""
    armed = [
        group
        for group in _iter_param_groups(fsdp_root)
        if not group.reduce_scatter_unused_params
        and not _is_loss_managed(group)
        and _group_selected(group, scope)
    ]
    for group in armed:
        group.reduce_scatter_unused_params = True
    return armed


def _has_weight_backwards(schedule: _PipelineScheduleRuntime) -> bool:
    """Weight-only backwards (zero-bubble/ZBV) bypass FSDP's post-backward
    hooks; checking this rank suffices (schedules are symmetric)."""

    def leaves(action):
        if action.computation_type == OVERLAP_F_B and action.sub_actions:
            return action.sub_actions
        return (action,)

    return any(
        leaf.computation_type == BACKWARD_WEIGHT
        for action in schedule.pipeline_order_with_comms[schedule.rank]
        if action is not None
        for leaf in leaves(action)
    )


def _install_synced_backward(stage: _PipelineStageBase, scope: str) -> None:
    """Force gradient sync on every full backward via a setter shadow (the
    stock method clears the flag) and defer replicate-axis all-reduces to
    the last backward; with a selective ``scope`` only matching param groups
    participate and the rest keep the stock flags. Every deferred reduction
    of a group must cover the same params (fixed sharded buffer); the FSDP
    grouping keeps the loss-reduced head separate from the norm, which
    guarantees that."""
    if "backward_maybe_with_nosync" in stage.__dict__:
        return  # already installed; a second wrapper would break the finally-del
    orig_backward = stage.backward_maybe_with_nosync
    sync_verified = False
    fp32_accumulators: dict = {}

    def backward_with_sync(backward_type, bwd_kwargs, last_backward=False, **kwargs):
        if backward_type != "full":
            return orig_backward(
                backward_type, bwd_kwargs, last_backward=last_backward, **kwargs
            )
        # re-read: submod may be swapped after schedule build (compile, wrappers)
        fsdp_root = cast(FSDPModule, stage.submod)  # installer checked isinstance
        orig_set_sync = fsdp_root.set_requires_gradient_sync
        defer_all_reduce = not last_backward

        def _forced_sync(requires_gradient_sync: bool, *, recurse: bool = True) -> None:
            # the stock False being passed in is the very thing we override
            if scope == "full":
                orig_set_sync(True, recurse=recurse)
                if defer_all_reduce:
                    fsdp_root.set_requires_all_reduce(False, recurse=recurse)
                return
            # Selection is per param group (dense and expert params of one
            # block live in different groups of the same module), so the
            # module-level setters are too coarse here.
            for group in _iter_param_groups(fsdp_root):
                selected = _group_selected(group, scope)
                group.reduce_grads = selected
                group.all_reduce_grads = selected and not defer_all_reduce

        # Nested toggles (e.g. ChunkedCELoss) hit child modules; unaffected.
        fsdp_root.set_requires_gradient_sync = _forced_sync
        armed = _arm_unused_param_reduces(fsdp_root, scope)
        try:
            result = orig_backward(
                backward_type, bwd_kwargs, last_backward=last_backward, **kwargs
            )
        finally:
            del fsdp_root.set_requires_gradient_sync
            for group in armed:
                group.reduce_scatter_unused_params = False
        nonlocal sync_verified
        if not sync_verified:
            _assert_sync_took_effect(fsdp_root, scope)
            sync_verified = True
        _fold_into_fp32_accumulators(fsdp_root, scope, fp32_accumulators, last_backward)
        _release_reduce_scatter_keepalives(fsdp_root)
        return result

    stage.backward_maybe_with_nosync = backward_with_sync


def install_per_microbatch_grad_reduce(
    schedule: _PipelineSchedule, scope: str = "full"
) -> None:
    """Make the ``scope``-selected FSDP param groups of every stage of
    ``schedule`` reduce-scatter gradients inside each microbatch backward;
    wraps the backward entry point, not the IR."""
    if isinstance(schedule, PipelineScheduleSingle):
        stages = [schedule._stage]
    elif isinstance(schedule, PipelineScheduleMulti):
        stages = schedule._stages
    else:
        raise ValueError(
            "parallelism.pipeline_parallel_grad_reduce supports only torch "
            "pipeline schedules (PipelineScheduleSingle/Multi), got "
            f"{type(schedule).__name__}."
        )
    # _build_pipeline_schedule always passes scale_grads=False under PP
    assert not schedule.scale_grads, "requires scale_grads=False"
    if isinstance(schedule, _PipelineScheduleRuntime) and _has_weight_backwards(
        schedule
    ):
        raise ValueError(
            "parallelism.pipeline_parallel_grad_reduce does not support "
            "split weight backwards (zero-bubble/ZBV): they bypass FSDP's "
            "post-backward hooks."
        )

    for stage in stages:
        _reject_gfsdp_stage(stage)
        if not isinstance(stage.submod, FSDPModule):
            logger.warning(
                "parallelism.pipeline_parallel_grad_reduce: stage "
                f"{stage.stage_index} is not an FSDPModule; leaving its "
                "backward unchanged."
            )
            continue
        _install_synced_backward(stage, scope)
