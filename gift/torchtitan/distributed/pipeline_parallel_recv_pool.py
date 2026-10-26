# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pool pipeline P2P recv buffers by their lifetimes in the lowered schedule.

Each buffer stays live from RECV until its last reader: forward, every
backward flavor (weight backward reads the stage input; split backward
retains output gradients until weight backward), and the stage's own SEND —
a stage output may alias the stage input (passthrough, e.g. the MTP
``input_embeds`` carry), so sending microbatch ``i`` onward reads its fwd
recv buffer. Adjacent local stages can pass aliases without copying, so
their readers extend the same lifetime. Greedy interval colouring assigns
the minimum number of buffer sets per stage and direction.

Reuse relies on the runtime executing its action list in order: a reusing
RECV is posted only after every reader of the slot's previous microbatch,
and NCCL synchronizes the posted recv with the current compute stream.
Positions of RECV actions and the streams they use do not change, so pooling
adds no ordering between communication and compute beyond stock.

Each forward gets fresh leaf tensors over the slot's storage, so autograd
state (``.grad``, module hooks) dies with the microbatch's graph. When the
slot moves on to its next owner the previous owner's leaves are cut off onto
empty storage: a late reader fails loudly instead of silently reading the
next microbatch's data.

Where a slot's contents can legitimately outlive the modeled interval, compute
gets a copy instead: output gradients (autograd may hand the gradient tensor
itself to ``parameter.grad`` or to the input gradient that SEND_B ships), and
the last stage's inputs when its outputs are kept for the merge after the
step (``return_outputs``): they may alias the inputs.

Only _PipelineScheduleRuntime subclasses expose and execute this action list.

Private torch API this module relies on (beyond ``PipelineStage`` methods
``RecvPoolStage`` overrides): ``_PipelineScheduleRuntime`` with
``pipeline_order_with_comms``, ``_stages`` and ``_n_microbatches``;
``_Action`` fields (``stage_index``, ``computation_type``,
``microbatch_index``, ``sub_actions``); ``_ComputationType`` members;
``_RecvInfo`` and ``args_recv_info``/``grad_recv_info`` as
``dict[mb] -> tuple`` of infos with a ``buffer`` tensor;
``_setup_backward_recv_info`` recording its allocation count in
``self.chunks``.
"""

from collections.abc import Sequence
from heapq import heappop, heappush
from typing import cast

from torch import Tensor
from torch.distributed.pipelining.schedules import (
    _Action,
    _ComputationType,
    _PipelineScheduleRuntime,
)
from torch.distributed.pipelining.stage import PipelineStage

from torchtitan.tools.logging import logger

_FWD_RECV = _ComputationType.RECV_F
_BWD_RECV = _ComputationType.RECV_B
# Compute actions that read the *stage input* (forward recv buffer) of their
# microbatch. Weight backward is included: computing weight gradients reads
# the input. SEND_F is included because a stage output may alias the stage
# input (passthrough, e.g. the MTP input_embeds carry), so the send posts a
# read of the fwd recv buffer. Note the interval model can only order the
# reusing RECV after the *posting* of that send; the send kernel itself is
# ordered before the reuse transitively, through backward: the reusing RECV
# waits for backward(i) on the compute stream, backward(i) needs the gradient
# from downstream, and downstream can only produce it after consuming our
# send(i). A forward-only run (``schedule.eval()``) has no such chain, so
# forward recvs are not pooled without backward.
_FWD_CONSUMERS = frozenset(
    {
        _ComputationType.FORWARD,
        _ComputationType.FULL_BACKWARD,
        _ComputationType.BACKWARD_INPUT,
        _ComputationType.BACKWARD_WEIGHT,
        _ComputationType.SEND_F,
    }
)
# Readers of the *output gradient* (backward recv buffer). RecvPoolStage hands
# compute a copy, so the slot itself is only read there; the later consumers
# keep the interval model conservative.
_BWD_CONSUMERS = frozenset(
    {
        _ComputationType.FULL_BACKWARD,
        _ComputationType.BACKWARD_INPUT,
        _ComputationType.BACKWARD_WEIGHT,
        _ComputationType.SEND_B,
    }
)


class RecvPoolStage(PipelineStage):
    """PipelineStage whose per-microbatch recv slots can share buffer sets.

    Dormant (stock behavior) until :func:`pool_pipeline_recv_buffers` assigns
    the slot maps. Recv infra is set up lazily on first use and re-set-up on
    train/eval mode switches; overriding the setup methods keeps the pool
    applied across re-inits and keeps the private-API surface visible to type
    checkers, unlike per-instance monkeypatching.
    """

    # microbatch -> slot, per direction; None means stock allocation.
    _fwd_recv_slots: dict[int, int] | None = None
    _bwd_recv_slots: dict[int, int] | None = None
    # microbatch -> the slot's previous owner (circular across steps), only for
    # slots shared by several microbatches.
    _fwd_reuse_prev: dict[int, int] | None = None
    # microbatch -> leaves handed to compute, while fwd is pooled.
    _fwd_lease: dict[int, tuple[Tensor, ...]] | None = None
    _bwd_pooled: bool = False
    _save_output: bool = False

    def _setup_forward_recv_info(
        self, num_microbatches: int, has_backward: bool
    ) -> None:
        slots = self._fwd_recv_slots
        self._fwd_lease = None
        # Forward-only runs have no backward chain ordering SEND before reuse.
        if (
            slots is None
            or not has_backward
            or set(slots) != set(range(num_microbatches))
        ):
            # pyrefly: ignore [missing-attribute]
            return super()._setup_forward_recv_info(num_microbatches, has_backward)
        # pyrefly: ignore [missing-attribute]
        super()._setup_forward_recv_info(max(slots.values()) + 1, has_backward)
        self.args_recv_info = {
            mb: self.args_recv_info[slot] for mb, slot in slots.items()
        }
        self._fwd_lease = {}

    def _setup_backward_recv_info(self, num_microbatches: int) -> None:
        slots = self._bwd_recv_slots
        self._bwd_pooled = False
        if slots is None or set(slots) != set(range(num_microbatches)):
            # pyrefly: ignore [missing-attribute]
            return super()._setup_backward_recv_info(num_microbatches)
        # pyrefly: ignore [missing-attribute]
        super()._setup_backward_recv_info(max(slots.values()) + 1)
        # The original setup records the allocation count in self.chunks;
        # chunk-id validation needs the schedule's full microbatch count.
        self.chunks = num_microbatches
        self.grad_recv_info = {
            mb: self.grad_recv_info[slot] for mb, slot in slots.items()
        }
        self._bwd_pooled = True

    def forward_one_chunk(
        self, fwd_chunk_id, args, kwargs=None, save_forward_output=True
    ):
        self._save_output = self.is_last and save_forward_output
        return super().forward_one_chunk(
            fwd_chunk_id, args, kwargs, save_forward_output
        )

    def _retrieve_recv_activations(self, fwd_chunk_id: int):
        # pyrefly: ignore [missing-attribute]
        tensors = super()._retrieve_recv_activations(fwd_chunk_id)
        if self._fwd_lease is not None and self._save_output:
            return _copy(tensors)
        return _lease(self._fwd_lease, fwd_chunk_id, tensors)

    def _retrieve_recv_grads(self, bwd_chunk_id: int):
        # pyrefly: ignore [missing-attribute]
        tensors = super()._retrieve_recv_grads(bwd_chunk_id)
        return _copy(tensors) if self._bwd_pooled else tensors

    def get_fwd_recv_ops(self, fwd_chunk_id: int):
        _release(self._fwd_lease, self._fwd_reuse_prev, fwd_chunk_id)
        return super().get_fwd_recv_ops(fwd_chunk_id)


def _lease(
    lease: dict[int, tuple[Tensor, ...]] | None, mb: int, tensors: tuple
) -> tuple:
    fresh = tuple(
        t.detach().requires_grad_(t.requires_grad) if isinstance(t, Tensor) else t
        for t in tensors
    )
    if lease is not None:
        lease[mb] = tuple(t for t in fresh if isinstance(t, Tensor))
    return fresh


def _copy(tensors: tuple) -> tuple:
    return tuple(
        t.detach().clone().requires_grad_(t.requires_grad)
        if isinstance(t, Tensor)
        else t
        for t in tensors
    )


def _release(
    lease: dict[int, tuple[Tensor, ...]] | None,
    reuse_prev: dict[int, int] | None,
    mb: int,
) -> None:
    prev = (reuse_prev or {}).get(mb)
    if not lease or prev is None:
        return
    for t in lease.pop(prev, ()):
        t.grad = None
        t.data = t.new_empty(0)


def _reuse_chain(
    intervals: dict[int, tuple[int, int]], slots: dict[int, int]
) -> dict[int, int]:
    """Map each microbatch of a shared slot to the slot's previous owner.

    Owners are ordered by recv position; the chain is circular so that the
    first recv of a step cuts off the last owner of the previous step.
    """
    owners: dict[int, list[int]] = {}
    for mb in sorted(slots, key=lambda mb: intervals[mb][0]):
        owners.setdefault(slots[mb], []).append(mb)
    chain: dict[int, int] = {}
    for mbs in owners.values():
        if len(mbs) > 1:
            for prev, mb in zip([mbs[-1]] + mbs[:-1], mbs):
                chain[mb] = prev
    return chain


def _flatten_actions(actions: Sequence[_Action | None]) -> list[_Action]:
    """Expand composite actions (e.g. OVERLAP_F_B) into their compute actions."""
    flat: list[_Action] = []
    for action in actions:
        if action is None:
            continue
        flat.extend(action.sub_actions or [action])
    return flat


def _busy_intervals(
    actions: list[_Action],
    stage_index: int,
    recv_type: _ComputationType,
    consumer_types: frozenset[_ComputationType],
    local_stage_indices: set[int],
) -> dict[int, tuple[int, int]] | None:
    """Map microbatch -> (recv position, last-read position) in the action list.

    Returns None when this stage posts no recvs of the given type (first/last
    stage, same-rank V-schedule neighbor, or no-backward run) — nothing to pool.
    """
    # Local handoffs can preserve storage: e.g. Bias backward passes its recv
    # gradient to the preceding V-stage, which still needs it for weight backward.
    # Remote handoffs copy through P2P and end this chain of potential aliases.
    consumer_stages = {stage_index}
    direction = 1 if recv_type == _FWD_RECV else -1
    neighbor = stage_index + direction
    while neighbor in local_stage_indices:
        consumer_stages.add(neighbor)
        neighbor += direction

    starts: dict[int, int] = {}
    ends: dict[int, int] = {}
    for pos, action in enumerate(actions):
        if action.microbatch_index is None:
            continue
        mb = action.microbatch_index
        if action.stage_index == stage_index and action.computation_type == recv_type:
            starts[mb] = pos
        elif (
            action.stage_index in consumer_stages
            and action.computation_type in consumer_types
        ):
            ends[mb] = max(ends.get(mb, -1), pos)
    if not starts:
        return None
    missing_end = [mb for mb in starts if mb not in ends]
    if missing_end:
        raise ValueError(
            f"Stage {stage_index}: recv posted but no consuming compute action "
            f"found for microbatches {sorted(missing_end)[:4]}..."
        )
    return {mb: (starts[mb], ends[mb]) for mb in starts}


def _assign_slots(intervals: dict[int, tuple[int, int]]) -> dict[int, int]:
    """Assign the minimum number of slots in O(n log n), regardless of MB order."""
    slots: dict[int, int] = {}
    busy: list[tuple[int, int]] = []  # (last read, slot), one entry per slot
    for mb, (recv, last_read) in sorted(intervals.items(), key=lambda item: item[1][0]):
        slot = heappop(busy)[1] if busy and busy[0][0] < recv else len(busy)
        slots[mb] = slot
        heappush(busy, (last_read, slot))
    return slots


def pool_pipeline_recv_buffers(schedule: _PipelineScheduleRuntime) -> None:
    """Assign pooled recv slot maps to all local stages of the schedule.

    Must be called after the schedule is constructed (its lowered action list
    exists) and before the first ``step()`` (recv infra is set up lazily on
    first use). Stages must be :class:`RecvPoolStage`; otherwise a warning is
    logged and stock allocation is kept.
    """
    stages = cast(list[PipelineStage], schedule._stages)
    if not all(isinstance(stage, RecvPoolStage) for stage in stages):
        logger.warning(
            "Recv buffer pooling needs RecvPoolStage stages; got %s, so recv "
            "buffer pooling is disabled.",
            sorted({type(stage).__name__ for stage in stages}),
        )
        return
    actions = _flatten_actions(schedule.pipeline_order_with_comms[stages[0].group_rank])
    num_microbatches = schedule._n_microbatches
    local_stage_indices = {stage.stage_index for stage in stages}

    for stage in stages:
        counts = {"fwd": 0, "bwd": 0}  # 0 = this direction posts no recvs
        for direction, recv_type, consumers in (
            ("fwd", _FWD_RECV, _FWD_CONSUMERS),
            ("bwd", _BWD_RECV, _BWD_CONSUMERS),
        ):
            intervals = _busy_intervals(
                actions, stage.stage_index, recv_type, consumers, local_stage_indices
            )
            if intervals is None:
                continue
            # Stock allocation unless pooling is installed below.
            counts[direction] = num_microbatches
            if set(intervals) != set(range(num_microbatches)):
                logger.warning(
                    "Stage %d: %s recvs cover %d of %d microbatches; "
                    "skipping recv buffer pooling for this direction.",
                    stage.stage_index,
                    recv_type,
                    len(intervals),
                    num_microbatches,
                )
                continue
            slots = _assign_slots(intervals)
            num_slots = max(slots.values()) + 1
            if num_slots < num_microbatches:
                setattr(stage, f"_{direction}_recv_slots", slots)
                if direction == "fwd":
                    stage._fwd_reuse_prev = _reuse_chain(intervals, slots)
                counts[direction] = num_slots
        logger.info(
            "PP recv buffer pool: stage %d fwd=%d bwd=%d (microbatches=%d)",
            stage.stage_index,
            counts["fwd"],
            counts["bwd"],
            num_microbatches,
        )
