"""GFSDP callback and shared-slot lifetime checks for chunked head backward."""

from types import SimpleNamespace

import pytest
import torch

from torchtitan.config.configs import ParallelismConfig
from torchtitan.distributed.gfsdp._common import TrainingState
from torchtitan.distributed.gfsdp._param_group import (
    GFSDPBufferContext,
    GFSDPParamGroup,
)
from torchtitan.distributed.gfsdp._state import GFSDPState, GFSDPStateContext
from torchtitan.distributed.gfsdp.fully_shard import GFSDPModule


class _ProfileHandle:
    def __init__(self):
        self.calls = 0

    def process_profiling_events(self):
        self.calls += 1


class _CallbackGroup:
    def __init__(self, training_state):
        self._training_state = training_state
        self.post_backward_calls = 0
        self.data_handle = _ProfileHandle()
        self.grad_handle = _ProfileHandle()
        self.data_buffer_ctx = SimpleNamespace(yccl_handles={None: self.data_handle})
        self.grad_buffer_ctx = SimpleNamespace(yccl_handles={None: self.grad_handle})

    def post_backward(self):
        self.post_backward_calls += 1
        self._training_state = TrainingState.POST_BACKWARD


class _CallbackState:
    def __init__(self, group, training_state):
        self._fsdp_param_groups = [group]
        self._training_state = training_state
        self.finalize_calls = 0

    def _finalize_backward(self):
        self.finalize_calls += 1


def _run_root_callback(*, retain_tail_weights, phases, is_last_backward=False):
    root = GFSDPState()
    root._state_ctx = GFSDPStateContext()
    root._state_ctx.retain_tail_weights = retain_tail_weights
    root._state_ctx.is_last_backward = is_last_backward
    root._state_ctx.post_backward_final_callback_queued = True
    root._comm_ctx = SimpleNamespace(post_forward_order=["preserve-until-final"])

    states = []
    groups = []
    for group_phase, state_phase in phases:
        group = _CallbackGroup(group_phase)
        state = _CallbackState(group, state_phase)
        groups.append(group)
        states.append(state)
    root._state_ctx.all_states = states
    root._root_post_backward_final_callback()
    return root, states, groups


def test_nonfinal_callback_keeps_nonparticipants_and_resets_participants():
    phases = [
        (TrainingState.PRE_BACKWARD, TrainingState.PRE_BACKWARD),
        (TrainingState.POST_BACKWARD, TrainingState.PRE_BACKWARD),
        (TrainingState.IDLE, TrainingState.IDLE),
        (TrainingState.FORWARD, TrainingState.FORWARD),
    ]
    root, states, groups = _run_root_callback(
        retain_tail_weights=True, phases=phases
    )

    assert [group.post_backward_calls for group in groups] == [1, 0, 0, 0]
    assert [group._training_state for group in groups] == [
        TrainingState.IDLE,
        TrainingState.IDLE,
        TrainingState.IDLE,
        TrainingState.FORWARD,
    ]
    assert [state._training_state for state in states] == [
        TrainingState.IDLE,
        TrainingState.IDLE,
        TrainingState.IDLE,
        TrainingState.FORWARD,
    ]
    assert [state.finalize_calls for state in states] == [0, 0, 0, 0]
    assert all(group.data_handle.calls == 1 for group in groups)
    assert all(group.grad_handle.calls == 1 for group in groups)
    assert root._comm_ctx.post_forward_order == ["preserve-until-final"]
    assert root._state_ctx.post_backward_final_callback_queued is False


def test_default_nonfinal_callback_keeps_the_existing_fallback_behavior():
    _, states, groups = _run_root_callback(
        retain_tail_weights=False,
        phases=[(TrainingState.IDLE, TrainingState.IDLE)],
    )

    assert groups[0].post_backward_calls == 1
    assert groups[0]._training_state == TrainingState.IDLE
    assert states[0]._training_state == TrainingState.IDLE


def test_tail_weight_retention_config_defaults_off():
    assert ParallelismConfig().gfsdp_retain_tail_weights is False


def test_tail_weight_retention_rejects_fsdp2():
    from torchtitan.models.gift_moe._base.parallelize import (
        _validate_tail_weight_retention_request,
    )

    with pytest.raises(ValueError, match="requires fsdp_backend='gfsdp2'"):
        _validate_tail_weight_retention_request(
            ParallelismConfig(
                fsdp_backend="fsdp2", gfsdp_retain_tail_weights=True
            )
        )


def test_root_tail_retention_policy_is_set_before_lazy_init():
    state = GFSDPState()

    class _Module:
        def _get_fsdp_state(self):
            return state

    GFSDPModule.set_retain_tail_weights(_Module(), True)
    assert state._state_ctx.retain_tail_weights is True

    state._is_root = True
    with pytest.raises(RuntimeError, match="before the first forward"):
        GFSDPModule.set_retain_tail_weights(_Module(), False)


def _storage_context():
    context = GFSDPBufferContext(GFSDPBufferContext.BufferType.ALL_GATHER)
    context._storage_ctx = context
    context._regions = [context]
    context._ranges = {None: (0, 8)}
    context.buffers = {None: torch.empty(8)}
    context.yccl_handles = {None: None}
    context.owner = None
    context.release_event = None
    return context


class _PendingGather:
    def __init__(self, events):
        self.events = events

    def wait(self):
        self.events.append("gather-complete")


class _RecordedStream:
    def __init__(self, events):
        self.events = events

    def record_event(self):
        event = object()
        self.events.append(("release-published", event))
        return event


def _lifecycle_group(context, pending, events, *, retain_tail_weights=True):
    group = object.__new__(GFSDPParamGroup)
    group._module_fqn = "test.group"
    group.data_buffer_ctx = context
    group._all_gather_result = pending
    group._post_reduce_event = None
    group._post_forward_indices = []
    group._bf16_cache_enabled = False
    group._fp8_cache = None
    group._all_gather_input_set_dtypes = set()
    group.comm_ctx = SimpleNamespace(retain_tail_weights=retain_tail_weights)
    group.device_handle = SimpleNamespace(
        current_stream=lambda: _RecordedStream(events)
    )
    return group


def test_pending_prefetch_finalization_drains_before_owner_release():
    context = _storage_context()
    events = []
    group = _lifecycle_group(context, _PendingGather(events), events)
    context.owner = group

    group.finalize_backward()

    assert events[0] == "gather-complete"
    assert events[1][0] == "release-published"
    assert context.release_event is events[1][1]
    assert context.owner is None
    assert group._all_gather_result is None


def test_reshard_drains_pending_gather_before_publishing_release_event():
    context = _storage_context()
    events = []
    group = _lifecycle_group(context, _PendingGather(events), events)
    group._training_state = TrainingState.IDLE
    group._to_sharded = lambda: events.append("parameters-resharded")
    context.owner = group

    group.reshard()

    assert events[0] == "gather-complete"
    assert events[1] == "parameters-resharded"
    assert events[2][0] == "release-published"
    assert context.release_event is events[2][1]
    assert context.owner is None
    assert group._all_gather_result is None


def test_default_finalization_keeps_its_original_pending_gather_release_path():
    context = _storage_context()
    events = []
    group = _lifecycle_group(
        context, _PendingGather(events), events, retain_tail_weights=False
    )
    context.owner = group

    group.finalize_backward()

    assert events == ["gather-complete"]
    assert context.owner is None
    assert context.release_event is None
    assert group._all_gather_result is None


def test_pending_gather_cannot_wait_under_a_conflicting_shared_slot_owner():
    storage = _storage_context()
    target_context = storage.region({None: 8}, {None: 0})
    events = []
    group = _lifecycle_group(target_context, _PendingGather(events), events)
    storage.owner = object()

    with pytest.raises(RuntimeError, match="data buffer is in use"):
        group.unshard()
    with pytest.raises(RuntimeError, match="data buffer is in use"):
        group.wait_for_unshard()

    assert events == []
    assert group._all_gather_result is not None


def test_buffer_release_does_not_clear_another_groups_owner_or_publish_event():
    context = _storage_context()
    other_owner = object()
    context.owner = other_owner
    events = []

    assert context.release(object(), _RecordedStream(events)) is False
    assert context.owner is other_owner
    assert context.release_event is None
    assert events == []
