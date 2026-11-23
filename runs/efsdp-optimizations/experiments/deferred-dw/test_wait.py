"""Test the real lease and slot owner with CPU tensors and explicit stream logs."""

from unittest.mock import Mock

import pytest
import torch

from component import OFFSETS, SHAPES, SLOT_BYTES, make_group, warm_slot
from torchtitan.distributed.gfsdp._param_group import GFSDPBufferContext
from torchtitan.ops.float8 import direct_expert_dw as direct


@pytest.fixture
def lease_env(monkeypatch):
    log = []

    class Stream:
        def wait_event(self, event):
            log.append(("wait", self, event))

        def record_event(self):
            event = object()
            log.append(("record", self, event))
            return event

    original = Stream()
    current = Mock(return_value=original)
    monkeypatch.setattr(torch.cuda, "current_stream", current)
    monkeypatch.setattr(direct, "_slot_key", lambda slot: (0, slot.data_ptr()))
    monkeypatch.setattr(direct, "_leases", {})

    def writer(a, b, destination, ks, ks_tensor):
        log.append(("native", original, destination.shape))
        destination.fill_(3.25)

    monkeypatch.setattr(direct, "_native_writer", writer)
    storage = GFSDPBufferContext(GFSDPBufferContext.BufferType.REDUCE_SCATTER)
    storage.lazy_init({None: SLOT_BYTES * 2}, torch.device("cpu"))
    claim = storage.region({None: SLOT_BYTES}, {None: 0})
    overlap = storage.region({None: SLOT_BYTES}, {None: 0})
    disjoint = storage.region({None: SLOT_BYTES}, {None: SLOT_BYTES})
    slot = claim.buffers[None].view(torch.float32).fill_(-2.0)
    group = make_group(claim, slot)
    return group, slot, storage, overlap, disjoint, log, original, current


def write_projection(slot, index, *, bad_scale=False):
    _, m, n = SHAPES[index]
    counts = torch.tensor([128, 0], dtype=torch.int32)
    a = torch.empty(128 * m)
    b = torch.empty(128 * n)
    a_scale = torch.empty(m) if bad_scale else torch.empty(m, 1)
    direct._write_direct_expert_dw(
        a, a_scale, b, torch.empty(n, 1), slot, OFFSETS[index], None, counts, counts
    )


@pytest.mark.parametrize("zero_first", [False, True])
def test_deferred_once_at_first_writer(lease_env, zero_first):
    group, slot, storage, overlap, disjoint, log, compute, _ = lease_env
    first, second = object(), object()
    storage.release_event = first
    group.grad_buffer_ctx.release_event = first  # The same event must be joined once.
    overlap.release_event = second
    disjoint.release_event = object()
    group._acquire_direct_expert_dw(slot)
    lease = group._direct_expert_dw_lease
    assert group.grad_buffer_ctx.owner is group
    assert not log

    disjoint.release_event = storage.release_event = object()
    overlap.release_event = group.grad_buffer_ctx.release_event = None
    with pytest.raises(ValueError, match="outside the lease"):
        direct._zero_direct_expert_dw(slot, 1, torch.empty(0))
    with pytest.raises(IndexError):
        write_projection(slot, 0, bad_scale=True)
    assert not log
    assert torch.all(slot == -2).item()

    expected = [("wait", compute, first), ("wait", compute, second)]
    if zero_first:
        direct._zero_direct_expert_dw(slot, OFFSETS[0], torch.empty(0))
        assert log == expected
        assert torch.count_nonzero(slot[: OFFSETS[1]]).item() == 0
    write_projection(slot, int(zero_first))
    assert log == [*expected, ("native", compute, torch.Size(SHAPES[int(zero_first)]))]
    if not zero_first:
        direct._zero_direct_expert_dw(slot, OFFSETS[1], torch.empty(0))
    assert len(log) == 3
    group._finish_direct_expert_dw(lease)
    assert all(
        param.unsharded_accumulated_grad is param._unsharded_accumulated_grad
        for param in group.fsdp_params
    )
    assert group.grad_buffer_ctx.owner is group
    assert group._direct_expert_dw_lease is None


@pytest.mark.parametrize("write_first", [False, True])
@pytest.mark.parametrize("wrong_stream", [False, True])
def test_abort_orders_previous_release_before_new_fence(
    lease_env, write_first, wrong_stream
):
    group, slot, storage, overlap, _, log, compute, current = lease_env
    first, second = object(), object()
    storage.release_event, overlap.release_event = first, second
    group._acquire_direct_expert_dw(slot)
    lease = group._direct_expert_dw_lease
    assert not log
    if write_first:
        write_projection(slot, 0)
    if wrong_stream:
        current.return_value = object()
        log_before = log.copy()
        with pytest.raises(RuntimeError, match="different CUDA stream"):
            direct._zero_direct_expert_dw(slot, OFFSETS[1], torch.empty(0))
        assert log == log_before
    with pytest.raises(RuntimeError, match="different CUDA stream|one write"):
        group._finish_direct_expert_dw(lease)
    expected = [("wait", compute, first), ("wait", compute, second)]
    if write_first:
        expected.append(("native", compute, torch.Size(SHAPES[0])))
    assert log == [*expected, ("record", compute, group.grad_buffer_ctx.release_event)]
    assert group.grad_buffer_ctx.owner is None
    assert group._direct_expert_dw_lease is None
    assert all(param.unsharded_accumulated_grad is None for param in group.fsdp_params)
    assert not direct._leases


def test_stale_lease_cannot_fence_replacement_and_no_event_slot_needs_no_wait(
    lease_env,
):
    group, slot, _, _, _, log, compute, _ = lease_env
    group._acquire_direct_expert_dw(slot)
    old = group._direct_expert_dw_lease
    for offset in OFFSETS:
        direct._zero_direct_expert_dw(slot, offset, torch.empty(0))
    group._finish_direct_expert_dw(old)
    assert not log
    assert group.grad_buffer_ctx.release(group, None)
    for param in group.fsdp_params:
        param.unsharded_accumulated_grad = None
    group._acquire_direct_expert_dw(slot)
    current = group._direct_expert_dw_lease
    assert old.abort() is None
    with pytest.raises(RuntimeError, match="stale"):
        group._finish_direct_expert_dw(old)
    assert direct._get_lease(slot) is current and group.grad_buffer_ctx.owner is group
    assert not log
    with pytest.raises(RuntimeError, match="one write"):
        group._finish_direct_expert_dw(current)
    assert log == [("record", compute, group.grad_buffer_ctx.release_event)]


def test_native_component_warm_slot_before_old_reader_claim(lease_env):
    group, slot, _, overlap, _, log, compute, _ = lease_env
    warm_slot(
        group,
        slot,
        compute,
        lambda: write_projection(slot, 0),
        lambda: direct._zero_direct_expert_dw(slot, OFFSETS[1], torch.empty(0)),
    )
    assert group.grad_buffer_ctx.owner is None and group._direct_expert_dw_lease is None
    assert all(param.unsharded_accumulated_grad is None for param in group.fsdp_params)
    assert log == [
        ("native", compute, torch.Size(SHAPES[0])),
        ("record", compute, group.grad_buffer_ctx.release_event),
    ]
    overlap.owner = object()
    with pytest.raises(RuntimeError, match="owned by"):
        group._acquire_direct_expert_dw(slot)
    assert overlap.release(overlap.owner, compute)
    group._acquire_direct_expert_dw(slot)
    assert group.grad_buffer_ctx.owner is group and log[-1][0] == "record"
    with pytest.raises(RuntimeError, match="one write"):
        group._finish_direct_expert_dw(group._direct_expert_dw_lease)
