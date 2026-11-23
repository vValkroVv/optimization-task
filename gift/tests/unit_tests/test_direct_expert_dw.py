"""Exercise the direct expert dW dispatcher boundary and host lease."""

from math import prod
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from torchtitan.ops.float8 import direct_expert_dw as direct
from torchtitan.distributed.gfsdp._param_group import (
    GFSDPBufferContext,
    GFSDPParamGroup,
)


def test_fake_dispatch_accepts_mutable_slot():
    with FakeTensorMode():
        first_projection = 32 * 3072 * 4096
        slot = torch.empty(first_projection + 32 * 4096 * 1536, device="cuda")
        a = torch.empty(0, dtype=torch.float8_e4m3fn, device="cuda")
        scale = torch.empty((0, 0), device="cuda")
        ks = torch.empty(32, dtype=torch.int32, device="cuda")
        assert (
            direct.write_direct_expert_dw(a, scale, a, scale, slot, 0, None, None, ks)
            is None
        )
        assert direct.zero_direct_expert_dw(slot, first_projection, a) is None
    for name in ("write_direct_expert_dw", "zero_direct_expert_dw"):
        schema = getattr(torch.ops.torchtitan, name).default._schema
        written = [
            arg.name
            for arg in schema.arguments
            if arg.alias_info and arg.alias_info.is_write
        ]
        assert written == ["slot"]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("local_experts", [4, 16, 32])
def test_lease_and_writes(monkeypatch, reverse, local_experts):
    # Exercise the real host checks and write bodies with small CPU stand-ins.
    shapes = w13, w2 = (local_experts, 256, 128), (local_experts, 128, 128)
    monkeypatch.setattr(direct, "_native_writer", None)
    monkeypatch.setattr(direct, "_leases", {})
    monkeypatch.setattr(direct, "_slot_key", lambda slot: (0, slot.data_ptr()))
    stream = object()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _: stream)
    expected_k = [128] + [0] * (local_experts - 1)
    counts = torch.tensor(expected_k, dtype=torch.int32)
    calls = []

    def writer(a, b, destination, ks, ks_tensor):
        calls.append(destination.data_ptr())
        assert (a[0].numel(), b[0].numel()) == (128 * w13[1], 128 * w13[2])
        assert (a[1].shape, b[1].shape) == ((w13[1], 1), (w13[2], 1))
        assert ks == expected_k and ks_tensor is counts
        assert destination.shape == w13
        destination.fill_(2)

    direct.configure_direct_expert_dw(writer)
    offset_w13, offset_w2 = (prod(w2), 0) if reverse else (0, prod(w13))
    offsets = (offset_w13, offset_w2)
    backing = torch.full((4 + prod(w13) + prod(w2) + 4,), -1.0)
    slot = backing[4:-4]
    grad_output = torch.empty(0)
    lease = direct.acquire_direct_expert_dw(slot, offsets, shapes)
    with pytest.raises(RuntimeError, match="already has an active lease"):
        direct.acquire_direct_expert_dw(slot, offsets, shapes)
    with pytest.raises(ValueError, match="outside the lease"):
        direct._zero_direct_expert_dw(slot, 1, grad_output)
    with pytest.raises(RuntimeError, match="did not receive the leased"):
        direct._zero_direct_expert_dw(slot.clone(), offset_w2, grad_output)
    with pytest.raises(RuntimeError, match="one write for each projection"):
        lease.finish()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _: object())
    with pytest.raises(RuntimeError, match="different CUDA stream"):
        direct._zero_direct_expert_dw(slot, offset_w2, grad_output)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _: stream)
    a = torch.empty(256 * w13[1], dtype=torch.float8_e4m3fn)
    b = torch.empty(256 * w13[2], dtype=torch.float8_e4m3fn)
    sa = torch.empty((2, w13[1])).T
    sb = torch.empty((2, w13[2])).T
    direct._write_direct_expert_dw(a, sa, b, sb, slot, offset_w13, None, counts, counts)
    with pytest.raises(RuntimeError, match="was written twice"):
        direct._zero_direct_expert_dw(slot, offset_w13, grad_output)
    direct._zero_direct_expert_dw(slot, offset_w2, grad_output)
    lease.finish()
    assert calls == [slot.data_ptr() + offset_w13 * slot.element_size()]
    torch.testing.assert_close(
        slot[offset_w13 : offset_w13 + prod(w13)], torch.full((prod(w13),), 2.0)
    )
    torch.testing.assert_close(
        slot[offset_w2 : offset_w2 + prod(w2)], torch.zeros(prod(w2))
    )
    assert torch.all(backing[:4] == -1) and torch.all(backing[-4:] == -1)

    replacement = direct.acquire_direct_expert_dw(slot, offsets, shapes)
    lease.abort()
    with pytest.raises(RuntimeError, match="lease is stale"):
        lease.finish()
    replacement.abort()
    assert not direct._leases


@pytest.mark.parametrize("local_experts", [4, 16, 32])
def test_supports_crop_projection_shapes(local_experts):
    assert direct.is_supported_direct_expert_dw_shape((local_experts, 3072, 4096))
    assert direct.is_supported_direct_expert_dw_shape((local_experts, 4096, 1536))


@pytest.mark.parametrize(
    "shape", [(0, 128, 128), (1, 192, 128), (1, 128, 96), (2**24, 128, 128)]
)
def test_rejects_unsupported_projection_shapes(shape):
    assert not direct.is_supported_direct_expert_dw_shape(shape)


def test_lease_rejects_incomplete_or_overlapping_slots(monkeypatch):
    monkeypatch.setattr(direct, "_native_writer", Mock())
    shapes = ((4, 256, 128), (4, 128, 128))
    first, second = map(prod, shapes)
    slot = torch.empty(first + second)
    for invalid_slot, offsets in ((slot, (0, first - 1)), (slot[:-1], (0, first))):
        with pytest.raises(ValueError, match="exactly cover the flat slot"):
            direct.acquire_direct_expert_dw(invalid_slot, offsets, shapes)


@pytest.mark.parametrize("complete", [True, False])
def test_gfsdp_publishes_direct_gradients_only_after_lease_completion(
    monkeypatch, complete
):
    group = object.__new__(GFSDPParamGroup)
    group._module_fqn = "layers.7[efsdp]"
    ctx = GFSDPBufferContext(GFSDPBufferContext.BufferType.REDUCE_SCATTER)
    ctx._ranges = {None: (0, 8)}
    ctx.owner = None
    previous_release = object()
    ctx.release_event = previous_release
    stream = Mock()
    group.device_handle = SimpleNamespace(current_stream=lambda: stream)
    group.grad_buffer_ctx = ctx
    group.reduce_grads = True
    group._direct_expert_dw_lease = None
    group._direct_expert_dw_offsets = (0, 4)
    group._direct_expert_dw_shapes = ((1, 2, 2), (1, 2, 2))
    slot = torch.arange(8, dtype=torch.float32)
    group.fsdp_params = [
        SimpleNamespace(
            unsharded_param=SimpleNamespace(grad=None),
            unsharded_accumulated_grad=None,
            _unsharded_accumulated_grad=view,
        )
        for view in slot.split(4)
    ]
    lease = Mock()
    if not complete:
        lease.finish.side_effect = RuntimeError("unfinished direct gradient")

    def acquire(real_slot, offsets, shapes):
        stream.wait_event.assert_called_once_with(previous_release)
        assert real_slot is slot and offsets == (0, 4)
        assert shapes == ((1, 2, 2), (1, 2, 2))
        return lease

    monkeypatch.setattr(direct, "acquire_direct_expert_dw", acquire)
    group._acquire_direct_expert_dw(slot)
    assert ctx.owner is group and group._direct_expert_dw_lease is lease
    if complete:
        group._finish_direct_expert_dw(lease)
        assert ctx.owner is group
        for param in group.fsdp_params:
            assert param.unsharded_accumulated_grad is param._unsharded_accumulated_grad
        lease.abort.assert_not_called()
    else:
        with pytest.raises(RuntimeError, match="unfinished direct gradient"):
            group._finish_direct_expert_dw(lease)
        assert ctx.owner is None
        assert ctx.release_event is stream.record_event.return_value
        assert all(
            param.unsharded_accumulated_grad is None for param in group.fsdp_params
        )
        lease.abort.assert_called_once_with()
    lease.finish.assert_called_once_with()
    assert group._direct_expert_dw_lease is None
