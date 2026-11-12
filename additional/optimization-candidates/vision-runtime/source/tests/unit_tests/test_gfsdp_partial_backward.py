"""An inner CE backward must leave unused decoder units live."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from torchtitan.distributed.gfsdp._state import GFSDPState, TrainingState


@pytest.mark.parametrize(
    ("last", "phase", "calls", "expected"),
    [
        (False, TrainingState.FORWARD, 0, TrainingState.FORWARD),
        (False, TrainingState.IDLE, 0, TrainingState.IDLE),
        (False, TrainingState.PRE_BACKWARD, 1, TrainingState.IDLE),
        (False, TrainingState.POST_BACKWARD, 0, TrainingState.IDLE),
        (True, TrainingState.FORWARD, 1, TrainingState.IDLE),
        (True, TrainingState.IDLE, 1, TrainingState.IDLE),
    ],
)
def test_partial_backward_only_finishes_participating_groups(
    last, phase, calls, expected
):
    group = SimpleNamespace(
        _training_state=phase,
        post_backward=Mock(),
        data_buffer_ctx=SimpleNamespace(yccl_handles={}),
        grad_buffer_ctx=None,
    )
    child = SimpleNamespace(
        _fsdp_param_groups=[group],
        _training_state=phase,
        _finalize_backward=Mock(),
    )
    root = GFSDPState()
    root._comm_ctx.post_forward_order = []
    root._state_ctx.all_states = [child]
    root._state_ctx.is_last_backward = last
    root._root_post_backward_final_callback()
    assert group.post_backward.call_count == calls
    assert group._training_state == expected
    assert child._finalize_backward.call_count == int(last)
    assert not root._state_ctx.post_backward_final_callback_queued


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_chunked_head_restores_gfsdp_backward_boundary(monkeypatch, initial, fail):
    import torch
    import torchtitan.components.loss as loss
    import torchtitan.distributed.gfsdp as api

    head = torch.nn.Linear(1, 1)
    context = SimpleNamespace(is_last_backward=initial)
    state = SimpleNamespace(_state_ctx=context)
    head._get_fsdp_state = lambda: state
    head.set_is_last_backward = lambda value: setattr(
        context, "is_last_backward", value
    )
    group = SimpleNamespace(
        post_forward_mesh_info=object(),
        reshard_after_backward=True,
        reduce_grads=True,
    )
    monkeypatch.setattr(loss, "FSDPModule", type(head))
    monkeypatch.setattr(api, "GFSDPModule", type(head))
    monkeypatch.setattr(loss, "_get_fsdp_param_groups", lambda _: [group])
    try:
        with loss._chunked_fsdp_head(head, requires_grad=True) as chunks:
            assert context.is_last_backward is False
            chunks[0].last_chunk()
            assert context.is_last_backward is False
            if fail:
                raise RuntimeError("chunk failed")
    except RuntimeError as error:
        assert fail and str(error) == "chunk failed"
    assert context.is_last_backward is initial
    assert group.reduce_grads is True
    assert group.reshard_after_backward is True
