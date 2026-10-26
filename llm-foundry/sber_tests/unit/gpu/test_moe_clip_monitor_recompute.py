"""Clip-count gate must be identical in forward and act-ckpt recompute (CheckpointError 13 vs 11)."""
import types

import pytest
import torch
import torch.utils.checkpoint as ckpt

from composer.utils.shared_state import global_actlog_state
from llmfoundry.models.layers.moe.experts import (
    _clip_monitor_enabled,
    _clip_monitor_reporting,
    fused_swiglu,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_gpu]


@pytest.fixture
def monitor_state():
    saved = (global_actlog_state.enable_monitor, global_actlog_state.monitor)
    fake = types.SimpleNamespace(is_forward=True)
    global_actlog_state.enable_monitor = True
    global_actlog_state.monitor = fake
    try:
        yield fake
    finally:
        global_actlog_state.enable_monitor, global_actlog_state.monitor = saved


def test_kernel_gate_is_recompute_invariant_but_reporting_is_not(monitor_state):
    monitor_state.is_forward = True
    assert _clip_monitor_enabled(10.0) and _clip_monitor_reporting(10.0)
    monitor_state.is_forward = False
    assert _clip_monitor_enabled(10.0), "kernel path must not change between forward and recompute"
    assert not _clip_monitor_reporting(10.0), "EP all_reduce must not run from the recompute"
    assert not _clip_monitor_enabled(0.0) and not _clip_monitor_reporting(0.0)


def _count_saved(fn):
    n = [0]
    with torch.autograd.graph.saved_tensors_hooks(lambda t: n.__setitem__(0, n[0] + 1) or t, lambda t: t):
        out = fn()
    out.sum().backward()
    return n[0]


def test_fused_swiglu_saved_tensor_count_depends_on_clip_counts():
    x = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    probs = torch.rand(256, 1, device="cuda", dtype=torch.bfloat16)
    counts = torch.zeros(2, dtype=torch.int32, device="cuda")
    without = _count_saved(lambda: fused_swiglu(x, probs, swiglu_limit=10.0, clip_counts=None))
    with_buf = _count_saved(lambda: fused_swiglu(x, probs, swiglu_limit=10.0, clip_counts=counts))
    assert with_buf != without


def test_checkpoint_recompute_with_monitor_toggled_matches_forward(monitor_state):
    x = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    probs = torch.rand(256, 1, device="cuda", dtype=torch.bfloat16)
    counts = torch.zeros(2, dtype=torch.int32, device="cuda")

    def region(inp):
        return fused_swiglu(inp, probs, swiglu_limit=10.0,
                            clip_counts=counts if _clip_monitor_enabled(10.0) else None)

    monitor_state.is_forward = True
    out = ckpt.checkpoint(region, x, use_reentrant=False)
    monitor_state.is_forward = False
    out.sum().backward()
    assert x.grad is not None
