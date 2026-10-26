# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the pieces llm-foundry's DSA stage-1 leans on.

Three of them had no coverage here at all, which meant a regression could only be
caught downstream, on a multi-node training run:

* ``is_recomputing()`` -- the hook a consumer uses to replay expensive gradient-free
  work instead of recomputing it. It has to be false on the real forward, true during
  the recompute and, because the recompute runs on the autograd worker thread, it has
  to be thread-local.
* the offload synchronizer's armed-slot bookkeeping -- a frozen layer prefix leaves
  boundaries with no autograd graph, so their slots are never visited by backward and
  stay dirty into the next cycle. Without the sweep this asserts on the *second* batch,
  which is why a single-step test cannot catch it.
* ``_report_grad_nan_origin`` -- must stay a no-op when GigaFSDP is not installed.
"""
import sys
import threading
import warnings
import types

import pytest
import torch

from composer.callbacks.weight_monitor import WeightMonitor
from composer.optim import decoupled_weight_decay
from composer.trainer import cpu_offload
from composer.trainer.torch_utils_checkpoint import checkpoint, is_recomputing


class TestIsRecomputing:

    def test_false_outside_any_checkpoint(self):
        assert is_recomputing() is False

    def test_false_on_the_real_forward_and_true_on_the_recompute(self):
        seen = []

        def fn(x):
            seen.append(is_recomputing())
            return x.sin()  # backward needs the saved input, so the recompute fires

        x = torch.randn(4, requires_grad=True)
        out = checkpoint(fn, x, use_reentrant=False)
        assert seen == [False]
        out.sum().backward()
        assert seen == [False, True]

    def test_restored_after_the_recompute(self):
        x = torch.randn(4, requires_grad=True)
        checkpoint(lambda t: t.sin(), x, use_reentrant=False).sum().backward()
        assert is_recomputing() is False

    def test_is_thread_local(self):
        """A concurrent forward on another thread must not observe the recompute flag."""
        entered = threading.Event()
        release = threading.Event()
        other = {}

        def fn(x):
            if is_recomputing():
                entered.set()
                release.wait(timeout=5)
            return x.sin()

        def observer():
            entered.wait(timeout=5)
            other['flag'] = is_recomputing()
            release.set()

        watcher = threading.Thread(target=observer)
        watcher.start()
        x = torch.randn(4, requires_grad=True)
        checkpoint(fn, x, use_reentrant=False).sum().backward()
        watcher.join(timeout=5)
        assert other['flag'] is False


class TestReportGradNanOrigin:

    def test_no_op_without_gigafsdp(self, monkeypatch):
        monkeypatch.setitem(sys.modules, 'giga_fsdp.giga_fsdp', None)
        decoupled_weight_decay._report_grad_nan_origin()  # must not raise

    def test_calls_through_when_gigafsdp_is_present(self, monkeypatch):
        called = []
        fake = types.ModuleType('giga_fsdp.giga_fsdp')
        fake.report_grad_nan_origin_and_arm = lambda: called.append(True)
        monkeypatch.setitem(sys.modules, 'giga_fsdp', types.ModuleType('giga_fsdp'))
        monkeypatch.setitem(sys.modules, 'giga_fsdp.giga_fsdp', fake)
        decoupled_weight_decay._report_grad_nan_origin()
        assert called == [True]


class TestWeightMonitorGather:
    """Without an expert-parallel group the metrics are already global per rank."""

    def _monitor(self):
        return WeightMonitor.__new__(WeightMonitor)

    def test_trivial_ep_group_returns_local_metrics_without_a_collective(self, monkeypatch):
        gathered = []
        monkeypatch.setattr(torch.distributed, 'is_available', lambda: True)
        monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
        monkeypatch.setattr(
            torch.distributed,
            'all_gather_object',
            lambda *args, **kwargs: gathered.append(True),
        )
        from composer.utils import dist as composer_dist
        monkeypatch.setattr(composer_dist, 'get_ep_group', lambda: None)
        monkeypatch.setattr(composer_dist, 'is_trivial_process_group', lambda group: True)

        out = self._monitor()._gather_weight_metrics({'a': torch.tensor(1.0)})
        assert out == {'a': 1.0}
        assert gathered == []

    def test_no_distributed_returns_local_metrics(self, monkeypatch):
        monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: False)
        out = self._monitor()._gather_weight_metrics({'a': torch.tensor(2.0)})
        assert out == {'a': 2.0}


def _ctx(num_layers=2, model_layers=3):
    # num_layers == model_layers - 1 only costs overlap, and it keeps the cycle short
    # enough that one wraparound is three forwards.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        ctx, sync = cpu_offload.get_cpu_offload_context(
            enabled=True,
            num_layers=num_layers,
            model_layers=model_layers,
        )
    push, _pop = ctx.get_hooks()
    return ctx, sync, push


def _offloadable():
    t = torch.randn(64, 64, device='cuda')
    t.activation_offloading = True
    return t


def _boundary():
    x = torch.randn(8, device='cuda', requires_grad=True)
    return x * 2.0  # non-leaf, so it carries a grad_fn


def _cycle(ctx, sync, push, arm=(True, True, True), backward=True):
    """One forward cycle over three slots; ``arm[i]`` false models a frozen boundary."""
    boundaries = []
    for slot in range(3):
        with ctx:
            push(_offloadable())
        if arm[slot]:
            y = _boundary()
            sync(y)
            boundaries.append(y)
    if backward:
        for y in boundaries:
            y.sum().backward()
    return boundaries


@pytest.mark.gpu
class TestOffloadArmedSlots:

    def test_a_frozen_slot_is_released_at_the_next_cycle(self):
        """The slot was pushed but never armed, so backward never visits it."""
        ctx, sync, push = _ctx()
        _cycle(ctx, sync, push, arm=(False, True, True), backward=True)
        with ctx:  # second batch must not assert on a dirty slot
            push(_offloadable())

    def test_slots_still_awaiting_backward_fail_fast(self):
        """Armed slots mean a live graph; releasing them would corrupt it."""
        ctx, sync, push = _ctx()
        _cycle(ctx, sync, push, arm=(True, True, True), backward=False)
        with pytest.raises(AssertionError, match='await'):
            with ctx:
                pass

    def test_ordinary_cycles_are_unaffected(self):
        ctx, sync, push = _ctx()
        for _ in range(3):
            _cycle(ctx, sync, push, arm=(True, True, True), backward=True)

    def test_a_graphless_boundary_is_passed_through(self):
        """A tensor with no grad_fn has nothing to synchronize on."""
        ctx, sync, push = _ctx()
        with ctx:
            push(_offloadable())
        frozen = torch.randn(8, device='cuda')  # leaf, requires_grad False
        assert sync(frozen) is frozen
