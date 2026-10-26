"""WeightMonitor must never issue a collective that skipped ranks are expected to join.
with ep_size == 1 the "EP-FSDP" complement group is the whole world, 
so `_should_capture_expert_weights` was true only on global rank 0, and rank 0 alone entered a world-wide
`all_gather_object` in `_gather_weight_metrics` while every other rank had
already early-returned from `batch_end` and moved on to the next callback's
`all_reduce` (NCCL watchdog: rank 0 ALLGATHER 1->world vs others ALLREDUCE).
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

import torch


def _weight_monitor_path() -> Path:
    vendored = (
        Path(__file__).parents[3]
        / "contrib" / "composer" / "composer" / "callbacks" / "weight_monitor.py"
    )
    if vendored.is_file():
        return vendored
    spec = importlib.util.find_spec("composer.callbacks.weight_monitor")
    if spec is None or spec.origin is None:
        pytest.fail(
            "composer.callbacks.weight_monitor not found: vendored submodule "
            "is not checked out and composer is not installed (broken env)")
    return Path(spec.origin)


class _FakeGroup:
    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


class _Dist:
    """Configurable stand-in for composer.utils.dist group accessors."""

    def __init__(self):
        self.global_rank = 0
        self.world_size = 1
        self.ep_group = None
        self.ep_group_size = None
        self.ep_group_rank = None
        self.ep_fsdp_group_rank = None

    def get_global_rank(self):
        return self.global_rank

    def get_world_size(self):
        return self.world_size

    def get_ep_group(self):
        return self.ep_group

    def get_ep_group_size(self):
        return self.ep_group_size

    def get_ep_group_rank(self):
        return self.ep_group_rank

    def get_ep_fsdp_group_rank(self):
        return self.ep_fsdp_group_rank

    def is_trivial_process_group(self, group):
        return group is None or group.size() == 1


class _GlobalWeightLogState:
    def __init__(self):
        self.monitor = None
        self.enabled = False

    def set_monitor_variable(self, monitor):
        self.monitor = monitor

    def set_enable_monitor_variable(self, value):
        self.enabled = value


def _load_weight_monitor_module(monkeypatch):
    path = _weight_monitor_path()

    class _Callback:
        pass

    class _State:
        pass

    class _Logger:
        pass

    class _WandBLogger:
        pass

    class _TimeUnit:
        BATCH = "ba"
        EPOCH = "ep"

    class _Time:
        def __init__(self, value=1, unit=_TimeUnit.BATCH):
            self.value = value
            self.unit = unit

        @classmethod
        def from_input(cls, value, unit):
            if isinstance(value, cls):
                return value
            if isinstance(value, int):
                return cls(value, unit)
            if isinstance(value, str):
                digits = "".join(ch for ch in value if ch.isdigit())
                return cls(int(digits or 1), unit)
            return cls(1, unit)

        @classmethod
        def from_timestring(cls, value):
            digits = "".join(ch for ch in value if ch.isdigit())
            return cls(int(digits or 1), _TimeUnit.BATCH)

        def __lt__(self, other):
            return self.value < other.value

        def __str__(self):
            return f"{self.value}{self.unit}"

    composer = types.ModuleType("composer")
    composer_core = types.ModuleType("composer.core")
    composer_core.Callback = _Callback
    composer_core.State = _State
    composer_core.Time = _Time
    composer_core.TimeUnit = _TimeUnit
    composer_loggers = types.ModuleType("composer.loggers")
    composer_loggers.Logger = _Logger
    composer_wandb = types.ModuleType("composer.loggers.wandb_logger")
    composer_wandb.WandBLogger = _WandBLogger
    composer_utils = types.ModuleType("composer.utils")
    composer_utils.dist = _Dist()
    composer_utils.global_weightlog_state = _GlobalWeightLogState()

    for name, module in {
        "composer": composer,
        "composer.core": composer_core,
        "composer.loggers": composer_loggers,
        "composer.loggers.wandb_logger": composer_wandb,
        "composer.utils": composer_utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("_composer_weight_monitor_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_monitor(module):
    monitor = object.__new__(module.WeightMonitor)
    monitor.stat_list = ["amax"]
    monitor.weight_metrics = {}
    monitor.external_metrics = {}
    return monitor


def _configure_trivial_ep(dist, rank, world_size):
    """The topology of the hanging run: ep_size=1, ep_fsdp group = world."""
    dist.global_rank = rank
    dist.world_size = world_size
    dist.ep_group = None
    dist.ep_group_size = None
    dist.ep_group_rank = None
    dist.ep_fsdp_group_rank = rank


def _configure_real_ep(dist, rank, world_size, ep_size):
    """Row-major EP groups of consecutive ranks, column-major complement."""
    dist.global_rank = rank
    dist.world_size = world_size
    dist.ep_group = _FakeGroup(ep_size)
    dist.ep_group_size = ep_size
    dist.ep_group_rank = rank % ep_size
    dist.ep_fsdp_group_rank = rank // ep_size


def _forbid_collectives(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("collective issued where skipped ranks cannot join it")

    monkeypatch.setattr(torch.distributed, "all_gather_object", _boom)


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.parametrize("rank", [0, 1, 5])
def test_trivial_ep_gather_issues_no_collective(monkeypatch, rank):
    module = _load_weight_monitor_module(monkeypatch)
    _configure_trivial_ep(module.dist, rank=rank, world_size=256)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    _forbid_collectives(monkeypatch)

    monitor = _make_monitor(module)
    local = {"weights_amax/model.lm_head.weight": torch.tensor([3.0])}
    gathered = monitor._gather_weight_metrics(dict(local))
    assert gathered == {"weights_amax/model.lm_head.weight": 3.0}


@pytest.mark.unit
@pytest.mark.fast
def test_trivial_ep_captures_experts_on_rank0_only(monkeypatch):
    module = _load_weight_monitor_module(monkeypatch)

    _configure_trivial_ep(module.dist, rank=0, world_size=256)
    monitor = _make_monitor(module)
    assert monitor._should_capture_expert_weights() is True

    for rank in (1, 7, 255):
        _configure_trivial_ep(module.dist, rank=rank, world_size=256)
        assert monitor._should_capture_expert_weights() is False


@pytest.mark.unit
@pytest.mark.fast
def test_collective_participants_match_group_in_both_topologies(monkeypatch):
    """The set of ranks that reach the gather must exactly cover its group."""
    module = _load_weight_monitor_module(monkeypatch)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    gather_calls = []

    def fake_all_gather_object(output, obj, group=None):
        gather_calls.append(module.dist.global_rank)
        for i in range(len(output)):
            output[i] = obj if i == module.dist.ep_group_rank else {}

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)

    world_size, ep_size = 8, 4

    def capturing_ranks(configure, *args):
        ranks = set()
        for rank in range(world_size):
            configure(module.dist, rank, world_size, *args)
            monitor = _make_monitor(module)
            capture_dense = rank == 0
            capture_expert = monitor._should_capture_expert_weights()
            if capture_dense or capture_expert:
                monitor._gather_weight_metrics({})
                ranks.add(rank)
        return ranks

    # ep_size == 1: nobody may issue a collective, rank 0 logs alone.
    gather_calls.clear()
    trivial = capturing_ranks(lambda d, r, w: _configure_trivial_ep(d, r, w))
    assert trivial == {0}
    assert gather_calls == []

    # Real EP: the participants are exactly one full EP group (ranks 0..ep-1),
    # each issuing exactly one group-wide gather.
    gather_calls.clear()
    real = capturing_ranks(_configure_real_ep, ep_size)
    assert real == set(range(ep_size))
    assert sorted(gather_calls) == list(range(ep_size))


@pytest.mark.unit
@pytest.mark.fast
def test_real_ep_merge_keeps_metrics_from_all_ep_ranks(monkeypatch):
    module = _load_weight_monitor_module(monkeypatch)
    _configure_real_ep(module.dist, rank=0, world_size=8, ep_size=4)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def fake_all_gather_object(output, obj, group=None):
        assert group is module.dist.ep_group
        output[0] = obj
        for i in range(1, len(output)):
            output[i] = {f"weights_amax/experts.weight1__expert_idx_{i}": float(i)}

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)

    monitor = _make_monitor(module)
    local = {"weights_amax/experts.weight1__expert_idx_0": torch.tensor([9.0])}
    merged = monitor._gather_weight_metrics(dict(local))
    assert merged == {
        "weights_amax/experts.weight1__expert_idx_0": 9.0,
        "weights_amax/experts.weight1__expert_idx_1": 1.0,
        "weights_amax/experts.weight1__expert_idx_2": 2.0,
        "weights_amax/experts.weight1__expert_idx_3": 3.0,
    }
