import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)

from torchtitan.callbacks.manager import CallbackManager
from torchtitan.config.decode import decode
from torchtitan.config.errors import ConfigFieldError
from torchtitan.config.schema.experiments.cuda_graph_campaign.observer import (
    CampaignTimingObserverConfig,
)
from torchtitan.experiments.cuda_graph_campaign.observer import (
    CampaignTimingObserver,
)


class FakeEvent:
    def __init__(self, value: float) -> None:
        self.value = value

    def elapsed_time(self, other) -> float:
        return (other.value - self.value) * 1000


class CPUObserver(CampaignTimingObserver):
    def __init__(self, config, *, max_elapsed=5.0):
        super().__init__(config)
        self.max_elapsed = max_elapsed
        self.barrier_count = 0
        self.sync_count = 0
        self.memory_reads = 0
        self._next_event = 0

    @staticmethod
    def _validate_timing_device(device):
        del device

    @staticmethod
    def _runtime_versions(device):
        del device
        return {"python": "test", "torch": "test", "cuda_build": None}

    def _align_ranks(self):
        self.barrier_count += 1

    def _new_cuda_event(self):
        event = FakeEvent(self._next_event / 10)
        self._next_event += 1
        return event

    @staticmethod
    def _record_cuda_event(event, device):
        del event, device

    def _synchronize_device(self, device):
        del device
        self.sync_count += 1

    def _memory_boundary(self, device):
        del device
        self.memory_reads += 1
        return {
            "current_allocated": self.memory_reads,
            "current_reserved": self.memory_reads + 10,
            "peak_allocated": self.memory_reads + 20,
            "peak_reserved": self.memory_reads + 30,
        }


def make_state(tmp_path, *, step=0, world_size=1):
    training = SimpleNamespace(
        local_batch_size=1,
        global_batch_size=4,
        seq_len=2,
    )
    config = SimpleNamespace(
        dump_folder=str(tmp_path),
        training=training,
        dataloader=SimpleNamespace(dataset="c4_test"),
        compile=SimpleNamespace(
            enable=True, components=["model", "loss"], backend="inductor"
        ),
        activation_checkpoint=SimpleNamespace(mode="selective"),
        fp8=SimpleNamespace(
            dense_include_linear_layers=r".*\.(w1|w2|w3|wo)$",
            dense_use_shared_quantizer=True,
            sparse_use_grouped_gemm=True,
        ),
    )
    parallel_dims = SimpleNamespace(
        world_size=world_size,
        pp=1,
        dp_replicate=1,
        dp_shard=2,
        cp=1,
        tp=1,
        ep=world_size,
        get_all_one_dimensional_meshes=lambda: {},
        world_mesh=SimpleNamespace(
            size=lambda: world_size, mesh=torch.arange(world_size)
        ),
    )
    return SimpleNamespace(
        step=step,
        device="cpu",
        config=config,
        parallel_dims=parallel_dims,
        model_parts=(),
        loss_fn_type=type("ChunkedCELoss", (), {}),
        gradient_accumulation_steps=2,
    )


def load_records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_config_validation_and_absent_callback_default():
    config = decode(
        CampaignTimingObserverConfig,
        {"start_step": 10, "update_count": 2},
    )
    assert config.output_dir == "cuda_graph_timing"
    for invalid in (
        {"start_step": True, "update_count": 2},
        {"start_step": 10, "update_count": 0},
    ):
        with pytest.raises(ConfigFieldError):
            CampaignTimingObserverConfig(**invalid)

    manager = CallbackManager({}, sharding_backend=None)
    assert list(manager.handler.callbacks) == []


def test_window_boundaries_count_global_tokens_once_and_use_slowest_rank(
    tmp_path, monkeypatch
):
    config = CampaignTimingObserverConfig(
        start_step=10, update_count=2, output_dir="observer"
    )
    observer = CPUObserver(config, max_elapsed=5.0)
    state = make_state(tmp_path, step=0, world_size=2)
    time_calls = []
    times = iter([10.0, 11.0, 13.0, 13.0, 14.0, 15.0])

    def timestamp():
        time_calls.append(None)
        return next(times)

    monkeypatch.setattr(
        "torchtitan.experiments.cuda_graph_campaign.observer.time.perf_counter",
        timestamp,
    )
    monkeypatch.setattr(
        "torchtitan.experiments.cuda_graph_campaign.observer.dist.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "torchtitan.experiments.cuda_graph_campaign.observer.dist.is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "torchtitan.experiments.cuda_graph_campaign.observer.dist.get_rank", lambda: 0
    )
    monkeypatch.setattr(
        "torchtitan.experiments.cuda_graph_campaign.observer.dist.get_world_size",
        lambda: 2,
    )
    reduced_payloads = []

    def reduce_slowest_rank(payload, *, op):
        reduced_payloads.append(payload.tolist())
        assert op == torch.distributed.ReduceOp.MAX
        payload[0] = max(float(payload[0]), 5.0)

    monkeypatch.setattr(
        "torchtitan.experiments.cuda_graph_campaign.observer.dist.all_reduce",
        reduce_slowest_rank,
    )

    observer.fit_start(state)
    observer.batch_start(SimpleNamespace(step=9))
    output = tmp_path / "observer" / "rank-00000.jsonl"
    assert not output.exists()

    observer.batch_start(SimpleNamespace(step=10, device="cpu"))
    observer.batch_end(SimpleNamespace(step=10, global_valid_tokens=8.0))
    observer.batch_start(SimpleNamespace(step=11, device="cpu"))
    observer.batch_end(SimpleNamespace(step=11, global_valid_tokens=8.0))
    observer.batch_start(SimpleNamespace(step=12, device="cpu"))
    observer.fit_end(SimpleNamespace(step=12))

    activation, window = load_records(output)
    assert (
        activation["training_contract"]["expected_global_valid_tokens_per_update"] == 8
    )
    assert activation["parallelism"]["process_groups"]["world"] == {
        "degree": 2,
        "global_ranks": [0, 1],
    }
    assert window["status"] == "complete"
    assert window["global_valid_tokens_per_update_this_rank"] == [8, 8]
    assert window["global_valid_tokens_total_this_rank"] == 16
    assert window["host_update_submission_spans_seconds"] == [1.0, 2.0]
    assert window["local_host_submission_elapsed_seconds"] == 3.0
    assert window["local_completed_window_elapsed_seconds"] == 3.0
    assert window["slowest_rank_completed_window_elapsed_seconds"] == 5.0
    assert window["global_valid_tokens_per_second"] == 3.2
    assert reduced_payloads == [[3.0, 0.0]]
    assert observer.barrier_count == 1
    assert observer.sync_count == 2
    assert observer.memory_reads == 2

    output_before_later_callbacks = output.read_bytes()
    counts_before_later_callbacks = (
        len(time_calls),
        observer._next_event,
        observer.barrier_count,
        observer.sync_count,
        observer.memory_reads,
        len(reduced_payloads),
        len(observer._errors),
    )
    for step in (13, 14):
        observer.batch_start(SimpleNamespace(step=step, device="cpu"))
        observer.batch_end(SimpleNamespace(step=step, global_valid_tokens=8.0))

    assert (
        len(time_calls),
        observer._next_event,
        observer.barrier_count,
        observer.sync_count,
        observer.memory_reads,
        len(reduced_payloads),
        len(observer._errors),
    ) == counts_before_later_callbacks
    assert output.read_bytes() == output_before_later_callbacks


def test_non_python_token_scalar_fails_without_reading_it(tmp_path, monkeypatch):
    config = CampaignTimingObserverConfig(
        start_step=10, update_count=1, output_dir="observer"
    )
    observer = CPUObserver(config)
    state = make_state(tmp_path)
    times = iter([1.0, 2.0, 2.0])
    monkeypatch.setattr(
        "torchtitan.experiments.cuda_graph_campaign.observer.time.perf_counter",
        lambda: next(times),
    )

    class NoReadTensor:
        def item(self):
            raise AssertionError("observer must not read tensor scalars")

    observer.fit_start(state)
    observer.batch_start(SimpleNamespace(step=10, device="cpu"))
    observer.batch_end(SimpleNamespace(step=10, global_valid_tokens=NoReadTensor()))
    with pytest.raises(RuntimeError, match="marked invalid"):
        observer.batch_start(SimpleNamespace(step=11, device="cpu"))

    _, window = load_records(tmp_path / "observer" / "rank-00000.jsonl")
    assert window["status"] == "invalid"
    assert window["global_valid_tokens_per_second"] is None
    assert "not the native Python scalar" in window["failure_reasons"][0]


def test_completed_window_includes_async_tail_but_excludes_setup_and_reduction(
    tmp_path, monkeypatch
):
    clock = [0.0]

    class DelayedCompletionObserver(CPUObserver):
        def _synchronize_device(self, device):
            del device
            self.sync_count += 1
            clock[0] += 10.0 if self.sync_count == 1 else 3.0

    config = CampaignTimingObserverConfig(
        start_step=10, update_count=1, output_dir="observer"
    )
    observer = DelayedCompletionObserver(config)
    state = make_state(tmp_path, world_size=2)
    monkeypatch.setattr(
        "torchtitan.experiments.cuda_graph_campaign.observer.time.perf_counter",
        lambda: clock[0],
    )
    monkeypatch.setattr(
        "torchtitan.experiments.cuda_graph_campaign.observer.dist.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "torchtitan.experiments.cuda_graph_campaign.observer.dist.is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "torchtitan.experiments.cuda_graph_campaign.observer.dist.get_rank", lambda: 0
    )
    monkeypatch.setattr(
        "torchtitan.experiments.cuda_graph_campaign.observer.dist.get_world_size",
        lambda: 2,
    )

    def slow_max_collective(payload, *, op):
        assert op == torch.distributed.ReduceOp.MAX
        clock[0] += 100.0

    monkeypatch.setattr(
        "torchtitan.experiments.cuda_graph_campaign.observer.dist.all_reduce",
        slow_max_collective,
    )

    observer.fit_start(state)
    observer.batch_start(SimpleNamespace(step=10, device="cpu"))
    start_time = clock[0]
    observer.batch_end(SimpleNamespace(step=10, global_valid_tokens=8))
    clock[0] = start_time + 2.0
    observer.batch_start(SimpleNamespace(step=11, device="cpu"))
    observer.fit_end(SimpleNamespace(step=11))

    _, window = load_records(tmp_path / "observer" / "rank-00000.jsonl")
    assert window["local_host_submission_elapsed_seconds"] == 2.0
    assert window["local_completed_window_elapsed_seconds"] == 5.0
    assert window["slowest_rank_completed_window_elapsed_seconds"] == 5.0
    assert window["global_valid_tokens_per_second"] == pytest.approx(1.6)
    assert observer.sync_count == 2
    assert clock[0] == 115.0


def test_missing_following_boundary_writes_incomplete_record_and_rejects(
    tmp_path, monkeypatch
):
    config = CampaignTimingObserverConfig(
        start_step=10, update_count=1, output_dir="observer"
    )
    observer = CPUObserver(config)
    state = make_state(tmp_path)
    times = iter([1.0])
    monkeypatch.setattr(
        "torchtitan.experiments.cuda_graph_campaign.observer.time.perf_counter",
        lambda: next(times),
    )

    observer.fit_start(state)
    observer.batch_start(SimpleNamespace(step=10, device="cpu"))
    observer.batch_end(SimpleNamespace(step=10, global_valid_tokens=8))
    with pytest.raises(RuntimeError, match="window is incomplete"):
        observer.fit_end(SimpleNamespace(step=10))

    _, window = load_records(tmp_path / "observer" / "rank-00000.jsonl")
    assert window["status"] == "incomplete"
    assert window["valid"] is False
    assert window["global_valid_tokens_per_second"] is None


def test_multi_rank_observer_requires_initialized_world_group(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "torchtitan.experiments.cuda_graph_campaign.observer.dist.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "torchtitan.experiments.cuda_graph_campaign.observer.dist.is_initialized",
        lambda: False,
    )
    state = make_state(tmp_path, world_size=2)
    observer = CampaignTimingObserver(
        CampaignTimingObserverConfig(start_step=1, update_count=1)
    )
    monkeypatch.setattr(observer, "_validate_timing_device", lambda device: None)

    with pytest.raises(RuntimeError, match="initialized distributed process group"):
        observer.fit_start(state)


def test_activation_inventory_uses_constructed_markers():
    fp8_linear_type = type(
        "FP8Linear",
        (nn.Module,),
        {"__module__": "torchtitan.ops.float8.cuda_gemm"},
    )
    grouped_type = type(
        "GroupedExperts",
        (nn.Module,),
        {"__module__": "torchtitan.models.common.moe.experts.gmm"},
    )
    dispatcher_type = type(
        "HybridEPTokenDispatcher",
        (),
        {"__module__": "torchtitan.models.common.moe.dispatch.hybridep"},
    )
    model = nn.Module()
    model.compiled = nn.Linear(1, 1)
    model.compiled._compiled_call_impl = object()
    model.checkpoint = checkpoint_wrapper(nn.Linear(1, 1))
    model.sharded_checkpoint = checkpoint_wrapper(nn.Linear(1, 1))
    # FSDP changes the wrapper's class to a subclass at runtime.
    model.sharded_checkpoint.__class__ = type(
        "FSDPCheckpointWrapper", (type(model.sharded_checkpoint),), {}
    )
    model.fp8 = fp8_linear_type()
    model.grouped = grouped_type()
    model.grouped._use_float8_grouped_gemm = True
    model.shared = nn.Module()
    model.shared.use_shared_quantizer = True
    model.moe = nn.Module()
    model.moe.token_dispatcher = dispatcher_type()
    model.moe.token_dispatcher._capacity_config = SimpleNamespace(comm_stream=True)

    markers = CampaignTimingObserver._module_markers((model,))
    assert markers["compiled_model_callables_installed"]["count"] == 1
    assert markers["activation_checkpoint_wrappers"]["count"] == 2
    assert markers["dense_fp8_linear_modules"]["count"] == 1
    assert markers["sparse_fp8_grouped_expert_modules"]["count"] == 1
    assert markers["dense_shared_quantizer_modules"]["count"] == 1
    assert markers["hybridep_dispatchers"]["examples"][0]["comm_stream"] is True


def test_activation_records_singleton_expert_fsdp_mesh(tmp_path):
    observer = CPUObserver(CampaignTimingObserverConfig(start_step=1, update_count=1))
    state = make_state(tmp_path, world_size=8)
    singleton = SimpleNamespace(size=lambda: 1, mesh=torch.tensor([0]))
    requested = []

    def optional_mesh(name):
        requested.append(name)
        assert name == "efsdp"
        return singleton

    state.parallel_dims.get_optional_mesh = optional_mesh
    activation = observer._build_activation_record(state)
    assert activation["parallelism"]["process_groups"]["efsdp"] == {
        "degree": 1,
        "global_ranks": [0],
    }
    assert requested == ["efsdp"]
