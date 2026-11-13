from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import measure

LOCAL_PHYSICAL_PER_UPDATE = 262_144
GLOBAL_SUPERVISED_PER_UPDATE = 2_096_640


class FakeParameter:
    def __init__(self, *, requires_grad: bool, local_numel: int, dtype: str) -> None:
        self.requires_grad = requires_grad
        self._local_numel = local_numel
        self.dtype = dtype

    def to_local(self) -> FakeParameter:
        return self

    def numel(self) -> int:
        return self._local_numel


class FakePart:
    def __init__(self, parameters: list[FakeParameter]) -> None:
        self._parameters = parameters

    def parameters(self) -> list[FakeParameter]:
        return self._parameters


class FakeScalar:
    def __init__(self) -> None:
        self.value = 0

    def item(self) -> int:
        return self.value


class FakeMetricsProcessor:
    def __init__(self) -> None:
        self.ntokens_since_last_log = 0

    def interval_metrics(self) -> int:
        value = self.ntokens_since_last_log
        self.ntokens_since_last_log = 0
        return value


class FakeProfiler:
    def step(self) -> None:
        return None


class FakeRuntime:
    def __init__(self) -> None:
        self.stats = self._reset_stats(0, 0)

    @staticmethod
    def _reset_stats(retries: int, ooms: int) -> dict[str, int]:
        return {
            "allocated_bytes.all.peak": 100,
            "active_bytes.all.peak": 200,
            "reserved_bytes.all.peak": 300,
            "num_alloc_retries": retries,
            "num_ooms": ooms,
        }

    def memory_stats(self) -> dict[str, int]:
        return dict(self.stats)

    def reset_peak_stats(self) -> None:
        retries = self.stats["num_alloc_retries"]
        ooms = self.stats["num_ooms"]
        self.stats = self._reset_stats(retries, ooms)


def _fake_classes(stop_after: int):
    runtime = FakeRuntime()
    metrics_module = SimpleNamespace(reset_peak_stats=runtime.reset_peak_stats)

    class FakeTrainer:
        def __init__(self) -> None:
            self.step = 0
            self.metrics_processor = FakeMetricsProcessor()
            self.ntokens_seen = FakeScalar()
            self.model_parts = [
                FakePart(
                    [
                        FakeParameter(
                            requires_grad=True,
                            local_numel=17,
                            dtype="float32",
                        ),
                        FakeParameter(
                            requires_grad=False,
                            local_numel=5,
                            dtype="bfloat16",
                        ),
                    ]
                )
            ]
            self.profiler = FakeProfiler()

        def forward_backward_step(self) -> None:
            return None

        def train(self) -> None:
            for step in range(1, stop_after + 1):
                self.step = step
                self.metrics_processor.ntokens_since_last_log += LOCAL_PHYSICAL_PER_UPDATE
                self.ntokens_seen.value += GLOBAL_SUPERVISED_PER_UPDATE
                self.metrics_processor.interval_metrics()
                self.runtime.stats = {
                    "allocated_bytes.all.peak": 1_000 + step,
                    "active_bytes.all.peak": 2_000 + step,
                    "reserved_bytes.all.peak": 3_000 + step,
                    "num_alloc_retries": max(0, step - 20),
                    "num_ooms": int(step >= 24),
                }
                self.metrics_module.reset_peak_stats()
                for _ in range(32):
                    self.forward_backward_step()
                self.profiler.step()

    FakeTrainer.runtime = runtime
    FakeTrainer.metrics_module = metrics_module
    return FakeTrainer, FakeProfiler, FakeMetricsProcessor, metrics_module, runtime


def _adapter(
    tmp_path: Path,
    *,
    steps: int,
    warmup: int,
    window: int,
    monitor_interval: int,
    capture_microbatches: bool,
):
    trainer_class, profiler_class, metrics_class, metrics_module, runtime = _fake_classes(steps)
    sync_calls: list[int] = []
    adapter = measure.NativeAdapter(
        trainer_class=trainer_class,
        profiler_class=profiler_class,
        metrics_class=metrics_class,
        metrics_module=metrics_module,
        options={
            "run_id": "unit-test",
            "output_dir": tmp_path / "measurement",
            "expected_steps": steps,
            "warmup_updates": warmup,
            "window_updates": window,
            "monitor_interval": monitor_interval,
            "capture_microbatches": capture_microbatches,
        },
        cuda_synchronize=lambda: sync_calls.append(1),
        memory_stats=runtime.memory_stats,
    )
    return trainer_class, adapter, runtime, sync_calls


def test_smoke_mode_captures_native_updates_without_adapter_syncs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    trainer_class, adapter, _, sync_calls = _adapter(
        tmp_path,
        steps=10,
        warmup=0,
        window=0,
        monitor_interval=5,
        capture_microbatches=True,
    )

    with adapter:
        trainer_class().train()

    record = json.loads((tmp_path / "measurement/rank-00.json").read_text())
    assert record["training_status"] == "complete"
    assert record["measurement_status"] == "disabled"
    assert record["measurement"] is None
    assert len(record["cadence"]) == 10
    assert record["microbatch_calls_by_step"] == {str(step): 32 for step in range(1, 11)}
    assert sync_calls == []


def test_full_window_uses_two_boundaries_and_tracks_native_resets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    trainer_class, adapter, _, sync_calls = _adapter(
        tmp_path,
        steps=70,
        warmup=20,
        window=50,
        monitor_interval=50,
        capture_microbatches=False,
    )

    with adapter:
        trainer_class().train()

    record = json.loads((tmp_path / "measurement/rank-00.json").read_text())
    window = record["measurement"]
    assert record["training_status"] == "complete"
    assert record["measurement_status"] == "complete"
    assert sync_calls == [1, 1]
    assert window["cuda_synchronize_calls"] == 2
    assert window["start_after_update"] == 20
    assert window["end_after_update"] == 70
    assert window["monitoring_updates"] == [50]
    assert window["physical_input_tokens_local"] == 50 * LOCAL_PHYSICAL_PER_UPDATE
    assert window["supervised_tokens_global"] == 50 * GLOBAL_SUPERVISED_PER_UPDATE
    assert window["memory"]["peak_bytes"] == {
        "allocated_bytes.all.peak": 1_070,
        "active_bytes.all.peak": 2_070,
        "reserved_bytes.all.peak": 3_070,
    }
    assert window["memory"]["allocator_counter_deltas"] == {
        "num_alloc_retries": 50,
        "num_ooms": 1,
    }
    assert record["parameter_census"]["trainable"]["local_numel"] == 17
    assert record["parameter_census"]["frozen"]["local_numel"] == 5


def test_incomplete_window_is_written_as_incomplete_and_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    short_class, short_adapter, _, short_sync_calls = _adapter(
        tmp_path / "short",
        steps=70,
        warmup=20,
        window=50,
        monitor_interval=50,
        capture_microbatches=False,
    )
    short_class.train = lambda trainer: _run_short(trainer, 40)
    with short_adapter, pytest.raises(measure.IncompleteWindowError):
        short_class().train()

    record = json.loads((tmp_path / "short/measurement/rank-00.json").read_text())
    assert record["training_status"] == "incomplete"
    assert record["measurement_status"] == "incomplete"
    assert record["measurement"] is None
    assert short_sync_calls == [1]


def test_real_deep_gemm_setup_and_share_stay_in_task_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torchtitan.ops.warmup.deep_gemm_fast_warmup as warmup
    import torchtitan.trainer as trainer_module

    rank_root = tmp_path / "rank-cache"
    rank_root.mkdir()
    shared_root = tmp_path / "shared-cache"
    monkeypatch.setenv("DG_JIT_SHARED_CACHE_DIR", str(shared_root))
    monkeypatch.delenv("PROPAGATE_DG_CACHE", raising=False)
    monkeypatch.delenv("PROPAGATE_DG_CACHE_DIR", raising=False)

    calls: list[tuple[object, int, int, int]] = []

    def fake_warmup(model_config, ep_degree: int, num_sms: int, local_rank: int) -> None:
        calls.append((model_config, ep_degree, num_sms, local_rank))
        cache = Path(warmup._rank_cache_dir(local_rank)) / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        sentinel = cache / f"kernel.rank{local_rank}"
        sentinel.mkdir()
        (sentinel / "owner.txt").write_text(str(local_rank), encoding="utf-8")

    def forbidden_cache_manager(local_rank: int) -> None:
        raise AssertionError(f"hard-coded /temp manager was called for local rank {local_rank}")

    monkeypatch.setattr(warmup, "_warmup_node", fake_warmup)
    monkeypatch.setattr(warmup, "_deepgemm_cache_manager", forbidden_cache_manager)
    original_rank_cache_dir = warmup._rank_cache_dir
    original_setup = trainer_module.setup_deep_gemm_cache
    redirect = measure.DeepGemmCacheRedirect(
        warmup, trainer_module, rank_root, task_root=tmp_path
    )
    model_config = object()
    fp8 = SimpleNamespace(
        deep_gemm_warmup_on_start=True,
        sparse_deep_gemm_num_sms=116,
    )

    with redirect:
        for local_rank in range(8):
            trainer_module.setup_deep_gemm_cache(local_rank, model_config, 8, fp8)
        for local_rank in range(8):
            warmup._share_node_kernels(local_rank, 8)

        for local_rank in range(8):
            rank_dir = rank_root / f"rank_{local_rank}" / "deepgemm" / "cache"
            assert rank_dir.is_relative_to(rank_root)
            for owner in range(8):
                marker = rank_dir / f"kernel.rank{owner}" / "owner.txt"
                assert marker.read_text(encoding="utf-8") == str(owner)

    assert calls == [(model_config, 8, 116, rank) for rank in range(8)]
    assert warmup._rank_cache_dir is original_rank_cache_dir
    assert trainer_module.setup_deep_gemm_cache is original_setup
    assert all(
        path.resolve().is_relative_to(tmp_path.resolve())
        for path in rank_root.rglob("owner.txt")
    )


def test_deep_gemm_cache_adapter_rejects_alternate_writers_before_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torchtitan.ops.warmup.deep_gemm_fast_warmup as warmup
    import torchtitan.trainer as trainer_module

    rank_root = tmp_path / "rank-cache"
    rank_root.mkdir()
    shared_root = tmp_path / "shared-cache"
    monkeypatch.setenv("DG_JIT_SHARED_CACHE_DIR", str(shared_root))
    monkeypatch.delenv("PROPAGATE_DG_CACHE", raising=False)
    monkeypatch.delenv("PROPAGATE_DG_CACHE_DIR", raising=False)
    redirect = measure.DeepGemmCacheRedirect(
        warmup, trainer_module, rank_root, task_root=tmp_path
    )

    with redirect:
        with pytest.raises(RuntimeError, match="requires the fixed DeepGEMM warmup"):
            trainer_module.setup_deep_gemm_cache(
                0,
                object(),
                8,
                SimpleNamespace(deep_gemm_warmup_on_start=False),
            )
        assert not shared_root.exists()
        assert not any(rank_root.iterdir())

        monkeypatch.setenv("PROPAGATE_DG_CACHE", "1")
        with pytest.raises(RuntimeError, match="PROPAGATE_DG_CACHE paths"):
            trainer_module.setup_deep_gemm_cache(
                0,
                object(),
                8,
                SimpleNamespace(deep_gemm_warmup_on_start=True),
            )
        assert not shared_root.exists()
        assert not any(rank_root.iterdir())


def _run_short(trainer, stop_after: int) -> None:
    for step in range(1, stop_after + 1):
        trainer.step = step
        trainer.metrics_processor.ntokens_since_last_log += LOCAL_PHYSICAL_PER_UPDATE
        trainer.ntokens_seen.value += GLOBAL_SUPERVISED_PER_UPDATE
        trainer.metrics_processor.interval_metrics()
        trainer.runtime.stats = {
            "allocated_bytes.all.peak": 1_000 + step,
            "active_bytes.all.peak": 2_000 + step,
            "reserved_bytes.all.peak": 3_000 + step,
            "num_alloc_retries": max(0, step - 20),
            "num_ooms": int(step >= 24),
        }
        trainer.metrics_module.reset_peak_stats()
        trainer.profiler.step()
