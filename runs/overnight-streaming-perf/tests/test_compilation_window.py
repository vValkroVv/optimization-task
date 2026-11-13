from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compiler_stats
import measure
import parse_results


def _metadata() -> dict:
    files = {
        name: {"expected_sha256": digest, "sha256": digest}
        for name, digest in compiler_stats.SOURCE_PINS.items()
    }
    files["torch._inductor.async_compile"] = {"sha256": "a" * 64}
    return {"source_files": files}


def _snapshot() -> dict:
    return {
        "schema_version": compiler_stats.SCHEMA_VERSION,
        "sources": {
            "dynamo": {
                "available": True,
                "mapping_id": 10,
                "timing_lists": {
                    "_compile.compile_inner": {"identity": 11, "length": 2},
                    "runtime.metrics": {"identity": 12, "length": 4},
                },
            },
            "inductor": {
                "available": True,
                "counters_id": 20,
                "inductor_mapping_id": 21,
                "counters": {
                    "async_compile_cache_miss": {"present": True, "value": 3},
                    "async_compile_cache_hit": {"present": True, "value": 7},
                },
            },
            "cute": {
                "available": True,
                "module_id": 30,
                "mapping_id": 31,
                "entries": {"shape-a": 32},
            },
            "triton": {
                "available": True,
                "registry_id": 40,
                "function_ids": {"module.kernel": 41},
                "device_caches_ids": {"module.kernel": 42},
                "kernel_cache_ids": {"kernel-device": 43},
                "kernel_entry_ids": {"kernel-device-specialization-a": 44},
            },
        },
        "snapshot_errors": [],
    }


def _comparison(before: dict, after: dict | None = None) -> dict:
    return compiler_stats.compare_compilation_snapshots(
        before, before if after is None else after, _metadata()
    )


def _guard_identity() -> dict:
    return {
        "build_version": compiler_stats.DEEP_GEMM_GUARD_BUILD,
        "extension_path": "/opt/torchtitan-venv/lib/python3.12/site-packages/deep_gemm/_C.so",
        "sha256": compiler_stats.DEEP_GEMM_GUARD_EXTENSION_SHA256,
    }


def _guard_receipt(activation: int, end: int, sync_calls: int) -> dict:
    return {
        "environment_variable": compiler_stats.DEEP_GEMM_GUARD_ENV,
        "guarded_value": compiler_stats.DEEP_GEMM_GUARD_VALUE,
        "activation_after_update": activation,
        "previous_value": None,
        "flag_after_activation": "0",
        "activation_sync_calls": sync_calls,
        "end_after_update": end,
        "flag_at_end_boundary": "0",
        "binary_identity": _guard_identity(),
    }


def test_stable_snapshot_allows_runtime_only_growth() -> None:
    before = _snapshot()
    after = deepcopy(before)
    after["sources"]["dynamo"]["timing_lists"]["runtime.metrics"]["length"] += 5
    # An unrelated runtime counter is intentionally outside the selected compile counters.
    after["sources"]["inductor"]["counters"]["runtime_kernel_count"] = 900

    result = _comparison(before, after)

    assert result["status"] == "observed_stable"
    assert result["sources"]["dynamo"]["runtime_only_growth"] == ["runtime.metrics"]
    assert result["sources"]["dynamo"]["compile_only_growth"] == []
    assert result["source_coverage"]["status"] == "partial"
    assert result["source_coverage"]["unverified"] == [
        "asynchronous compiler worker completion",
        "DeepGEMM native JIT paths outside the tested grouped APIs",
    ]


@pytest.mark.parametrize(
    "malformation",
    [
        "missing_dynamo_identity",
        "missing_triton_identity",
        "missing_inductor_identity",
        "empty_inductor_records",
        "missing_inductor_counter_value",
        "nonboolean_inductor_presence",
        "inconsistent_absent_inductor_value",
    ],
)
def test_identical_malformed_source_states_are_incomplete(malformation: str) -> None:
    before = _snapshot()
    after = deepcopy(before)
    if malformation == "missing_dynamo_identity":
        before["sources"]["dynamo"].pop("mapping_id")
        after["sources"]["dynamo"].pop("mapping_id")
    elif malformation == "missing_triton_identity":
        before["sources"]["triton"].pop("registry_id")
        after["sources"]["triton"].pop("registry_id")
    elif malformation == "missing_inductor_identity":
        before["sources"]["inductor"].pop("inductor_mapping_id")
        after["sources"]["inductor"].pop("inductor_mapping_id")
    elif malformation == "empty_inductor_records":
        for state in (before, after):
            state["sources"]["inductor"]["counters"] = {
                "async_compile_cache_miss": {},
                "async_compile_cache_hit": {},
            }
    elif malformation == "nonboolean_inductor_presence":
        for state in (before, after):
            state["sources"]["inductor"]["counters"]["async_compile_cache_hit"][
                "present"
            ] = "false"
    elif malformation == "missing_inductor_counter_value":
        for state in (before, after):
            state["sources"]["inductor"]["counters"]["async_compile_cache_hit"].pop(
                "value"
            )
    else:
        for state in (before, after):
            state["sources"]["inductor"]["counters"]["async_compile_cache_hit"] = {
                "present": False,
                "value": 0,
            }

    result = _comparison(before, after)

    assert result["status"] == "incomplete"
    assert not result["observed_stable"]
    assert result["errors"]


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda state: state["sources"]["dynamo"]["timing_lists"].update(
                {"compile_attempt_3": {"identity": 100, "length": 1}}
            ),
            "changes_observed",
        ),
        (
            lambda state: state["sources"]["dynamo"]["timing_lists"]["_compile.compile_inner"].update(
                {"length": 3}
            ),
            "changes_observed",
        ),
        (
            lambda state: state["sources"]["inductor"]["counters"]["async_compile_cache_miss"].update(
                {"value": 4}
            ),
            "changes_observed",
        ),
        (
            lambda state: state["sources"]["cute"]["entries"].update({"shape-b": 45}),
            "changes_observed",
        ),
        (
            lambda state: state["sources"]["triton"]["kernel_entry_ids"].update(
                {"kernel-device-specialization-b": 46}
            ),
            "changes_observed",
        ),
        (
            lambda state: state["sources"]["cute"].update({"mapping_id": 999}),
            "incomplete",
        ),
        (
            lambda state: state["sources"]["dynamo"]["timing_lists"]["runtime.metrics"].update(
                {"identity": 999}
            ),
            "incomplete",
        ),
        (
            lambda state: state["sources"]["dynamo"]["timing_lists"]["_compile.compile_inner"].update(
                {"length": 1}
            ),
            "incomplete",
        ),
        (
            lambda state: state["sources"]["inductor"]["counters"]["async_compile_cache_hit"].update(
                {"value": 1}
            ),
            "incomplete",
        ),
        (
            lambda state: state["sources"]["triton"]["kernel_entry_ids"].pop(
                "kernel-device-specialization-a"
            ),
            "incomplete",
        ),
        (
            lambda state: state["sources"].pop("cute"),
            "incomplete",
        ),
    ],
)
def test_compile_growth_and_state_resets_open_the_gate(mutate, expected: str) -> None:
    before = _snapshot()
    after = deepcopy(before)
    mutate(after)

    assert _comparison(before, after)["status"] == expected


def test_observer_does_not_insert_defaultdict_keys_or_bind_missing_devices(
    monkeypatch,
) -> None:
    calls: list[str] = []
    counters = defaultdict(Counter)

    def binder():
        calls.append("device-bound")
        raise AssertionError("snapshot must not invoke the Triton device binder")

    device_caches = defaultdict(binder)
    function = SimpleNamespace(device_caches=device_caches)
    dynamo = SimpleNamespace(compilation_time_metrics={}, counters=counters)
    triton = SimpleNamespace(_triton_jit_function_registry={"module.kernel": function})
    observer = object.__new__(compiler_stats.CompilationWindowObserver)
    observer._imports = {
        "torch._dynamo.utils": dynamo,
        "triton.runtime.jit": triton,
    }
    observer._import_errors = {}
    monkeypatch.setitem(
        sys.modules,
        compiler_stats.CUTE_MODULE,
        SimpleNamespace(COMPILED={}),
    )

    state = observer.snapshot()
    result = compiler_stats.compare_compilation_snapshots(state, state, _metadata())

    assert state["sources"]["inductor"]["inductor_mapping_id"] is None
    assert result["status"] == "observed_stable"
    assert result["errors"] == []
    assert "inductor" not in counters
    assert not calls
    assert not device_caches
    assert state["sources"]["triton"]["function_ids"] == {
        "module.kernel": id(function)
    }


class _Part:
    def parameters(self) -> list:
        return []


class _Metrics:
    ntokens_since_last_log = 0


class _Trainer:
    def __init__(self) -> None:
        self.step = 0
        self.model_parts = [_Part()]
        self.metrics_processor = _Metrics()
        self.ntokens_seen = 0


def test_window_snapshots_stay_outside_primary_timer_and_keep_two_syncs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    clock_values = iter((100, 200))
    wall_values = iter((1_000, 2_000))
    trainer = _Trainer()
    monkeypatch.setattr(measure, "parameter_census", lambda _: {"trainable": {}, "frozen": {}})
    measurement = measure.WindowMeasurement(
        trainer=trainer,
        run_id="observer-test",
        output_dir=tmp_path,
        expected_steps=2,
        warmup_updates=1,
        window_updates=1,
        monitor_interval=50,
        rank=0,
        world_size=8,
        cuda_synchronize=lambda: events.append("sync"),
        memory_stats=lambda: events.append("memory") or {},
        capture_microbatches=False,
        clock_ns=lambda: events.append("clock") or next(clock_values),
        wall_clock_ns=lambda: events.append("wall") or next(wall_values),
        compiler_snapshot=lambda: events.append("snapshot") or _snapshot(),
        compiler_source_metadata=_metadata(),
        compiler_guard_activate=lambda step, sync_calls: (
            events.append("guard")
            or monkeypatch.setenv(compiler_stats.DEEP_GEMM_GUARD_ENV, "0")
            or _guard_receipt(step, step + 1, sync_calls)
        ),
    )

    trainer.step = 1
    measurement.on_profiler_step(1)
    trainer.step = 2
    measurement.on_profiler_step(2)
    assert measurement.finish(training_succeeded=True) is None
    record = json.loads((tmp_path / "rank-00.json").read_text(encoding="utf-8"))

    assert events == [
        "sync", "guard", "snapshot", "wall", "clock", "memory",
        "sync", "clock", "wall", "snapshot", "memory",
    ]
    assert record["measurement"]["start_monotonic_ns"] == 100
    assert record["measurement"]["end_monotonic_ns"] == 200
    assert record["measurement"]["cuda_synchronize_calls"] == 2
    assert record["measurement"]["start_wall_time_ns"] == 1_000
    assert record["measurement"]["end_wall_time_ns"] == 2_000
    assert record["measurement"]["compilation_observer"]["before"] == _snapshot()
    assert record["measurement"]["compilation_observer"]["comparison"]["status"] == "observed_stable"
    assert record["compiler_guard"]["activation_after_update"] == 1
    assert record["compiler_guard"]["end_after_update"] == 2
    assert record["compiler_guard"]["flag_at_end_boundary"] == "0"


def test_disabled_window_does_not_call_or_serialize_observer(tmp_path: Path, monkeypatch) -> None:
    trainer = _Trainer()
    monkeypatch.setattr(measure, "parameter_census", lambda _: {"trainable": {}, "frozen": {}})
    measurement = measure.WindowMeasurement(
        trainer=trainer,
        run_id="disabled-test",
        output_dir=tmp_path,
        expected_steps=1,
        warmup_updates=0,
        window_updates=0,
        monitor_interval=5,
        rank=0,
        world_size=8,
        cuda_synchronize=lambda: pytest.fail("disabled windows do not synchronize"),
        memory_stats=lambda: pytest.fail("disabled windows do not sample memory"),
        capture_microbatches=False,
        wall_clock_ns=lambda: pytest.fail("disabled windows do not sample wall time"),
        compiler_snapshot=lambda: pytest.fail("disabled windows do not sample compilers"),
        compiler_source_metadata=_metadata(),
        compiler_guard_activate=lambda *_: pytest.fail("disabled mode does not activate a guard"),
    )
    trainer.step = 1
    measurement.on_profiler_step(1)

    assert measurement.finish(training_succeeded=True) is None
    record = json.loads((tmp_path / "rank-00.json").read_text(encoding="utf-8"))
    assert record["measurement"] is None
    assert "compilation_observer" not in record
    assert "compiler_guard" not in record


def test_profile_guard_activates_after_warmup_and_records_target_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    trainer = _Trainer()
    monkeypatch.setattr(measure, "parameter_census", lambda _: {"trainable": {}, "frozen": {}})

    def activate(step: int, sync_calls: int) -> dict:
        events.append("guard")
        monkeypatch.setenv(compiler_stats.DEEP_GEMM_GUARD_ENV, "0")
        return _guard_receipt(step, 3, sync_calls)

    measurement = measure.WindowMeasurement(
        trainer=trainer,
        run_id="profile-guard-test",
        output_dir=tmp_path,
        expected_steps=3,
        warmup_updates=1,
        window_updates=0,
        monitor_interval=50,
        rank=0,
        world_size=8,
        cuda_synchronize=lambda: events.append("sync"),
        memory_stats=lambda: pytest.fail("profile guard does not sample window memory"),
        capture_microbatches=False,
        profile_update=3,
        clock_ns=lambda: events.append("clock") or len(events),
        compiler_guard_activate=activate,
    )

    trainer.step = 1
    measurement.on_profiler_step(1)
    trainer.step = 2
    assert os.environ[compiler_stats.DEEP_GEMM_GUARD_ENV] == "0"
    measurement.on_profiler_step(2)
    trainer.step = 3
    measurement.on_profiler_step(3)
    assert measurement.finish(training_succeeded=True) is None
    record = json.loads((tmp_path / "rank-00.json").read_text(encoding="utf-8"))

    assert events[:3] == ["sync", "guard", "clock"]
    assert events.count("sync") == 1
    assert record["compiler_guard"]["activation_after_update"] == 1
    assert record["compiler_guard"]["activation_sync_calls"] == 1
    assert record["compiler_guard"]["end_after_update"] == 3
    assert record["compiler_guard"]["flag_at_end_boundary"] == "0"
    assert "measurement" not in record or record["measurement"] is None


@pytest.mark.parametrize("previous", [None, "1"])
@pytest.mark.parametrize("fail_inside", [False, True])
def test_native_adapter_restores_guard_environment_on_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    previous: str | None,
    fail_inside: bool,
) -> None:
    class Trainer:
        def train(self):
            return None

        def forward_backward_step(self):
            return None

    class Profiler:
        def step(self):
            return None

    class Metrics:
        def interval_metrics(self):
            return None

    if previous is None:
        monkeypatch.delenv(compiler_stats.DEEP_GEMM_GUARD_ENV, raising=False)
    else:
        monkeypatch.setenv(compiler_stats.DEEP_GEMM_GUARD_ENV, previous)
    adapter = measure.NativeAdapter(
        trainer_class=Trainer,
        profiler_class=Profiler,
        metrics_class=Metrics,
        metrics_module=SimpleNamespace(reset_peak_stats=lambda: None),
        options={
            "run_id": "guard-restore-test",
            "output_dir": tmp_path,
            "expected_steps": 1,
            "warmup_updates": 0,
            "window_updates": 0,
            "monitor_interval": 50,
            "capture_microbatches": False,
            "profile_update": None,
            "compiler_guard_identity": _guard_identity(),
        },
        cuda_synchronize=lambda: None,
        memory_stats=lambda: {},
    )

    if fail_inside:
        with pytest.raises(LookupError, match="sentinel"):
            with adapter:
                receipt = adapter._activate_compiler_guard(60, 0)
                assert receipt["previous_value"] == previous
                assert os.environ[compiler_stats.DEEP_GEMM_GUARD_ENV] == "0"
                raise LookupError("sentinel")
    else:
        with adapter:
            receipt = adapter._activate_compiler_guard(60, 0)
            assert receipt["previous_value"] == previous
            assert os.environ[compiler_stats.DEEP_GEMM_GUARD_ENV] == "0"

    assert os.environ.get(compiler_stats.DEEP_GEMM_GUARD_ENV) == previous


def test_mismatched_guard_binary_fails_before_snapshot_or_timer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    identity = _guard_identity()
    identity["sha256"] = "0" * 64
    adapter = object.__new__(measure.NativeAdapter)
    adapter.compiler_guard_identity = identity
    adapter._compiler_guard_previous_value = None
    adapter._compiler_guard_activated = False
    trainer = _Trainer()
    monkeypatch.setattr(measure, "parameter_census", lambda _: {"trainable": {}, "frozen": {}})
    measurement = measure.WindowMeasurement(
        trainer=trainer,
        run_id="bad-guard-binary-test",
        output_dir=tmp_path,
        expected_steps=2,
        warmup_updates=1,
        window_updates=1,
        monitor_interval=50,
        rank=0,
        world_size=8,
        cuda_synchronize=lambda: events.append("sync"),
        memory_stats=lambda: {},
        capture_microbatches=False,
        clock_ns=lambda: events.append("clock") or 1,
        compiler_snapshot=lambda: events.append("snapshot") or _snapshot(),
        compiler_guard_activate=adapter._activate_compiler_guard,
    )
    trainer.step = 1

    with pytest.raises(RuntimeError, match="extension hash"):
        measurement.on_profiler_step(1)

    assert events == ["sync"]
    assert not adapter._compiler_guard_activated
    assert compiler_stats.DEEP_GEMM_GUARD_ENV not in os.environ


def _parser_records() -> tuple[list[dict], dict]:
    before = _snapshot()
    comparison = _comparison(before)
    records = []
    for rank in range(8):
        start = 1_000_000_000 + rank * 10
        measurement = {
            "warmup_updates": 60,
            "window_updates": 50,
            "monitoring_updates": [100],
            "cuda_synchronize_calls": 2,
            "start_monotonic_ns": start,
            "end_monotonic_ns": start + 50_000_000_000,
            "local_elapsed_s": 50.0,
            "start_wall_time_ns": 2_000_000_000,
            "end_wall_time_ns": 52_000_000_000,
            "physical_input_tokens_local": 50 * parse_results.LOCAL_PHYSICAL_TOKENS_PER_UPDATE,
            "supervised_tokens_global": 50 * parse_results.GLOBAL_SUPERVISED_TOKENS_PER_UPDATE,
            "memory": {"peak_bytes": {}, "allocator_counter_deltas": {}},
            "compilation_observer": {
                "source_metadata": _metadata(),
                "before": deepcopy(before),
                "after": deepcopy(before),
                "comparison": deepcopy(comparison),
            },
        }
        records.append(
            {
                "rank": rank,
                "world_size": 8,
                "run_id": "parser-test",
                "host_name": "one-host",
                "training_status": "complete",
                "expected_steps": 110,
                "completed_steps": 110,
                "measurement_status": "complete",
                "measurement": measurement,
                "compiler_guard": _guard_receipt(60, 110, 0),
                "parameter_census": {"trainable": {}, "frozen": {}},
            }
        )
    metadata = {
        "run_id": "parser-test",
        "steps": 110,
        "warmup_updates": 60,
        "window_updates": 50,
        "monitor_interval": 50,
        "deep_gemm_compilation_guard": compiler_stats.deep_gemm_guard_manifest(
            enabled=True, activate_after_update=60
        ),
    }
    return records, metadata


def test_parser_requires_stable_rank_boundaries_but_keeps_coverage_partial() -> None:
    records, metadata = _parser_records()

    summary = parse_results.summarize_records(records, metadata)

    assert summary["status"] == "complete"
    assert summary["compilation_observer"]["status"] == "observed_stable"
    assert summary["compilation_observer"]["coverage_status"] == "partial"
    assert summary["compilation_observer"]["unverified"] == [
        "asynchronous compiler worker completion",
        "DeepGEMM native JIT paths outside the tested grouped APIs",
    ]
    assert summary["deep_gemm_compilation_guard"]["status"] == "complete"


@pytest.mark.parametrize("failure", ["missing", "changed_counter", "missing_comparison"])
def test_parser_rejects_missing_or_changed_compiler_receipts(failure: str) -> None:
    records, metadata = _parser_records()
    observer = records[2]["measurement"]["compilation_observer"]
    if failure == "missing":
        observer["after"] = None
    elif failure == "missing_comparison":
        observer.pop("comparison")
    else:
        observer["after"]["sources"]["inductor"]["counters"][
            "async_compile_cache_miss"
        ]["value"] += 1

    summary = parse_results.summarize_records(records, metadata)

    assert summary["status"] == "incomplete"
    assert any("compiler" in error.lower() for error in summary["errors"])


@pytest.mark.parametrize(
    "failure",
    ["missing", "missing_previous_value", "changed_flag", "wrong_boundary", "wrong_binary"],
)
def test_parser_rejects_missing_or_malformed_deep_gemm_guard(failure: str) -> None:
    records, metadata = _parser_records()
    if failure == "missing":
        records[2].pop("compiler_guard")
    elif failure == "missing_previous_value":
        records[2]["compiler_guard"].pop("previous_value")
    elif failure == "changed_flag":
        records[2]["compiler_guard"]["flag_at_end_boundary"] = "1"
    elif failure == "wrong_boundary":
        records[2]["compiler_guard"]["activation_after_update"] = 59
    else:
        records[2]["compiler_guard"]["binary_identity"]["sha256"] = "0" * 64

    summary = parse_results.summarize_records(records, metadata)

    assert summary["status"] == "incomplete"
    assert any("DeepGEMM guard" in error for error in summary["errors"])

