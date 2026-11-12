"""CPU behavior checks for the staged native phase-memory adapter."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import deque
from pathlib import Path
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import pytest


GIFT_ROOT = Path(__file__).resolve().parents[2]
ADAPTER_PATH = (
    GIFT_ROOT / "torchtitan" / "experiments" / "cuda_graph_campaign" / "phase_memory.py"
)


class FakeClock:
    def __init__(self):
        self.utc = 1_000_000
        self.monotonic = 10_000
        self.perf = 1.0
        self.perf_calls = 0

    def time_ns(self):
        self.utc += 1
        return self.utc

    def perf_counter_ns(self):
        self.monotonic += 1
        return self.monotonic

    def perf_counter(self):
        self.perf_calls += 1
        self.perf += 0.25
        return self.perf


class FakeDevice:
    def __init__(self, stats):
        self.stats = deque(stats)
        self.memory_stats_calls = 0
        self.mem_get_info_calls = 0
        self.reset_calls = 0
        self.events = []
        self.reset_error = None
        self.helper_error = None

    def memory_stats(self):
        self.memory_stats_calls += 1
        self.events.append("memory_stats")
        return self.stats.popleft()

    def mem_get_info(self):
        self.mem_get_info_calls += 1
        self.events.append("mem_get_info")
        return 100, 1_000

    def reset_peak_memory_stats(self):
        self.reset_calls += 1
        self.events.append("device_reset")
        if self.reset_error is not None:
            raise self.reset_error
        return None


def stats(*, allocated_peak, reserved_peak, active_peak, allocated_current=10):
    return {
        "allocated_bytes.all.current": allocated_current,
        "allocated_bytes.all.peak": allocated_peak,
        "reserved_bytes.all.current": min(reserved_peak, 1_000),
        "reserved_bytes.all.peak": reserved_peak,
        "active_bytes.all.current": 5,
        "active_bytes.all.peak": active_peak,
        "num_alloc_retries": 0,
        "num_ooms": 0,
    }


def fake_modules(stats_values):
    utils = ModuleType("torchtitan.tools.utils")
    metrics = ModuleType("torchtitan.components.metrics")
    timing = ModuleType("torchtitan.experiments.cuda_graph_campaign.observer")
    device = FakeDevice(stats_values)
    device_module = SimpleNamespace(
        memory_stats=device.memory_stats,
        mem_get_info=device.mem_get_info,
        reset_peak_memory_stats=device.reset_peak_memory_stats,
    )
    utils.device_module = device_module
    utils._capacity_bytes = 1_000
    helper_calls = []
    helper_results = deque(object() for _ in stats_values)

    def device_memory_usage(memory_stats):
        device.events.append("device_memory_usage")
        helper_calls.append(memory_stats)
        if device.helper_error is not None:
            raise device.helper_error
        device.mem_get_info()
        return helper_results.popleft()

    def reset_peak_stats():
        device.events.append("reset_peak_stats")
        return device.reset_peak_memory_stats()

    utils.device_memory_usage = device_memory_usage
    utils.reset_peak_stats = reset_peak_stats
    metrics.reset_peak_stats = reset_peak_stats

    class MetricsProcessor:
        def __init__(self, log_freq=10):
            self.config = SimpleNamespace(log_freq=log_freq)
            self.log_calls = []
            self.validation_calls = []
            self.skip_usage = False
            self.skip_reset = False
            self.nest_validation = False
            self.log_error = None
            metrics.reset_peak_stats()

        def log(
            self,
            step,
            global_avg_loss,
            global_max_loss,
            grad_norm,
            extra_metrics=None,
        ):
            self.log_calls.append(step)
            if not self.skip_usage:
                memory_stats = device.memory_stats()
                result = utils.device_memory_usage(memory_stats)
            else:
                result = None
            if self.nest_validation:
                self.log_validation(0.5, step)
            device.events.append(("logger", step))
            if not self.skip_reset:
                metrics.reset_peak_stats()
            if self.log_error is not None:
                raise self.log_error
            return result

        def log_validation(self, loss, step, extra_metrics=None):
            self.validation_calls.append(step)
            memory_stats = device.memory_stats()
            result = utils.device_memory_usage(memory_stats)
            device.events.append(("validation_logger", step))
            metrics.reset_peak_stats()
            return result

    metrics.MetricsProcessor = MetricsProcessor

    clock = FakeClock()
    timing.time = clock

    class CampaignTimingObserver:
        def __init__(self, start_step=5, update_count=2):
            self.config = SimpleNamespace(
                start_step=start_step, update_count=update_count
            )
            self._memory_start = None
            self._memory_end = None
            self._last_boundary_step = None
            self.boundary_calls = []
            self.sync_calls = 0
            self.event_calls = 0

        def _capture_boundary(self, step):
            self._last_boundary_step = step
            self.boundary_calls.append(step)
            self.event_calls += 1
            if step == self.config.start_step:
                self._memory_start = {
                    "current_allocated": 11,
                    "current_reserved": 22,
                    "peak_allocated": 33,
                    "peak_reserved": 44,
                }
            return timing.time.perf_counter()

        def _finish_window(self, state, expected_end_step):
            del state, expected_end_step
            self.sync_calls += 1
            stop = timing.time.perf_counter()
            self._memory_end = {
                "current_allocated": 7,
                "current_reserved": 14,
                "peak_allocated": 21,
                "peak_reserved": 28,
            }
            return stop

    timing.CampaignTimingObserver = CampaignTimingObserver

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(get_allocator_backend=lambda: "native")

    packages = {
        "torchtitan": ModuleType("torchtitan"),
        "torchtitan.tools": ModuleType("torchtitan.tools"),
        "torchtitan.components": ModuleType("torchtitan.components"),
        "torchtitan.experiments": ModuleType("torchtitan.experiments"),
        "torchtitan.experiments.cuda_graph_campaign": ModuleType(
            "torchtitan.experiments.cuda_graph_campaign"
        ),
    }
    modules = {
        **packages,
        "torchtitan.tools.utils": utils,
        "torchtitan.components.metrics": metrics,
        "torchtitan.experiments.cuda_graph_campaign.observer": timing,
        "torch": torch,
    }
    packages["torchtitan"].tools = packages["torchtitan.tools"]
    packages["torchtitan"].components = packages["torchtitan.components"]
    packages["torchtitan"].experiments = packages["torchtitan.experiments"]
    packages["torchtitan.tools"].utils = utils
    packages["torchtitan.components"].metrics = metrics
    packages["torchtitan.experiments"].cuda_graph_campaign = packages[
        "torchtitan.experiments.cuda_graph_campaign"
    ]
    packages["torchtitan.experiments.cuda_graph_campaign"].observer = timing
    return SimpleNamespace(
        modules=modules,
        utils=utils,
        metrics=metrics,
        timing=timing,
        device=device,
        helper_calls=helper_calls,
        processor_type=MetricsProcessor,
        timing_type=CampaignTimingObserver,
        clock=clock,
    )


def load_adapter():
    name = f"staged_phase_memory_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, ADAPTER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, name


@pytest.fixture
def harness(monkeypatch, tmp_path):
    native = fake_modules(
        [
            stats(allocated_peak=900, reserved_peak=950, active_peak=90),
            stats(allocated_peak=600, reserved_peak=700, active_peak=50),
            stats(allocated_peak=999, reserved_peak=1_000, active_peak=999),
            stats(allocated_peak=500, reserved_peak=550, active_peak=40),
            stats(allocated_peak=450, reserved_peak=525, active_peak=35),
        ]
    )
    for name, module in native.modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    for name, value in {
        "RANK": "1",
        "LOCAL_RANK": "1",
        "WORLD_SIZE": "8",
        "CUDA_VISIBLE_DEVICES": "GPU-AAA,GPU-BBB",
        "TORCHTITAN_PHASE_MEMORY_RUN_ID": "run-test",
        "TORCHTITAN_PHASE_MEMORY_SOURCE_ID": "source-007-test",
        "TORCHTITAN_PHASE_MEMORY_CONFIG_ID": "config-sha-test",
    }.items():
        monkeypatch.setenv(name, value)
    adapter, module_name = load_adapter()
    observer = adapter.install_phase_memory_observer(
        tmp_path / "phase-memory",
        {"startup_end_step": 20, "expected_end_step": 30},
    )
    try:
        yield SimpleNamespace(adapter=adapter, native=native, observer=observer)
    finally:
        if not observer._restore_done:
            observer.restore()
        sys.modules.pop(module_name, None)


def run_log(processor, step):
    return processor.log(step, 1.0, 1.1, 2.0)


def test_native_training_validation_windows_keep_exact_raw_counters_and_calls(harness):
    native = harness.native
    processor = native.processor_type()
    results = [run_log(processor, step) for step in (1, 10)]
    validation_result = processor.log_validation(0.5, 10)
    results.append(run_log(processor, 20))
    results.append(run_log(processor, 30))

    assert len({id(value) for value in results}) == len(results)
    assert id(validation_result) not in {id(value) for value in results}
    assert native.device.memory_stats_calls == 5
    assert len(native.helper_calls) == 5
    assert native.device.mem_get_info_calls == 5
    assert native.device.reset_calls == 6  # constructor plus one per native log
    assert native.device.events.count("reset_peak_stats") == 6
    assert native.device.events == [
        "reset_peak_stats",
        "device_reset",
        "memory_stats",
        "device_memory_usage",
        "mem_get_info",
        ("logger", 1),
        "reset_peak_stats",
        "device_reset",
        "memory_stats",
        "device_memory_usage",
        "mem_get_info",
        ("logger", 10),
        "reset_peak_stats",
        "device_reset",
        "memory_stats",
        "device_memory_usage",
        "mem_get_info",
        ("validation_logger", 10),
        "reset_peak_stats",
        "device_reset",
        "memory_stats",
        "device_memory_usage",
        "mem_get_info",
        ("logger", 20),
        "reset_peak_stats",
        "device_reset",
        "memory_stats",
        "device_memory_usage",
        "mem_get_info",
        ("logger", 30),
        "reset_peak_stats",
        "device_reset",
    ]

    records = harness.observer._records
    snapshots = [
        record for record in records if record["event_kind"] == "native_snapshot"
    ]
    training = [record for record in snapshots if record["context"] == "training"]
    validation = [record for record in snapshots if record["context"] == "validation"]
    assert [record["step"] for record in training] == [1, 10, 20, 30]
    assert [record["phase_label"] for record in training] == [
        "initial_through_20",
        "initial_through_20",
        "initial_through_20",
        "steady_21_to_30",
    ]
    assert validation[0]["phase_label"] == "validation"
    assert validation[0]["allocator_counters_bytes"]["allocated_bytes.all.peak"] == 999
    assert all(record["status"] == "complete" for record in snapshots)
    assert training[0]["allocator_counters_bytes"]["allocated_bytes.all.peak"] == 900
    assert training[0]["allocator_counters_bytes"]["active_bytes.all.peak"] == 90
    assert training[3]["actual_window"]["label"] == "21_to_30"
    assert (
        training[2]["actual_window"]["reset_origin"] == "validation_reset_after_step_10"
    )

    harness.observer.finish_and_flush()
    output = json.loads(harness.observer.output_path.read_text())
    envelopes = {item["phase_label"]: item for item in output["phase_envelopes"]}
    assert output["status"] == "complete"
    assert envelopes["initial_through_20"]["max_allocated_peak_bytes"] == 900
    assert envelopes["initial_through_20"]["max_reserved_peak_bytes"] == 950
    assert envelopes["steady_21_to_30"]["max_allocated_peak_bytes"] == 450
    assert output["device_samples"]["expected_sample_period_ms"] == 500
    assert output["device_samples"]["sample_count"] is None
    assert output["device_samples"]["status"] == "pending_external_join"
    assert output["device_capacity_bytes"] == 1_000
    assert output["device"]["device_uuid"] == "GPU-BBB"
    assert output["allocator_backend"] == "native"
    harness.observer.restore()
    assert native.metrics.MetricsProcessor.log is harness.observer._bindings[2][2]
    assert native.utils.device_memory_usage is harness.observer._bindings[0][2]


def test_missing_required_peak_stays_invalid_in_snapshot_and_envelope(harness):
    native = harness.native
    native.device.stats[0].pop("allocated_bytes.all.peak")
    processor = native.processor_type()
    run_log(processor, 1)

    snapshot = next(
        record
        for record in harness.observer._records
        if record["event_kind"] == "native_snapshot"
    )
    assert snapshot["status"] == "invalid"
    assert any("allocated_bytes.all.peak" in reason for reason in snapshot["reasons"])
    assert native.device.memory_stats_calls == 1
    assert native.device.mem_get_info_calls == 1
    assert native.device.reset_calls == 2  # constructor and log reset

    harness.observer.finish_and_flush()
    output = json.loads(harness.observer.output_path.read_text())
    envelope = output["phase_envelopes"][0]
    assert output["status"] == "invalid"
    assert envelope["status"] == "invalid"
    assert envelope["partial_values_only"] is True
    assert envelope["max_allocated_peak_bytes"] is None
    assert envelope["max_reserved_peak_bytes"] == 950
    harness.observer.restore()


def test_unknown_reset_origin_stays_invalid_in_snapshot_and_envelope(harness):
    native = harness.native
    processor = native.processor_type()
    native.metrics.reset_peak_stats()
    run_log(processor, 1)

    snapshot = next(
        record
        for record in harness.observer._records
        if record["event_kind"] == "native_snapshot"
    )
    assert snapshot["status"] == "invalid"
    assert any("unknown reset origin" in reason for reason in snapshot["reasons"])
    assert native.device.memory_stats_calls == 1
    assert native.device.mem_get_info_calls == 1
    assert native.device.reset_calls == 3  # constructor, unknown reset, and log reset

    harness.observer.finish_and_flush()
    output = json.loads(harness.observer.output_path.read_text())
    envelope = output["phase_envelopes"][0]
    assert output["status"] == "invalid"
    assert envelope["status"] == "invalid"
    assert envelope["partial_values_only"] is True
    assert envelope["max_allocated_peak_bytes"] == 900
    assert envelope["max_reserved_peak_bytes"] == 950
    harness.observer.restore()


def test_timing_anchors_reuse_existing_timer_and_memory_samples(harness):
    native = harness.native
    timing = native.timing_type(start_step=5, update_count=2)
    original_clock = native.timing.time
    timing._capture_boundary(5)
    timing._capture_boundary(6)
    timing._capture_boundary(7)
    timing._finish_window(SimpleNamespace(device="cpu"), 7)

    assert timing.boundary_calls == [5, 6, 7]
    assert timing.event_calls == 3
    assert timing.sync_calls == 1
    assert native.clock.perf_calls == 4
    assert native.timing.time is original_clock
    assert harness.observer._timing_anchors["window_open"]["step"] == 5
    assert harness.observer._timing_anchors["completed_close"]["step"] == 7
    assert (
        harness.observer._timing_anchors["completed_close"]["completed_close"] is True
    )
    assert (
        harness.observer._timing_memory_samples["opening"]["counters_bytes"][
            "peak_allocated"
        ]
        == 33
    )
    assert (
        harness.observer._timing_memory_samples["completed_close"]["counters_bytes"][
            "peak_allocated"
        ]
        == 21
    )
    assert (
        harness.observer._timing_memory_samples["completed_close"]["reset_generation"]
        == -1
    )


def test_native_exception_is_preserved_and_partial_capture_is_invalid(harness):
    native = harness.native
    processor = native.processor_type()
    failure = RuntimeError("native logger failed")
    processor.log_error = failure
    with pytest.raises(RuntimeError) as raised:
        run_log(processor, 1)

    assert raised.value is failure
    assert native.device.memory_stats_calls == 1
    assert native.device.mem_get_info_calls == 1
    assert (
        native.device.reset_calls == 2
    )  # native log resets before its final logger line
    snapshot = next(
        record
        for record in harness.observer._records
        if record["event_kind"] == "native_snapshot"
    )
    assert snapshot["status"] == "invalid"
    assert any(
        "native training metrics call raised" in reason
        for reason in snapshot["reasons"]
    )

    original_log = harness.observer._bindings[2][2]
    original_usage = harness.observer._bindings[0][2]
    harness.observer.finish_and_flush(failure)
    harness.observer.restore()
    harness.observer.restore()
    assert native.metrics.MetricsProcessor.log is original_log
    assert native.utils.device_memory_usage is original_usage


def test_helper_exception_keeps_the_original_exception(harness):
    native = harness.native
    processor = native.processor_type()
    helper_failure = ValueError("helper failed")
    native.device.helper_error = helper_failure
    with pytest.raises(ValueError) as raised:
        run_log(processor, 1)
    assert raised.value is helper_failure
    assert native.device.memory_stats_calls == 1
    assert native.device.mem_get_info_calls == 0
    assert native.device.reset_calls == 1  # constructor only
    snapshot = next(
        record
        for record in harness.observer._records
        if record["event_kind"] == "native_snapshot"
    )
    assert snapshot["status"] == "invalid"
    harness.observer.finish_and_flush(helper_failure)
    original_usage = harness.observer._bindings[0][2]
    harness.observer.restore()
    assert native.utils.device_memory_usage is original_usage


def test_reset_exception_keeps_the_original_exception(harness):
    native = harness.native
    processor = native.processor_type()
    reset_failure = RuntimeError("native peak reset failed")
    native.device.reset_error = reset_failure
    with pytest.raises(RuntimeError) as raised:
        run_log(processor, 1)
    assert raised.value is reset_failure
    assert native.device.memory_stats_calls == 1
    assert native.device.mem_get_info_calls == 1
    assert native.device.reset_calls == 2  # constructor and failed log reset
    snapshot = next(
        record
        for record in harness.observer._records
        if record["event_kind"] == "native_snapshot"
    )
    assert snapshot["status"] == "invalid"
    assert any(
        "native reset raised RuntimeError" in reason for reason in snapshot["reasons"]
    )
    harness.observer.finish_and_flush(reset_failure)
    harness.observer.restore()


def test_short_normal_run_flushes_as_incomplete(harness):
    native = harness.native
    processor = native.processor_type()
    run_log(processor, 1)
    harness.observer.finish_and_flush()
    output = json.loads(harness.observer.output_path.read_text())
    assert output["status"] == "incomplete"
    assert output["coverage"]["last_training_metrics_step"] == 1
    assert any(
        "expected training metrics through step 30" in reason
        for reason in output["coverage"]["incomplete_reasons"]
    )
    harness.observer.restore()


def test_nested_context_missing_capture_unknown_reset_and_restore_conflict(
    tmp_path, monkeypatch
):
    native = fake_modules(
        [
            stats(allocated_peak=100, reserved_peak=120, active_peak=80),
            stats(allocated_peak=90, reserved_peak=110, active_peak=70),
        ]
    )
    for name, module in native.modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    for name, value in {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "TORCHTITAN_PHASE_MEMORY_RUN_ID": "run-nested",
        "TORCHTITAN_PHASE_MEMORY_SOURCE_ID": "source-007-test",
        "TORCHTITAN_PHASE_MEMORY_CONFIG_ID": "config-sha-test",
    }.items():
        monkeypatch.setenv(name, value)
    adapter, name = load_adapter()
    observer = adapter.install_phase_memory_observer(
        tmp_path / "nested", {"startup_end_step": 1, "expected_end_step": 10}
    )
    try:
        processor = native.processor_type()
        native.metrics.reset_peak_stats()  # unpaired reset invalidates its new origin
        assert any(
            "outside a recognized native metrics context" in reason
            for reason in observer._invalid_reasons
        )
        processor.nest_validation = True
        run_log(processor, 1)
        assert any(
            "nested native metrics contexts" in reason
            for reason in observer._invalid_reasons
        )
        assert native.device.memory_stats_calls == 2
        assert native.device.mem_get_info_calls == 2

        processor.nest_validation = False
        processor.skip_usage = True
        run_log(processor, 10)
        assert any(
            record["event_kind"] == "missing_capture" for record in observer._records
        )
        observer.finish_and_flush()
        output = json.loads(observer.output_path.read_text())
        assert output["status"] == "invalid"
        assert any(
            record["event_kind"] == "missing_capture" for record in output["events"]
        )

        original_usage = observer._bindings[0][2]

        def replacement(memory_stats):
            return None

        native.utils.device_memory_usage = replacement
        with pytest.raises(RuntimeError, match="refused to restore replaced bindings"):
            observer.restore()
        assert native.utils.device_memory_usage is replacement
        native.utils.device_memory_usage = observer._bindings[0][3]
        observer.restore()
        assert native.utils.device_memory_usage is original_usage
    finally:
        sys.modules.pop(name, None)


def test_duplicate_install_and_existing_rank_output_fail_before_patching(
    harness, tmp_path
):
    with pytest.raises(RuntimeError, match="only one phase-memory observer"):
        harness.adapter.install_phase_memory_observer(
            tmp_path / "duplicate", {"startup_end_step": 1, "expected_end_step": 2}
        )

    harness.observer.restore()
    existing = tmp_path / "occupied"
    existing.mkdir()
    (existing / "rank-00001.json").write_text("old", encoding="utf-8")
    original = harness.native.utils.device_memory_usage
    with pytest.raises(FileExistsError, match="already contains rank records"):
        harness.adapter.install_phase_memory_observer(
            existing, {"startup_end_step": 1, "expected_end_step": 2}
        )
    assert harness.native.utils.device_memory_usage is original


def test_opt_in_entry_restores_bindings_when_partial_flush_fails(
    harness, monkeypatch, tmp_path
):
    native = harness.native
    harness.observer.restore()
    package = sys.modules["torchtitan.experiments.cuda_graph_campaign"]
    logging_module = ModuleType("torchtitan.tools.logging")
    logging_module.init_logger = lambda: None
    scan_module = ModuleType(
        "torchtitan.experiments.cuda_graph_campaign.backend_scan_fix"
    )
    scan_calls = []
    scan_module.install_hybrid_ep_scan_fix = lambda: scan_calls.append("installed")
    monkeypatch.setitem(sys.modules, "torchtitan.tools.logging", logging_module)
    monkeypatch.setitem(
        sys.modules,
        "torchtitan.experiments.cuda_graph_campaign.backend_scan_fix",
        scan_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "torchtitan.experiments.cuda_graph_campaign.phase_memory",
        harness.adapter,
    )
    monkeypatch.setattr(
        sys.modules["torchtitan.tools"], "logging", logging_module, raising=False
    )
    monkeypatch.setattr(package, "backend_scan_fix", scan_module, raising=False)
    monkeypatch.setattr(package, "phase_memory", harness.adapter, raising=False)

    output_dir = tmp_path / "entry-output"
    monkeypatch.setenv("TORCHTITAN_PHASE_MEMORY_OUTPUT_DIR", str(output_dir))
    forwarded_argv = ["preset.yaml", "--training.steps", "2"]
    failure = KeyboardInterrupt("native execution stopped")
    train_module = ModuleType("torchtitan.train")

    def native_main(argv):
        assert argv is forwarded_argv
        rank_file = output_dir / "rank-00001.json"
        rank_file.parent.mkdir(parents=True, exist_ok=True)
        rank_file.write_text("existing record", encoding="utf-8")
        raise failure

    train_module.main = native_main
    monkeypatch.setitem(sys.modules, "torchtitan.train", train_module)
    entry_name = f"staged_train_phase_memory_{uuid4().hex}"
    entry_spec = importlib.util.spec_from_file_location(
        entry_name,
        GIFT_ROOT
        / "torchtitan"
        / "experiments"
        / "cuda_graph_campaign"
        / "train_phase_memory.py",
    )
    assert entry_spec is not None and entry_spec.loader is not None
    entry = importlib.util.module_from_spec(entry_spec)
    sys.modules[entry_name] = entry
    entry_spec.loader.exec_module(entry)
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            entry.main(forwarded_argv)
        assert raised.value is failure
        assert any(
            "phase-memory flush also failed" in note for note in failure.__notes__
        )
        assert scan_calls == ["installed"]
        assert (output_dir / "rank-00001.json").read_text() == "existing record"
        assert native.metrics.MetricsProcessor.log is harness.observer._bindings[2][2]
        assert native.utils.device_memory_usage is harness.observer._bindings[0][2]
        assert native.metrics.reset_peak_stats is harness.observer._bindings[1][2]
    finally:
        sys.modules.pop(entry_name, None)
