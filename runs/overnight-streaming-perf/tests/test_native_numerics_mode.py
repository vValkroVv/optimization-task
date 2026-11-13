from __future__ import annotations

import builtins
import hashlib
import json
import struct
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import measure
import parse_results
import run


MANIFEST = {
    "schema": "native-numerical-receipt-v1",
    "capture_updates": [1, 2, 50],
    "expected_microbatches": 32,
    "scope": "diagnostic only; no TPS",
}


def _tensor_observation(update, phase, name, *, rank=0, microbatch=None, dtype="torch.float32"):
    return {
        "update": update,
        "phase": phase,
        "name": name,
        "microbatch": microbatch,
        "present": True,
        "rank": rank,
        "dtype": dtype,
        "source_device_type": "cuda",
        "global_shape": [],
        "local_shape": [],
        "local_stride": [],
        "local_numel": 1,
        "mesh": None,
        "placements": [],
        "sha256": "a" * 64,
        "is_finite": True if dtype.startswith("torch.float") else None,
    }


def _counter_observation(rank, update, value, phase):
    digest = hashlib.sha256(struct.pack("=q", value)).hexdigest()
    item = _tensor_observation(update, phase, "trainer.ntokens_seen", rank=rank, dtype="torch.int64")
    item["sha256"] = digest
    item["is_finite"] = None
    return item


def _receipt(rank=0):
    observations = [
        {"update": 0, "phase": "after_load", "name": "trainer.step", "value": 0},
        _counter_observation(rank, 0, 0, "after_load"),
        _tensor_observation(0, "initial_parameter", "model.0.weight", rank=rank),
    ]
    for update, chunks, tokens in ((1, 2, 2_096_640), (2, 2, 4_193_280), (50, 32, 104_832_000)):
        for microbatch in range(32):
            observations.extend(
                [
                    _tensor_observation(update, "dataloader_input", "input.tokens", rank=rank, microbatch=microbatch, dtype="torch.int64"),
                    _tensor_observation(update, "dataloader_input", "labels", rank=rank, microbatch=microbatch, dtype="torch.int64"),
                    {
                        "update": update,
                        "phase": "microbatch_metadata",
                        "name": "trainer.forward_backward_step",
                        "microbatch": microbatch,
                        "active_num_chunks": chunks,
                        "global_valid_tokens": {
                            "shape": [], "dtype": "python.float", "device_type": None,
                            "python_type": "float", "value": 2_096_640.0,
                        },
                    },
                    _tensor_observation(update, "microbatch_output", "loss.local_normalized", rank=rank, microbatch=microbatch),
                ]
            )
        observations.extend(
            [
                {
                    "update": update, "phase": "update_summary", "name": "global_valid_tokens",
                    "microbatch": None, "present": True, "python_type": "float", "dtype": "python.float",
                    "shape": [], "global_shape": [], "local_shape": [], "local_stride": [],
                    "local_numel": 1, "source_device_type": None, "value": 2_096_640.0,
                    "sha256": None, "is_finite": True,
                },
                _tensor_observation(update, "grad_preclip", "model.0.weight.grad", rank=rank),
                _tensor_observation(update, "clip_result", "clip.total_norm", rank=rank),
                _tensor_observation(update, "grad_postclip", "model.0.weight.grad", rank=rank),
                {"update": update, "phase": "optimizer_group", "name": "optimizer.0.group.0", "optimizer_class": "torch.optim.AdamW", "hyperparameters": {"lr": 0.001}},
                _tensor_observation(update, "post_optimizer_parameter", "model.0.weight", rank=rank),
                {"update": update, "phase": "callback_state", "name": "callbacks.batch_end", "microbatch": None, "event": "batch_end", "step": update, "record_count": 0},
                {"update": update, "phase": "batch_end", "name": "trainer.step", "value": update},
                _counter_observation(rank, update, tokens, "batch_end"),
            ]
        )
    return {
        "schema": "native-numerical-receipt-v1",
        "complete": True,
        "rank": rank,
        "capture_updates": [1, 2, 50],
        "expected_microbatches": 32,
        "hash_semantics": "exact bytes",
        "clip_norm_semantics": "native total norm",
        "ema_state": "not captured",
        "callback_state_scope": "batch end only",
        "training_metadata": {"initial_step": 0, "mtp_loss_weights": [0.3], "initial_num_chunks": 2},
        "positions_semantics": "source positions",
        "microbatch_capture": "CPU inputs and deferred scalars",
        "scope": "Parameter, gradient, optimizer-state and update-summary copies to CPU synchronize and are diagnostic instrumentation, not TPS.",
        "coverage": {
            str(update): {
                "input_batches": 32, "microbatches": 32, "summary_calls": 1,
                "clip_calls": 1, "optimizer_calls": 1, "batch_end_calls": 1,
            }
            for update in (1, 2, 50)
        },
        "observations": observations,
    }


def _write_run(tmp_path: Path, *, ranks=range(8), mutate=None):
    measurement = tmp_path / "measurement"
    measurement.mkdir(exist_ok=True)
    metadata = {
        "run_id": "native-numerics-test",
        "steps": 50,
        "warmup_updates": 0,
        "window_updates": 0,
        "monitor_interval": 50,
        "native_numerics": MANIFEST,
        "config_hashes_before": {"native_numerics_sha256": "b" * 64},
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "config-hashes-final.json").write_text(json.dumps({"native_numerics_sha256": "b" * 64}))
    for rank in ranks:
        receipt = _receipt(rank)
        if mutate is not None and rank == 0:
            mutate(receipt)
        (measurement / f"native-numerics-rank{rank}.json").write_text(json.dumps(receipt))
    return metadata


def test_native_numerics_runner_and_helper_flags_are_forwarded_and_bounded(tmp_path, monkeypatch):
    args = run._parse_args(["--run-id", "n1", "--steps", "50", "--native-numerics"])
    assert args.native_numerics
    paths = {key: tmp_path / key for key in ("measurement", "config", "native", "logs")}
    command = run._command(args, paths)
    assert "--native-numerics" in command
    ordinary = run._parse_args(["--run-id", "n2", "--steps", "50"])
    assert "--native-numerics" not in run._command(ordinary, paths)

    for extra in (
        ["--steps", "49"],
        ["--warmup", "1"],
        ["--window", "50", "--warmup", "1", "--steps", "51"],
        ["--monitor-interval", "5"],
        ["--profile-update", "2", "--warmup", "1", "--steps", "2"],
    ):
        with pytest.raises(SystemExit):
            run._parse_args(["--run-id", "n3", "--steps", "50", "--native-numerics", *extra])

    helper = [
        "--measurement-dir", str(tmp_path), "--run-id", "n1", "--steps", "50",
        "--warmup", "0", "--window", "0", "--monitor-interval", "50",
        "--native-numerics", "--", "experiment.yaml",
    ]
    assert measure._parse_args(helper)[0].native_numerics
    with pytest.raises(SystemExit):
        measure._parse_args(helper[:helper.index("--steps") + 1] + ["49"] + helper[helper.index("--steps") + 2:])


def test_config_hashes_include_numerics_module_only_when_enabled():
    default_hashes = run._config_hashes()
    enabled_hashes = run._config_hashes(native_numerics=True)
    assert "native_numerics_sha256" not in default_hashes
    assert enabled_hashes["native_numerics_sha256"] == run.sha256_file(
        run.HELPER_DIR / "numerics.py"
    )


def test_runner_failure_finally_retains_enabled_numerics_hash(tmp_path, monkeypatch):
    attempt = tmp_path / "attempt"
    paths = {"attempt": attempt}
    attempt.mkdir()
    for name in ("logs", "preflight", "config", "native", "measurement"):
        paths[name] = attempt / name
        paths[name].mkdir()
    owned_tmp = tmp_path / "owned-tmp"
    owned_tmp.mkdir()
    monkeypatch.setattr(run, "PYTHON", Path(sys.executable))
    monkeypatch.setattr(run, "verify_task_root", lambda root: None)
    monkeypatch.setattr(run, "_make_run_dirs", lambda run_id: paths)
    monkeypatch.setattr(run, "_data_root", lambda root: tmp_path / "mds")
    monkeypatch.setattr(run, "_filesystem_status", lambda run_paths, data: {})
    monkeypatch.setattr(run, "_make_local_tmpdir", lambda run_id: (owned_tmp, {}))
    monkeypatch.setattr(run, "_run_environment", lambda run_paths: {})
    monkeypatch.setattr(
        run,
        "_write_experiment_config",
        lambda path: path.write_text("test config\n", encoding="utf-8"),
    )
    source = {"head": "fixed", "tracked_and_untracked_sha256": "a" * 64}
    monkeypatch.setattr(run, "source_snapshot", lambda root: source)
    monkeypatch.setattr(
        run,
        "_runtime_preflight",
        lambda env, run_paths: (_ for _ in ()).throw(RuntimeError("sentinel preflight failure")),
    )
    monkeypatch.setattr(run, "_cleanup_local_tmpdir", lambda path: path.rmdir())
    monkeypatch.setattr(run, "_launch", lambda *args, **kwargs: pytest.fail("launch must not run"))

    original_hashes = run._config_hashes
    native_flags = []
    observed_native_hashes = []

    def record_hash_call(*, native_numerics=False):
        native_flags.append(native_numerics)
        hashes = original_hashes(native_numerics=native_numerics)
        observed_native_hashes.append(hashes.get("native_numerics_sha256"))
        return hashes

    monkeypatch.setattr(run, "_config_hashes", record_hash_call)
    assert run.main(["--run-id", "n-failure", "--steps", "50", "--native-numerics"]) == 1
    assert native_flags == [True, True]
    assert observed_native_hashes == [run.sha256_file(run.HELPER_DIR / "numerics.py")] * 2
    final_hashes = json.loads((attempt / "config-hashes-final.json").read_text())
    assert final_hashes["native_numerics_sha256"] == run.sha256_file(
        run.HELPER_DIR / "numerics.py"
    )
    assert json.loads((attempt / "failure.json").read_text())["status"] == "FAIL"


@pytest.mark.parametrize("change", ["input", "gradient", "optimizer_dtype", "denominator"])
def test_exact_comparator_detects_changed_receipt_fields(tmp_path, change):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    a = _receipt(0)
    b = _receipt(0)
    if change == "input":
        next(item for item in b["observations"] if item["phase"] == "dataloader_input")["sha256"] = "c" * 64
    elif change == "gradient":
        next(item for item in b["observations"] if item["phase"] == "grad_postclip")["sha256"] = "d" * 64
    elif change == "optimizer_dtype":
        next(item for item in b["observations"] if item["phase"] == "post_optimizer_parameter")["dtype"] = "torch.bfloat16"
    else:
        next(item for item in b["observations"] if item["phase"] == "update_summary")["value"] += 1.0
    left.write_text(json.dumps(a))
    right.write_text(json.dumps(b))
    assert parse_results.compare_native_numerical_receipts(left, left) == {"equal": True, "mismatch_paths": []}
    comparison = parse_results.compare_native_numerical_receipts(left, right)
    assert not comparison["equal"]
    assert comparison["mismatch_paths"]


def test_native_numerics_parser_accepts_valid_eight_rank_capture(tmp_path):
    _write_run(tmp_path)
    result = parse_results.summarize_run(tmp_path)
    assert result["status"] == "complete"
    assert result["mode"] == "native_numerics"
    assert result["scope"] == "diagnostic only; no TPS"
    assert "tps" not in result and "throughput" not in result


@pytest.mark.parametrize(
    "phase",
    [
        "dataloader_input",
        "microbatch_output",
        "grad_preclip",
        "grad_postclip",
        "post_optimizer_parameter",
        "clip_result",
    ],
)
def test_parser_rejects_identity_only_tensor_observations(tmp_path, phase):
    def strip_payload(receipt):
        row = next(
            item for item in receipt["observations"]
            if item["update"] == 1 and item["phase"] == phase
        )
        for key in (
            "present", "rank", "dtype", "source_device_type", "global_shape",
            "local_shape", "local_stride", "local_numel", "mesh", "placements",
            "sha256", "is_finite",
        ):
            row.pop(key, None)

    _write_run(tmp_path, mutate=strip_payload)
    result = parse_results.summarize_run(tmp_path)
    assert result["status"] == "incomplete"
    assert any("malformed or non-finite records" in error for error in result["errors"])


def test_parser_requires_floating_tensor_finite_flag(tmp_path):
    def omit_finite(receipt):
        row = next(
            item for item in receipt["observations"]
            if item["update"] == 1 and item["phase"] == "microbatch_output"
        )
        row["is_finite"] = None

    _write_run(tmp_path, mutate=omit_finite)
    result = parse_results.summarize_run(tmp_path)
    assert result["status"] == "incomplete"
    assert any("malformed or non-finite records" in error for error in result["errors"])


def test_parser_accepts_none_gradients(tmp_path):
    def none_grad(receipt):
        grad = next(
            item for item in receipt["observations"]
            if item["update"] == 1 and item["phase"] == "grad_preclip"
        )
        grad.clear()
        grad.update(
            update=1,
            phase="grad_preclip",
            name="model.0.optional.grad",
            microbatch=None,
            present=False,
            dtype=None,
            shape=None,
            sha256=None,
            is_finite=None,
        )

    _write_run(tmp_path, mutate=none_grad)
    result = parse_results.summarize_run(tmp_path)
    assert result["status"] == "complete"


def test_parser_requires_captured_trainer_steps(tmp_path):
    def remove_steps(receipt):
        receipt["observations"] = [
            item for item in receipt["observations"]
            if not (item.get("phase") == "batch_end" and item.get("name") == "trainer.step")
        ]

    _write_run(tmp_path, mutate=remove_steps)
    result = parse_results.summarize_run(tmp_path)
    assert result["status"] == "incomplete"
    assert any("captured trainer.step does not match update" in error for error in result["errors"])


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("missing_rank", "exactly eight"),
        ("malformed", "schema or completion"),
        ("count", "call counts"),
        ("chunks", "active chunk count"),
        ("duplicate", "duplicate observation"),
        ("nonfinite", "malformed or non-finite"),
        ("manifest", "manifest is missing or differs"),
    ],
)
def test_native_numerics_parser_rejects_invalid_receipts(tmp_path, case, expected):
    if case == "missing_rank":
        _write_run(tmp_path, ranks=range(7))
    elif case == "malformed":
        _write_run(tmp_path, mutate=lambda receipt: receipt.update(schema="wrong"))
    elif case == "count":
        _write_run(tmp_path, mutate=lambda receipt: receipt["coverage"]["1"].update(clip_calls=2))
    elif case == "chunks":
        def change_chunks(receipt):
            next(item for item in receipt["observations"] if item["phase"] == "microbatch_metadata")["active_num_chunks"] = 8
        _write_run(tmp_path, mutate=change_chunks)
    elif case == "duplicate":
        def duplicate(receipt):
            row = next(item for item in receipt["observations"] if item["phase"] == "after_load")
            receipt["observations"].append(row.copy())
        _write_run(tmp_path, mutate=duplicate)
    elif case == "nonfinite":
        def nonfinite(receipt):
            next(item for item in receipt["observations"] if item["phase"] == "microbatch_output")["is_finite"] = False
        _write_run(tmp_path, mutate=nonfinite)
    else:
        metadata = _write_run(tmp_path)
        metadata["native_numerics"] = {**MANIFEST, "scope": "throughput"}
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    result = parse_results.summarize_run(tmp_path)
    assert result["status"] == "incomplete"
    assert any(expected in error for error in result["errors"])


def test_native_observer_installs_before_native_main_and_restores_on_error(tmp_path, monkeypatch):
    class FakeTrainer:
        def forward_backward_step(self):
            pass

    class FakeOptimizer:
        def step(self):
            pass

    class FakeCallbacks:
        def emit(self):
            pass

    def clip(*args, **kwargs):
        pass

    dist_utils = SimpleNamespace(clip_grad_norm_=clip)
    trainer_module = SimpleNamespace(
        Trainer=FakeTrainer, OptimizersContainer=FakeOptimizer,
        CallbackManager=FakeCallbacks, dist_utils=dist_utils,
    )
    originals = (FakeTrainer.forward_backward_step, FakeOptimizer.step, FakeCallbacks.emit, clip)
    lifecycle = []

    class Adapter:
        def __enter__(self):
            lifecycle.append("adapter_enter")
            return self
        def __exit__(self, exc_type, exc, tb):
            lifecycle.append(("adapter_exit", exc_type))

    def native_main(argv):
        lifecycle.append("native_main")
        assert FakeTrainer.forward_backward_step is not originals[0]
        assert FakeOptimizer.step is not originals[1]
        assert FakeCallbacks.emit is not originals[2]
        assert dist_utils.clip_grad_norm_ is not originals[3]
        raise RuntimeError("sentinel config.build failure")

    package = ModuleType("torchtitan")
    package.__path__ = []
    train_module = ModuleType("torchtitan.train")
    train_module.main = native_main
    monkeypatch.setitem(sys.modules, "torchtitan", package)
    monkeypatch.setitem(sys.modules, "torchtitan.train", train_module)
    monkeypatch.setenv("RANK", "0")
    args = SimpleNamespace(native_numerics=True, measurement_dir=tmp_path)
    with pytest.raises(RuntimeError, match="sentinel config.build"):
        measure._run_native_main(args, [], Adapter(), trainer_module)
    assert lifecycle[0] == "adapter_enter"
    assert lifecycle[1] == "native_main"
    assert lifecycle[2] == ("adapter_exit", RuntimeError)
    assert (FakeTrainer.forward_backward_step, FakeOptimizer.step, FakeCallbacks.emit, dist_utils.clip_grad_norm_) == originals
    assert list(tmp_path.iterdir()) == []


def test_default_native_path_does_not_import_numerics_or_write_receipts(tmp_path, monkeypatch):
    calls = []
    original_import = builtins.__import__

    def track_import(name, *args, **kwargs):
        if name == "numerics":
            calls.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", track_import)
    package = ModuleType("torchtitan")
    package.__path__ = []
    train_module = ModuleType("torchtitan.train")
    train_module.main = lambda argv: None
    monkeypatch.setitem(sys.modules, "torchtitan", package)
    monkeypatch.setitem(sys.modules, "torchtitan.train", train_module)

    class Adapter:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False

    args = SimpleNamespace(native_numerics=False, measurement_dir=tmp_path)
    measure._run_native_main(args, [], Adapter(), SimpleNamespace())
    assert calls == []
    assert list(tmp_path.iterdir()) == []
