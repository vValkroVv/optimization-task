"""Focused CPU checks for opt-in reuse of completed task compiler caches."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import measure
import run as runner


CANONICAL_HASH = "canonical-config-hash"
PRODUCER_RUNTIME = {
    "python": "3.12.3",
    "torch": "torch-build",
    "torch_cuda": "13.0",
    "nccl": [2, 27, 3],
    "gpus": [{"name": "NVIDIA H100", "capability": [9, 0]}],
    "packages": {
        "deep_gemm": {
            "build_version": "deep-gemm-build",
            "header_sha256": "header-hash",
        },
        "deep_ep": {
            "hybrid_ep_wrapper_sha256": "wrapper-hash",
            "hybrid_ep_backend_sha256": "backend-hash",
        },
    },
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _task_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    task_root = tmp_path / "task"
    raw_root = task_root / "runs/overnight-streaming-perf/artifacts/raw"
    raw_root.mkdir(parents=True)
    monkeypatch.setattr(runner, "TASK_ROOT", task_root)
    monkeypatch.setattr(runner, "RAW_ROOT", raw_root)
    monkeypatch.setattr(
        runner, "SHARED_HF_CACHE", raw_root / "shared-cache/huggingface"
    )
    monkeypatch.setattr(measure, "TASK_ROOT", task_root)
    monkeypatch.setattr(measure, "RAW_ROOT", raw_root)
    original_git = runner._git

    def task_git(root, *args, check=True):
        if args and args[0] == "check-ignore":
            return SimpleNamespace(returncode=0)
        return original_git(root, *args, check=check)

    monkeypatch.setattr(runner, "_git", task_git)
    monkeypatch.setattr(runner, "_group_exists", lambda _pgid: False)
    monkeypatch.delenv("TORCHINDUCTOR_FORCE_DISABLE_CACHES", raising=False)
    return task_root, raw_root


def _make_producer(raw_root: Path, run_id: str = "baseline003") -> Path:
    attempt = raw_root / run_id
    attempt.mkdir()
    cache_root = attempt / "cache"
    for name in ("torchinductor", "triton", "deep_gemm", "hybridep"):
        (cache_root / name).mkdir(parents=True)
    for rank in range(8):
        (cache_root / "triton" / f"rank_{rank}").mkdir()
    (cache_root / "deep_gemm" / "rank-cache").mkdir()
    (attempt / "status.txt").write_text("PASS\n", encoding="utf-8")
    _write_json(attempt / "completion.json", {"status": "PASS"})
    _write_json(
        attempt / "process.json",
        {
            "process_group_id": 987654321,
            "process_group_stopped": True,
            "orphaned_process_group": False,
        },
    )
    _write_json(
        attempt / "metadata.json",
        {
            "run_id": run_id,
            "runtime": PRODUCER_RUNTIME,
            "config_hashes_before": {
                "canonical_experiment_sha256": CANONICAL_HASH
            },
        },
    )
    return cache_root


def _run_dirs(
    raw_root: Path, run_id: str, cache_root: Path | None = None
) -> dict[str, Path]:
    return runner._make_run_dirs(
        run_id, raw_root, compiler_cache_root=cache_root
    )


def test_default_creates_fresh_cache_and_exports_its_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, raw_root = _task_roots(tmp_path, monkeypatch)
    args = runner._parse_args(["--run-id", "fresh-default"])
    assert args.compiler_cache_root is None
    paths = _run_dirs(raw_root, "fresh-default")
    task_tmpdir, _ = runner._make_task_tmpdir()
    paths["tmp_dir"] = task_tmpdir
    try:
        env = runner._run_environment(paths)
        assert paths["cache"] == paths["attempt"] / "cache"
        assert env["TASK_COMPILER_CACHE_ROOT"] == str(paths["cache"].resolve())
        assert env["TRITON_CACHE_DIR"] == str(paths["cache_triton"].resolve())
        assert all((paths["cache_triton"] / f"rank_{rank}").is_dir() for rank in range(8))
    finally:
        runner._cleanup_task_tmpdir(task_tmpdir)


def test_two_attempts_reuse_one_completed_root_with_rank_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_root, raw_root = _task_roots(tmp_path, monkeypatch)
    producer_cache = _make_producer(raw_root)
    first = _run_dirs(raw_root, "candidate-a", producer_cache)
    second = _run_dirs(raw_root, "candidate-b", producer_cache)
    assert first["attempt"] != second["attempt"]
    assert first["measurement"] != second["measurement"]
    assert first["cache"] == second["cache"] == producer_cache

    task_tmp_a, _ = runner._make_task_tmpdir()
    task_tmp_b, _ = runner._make_task_tmpdir()
    first["tmp_dir"] = task_tmp_a
    second["tmp_dir"] = task_tmp_b
    try:
        env = runner._run_environment(second)
        expected = {
            "TORCHINDUCTOR_CACHE_DIR": producer_cache / "torchinductor",
            "TRITON_CACHE_DIR": producer_cache / "triton",
            "DG_JIT_SHARED_CACHE_DIR": producer_cache / "deep_gemm",
            "TASK_DEEP_GEMM_RANK_CACHE_ROOT": producer_cache / "deep_gemm/rank-cache",
            "HYBRID_EP_CACHE_DIR": producer_cache / "hybridep",
        }
        for name, path in expected.items():
            assert env[name] == str(path.resolve())
        assert env["TASK_COMPILER_CACHE_ROOT"] == str(producer_cache.resolve())
        assert first["measurement"].is_relative_to(task_root)
        assert second["measurement"].is_relative_to(task_root)
        assert task_tmp_a != task_tmp_b
        assert task_tmp_a.is_relative_to(task_root)
        assert task_tmp_b.is_relative_to(task_root)

        monkeypatch.setenv("TASK_COMPILER_CACHE_ROOT", env["TASK_COMPILER_CACHE_ROOT"])
        selected: set[Path] = set()
        for rank in range(8):
            monkeypatch.setenv("RANK", str(rank))
            monkeypatch.setenv("TRITON_CACHE_DIR", env["TRITON_CACHE_DIR"])
            routed_rank, rank_path = measure._route_rank_triton_cache(
                second["measurement"]
            )
            assert routed_rank == rank
            assert rank_path == producer_cache / "triton" / f"rank_{rank}"
            selected.add(rank_path)
            repeated_rank, repeated_path = measure._route_rank_triton_cache(
                second["measurement"]
            )
            assert (repeated_rank, repeated_path) == (rank, rank_path)
        assert len(selected) == 8
    finally:
        runner._cleanup_task_tmpdir(task_tmp_a)
        runner._cleanup_task_tmpdir(task_tmp_b)


@pytest.mark.parametrize(
    "bad_case",
    [
        "outside",
        "relative",
        "traversal",
        "symlink",
        "unignored",
        "missing-layout",
        "active-producer",
        "live-process-group",
    ],
)
def test_invalid_or_active_producer_roots_are_rejected_before_attempt_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_case: str,
) -> None:
    task_root, raw_root = _task_roots(tmp_path, monkeypatch)
    producer_cache = _make_producer(raw_root)
    selected = producer_cache
    if bad_case == "outside":
        selected = tmp_path / "outside" / "cache"
    elif bad_case == "relative":
        selected = Path("runs/overnight-streaming-perf/artifacts/raw/baseline003/cache")
    elif bad_case == "traversal":
        selected = task_root / "runs/overnight-streaming-perf/../overnight-streaming-perf/artifacts/raw/baseline003/cache"
    elif bad_case == "symlink":
        outside = tmp_path / "outside"
        outside.mkdir()
        alias = raw_root / "alias"
        alias.symlink_to(outside, target_is_directory=True)
        selected = alias / "cache"
    elif bad_case == "unignored":
        original_git = runner._git

        def unignore_selected(root, *args, check=True):
            if args and args[0] == "check-ignore" and args[-1] == str(
                producer_cache.relative_to(task_root)
            ):
                return SimpleNamespace(returncode=1)
            return original_git(root, *args, check=check)

        monkeypatch.setattr(runner, "_git", unignore_selected)
    elif bad_case == "missing-layout":
        shutil.rmtree(producer_cache / "triton" / "rank_7")
    elif bad_case == "live-process-group":
        monkeypatch.setattr(runner, "_group_exists", lambda _pgid: True)
    elif bad_case == "active-producer":
        _write_json(
            producer_cache.parent / "process.json",
            {"process_group_stopped": False, "orphaned_process_group": False},
        )
        (producer_cache.parent / "status.txt").write_text(
            "RUNNING\n", encoding="utf-8"
        )

    with pytest.raises(RuntimeError):
        _run_dirs(raw_root, "must-not-be-created", selected)
    assert not (raw_root / "must-not-be-created").exists()


def test_runtime_and_canonical_config_guard_allows_new_triton_fingerprint_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, raw_root = _task_roots(tmp_path, monkeypatch)
    producer_cache = _make_producer(raw_root)
    candidate = raw_root / "candidate"
    candidate.mkdir()
    paths = {"attempt": candidate, "cache": producer_cache}
    current_runtime = {
        **PRODUCER_RUNTIME,
        "packages": {
            **PRODUCER_RUNTIME["packages"],
            "triton": {"version": "current-triton", "path": "/venv/triton"},
        },
    }
    result = runner._validate_compiler_cache_reuse(
        paths, current_runtime, CANONICAL_HASH
    )
    assert result == {
        "root": str(producer_cache.resolve()),
        "policy": "reuse",
        "producer_run_id": "baseline003",
    }

    mismatched_runtime = {**current_runtime, "torch": "different-torch-build"}
    with pytest.raises(RuntimeError, match="runtime differs"):
        runner._validate_compiler_cache_reuse(
            paths, mismatched_runtime, CANONICAL_HASH
        )
    with pytest.raises(RuntimeError, match="canonical experiment config differs"):
        runner._validate_compiler_cache_reuse(
            paths, current_runtime, "different-canonical-hash"
        )


def test_cache_option_does_not_change_training_argv_or_update_window(
    tmp_path: Path,
) -> None:
    common = [
        "--run-id",
        "candidate",
        "--steps",
        "140",
        "--warmup",
        "40",
        "--window",
        "100",
    ]
    fresh = runner._parse_args(common)
    reused = runner._parse_args(
        [*common, "--compiler-cache-root", str(tmp_path / "cache")]
    )
    assert (reused.steps, reused.warmup, reused.window) == (140, 40, 100)
    paths = {
        "config": tmp_path / "config",
        "logs": tmp_path / "logs",
        "measurement": tmp_path / "measurement",
        "native": tmp_path / "native",
    }
    fresh_command = runner._command(fresh, paths)
    reused_command = runner._command(reused, paths)
    assert fresh_command == reused_command
    helper_start = reused_command.index(str((runner.HELPER_DIR / "measure.py").resolve())) + 1
    helper_args, train_argv = measure._parse_args(reused_command[helper_start:])
    assert (helper_args.steps, helper_args.warmup, helper_args.window) == (140, 40, 100)
    assert train_argv[0] == str((paths["config"] / "experiment.yaml").resolve())
    assert "--compiler-cache-root" not in reused_command
