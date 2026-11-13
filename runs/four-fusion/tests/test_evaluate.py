from __future__ import annotations

import hashlib
import importlib.util
import json
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "four_fusion_evaluate", ROOT / "runs/four-fusion/evaluate.py"
)
assert SPEC is not None and SPEC.loader is not None
evaluate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluate
SPEC.loader.exec_module(evaluate)


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def write_patch(root: Path, path: Path, relative_target: str, new_text: str) -> None:
    target = root / relative_target
    original = target.read_text(encoding="utf-8")
    target.write_text(new_text, encoding="utf-8")
    path.write_bytes(
        subprocess.run(
            ["git", "-C", str(root), "diff", "--binary", "--", relative_target],
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
    )
    target.write_text(original, encoding="utf-8")


@pytest.fixture
def task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    root = tmp_path / "task"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "Four Fusion Test")
    git(root, "config", "user.email", "four-fusion@example.invalid")
    for relative in evaluate.CORE_TARGETS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("baseline\n", encoding="utf-8")
    git(root, "add", "gift")
    git(root, "commit", "-qm", "baseline")

    fusion = root / "runs/four-fusion"
    fusion.mkdir(parents=True)
    runner_dir = root / "runs/overnight-streaming-perf"
    runner_dir.mkdir(parents=True)
    runner = runner_dir / "run.py"
    runner.write_text("# test runner path\n", encoding="utf-8")
    raw = runner_dir / "artifacts/raw"
    records = fusion / "driver-records"
    patches = {
        name: fusion / filename
        for name, filename in {
            "input": "input-prepare.patch",
            "routing": "routing-metadata.patch",
            "linear-ce": "linear-ce.patch",
            "ce": "ce.patch",
        }.items()
    }
    write_patch(
        root,
        patches["linear-ce"],
        "gift/torchtitan/components/loss.py",
        "baseline\nlinear-ce candidate\n",
    )
    for name, value in {
        "TASK_ROOT": root,
        "FUSION_DIR": fusion,
        "RUNNER": runner,
        "RUNNER_PYTHON": Path(sys.executable),
        "RAW_ROOT": raw,
        "RECORDS_DIR": records,
        "LOCK_PATH": root / ".tmp/four-fusion-evaluate.lock",
        "PATCH_FILES": patches,
    }.items():
        monkeypatch.setattr(evaluate, name, value)
    return {"root": root, "fusion": fusion, "raw": raw, "records": records, **patches}


def write_runner_receipts(
    task: dict[str, Path],
    command: list[str] | tuple[str, ...],
    *,
    group_stopped: bool = True,
    completion_status: str = "PASS",
) -> None:
    runner_id = command[command.index("--run-id") + 1]
    attempt = task["raw"] / runner_id
    attempt.mkdir(parents=True)
    source_hash = "same-before-and-after"
    (attempt / "process.json").write_text(
        json.dumps(
            {
                "pid": 4242,
                "process_group_id": 4242,
                "process_group_stopped": group_stopped,
            }
        ),
        encoding="utf-8",
    )
    (attempt / "completion.json").write_text(
        json.dumps({"status": completion_status}), encoding="utf-8"
    )
    (attempt / "metadata.json").write_text(
        json.dumps(
            {"source_snapshot_before": {"tracked_and_untracked_sha256": source_hash}}
        ),
        encoding="utf-8",
    )
    (attempt / "source-final.json").write_text(
        json.dumps({"tracked_and_untracked_sha256": source_hash}), encoding="utf-8"
    )


def read_record(task: dict[str, Path], run_id: str) -> dict[str, Any]:
    return json.loads((task["records"] / f"{run_id}.json").read_text(encoding="utf-8"))


def test_real_loss_patches_parse_and_apply_check_read_only(tmp_path: Path) -> None:
    relative_target = "gift/torchtitan/components/loss.py"
    baseline = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"HEAD:{relative_target}"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    for variant in ("linear-ce", "ce"):
        patch = evaluate.PATCH_FILES[variant].read_bytes()
        assert evaluate._patch_paths(patch) == (relative_target,)

        test_root = tmp_path / variant
        target = test_root / relative_target
        target.parent.mkdir(parents=True)
        target.write_bytes(baseline)
        git(test_root, "init", "-q")
        git(test_root, "config", "user.name", "Patch Parser Test")
        git(test_root, "config", "user.email", "patch-parser@example.invalid")
        git(test_root, "add", "gift")
        git(test_root, "commit", "-qm", "baseline")
        check = subprocess.run(
            ["git", "-C", str(test_root), "apply", "--check"],
            input=patch,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert check.returncode == 0, check.stderr.decode(errors="replace")


def test_tps_dry_run_prints_one_patch_and_fixed_window(
    task: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def should_not_launch(_command: list[str]) -> int:
        raise AssertionError("dry-run launched the runner")

    monkeypatch.setattr(evaluate, "_run_runner", should_not_launch)
    result = evaluate.run_evaluation(
        "linear-ce",
        "tps",
        "tps-001",
        driver_argv=["linear-ce", "tps", "--run-id", "tps-001", "--dry-run"],
        dry_run=True,
    )
    output = json.loads(capsys.readouterr().out)
    command = output["runner_command_argv"]
    assert result == 0
    assert output["decision"] == "apply exactly one selected patch"
    assert output["patch"]["path"].endswith("linear-ce.patch")
    assert command[command.index("--steps") + 1] == "20"
    assert command[command.index("--warmup") + 1] == "10"
    assert command[command.index("--window") + 1] == "10"
    assert command[command.index("--epoch-size") + 1] == "51200"
    assert command[command.index("--num-prefetch-batches") + 1] == "32"
    assert "--profile-update" not in command
    assert not (task["records"] / "tps-001.json").exists()
    assert (
        task["root"] / "gift/torchtitan/components/loss.py"
    ).read_text() == "baseline\n"


def test_trace_control_uses_existing_runner_profile_contract(
    task: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[str]] = []

    def execute(command: list[str]) -> int:
        seen.append(command)
        write_runner_receipts(task, command)
        return 0

    monkeypatch.setattr(evaluate, "_run_runner", execute)
    result = evaluate.run_evaluation(
        "control",
        "trace",
        "control-trace-001",
        driver_argv=["control", "trace", "--run-id", "control-trace-001"],
    )
    command = seen[0]
    record = read_record(task, "control-trace-001")
    assert result == 0
    assert command[command.index("--steps") + 1] == "11"
    assert command[command.index("--warmup") + 1] == "10"
    assert command[command.index("--window") + 1] == "0"
    assert command[command.index("--profile-update") + 1] == "11"
    assert command[command.index("--profile-microbatches") + 1] == "3"
    assert command[command.index("--epoch-size") + 1] == "51200"
    assert command[command.index("--num-prefetch-batches") + 1] == "32"
    assert "--gradient-accumulation-steps" not in command
    assert record["patch"] is None
    assert record["group_exit_confirmed"] is True
    assert record["completion_status"] == "PASS"


def test_only_selected_patch_is_applied_and_reversed(
    task: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    write_patch(
        task["root"],
        task["routing"],
        "gift/torchtitan/distributed/deepep/hybridep.py",
        "baseline\nrouting candidate\n",
    )
    helper = task["root"] / "gift/torchtitan/experiments/four_fusion/linear_ce.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("# unapplied helper is allowed\n", encoding="utf-8")
    loss = task["root"] / "gift/torchtitan/components/loss.py"

    def execute(command: list[str]) -> int:
        assert loss.read_text() == "baseline\nlinear-ce candidate\n"
        assert (
            task["root"] / "gift/torchtitan/distributed/deepep/hybridep.py"
        ).read_text() == "baseline\n"
        write_runner_receipts(task, command)
        return 0

    monkeypatch.setattr(evaluate, "_run_runner", execute)
    result = evaluate.run_evaluation(
        "linear-ce",
        "tps",
        "linear-ce-001",
        driver_argv=["linear-ce", "tps", "--run-id", "linear-ce-001"],
    )
    record = read_record(task, "linear-ce-001")
    assert result == 0
    assert loss.read_text() == "baseline\n"
    assert record["selected_patch_restored"] is True
    assert (
        record["patch"]["sha256"]
        == hashlib.sha256(task["linear-ce"].read_bytes()).hexdigest()
    )
    status = git(
        task["root"], "status", "--short", "--untracked-files=all", "--", "gift"
    )
    assert "?? gift/torchtitan/experiments/four_fusion/linear_ce.py" in status


def test_nonzero_child_with_stopped_group_reverses_patch(
    task: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def execute(command: list[str]) -> int:
        write_runner_receipts(task, command, completion_status="FAIL")
        return 17

    monkeypatch.setattr(evaluate, "_run_runner", execute)
    result = evaluate.run_evaluation(
        "linear-ce",
        "tps",
        "linear-ce-failed-001",
        driver_argv=["linear-ce", "tps", "--run-id", "linear-ce-failed-001"],
    )
    record = read_record(task, "linear-ce-failed-001")
    assert result == 17
    assert (
        task["root"] / "gift/torchtitan/components/loss.py"
    ).read_text() == "baseline\n"
    assert record["status"] == "FAIL"
    assert record["group_exit_confirmed"] is True
    assert record["selected_patch_restored"] is True


def test_uncertain_runner_group_preserves_selected_patch(
    task: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def execute(command: list[str]) -> int:
        write_runner_receipts(
            task, command, group_stopped=False, completion_status="FAIL"
        )
        return 1

    monkeypatch.setattr(evaluate, "_run_runner", execute)
    with pytest.raises(evaluate.DriverError, match="group exit is not confirmed"):
        evaluate.run_evaluation(
            "linear-ce",
            "tps",
            "linear-ce-uncertain-001",
            driver_argv=["linear-ce", "tps", "--run-id", "linear-ce-uncertain-001"],
        )
    record = read_record(task, "linear-ce-uncertain-001")
    assert (task["root"] / "gift/torchtitan/components/loss.py").read_text() == (
        "baseline\nlinear-ce candidate\n"
    )
    assert record["status"] == "RUNNER_EXIT_UNCERTAIN"
    assert record["group_exit_confirmed"] is False
    assert record["selected_patch_restored"] is False


def test_dirty_core_target_rejects_before_mutation(
    task: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    target = task["root"] / "gift/torchtitan/components/loss.py"
    target.write_text("baseline\nother integration\n", encoding="utf-8")

    def should_not_launch(_command: list[str]) -> int:
        raise AssertionError("dirty target launched the runner")

    monkeypatch.setattr(evaluate, "_run_runner", should_not_launch)
    with pytest.raises(evaluate.DriverError, match="pristine V2 target files"):
        evaluate.run_evaluation(
            "linear-ce",
            "tps",
            "dirty-core-001",
            driver_argv=["linear-ce", "tps", "--run-id", "dirty-core-001"],
        )
    assert target.read_text() == "baseline\nother integration\n"
    assert not (task["raw"] / "four-fusion-linear-ce-dirty-core-001").exists()


def test_termination_signal_is_forwarded_to_runner_as_sigint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evaluate, "TASK_ROOT", tmp_path)
    installed: dict[int, Any] = {}
    sent: list[int] = []

    class FakeChild:
        pid = 9876

        def poll(self) -> None:
            return None

        def send_signal(self, signum: int) -> None:
            sent.append(signum)

        def wait(self) -> int:
            installed[signal.SIGTERM](signal.SIGTERM, None)
            return 130

    def fake_signal(signum: int, handler: Any) -> Any:
        previous = installed.get(signum, signal.SIG_DFL)
        installed[signum] = handler
        return previous

    def fake_popen(
        command: list[str], *, cwd: Path, start_new_session: bool
    ) -> FakeChild:
        assert command == ["runner"]
        assert cwd == tmp_path
        assert start_new_session is True
        return FakeChild()

    monkeypatch.setattr(evaluate.signal, "signal", fake_signal)
    monkeypatch.setattr(evaluate.subprocess, "Popen", fake_popen)
    result = evaluate._run_runner(["runner"])
    assert result == 130
    assert sent == [signal.SIGINT]
