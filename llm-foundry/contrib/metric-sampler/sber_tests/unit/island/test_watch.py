from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from metric_sampler.contract.metrics import Family
from metric_sampler.island import watch

pytestmark = pytest.mark.unit


class _StopLoop(Exception):
    pass


class _Slurm:
    """squeue and sbatch as the watcher calls them: squeue lists `queued` job ids, sbatch submits one."""

    def __init__(self, queued: str) -> None:
        self.queued = queued
        self.commands: list[list[str]] = []

    def run(self, command: list[str], **kwargs) -> subprocess.CompletedProcess:
        assert kwargs == {"capture_output": True, "text": True, "check": True}
        self.commands.append(command)
        stdout = self.queued if command[0] == "squeue" else "Submitted batch job 4242\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout)


def _run_one_pass(monkeypatch: pytest.MonkeyPatch, slurm: _Slurm, argv: list[str]) -> list[int]:
    sleeps: list[int] = []

    def stop_after_first_pass(seconds: int) -> None:
        sleeps.append(seconds)
        raise _StopLoop

    monkeypatch.setattr(watch.subprocess, "run", slurm.run)
    monkeypatch.setattr(watch.getpass, "getuser", lambda: "collector-user")
    monkeypatch.setattr(watch.time, "sleep", stop_after_first_pass)
    with pytest.raises(_StopLoop):
        watch.main(argv)
    return sleeps


def _argv(raw_root: Path) -> list[str]:
    return ["--raw-root", str(raw_root), "--job-name", "ms-collector", "--sbatch", "collector.sbatch"]


_SQUEUE = ["squeue", "--noheader", "--user", "collector-user", "--name", "ms-collector", "--format", "%i"]


def test_watch_submits_the_collector_when_a_window_waits_and_none_is_queued(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], write_window,
) -> None:
    raw_root = tmp_path / "raw"
    write_window(raw_root, "expA", "run1", "0", "host1", 0, 9, families=(Family.SCALARS,))
    slurm = _Slurm(queued="")

    sleeps = _run_one_pass(monkeypatch, slurm, _argv(raw_root))

    assert slurm.commands == [_SQUEUE, ["sbatch", "collector.sbatch"]]
    assert sleeps == [300]
    assert capsys.readouterr().err.rstrip().endswith(" windows to collect: Submitted batch job 4242")


def test_watch_leaves_a_queued_collector_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write_window,
) -> None:
    raw_root = tmp_path / "raw"
    write_window(raw_root, "expA", "run1", "0", "host1", 0, 9, families=(Family.SCALARS,))
    slurm = _Slurm(queued="4241\n")

    sleeps = _run_one_pass(monkeypatch, slurm, [*_argv(raw_root), "--interval-seconds", "60"])

    assert slurm.commands == [_SQUEUE]
    assert sleeps == [60]


@pytest.mark.parametrize(("dumps_flag", "expected_commands"), [([], []), (["--dumps"], [_SQUEUE])])
def test_watch_counts_a_waiting_dump_only_for_a_collector_that_uploads_dumps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write_dump, dumps_flag: list[str],
    expected_commands: list[list[str]],
) -> None:
    raw_root = tmp_path / "raw"
    write_dump(raw_root, "expA", "run1", "0", step=100, rank=0)
    slurm = _Slurm(queued="4241\n")

    _run_one_pass(monkeypatch, slurm, _argv(raw_root) + dumps_flag)

    assert slurm.commands == expected_commands
