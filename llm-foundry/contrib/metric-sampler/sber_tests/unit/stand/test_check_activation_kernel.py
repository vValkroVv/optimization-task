import sys
from types import SimpleNamespace

import pytest
import torch

from stand import check_activation_kernel

pytestmark = pytest.mark.unit

CUDA = torch.device("cuda")


def run_main(monkeypatch, arguments, *, cuda=True, extension="extension", tuning_problems=()):
    """The calls the check makes to its GPU phases for a command line, and its exit code (None for a return)."""
    calls = []
    monkeypatch.setattr(sys, "argv", ["check_activation_kernel.py", *arguments])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(
        check_activation_kernel, "activation_ext",
        SimpleNamespace(build=lambda: calls.append(("build",)), load=lambda: extension),
    )
    monkeypatch.setattr(
        check_activation_kernel, "build_baseline", lambda commit: calls.append(("baseline", commit)) or "baseline"
    )
    monkeypatch.setattr(check_activation_kernel, "check_correctness", lambda *a: calls.append(("correctness", *a)))
    monkeypatch.setattr(check_activation_kernel, "time_kernels", lambda *a: calls.append(("kernel table", *a)))
    monkeypatch.setattr(
        check_activation_kernel, "time_extension_paths",
        lambda *a: calls.append(("extension paths", *a)) or list(tuning_problems),
    )
    try:
        check_activation_kernel.main()
    except SystemExit as stop:
        return calls, stop.code
    return calls, None


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ([], [("correctness", CUDA, "extension", None, None), ("kernel table", "extension", None, False),
              ("extension paths", "extension")]),
        (["--correctness-only"], [("correctness", CUDA, "extension", None, None)]),
        (["--timing-only"], [("kernel table", "extension", None, False)]),
        (["--timing-only", "--breakdown"], [("kernel table", "extension", None, True)]),
        (["--breakdown"], [("correctness", CUDA, "extension", None, None), ("kernel table", "extension", None, True),
                           ("extension paths", "extension")]),
        (["--baseline-ref", "abc123"],
         [("baseline", "abc123"), ("correctness", CUDA, "extension", "baseline", "abc123"),
          ("kernel table", "extension", "baseline", False), ("extension paths", "extension")]),
        (["--correctness-only", "--baseline-ref", "abc123"],
         [("baseline", "abc123"), ("correctness", CUDA, "extension", "baseline", "abc123")]),
        (["--timing-only", "--baseline-ref", "abc123"],
         [("baseline", "abc123"), ("kernel table", "extension", "baseline", False)]),
    ],
    ids=["everything", "correctness only", "timing only", "timing only with kernels", "everything with kernels",
         "everything against a commit", "correctness against a commit", "timing against a commit"],
)
def test_the_command_line_selects_the_phases_that_run_after_the_extension_is_built(monkeypatch, arguments, expected):
    calls, exit_code = run_main(monkeypatch, arguments)

    assert calls == [("build",), *expected]
    assert exit_code is None


def test_problems_of_the_tuning_runs_fail_the_check_after_every_run_finished(monkeypatch):
    calls, exit_code = run_main(monkeypatch, [], tuning_problems=["tile=2048 warps=2: differs"])

    assert [call[0] for call in calls] == ["build", "correctness", "kernel table", "extension paths"]
    assert exit_code == 1


def test_a_machine_without_cuda_fails_before_anything_is_built(monkeypatch, capsys):
    calls, exit_code = run_main(monkeypatch, [], cuda=False)

    assert calls == []
    assert exit_code == 1
    assert capsys.readouterr().err == "this check needs a CUDA device\n"


def test_an_extension_that_does_not_load_fails_the_check_after_its_build(monkeypatch, capsys):
    calls, exit_code = run_main(monkeypatch, ["--baseline-ref", "abc123"], extension=None)

    assert calls == [("build",)]
    assert exit_code == 1
    assert capsys.readouterr().err == "activation_ext failed to build/load\n"


def test_correctness_only_and_timing_only_exclude_each_other(monkeypatch, capsys):
    calls, exit_code = run_main(monkeypatch, ["--correctness-only", "--timing-only"])

    assert calls == []
    assert exit_code == 2
    assert "argument --timing-only: not allowed with argument --correctness-only" in capsys.readouterr().err
