import subprocess
import tempfile

import pytest
import torch
from torch.utils import cpp_extension

from stand.activation_kernel_checks.baseline import REPOSITORY_ROOT, build_baseline, check_baseline_parity
from stand.activation_kernel_checks.fake_recorder import (
    AMAX_OFFSET,
    COUNTS_OFFSET,
    MOMENTS_OFFSET,
    POWER_SUMS_OFFSET,
    WORDS_WITH_QUANTILES,
)

pytestmark = pytest.mark.unit


class _RecordingExtension:
    """observe_with_quantiles of a recorder that writes the words it was given, whatever it observes."""

    def __init__(self, recorded: torch.Tensor) -> None:
        self.recorded = recorded

    def observe_with_quantiles(self, x, words, *offsets) -> None:
        words.copy_(self.recorded)


def _recorded_words(**changes: float) -> torch.Tensor:
    words = torch.zeros(WORDS_WITH_QUANTILES)
    words.view(torch.int32)[COUNTS_OFFSET + 1] = 6
    words[AMAX_OFFSET] = 3.0
    for index, value in changes.items():
        words[int(index.removeprefix("word_"))] = value
    return words


def test_baseline_parity_compares_every_integer_word_but_not_the_float_sums():
    tensors = [("tensor", torch.ones(2, 3))]
    baseline = _RecordingExtension(_recorded_words())
    sums = (MOMENTS_OFFSET, POWER_SUMS_OFFSET + 1)
    rounded = _recorded_words(**{f"word_{word}": 1.5 for word in sums})

    assert check_baseline_parity(tensors, _RecordingExtension(rounded), baseline) == []

    other_amax = _recorded_words(**{f"word_{AMAX_OFFSET}": 4.0})
    baseline_bits = _recorded_words().view(torch.int32)[AMAX_OFFSET].item()
    other_bits = other_amax.view(torch.int32)[AMAX_OFFSET].item()
    problems = check_baseline_parity(tensors, _RecordingExtension(other_amax), baseline)
    assert problems == [
        f"parity tensor: the baseline and the extension differ at words [{AMAX_OFFSET}]: "
        f"[{baseline_bits}] vs [{other_bits}]"
    ]


@pytest.mark.parametrize(
    ("files", "sources"),
    [
        (["activation_stats.cpp", "activation_stats.cu"], ["activation_stats.cpp", "activation_stats.cu"]),
        (
            ["bindings.cpp", "kernel_common.cuh", "norm_kernels.cu", "observer_helpers.h", "observers.cpp"],
            ["bindings.cpp", "norm_kernels.cu", "observers.cpp"],
        ),
    ],
    ids=["a source of each language", "sources and headers"],
)
def test_baseline_extension_is_built_from_the_csrc_files_of_a_commit(tmp_path, monkeypatch, files, sources):
    commands = []

    def run(command, **options):
        commands.append((command, options))
        if command[3] == "ls-tree":
            return subprocess.CompletedProcess(command, 0, stdout="".join(f"{name}\n" for name in files))
        return subprocess.CompletedProcess(command, 0, stdout=f"text of {command[-1].split(':')[1]}")

    loaded = {}
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(tempfile, "mkdtemp", lambda prefix: str(tmp_path))
    monkeypatch.setattr(cpp_extension, "load", lambda **options: loaded.update(options) or "module")

    assert build_baseline("abc123") == "module"

    csrc = "abc123:metric_sampler/capture/csrc"
    assert [command[3:] for command, _ in commands] == [
        ["ls-tree", "--name-only", csrc],
        *(["show", f"{csrc}/{name}"] for name in files),
    ]
    assert all(command[:3] == ["git", "-C", str(REPOSITORY_ROOT)] for command, _ in commands)
    assert all(options == {"check": True, "capture_output": True, "text": True} for _, options in commands)
    for name in files:
        assert (tmp_path / name).read_text() == f"text of metric_sampler/capture/csrc/{name}"
    assert loaded == {
        "name": "metric_sampler_activation_ext_baseline",
        "sources": [str(tmp_path / name) for name in sources],
        "build_directory": str(tmp_path),
        "extra_cflags": ["-O3"],
        "extra_cuda_cflags": ["-O3", "--expt-extended-lambda"],
    }


def test_the_baseline_is_read_from_the_repository_that_holds_the_checks():
    assert (REPOSITORY_ROOT / "stand" / "activation_kernel_checks" / "baseline.py").is_file()
    assert (REPOSITORY_ROOT / "metric_sampler" / "capture" / "csrc").is_dir()
