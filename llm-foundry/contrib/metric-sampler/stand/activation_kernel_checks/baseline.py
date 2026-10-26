from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import torch
from torch.utils import cpp_extension

from metric_sampler.capture.kernels import activation_ext
from stand.activation_kernel_checks.fake_recorder import FLOAT_SUM_WORDS, WORDS_WITH_QUANTILES
from stand.activation_kernel_checks.kernel_calls import observe_with_quantiles
from stand.activation_kernel_checks.statistics_oracle import MAX_REPORTED_WORDS

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _git_output(root: Path, *arguments: str) -> str:
    return subprocess.run(["git", "-C", str(root), *arguments], check=True, capture_output=True, text=True).stdout


def build_baseline(commit: str):
    """The extension as the commit `commit` has it, with whatever files its csrc directory holds, built beside the
    current one under another name."""
    directory = Path(tempfile.mkdtemp(prefix="activation_ext_baseline_"))
    csrc = f"{commit}:metric_sampler/capture/csrc"
    names = _git_output(REPOSITORY_ROOT, "ls-tree", "--name-only", csrc).splitlines()
    for name in names:
        (directory / name).write_text(_git_output(REPOSITORY_ROOT, "show", f"{csrc}/{name}"))
    sources = [str(directory / name) for name in names if name.endswith((".cpp", ".cu"))]
    return cpp_extension.load(
        name="metric_sampler_activation_ext_baseline", sources=sources, build_directory=str(directory),
        extra_cflags=list(activation_ext._CXX_FLAGS), extra_cuda_cflags=list(activation_ext._CUDA_FLAGS),
    )


def check_baseline_parity(tensors: list[tuple[str, torch.Tensor]], extension, baseline) -> list[str]:
    """Every integer word the extension records for a tensor with its exact quantiles -- the counts, both
    histograms, the absolute max, the quantiles and their counts -- equals the baseline's bit for bit. The
    float sums are not compared: they differ by rounding, and the references of `compare` bound them."""
    problems = []
    for name, tensor in tensors:
        recorded = []
        for module in (baseline, extension):
            words = torch.zeros(WORDS_WITH_QUANTILES, device=tensor.device)
            observe_with_quantiles(module, tensor.reshape(-1, tensor.shape[-1]), words)
            recorded.append(words.view(torch.int32))
        different = (recorded[0] != recorded[1]).nonzero().flatten().tolist()
        different = [word for word in different if word not in FLOAT_SUM_WORDS]
        if different:
            problems.append(f"parity {name}: the baseline and the extension differ at words "
                            f"{different[:MAX_REPORTED_WORDS]}: "
                            f"{recorded[0][different[:MAX_REPORTED_WORDS]].tolist()} vs "
                            f"{recorded[1][different[:MAX_REPORTED_WORDS]].tolist()}")
    return problems
