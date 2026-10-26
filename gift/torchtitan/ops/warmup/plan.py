"""Expansion of a warmup config into single-GPU warmup passes.

Warming the DeepGEMM JIT cache means running :mod:`deep_gemm_warmup` once per
GEMM group on a single GPU, plus extra passes that cross the SM90 ``num_stages``
boundary.  A plan owns everything a cluster job wrapper needs -- a command and a
label -- so that job wrappers stay thin, and the passes are independent, so a
submitter can fan them out across the cluster.

This lives next to the entrypoint because it only describes how to invoke it:
the JIT environment is the entrypoint's own (see its ``__main__``) and the
config schema is in :mod:`torchtitan.ops.warmup.config`, which the entrypoint
validates too.  Only stdlib, ``yaml`` and that schema are imported, so a submit
host without CUDA, ``torch`` or ``deep_gemm`` can build a plan.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path

from torchtitan.ops.warmup.config import (
    DEEP_GEMM_SECTION,
    load_warmup_config,
    SM90_STAGES_THRESHOLD,
    TOKEN_BLOCK,
)

ENTRYPOINT_RELPATH = Path("torchtitan") / "ops" / "warmup" / "deep_gemm_warmup.py"

# Conservative budget for an 80 GB GPU, used to size the stages-coverage pass.
_GPU_MEM_BUDGET = 70 * (1 << 30)

# A stages-coverage pass holds x, w and out in bf16 at once.
_PEAK_BYTES_PER_TOKEN_ROW = 2 * 3


def compute_stages_coverage(warmup_config: dict) -> list[tuple[int, int | None]]:
    """Extra ``(max_size_override, max_buffer_tokens)`` pairs to warm.

    A config whose ``max_size`` stays below
    :data:`~torchtitan.ops.warmup.config.SM90_STAGES_THRESHOLD` only compiles one
    of the two ``num_stages`` variants of the k-grouped wgrad kernel; the missing
    one is compiled by an extra pass that pushes ``max_size`` across the
    threshold.  ``max_buffer_tokens`` is ``None`` when the resulting token
    buffers already fit in the GPU memory budget, otherwise it caps them.

    Args:
        warmup_config: The validated ``deep_gemm`` section of a warmup config.

    Returns:
        One pair per extra pass, empty when ``max_size`` already crosses the
        threshold.
    """
    if warmup_config["max_size"] >= SM90_STAGES_THRESHOLD:
        return []

    num_groups = warmup_config["num_groups"]
    # w1 widens to 2 * intermediate, so that is the largest GEMM dimension.
    largest_dim = max(warmup_config["hid_dim"], 2 * warmup_config["int_dim"])
    buf_tokens = SM90_STAGES_THRESHOLD * num_groups
    peak_est = buf_tokens * largest_dim * _PEAK_BYTES_PER_TOKEN_ROW

    if peak_est < _GPU_MEM_BUDGET:
        return [(SM90_STAGES_THRESHOLD, None)]

    stages_buf = min(
        SM90_STAGES_THRESHOLD * 2,
        _GPU_MEM_BUDGET // (largest_dim * _PEAK_BYTES_PER_TOKEN_ROW),
    )
    stages_buf = (stages_buf // TOKEN_BLOCK) * TOKEN_BLOCK
    return [(SM90_STAGES_THRESHOLD, stages_buf)]


@dataclass(frozen=True, kw_only=True)
class WarmupPass:
    """One single-GPU invocation of the DeepGEMM warmup entrypoint."""

    gemm_group: str
    entrypoint: str
    config_path: str
    device: int = 0
    # Stages-coverage overrides; see compute_stages_coverage().
    max_size_override: int | None = None
    max_buffer_tokens: int | None = None

    @property
    def label(self) -> str:
        """Short name for the pass, unique within a plan."""
        suffix = "_stages" if self.max_size_override is not None else ""
        return f"{DEEP_GEMM_SECTION}_{self.gemm_group}{suffix}"

    @property
    def command(self) -> str:
        """Shell command that runs the pass.

        Returns:
            The entrypoint invocation, with the cache layout and the JIT
            environment left to the entrypoint itself.
        """
        command = (
            f"python {shlex.quote(self.entrypoint)}"
            f" --config {shlex.quote(self.config_path)}"
            f" --device {self.device}"
            f" --gemm {shlex.quote(self.gemm_group)}"
        )
        if self.max_size_override is not None:
            command += f" --max-size-override {self.max_size_override}"
        if self.max_buffer_tokens is not None:
            command += f" --max-buffer-tokens {self.max_buffer_tokens}"
        return command


@dataclass(frozen=True, kw_only=True)
class WarmupPlan:
    """Every pass needed to warm one DeepGEMM cache directory."""

    config_path: str
    cache_dir: str
    passes: tuple[WarmupPass, ...]

    def prepare_cache_dir(self) -> None:
        """Create the shared cache directory before the first pass starts.

        Each pass compiles into a node-local temp directory and copies the
        result here, so this is the only directory that has to exist; creating it
        from the submit host fails fast when the shared path is wrong.
        """
        os.makedirs(self.cache_dir, exist_ok=True)


def build_warmup_plan(
    config_path: str | Path, *, gift_root: str | Path, device: int = 0
) -> WarmupPlan:
    """Expand a warmup config into one pass per GEMM group.

    Args:
        config_path: Path to the warmup YAML. It is resolved to an absolute path
            because a job runs with a working directory of its own; the submit
            host and the cluster nodes must share the filesystem it is on, as
            they already must for ``gift_root``.
        gift_root: torchtitan checkout the passes run from. The entrypoint path
            is derived from it, so it must be the same checkout the job puts on
            ``PYTHONPATH``, or a pass would run one checkout against another.
        device: CUDA device index each pass runs on.

    Returns:
        The plan, with the stages-coverage passes last.

    Raises:
        ValueError: ``gift_root`` holds no warmup entrypoint, or the config is
            invalid.
    """
    config_path = str(Path(config_path).expanduser().resolve())
    warmup_config = load_warmup_config(config_path)

    entrypoint = Path(gift_root).expanduser().resolve() / ENTRYPOINT_RELPATH
    if not entrypoint.is_file():
        raise ValueError(
            f"Warmup entrypoint '{entrypoint}' does not exist; 'gift_root' must "
            f"point at a torchtitan checkout the cluster nodes can read"
        )

    gemm_groups = warmup_config["gemm_groups"]
    passes = [
        WarmupPass(
            gemm_group=group,
            entrypoint=str(entrypoint),
            config_path=config_path,
            device=device,
        )
        for group in gemm_groups
    ]
    for max_size_override, max_buffer_tokens in compute_stages_coverage(warmup_config):
        passes.extend(
            WarmupPass(
                gemm_group=group,
                entrypoint=str(entrypoint),
                config_path=config_path,
                device=device,
                max_size_override=max_size_override,
                max_buffer_tokens=max_buffer_tokens,
            )
            for group in gemm_groups
        )

    return WarmupPlan(
        config_path=config_path,
        cache_dir=warmup_config["cache_dir"],
        passes=tuple(passes),
    )
