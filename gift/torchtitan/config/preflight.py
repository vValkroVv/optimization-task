"""In-container preflight: environment checks that pre-submit cannot do.

Runs on the decoded trainer config and its materialized model, before any
runtime object is built. Not in ``__post_init__``: config parsing must stay
side-effect free.
"""

import os
from typing import Any

from .validation import RuntimeInfo, ValidationReport


def preflight(
    config: Any, model_config: Any, runtime: RuntimeInfo | None = None
) -> ValidationReport:
    """Validate the trainer *config* and the materialized *model_config*
    against the container they are about to run in.

    Sections a trainer type does not have (RL keeps ``parallelism`` under
    ``policy_trainer``) are skipped.
    """
    runtime = runtime or RuntimeInfo.from_env()
    report = ValidationReport(stage="in_container")

    _check_output_dir(config.dump_folder, report)
    _check_hf_assets(config.hf_assets_path, report)
    _check_packages(config, model_config, report)
    _check_accelerators(runtime, report)
    _check_symm_mem(getattr(config, "parallelism", None), report)

    return report


def _check_output_dir(dump_folder: str, report: ValidationReport) -> None:
    probe = dump_folder
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(os.path.abspath(probe))
        if parent == probe:
            break
        probe = parent
    if not probe or not os.path.isdir(probe):
        report.error("dump_folder", f"no existing parent directory for {dump_folder!r}")
        return
    if not os.access(probe, os.W_OK):
        report.error("dump_folder", f"not writable: {probe}")


def _check_hf_assets(path: str, report: ValidationReport) -> None:
    if not os.path.exists(path):
        report.error(
            "hf_assets_path",
            f"not found in the container: {path!r}. Download it with "
            "`python scripts/download_hf_assets.py` or fix the mount.",
        )
        return
    if os.path.isfile(path) and path.endswith("tokenizer.model"):
        report.error(
            "hf_assets_path",
            "points at a legacy tokenizer.model file; it must be the directory "
            "holding the HF tokenizer files",
        )


def _check_packages(config: Any, model_config: Any, report: ValidationReport) -> None:
    from torchtitan.config.schema.models.common.moe.dispatch.deepep import (
        DeepEPTokenDispatcherConfig,
    )

    metrics = getattr(config, "metrics", None)
    if metrics is not None:
        if metrics.enable_tensorboard and not _has_module("tensorboard"):
            report.error("metrics.enable_tensorboard", "tensorboard is not installed")
        if metrics.enable_wandb and not _has_module("wandb"):
            report.error("metrics.enable_wandb", "wandb is not installed")
    deepep = next(model_config.traverse(DeepEPTokenDispatcherConfig), None)
    if deepep is not None and not _has_module("deep_ep"):
        report.error(
            f"model.{deepep[0]}", "the deepep backend requires the deep_ep package"
        )


def _has_module(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _check_accelerators(runtime: RuntimeInfo, report: ValidationReport) -> None:
    if runtime.device_count is None:
        report.warn("", "no accelerator visible; the run will fall back to CPU")
        return
    if runtime.local_rank >= runtime.device_count:
        report.error(
            "",
            f"LOCAL_RANK {runtime.local_rank} exceeds the {runtime.device_count} "
            "visible accelerator(s)",
        )


def _check_symm_mem(parallelism: Any, report: ValidationReport) -> None:
    """``parallelism.enable_fsdp_symm_mem`` needs an NVIDIA GPU with compute
    capability >= 9.0 (re-checked at use in ``torchtitan.distributed.fsdp``)."""
    if parallelism is None or not parallelism.enable_fsdp_symm_mem:
        return
    from torchtitan.distributed.fsdp_capabilities import fsdp_symmetric_memory_supported

    if not fsdp_symmetric_memory_supported():
        report.error(
            "parallelism.enable_fsdp_symm_mem",
            "only supported on NVIDIA GPUs with compute capability 9.0 or newer",
        )
