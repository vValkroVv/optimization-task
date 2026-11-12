"""Install short-run native allocator observation around the canonical entry."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import runpy

from torchtitan.experiments.cuda_graph_campaign.phase_memory import (
    install_phase_memory_observer,
)


logger = logging.getLogger(__name__)
BOUNDARIES = {"startup_end_step": 20, "expected_end_step": 40}
ENVIRONMENT_WHITELIST = {
    "CUDA_DEVICE_ORDER",
    "CUDA_HOME",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES",
    "DG_JIT_SHARED_CACHE_DIR",
    "GIFT_HYBRID_EP_SCAN_RUNTIME_DIR",
    "HF_DATASETS_CACHE",
    "HF_DATASETS_OFFLINE",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_HUB_OFFLINE",
    "HOME",
    "LD_LIBRARY_PATH",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "NCCL_CROSS_NIC",
    "NCCL_DEBUG",
    "NCCL_IB_HCA",
    "NCCL_P2P_LEVEL",
    "NCCL_SOCKET_IFNAME",
    "OMP_NUM_THREADS",
    "PATH",
    "PYTHONPATH",
    "PYTORCH_ALLOC_CONF",
    "RANK",
    "TORCHFT_LIGHTHOUSE",
    "TORCHTITAN_PHASE_MEMORY_RUN_ID",
    "TORCHTITAN_PHASE_MEMORY_SOURCE_ID",
    "TORCHTITAN_PHASE_MEMORY_CONFIG_ID",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCHINDUCTOR_CUDAGRAPHS",
    "TORCHINDUCTOR_MIX_ORDER_REDUCTION",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "TRANSFORMERS_OFFLINE",
    "TRITON_CACHE_DIR",
    "WORLD_SIZE",
}


def _report_secondary(
    primary: BaseException, label: str, secondary: BaseException
) -> None:
    note = f"phase-memory {label} also failed: {type(secondary).__name__}: {secondary}"
    add_note = getattr(primary, "add_note", None)
    if callable(add_note):
        add_note(note)
    logger.error(note)


def _assert_complete(output_path: Path) -> None:
    document = json.loads(output_path.read_text(encoding="utf-8"))
    if document.get("status") != "complete":
        raise RuntimeError(
            f"phase-memory record is {document.get('status')!r}: {output_path}"
        )
    if document.get("phase_boundaries") != BOUNDARIES:
        raise RuntimeError("phase-memory record has unexpected boundaries")
    if document.get("training_log_frequency") != 10:
        raise RuntimeError("native metrics log frequency was not the expected 10")
    coverage = document.get("coverage", {})
    if coverage.get("last_training_metrics_step") != 40:
        raise RuntimeError("phase-memory did not observe native metrics step 40")


def _record_environment() -> None:
    rank = int(os.environ["RANK"])
    runtime_dir = Path(__file__).resolve().parent / "runtime"
    runtime_dir.mkdir(exist_ok=True)
    values = {
        name: os.environ.get(name) for name in sorted(ENVIRONMENT_WHITELIST)
    }
    values = {name: value for name, value in values.items() if value is not None}
    record = {
        "rank": rank,
        "allowlisted_environment": values,
        "unset_or_unrecorded_allowlist_names": sorted(
            ENVIRONMENT_WHITELIST - values.keys()
        ),
        "secret_values_excluded": True,
    }
    path = runtime_dir / f"rank-{rank:05d}-environment.json"
    with path.open("x", encoding="utf-8") as output:
        json.dump(record, output, indent=2, sort_keys=True)
        output.write("\n")


def main() -> None:
    _record_environment()
    output_dir = Path(__file__).resolve().parent / "phase_memory"
    observer = install_phase_memory_observer(output_dir, BOUNDARIES)
    compiler_entry = Path(__file__).resolve().with_name("compiler_entry.py")

    try:
        # The canonical entry records compiler settings, then enters native Trainer.
        runpy.run_path(str(compiler_entry), run_name="__main__")
    except BaseException as primary:
        try:
            observer.finish_and_flush(primary)
        except BaseException as flush_error:
            _report_secondary(primary, "flush", flush_error)
        try:
            observer.restore()
        except BaseException as restore_error:
            _report_secondary(primary, "restoration", restore_error)
        raise
    else:
        cleanup_error: BaseException | None = None
        try:
            observer.finish_and_flush()
        except BaseException as exc:
            cleanup_error = exc
        try:
            observer.restore()
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
            else:
                _report_secondary(cleanup_error, "restoration", exc)
        if cleanup_error is not None:
            raise cleanup_error.with_traceback(cleanup_error.__traceback__)

        rank = int(os.environ["RANK"])
        _assert_complete(output_dir / f"rank-{rank:05d}.json")


if __name__ == "__main__":
    main()
