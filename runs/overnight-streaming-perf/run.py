#!/usr/bin/env python3
"""Launch one bounded, task-local eight-H100 native TorchTitan run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TASK_ROOT = Path(__file__).resolve().parents[2]
GIFT_ROOT = TASK_ROOT / "gift"
HELPER_DIR = Path(__file__).resolve().parent
PYTHON = Path("/opt/torchtitan-venv/bin/python3.12")
BASE_CONFIG = GIFT_ROOT / "configs/experiments/gift_100b_crop8_ep8_fp8_compile_cute.yaml"
DATA_FRAGMENT = HELPER_DIR / "mds-dataloader.yaml"
TOKENIZER = GIFT_ROOT / "tests/assets/tokenizer"
RAW_ROOT = HELPER_DIR / "artifacts/raw"
SHARED_HF_CACHE = RAW_ROOT / "shared-cache/huggingface"
DEFAULT_TIMEOUT_SECONDS = 4 * 60 * 60
MIN_SHM_FREE_BYTES = 512 * 1024 * 1024
PROFILE_RANKS = [0, 1]

EXPECTED = {
    "python": "3.12.3",
    "torch": "2.15.0.dev20260822+cu130",
    "torch_cuda": "13.0",
    "streaming": "0.13.0",
    "transformers": "4.57.1",
    "tokenizers": "0.22.2",
    "deep_gemm_build": "2.3.0+f093a4c",
}
DEEP_GEMM_HEADER_SHA256 = "fd8a620440cc31ae7d01010fde1f914aae6998f2dcf6ede2a65e3b5e76001726"
HYBRID_EP_WRAPPER_SHA256 = "63e425916b3c6b0c881ad402af35a2b00c4c84203fcde611a6715744b89210cb"
HYBRID_EP_BACKEND_SHA256 = "b507f65cfa4580f16342076fc2d00afe81e43f64ddefcc4a3cf43681aef11b4b"
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")

WM_SCRIPT = "/etc/profile.d/wm.sh"
WM_BASH = (
    'set -Eeuo pipefail; source /etc/profile.d/wm.sh; '
    'ulimit -c 0; '
    'unset TORCH_LOGS TORCH_COMPILE_DEBUG TORCH_TRACE CUDA_LAUNCH_BLOCKING '
    'TITAN_STRUCT_LOGGER_HANDLERS CUDA_ENABLE_COREDUMP_ON_EXCEPTION '
    'CUDA_COREDUMP_GENERATION_FLAGS CUDA_COREDUMP_SHOW_PROGRESS CUDA_COREDUMP_FILE; '
    'export TMPDIR="${TASK_LOCAL_TMPDIR:?TASK_LOCAL_TMPDIR is required}"; '
    'export TMP="$TMPDIR" TEMP="$TMPDIR"; exec "$@"'
)

RUNTIME_PROBE = r"""
import importlib.metadata as metadata
import importlib.util
import json
import os
import platform
import resource
import sys
from pathlib import Path

import torch
import deep_gemm
import deep_ep.hybrid_ep_buffer as hybrid_ep_buffer
import streaming
import tokenizers
import transformers

def distribution(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None

def origin(name):
    spec = importlib.util.find_spec(name)
    return None if spec is None else spec.origin or str(spec.submodule_search_locations)

gpu_rows = []
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        capability = torch.cuda.get_device_capability(index)
        gpu_rows.append({
            "index": index,
            "name": props.name,
            "capability": list(capability),
            "total_memory_bytes": props.total_memory,
        })

package_dir = Path(deep_gemm.__file__).resolve().parent
header = package_dir / "include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh"
wrapper = Path(hybrid_ep_buffer.__file__).resolve()
backend = wrapper.parent / "backend/hybrid_ep_backend.cuh"

result = {
    "python": platform.python_version(),
    "python_executable": str(Path(sys.executable).resolve()),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "nccl": torch.cuda.nccl.version() if torch.cuda.is_available() else None,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "gpus": gpu_rows,
    "cuda_home": os.environ.get("CUDA_HOME"),
    "core_file_limit": list(resource.getrlimit(resource.RLIMIT_CORE)),
    "packages": {
        "torchtitan": {"version": distribution("torchtitan"), "path": origin("torchtitan")},
        "streaming": {"version": distribution("mosaicml-streaming"), "path": origin("streaming")},
        "transformers": {"version": transformers.__version__, "path": transformers.__file__},
        "tokenizers": {"version": tokenizers.__version__, "path": tokenizers.__file__},
        "deep_gemm": {
            "version": distribution("deep-gemm"),
            "build_version": getattr(deep_gemm, "__build_version__", None),
            "path": deep_gemm.__file__,
            "extension": deep_gemm._C.__file__,
            "header": str(header),
            "header_sha256": __import__("hashlib").sha256(header.read_bytes()).hexdigest(),
        },
        "deep_ep": {
            "version": distribution("deep-ep"),
            "path": origin("deep_ep"),
            "hybrid_ep_wrapper": str(wrapper),
            "hybrid_ep_wrapper_sha256": __import__("hashlib").sha256(wrapper.read_bytes()).hexdigest(),
            "hybrid_ep_backend": str(backend),
            "hybrid_ep_backend_sha256": __import__("hashlib").sha256(backend.read_bytes()).hexdigest(),
        },
        "cutlass_dsl": {"version": distribution("nvidia-cutlass-dsl")},
        "tvm_ffi": {"version": distribution("tvm-ffi")},
    },
}
print(json.dumps(result, sort_keys=True))
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def verify_task_root(root: Path) -> None:
    actual = Path(
        _git(root, "rev-parse", "--show-toplevel").stdout.decode().strip()
    ).resolve()
    if actual != root.resolve():
        raise RuntimeError(f"expected task Git root {root}, found {actual}")


def source_snapshot(root: Path) -> dict[str, Any]:
    tracked = _git(root, "ls-files", "-z", "--cached", "--", "gift/").stdout
    untracked = _git(
        root, "ls-files", "-z", "--others", "--exclude-standard", "--", "gift/"
    ).stdout
    tracked_paths = sorted(path for path in tracked.split(b"\0") if path)
    untracked_paths = sorted(path for path in untracked.split(b"\0") if path)

    def manifest(paths: list[bytes]) -> str:
        digest = hashlib.sha256()
        for raw_path in paths:
            relative = raw_path.decode("utf-8", errors="strict")
            path = root / relative
            digest.update(raw_path)
            digest.update(b"\0")
            if not path.exists() and not path.is_symlink():
                digest.update(b"missing\0")
                continue
            if path.is_symlink():
                digest.update(b"symlink\0")
                digest.update(os.readlink(path).encode("utf-8"))
                digest.update(b"\0")
                continue
            mode = path.stat().st_mode & 0o7777
            digest.update(f"{mode:o}\0".encode("ascii"))
            digest.update(sha256_file(path).encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()

    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all", "--", "gift/").stdout
    return {
        "head": _git(root, "rev-parse", "HEAD").stdout.decode().strip(),
        "tracked_file_count": len(tracked_paths),
        "untracked_file_count": len(untracked_paths),
        "tracked_sha256": manifest(tracked_paths),
        "untracked_sha256": manifest(untracked_paths),
        "tracked_and_untracked_sha256": manifest(
            sorted(set(tracked_paths).union(untracked_paths))
        ),
        "git_status_sha256": hashlib.sha256(status).hexdigest(),
    }


def _inside(root: Path, path: Path, label: str) -> Path:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(f"{label} must remain inside the task root: {path}") from exc
    return resolved_path


def _wm_run(argv: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", "-c", WM_BASH, "campaign-wm", *argv],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


def validate_args(args: argparse.Namespace) -> None:
    if not RUN_ID_RE.fullmatch(args.run_id) or args.run_id in {".", ".."}:
        raise ValueError("run-id must contain only letters, digits, dot, underscore, or hyphen")
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.warmup < 0 or args.window < 0:
        raise ValueError("warmup and window must be nonnegative optimizer-update counts")
    if args.window and (args.warmup <= 0 or args.steps != args.warmup + args.window):
        raise ValueError("measured runs require warmup>0 and steps == warmup + window")
    if not args.window and args.warmup and args.profile_update is None:
        raise ValueError("warmup must be zero when window is disabled")
    if args.profile_update is not None:
        if args.window:
            raise ValueError("profile-update requires window 0")
        if args.warmup <= 0:
            raise ValueError("profile-update requires warmup > 0")
        if args.profile_update < 2 or args.profile_update <= args.warmup:
            raise ValueError("profile-update must be at least 2 and greater than warmup")
        if args.steps != args.profile_update:
            raise ValueError("profile-update requires steps == profile-update")
    if args.timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be positive")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", type=int, default=20, help="optimizer updates before stopping")
    parser.add_argument("--warmup", type=int, default=10, help="optimizer updates before timing")
    parser.add_argument("--window", type=int, default=10, help="measured optimizer updates; 0 disables timing")
    parser.add_argument("--observe-compiler-cache", action="store_true", help="optional boundary cache observations; changes are diagnostic only")
    parser.add_argument("--profile-update", type=int, help="capture one native profiler update on global ranks 0 and 1")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a YAML mapping in {path}")
    return value


def _data_root(root: Path) -> Path:
    fragment = _load_yaml(DATA_FRAGMENT)
    try:
        stream = fragment["dataloader"]["dataset"]["streams"][0]
        local = Path(stream["local"])
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"{DATA_FRAGMENT} lacks dataloader.dataset.streams[0].local") from exc
    if not local.is_absolute():
        raise RuntimeError("MDS stream local path must be absolute")
    local = _inside(root, local, "MDS data path")
    if not local.is_dir():
        raise RuntimeError(f"MDS data directory does not exist: {local}")
    return local


def _make_run_dirs(run_id: str) -> dict[str, Path]:
    raw_root = _inside(TASK_ROOT, RAW_ROOT, "raw artifact root")
    ignored = subprocess.run(
        ["git", "-C", str(TASK_ROOT), "check-ignore", "-q", "--", str(RAW_ROOT.relative_to(TASK_ROOT))],
        check=False,
    )
    if ignored.returncode != 0:
        raise RuntimeError(f"task raw artifact path is not Git-ignored: {raw_root}")
    raw_root.mkdir(parents=True, exist_ok=True)
    attempt = raw_root / run_id
    attempt.mkdir(mode=0o755, exist_ok=False)

    paths = {
        "attempt": attempt,
        "logs": attempt / "logs",
        "preflight": attempt / "preflight",
        "config": attempt / "config",
        "native": attempt / "native",
        "measurement": attempt / "measurement",
        "cache": attempt / "cache",
        "shared_hf_cache": SHARED_HF_CACHE,
    }
    for key, path in paths.items():
        _inside(TASK_ROOT, path, key)
        if key == "shared_hf_cache":
            path.mkdir(parents=True, exist_ok=True)
        elif key != "attempt":
            path.mkdir(parents=True, exist_ok=False)
    if paths["shared_hf_cache"].is_symlink():
        raise RuntimeError("persistent Hugging Face cache must not be a symlink")
    for name in ("torchinductor", "triton", "deep_gemm", "hybridep"):
        cache = paths["cache"] / name
        cache.mkdir(exist_ok=False)
        paths[f"cache_{name}"] = cache
    rank_cache = paths["cache_deep_gemm"] / "rank-cache"
    rank_cache.mkdir(exist_ok=False)
    paths["cache_deep_gemm_ranks"] = rank_cache
    return paths


def _make_local_tmpdir(run_id: str) -> tuple[Path, dict[str, int]]:
    local_root = Path("/dev/shm")
    if not local_root.is_dir():
        raise RuntimeError("local /dev/shm is unavailable for task-owned TMPDIR")
    stats = os.statvfs(local_root)
    free_bytes = stats.f_bavail * stats.f_frsize
    free_inodes = stats.f_favail
    if free_bytes < MIN_SHM_FREE_BYTES or free_inodes <= 0:
        raise RuntimeError(
            f"/dev/shm has insufficient free capacity: {free_bytes} bytes, "
            f"{free_inodes} inodes"
        )
    path = Path(
        tempfile.mkdtemp(
            prefix=f"vkropotin-overnight-{run_id}-",
            dir=local_root,
        )
    )
    return path, {
        "total_bytes": stats.f_blocks * stats.f_frsize,
        "available_bytes_before_create": free_bytes,
        "available_inodes_before_create": free_inodes,
    }


def _cleanup_local_tmpdir(path: Path) -> None:
    if path.is_symlink():
        raise RuntimeError(f"refusing to remove a symlink temporary path: {path}")
    resolved = path.resolve()
    if resolved.parent != Path("/dev/shm").resolve():
        raise RuntimeError(f"refusing to remove temporary path outside /dev/shm: {path}")
    if not resolved.name.startswith("vkropotin-overnight-"):
        raise RuntimeError(f"refusing to remove a non-task temporary path: {path}")
    shutil.rmtree(resolved)


def _filesystem_status(paths: dict[str, Path], mds_root: Path) -> dict[str, Any]:
    result = {}
    check_paths = {
        "attempt": paths["attempt"],
        "cache": paths["cache"],
        "shared_hf_cache": paths["shared_hf_cache"],
        "mds_data": mds_root,
    }
    for key, path in check_paths.items():
        stats = os.statvfs(path)
        free_bytes = stats.f_bavail * stats.f_frsize
        free_inodes = stats.f_favail
        result[key] = {
            "path": str(path.resolve()),
            "available_bytes": free_bytes,
            "available_inodes": free_inodes,
        }
        if free_bytes <= 0 or free_inodes <= 0:
            raise RuntimeError(f"no free disk blocks or inodes at {path}")
    return result


def _write_new_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")


def _write_atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return value


def _write_replace_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".update.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _write_replace_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".update.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(value)
        if not value.endswith("\n"):
            handle.write("\n")
    temporary.replace(path)


class RunInterrupted(Exception):
    def __init__(self, pid: int, process_group_stopped: bool) -> None:
        super().__init__("runner interrupted; owned process group cleanup was attempted")
        self.pid = pid
        self.process_group_stopped = process_group_stopped


def _write_new_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)
        if not value.endswith("\n"):
            handle.write("\n")


def _runtime_preflight(env: dict[str, str], paths: dict[str, Path]) -> dict[str, Any]:
    if not PYTHON.is_file() or not os.access(PYTHON, os.X_OK):
        raise RuntimeError(f"verified Python is unavailable: {PYTHON}")
    if not Path(WM_SCRIPT).is_file():
        raise RuntimeError(f"job image CUDA profile is unavailable: {WM_SCRIPT}")

    apps = _wm_run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"],
        env=env,
    ).stdout
    _write_new_text(paths["preflight"] / "compute-apps.txt", apps or "empty\n")
    if apps.strip():
        raise RuntimeError("GPU compute applications are already present")

    gpu_csv = _wm_run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,compute_mode,driver_version,memory.used",
            "--format=csv,noheader",
        ],
        env=env,
    ).stdout
    _write_new_text(paths["preflight"] / "gpus.csv", gpu_csv)
    gpu_rows = list(csv.reader(gpu_csv.splitlines()))
    if len(gpu_rows) != 8 or any("H100" not in row[0] for row in gpu_rows):
        raise RuntimeError("expected exactly eight visible H100 GPUs")

    topology = _wm_run(["nvidia-smi", "topo", "-m"], env=env).stdout
    _write_new_text(paths["preflight"] / "topology.txt", topology)

    runtime = json.loads(_wm_run([str(PYTHON), "-c", RUNTIME_PROBE], env=env).stdout)
    if runtime["python"] != EXPECTED["python"]:
        raise RuntimeError(f"unexpected Python version {runtime['python']}")
    if runtime["torch"] != EXPECTED["torch"] or runtime["torch_cuda"] != EXPECTED["torch_cuda"]:
        raise RuntimeError(f"unexpected torch/CUDA build: {runtime['torch']} / {runtime['torch_cuda']}")
    if not runtime["cuda_available"] or runtime["cuda_device_count"] != 8:
        raise RuntimeError("PyTorch must expose exactly eight CUDA devices")
    if any(gpu["capability"] != [9, 0] or "H100" not in gpu["name"] for gpu in runtime["gpus"]):
        raise RuntimeError("all visible CUDA devices must be H100/SM90")
    for package in ("streaming", "transformers", "tokenizers"):
        actual = runtime["packages"][package]["version"]
        if actual != EXPECTED[package]:
            raise RuntimeError(f"unexpected {package} version {actual}; expected {EXPECTED[package]}")
    if runtime["packages"]["deep_gemm"]["build_version"] != EXPECTED["deep_gemm_build"]:
        raise RuntimeError("unexpected DeepGEMM build version")
    if runtime["packages"]["deep_gemm"]["header_sha256"] != DEEP_GEMM_HEADER_SHA256:
        raise RuntimeError("DeepGEMM SM90 FP8 header hash does not match the smoke baseline")
    if runtime["packages"]["deep_ep"]["hybrid_ep_wrapper_sha256"] != HYBRID_EP_WRAPPER_SHA256:
        raise RuntimeError("installed HybridEP wrapper hash does not match the smoke baseline")
    if runtime["packages"]["deep_ep"]["hybrid_ep_backend_sha256"] != HYBRID_EP_BACKEND_SHA256:
        raise RuntimeError("installed HybridEP backend hash does not match the smoke baseline")
    expected_torchtitan = (GIFT_ROOT / "torchtitan/__init__.py").resolve()
    if Path(runtime["packages"]["torchtitan"]["path"]).resolve() != expected_torchtitan:
        raise RuntimeError("torchtitan import does not resolve to this task's GIFT tree")
    if Path(runtime["packages"]["streaming"]["path"]).resolve().is_relative_to(
        (GIFT_ROOT / "contrib/streaming").resolve()
    ):
        raise RuntimeError("streaming must resolve to the installed native package")

    nvcc = Path(runtime["cuda_home"] or "/usr/local/cuda") / "bin/nvcc"
    nvcc_output = _wm_run([str(nvcc), "--version"], env=env).stdout
    _write_new_text(paths["preflight"] / "nvcc-version.txt", nvcc_output)
    if "V13.2.51" not in nvcc_output:
        raise RuntimeError("expected CUDA nvcc V13.2.51 from the verified smoke image")
    _write_new_json(paths["preflight"] / "runtime.json", runtime)
    return runtime


def _write_experiment_config(path: Path) -> None:
    import yaml

    document = {
        "config_version": 1,
        "imports": [str(BASE_CONFIG.resolve()), str(DATA_FRAGMENT.resolve())],
        # Null removes inherited callbacks before CallbackManager builds hooks.
        "callbacks": {"monitoring": None, "system_metrics": None, "timers": None},
        "loss": {"num_chunks_on_monitoring_steps": None},
    }
    with path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(document, handle, sort_keys=False)


def _profile_config_values(args: argparse.Namespace) -> dict[str, Any]:
    values: dict[str, Any] = {
        "profiler.enable_profiling": args.profile_update is not None,
        "profiler.enable_memory_snapshot": False,
    }
    if args.profile_update is not None:
        values.update(
            {
                "profiler.profile_ranks": PROFILE_RANKS,
                "profiler.save_traces_folder": "profiling/traces",
                "profiler.record_shapes": False,
                "profiler.profile_freq": 2,
                "profiler.profiler_warmup": 1,
                "profiler.profiler_active": 1,
                "profiler.profiler_repeat": 1,
                "profiler.profiler_skip_first": args.profile_update - 2,
                "profiler.profiler_skip_first_wait": 0,
            }
        )
    return values


def _profile_capture_spec(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.profile_update is None:
        return None
    return {
        "target_update": args.profile_update,
        "warmup_updates": args.warmup,
        "profile_ranks": PROFILE_RANKS,
        "settings": _profile_config_values(args),
        "native_hardcoded": {"with_stack": False, "profile_memory": False},
        "trace_directory": (
            f"native/profiling/traces/iteration_{args.profile_update}"
        ),
    }


def _training_overrides(args: argparse.Namespace, paths: dict[str, Path]) -> list[str]:
    overrides = [
        "--training.steps",
        str(args.steps),
        "--lr_scheduler.total_steps",
        "65000",
        "--lr_scheduler.warmup_steps",
        "2000",
        "--debug.seed",
        "2048",
        "--metrics.log_freq",
        "1",
        "--checkpoint.enable",
        "false",
        "--validator.enable",
        "false",
        "--profiler.enable_profiling",
        str(args.profile_update is not None).lower(),
        "--profiler.enable_memory_snapshot",
        "false",
        "--dump_folder",
        str(paths["native"].resolve()),
        "--hf_assets_path",
        str(TOKENIZER.resolve()),
    ]
    for dotted, value in _profile_config_values(args).items():
        if dotted in {
            "profiler.enable_profiling",
            "profiler.enable_memory_snapshot",
        }:
            continue
        if isinstance(value, bool):
            raw_value = str(value).lower()
        elif isinstance(value, list):
            raw_value = "[" + ", ".join(str(item) for item in value) + "]"
        else:
            raw_value = str(value)
        overrides.extend((f"--{dotted}", raw_value))
    return overrides


def _run_environment(paths: dict[str, Path]) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(GIFT_ROOT), str(HELPER_DIR)))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["HF_HOME"] = str(paths["shared_hf_cache"].resolve())
    env["HF_HUB_CACHE"] = str((paths["shared_hf_cache"] / "hub").resolve())
    env["HF_DATASETS_CACHE"] = str((paths["shared_hf_cache"] / "datasets").resolve())
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["TORCHINDUCTOR_CACHE_DIR"] = str(paths["cache_torchinductor"].resolve())
    env["TRITON_CACHE_DIR"] = str(paths["cache_triton"].resolve())
    env["DG_JIT_SHARED_CACHE_DIR"] = str(paths["cache_deep_gemm"].resolve())
    env["TASK_DEEP_GEMM_RANK_CACHE_ROOT"] = str(
        paths["cache_deep_gemm_ranks"].resolve()
    )
    env["HYBRID_EP_CACHE_DIR"] = str(paths["cache_hybridep"].resolve())
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["TORCHINDUCTOR_CUDAGRAPHS"] = "0"
    env["TORCHINDUCTOR_MIX_ORDER_REDUCTION"] = "0"
    env["GIFT_LOG_RANK"] = "all"
    env["TORCHFT_LIGHTHOUSE"] = "http://localhost:29510"
    env["TASK_LOCAL_TMPDIR"] = str(paths["tmp_dir"])
    for key in (
        "GIFT_HYBRID_EP_SCAN_RUNTIME_DIR",
        "DG_CACHE_FORCE_SKIP",
        "PROPAGATE_DG_CACHE",
        "PROPAGATE_DG_CACHE_DIR",
        "TORCH_LOGS",
        "TORCH_COMPILE_DEBUG",
        "TORCH_TRACE",
        "CUDA_LAUNCH_BLOCKING",
        "TITAN_STRUCT_LOGGER_HANDLERS",
        "CUDA_ENABLE_COREDUMP_ON_EXCEPTION",
        "CUDA_COREDUMP_GENERATION_FLAGS",
        "CUDA_COREDUMP_SHOW_PROGRESS",
        "CUDA_COREDUMP_FILE",
    ):
        env.pop(key, None)
    return env


def _command(args: argparse.Namespace, paths: dict[str, Path]) -> list[str]:
    helper_args = [
        "--measurement-dir",
        str(paths["measurement"].resolve()),
        "--run-id",
        args.run_id,
        "--steps",
        str(args.steps),
        "--warmup",
        str(args.warmup),
        "--window",
        str(args.window),
    ]
    if args.profile_update is not None:
        helper_args.extend(("--profile-update", str(args.profile_update)))
    if args.observe_compiler_cache:
        helper_args.append("--observe-compiler-cache")
    return [
        str(PYTHON),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=8",
        "--max_restarts=0",
        "--log_dir",
        str(paths["logs"] / "torchrun"),
        "--redirects=3",
        "--tee=3",
        str((HELPER_DIR / "measure.py").resolve()),
        *helper_args,
        "--",
        str((paths["config"] / "experiment.yaml").resolve()),
        *_training_overrides(args, paths),
    ]


def _terminate_owned_group(process: subprocess.Popen[Any]) -> bool:
    group_id = process.pid
    if process.poll() is None:
        try:
            os.killpg(group_id, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            os.killpg(group_id, 0)
        except ProcessLookupError:
            break
        time.sleep(1)
    else:
        try:
            os.killpg(group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    return not _group_exists(group_id)


def _group_exists(group_id: int) -> bool:
    try:
        os.killpg(group_id, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_for_group_exit(group_id: int, timeout: int) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _group_exists(group_id):
            return True
        time.sleep(0.2)
    return not _group_exists(group_id)


def _launch(
    command: list[str], env: dict[str, str], paths: dict[str, Path], timeout: int
) -> tuple[int, bool, int | None, bool, bool]:
    launcher_log = paths["logs"] / "launcher.log"
    with launcher_log.open("xb") as log:
        process = subprocess.Popen(
            ["/bin/bash", "-c", WM_BASH, "campaign-training", *command],
            cwd=GIFT_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        pid = process.pid
        _write_new_text(paths["attempt"] / "launcher.pid", f"{pid}\n")
        try:
            child_code = process.wait(timeout=timeout)
            orphaned_group = False
            group_stopped = _wait_for_group_exit(process.pid, timeout=2)
            if not group_stopped:
                orphaned_group = True
                group_stopped = _terminate_owned_group(process)
            return child_code, False, pid, orphaned_group, group_stopped
        except subprocess.TimeoutExpired:
            group_stopped = _terminate_owned_group(process)
            return 124, True, pid, False, group_stopped
        except KeyboardInterrupt:
            group_stopped = _terminate_owned_group(process)
            raise RunInterrupted(process.pid, group_stopped) from None


def _config_hashes() -> dict[str, str]:
    hashes = {
        "canonical_experiment_sha256": sha256_file(BASE_CONFIG),
        "mds_fragment_sha256": sha256_file(DATA_FRAGMENT),
        "runner_sha256": sha256_file(HELPER_DIR / "run.py"),
        "measurement_adapter_sha256": sha256_file(HELPER_DIR / "measure.py"),
        "parser_sha256": sha256_file(HELPER_DIR / "parse_results.py"),
        "compiler_observer_sha256": sha256_file(HELPER_DIR / "compiler_stats.py"),
    }
    return hashes


def _verify_resolved_config(paths: dict[str, Path], args: argparse.Namespace) -> str:
    resolved_path = paths["native"] / "config/resolved_config.yaml"
    if not resolved_path.is_file():
        raise RuntimeError(f"native resolved config was not written: {resolved_path}")
    resolved = _load_yaml(resolved_path)
    required = {
        "training.steps": args.steps,
        "training.local_batch_size": 2,
        "training.global_batch_size": 512,
        "training.seq_len": 4096,
        "lr_scheduler.total_steps": 65000,
        "lr_scheduler.warmup_steps": 2000,
        "debug.seed": 2048,
        "loss.num_chunks": 2,
        "metrics.log_freq": 1,
        "checkpoint.enable": False,
        "validator.enable": False,
        "compile.enable": True,
        "activation_checkpoint.mode": "selective",
        "parallelism.expert_parallel_degree": 8,
        "parallelism.pipeline_parallel_degree": 1,
        "parallelism.tensor_parallel_degree": 1,
        "parallelism.context_parallel_degree": 1,
        "fp8.sparse_use_grouped_gemm": True,
        "fp8.deep_gemm_warmup_on_start": True,
    }
    required.update(_profile_config_values(args))
    for dotted, expected in required.items():
        current: Any = resolved
        for component in dotted.split("."):
            if not isinstance(current, dict) or component not in current:
                raise RuntimeError(f"resolved config lacks {dotted}")
            current = current[component]
        if current != expected:
            raise RuntimeError(f"resolved config {dotted}={current!r}, expected {expected!r}")
    callbacks = resolved.get("callbacks", {})
    if any(name in callbacks for name in ("monitoring", "system_metrics", "timers")):
        raise RuntimeError("resolved performance config contains detailed monitoring callbacks")
    if "ema" not in callbacks:
        raise RuntimeError("resolved performance config lost the EMA callback")
    if resolved.get("loss", {}).get("num_chunks_on_monitoring_steps") is not None:
        raise RuntimeError("resolved performance config contains monitoring loss chunking")
    if resolved.get("dataloader", {}).get("type") != "streaming_text":
        raise RuntimeError("resolved config did not select native streaming_text data")
    if Path(resolved.get("dump_folder", "")).resolve() != paths["native"].resolve():
        raise RuntimeError("resolved dump_folder is outside this run's native output")
    if Path(resolved.get("hf_assets_path", "")).resolve() != TOKENIZER.resolve():
        raise RuntimeError("resolved tokenizer path differs from the task-owned tokenizer")
    return sha256_file(resolved_path)


def _load_parser():
    import parse_results

    return parse_results


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths: dict[str, Path] | None = None
    status = "FAIL"
    exit_code = 1
    error: str | None = None
    timed_out = False
    orphaned_process_group = False
    process_group_stopped = True
    pid: int | None = None
    tmp_dir: Path | None = None
    source_before: dict[str, Any] | None = None
    config_before: dict[str, str] | None = None
    try:
        verify_task_root(TASK_ROOT)
        if Path(sys.executable).resolve() != PYTHON.resolve():
            raise RuntimeError(f"run.py must use {PYTHON}, got {sys.executable}")
        for required in (BASE_CONFIG, DATA_FRAGMENT, TOKENIZER / "tokenizer.json"):
            if not required.is_file():
                raise RuntimeError(f"required task input is missing: {required}")
        paths = _make_run_dirs(args.run_id)
        mds_root = _data_root(TASK_ROOT)
        filesystem = _filesystem_status(paths, mds_root)
        tmp_dir, tmp_capacity = _make_local_tmpdir(args.run_id)
        paths["tmp_dir"] = tmp_dir
        _write_new_json(
            paths["preflight"] / "local-tmpdir.json",
            {"path": str(tmp_dir), "capacity": tmp_capacity},
        )

        experiment_config = paths["config"] / "experiment.yaml"
        _write_experiment_config(experiment_config)
        env = _run_environment(paths)
        source_before = source_snapshot(TASK_ROOT)
        config_before = _config_hashes()
        config_before["generated_experiment_sha256"] = sha256_file(experiment_config)
        runtime = _runtime_preflight(env, paths)

        metadata = {
            "run_id": args.run_id,
            "task_root": str(TASK_ROOT),
            "source_head_before": source_before["head"],
            "source_snapshot_before": source_before,
            "config_hashes_before": config_before,
            "steps": args.steps,
            "monitor_interval": 0,
            "detailed_monitoring": "disabled; callbacks omitted; ordinary two-chunk loss",
            "warmup_updates": args.warmup,
            "window_updates": args.window,
            "scheduler_total_steps": 65000,
            "scheduler_warmup_steps": 2000,
            "seed": 2048,
            "mds_data_root": str(mds_root),
            "filesystem": filesystem,
            "local_tmpdir": {
                "path": str(tmp_dir),
                "capacity": tmp_capacity,
                "cleanup": "remove this exact directory after the owned process group stops",
            },
            "runtime": runtime,
            "profiler": (
                "disabled"
                if args.profile_update is None
                else "enabled for one update on global ranks 0 and 1"
            ),
            "checkpointing": "disabled",
            "validation": "disabled",
            "memory_snapshots": "disabled",
            "compiler_cache_policy": "fresh task-owned TorchInductor, Triton, DeepGEMM shared-kernel and rank-cache directories per run",
            "deep_gemm_rank_cache_root": str(paths["cache_deep_gemm_ranks"].resolve()),
            "deep_gemm_warmup": "enabled; adapter redirects rank cache; PROPAGATE_DG_CACHE unset",
            "observe_compiler_cache": args.observe_compiler_cache,
            "huggingface_cache_policy": "persistent task-owned cache with offline mode",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if args.profile_update is not None:
            metadata["profile_capture"] = _profile_capture_spec(args)
        command = _command(args, paths)
        metadata["argv"] = command
        _write_new_json(paths["attempt"] / "metadata.json", metadata)
        _write_new_text(paths["attempt"] / "status.txt", "RUNNING\n")
        print(f"run_id={args.run_id}")
        print(f"attempt={paths['attempt']}")
        print(f"world_size=8 timeout_seconds={args.timeout_seconds}")
        print("launching exactly one native torchrun process group")

        child_code, timed_out, pid, orphaned_process_group, process_group_stopped = _launch(
            command, env, paths, args.timeout_seconds
        )
        _write_atomic_json(
            paths["attempt"] / "process.json",
            {
                "pid": pid,
                "process_group_id": pid,
                "timeout": timed_out,
                "orphaned_process_group": orphaned_process_group,
                "process_group_stopped": process_group_stopped,
                "child_exit_code": child_code,
            },
        )

        source_after = source_snapshot(TASK_ROOT)
        config_after = _config_hashes()
        config_after["generated_experiment_sha256"] = sha256_file(experiment_config)
        source_unchanged = (
            source_after["tracked_and_untracked_sha256"]
            == source_before["tracked_and_untracked_sha256"]
        )
        config_unchanged = config_after == config_before
        _write_atomic_json(paths["attempt"] / "source-final.json", source_after)
        _write_atomic_json(paths["attempt"] / "config-hashes-final.json", config_after)

        resolved_hash = None
        if child_code == 0 and not timed_out:
            resolved_hash = _verify_resolved_config(paths, args)
            parser = _load_parser()
            summary = parser.summarize_run(paths["attempt"])
            _write_atomic_json(paths["attempt"] / "summary.json", summary)
            if summary.get("status") != "complete":
                raise RuntimeError(f"rank measurement records are incomplete: {summary.get('errors')}")
        if not source_unchanged:
            raise RuntimeError("GIFT training source changed during the run")
        if not config_unchanged:
            raise RuntimeError("canonical or generated training config changed during the run")
        if orphaned_process_group:
            raise RuntimeError("owned torchrun process group did not exit cleanly")
        if not process_group_stopped:
            raise RuntimeError("owned torchrun process group survived exact-PID cleanup")
        if child_code != 0 or timed_out:
            raise RuntimeError(
                f"native torchrun exited with code {child_code}"
                + (" after timeout" if timed_out else "")
            )
        _write_atomic_json(
            paths["attempt"] / "completion.json",
            {
                "status": "PASS",
                "resolved_config_sha256": resolved_hash,
                "source_unchanged": source_unchanged,
                "config_unchanged": config_unchanged,
                "local_tmpdir": str(tmp_dir),
                "local_tmpdir_removed": False,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        status = "PASS"
        exit_code = 0
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"runner: {error}", file=sys.stderr)
        if isinstance(exc, RunInterrupted):
            pid = exc.pid
            process_group_stopped = exc.process_group_stopped
            failure_status = "INTERRUPTED"
        elif isinstance(exc, KeyboardInterrupt):
            failure_status = "INTERRUPTED"
        else:
            failure_status = "TIMEOUT" if timed_out else "FAIL"
        if paths is not None:
            try:
                _write_atomic_json(
                    paths["attempt"] / "failure.json",
                    {
                        "status": failure_status,
                        "error": error,
                        "pid": pid,
                        "timed_out": timed_out,
                        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
            except FileExistsError:
                pass
        status = "TIMEOUT" if timed_out else "FAIL"
        exit_code = 124 if timed_out else 1
        if isinstance(exc, RunInterrupted):
            exit_code = 130
            status = "INTERRUPTED"
        elif isinstance(exc, KeyboardInterrupt):
            exit_code = 130
            status = "INTERRUPTED"
    finally:
        if paths is not None:
            if (
                source_before is not None
                and process_group_stopped
                and not (paths["attempt"] / "source-final.json").exists()
            ):
                try:
                    source_after = source_snapshot(TASK_ROOT)
                    config_after = _config_hashes()
                    experiment_path = paths["config"] / "experiment.yaml"
                    if experiment_path.is_file():
                        config_after["generated_experiment_sha256"] = sha256_file(
                            experiment_path
                        )
                    _write_atomic_json(paths["attempt"] / "source-final.json", source_after)
                    _write_atomic_json(
                        paths["attempt"] / "config-hashes-final.json", config_after
                    )
                except Exception as freeze_error:
                    print(f"runner: could not record final source/config hashes: {freeze_error}", file=sys.stderr)
                    if exit_code == 0:
                        exit_code = 1
                        status = "FAIL"
            tmp_dir = paths.get("tmp_dir")
            if tmp_dir is not None and tmp_dir.exists() and process_group_stopped:
                try:
                    _cleanup_local_tmpdir(tmp_dir)
                    _write_atomic_json(
                        paths["attempt"] / "local-tmpdir-cleanup.json",
                        {"path": str(tmp_dir), "removed": True},
                    )
                except Exception as cleanup_error:
                    print(f"runner: temporary directory cleanup failed: {cleanup_error}", file=sys.stderr)
                    if exit_code == 0:
                        exit_code = 1
                        status = "FAIL"
            if status == "PASS":
                completion_path = paths["attempt"] / "completion.json"
                if completion_path.is_file():
                    completion = _load_json(completion_path)
                    completion["local_tmpdir_removed"] = not tmp_dir.exists() if tmp_dir else True
                    _write_replace_json(completion_path, completion)
            elif tmp_dir is not None and tmp_dir.exists() and not process_group_stopped:
                print(
                    f"runner: preserving {tmp_dir}; owned children may still be active",
                    file=sys.stderr,
                )
            if status != "PASS":
                completion_path = paths["attempt"] / "completion.json"
                if completion_path.is_file():
                    completion = _load_json(completion_path)
                    completion["status"] = status
                    completion["failure"] = error or "temporary-directory cleanup or launch failed"
                    completion["local_tmpdir_removed"] = (
                        not tmp_dir.exists() if tmp_dir else True
                    )
                    _write_replace_json(completion_path, completion)
            try:
                _write_new_text(paths["attempt"] / "exit-code.txt", f"{exit_code}\n")
                _write_replace_text(paths["attempt"] / "status.txt", f"{status}\n")
            except FileExistsError:
                pass
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
