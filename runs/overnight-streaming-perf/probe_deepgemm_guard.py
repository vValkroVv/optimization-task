#!/usr/bin/env python3
"""One-shot DeepGEMM cache-miss guard probe; not training or a benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_ROOT = Path(__file__).resolve().parents[2]
GIFT_ROOT = TASK_ROOT / "gift"
EXPERIMENT = GIFT_ROOT / "configs/experiments/gift_100b_crop8_ep8_fp8_compile_cute.yaml"
PRESET = GIFT_ROOT / "configs/presets/models/gift_moe/gift_3_5/100b_crop8.yaml"
WARMUP = GIFT_ROOT / "torchtitan/ops/warmup/deep_gemm_fast_warmup.py"
RAW_BASE = TASK_ROOT / "runs/overnight-streaming-perf/artifacts/raw/g0"
RUN_ROOT = RAW_BASE / "deepgemm-guard-probe-001"
PYTHON = Path("/opt/torchtitan-venv/bin/python3.12")
BUILD_VERSION = "2.3.0+f093a4c"
EXTENSION_SHA256 = "99b7afad2e2fe7ebabd26c22afc99dc652e2e6935fcbf37324359eb5aeeefe26"
TORCH_VERSION = "2.15.0.dev20260822+cu130"
TIMEOUT_SECONDS = 600


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def cache_env(root: Path, visible_gpu: str) -> dict[str, str]:
    """Give all caches used by this helper call a fresh task-owned location."""
    paths = {
        "DG_JIT_CACHE_DIR": root / "rank_0" / "deepgemm",
        "DG_JIT_SHARED_CACHE_DIR": root / "shared",
        "TRITON_CACHE_DIR": root / "triton",
        "TORCHINDUCTOR_CACHE_DIR": root / "inductor",
        "CUTE_DSL_CACHE_DIR": root / "cute_dsl",
        "TVM_FFI_CACHE_DIR": root / "tvm_ffi",
        "TORCH_EXTENSIONS_DIR": root / "torch_extensions",
        "CUDA_CACHE_PATH": root / "cuda_driver",
        "XDG_CACHE_HOME": root / "xdg",
        "TMPDIR": root / "tmp",
    }
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(GIFT_ROOT),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        CUDA_VISIBLE_DEVICES=visible_gpu,
    )
    env.pop("DG_CACHE_FORCE_SKIP", None)
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path.resolve())
    # The inspected helper/package cache layout appends `cache` to both roots.
    (paths["DG_JIT_CACHE_DIR"] / "cache").mkdir(exist_ok=True)
    (paths["DG_JIT_SHARED_CACHE_DIR"] / "cache").mkdir(exist_ok=True)
    env["TMP"] = env["TEMP"] = env["TMPDIR"]
    return env


def require_empty_deepgemm_caches(env: dict[str, str]) -> None:
    for key in ("DG_JIT_CACHE_DIR", "DG_JIT_SHARED_CACHE_DIR"):
        path = Path(env[key]) / "cache"
        if not path.is_dir() or any(path.iterdir()):
            raise RuntimeError(f"expected empty private cache before import: {key}={path}")


def load_contract() -> dict[str, int]:
    import yaml

    experiment = yaml.safe_load(EXPERIMENT.read_text())
    preset = yaml.safe_load(PRESET.read_text())
    model = preset["model"]["config"]
    dim = int(model["dim"])
    expert_shapes = {
        (int(block["moe"]["experts"]["hidden_dim"]), int(block["moe"]["experts"]["num_experts"]))
        for block in model["blocks"].values()
        if "moe" in block and "experts" in block["moe"]
    }
    if len(expert_shapes) != 1:
        raise RuntimeError(f"expected one MoE geometry, got {sorted(expert_shapes)}")
    hidden_dim, experts = next(iter(expert_shapes))
    override = experiment.get("model", {}).get("config", {})
    if "dim" in override and int(override["dim"]) != dim:
        raise RuntimeError("experiment overrides the preset model dimension")
    for block in override.get("blocks", {}).values():
        experts_override = block.get("moe", {}).get("experts", {})
        for key, expected in (("hidden_dim", hidden_dim), ("num_experts", experts)):
            if key in experts_override and int(experts_override[key]) != expected:
                raise RuntimeError(f"experiment overrides expert {key}")
    ep = int(experiment["parallelism"]["expert_parallel_degree"])
    sms = int(experiment["fp8"]["sparse_deep_gemm_num_sms"])
    if experts % ep or (dim, hidden_dim, experts, ep, sms) != (4096, 1536, 128, 8, 116):
        raise RuntimeError(
            f"unexpected crop contract D/H/E/EP/num_sms={dim}/{hidden_dim}/{experts}/{ep}/{sms}"
        )
    return {
        "dim": dim,
        "expert_hidden_dim": hidden_dim,
        "expert_parallel_degree": ep,
        "experts_per_rank": experts // ep,
        "sparse_deep_gemm_num_sms": sms,
    }


def tensor_budget(mode: str, call: tuple, groups: int) -> dict[str, int]:
    """Count named tensors; opaque quantizer scratch and kernel workspace are excluded."""
    def ceil_div(n: int, d: int) -> int:
        return (n + d - 1) // d

    if mode == "m_grouped":
        _, m, n, k, _ = call
        source = m * k + groups * n * k
        scale = m * ceil_div(k, 128) + groups * ceil_div(n, 128) * ceil_div(k, 128)
        return {
            "bf16_source_bytes": 2 * source,
            "fp8_payload_bytes": source,
            "bf16_output_bytes": 2 * m * n,
            "scale_bytes": 4 * scale,
            "index_bytes": 4 * m,
            "named_tensor_budget_bytes": 3 * source + 2 * m * n + 4 * scale + 4 * m,
        }
    if mode == "k_grouped":
        _, m, n, max_k = call
        rows = max_k + (groups - 1) * 128
        source = rows * (m + n)
        scale_rows = rows * (ceil_div(m, 128) + ceil_div(n, 128))
        scale_copy = rows * max(ceil_div(m, 128), ceil_div(n, 128))
        output = groups * m * n
        return {
            "input_rows": rows,
            "bf16_source_bytes": 2 * source,
            "fp8_rowwise_payload_bytes": source,
            "fp8_columnwise_payload_bytes": source,
            "bf16_output_bytes": 2 * output,
            "scale_bytes": 4 * (2 * scale_rows + scale_copy),
            "index_and_group_size_bytes": 4 * rows + 8 * groups,
            "named_tensor_budget_bytes": 4 * source + 2 * output + 4 * (2 * scale_rows + scale_copy) + 4 * rows + 8 * groups,
        }
    raise ValueError(mode)


def choose_pair(planned: list[tuple], mode: str, groups: int) -> dict[str, Any]:
    by_name: dict[str, list[tuple[tuple, tuple]]] = {}
    for name, call, config in planned:
        if call[0] == mode:
            by_name.setdefault(name, []).append((tuple(call), tuple(config)))
    candidates = []
    per_name = []
    for name, entries in sorted(by_name.items()):
        pairs = []
        for i, (call_a, config_a) in enumerate(entries):
            for call_b, config_b in entries[i + 1 :]:
                if config_a == config_b:
                    continue
                a_key = call_a[1] if mode == "m_grouped" else call_a[3]
                b_key = call_b[1] if mode == "m_grouped" else call_b[3]
                if b_key < a_key:
                    call_a, call_b, config_a, config_b = call_b, call_a, config_b, config_a
                estimate_a = tensor_budget(mode, call_a, groups)
                estimate_b = tensor_budget(mode, call_b, groups)
                pairs.append(
                    {
                        "name": name,
                        "a": {"call": call_a, "config": config_a, "estimate": estimate_a},
                        "b": {"call": call_b, "config": config_b, "estimate": estimate_b},
                        "max_estimated_bytes": max(
                            estimate_a["named_tensor_budget_bytes"],
                            estimate_b["named_tensor_budget_bytes"],
                        ),
                    }
                )
        if pairs:
            best = min(pairs, key=lambda item: item["max_estimated_bytes"])
            candidates.append(best)
            per_name.append({"name": name, "max_estimated_bytes": best["max_estimated_bytes"]})
    if not candidates:
        raise RuntimeError(f"actual _plan has no distinct {mode} configuration pair")
    selected = min(candidates, key=lambda item: (item["max_estimated_bytes"], item["name"]))
    if selected["a"]["config"] == selected["b"]["config"]:
        raise RuntimeError(f"selected {mode} configs are equal")
    return {
        "selected": selected,
        "per_name_minimums": per_name,
        "selection_rule": "minimum maximum named-tensor budget among same-name _plan pairs with distinct configs",
    }


def build_plan() -> dict[str, Any]:
    sys.path.insert(0, str(GIFT_ROOT))
    from torchtitan.ops.warmup import deep_gemm_fast_warmup as warmup

    contract = load_contract()
    planned = warmup._plan(
        contract["dim"],
        contract["expert_hidden_dim"],
        contract["experts_per_rank"],
        contract["sparse_deep_gemm_num_sms"],
    )
    return {
        "contract": {**contract, "planned_template_count": len(planned)},
        "m_grouped": choose_pair(planned, "m_grouped", contract["experts_per_rank"]),
        "k_grouped": choose_pair(planned, "k_grouped", contract["experts_per_rank"]),
        "interpretation": "Guard/cache behavior only; not training, parity, performance, or convergence evidence.",
        "memory_budget_note": (
            "Named BF16/FP8/source/conversion/output/scale/index tensors are counted; opaque quantizer scratch and "
            "DeepGEMM workspace are excluded. Each child requires 2x the largest named-tensor budget plus 512 MiB free."
        ),
    }


def gpu_state(*, require_zero_utilization: bool = True) -> list[dict[str, Any]]:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory,gpu_uuid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    rows = []
    for row in csv.reader(gpu.stdout.splitlines()):
        if row:
            index, name, used, total, util = [value.strip() for value in row]
            rows.append(
                {
                    "index": int(index),
                    "name": name,
                    "memory_used_mib": int(used),
                    "memory_total_mib": int(total),
                    "utilization_percent": int(util),
                }
            )
    active = [
        row.strip()
        for row in apps.stdout.splitlines()
        if row.strip() and row.strip().lower() != "no running processes found"
    ]
    if len(rows) != 8 or active:
        raise RuntimeError(f"expected eight visible idle GPUs; count={len(rows)} compute_apps={active}")
    busy = [
        row
        for row in rows
        if "H100" not in row["name"]
        or row["memory_total_mib"] < 75 * 1024
        or row["memory_used_mib"] > 512
        or (require_zero_utilization and row["utilization_percent"] != 0)
    ]
    if busy:
        raise RuntimeError(f"GPU preflight found busy or unexpected devices: {busy}")
    return rows


def wait_for_gpu_release() -> list[dict[str, Any]]:
    last_error = None
    for attempt in range(6):
        try:
            return gpu_state(require_zero_utilization=False)
        except Exception as exc:
            last_error = exc
            if attempt < 5:
                time.sleep(2)
    raise RuntimeError(f"GPU ownership did not release within 10s: {last_error}")


def cache_entries(root: Path) -> list[dict[str, Any]]:
    result = []
    for entry in sorted((root / "cache").glob("kernel.*")):
        if not entry.is_dir():
            continue
        files = []
        for path in sorted(p for p in entry.rglob("*") if p.is_file()):
            files.append({"path": str(path.relative_to(entry)), "bytes": path.stat().st_size})
        result.append({"name": entry.name, "files": files, "bytes": sum(f["bytes"] for f in files)})
    return result


def cache_state(env: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    return {
        "rank": cache_entries(Path(env["DG_JIT_CACHE_DIR"])),
        "shared": cache_entries(Path(env["DG_JIT_SHARED_CACHE_DIR"])),
    }


def invoke(warmup: Any, handle: Any, device: Any, groups: int, call: tuple, flag: str, label: str, *, miss: bool = False, compile: bool = False) -> dict[str, Any]:
    env = os.environ.copy()
    before = cache_state(env)
    os.environ["DG_CACHE_FORCE_SKIP"] = flag
    try:
        warmup._run(call, groups, handle, device)
    except Exception as exc:
        after = cache_state(env)
        message = str(exc)
        if not miss or "CACHE MISS" not in message or after != before:
            raise RuntimeError(f"{label}: unexpected {type(exc).__name__}: {message}") from exc
        return {
            "phase": label,
            "flag": flag,
            "result": "expected_cache_miss",
            "exception_type": type(exc).__name__,
            "exception": message,
            "cache_unchanged": True,
            "cache_entry_names": {
                family: [item["name"] for item in entries]
                for family, entries in after.items()
            },
        }
    after = cache_state(env)
    new_names = {
        (family, item["name"])
        for family, entries in after.items()
        for item in entries
    } - {
        (family, item["name"])
        for family, entries in before.items()
        for item in entries
    }
    if miss or (compile and not new_names) or (not compile and after != before):
        raise RuntimeError(f"{label}: unexpected cache outcome; new_entries={sorted(new_names)}")
    new_entries = [
        {"cache": family, **item}
        for family, entries in after.items()
        for item in entries
        if (family, item["name"]) in new_names
    ]
    return {
        "phase": label,
        "flag": flag,
        "result": "success",
        "new_cache_entries": new_entries,
        "cache_unchanged": after == before,
        "cache_entry_names": {
            family: [item["name"] for item in entries]
            for family, entries in after.items()
        },
    }


def child(mode: str, plan_path: Path, result_path: Path) -> int:
    env = os.environ.copy()
    require_empty_deepgemm_caches(env)
    if env.get("CUDA_VISIBLE_DEVICES") != "0" or "DG_CACHE_FORCE_SKIP" in env:
        raise RuntimeError("child must start with only GPU 0 visible and no guard override")
    core = resource.getrlimit(resource.RLIMIT_CORE)
    if core[0] != 0:
        raise RuntimeError(f"child core dump soft limit must be zero, got {core}")

    import torch
    import deep_gemm
    from torchtitan.ops.float8.grouped_gemm import _get_deep_gemm_handle
    from torchtitan.ops.warmup import deep_gemm_fast_warmup as warmup

    extension = Path(deep_gemm._C.__file__).resolve()
    extension_sha = sha256(extension)
    if (
        getattr(deep_gemm, "__build_version__", None) != BUILD_VERSION
        or extension_sha != EXTENSION_SHA256
        or torch.__version__ != TORCH_VERSION
        or torch.version.cuda != "13.0"
        or Path(warmup.__file__).resolve() != WARMUP.resolve()
    ):
        raise RuntimeError("installed runtime/helper identity differs from the reviewed probe target")
    if os.environ.get("DG_JIT_MINIMIZE_NUM_SMS", "0") not in ("", "0"):
        raise RuntimeError("unsupported inherited DG_JIT_MINIMIZE_NUM_SMS override")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected exactly one available CUDA device")
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    if "H100" not in props.name or torch.cuda.get_device_capability(0) != (9, 0):
        raise RuntimeError(f"expected H100 SM90, got {props.name}")

    spec = json.loads(plan_path.read_text())
    if spec["selected_mode"] != mode:
        raise RuntimeError("child mode does not match selected plan")
    selected = spec["selected"]
    call_a, call_b = tuple(selected["a"]["call"]), tuple(selected["b"]["call"])
    config_a, config_b = tuple(selected["a"]["config"]), tuple(selected["b"]["config"])
    if config_a == config_b:
        raise RuntimeError("A and B must have distinct configurations")
    contract = load_contract()
    planned = warmup._plan(
        contract["dim"],
        contract["expert_hidden_dim"],
        contract["experts_per_rank"],
        contract["sparse_deep_gemm_num_sms"],
    )
    planned = {(name, tuple(call), tuple(config)) for name, call, config in planned}
    if (selected["name"], call_a, config_a) not in planned or (selected["name"], call_b, config_b) not in planned:
        raise RuntimeError("A/B calls or configs are no longer returned by the actual _plan")

    groups = contract["experts_per_rank"]
    budget = max(
        selected["a"]["estimate"]["named_tensor_budget_bytes"],
        selected["b"]["estimate"]["named_tensor_budget_bytes"],
    )
    free, total = torch.cuda.mem_get_info(0)
    if free < 2 * budget + 512 * 1024**2:
        raise RuntimeError(f"insufficient free memory: free={free}, named_tensor_budget={budget}")
    torch.manual_seed(2048)
    handle = _get_deep_gemm_handle(num_sms=contract["sparse_deep_gemm_num_sms"])
    device = torch.device("cuda", torch.cuda.current_device())
    phases = [
        invoke(warmup, handle, device, groups, call_a, "1", "A compile flag1", compile=True),
        invoke(warmup, handle, device, groups, call_a, "0", "A cache hit flag0"),
        invoke(warmup, handle, device, groups, call_b, "0", "B absent-cache flag0", miss=True),
        invoke(warmup, handle, device, groups, call_b, "1", "B compile flag1", compile=True),
    ]
    torch.cuda.synchronize(0)
    result = {
        "status": "PASS",
        "mode": mode,
        "completed_at_utc": now(),
        "seed": 2048,
        "python": {"version": sys.version, "executable": str(Path(sys.executable).resolve())},
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
        "gpu": {
            "name": props.name,
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_bytes": total,
            "free_bytes_before_calls": free,
        },
        "deep_gemm": {"build_version": BUILD_VERSION, "extension": str(extension), "sha256": extension_sha},
        "warmup_source": str(Path(warmup.__file__).resolve()),
        "private_rank_and_shared_caches_empty_before_import": True,
        "cache_roots": {
            key: env[key]
            for key in (
                "DG_JIT_CACHE_DIR", "DG_JIT_SHARED_CACHE_DIR", "TRITON_CACHE_DIR",
                "TORCHINDUCTOR_CACHE_DIR", "CUTE_DSL_CACHE_DIR", "TVM_FFI_CACHE_DIR",
                "TORCH_EXTENSIONS_DIR", "CUDA_CACHE_PATH", "TMPDIR",
            )
        },
        "phases": phases,
        "cache_entries_after": cache_state(env),
        "interpretation": "Guard/cache behavior only; no training, parity, timing, speedup, or convergence claim.",
    }
    write_json(result_path, result)
    print(json.dumps({"mode": mode, "status": "PASS"}), flush=True)
    return 0


def stop_owned_group(process: subprocess.Popen, *, timed_out: bool) -> tuple[int, bool, bool]:
    if str(TASK_ROOT / "runs/overnight-streaming-perf") not in sys.path:
        sys.path.insert(0, str(TASK_ROOT / "runs/overnight-streaming-perf"))
    import run as task_runner

    if Path(task_runner.__file__).resolve() != TASK_ROOT / "runs/overnight-streaming-perf/run.py":
        raise RuntimeError("refusing an unexpected process-group cleanup helper")
    group_stopped = task_runner._terminate_owned_group(process)
    return process.returncode if process.returncode is not None else 1, timed_out, group_stopped


def plan_only() -> dict[str, Any]:
    RAW_BASE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".deepgemm-plan-", dir=RAW_BASE) as temp:
        env = cache_env(Path(temp) / "cache", visible_gpu="")
        result = subprocess.run(
            [str(PYTHON), str(Path(__file__).resolve()), "--_internal-plan"],
            cwd=TASK_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return json.loads(result.stdout)


def execute() -> int:
    if Path.cwd().resolve() != TASK_ROOT or Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError(f"run from {TASK_ROOT} with {PYTHON}")
    if RUN_ROOT.exists():
        raise RuntimeError(f"refusing to reuse existing artifact root {RUN_ROOT}")
    RUN_ROOT.mkdir(parents=True, exist_ok=False)
    log_path, receipt_path = RUN_ROOT / "probe.log", RUN_ROOT / "receipt.json"
    log_path.touch(exist_ok=False)
    receipt: dict[str, Any] = {
        "status": "RUNNING",
        "run_id": RUN_ROOT.name,
        "started_at_utc": now(),
        "task_root": str(TASK_ROOT),
        "python": str(PYTHON),
        "source_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=TASK_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "probe_script_sha256": sha256(Path(__file__).resolve()),
        "warmup_source_sha256": sha256(WARMUP),
        "experiment_sha256": sha256(EXPERIMENT),
        "model_preset_sha256": sha256(PRESET),
        "launch_prefix": "source /etc/profile.d/wm.sh; ulimit -c 0",
        "timeout_seconds_total": TIMEOUT_SECONDS,
        "log": str(log_path.relative_to(TASK_ROOT)),
    }
    write_json(receipt_path, receipt)
    deadline = time.monotonic() + TIMEOUT_SECONDS
    preflight = None
    code = 1
    children = []
    with log_path.open("a") as log:
        try:
            preflight = gpu_state()
            receipt["preflight_gpus"] = preflight
            plan_env = cache_env(RUN_ROOT / "plan", visible_gpu="")
            if plan_env.get("DG_JIT_MINIMIZE_NUM_SMS", "0") not in ("", "0"):
                raise RuntimeError("unsupported inherited DG_JIT_MINIMIZE_NUM_SMS override")
            require_empty_deepgemm_caches(plan_env)
            plan_result = subprocess.run(
                [str(PYTHON), str(Path(__file__).resolve()), "--_internal-plan"],
                cwd=TASK_ROOT,
                env=plan_env,
                check=True,
                capture_output=True,
                text=True,
                timeout=min(60, max(1, int(deadline - time.monotonic()))),
            )
            if plan_result.stderr:
                log.write(plan_result.stderr)
                log.flush()
            plan = {**json.loads(plan_result.stdout), "started_at_utc": now()}
            plan_path = RUN_ROOT / "plan.json"
            write_json(plan_path, plan)
            receipt["plan"] = str(plan_path.relative_to(TASK_ROOT))
            receipt["plan_sha256"] = sha256(plan_path)
            receipt["compiler_cache_policy"] = (
                "backend variables inherited; only DG_CACHE_FORCE_SKIP toggled; DeepGEMM local/shared, "
                "Triton, Inductor, CuTe DSL, TVM-FFI, extension, CUDA driver, XDG, and TMP caches private"
            )
            write_json(receipt_path, receipt)

            for mode in ("m_grouped", "k_grouped"):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("600-second total deadline exhausted")
                spec_path = RUN_ROOT / f"{mode}.plan.json"
                result_path = RUN_ROOT / f"{mode}.result.json"
                spec = {
                    "selected_mode": mode,
                    "selected": plan[mode]["selected"],
                    "contract": plan["contract"],
                    "started_at_utc": plan["started_at_utc"],
                }
                write_json(spec_path, spec)
                env = cache_env(RUN_ROOT / mode, visible_gpu="0")
                if env.get("DG_JIT_MINIMIZE_NUM_SMS", "0") not in ("", "0"):
                    raise RuntimeError("unsupported inherited DG_JIT_MINIMIZE_NUM_SMS override")
                require_empty_deepgemm_caches(env)
                command = [
                    str(PYTHON), str(Path(__file__).resolve()), "--_child-mode", mode,
                    "--plan-json", str(spec_path), "--result-json", str(result_path),
                ]
                receipt.setdefault("child_commands", []).append(command)
                receipt.setdefault("child_cache_env", {})[mode] = {
                    key: env[key]
                    for key in (
                        "CUDA_VISIBLE_DEVICES", "DG_JIT_CACHE_DIR", "DG_JIT_SHARED_CACHE_DIR",
                        "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "CUTE_DSL_CACHE_DIR",
                        "TVM_FFI_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "CUDA_CACHE_PATH", "TMPDIR",
                    )
                }
                write_json(receipt_path, receipt)
                log.write(f"starting {mode}; one child, CUDA_VISIBLE_DEVICES=0\n")
                log.flush()
                print(f"Starting {mode} guard probe", flush=True)
                process = subprocess.Popen(
                    command,
                    cwd=TASK_ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                timed_out = False
                try:
                    process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    timed_out = True
                except BaseException:
                    stop_owned_group(process, timed_out=True)
                    raise
                child_code, timed_out, group_stopped = stop_owned_group(process, timed_out=timed_out)
                child_result = json.loads(result_path.read_text()) if result_path.is_file() else None
                row = {
                    "mode": mode,
                    "exit_code": child_code,
                    "timed_out_or_orphan_group": timed_out,
                    "process_group_stopped": group_stopped,
                    "result_path": str(result_path.relative_to(TASK_ROOT)),
                    "result": child_result,
                }
                children.append(row)
                receipt["children"] = children
                write_json(receipt_path, receipt)
                if child_code != 0 or timed_out or not group_stopped or not child_result or child_result.get("status") != "PASS":
                    raise RuntimeError(f"{mode} child failed; inspect the preserved probe log")
            receipt["status"] = "PASS"
            code = 0
        except BaseException as exc:
            receipt["status"] = "FAIL"
            receipt["failure"] = {"type": type(exc).__name__, "message": str(exc)}
            receipt["children"] = children
            code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        finally:
            if preflight is not None:
                try:
                    receipt["final_gpus"] = wait_for_gpu_release()
                except Exception as exc:
                    receipt["final_gpu_state_error"] = f"{type(exc).__name__}: {exc}"
                    if receipt["status"] == "PASS":
                        receipt["status"] = "FAIL"
                        receipt["failure"] = {"type": "GPUPostflightError", "message": str(exc)}
                        code = 1
            receipt["completed_at_utc"] = now()
            write_json(receipt_path, receipt)
    print(f"DeepGEMM guard probe {receipt['status']}: {receipt_path}", flush=True)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    public = parser.add_mutually_exclusive_group()
    public.add_argument("--plan-only", action="store_true", help="print the CPU-only call and memory plan")
    public.add_argument("--execute", action="store_true", help="run the bounded guard probe")
    parser.add_argument("--_internal-plan", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_child-mode", choices=("m_grouped", "k_grouped"), help=argparse.SUPPRESS)
    parser.add_argument("--plan-json", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result-json", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._internal_plan:
        env = os.environ.copy()
        require_empty_deepgemm_caches(env)
        if args.execute or args.plan_only or args._child_mode:
            parser.error("internal planner mode cannot be combined with other modes")
        print(json.dumps(build_plan(), sort_keys=True))
        return 0
    if args._child_mode:
        if args.execute or args.plan_only or args.plan_json is None or args.result_json is None:
            parser.error("child mode requires plan/result paths and no public mode")
        return child(args._child_mode, args.plan_json, args.result_json)
    if args.plan_only:
        print(json.dumps(plan_only(), indent=2, sort_keys=True))
        return 0
    if args.execute:
        return execute()
    parser.error("choose --plan-only or --execute")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
