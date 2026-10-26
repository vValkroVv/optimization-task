#!/usr/bin/env python3
"""End-to-end model inference performance benchmark launcher.

Default mode submits an inf_bench InfiniteJob that runs the whole pipeline on cluster:
random HF checkpoint init, optional FP8 quantization, custom-SGLang inference job,
aiperf benchmark, optional profiler traces, and inference job cleanup.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import urllib.error
import urllib.request


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_IMAGE = (
    "cr.ai.cloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/"
    "pretrain-auto-metrics_opensource:2026_06_12_commit8f7a8687"
)
PROXY_ENV_VARS = (
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "ALL_PROXY",
    "https_proxy",
    "http_proxy",
    "all_proxy",
)


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value!r}")


def optional_int(value: str | None) -> int | None:
    if value is None or value == "" or value.lower() == "none":
        return None
    return int(value)


def abs_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def json_dump(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


class CheckpointValidationError(RuntimeError):
    pass


SUCCESS_JOB_STATUSES = {"Completed", "Succeeded", "Success"}
ACTIVE_JOB_STATUSES = {"Pending", "Running"}
FAILED_JOB_STATUSES = {"Failed", "Cancelled", "Canceled", "Error"}


def path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def safe_rmtree(path: Path, allowed_root: Path, *, allow_outside: bool = False) -> None:
    path = path.resolve()
    allowed_root = allowed_root.resolve()
    if not path.exists():
        return
    if path == allowed_root:
        raise ValueError(f"Refusing to remove artifacts root: {path}")
    inside_allowed_root = path_is_relative_to(path, allowed_root)
    if not inside_allowed_root and not allow_outside:
        raise ValueError(
            "Unsafe checkpoint removal requested for "
            f"{path}; it is outside artifacts dir {allowed_root}. "
            "If you know exactly what you are doing and want to delete existing "
            f"checkpoint(s) in {path}, set FORCE_REMOVE_OUTSIDE_ARTIFACTS=1."
        )
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        raise CheckpointValidationError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CheckpointValidationError(f"JSON file {path} must contain an object")
    return data


def _sample(items: list[str], limit: int = 10) -> str:
    suffix = "" if len(items) <= limit else f" ... (+{len(items) - limit} more)"
    return ", ".join(items[:limit]) + suffix


def validate_safetensors_checkpoint(path: Path) -> int:
    try:
        from safetensors import safe_open
    except Exception as exc:
        raise CheckpointValidationError(
            f"safetensors is required for checkpoint validation: {exc}"
        ) from exc

    index_path = path / "model.safetensors.index.json"
    safetensors_files = sorted(path.glob("*.safetensors"))
    if not safetensors_files:
        if any(path.glob("pytorch_model*.bin")):
            raise CheckpointValidationError(
                "strict validation supports safetensors checkpoints only; found pytorch_model*.bin"
            )
        raise CheckpointValidationError("no .safetensors weight files found")

    if not index_path.is_file():
        if len(safetensors_files) != 1:
            raise CheckpointValidationError(
                "multiple .safetensors files require model.safetensors.index.json"
            )
        shard_path = safetensors_files[0]
        try:
            with safe_open(shard_path, framework="pt", device="cpu") as shard:
                keys = list(shard.keys())
        except Exception as exc:
            raise CheckpointValidationError(
                f"cannot read safetensors shard {shard_path}: {exc}"
            ) from exc
        if not keys:
            raise CheckpointValidationError(
                f"safetensors shard has no tensors: {shard_path}"
            )
        return len(keys)

    index = _read_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise CheckpointValidationError(
            f"{index_path} must contain a non-empty weight_map"
        )

    bad_shard_values = sorted(
        {repr(v) for v in weight_map.values() if not isinstance(v, str) or not v}
    )
    if bad_shard_values:
        raise CheckpointValidationError(
            f"weight_map contains invalid shard names: {_sample(bad_shard_values)}"
        )

    expected_keys = set(weight_map)
    shard_names = sorted(set(weight_map.values()))
    missing_shards = [name for name in shard_names if not (path / name).is_file()]
    if missing_shards:
        raise CheckpointValidationError(
            f"missing shard file(s): {_sample(missing_shards)}"
        )

    actual_key_to_shard: dict[str, str] = {}
    duplicate_keys: list[str] = []
    for shard_name in shard_names:
        shard_path = path / shard_name
        try:
            with safe_open(shard_path, framework="pt", device="cpu") as shard:
                keys = list(shard.keys())
        except Exception as exc:
            raise CheckpointValidationError(
                f"cannot read safetensors shard {shard_path}: {exc}"
            ) from exc
        if not keys:
            raise CheckpointValidationError(
                f"safetensors shard has no tensors: {shard_path}"
            )
        for key in keys:
            if key in actual_key_to_shard:
                duplicate_keys.append(key)
            actual_key_to_shard[key] = shard_name

    if duplicate_keys:
        duplicate_keys = sorted(set(duplicate_keys))
        raise CheckpointValidationError(
            f"duplicate tensor keys across shards: {_sample(duplicate_keys)}"
        )

    actual_keys = set(actual_key_to_shard)
    missing_keys = sorted(expected_keys - actual_keys)
    if missing_keys:
        raise CheckpointValidationError(
            f"weight_map keys missing from shards: {_sample(missing_keys)}"
        )

    extra_keys = sorted(actual_keys - expected_keys)
    if extra_keys:
        raise CheckpointValidationError(
            f"shard keys missing from weight_map: {_sample(extra_keys)}"
        )

    wrong_shards = sorted(
        key
        for key, expected_shard in weight_map.items()
        if actual_key_to_shard.get(key) != expected_shard
    )
    if wrong_shards:
        raise CheckpointValidationError(
            f"weight_map points to wrong shard for: {_sample(wrong_shards)}"
        )

    return len(actual_keys)


def validate_checkpoint(path: Path, *, require_fp8: bool = False) -> int:
    if not path.is_dir():
        raise CheckpointValidationError(f"checkpoint path is not a directory: {path}")
    config_path = path / "config.json"
    if not config_path.is_file():
        raise CheckpointValidationError(f"missing config.json: {config_path}")
    config = _read_json(config_path)
    if require_fp8:
        qcfg = config.get("quantization_config") or {}
        if not isinstance(qcfg, dict) or qcfg.get("quant_method") != "fp8":
            raise CheckpointValidationError(
                "config.json does not describe an FP8 checkpoint"
            )
    return validate_safetensors_checkpoint(path)


def maybe_use_existing_checkpoint(
    path: Path,
    *,
    artifacts_dir: Path,
    checkpoint_strict: bool,
    force_remove_outside_artifacts: bool,
    require_fp8: bool,
    label: str,
) -> bool:
    if not path.exists():
        return False
    try:
        tensor_count = validate_checkpoint(path, require_fp8=require_fp8)
    except CheckpointValidationError as exc:
        message = f"invalid checkpoint {path}: {exc}"
        if checkpoint_strict:
            raise RuntimeError(message) from exc
        print(f"WARNING: {message}. Recreating it from scratch.", flush=True)
        safe_rmtree(
            path,
            artifacts_dir,
            allow_outside=force_remove_outside_artifacts,
        )
        return False
    print(
        f"{label} checkpoint already exists and passed validation "
        f"({tensor_count} tensors), skipping: {path}",
        flush=True,
    )
    return True


def normalize_job_status(raw_status: str | None) -> str:
    if not raw_status:
        return ""
    prefix = "Job status="
    return raw_status[len(prefix) :] if raw_status.startswith(prefix) else raw_status


def ensure_job_completed(job_label: str, raw_status: str) -> None:
    status = normalize_job_status(raw_status)
    if status not in SUCCESS_JOB_STATUSES:
        raise RuntimeError(
            f"{job_label} finished with non-success status: {raw_status}"
        )


def run_command(
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    timeout_seconds: int | None = None,
) -> None:
    print("\n===== RUN =====", flush=True)
    print(" ".join(shlex.quote(part) for part in argv), flush=True)
    if timeout_seconds is not None and timeout_seconds > 0:
        print(f"Command timeout: {timeout_seconds}s", flush=True)
    try:
        subprocess.run(
            argv,
            cwd=str(cwd or REPO_ROOT),
            env=env,
            check=True,
            timeout=timeout_seconds
            if timeout_seconds and timeout_seconds > 0
            else None,
        )
    except subprocess.TimeoutExpired as exc:
        cmd = " ".join(shlex.quote(part) for part in argv)
        raise TimeoutError(
            f"Command timed out after {timeout_seconds}s: {cmd}"
        ) from exc


def remove_proxy_env() -> None:
    for key in PROXY_ENV_VARS:
        os.environ.pop(key, None)


def inference_heartbeat_enabled(state: dict[str, Any]) -> bool:
    return (
        not state.get("keep_inference_job")
        and int(state.get("inference_heartbeat_stale_seconds") or 0) > 0
    )


class InferenceHeartbeat:
    def __init__(self, path: Path, interval_seconds: int) -> None:
        self.path = path
        self.interval_seconds = max(1, int(interval_seconds))
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._job_name = ""

    def start(self, job_name: str) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._job_name = job_name
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._write("running")
            self._thread = threading.Thread(
                target=self._run,
                name="model-perf-inference-heartbeat",
                daemon=True,
            )
            self._thread.start()
        print(f"Inference heartbeat started: {self.path}", flush=True)

    def stop(self, status: str = "stopped") -> None:
        with self._lock:
            thread = self._thread
            if thread is None:
                return
            self._stop.set()
        try:
            self._write(status)
        except Exception as exc:
            print(
                f"WARNING: failed to write final inference heartbeat: {exc}", flush=True
            )
        thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self._write("running")
            except Exception as exc:
                print(
                    f"WARNING: failed to update inference heartbeat: {exc}", flush=True
                )

    def _write(self, status: str) -> None:
        payload = {
            "job_name": self._job_name,
            "pid": os.getpid(),
            "status": status,
            "time": time.time(),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        tmp_path = self.path.with_name(self.path.name + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        tmp_path.replace(self.path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-in-job", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--state-json", help="Internal state JSON path for --run-in-job"
    )

    parser.add_argument(
        "--config", help="llm-foundry config used to build the random HF checkpoint"
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--user-login", default="nvokhotnikov")
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument(
        "--venv", default="", help="Optional venv to source inside the inf_bench job"
    )
    parser.add_argument("--job-python", default="/home/user/sglang/bin/python")

    parser.add_argument("--queue-name", default="pretrain-ml")
    parser.add_argument("--priority-class", default="medium")
    parser.add_argument("--prep-n-gpus", type=int, default=4)
    parser.add_argument("--prep-n-workers", type=int, default=1)
    parser.add_argument("--base-image", default=DEFAULT_BASE_IMAGE)
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Monitor the inf_bench job until completion",
    )
    parser.add_argument("--monitor-poll-seconds", type=int, default=30)

    parser.add_argument("--hf-checkpoint-dir", default=None)
    parser.add_argument("--force-init", action="store_true")
    parser.add_argument("--checkpoint-strict", type=str_to_bool, default=True)
    parser.add_argument("--force-remove-outside-artifacts", action="store_true")
    parser.add_argument("--include-mtp", type=str_to_bool, default=True)
    parser.add_argument("--use-custom-sglang", type=str_to_bool, default=True)
    parser.add_argument(
        "--model-type", default="gigachat35", choices=("gigachat35", "deepseek_v3")
    )
    parser.add_argument("--model-tag", default=None)
    parser.add_argument("--init-device", default="cuda")
    parser.add_argument("--init-cuda-visible-devices", default="0")
    parser.add_argument("--uniform-moe", type=str_to_bool, default=False)
    parser.add_argument("--expand-context-len-to", type=optional_int, default=262144)

    parser.add_argument("--fp8-checkpoint-dir", default=None)
    parser.add_argument("--quantize-fp8", type=str_to_bool, default=True)
    parser.add_argument("--force-quant", action="store_true")
    parser.add_argument("--quant-device", default="cuda:0")
    parser.add_argument("--weight-block-size", type=int, nargs=2, default=(128, 128))
    parser.add_argument(
        "--activation-scheme", default="dynamic", choices=("dynamic", "static")
    )
    parser.add_argument("--modules-to-not-convert", nargs="*", default=None)

    parser.add_argument("--inference-n-gpus", type=int, default=4)
    parser.add_argument("--inference-n-workers", type=int, default=1)
    parser.add_argument("--inference-queue-name", default=None)
    parser.add_argument("--inference-priority-class", default=None)
    parser.add_argument("--inference-base-image", default=None)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--ep", type=int, default=4)
    parser.add_argument("--dp", type=optional_int, default=None)
    parser.add_argument("--pp", type=optional_int, default=None)
    parser.add_argument("--router-port", type=int, default=30000)
    parser.add_argument("--sglang-communication-port", type=int, default=50000)
    parser.add_argument("--max-seq-len", type=int, default=262144)
    parser.add_argument(
        "--gated-norm-impl",
        default="torch_compile",
        choices=("native", "torch_compile", "triton"),
    )
    parser.add_argument("--moe-runner-backend", default="triton_llmf")
    parser.add_argument("--mem-fraction-static", type=float, default=0.8)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--copy-tokenizer-to-artifacts", type=str_to_bool, default=True)
    parser.add_argument("--server-timeout-minutes", type=int, default=300)
    parser.add_argument("--server-poll-seconds", type=int, default=15)
    parser.add_argument("--inference-heartbeat-interval-seconds", type=int, default=30)
    parser.add_argument("--inference-heartbeat-stale-seconds", type=int, default=900)
    parser.add_argument("--inference-heartbeat-poll-seconds", type=int, default=30)
    parser.add_argument("--keep-inference-job", action="store_true")

    parser.add_argument("--skip-aiperf", action="store_true")
    parser.add_argument("--aiperf-model", default="default")
    parser.add_argument("--aiperf-endpoint-type", default="completions")
    parser.add_argument("--aiperf-streaming", type=str_to_bool, default=True)
    parser.add_argument("--aiperf-input-len", type=int, default=1024)
    parser.add_argument("--aiperf-output-len", type=int, default=8192)
    parser.add_argument("--aiperf-input-stddev", type=int, default=0)
    parser.add_argument("--aiperf-concurrency", type=int, default=64)
    parser.add_argument("--aiperf-requests", type=int, default=128)
    parser.add_argument("--aiperf-request-timeout-seconds", type=int, default=3600)
    parser.add_argument("--aiperf-timeout-seconds", type=int, default=4500)
    parser.add_argument("--aiperf-output-dir", default=None)
    parser.add_argument("--aiperf-extra-inputs", default=None)
    parser.add_argument("--aiperf-bin", default=None)
    parser.add_argument("--aiperf-python", default=None)
    parser.add_argument("--aiperf-venv", default=None)

    parser.add_argument("--collect-traces", action="store_true")
    parser.add_argument("--trace-input-len", type=int, default=10240)
    parser.add_argument("--trace-output-len", type=int, default=1024)
    parser.add_argument("--trace-num-prompts", type=int, default=32)
    parser.add_argument("--trace-max-concurrency", type=int, default=0)
    parser.add_argument("--trace-rps", type=float, default=8)
    parser.add_argument("--trace-save-details", type=str_to_bool, default=True)
    parser.add_argument("--trace-warmup-steps", type=int, default=5)
    parser.add_argument("--trace-timeout-seconds", type=int, default=3600)
    parser.add_argument("--trace-output-dir", default=None)
    return parser


def infer_tokenizer_path(config_path: Path) -> str | None:
    model_path = None
    tokenizer_name = None
    for raw_line in config_path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line.startswith("model_path:"):
            model_path = line.split(":", 1)[1].strip().strip("\"'")
        elif line.startswith("name:") and tokenizer_name is None:
            # This is intentionally simple: in the 700B configs the tokenizer
            # section uses `name: ${model_path}` and appears after model_path.
            value = line.split(":", 1)[1].strip().strip("\"'")
            if value == "${model_path}" and model_path:
                tokenizer_name = model_path
    return tokenizer_name or model_path


def resolve_tokenizer_path(config_path: Path, explicit_path: str | None) -> str:
    tokenizer_path = explicit_path or infer_tokenizer_path(config_path)
    if not tokenizer_path:
        raise ValueError(
            "Tokenizer path is required for this benchmark; set TOKENIZER_PATH in "
            "run_model_perf_benchmark.sh or provide an existing tokenizer/model path in the config."
        )

    resolved = abs_path(tokenizer_path)
    if resolved.exists():
        return str(resolved)

    source = "explicit" if explicit_path else "inferred from config"
    raise FileNotFoundError(f"Tokenizer path ({source}) does not exist: {resolved}")


def materialize_tokenizer_path(
    tokenizer_path: str | None, artifacts_dir: Path, copy_to_artifacts: bool
) -> str | None:
    if not tokenizer_path or not copy_to_artifacts:
        return tokenizer_path

    src = Path(tokenizer_path)
    if not src.is_dir():
        return tokenizer_path

    dst = artifacts_dir / "tokenizer"
    if src.resolve() == dst.resolve():
        return str(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return str(dst)


def build_state(args: argparse.Namespace) -> dict[str, Any]:
    if not args.config:
        raise SystemExit("--config is required")
    config_path = abs_path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    artifacts_dir = (
        abs_path(args.artifacts_dir)
        if args.artifacts_dir
        else config_path.parent / "artifacts"
    )
    run_id = args.run_id or datetime.now().strftime("model_perf_%Y%m%d_%H%M%S")
    tokenizer_path = resolve_tokenizer_path(config_path, args.tokenizer_path)
    tokenizer_path = materialize_tokenizer_path(
        tokenizer_path, artifacts_dir, args.copy_tokenizer_to_artifacts
    )
    if not tokenizer_path or not Path(tokenizer_path).exists():
        raise FileNotFoundError(
            f"Tokenizer path is required and must exist: {tokenizer_path}"
        )
    run_dir = artifacts_dir / "model_perf" / run_id
    hf_checkpoint_dir = (
        abs_path(args.hf_checkpoint_dir)
        if args.hf_checkpoint_dir
        else artifacts_dir / "random_hf_checkpoint_uniform"
    )
    fp8_checkpoint_dir = (
        abs_path(args.fp8_checkpoint_dir)
        if args.fp8_checkpoint_dir
        else artifacts_dir / "random_hf_checkpoint_uniform_fp8"
    )
    aiperf_output_dir = (
        abs_path(args.aiperf_output_dir)
        if args.aiperf_output_dir
        else run_dir / "aiperf"
    )
    trace_output_dir = (
        abs_path(args.trace_output_dir) if args.trace_output_dir else run_dir / "traces"
    )
    job_python = args.job_python

    state = vars(args).copy()
    state.update(
        {
            "repo_root": str(REPO_ROOT),
            "config": str(config_path),
            "artifacts_dir": str(artifacts_dir),
            "run_id": run_id,
            "run_dir": str(run_dir),
            "hf_checkpoint_dir": str(hf_checkpoint_dir),
            "fp8_checkpoint_dir": str(fp8_checkpoint_dir),
            "aiperf_output_dir": str(aiperf_output_dir),
            "trace_output_dir": str(trace_output_dir),
            "pam_config_path": str(run_dir / "inference_pam_config.yaml"),
            "tokenizer_path": tokenizer_path,
            "status_json": str(run_dir / "status.json"),
            "inference_heartbeat_path": str(
                run_dir / "inference_parent_heartbeat.json"
            ),
            "job_python": job_python,
        }
    )
    state.pop("run_in_job", None)
    state.pop("state_json", None)
    return state


def validate_preflight(state: dict[str, Any]) -> None:
    errors: list[str] = []
    artifacts_dir = Path(state["artifacts_dir"])
    hf_dir = Path(state["hf_checkpoint_dir"])
    fp8_dir = Path(state["fp8_checkpoint_dir"])

    required_files = [
        ("config", Path(state["config"])),
        ("init helper", REPO_ROOT / "sber_benchmarks" / "run_init_random_hf_model.sh"),
        (
            "init implementation",
            REPO_ROOT / "sber_benchmarks" / "init_random_hf_model.py",
        ),
        ("aiperf helper", REPO_ROOT / "sber_benchmarks" / "aiperf_benchmark.sh"),
    ]
    if state.get("quantize_fp8", True):
        required_files.append(
            (
                "FP8 quantizer",
                REPO_ROOT
                / "contrib"
                / "gigachat-utils"
                / "convertation"
                / "quantization"
                / "fine_grained_fp8_quantization_layer_by_layer.py",
            )
        )
    if state.get("collect_traces"):
        required_files.append(
            ("trace helper", REPO_ROOT / "sber_benchmarks" / "get_inference_traces.py")
        )

    for label, path in required_files:
        if not path.is_file():
            errors.append(f"missing {label}: {path}")

    tokenizer_path = Path(state["tokenizer_path"])
    if not tokenizer_path.exists():
        errors.append(f"tokenizer path does not exist: {tokenizer_path}")

    if state.get("aiperf_bin") and not Path(state["aiperf_bin"]).is_file():
        errors.append(f"aiperf binary does not exist: {state['aiperf_bin']}")
    if (
        state.get("aiperf_venv")
        and not (Path(state["aiperf_venv"]) / "bin" / "activate").is_file()
    ):
        errors.append(
            f"aiperf venv activation script does not exist: {Path(state['aiperf_venv']) / 'bin' / 'activate'}"
        )

    if state.get("use_custom_sglang"):
        sglang_fork = (
            REPO_ROOT
            / "contrib"
            / "pretrain_auto_metrics"
            / "contrib"
            / "sglang_pretrain"
            / "python"
        )
        if not sglang_fork.is_dir():
            errors.append(f"custom SGLang fork is missing: {sglang_fork}")

    if not state.get("force_remove_outside_artifacts"):
        for flag_name, checkpoint_path in (
            ("FORCE_INIT", hf_dir),
            ("FORCE_QUANT", fp8_dir),
        ):
            force_enabled = (
                state.get("force_init")
                if flag_name == "FORCE_INIT"
                else state.get("force_quant")
            )
            if (
                force_enabled
                and checkpoint_path.exists()
                and not path_is_relative_to(checkpoint_path, artifacts_dir)
            ):
                errors.append(
                    f"{flag_name}=1 would remove checkpoint outside artifacts dir: {checkpoint_path}. "
                    "Set FORCE_REMOVE_OUTSIDE_ARTIFACTS=1 only if this deletion is intentional."
                )

    if errors:
        raise RuntimeError("Preflight failed:\n- " + "\n- ".join(errors))


def write_pam_config(state: dict[str, Any]) -> Path:
    pam_path = Path(state["pam_config_path"])
    inference: dict[str, Any] = {
        "tp": state["tp"],
        "ep": state["ep"],
        "trust_remote_code": True,
        "allow_auto_truncate": True,
        "mem_fraction_static": state["mem_fraction_static"],
        "gated_norm_impl": state["gated_norm_impl"],
        "moe_runner_backend": state["moe_runner_backend"],
    }
    if state.get("dp") is not None:
        inference["dp"] = state["dp"]
    if state.get("pp") is not None:
        inference["pp"] = state["pp"]
    if state.get("tokenizer_path"):
        inference["tokenizer_path"] = state["tokenizer_path"]

    cfg = {
        "run_name": state.get("exp_name") or state["run_id"],
        "exp_name": state.get("exp_name") or state["run_id"],
        "user": state["user_login"],
        "tensorboard_path": str(
            Path(state["artifacts_dir"]) / "tensorboard" / state["run_id"]
        ),
        "max_seq_len": state["max_seq_len"],
        "autometrics": {
            "globals": {
                "delete_converted_hf": False,
                "sglang_communication_port": state["sglang_communication_port"],
            },
            "router": {
                "host": "0.0.0.0",
                "port": state["router_port"],
            },
            "convertation": {
                "skip": True,
                "mappers": ["map_gigafsdp_to_hf_safetensor"],
                "input_tp_size": 1,
            },
            "preparation": {"skip": True},
            "inference": inference,
            "obm_proxy": {"tokenizer_path": state["tokenizer_path"]}
            if state.get("tokenizer_path")
            else {},
            "model_info": {
                "model_class": "sft",
                "size": "custom",
                "username": state["user_login"],
            },
            "evaluators": {},
        },
    }
    pam_path.parent.mkdir(parents=True, exist_ok=True)
    with pam_path.open("w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    return pam_path


def submit_inf_bench_job(state: dict[str, Any], state_path: Path) -> Any:
    sys.path.insert(0, str(REPO_ROOT))
    from contrib.job_submitter.job_submitter.jobs.infinite import InfiniteJob

    venv = Path(state["venv"]) if state.get("venv") else None
    python_bin = state["job_python"]
    cmd = [
        python_bin,
        str(REPO_ROOT / "sber_benchmarks" / "model_perf_benchmark.py"),
        "--run-in-job",
        "--state-json",
        str(state_path),
    ]
    script_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        "export PYTHONUNBUFFERED=1",
        "unset HTTPS_PROXY HTTP_PROXY ALL_PROXY https_proxy http_proxy all_proxy",
    ]
    if venv:
        activate = venv / "bin" / "activate"
        script_lines.append(
            f"if [ -f {shlex.quote(str(activate))} ]; then source {shlex.quote(str(activate))}; fi"
        )
    script_lines.append(" ".join(shlex.quote(part) for part in cmd))
    script_path = state_path.parent / "inf_bench_job.sh"
    script_path.write_text("\n".join(script_lines) + "\n")
    script_path.chmod(0o755)
    script = f"bash {script_path}"

    remove_proxy_env()
    job = InfiniteJob(
        user_login=state["user_login"],
        exp_name=state.get("exp_name") or f"model perf benchmark {state['run_id']}",
        script=script,
        n_gpus=state["prep_n_gpus"],
        n_workers=state["prep_n_workers"],
        queue_name=state["queue_name"],
        priority_class=state["priority_class"],
        base_image=state["base_image"],
        extra_env={"PYTHONUNBUFFERED": "1"},
    )
    submitted = job.submit()
    print(submitted)
    if state.get("monitor"):
        result = job.monitor(
            initial_wait_seconds=min(15, state["monitor_poll_seconds"]),
            poll_interval_seconds=state["monitor_poll_seconds"],
            logs_tail=0,
            logs_verbose=True,
            capture_logs=True,
        )
        print(f"inf_bench job final status: {result.final_status}")
        ensure_job_completed("inf_bench job", result.final_status)
    return submitted


def maybe_init_checkpoint(state: dict[str, Any]) -> None:
    hf_dir = Path(state["hf_checkpoint_dir"])
    artifacts_dir = Path(state["artifacts_dir"])
    if state.get("force_init"):
        safe_rmtree(
            hf_dir,
            artifacts_dir,
            allow_outside=state.get("force_remove_outside_artifacts", False),
        )
    if maybe_use_existing_checkpoint(
        hf_dir,
        artifacts_dir=artifacts_dir,
        checkpoint_strict=state.get("checkpoint_strict", True),
        force_remove_outside_artifacts=state.get(
            "force_remove_outside_artifacts", False
        ),
        require_fp8=False,
        label="HF",
    ):
        return

    env = os.environ.copy()
    env.update(
        {
            "CONFIG_PATH": state["config"],
            "OUTPUT_DIR": str(hf_dir),
            "USE_CUSTOM_SGLANG": "True" if state["use_custom_sglang"] else "False",
            "MODEL_TYPE": state["model_type"],
            "MODEL_TAG": state.get("model_tag") or "",
            "TOKENIZER_PATH": state.get("tokenizer_path") or "",
            "INIT_DEVICE": state["init_device"],
            "CUDA_VISIBLE_DEVICES": state["init_cuda_visible_devices"],
            "INCLUDE_MTP": "True" if state["include_mtp"] else "False",
            "UNIFORM_MOE": "True" if state["uniform_moe"] else "False",
            "EXPAND_CONTEXT_LEN_TO": ""
            if state.get("expand_context_len_to") is None
            else str(state["expand_context_len_to"]),
            "PYTHON": sys.executable,
        }
    )
    run_command(
        ["bash", str(REPO_ROOT / "sber_benchmarks" / "run_init_random_hf_model.sh")],
        env=env,
    )
    try:
        validate_checkpoint(hf_dir, require_fp8=False)
    except CheckpointValidationError as exc:
        raise RuntimeError(
            f"HF checkpoint was not created correctly: {hf_dir}: {exc}"
        ) from exc


def maybe_quantize_checkpoint(state: dict[str, Any]) -> None:
    fp8_dir = Path(state["fp8_checkpoint_dir"])
    artifacts_dir = Path(state["artifacts_dir"])
    if state.get("force_quant"):
        safe_rmtree(
            fp8_dir,
            artifacts_dir,
            allow_outside=state.get("force_remove_outside_artifacts", False),
        )
    if maybe_use_existing_checkpoint(
        fp8_dir,
        artifacts_dir=artifacts_dir,
        checkpoint_strict=state.get("checkpoint_strict", True),
        force_remove_outside_artifacts=state.get(
            "force_remove_outside_artifacts", False
        ),
        require_fp8=True,
        label="FP8",
    ):
        return

    script = (
        REPO_ROOT
        / "contrib"
        / "gigachat-utils"
        / "convertation"
        / "quantization"
        / "fine_grained_fp8_quantization_layer_by_layer.py"
    )
    if not script.is_file():
        raise FileNotFoundError(script)
    cmd = [
        sys.executable,
        str(script),
        "--input_checkpoint_path",
        state["hf_checkpoint_dir"],
        "--output_checkpoint_path",
        str(fp8_dir),
        "--weight_block_size",
        str(state["weight_block_size"][0]),
        str(state["weight_block_size"][1]),
        "--activation_scheme",
        state["activation_scheme"],
        "--device",
        state["quant_device"],
    ]
    if state.get("modules_to_not_convert"):
        cmd.extend(["--modules_to_not_convert", *state["modules_to_not_convert"]])
    run_command(cmd)
    try:
        validate_checkpoint(fp8_dir, require_fp8=True)
    except CheckpointValidationError as exc:
        raise RuntimeError(
            f"FP8 checkpoint was not created correctly: {fp8_dir}: {exc}"
        ) from exc


def inference_checkpoint_dir(state: dict[str, Any]) -> str:
    if state.get("quantize_fp8", True):
        return state["fp8_checkpoint_dir"]
    return state["hf_checkpoint_dir"]


def submit_inference_job(state: dict[str, Any]) -> tuple[Any, str, str]:
    sys.path.insert(0, str(REPO_ROOT))
    from contrib.job_submitter.job_submitter.jobs.llmf.autometrics import (
        AutometricsJobLLMF,
    )

    pam_config = write_pam_config(state)
    extra_env = {}
    if inference_heartbeat_enabled(state):
        extra_env.update(
            {
                "MODEL_PERF_PARENT_HEARTBEAT_PATH": state["inference_heartbeat_path"],
                "MODEL_PERF_PARENT_HEARTBEAT_STALE_SECONDS": str(
                    state["inference_heartbeat_stale_seconds"]
                ),
                "MODEL_PERF_PARENT_HEARTBEAT_POLL_SECONDS": str(
                    state["inference_heartbeat_poll_seconds"]
                ),
            }
        )
    remove_proxy_env()
    job = AutometricsJobLLMF(
        user_login=state["user_login"],
        exp_name=(state.get("exp_name") or state["run_id"]) + " inference",
        checkpoint_dir=inference_checkpoint_dir(state),
        path_to_config=str(pam_config),
        mode="inference",
        debug=True,
        sglang_tag="custom",
        n_gpus=state["inference_n_gpus"],
        n_workers=state["inference_n_workers"],
        queue_name=state.get("inference_queue_name") or state["queue_name"],
        priority_class=state.get("inference_priority_class") or state["priority_class"],
        base_image=state.get("inference_base_image"),
        extra_env=extra_env,
    )
    job.submit()
    handle = getattr(job, "_last_submitted_job", None)
    if handle is None or not getattr(handle, "job_name", None):
        raise RuntimeError("Inference job did not return a monitorable job handle")
    job_name = handle.job_name
    host = f"{job_name}-mpimaster-0.{job_name}"
    base_url = f"http://{host}:{state['router_port']}"
    print(f"Inference job: {job_name}", flush=True)
    print(f"Inference URL: {base_url}", flush=True)
    return handle, job_name, base_url


def _http_json(
    method: str, url: str, payload: dict[str, Any] | None = None, timeout: int = 10
) -> int | None:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def check_server_once(base_url: str) -> bool:
    if _http_json("GET", f"{base_url}/health", timeout=5) == 200:
        return True
    status = _http_json(
        "POST",
        f"{base_url}/generate",
        payload={"text": "Hello!", "max_tokens": 2, "temperature": 0.0},
        timeout=30,
    )
    return status == 200


def wait_for_server(
    base_url: str,
    timeout_minutes: int,
    poll_seconds: int,
    *,
    handle: Any | None = None,
    job_name: str = "",
    on_job_running: Callable[[], None] | None = None,
) -> None:
    readiness_deadline: float | None = None
    job_running_notified = False
    attempt = 0

    def start_readiness_timeout() -> None:
        nonlocal readiness_deadline
        if timeout_minutes > 0 and readiness_deadline is None:
            readiness_deadline = time.time() + timeout_minutes * 60

    def mark_job_running() -> None:
        nonlocal job_running_notified
        start_readiness_timeout()
        if not job_running_notified:
            job_running_notified = True
            if on_job_running is not None:
                on_job_running()

    status_method = getattr(handle, "status", None)
    if not callable(status_method):
        mark_job_running()

    while True:
        attempt += 1
        if check_server_once(base_url):
            mark_job_running()
            print(f"Inference server is ready: {base_url}", flush=True)
            return

        raw_status = ""
        status = ""
        if callable(status_method):
            raw_status = status_method()
            status = normalize_job_status(raw_status)
            if status in FAILED_JOB_STATUSES or status in SUCCESS_JOB_STATUSES:
                raise RuntimeError(
                    f"Inference job {job_name} stopped before the server became ready: {raw_status}"
                )
            if status == "Running":
                mark_job_running()

        if readiness_deadline is not None and time.time() >= readiness_deadline:
            raise TimeoutError(
                f"Inference server did not become ready in {timeout_minutes} minutes "
                f"after inference job started: {base_url}"
            )

        if timeout_minutes <= 0:
            timeout_msg = "without timeout"
        elif readiness_deadline is None:
            status_hint = f"; job status={status}" if status else ""
            timeout_msg = f"readiness timeout has not started{status_hint}"
        else:
            remaining_seconds = max(0, int(readiness_deadline - time.time()))
            timeout_msg = (
                f"timeout in {remaining_seconds // 60} min {remaining_seconds % 60} sec "
                "after inference job start"
            )
        print(
            f"Waiting for inference server ({attempt}, {timeout_msg}); next poll in {poll_seconds}s",
            flush=True,
        )
        time.sleep(poll_seconds)


def run_aiperf(state: dict[str, Any], base_url: str) -> None:
    if state.get("skip_aiperf"):
        print("Skipping aiperf benchmark", flush=True)
        return
    output_dir = Path(state["aiperf_output_dir"])
    extra_inputs = state.get("aiperf_extra_inputs")
    if not extra_inputs:
        output_len = state["aiperf_output_len"]
        extra_inputs = (
            f"max_tokens:{output_len};min_tokens:{output_len};ignore_eos:true"
        )
    env = os.environ.copy()
    env.update(
        {
            "AIPERF_URL": base_url,
            "AIPERF_MODEL": state["aiperf_model"],
            "AIPERF_TOKENIZER": state.get("tokenizer_path")
            or inference_checkpoint_dir(state),
            "AIPERF_ENDPOINT_TYPE": state["aiperf_endpoint_type"],
            "AIPERF_STREAMING": "True" if state["aiperf_streaming"] else "False",
            "AIPERF_INPUT_LEN": str(state["aiperf_input_len"]),
            "AIPERF_OUTPUT_LEN": str(state["aiperf_output_len"]),
            "AIPERF_INPUT_STDDEV": str(state["aiperf_input_stddev"]),
            "AIPERF_CONCURRENCY": str(state["aiperf_concurrency"]),
            "AIPERF_REQUESTS": str(state["aiperf_requests"]),
            "AIPERF_REQUEST_TIMEOUT_SECONDS": str(
                state["aiperf_request_timeout_seconds"]
            ),
            "AIPERF_OUTPUT_DIR": str(output_dir),
            "AIPERF_EXTRA_INPUTS": extra_inputs,
        }
    )
    if state.get("aiperf_bin"):
        env["AIPERF_BIN"] = state["aiperf_bin"]
    if state.get("aiperf_python"):
        env["AIPERF_PYTHON"] = state["aiperf_python"]
    if state.get("aiperf_venv"):
        env["AIPERF_VENV"] = state["aiperf_venv"]
    run_command(
        ["bash", str(REPO_ROOT / "sber_benchmarks" / "aiperf_benchmark.sh")],
        env=env,
        timeout_seconds=state.get("aiperf_timeout_seconds"),
    )


def run_traces(state: dict[str, Any], base_url: str) -> None:
    if not state.get("collect_traces"):
        print("Trace collection disabled", flush=True)
        return
    tokenizer_path = state.get("tokenizer_path")
    if not tokenizer_path or not Path(tokenizer_path).exists():
        raise FileNotFoundError(
            f"Trace collection requires an existing tokenizer path: {tokenizer_path}"
        )
    run_command(
        [
            sys.executable,
            str(REPO_ROOT / "sber_benchmarks" / "get_inference_traces.py"),
            "--base-url",
            base_url,
            "--tokenizer-path",
            tokenizer_path,
            "--output-folder",
            state["trace_output_dir"],
            "--input-len",
            str(state["trace_input_len"]),
            "--output-len",
            str(state["trace_output_len"]),
            "--num-prompts",
            str(state["trace_num_prompts"]),
            "--max-concurrency",
            str(state["trace_max_concurrency"]),
            "--rps",
            str(state["trace_rps"]),
            "--save-details",
            "True" if state["trace_save_details"] else "False",
            "--warmup-steps",
            str(state["trace_warmup_steps"]),
        ],
        timeout_seconds=state.get("trace_timeout_seconds"),
    )


def _kill_result_success(result: Any) -> bool:
    text = str(result).lower()
    return " deleted" in text or "deleted" in text or "not submitted" in text


def _wait_until_job_not_active(
    handle: Any, job_name: str, timeout_seconds: int = 120
) -> bool:
    status_method = getattr(handle, "status", None)
    if not callable(status_method):
        return True
    deadline = time.time() + timeout_seconds
    last_status = ""
    while time.time() < deadline:
        try:
            raw_status = status_method()
        except Exception as exc:
            print(
                f"WARNING: cannot read status for inference job {job_name} after kill: {exc}",
                flush=True,
            )
            return True
        last_status = raw_status
        status = normalize_job_status(raw_status)
        if status not in ACTIVE_JOB_STATUSES:
            print(f"Inference job {job_name} cleanup status: {raw_status}", flush=True)
            return True
        time.sleep(5)
    print(
        f"WARNING: inference job {job_name} is still active after kill request: {last_status}",
        flush=True,
    )
    return False


def kill_inference(handle: Any, job_name: str) -> None:
    if handle is None or not job_name:
        return

    from contrib.job_submitter.job_submitter.c_lib import client_lib

    errors: list[str] = []
    attempts: list[tuple[str, Callable[[], Any]]] = []
    kill_method = getattr(handle, "kill", None)
    if callable(kill_method):
        attempts.append(("handle.kill", kill_method))
    attempts.extend(
        [
            (
                "client_lib.kill(region=SR008)",
                lambda: client_lib.kill(job_name, region=client_lib.RegionMT.SR008),
            ),
            ("client_lib.kill(region=None)", lambda: client_lib.kill(job_name)),
        ]
    )

    for label, action in attempts:
        try:
            result = action()
            print(f"{label}: {result}", flush=True)
            if _kill_result_success(result):
                _wait_until_job_not_active(handle, job_name)
                return
            errors.append(f"{label}: {result}")
        except Exception as exc:
            errors.append(f"{label}: {exc!r}")

    if _wait_until_job_not_active(handle, job_name, timeout_seconds=5):
        return
    print(
        f"WARNING: failed to kill inference job {job_name}: {'; '.join(errors)}",
        flush=True,
    )


def update_phase_status(
    status: dict[str, Any],
    status_path: Path,
    phase: str,
    phase_status: str,
    **details: Any,
) -> None:
    phase_data = {
        "status": phase_status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    phase_data.update(details)
    status.setdefault("phases", {})[phase] = phase_data
    json_dump(status_path, status)


def run_phase(
    status: dict[str, Any],
    status_path: Path,
    phase: str,
    action: Callable[[], None],
    **details: Any,
) -> None:
    update_phase_status(status, status_path, phase, "running", **details)
    try:
        action()
    except Exception as exc:
        update_phase_status(
            status, status_path, phase, "failed", error=repr(exc), **details
        )
        raise
    update_phase_status(status, status_path, phase, "completed", **details)


def run_in_job(state_path: Path) -> None:
    state = json.loads(state_path.read_text())
    run_dir = Path(state["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = Path(state["status_json"])
    status: dict[str, Any] = {
        "run_id": state["run_id"],
        "status": "running",
        "phases": {},
    }
    json_dump(status_path, status)

    handle = None
    inference_job_name = ""
    heartbeat: InferenceHeartbeat | None = None
    try:
        run_phase(
            status,
            status_path,
            "hf_init",
            lambda: maybe_init_checkpoint(state),
            checkpoint_dir=state["hf_checkpoint_dir"],
        )
        status["hf_checkpoint_dir"] = state["hf_checkpoint_dir"]
        json_dump(status_path, status)

        if state.get("quantize_fp8", True):
            run_phase(
                status,
                status_path,
                "fp8_quantization",
                lambda: maybe_quantize_checkpoint(state),
                checkpoint_dir=state["fp8_checkpoint_dir"],
            )
            status["fp8_checkpoint_dir"] = state["fp8_checkpoint_dir"]
        else:
            print(
                "FP8 quantization disabled, using HF checkpoint for inference",
                flush=True,
            )
            update_phase_status(status, status_path, "fp8_quantization", "skipped")
        status["inference_checkpoint_dir"] = inference_checkpoint_dir(state)
        json_dump(status_path, status)

        handle, inference_job_name, base_url = submit_inference_job(state)
        if inference_heartbeat_enabled(state):
            heartbeat = InferenceHeartbeat(
                Path(state["inference_heartbeat_path"]),
                state["inference_heartbeat_interval_seconds"],
            )
        status.update({"inference_job_name": inference_job_name, "base_url": base_url})
        if heartbeat is not None:
            status["inference_heartbeat_path"] = state["inference_heartbeat_path"]
        json_dump(status_path, status)

        run_phase(
            status,
            status_path,
            "server_ready",
            lambda: wait_for_server(
                base_url,
                state["server_timeout_minutes"],
                state["server_poll_seconds"],
                handle=handle,
                job_name=inference_job_name,
                on_job_running=(
                    lambda: heartbeat.start(inference_job_name)
                    if heartbeat is not None
                    else None
                ),
            ),
            base_url=base_url,
        )

        if state.get("skip_aiperf"):
            update_phase_status(
                status,
                status_path,
                "aiperf",
                "skipped",
                output_dir=state["aiperf_output_dir"],
            )
        else:
            run_phase(
                status,
                status_path,
                "aiperf",
                lambda: run_aiperf(state, base_url),
                output_dir=state["aiperf_output_dir"],
            )

        if state.get("collect_traces"):
            run_phase(
                status,
                status_path,
                "traces",
                lambda: run_traces(state, base_url),
                output_dir=state["trace_output_dir"],
            )
        else:
            update_phase_status(
                status,
                status_path,
                "traces",
                "skipped",
                output_dir=state["trace_output_dir"],
            )

        status["status"] = "completed"
        json_dump(status_path, status)
    except Exception as exc:
        status["status"] = "failed"
        status["error"] = repr(exc)
        json_dump(status_path, status)
        raise
    finally:
        if heartbeat is not None:
            heartbeat.stop("parent_exiting")
        if handle is not None and not state.get("keep_inference_job"):
            kill_inference(handle, inference_job_name)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.run_in_job:
        if not args.state_json:
            raise SystemExit("--state-json is required with --run-in-job")
        run_in_job(abs_path(args.state_json))
        return

    state = build_state(args)
    validate_preflight(state)
    run_dir = Path(state["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.json"
    json_dump(state_path, state)
    print(f"State written to: {state_path}")
    print(f"Artifacts dir: {state['artifacts_dir']}")
    submit_inf_bench_job(state, state_path)


if __name__ == "__main__":
    main()
