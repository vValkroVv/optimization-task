"""
Collect PyTorch Profiler traces from a running sglang server.

Workflow:
  1. POST /start_profile  (with start_step for warmup)
  2. Run bench_serving to generate load (no --profile flag)
  3. POST /stop_profile   (blocks until traces are flushed)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import requests


# Previous ad-hoc defaults are kept for standalone runs. The end-to-end
# benchmark wrapper passes explicit CLI args.
SERVER_PORT = 30000
SERVER_IP = "..."
OUTPUT_FOLDER = (
    "/home/jovyan/nvokhotnikov/llm-foundry/sber_benchmarks/sglang_benchmark_results/..."
)
INPUT_LEN = 10240
OUTPUT_LEN = 1024
NUM_PROMPTS = 32
MAX_CONCURRENCY = 0
RPS = 8
SAVE_DETAILS = True
WARMUP_STEPS = 5
TOKENIZER_PATH = ""


_SGLANG_FORK = (
    Path(__file__).resolve().parent.parent
    / "contrib"
    / "pretrain_auto_metrics"
    / "contrib"
    / "sglang_pretrain"
    / "python"
)
assert _SGLANG_FORK.is_dir(), f"SGLang fork not found at {_SGLANG_FORK}"


def _env_default(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect SGLang profiler traces")
    parser.add_argument(
        "--server-ip", default=_env_default("TRACE_SERVER_IP", SERVER_IP)
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=int(_env_default("TRACE_SERVER_PORT", str(SERVER_PORT))),
    )
    parser.add_argument("--base-url", default=_env_default("TRACE_BASE_URL", ""))
    parser.add_argument(
        "--tokenizer-path", default=_env_default("TRACE_TOKENIZER_PATH", TOKENIZER_PATH)
    )
    parser.add_argument(
        "--output-folder",
        required=False,
        default=_env_default("TRACE_OUTPUT_FOLDER", OUTPUT_FOLDER),
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=int(_env_default("TRACE_INPUT_LEN", str(INPUT_LEN))),
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=int(_env_default("TRACE_OUTPUT_LEN", str(OUTPUT_LEN))),
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=int(_env_default("TRACE_NUM_PROMPTS", str(NUM_PROMPTS))),
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=int(_env_default("TRACE_MAX_CONCURRENCY", str(MAX_CONCURRENCY))),
    )
    parser.add_argument(
        "--rps", type=float, default=float(_env_default("TRACE_RPS", str(RPS)))
    )
    parser.add_argument(
        "--save-details",
        type=_str_to_bool,
        default=_str_to_bool(_env_default("TRACE_SAVE_DETAILS", str(SAVE_DETAILS))),
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=int(_env_default("TRACE_WARMUP_STEPS", str(WARMUP_STEPS))),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    base_url = args.base_url or f"http://{args.server_ip}:{args.server_port}"

    start_body = {
        "output_dir": str(output_folder),
        "activities": ["CPU", "GPU"],
    }
    if args.warmup_steps > 0:
        start_body["start_step"] = args.warmup_steps

    print(f"[1/3] POST {base_url}/start_profile  body={start_body}", flush=True)
    response = requests.post(f"{base_url}/start_profile", json=start_body, timeout=120)
    response.raise_for_status()
    print("      Profiling started.", flush=True)

    cmd = [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--base-url",
        base_url,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(args.input_len),
        "--random-output-len",
        str(args.output_len),
        "--num-prompts",
        str(args.num_prompts),
    ]
    if args.tokenizer_path:
        cmd += ["--tokenizer", args.tokenizer_path]
    if args.max_concurrency > 0:
        cmd += ["--max-concurrency", str(args.max_concurrency)]
    if args.rps > 0:
        cmd += ["--request-rate", str(args.rps)]
    if args.save_details:
        cmd += ["--output-details"]
        cmd += ["--output-file", str(output_folder / "details.json")]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(_SGLANG_FORK) + os.pathsep + env.get("PYTHONPATH", "")
    env["SGLANG_TORCH_PROFILER_DIR"] = str(output_folder)

    print(f"[2/3] Running bench_serving: {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, env=env, check=True)
    finally:
        print(
            "[3/3] POST {}/stop_profile  (waiting for trace flush)".format(base_url),
            flush=True,
        )
        response = requests.post(f"{base_url}/stop_profile", json={}, timeout=600)
        response.raise_for_status()
        print("      Profiling stopped. Traces saved to:", output_folder, flush=True)


if __name__ == "__main__":
    main()
