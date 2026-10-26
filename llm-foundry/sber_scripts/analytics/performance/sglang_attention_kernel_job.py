#!/usr/bin/env python3
"""Cluster launcher for SGLang attention kernel benchmarks.

The local process only submits and optionally monitors an InfiniteJob. The
benchmark commands themselves run inside the job image with the Python executable
configured by run_sglang_attention_kernel_benchmark.sh.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROXY_ENV_VARS = (
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "ALL_PROXY",
    "https_proxy",
    "http_proxy",
    "all_proxy",
)
SUCCESS_JOB_STATUSES = {"Completed", "Succeeded", "Success"}


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


def benchmark_logs_exist(state: dict[str, Any]) -> bool:
    return all(Path(run["log_path"]).exists() for run in state["runs"])


def should_retry_monitor(
    state: dict[str, Any], raw_status: str | None, logs_text: str | None
) -> bool:
    status = normalize_job_status(raw_status)
    logs = logs_text or ""
    if benchmark_logs_exist(state):
        return False
    if status.startswith("Cant get status"):
        return True
    return "in queue. Try later" in logs


def json_dump(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def remove_proxy_env() -> None:
    for key in PROXY_ENV_VARS:
        os.environ.pop(key, None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--user-login", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--n-gpus", type=int, required=True)
    parser.add_argument("--n-workers", type=int, required=True)
    parser.add_argument("--queue-name", required=True)
    parser.add_argument("--priority-class", required=True)
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--monitor-poll-seconds", type=int, default=30)
    parser.add_argument("--benchmark-label", action="append", required=True)
    parser.add_argument("--benchmark-log", action="append", required=True)
    parser.add_argument("--benchmark-command", action="append", required=True)
    return parser


def build_state(args: argparse.Namespace) -> dict[str, Any]:
    labels = args.benchmark_label
    logs = args.benchmark_log
    commands = args.benchmark_command
    if not (len(labels) == len(logs) == len(commands)):
        raise ValueError(
            "--benchmark-label, --benchmark-log and --benchmark-command must be repeated equally"
        )

    repo_root = Path(args.repo_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    if str(repo_root).startswith("/home/jovyan/"):
        raise ValueError(
            f"repo_root must be a cluster-visible physical path, got {repo_root}. "
            "Run through the entrypoint so symlinks are resolved to /workspace-SR008.nfs2/..."
        )
    benchmark_py = (
        repo_root / "sber_benchmarks" / "sglang_attention_kernel_benchmark.py"
    )
    if not benchmark_py.is_file():
        raise FileNotFoundError(benchmark_py)

    runs = []
    for label, log, command in zip(labels, logs, commands, strict=True):
        log_path = Path(log).resolve()
        runs.append(
            {
                "label": label,
                "log_path": str(log_path),
                "command": command,
            }
        )

    return {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "repo_root": str(repo_root),
        "user_login": args.user_login,
        "exp_name": args.exp_name,
        "n_gpus": args.n_gpus,
        "n_workers": args.n_workers,
        "queue_name": args.queue_name,
        "priority_class": args.priority_class,
        "base_image": args.base_image,
        "monitor": bool(args.monitor),
        "monitor_poll_seconds": args.monitor_poll_seconds,
        "runs": runs,
        "state_json": str(run_dir / "state.json"),
        "job_script": str(run_dir / "attention_kernel_job.sh"),
        "monitor_log": str(run_dir / "job_monitor.log"),
        "job_result_json": str(run_dir / "job_result.json"),
    }


def validate_state(state: dict[str, Any]) -> None:
    if state["n_gpus"] <= 0:
        raise ValueError(f"n_gpus must be positive, got {state['n_gpus']}")
    if state["n_workers"] <= 0:
        raise ValueError(f"n_workers must be positive, got {state['n_workers']}")
    if not state["runs"]:
        raise ValueError("at least one benchmark command is required")


def write_job_script(state: dict[str, Any]) -> Path:
    run_dir = Path(state["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(state["job_script"])

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(state['repo_root'])}",
        f"mkdir -p {shlex.quote(state['run_dir'])}",
        "echo '===== attention benchmark job ====='",
        f"echo 'run_id: {shlex.quote(state['run_id'])}'",
        f"echo 'repo_root: {shlex.quote(state['repo_root'])}'",
        f"echo 'run_dir: {shlex.quote(state['run_dir'])}'",
        "overall_status=0",
    ]

    for index, run in enumerate(state["runs"], start=1):
        label = run["label"]
        log_path = run["log_path"]
        command = run["command"]
        lines.extend(
            [
                f"echo '===== [{index}/{len(state['runs'])}] {shlex.quote(label)} ====='",
                f"mkdir -p {shlex.quote(str(Path(log_path).parent))}",
                "set +e",
                f"{command} 2>&1 | tee {shlex.quote(log_path)}",
                "status=${PIPESTATUS[0]}",
                "set -e",
                'if [ "${status}" -ne 0 ]; then',
                f"  echo 'Benchmark {shlex.quote(label)} failed with exit code '${{status}}",
                "  overall_status=1",
                "fi",
            ]
        )
    lines.extend(
        [
            'if [ "${overall_status}" -ne 0 ]; then',
            "  echo '===== attention benchmark job completed with failed benchmark command(s); exiting 0 by design ====='",
            "else",
            "  echo '===== attention benchmark job completed ====='",
            "fi",
        ]
    )

    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def submit_job(state: dict[str, Any], script_path: Path) -> Any:
    repo_root = Path(state["repo_root"])
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "contrib" / "job_submitter"))
    from contrib.job_submitter.job_submitter.jobs.infinite import InfiniteJob

    remove_proxy_env()
    job = InfiniteJob(
        user_login=state["user_login"],
        exp_name=state["exp_name"],
        script=f"bash {shlex.quote(str(script_path))}",
        n_gpus=state["n_gpus"],
        n_workers=state["n_workers"],
        queue_name=state["queue_name"],
        priority_class=state["priority_class"],
        base_image=state["base_image"],
    )
    submitted = job.submit()
    print(submitted)

    handle = getattr(job, "_last_submitted_job", None)
    job_name = getattr(handle, "job_name", None) if handle is not None else None
    if job_name:
        print(f"Attention benchmark job: {job_name}", flush=True)

    final_status = None
    logs_text = None
    monitor_chunks: list[str] = []
    if state.get("monitor"):
        while True:
            result = job.monitor(
                initial_wait_seconds=min(15, state["monitor_poll_seconds"]),
                poll_interval_seconds=state["monitor_poll_seconds"],
                logs_tail=0,
                logs_verbose=True,
                capture_logs=True,
            )
            final_status = result.final_status
            logs_text = result.logs
            print(f"attention benchmark job final status: {final_status}", flush=True)
            if logs_text:
                monitor_chunks.append(logs_text)
                Path(state["monitor_log"]).write_text(
                    "\n".join(monitor_chunks), encoding="utf-8"
                )
            if should_retry_monitor(state, final_status, logs_text):
                print(
                    "attention benchmark job is still queued or status is temporarily unavailable; retrying monitor",
                    flush=True,
                )
                time.sleep(state["monitor_poll_seconds"])
                continue
            if normalize_job_status(final_status).startswith(
                "Cant get status"
            ) and benchmark_logs_exist(state):
                print(
                    "attention benchmark job status endpoint is unavailable after benchmark logs appeared; keeping artifacts for validation",
                    flush=True,
                )
            else:
                ensure_job_completed("attention benchmark job", final_status)
            break

    json_dump(
        Path(state["job_result_json"]),
        {
            "job_name": job_name,
            "final_status": final_status,
            "monitor_log": state["monitor_log"] if logs_text else None,
        },
    )
    return submitted


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    state = build_state(args)
    validate_state(state)
    run_dir = Path(state["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = Path(state["state_json"])
    json_dump(state_path, state)
    script_path = write_job_script(state)
    print(f"State written to: {state_path}")
    print(f"Job script written to: {script_path}")
    print(f"Run dir: {run_dir}")
    submit_job(state, script_path)


if __name__ == "__main__":
    main()
