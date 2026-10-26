from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

from .src.token_budget import MAX_TOTAL_OUTPUT_TOKENS_ENV


DEFAULT_CONFIG: dict[str, Any] = {
    "dataset": "/home/user/terminal-bench-2",
    "agent": "terminus-2",
    "model-name": "openai/pam-model",
    "agent-timeout": None,
    "n-concurrent": 4,
    "pass-at-k": 1,
    "api_key": "dummy",
    "max-tokens": 32768,
    "max-total-output-tokens": None,
    "env-api-base": None,
    "env-prefix": "terminal-bench",
    "env-version": "1.0.0",
    "ttl-ms": 64800000,
    "verify-ssl": False,
    "allow-internet-access": True,
    "use-pip-proxy": False,
    "timeout-rli": 180,
    "request-retries": 2,
    "request-retry-backoff": 2.0,
    "transfer-retries": 2,
    "transfer-retry-backoff": 2.0,
    "session-ready-timeout": 600.0,
    "session-ready-poll-interval": 1.0,
    "command-timeout": 600,
    "llm-timeout": 6400,
    "enable-container-recycle-detection": False,
    "quiet": True,
    "yes": True,
    # Whole-trial retries (Harbor TrialQueue._execute_trial_with_retries). A
    # retry constructs a brand-new RLIEnvironment + start() on a fresh container
    # and re-runs the agent from scratch. We retry ONLY on exception types that
    # are, by construction, raised exclusively from pure-infra code paths --
    # never from agent or model behavior:
    #   - RLIContainerResetError: silent RLI pod recycle mid-task, confirmed via
    #     an alive-marker probe (see environment.py). Genuine model failures
    #     (reward 0 on a healthy container) are never retried.
    #   - RLISessionUnavailableError: RLI's own POST /v1/sessions was
    #     unavailable (503/410/transport error) -- raised only from
    #     _create_session_with_retries, the very first thing start() does,
    #     i.e. always before the agent exists (environment.py).
    #   - EnvironmentStartTimeoutError: Harbor's own type, raised from exactly
    #     one place (harbor/trial/trial.py _start_agent_environment) wrapping
    #     the whole agent_environment.start() call -- also always pre-agent
    #     (agent_result/verifier_result are null whenever this fires).
    #   - AgentSetupTimeoutError: Harbor's own type, raised from exactly one
    #     place (harbor/trial/trial.py _setup_agent) wrapping
    #     agent.setup(environment=...) -- the bootstrap step that runs BEFORE
    #     the agent's first turn (e.g. terminus-2's tmux-install check). Same
    #     asyncio.TimeoutError-subclass / same "always pre-agent" guarantee as
    #     EnvironmentStartTimeoutError above (agent_result/verifier_result are
    #     null whenever this fires) -- observed in production as the RLI
    #     step()/exec() call underneath agent.setup() hanging in an HTTP
    #     response read until Harbor's outer wait_for cancels it. Omitting
    #     this type left every such stall permanently un-retried even though
    #     it is structurally identical to EnvironmentStartTimeoutError.
    #   - DownloadVerifierDirError / AddTestsDirError: Harbor's own types,
    #     raised only when our upload_dir/download_dir transfer itself failed
    #     (harbor/verifier/verifier.py) -- the true reward is indeterminate in
    #     that case (we literally cannot read it), not "reward 0", so retrying
    #     is the only sound option and can't mask a genuine failure.
    # The Harbor-default non-retryable types (AgentTimeoutError,
    # RewardFileNotFoundError sourced from a real empty test run,
    # VerifierTimeoutError, ...) remain excluded by omission (this is an
    # allowlist), so we never mask real model behavior.
    "max-retries": 1,
    "retry-include": [
        "HTTPStatusError",
        "ReadTimeout",
        "RuntimeError",
        "RLIClientError",
        "RLIContainerResetError",
        "RLISessionUnavailableError",
        "RLIExecProxyBlipError",
        "RLILongRunningCommandLostError",
        "EnvironmentStartTimeoutError",
        "AgentSetupTimeoutError",
        "DownloadVerifierDirError",
        "AddTestsDirError",
    ],
    "retry-exclude": [],
}

_RLI_INFRASTRUCTURE_EXCEPTION_TYPES = frozenset(DEFAULT_CONFIG["retry-include"])

_AGENT_IMPORT_PATHS = {
    "terminus-2": "terminal_bench.agents.terminus.terminus_agent:RLITerminus2Agent",
    "mini-swe-agent": "terminal_bench.agents.mini_swe_agent.mini_swe_agent:RLIMiniSweAgent",
}

_COMMON_AGENT_KWARGS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("model-api-base", "model_api_base"), "model_api_base"),
    (("api-key", "api_key"), "api_key"),
    (("max-tokens", "max_tokens"), "max_tokens"),
    (("max-input-tokens", "max_input_tokens"), "max_input_tokens"),
    (("temperature",), "temperature"),
    (("top-p", "top_p"), "top_p"),
    (("reasoning-effort", "reasoning_effort"), "reasoning_effort"),
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Terminal-Bench through Harbor + RLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--metrics-dir", required=True)
    run_parser.add_argument("--config-json", required=True)
    return parser


def _normalized_config(raw: str) -> dict[str, Any]:
    raw_payload = json.loads(raw)
    if "n-attempts" in raw_payload or "n_attempts" in raw_payload:
        raise ValueError("n-attempts is not supported; use pass-at-k")
    payload = dict(DEFAULT_CONFIG)
    payload.update(raw_payload)
    payload["pass-at-k"] = int(payload["pass-at-k"])
    if payload["pass-at-k"] < 1:
        raise ValueError("pass-at-k must be a positive integer")
    return payload


def _append_flag(cmd: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def _append_repeated(cmd: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        values = [value]
    else:
        values = list(value)
    for item in values:
        cmd.extend([flag, str(item)])


def _common_agent_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for aliases, target in _COMMON_AGENT_KWARGS:
        for alias in aliases:
            if config.get(alias) is not None:
                kwargs[target] = config[alias]
                break
    return kwargs


def _build_rli_config(config: dict[str, Any], metrics_dir: Path) -> dict[str, Any]:
    trace_path = config.get("rli-trace-path") or str(metrics_dir / "rli_trace.jsonl")
    command_timeout = config.get("command-timeout")
    task_id_override = config.get("task-id") or config.get("task-name")
    if not isinstance(task_id_override, str):
        task_id_override = None
    return {
        "api_base": config.get("env-api-base"),
        "env_prefix": config.get("env-prefix"),
        "env_version": config.get("env-version"),
        "env_name": config.get("env-name"),
        "task_id": task_id_override,
        "ttl_ms": config.get("ttl-ms"),
        "agent_timeout_sec": config.get("agent-timeout"),
        "seed": config.get("seed"),
        "verify_ssl": config.get("verify-ssl"),
        "allow_internet_access": config.get("allow-internet-access", True),
        "use_pip_proxy": config.get("use-pip-proxy"),
        "timeout": config.get("timeout-rli"),
        "request_retries": config.get("request-retries"),
        "request_retry_backoff": config.get("request-retry-backoff"),
        "transfer_retries": config.get("transfer-retries"),
        "transfer_retry_backoff": config.get("transfer-retry-backoff"),
        "upload_chunk_kib": config.get("upload-chunk-kib"),
        "session_ready_timeout": config.get("session-ready-timeout"),
        "session_ready_poll_interval": config.get("session-ready-poll-interval"),
        "enable_probe_warmup": config.get("enable-probe-warmup"),
        "probe_warmup_command": config.get("probe-warmup-command"),
        "create_session_jitter_max_sec": config.get("create-session-jitter-max-sec"),
        "probe_retry_backoff_sec": config.get("probe-retry-backoff-sec"),
        "enable_container_recycle_detection": config.get(
            "enable-container-recycle-detection"
        ),
        "command_timeout_ms": (
            int(command_timeout) * 1000 if command_timeout is not None else None
        ),
        "trace_path": trace_path,
    }


def _selected_agent_import_path(config: dict[str, Any]) -> str:
    if config.get("agent-import-path") or config.get("agent_import_path"):
        raise ValueError("terminal_bench.agent-import-path is not supported")
    agent = config.get("agent")
    try:
        return _AGENT_IMPORT_PATHS[str(agent)]
    except KeyError as exc:
        allowed = ", ".join(sorted(_AGENT_IMPORT_PATHS))
        raise ValueError(
            f"Unsupported terminal_bench agent {agent!r}; use one of: {allowed}"
        ) from exc


def _build_harbor_job_config(config: dict[str, Any]) -> dict[str, Any]:
    agent_kwargs = _common_agent_kwargs(config)
    agent_kwargs.update(config.get("agent-kwargs") or {})
    llm_timeout = config.get("llm-timeout")
    if llm_timeout is not None:
        llm_call_kwargs = dict(agent_kwargs.get("llm_call_kwargs") or {})
        llm_call_kwargs.setdefault("timeout", llm_timeout)
        agent_kwargs["llm_call_kwargs"] = llm_call_kwargs
    return {
        "agents": [
            {
                "import_path": _selected_agent_import_path(config),
                "model_name": config.get("model-name"),
                "override_timeout_sec": config.get("agent-timeout"),
                "kwargs": agent_kwargs,
            }
        ]
    }


def _build_harbor_cmd(config: dict[str, Any], metrics_dir: Path) -> list[str]:
    harbor_config_path = metrics_dir / "harbor_job_config.json"
    harbor_config_path.write_text(
        json.dumps(_build_harbor_job_config(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    harbor_bin = shutil.which("harbor")
    cmd = [harbor_bin] if harbor_bin else [sys.executable, "-m", "harbor.cli.main"]
    cmd.extend(["run", "--config", str(harbor_config_path)])
    dataset = config.get("dataset")
    if dataset is not None and (
        str(dataset).startswith("/") or Path(str(dataset)).exists()
    ):
        _append_flag(cmd, "--path", dataset)
    else:
        _append_flag(cmd, "--dataset", dataset)
    _append_flag(cmd, "--n-concurrent", config.get("n-concurrent"))
    _append_flag(cmd, "--n-attempts", config.get("pass-at-k"))
    _append_flag(cmd, "--job-name", config.get("job-name"))
    _append_flag(
        cmd,
        "--environment-build-timeout-multiplier",
        config.get("environment-build-timeout-multiplier"),
    )
    _append_flag(cmd, "--jobs-dir", metrics_dir / "harbor_jobs")
    _append_flag(
        cmd,
        "--environment-import-path",
        "terminal_bench.src.environment:RLIEnvironment",
    )
    if not config.get("disable-verification"):
        _append_flag(
            cmd,
            "--verifier-import-path",
            "terminal_bench.src.verifier:TerminalBenchLocalRLIVerifier",
        )
    cmd.extend(
        [
            "--environment-kwarg",
            f"rli_config_json={json.dumps(_build_rli_config(config, metrics_dir))}",
        ]
    )
    _append_repeated(cmd, "--include-task-name", config.get("task-name"))
    _append_repeated(cmd, "--exclude-task-name", config.get("exclude-task-name"))
    _append_flag(cmd, "--n-tasks", config.get("n-tasks"))
    _append_flag(cmd, "--max-retries", config.get("max-retries"))
    _append_repeated(cmd, "--retry-include", config.get("retry-include"))
    _append_repeated(cmd, "--retry-exclude", config.get("retry-exclude"))

    if config.get("yes", True):
        cmd.append("--yes")
    if config.get("quiet", True):
        cmd.append("--quiet")
    if config.get("disable-verification"):
        cmd.append("--disable-verification")

    return cmd


def _build_env(config: dict[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    max_total_output_tokens = config.get("max-total-output-tokens")
    if max_total_output_tokens is None:
        env.pop(MAX_TOTAL_OUTPUT_TOKENS_ENV, None)
    else:
        max_total_output_tokens = int(max_total_output_tokens)
        if max_total_output_tokens < 1:
            raise ValueError("max-total-output-tokens must be a positive integer")
        env[MAX_TOTAL_OUTPUT_TOKENS_ENV] = str(max_total_output_tokens)
    api_base = config.get("model-api-base")
    if api_base:
        env["OPENAI_API_BASE"] = str(api_base)
        env["OPENAI_BASE_URL"] = str(api_base)
    api_key = config.get("api_key") or config.get("api-key") or "dummy"
    env.setdefault("OPENAI_API_KEY", str(api_key))
    env.setdefault("MSWEA_API_KEY", str(api_key))
    env_api_base = config.get("env-api-base")
    env_host = urlparse(str(env_api_base)).hostname if env_api_base else None
    if env_host:
        for key in ("NO_PROXY", "no_proxy"):
            existing = [item for item in env.get(key, "").split(",") if item]
            if env_host not in existing:
                existing.append(env_host)
            env[key] = ",".join(existing)
    python_paths = [str(Path(__file__).resolve().parents[1])]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(python_paths)
    return env


def _load_latest_job_result(
    jobs_dir: Path,
) -> tuple[Path | None, dict[str, Any] | None]:
    run_result_candidates = sorted(
        jobs_dir.glob("*/result.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    task_result_candidates = sorted(
        jobs_dir.glob("**/result.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    candidates = run_result_candidates + [
        path for path in task_result_candidates if path not in run_result_candidates
    ]
    for candidate in candidates:
        try:
            return candidate, json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
    return None, None


def _as_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _reward_counts(eval_stats: dict[str, Any]) -> tuple[int, int]:
    reward_stats = eval_stats.get("reward_stats") or {}
    reward_bucket = reward_stats.get("reward")
    if reward_bucket is None and reward_stats:
        reward_bucket = next(iter(reward_stats.values()))
    if not isinstance(reward_bucket, dict):
        return 0, 0

    passed = 0
    total = 0
    for raw_reward, trial_names in reward_bucket.items():
        count = len(trial_names) if isinstance(trial_names, list) else 0
        total += count
        reward = _as_number(raw_reward)
        if reward is not None and reward > 0:
            passed += count
    return passed, total


def _summarize_job_result(result: dict[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {"pass_rate": 0.0, "passed": 0, "total": 0, "errors": 0}

    stats = result.get("stats") or {}
    evals = stats.get("evals") or {}
    total = 0
    passed = 0
    errors = int(stats.get("n_errored_trials") or stats.get("n_errors") or 0)

    for eval_stats in evals.values():
        if not isinstance(eval_stats, dict):
            continue
        reward_passed, reward_total = _reward_counts(eval_stats)
        for metric in eval_stats.get("metrics") or []:
            if not isinstance(metric, dict):
                continue
            for key in (
                "pass_rate",
                "success_rate",
                "accuracy",
                "resolved_rate",
                "mean",
            ):
                metric_value = _as_number(metric.get(key))
                if metric_value is not None:
                    if key == "mean":
                        n_trials = int(
                            result.get("n_total_trials")
                            or eval_stats.get("n_trials")
                            or 0
                        )
                    else:
                        n_trials = int(
                            eval_stats.get("n_trials")
                            or result.get("n_total_trials")
                            or 0
                        )
                    return {
                        "pass_rate": metric_value,
                        "passed": (
                            reward_passed
                            if reward_total
                            else round(metric_value * n_trials)
                        ),
                        "total": n_trials,
                        "errors": errors,
                    }

        passed += reward_passed
        total += reward_total
    if total == 0:
        total = int(
            result.get("n_total_trials") or stats.get("n_completed_trials") or 0
        )
    return {
        "pass_rate": (passed / total) if total else 0.0,
        "passed": passed,
        "total": total,
        "errors": errors,
    }


def _compute_pass_at_k(
    job_dir: Path | None,
    result: dict[str, Any] | None,
    k: int,
) -> dict[str, Any]:
    total_attempts = int((result or {}).get("n_total_trials") or 0)
    total_tasks = total_attempts // k if total_attempts % k == 0 else 0
    successful_tasks: set[str] = set()
    attempt_counts: dict[str, int] = {}
    infrastructure_errors = 0
    parsed_attempts = 0

    if job_dir is not None and job_dir.is_dir():
        for result_path in job_dir.glob("*/result.json"):
            try:
                trial = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            task_name = trial.get("task_name")
            if not isinstance(task_name, str):
                continue
            parsed_attempts += 1
            attempt_counts[task_name] = attempt_counts.get(task_name, 0) + 1

            rewards = (trial.get("verifier_result") or {}).get("rewards") or {}
            if any(reward == 1 for reward in rewards.values()):
                successful_tasks.add(task_name)

            exception_type = (trial.get("exception_info") or {}).get("exception_type")
            if exception_type in _RLI_INFRASTRUCTURE_EXCEPTION_TYPES:
                infrastructure_errors += 1

    complete = (
        total_tasks > 0
        and parsed_attempts == total_attempts
        and len(attempt_counts) == total_tasks
        and all(attempts == k for attempts in attempt_counts.values())
        and infrastructure_errors == 0
    )
    return {
        "k": k,
        "score": len(successful_tasks) / total_tasks if total_tasks else 0.0,
        "passed_tasks": len(successful_tasks),
        "total_tasks": total_tasks,
        "complete": complete,
        "infrastructure_errors": infrastructure_errors,
    }


def _write_configured_pass_at_k(
    result_path: Path | None,
    result: dict[str, Any] | None,
    configured_pass_at_k: dict[str, Any],
) -> None:
    if result_path is None or result is None:
        return

    evals = ((result.get("stats") or {}).get("evals") or {})
    for eval_stats in evals.values():
        if not isinstance(eval_stats, dict):
            continue
        pass_at_k = eval_stats.setdefault("pass_at_k", {})
        pass_at_k[str(configured_pass_at_k["k"])] = configured_pass_at_k["score"]

    temporary_path = result_path.with_suffix(result_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )
    os.replace(temporary_path, result_path)


def _expected_task_names(config: dict[str, Any]) -> list[str]:
    names = config.get("task-name") or config.get("task-id")
    if isinstance(names, str):
        return [names]
    if isinstance(names, list) and names:
        return [str(name) for name in names]
    dataset = config.get("dataset")
    if dataset and Path(str(dataset)).is_dir():
        return sorted(
            entry.name
            for entry in Path(str(dataset)).iterdir()
            if entry.is_dir() and (entry / "task.toml").exists()
        )
    return []


def _reconcile_denominator(
    summary: dict[str, Any], job_dir: Path | None, expected: list[str]
) -> dict[str, Any]:
    """Fix #3: a task whose every retry attempt crashed leaves no result.json
    and silently drops out of Harbor's stats, inflating pass_rate (run3:
    compile-compcert -> 29/88 reported instead of the honest 29/89). Put such
    tasks back into the denominator as errored fails and name them."""
    if not expected or job_dir is None or not job_dir.is_dir():
        return summary
    found = {
        trial.name.rsplit("__", 1)[0]
        for trial in job_dir.iterdir()
        if trial.is_dir() and "__" in trial.name and (trial / "result.json").exists()
    }
    missing = sorted(set(expected) - found)
    if not missing:
        return summary
    for name in missing:
        print(
            f"[DENOMINATOR] task {name!r} produced no trial result.json "
            f"(every attempt crashed) — counted as an errored fail",
            file=sys.stderr,
        )
    summary = dict(summary)
    summary["missing_tasks"] = missing
    summary["errors"] = int(summary.get("errors") or 0) + len(missing)
    total = int(summary.get("total") or 0) + len(missing)
    passed = int(summary.get("passed") or 0)
    summary["total"] = total
    summary["pass_rate"] = (passed / total) if total else 0.0
    return summary


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def _write_run_observability(metrics_dir: Path, job_dir: Path | None) -> None:
    """One artifact answering "how often did the infra problems fire and how
    hard do serving latency / RLI transport bite" (2026-07-27 forensics):
    per-task outcome + wall + episodes + LLM-call latency from result.json,
    problem-dump counts by kind (fix #1/#2 frequency counters), and the
    exec-RTT distribution from the RLI wire trace. Best-effort: never fails
    the run."""
    obs: dict[str, Any] = {"tasks": {}, "problem_dump_counts": {}, "rli_exec_rtt": {}}
    if job_dir is not None and job_dir.is_dir():
        for trial in sorted(job_dir.iterdir()):
            if not trial.is_dir() or "__" not in trial.name:
                continue
            result_file = trial / "result.json"
            if not result_file.exists():
                continue
            try:
                data = json.loads(result_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            exception_type = (data.get("exception_info") or {}).get("exception_type")
            rewards = (data.get("verifier_result") or {}).get("rewards") or {}
            reward = next(iter(rewards.values()), None)
            metadata = (data.get("agent_result") or {}).get("metadata") or {}
            llm_times = [
                float(value)
                for value in metadata.get("api_request_times_msec") or []
                if isinstance(value, (int, float))
            ]
            entry: dict[str, Any] = {
                "outcome": exception_type or ("PASS" if reward else "ZERO"),
                "episodes": metadata.get("n_episodes"),
            }
            try:
                started = datetime.fromisoformat(
                    str(data.get("started_at")).replace("Z", "+00:00")
                )
                finished = datetime.fromisoformat(
                    str(data.get("finished_at")).replace("Z", "+00:00")
                )
                entry["wall_sec"] = round((finished - started).total_seconds(), 1)
            except Exception:
                pass
            if llm_times:
                entry["llm_ms_median"] = round(_percentile(llm_times, 0.5), 1)
                entry["llm_ms_p90"] = round(_percentile(llm_times, 0.9), 1)
                entry["llm_calls"] = len(llm_times)
            obs["tasks"][trial.name.rsplit("__", 1)[0]] = entry
    problems_dir = metrics_dir / "rli_problems"
    if problems_dir.is_dir():
        counts: dict[str, int] = {}
        for dump in problems_dir.glob("*.log"):
            parts = dump.name.split("__")
            kind = parts[2] if len(parts) >= 4 else "unknown"
            counts[kind] = counts.get(kind, 0) + 1
        obs["problem_dump_counts"] = counts
    trace_path = metrics_dir / "rli_trace.jsonl"
    if trace_path.is_file() and trace_path.stat().st_size < 2**31:
        durations: list[float] = []
        pattern = re.compile(r'"duration_ms":\s*([0-9.]+)')
        try:
            with trace_path.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    match = pattern.search(line)
                    if match:
                        durations.append(float(match.group(1)))
        except Exception:
            durations = []
        if durations:
            obs["rli_exec_rtt"] = {
                "events": len(durations),
                "median_ms": round(_percentile(durations, 0.5), 1),
                "p90_ms": round(_percentile(durations, 0.9), 1),
            }
    try:
        (metrics_dir / "run_observability.json").write_text(
            json.dumps(obs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        return
    for task, entry in obs["tasks"].items():
        print(
            f"[OBS] {task}: {entry.get('outcome')} wall={entry.get('wall_sec')}s "
            f"episodes={entry.get('episodes')} llm_med={entry.get('llm_ms_median')}ms "
            f"llm_p90={entry.get('llm_ms_p90')}ms"
        )
    print(f"[OBS] problem_dump_counts={obs['problem_dump_counts']}")
    print(f"[OBS] rli_exec_rtt={obs['rli_exec_rtt']}")


def _methodology_metric_name(config: dict[str, Any]) -> str:
    agent = str(config["agent"]).replace("-", "_").upper()
    execution_limit = (
        f"TIMEOUT_{config['agent-timeout']}"
        if config["agent-timeout"] is not None
        else f"N_CONCURRENT_{config['n-concurrent']}"
    )
    return (
        f"TERMINAL_BENCH_{agent}"
        f"_MAX_TOKENS_{config['max-tokens']}_{execution_limit}"
    )


def _write_metrics(
    metrics_dir: Path,
    summary: dict[str, Any],
    result_path: Path | None,
    config: dict[str, Any],
) -> None:
    payload = dict(summary)
    payload["job_result_path"] = str(result_path) if result_path is not None else None
    metric_name = _methodology_metric_name(config)
    payload["metrics"] = {metric_name: float(payload["pass_rate"])}
    pass_at_k = payload.get("pass_at_k")
    if isinstance(pass_at_k, dict) and int(pass_at_k["k"]) > 1:
        payload["metrics"][f"{metric_name}_PASS_AT_{pass_at_k['k']}"] = float(
            pass_at_k["score"]
        )
    (metrics_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _tail_text(text: str, max_lines: int = 200) -> str:
    return "\n".join(text.splitlines()[-max_lines:])


def run(args: argparse.Namespace) -> int:
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    config = _normalized_config(args.config_json)
    cmd = _build_harbor_cmd(config, metrics_dir)
    command_path = metrics_dir / "harbor_command.json"
    command_path.write_text(
        json.dumps(cmd, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    completed = subprocess.run(
        cmd,
        cwd=metrics_dir,
        env=_build_env(config),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    harbor_stdout = completed.stdout or ""
    (metrics_dir / "harbor_stdout.log").write_text(
        harbor_stdout,
        encoding="utf-8",
    )
    if completed.returncode != 0 and harbor_stdout:
        print(_tail_text(harbor_stdout), file=sys.stderr)

    result_path, result = _load_latest_job_result(metrics_dir / "harbor_jobs")
    summary = _summarize_job_result(result)
    job_dir = result_path.parent if result_path is not None else None
    summary = _reconcile_denominator(summary, job_dir, _expected_task_names(config))
    pass_at_k = int(config["pass-at-k"])
    if pass_at_k > 1:
        configured_pass_at_k = _compute_pass_at_k(job_dir, result, pass_at_k)
        summary["pass_at_k"] = configured_pass_at_k
        _write_configured_pass_at_k(
            result_path,
            result,
            configured_pass_at_k,
        )
    _write_metrics(metrics_dir, summary, result_path, config)
    try:
        _write_run_observability(metrics_dir, job_dir)
    except Exception as exc:
        print(f"[OBS] observability failed: {exc}", file=sys.stderr)
    return completed.returncode


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "run":
        return run(args)
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
