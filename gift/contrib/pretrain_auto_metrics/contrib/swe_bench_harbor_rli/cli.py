from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

from gigarli import RLIClient

from .src.token_budget import MAX_TOTAL_OUTPUT_TOKENS_ENV
from .utils.log_analysis import (
    correlate_rli_trace,
    load_target_trials,
    nearest_event,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "dataset": "/home/user/swe-bench-verified/datasets/swebench-verified",
    "agent": "mini-swe-agent",
    "agent-kwargs": {},
    "model-name": "openai/pam-model",
    "agent-timeout": None,
    "n-concurrent": 4,
    "pass-at-k": 1,
    "api_key": "dummy",
    "max-tokens": 32768,
    "max-total-output-tokens": None,
    "llm-timeout": 6400,
    "env-api-base": None,
    "env-prefix": "swe-bench",
    "env-version": "1.0.2",
    "ttl-ms": 64800000,
    "post-reverify": False,
    "post-reverify-grader": "rli",
    "verify-ssl": False,
    "allow-internet-access": False,
    "use-pip-proxy": False,
    "timeout-rli": 1620,
    "request-retries": 0,
    "request-retry-backoff": 2.0,
    "transfer-retries": 2,
    "transfer-retry-backoff": 2.0,
    "session-ready-timeout": 600.0,
    "session-ready-poll-interval": 1.0,
    "enable-probe-warmup": True,
    "create-session-jitter-max-sec": 60.0,
    "probe-retry-backoff-sec": 2.0,
    "command-timeout": 180,
    "verifier-timeout": 2400,
    "quiet": True,
    "yes": True,
    "max-retries": 1,
}

_AGENT_IMPORT_PATHS = {
    "terminus-2": "swe_bench_harbor_rli.agents.terminus.terminus_agent:RLITerminus2Agent",
    "mini-swe-agent-proxy": "swe_bench_harbor_rli.agents.mini_swe_agent.mini_swe_agent:RLIMiniSweAgent",
    "mini-swe-agent": "swe_bench_harbor_rli.agents.mini_swe_agent.rli_container_agent:RLIContainerMiniSweAgent",
    "agentless": "swe_bench_harbor_rli.agents.agentless.agentless_agent:RLIAgentlessAgent",
    "openhands": "swe_bench_harbor_rli.agents.openhands_sdk.openhands_sdk_agent:RLIOpenHandsSDKAgent",
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

_ASYNC_VERIFY_SCRIPT = (
    Path(__file__).resolve().parent / "local_verifier" / "async_verify.py"
)
_POST_REVERIFY_TIMEOUT_SEC = int(
    os.environ.get("PAM_POST_REVERIFY_TIMEOUT_SEC", str(2 * 3600))
)
_SWE_BENCH_VERIFIER_TIMEOUT_SEC = 3600
_RLI_INFRASTRUCTURE_EXCEPTION_TYPES = {
    "CancelledError",
    "HTTPStatusError",
    "JSONDecodeError",
    "ReadError",
    "RemoteProtocolError",
    "RLIExecProxyConnectionLostError",
    "RLILongRunningCommandLostError",
    "TransportError",
}
_RLI_REPORT_NAME = "rli_infrastructure_report.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SWE-bench through Harbor + RLI")
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
    verifier_command_timeout = config.get("verifier-command-timeout")
    if verifier_command_timeout is None:
        verifier_command_timeout = config.get("verifier-timeout")
    return {
        "api_base": config.get("env-api-base"),
        "env_prefix": config.get("env-prefix"),
        "env_version": config.get("env-version"),
        "env_name": config.get("env-name"),
        "task_id": config.get("task-id"),
        "ttl_ms": config.get("ttl-ms"),
        "agent_timeout_sec": config.get("agent-timeout"),
        "seed": config.get("seed"),
        "verify_ssl": config.get("verify-ssl"),
        "allow_internet_access": config.get("allow-internet-access", False),
        "use_pip_proxy": config.get("use-pip-proxy"),
        "timeout": config.get("timeout-rli"),
        "request_retries": config.get("request-retries"),
        "request_retry_backoff": config.get("request-retry-backoff"),
        "transfer_retries": config.get("transfer-retries"),
        "transfer_retry_backoff": config.get("transfer-retry-backoff"),
        "session_ready_timeout": config.get("session-ready-timeout"),
        "session_ready_poll_interval": config.get("session-ready-poll-interval"),
        "enable_probe_warmup": config.get("enable-probe-warmup"),
        "probe_warmup_command": config.get("probe-warmup-command"),
        "create_session_jitter_max_sec": config.get("create-session-jitter-max-sec"),
        "probe_retry_backoff_sec": config.get("probe-retry-backoff-sec"),
        "command_timeout_ms": (
            int(command_timeout) * 1000 if command_timeout is not None else None
        ),
        "verifier_command_timeout_ms": (
            int(verifier_command_timeout) * 1000
            if verifier_command_timeout is not None
            else None
        ),
        "trace_path": trace_path,
    }


def _selected_agent_import_path(config: dict[str, Any]) -> str:
    if config.get("agent-import-path") or config.get("agent_import_path"):
        raise ValueError("swe_bench.agent-import-path is not supported")
    agent = config.get("agent")
    try:
        return _AGENT_IMPORT_PATHS[str(agent)]
    except KeyError as exc:
        allowed = ", ".join(sorted(_AGENT_IMPORT_PATHS))
        raise ValueError(
            f"Unsupported swe_bench agent {agent!r}; use one of: {allowed}"
        ) from exc


def _build_agent_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    agent_kwargs = _common_agent_kwargs(config)
    agent_kwargs.update(config.get("agent-kwargs") or {})
    if (
        "command_timeout" not in agent_kwargs
        and config.get("command-timeout") is not None
    ):
        agent_kwargs["command_timeout"] = config["command-timeout"]
    agent_llm_timeout = agent_kwargs.pop("llm-timeout", None)
    llm_timeout = agent_kwargs.get("llm_timeout")
    if llm_timeout is None:
        llm_timeout = agent_llm_timeout
    if llm_timeout is None:
        llm_timeout = config.get("llm_timeout")
    if llm_timeout is None:
        llm_timeout = config.get("llm-timeout")
    if llm_timeout is not None:
        try:
            llm_timeout = int(llm_timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("llm-timeout must be a positive integer") from exc
        if llm_timeout < 1:
            raise ValueError("llm-timeout must be a positive integer")
        agent_kwargs["llm_timeout"] = llm_timeout
    return agent_kwargs


def _build_harbor_job_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "agents": [
            {
                "import_path": _selected_agent_import_path(config),
                "model_name": config.get("model-name"),
                "override_timeout_sec": config.get("agent-timeout"),
                "kwargs": _build_agent_kwargs(config),
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
        "swe_bench_harbor_rli.src.environment:RLIEnvironment",
    )
    if not config.get("disable-verification"):
        _append_flag(
            cmd,
            "--verifier-import-path",
            "swe_bench_harbor_rli.src.verifier:SWEBenchLocalRLIVerifier",
        )
        verifier_timeout = config.get("verifier-timeout")
        if verifier_timeout is not None:
            _append_flag(
                cmd,
                "--verifier-timeout-multiplier",
                float(verifier_timeout) / _SWE_BENCH_VERIFIER_TIMEOUT_SEC,
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
    if isinstance(reward_bucket, dict):
        passed = 0
        total = 0
        for raw_reward, trial_names in reward_bucket.items():
            count = len(trial_names) if isinstance(trial_names, list) else 0
            total += count
            reward = _as_number(raw_reward)
            if reward is not None and reward > 0:
                passed += count
        return passed, total

    n_trials = int(eval_stats.get("n_trials") or 0)
    for metric in eval_stats.get("metrics") or []:
        if not isinstance(metric, dict):
            continue
        for key in ("pass_rate", "success_rate", "accuracy", "resolved_rate", "mean"):
            value = _as_number(metric.get(key))
            if value is not None:
                return round(value * n_trials), n_trials
    return 0, 0


def _summarize_job_result(result: dict[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {
            "pass_rate": 0.0,
            "resolved_over_total": 0.0,
            "pass_rate_completed": 0.0,
            "passed": 0,
            "graded": 0,
            "completed": 0,
            "total": 0,
            "errors": 0,
            "exception_types": [],
            "rli_infrastructure_errors": [],
        }

    stats = result.get("stats") or {}
    evals = stats.get("evals") or {}
    errors = int(stats.get("n_errored_trials") or stats.get("n_errors") or 0)

    passed = 0
    graded = 0
    exception_types: set[str] = set()
    for eval_stats in evals.values():
        if not isinstance(eval_stats, dict):
            continue
        exception_types.update((eval_stats.get("exception_stats") or {}).keys())
        reward_passed, reward_total = _reward_counts(eval_stats)
        passed += reward_passed
        graded += reward_total

    completed = int(stats.get("n_completed_trials") or 0) or (graded + errors)
    expected_total = int(result.get("n_total_trials") or 0) or max(completed, graded)
    resolved_over_total = (passed / expected_total) if expected_total else 0.0

    return {
        "pass_rate": resolved_over_total,
        "resolved_over_total": resolved_over_total,
        "pass_rate_completed": (passed / graded) if graded else 0.0,
        "passed": passed,
        "graded": graded,
        "completed": completed,
        "total": expected_total,
        "errors": errors,
        "exception_types": sorted(exception_types),
        "rli_infrastructure_errors": sorted(
            _RLI_INFRASTRUCTURE_EXCEPTION_TYPES.intersection(exception_types)
        ),
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


def _methodology_metric_name(config: dict[str, Any]) -> str:
    agent = str(config["agent"]).replace("-", "_").upper()
    execution_limit = (
        f"TIMEOUT_{config['agent-timeout']}"
        if config["agent-timeout"] is not None
        else f"N_CONCURRENT_{config['n-concurrent']}"
    )
    return (
        f"SWE_BENCH_VERIFIED_{agent}"
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


def _rli_exception_stats(
    result: dict[str, Any] | None,
) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {}
    stats = (result or {}).get("stats") or {}
    for eval_stats in (stats.get("evals") or {}).values():
        if not isinstance(eval_stats, dict):
            continue
        exception_stats = eval_stats.get("exception_stats") or {}
        for exception_type in _RLI_INFRASTRUCTURE_EXCEPTION_TYPES:
            trial_names = exception_stats.get(exception_type)
            if not isinstance(trial_names, list):
                continue
            selected.setdefault(exception_type, []).extend(
                str(trial_name) for trial_name in trial_names
            )
    return {
        exception_type: list(dict.fromkeys(trial_names))
        for exception_type, trial_names in selected.items()
    }


def _container_periods(
    events: list[dict[str, Any]],
    session_id: str | None,
) -> list[dict[str, str | None]]:
    if session_id is None:
        return []
    periods: dict[str, dict[str, str | None]] = {}
    for event in events:
        if event.get("session_id") != session_id:
            continue
        timestamp = event.get("ts")
        for name in event.get("container_names") or []:
            if name not in periods:
                periods[name] = {
                    "name": name,
                    "first_seen_at": timestamp,
                    "last_seen_at": timestamp,
                }
            else:
                periods[name]["last_seen_at"] = timestamp
    return list(periods.values())


def _environment_for_session(
    events: list[dict[str, Any]],
    session_id: str | None,
) -> dict[str, str] | None:
    for event in events:
        if session_id is not None and event.get("session_id") != session_id:
            continue
        environment = event.get("environment")
        if isinstance(environment, dict):
            return environment
    return None


def _write_rli_infrastructure_report(
    metrics_dir: Path,
    result_path: Path | None,
    result: dict[str, Any] | None,
    config: dict[str, Any],
) -> None:
    report_path = metrics_dir / _RLI_REPORT_NAME
    exception_stats = _rli_exception_stats(result)
    if not exception_stats:
        report_path.unlink(missing_ok=True)
        return
    if result_path is None:
        return

    reports = load_target_trials(
        run_dir=result_path.parent,
        exception_stats=exception_stats,
        exception_types=sorted(exception_stats),
        snippet_chars=2_000,
    )
    trace_path = Path(
        str(config.get("rli-trace-path") or metrics_dir / "rli_trace.jsonl")
    )
    if trace_path.is_file():
        correlate_rli_trace(
            rli_trace_path=trace_path,
            reports=reports,
            snippet_chars=2_000,
            last_events=10_000,
        )

    errors: list[dict[str, Any]] = []
    for report in reports.values():
        events = list(report.last_trace_events)
        trace_event = nearest_event(
            report.failed_trace_events or report.problem_trace_events or events,
            report.exception_occurred_at,
        )
        session_id = trace_event.get("session_id") if trace_event else None
        if session_id is None and len(report.session_ids) == 1:
            session_id = next(iter(report.session_ids))
        errors.append(
            {
                "exception_type": report.exception_type,
                "occurred_at": report.exception_occurred_at,
                "containers": _container_periods(events, session_id),
                "environment": _environment_for_session(events, session_id),
                "session_id": session_id,
                "request_id": trace_event.get("request_id") if trace_event else None,
                "rli_error_code": (
                    trace_event.get("rli_error_code") if trace_event else None
                ),
                "http_status": (
                    trace_event.get("status_code") if trace_event else None
                ),
                "message": (
                    trace_event.get("rli_error_message")
                    if trace_event and trace_event.get("rli_error_message")
                    else report.exception_message
                ),
            }
        )
    errors.sort(key=lambda error: error.get("occurred_at") or "")
    temporary_path = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps({"errors": errors}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_path, report_path)


def _tail_text(text: str, max_lines: int = 200) -> str:
    return "\n".join(text.splitlines()[-max_lines:])


def _collect_in_job_resolved(metrics_dir: Path) -> dict[str, bool]:
    """In-job harbor verdicts: */verifier/reward.txt under metrics_dir,
    OR-reduced across repeated trials of the same instance."""
    resolved: dict[str, bool] = {}
    for reward_path in metrics_dir.rglob("verifier/reward.txt"):
        trial = reward_path.parent.parent
        name = trial.name
        iid = name.rsplit("__", 1)[0] if "__" in name else name
        try:
            value = float(reward_path.read_text().strip() or "0")
        except (OSError, ValueError):
            continue
        resolved[iid] = resolved.get(iid, False) or (value >= 0.5)
    return resolved


def _write_reverified_metrics(
    metrics_dir: Path,
    out_dir: Path,
    *,
    grader_label: str,
) -> None:
    in_job = _collect_in_job_resolved(metrics_dir)
    grader_results: dict[str, bool] = {}
    results_path = out_dir / "results_async.jsonl"
    if results_path.is_file():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = row.get("instance_id")
            if iid:
                grader_results[iid] = grader_results.get(iid, False) or bool(
                    row.get("resolved")
                )
    union = sorted(
        {iid for iid, ok in in_job.items() if ok}
        | {iid for iid, ok in grader_results.items() if ok}
    )
    payload: dict[str, Any] = {
        "resolved_count_reverified": len(union),
        "in_job_resolved_count": sum(1 for ok in in_job.values() if ok),
        "in_job_graded_count": len(in_job),
        "grader_resolved_count": sum(1 for ok in grader_results.values() if ok),
        "grader_graded_count": len(grader_results),
        "grader": grader_label,
        "resolved_instances": union,
        "note": (
            "advisory re-verify (in-job rewards UNION async grader verdicts); "
            "primary metrics.json untouched"
        ),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    metrics_path = metrics_dir / "metrics.json"
    if metrics_path.is_file():
        primary = json.loads(metrics_path.read_text(encoding="utf-8"))
        payload["primary_pass_rate"] = primary.get("pass_rate")
        total = primary.get("total")
        if isinstance(total, int) and total > 0:
            payload["pass_rate_reverified"] = len(union) / total
            payload["denominator"] = total
    target = metrics_dir / "metrics_reverified.json"
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp, target)
    print(
        f"post_reverify: wrote {target} "
        f"(resolved_reverified={len(union)}, grader_graded={len(grader_results)})",
        file=sys.stderr,
    )


def _config_bool(config: dict[str, Any], *keys: str) -> bool:
    return any(bool(config.get(key)) for key in keys)


def _post_reverify_grader(config: dict[str, Any]) -> str | None:
    raw = (
        config.get("post-reverify-grader")
        or config.get("post_reverify_grader")
        or "rli"
    )
    grader = str(raw).strip()
    if grader.lower() in {
        "",
        "default",
        "nodocker",
        "host",
        "host-nodocker",
        "host_nodocker",
    }:
        return None
    return grader


def _grader_label(grader: str | None) -> str:
    if grader is None:
        return "default"
    label = Path(grader).name or grader
    return "".join(ch if ch.isalnum() else "_" for ch in label).strip("_") or "custom"


def _post_reverify_out_dir(
    config: dict[str, Any], metrics_dir: Path, grader: str | None
) -> Path:
    configured = config.get("post-reverify-out-dir") or config.get(
        "post_reverify_out_dir"
    )
    if configured:
        return Path(str(configured))
    label = _grader_label(grader)
    if grader == "rli":
        return metrics_dir / "async_final"
    return metrics_dir / f"async_final_{label}"


def _run_post_reverify(config: dict[str, Any], metrics_dir: Path) -> None:
    if not _config_bool(config, "post-reverify", "post_reverify"):
        return
    if not _ASYNC_VERIFY_SCRIPT.is_file():
        print(
            f"post_reverify: driver not found at {_ASYNC_VERIFY_SCRIPT}, skipping",
            file=sys.stderr,
        )
        return
    # Advisory re-verify: a failure here must not discard the primary
    # metrics.json written above, so it is logged but never fails the run.
    try:
        grader = _post_reverify_grader(config)
        grader_label = _grader_label(grader)
        out_dir = _post_reverify_out_dir(config, metrics_dir, grader)
        par = int(config.get("n-concurrent") or 4)
        rli_base = RLIClient(base_url=config.get("env-api-base")).base_url
        cmd = [
            sys.executable,
            str(_ASYNC_VERIFY_SCRIPT),
            "--once",
            "--watch",
            str(metrics_dir),
            "--out",
            str(out_dir),
            "--par",
            str(par),
            "--min-age-sec",
            "0",
            "--rli-base",
            rli_base,
            "--rli-env-version",
            str(config["env-version"]),
            "--dataset-root",
            str(config.get("dataset")),
        ]
        cmd.append(
            "--allow-internet-access"
            if config.get("allow-internet-access", False)
            else "--no-allow-internet-access"
        )
        if grader is not None:
            cmd.extend(["--grader", grader])
        command_path = out_dir / "post_reverify_command.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        command_path.write_text(
            json.dumps(cmd, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"post_reverify: starting {cmd}", file=sys.stderr)
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=_POST_REVERIFY_TIMEOUT_SEC,
            check=False,
        )
        tail = (proc.stdout or b"")[-2000:].decode("utf-8", errors="replace")
        print(
            f"post_reverify: async_verify exited rc={proc.returncode}; tail:\n{tail}",
            file=sys.stderr,
        )
        _write_reverified_metrics(
            metrics_dir,
            out_dir,
            grader_label=grader_label,
        )
    except Exception as exc:
        print(f"post_reverify failed (advisory only): {exc}", file=sys.stderr)


def run(args: argparse.Namespace) -> int:
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    config = _normalized_config(args.config_json)
    skip_primary_run = _config_bool(config, "skip-primary-run", "skip_primary_run")

    if skip_primary_run:
        if not (metrics_dir / "metrics.json").is_file():
            print(
                f"skip-primary-run requested but {metrics_dir / 'metrics.json'} is missing",
                file=sys.stderr,
            )
            return 2
        completed_returncode = 0
        print(
            f"skip-primary-run: using existing primary metrics at {metrics_dir / 'metrics.json'}",
            file=sys.stderr,
        )
    else:
        cmd = _build_harbor_cmd(config, metrics_dir)
        command_path = metrics_dir / "harbor_command.json"
        command_path.write_text(
            json.dumps(cmd, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        completed = subprocess.run(
            cmd,
            cwd=metrics_dir,
            env=_build_env(config),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        completed_returncode = completed.returncode
        harbor_stdout = completed.stdout or ""
        (metrics_dir / "harbor_stdout.log").write_text(
            harbor_stdout,
            encoding="utf-8",
        )
        if completed.returncode != 0 and harbor_stdout:
            print(_tail_text(harbor_stdout), file=sys.stderr)

        result_path, result = _load_latest_job_result(metrics_dir / "harbor_jobs")
        summary = _summarize_job_result(result)
        pass_at_k = int(config["pass-at-k"])
        if pass_at_k > 1:
            configured_pass_at_k = _compute_pass_at_k(
                result_path.parent if result_path is not None else None,
                result,
                pass_at_k,
            )
            summary["pass_at_k"] = configured_pass_at_k
            _write_configured_pass_at_k(
                result_path,
                result,
                configured_pass_at_k,
            )
        _write_metrics(metrics_dir, summary, result_path, config)
        _write_rli_infrastructure_report(
            metrics_dir,
            result_path,
            result,
            config,
        )

    _run_post_reverify(config, metrics_dir)
    return completed_returncode


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "run":
        return run(args)
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
