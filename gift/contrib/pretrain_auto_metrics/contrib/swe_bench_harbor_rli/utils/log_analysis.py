import argparse
import json
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DEFAULT_METRICS_DIR = Path(
    "/workspace-SR008.nfs2/models/minimax-m2.7/artifacts/metrics/swe_bench"
)
DEFAULT_EVAL_KEY = "mini-swe-agent__pam-model__swebench-verified"
DEFAULT_EXCEPTION_TYPES = (
    "AgentTimeoutError",
    "RuntimeError",
    "RewardFileNotFoundError",
    "DownloadVerifierDirError",
    "HTTPStatusError",
    "InternalServerError",
    "RLIClientError",
    "RLIExecProxyConnectionLostError",
)
DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent / "errors_reports"
DEFAULT_SNIPPET_CHARS = 2_000
DEFAULT_LAST_EVENTS = 8
TRIAL_NAME_SEPARATOR = "__"
RLI_CONTAINER_PATTERN = re.compile(r"\broot@(rli-[^:\s]+):")


@dataclass
class TrialReport:
    trial_name: str
    task_id: str
    trial_dir: Path
    exception_type: str
    exception_message: str | None = None
    exception_traceback: str | None = None
    exception_occurred_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    environment_setup_started_at: str | None = None
    environment_setup_finished_at: str | None = None
    agent_setup_started_at: str | None = None
    agent_setup_finished_at: str | None = None
    session_ids: set[str] = field(default_factory=set)
    trace_stage_counts: Counter[str] = field(default_factory=Counter)
    trace_event_count: int = 0
    failed_trace_events: list[dict[str, Any]] = field(default_factory=list)
    problem_trace_events: list[dict[str, Any]] = field(default_factory=list)
    last_trace_events: deque[dict[str, Any]] = field(default_factory=deque)
    session_trace_event_counts: Counter[str] = field(default_factory=Counter)
    session_stage_counts: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    session_first_trace_events: dict[str, dict[str, Any]] = field(default_factory=dict)
    session_last_trace_events: dict[str, dict[str, Any]] = field(default_factory=dict)
    session_problem_trace_events: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    session_created_at: dict[str, str] = field(default_factory=dict)
    session_closed_at: dict[str, str] = field(default_factory=dict)
    exception_text: str | None = None
    trial_log_tail: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Harbor trials with selected exception types and correlate them with "
            "RLI trace events."
        )
    )
    parser.add_argument(
        "metrics_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_METRICS_DIR,
        help=(
            "SWE-bench metrics directory. It can be the backup root "
            "containing harbor_jobs/ and rli_trace.jsonl, or a concrete "
            "harbor run directory."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Concrete Harbor run directory. Overrides auto discovery.",
    )
    parser.add_argument(
        "--rli-trace",
        type=Path,
        default=None,
        help="Path to rli_trace.jsonl. Defaults to <metrics_dir>/rli_trace.jsonl.",
    )
    parser.add_argument(
        "--eval-key",
        default=DEFAULT_EVAL_KEY,
        help="Eval key inside top-level result.json stats.evals.",
    )
    parser.add_argument(
        "--exception-type",
        action="append",
        default=None,
        help=(
            "Exception type to extract from exception_stats. Can be repeated or "
            "comma-separated. Defaults to the common Harbor/RLI failure types "
            "(AgentTimeoutError, RuntimeError, RewardFileNotFoundError, "
            "DownloadVerifierDirError, HTTPStatusError, InternalServerError, "
            "RLIClientError, RLIExecProxyConnectionLostError). Use 'all' "
            "explicitly to include all exception_stats entries."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help=("Directory for task reports. Defaults to <script_dir>/errors_reports."),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional aggregate summary JSON path. Defaults to <output-dir>/_summary.json.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Only print a text summary; do not write report files.",
    )
    parser.add_argument(
        "--dump-traces-dir",
        type=Path,
        default=None,
        help=(
            "Directory for per-task JSONL trace snippets. Defaults to "
            "<output-dir>/traces."
        ),
    )
    parser.add_argument(
        "--snippet-chars",
        type=int,
        default=DEFAULT_SNIPPET_CHARS,
        help="Max characters kept from exception/trial logs and command output.",
    )
    parser.add_argument(
        "--last-events",
        type=int,
        default=DEFAULT_LAST_EVENTS,
        help="Number of last trace events kept per trial in the report.",
    )
    parser.add_argument(
        "--max-print-trials",
        type=int,
        default=None,
        help="Max number of trial details printed to stdout. Reports still include all trials.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_text_if_exists(path: Path, max_chars: int) -> str | None:
    if not path.exists():
        return None

    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def task_id_from_trial_name(trial_name: str) -> str:
    return trial_name.rsplit(TRIAL_NAME_SEPARATOR, 1)[0]


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None

    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_report_ts(value: str | None) -> str | None:
    parsed = parse_ts(value)
    if parsed is None:
        return value
    return parsed.strftime("%H:%M:%S___%Y-%m-%d")


def sortable_report_ts(value: str | None) -> str:
    if not value:
        return ""
    if "___" in value:
        time_part, date_part = value.split("___", 1)
        return f"{date_part}T{time_part}"
    parsed = parse_ts(value)
    if parsed is None:
        return value
    return parsed.strftime("%Y-%m-%dT%H:%M:%S")


def discover_run_dir(metrics_dir: Path, run_dir: Path | None) -> Path:
    if run_dir is not None:
        return run_dir

    if (metrics_dir / "result.json").is_file() and (metrics_dir / "job.log").is_file():
        return metrics_dir

    harbor_jobs = metrics_dir / "harbor_jobs"
    if not harbor_jobs.is_dir():
        raise FileNotFoundError(
            f"Cannot find harbor_jobs under {metrics_dir}. Pass --run-dir explicitly."
        )

    run_dirs = sorted(
        path
        for path in harbor_jobs.iterdir()
        if path.is_dir() and (path / "result.json").is_file()
    )
    if not run_dirs:
        raise FileNotFoundError(
            f"No Harbor run dirs with result.json under {harbor_jobs}"
        )
    if len(run_dirs) > 1:
        raise RuntimeError(
            "Multiple Harbor run dirs found. Pass --run-dir explicitly: "
            + ", ".join(str(path) for path in run_dirs)
        )
    return run_dirs[0]


def discover_rli_trace(
    metrics_dir: Path, run_dir: Path, rli_trace: Path | None
) -> Path:
    if rli_trace is not None:
        return rli_trace

    candidates = [
        metrics_dir / "rli_trace.jsonl",
        run_dir.parent.parent / "rli_trace.jsonl",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError("Cannot find rli_trace.jsonl. Pass --rli-trace explicitly.")


def exception_stats_for_eval(run_dir: Path, eval_key: str) -> dict[str, list[str]]:
    run_result = read_json(run_dir / "result.json")
    evals = run_result.get("stats", {}).get("evals", {})
    if eval_key not in evals:
        raise KeyError(
            f"Eval key {eval_key!r} is missing. Available keys: {sorted(evals)}"
        )

    raw_exception_stats = evals[eval_key].get("exception_stats", {})
    if not isinstance(raw_exception_stats, dict):
        return {}

    exception_stats: dict[str, list[str]] = {}
    for exception_type, trial_names in raw_exception_stats.items():
        if isinstance(trial_names, list):
            exception_stats[str(exception_type)] = [str(name) for name in trial_names]
    return exception_stats


def resolve_exception_types(
    exception_stats: dict[str, list[str]],
    requested_values: list[str] | None,
) -> list[str]:
    if requested_values is None:
        return [
            exception_type
            for exception_type in DEFAULT_EXCEPTION_TYPES
            if exception_type in exception_stats
        ]

    requested: list[str] = []
    for raw_value in requested_values:
        for value in str(raw_value).split(","):
            value = value.strip()
            if value:
                requested.append(value)

    if not requested or any(value.lower() == "all" for value in requested):
        return sorted(exception_stats)

    deduplicated = list(dict.fromkeys(requested))
    missing = [value for value in deduplicated if value not in exception_stats]
    if missing:
        raise KeyError(
            "Requested exception types are missing from exception_stats: "
            + ", ".join(missing)
            + ". Available types: "
            + ", ".join(sorted(exception_stats))
        )
    return deduplicated


def load_target_trials(
    run_dir: Path,
    exception_stats: dict[str, list[str]],
    exception_types: list[str],
    snippet_chars: int,
) -> dict[str, TrialReport]:
    reports: dict[str, TrialReport] = {}

    for exception_type in exception_types:
        trial_names = sorted(exception_stats.get(exception_type, []))
        for trial_name in trial_names:
            trial_dir = run_dir / trial_name
            task_id = task_id_from_trial_name(trial_name)
            report = TrialReport(
                trial_name=trial_name,
                task_id=task_id,
                trial_dir=trial_dir,
                exception_type=exception_type,
            )

            trial_result_path = trial_dir / "result.json"
            if trial_result_path.is_file():
                trial_result = read_json(trial_result_path)
                exception_info = trial_result.get("exception_info") or {}
                environment_setup = trial_result.get("environment_setup") or {}
                agent_setup = trial_result.get("agent_setup") or {}

                report.exception_message = exception_info.get("exception_message")
                report.exception_traceback = exception_info.get("exception_traceback")
                report.exception_occurred_at = exception_info.get("occurred_at")
                report.started_at = trial_result.get("started_at")
                report.finished_at = trial_result.get("finished_at")
                report.environment_setup_started_at = environment_setup.get(
                    "started_at"
                )
                report.environment_setup_finished_at = environment_setup.get(
                    "finished_at"
                )
                report.agent_setup_started_at = agent_setup.get("started_at")
                report.agent_setup_finished_at = agent_setup.get("finished_at")

            report.exception_text = read_text_if_exists(
                trial_dir / "exception.txt",
                snippet_chars,
            )
            report.trial_log_tail = read_text_if_exists(
                trial_dir / "trial.log",
                snippet_chars,
            )
            reports[trial_name] = report

    return reports


def assign_trial_for_event(
    task_id: str,
    event_ts: datetime | None,
    task_to_trials: dict[str, list[TrialReport]],
) -> list[TrialReport]:
    candidates = task_to_trials.get(task_id, [])
    if not candidates or event_ts is None:
        return candidates

    matched: list[TrialReport] = []
    slack = timedelta(seconds=10)
    for report in candidates:
        started_at = parse_ts(report.started_at)
        finished_at = parse_ts(report.finished_at)
        if started_at is None:
            continue
        if finished_at is None:
            finished_at = started_at + timedelta(days=365)
        if started_at - slack <= event_ts <= finished_at + slack:
            matched.append(report)

    return matched or candidates


def get_nested(mapping: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def response_body_payload(item: dict[str, Any]) -> dict[str, Any] | None:
    body = get_nested(item, ("response", "body"))
    if not isinstance(body, str) or not body.strip().startswith("{"):
        return None
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def container_names_from_trace_item(item: dict[str, Any]) -> list[str]:
    response_result = get_nested(item, ("response", "result"))
    if response_result is None:
        return []

    texts: list[str] = []

    def collect_text(value: Any) -> None:
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, list):
            for entry in value:
                collect_text(entry)
        elif isinstance(value, dict):
            for entry in value.values():
                collect_text(entry)

    collect_text(response_result)
    names: list[str] = []
    for text in texts:
        names.extend(RLI_CONTAINER_PATTERN.findall(text))
    return list(dict.fromkeys(names))


def environment_from_trace_item(item: dict[str, Any]) -> dict[str, str] | None:
    request_environment = get_nested(item, ("request", "json", "env"))
    response_result = get_nested(item, ("response", "result"))
    name = None
    version = None
    if isinstance(request_environment, dict):
        name = request_environment.get("name")
        version = request_environment.get("version")
    if isinstance(response_result, dict):
        name = name or response_result.get("env_name")
        version = version or response_result.get("env_version")
    environment = {
        key: str(value)
        for key, value in (("name", name), ("version", version))
        if value is not None
    }
    return environment or None


def session_id_from_trace_item(item: dict[str, Any]) -> str | None:
    response_meta_session = get_nested(item, ("response", "meta", "session_id"))
    if response_meta_session:
        return response_meta_session

    response_result_session = get_nested(item, ("response", "result", "session_id"))
    if response_result_session:
        return response_result_session

    request_session = get_nested(item, ("request", "session_id"))
    if request_session:
        return request_session

    request_session = get_nested(item, ("request", "json", "session_id"))
    if request_session:
        return request_session

    return None


def task_id_from_create_item(item: dict[str, Any]) -> str | None:
    return get_nested(item, ("request", "json", "env", "task_id"))


def trim_text(value: Any, max_chars: int) -> str | None:
    if value is None:
        return None
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars // 2] + "\n...[truncated]...\n" + text[-max_chars // 2 :]


def command_from_trace_item(item: dict[str, Any]) -> str | None:
    actions = get_nested(item, ("request", "json", "actions"))
    if not isinstance(actions, list) or not actions:
        return None

    commands: list[str] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        args = action.get("args")
        if isinstance(args, dict) and args.get("cmd"):
            commands.append(str(args["cmd"]))
    return "\n\n".join(commands) if commands else None


def underlying_rli_error(item: dict[str, Any]) -> dict[str, Any] | None:
    response = item.get("response")
    if not isinstance(response, dict):
        return None

    error = response.get("error")
    if not isinstance(error, dict):
        body = response.get("body")
        if isinstance(body, str) and body.strip().startswith("{"):
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                nested = parsed.get("error")
                error = nested if isinstance(nested, dict) else parsed

    if not isinstance(error, dict):
        body = response.get("body")
        status_code = response.get("status_code")
        body_text = body if isinstance(body, str) else ""
        if status_code == 503 and "No server is available" in body_text:
            return {
                "underlying_cause": "rli_gateway_unavailable",
                "rli_error_message": "RLI gateway returned 503: no server is available",
            }
        if status_code == 500:
            return {
                "underlying_cause": "rli_internal_server_error",
                "rli_error_message": "RLI gateway returned 500 Internal Server Error",
            }
        return None

    code = error.get("code")
    category = error.get("category")
    message = str(error.get("message") or error.get("error") or "")
    if not (code or category or message):
        return None

    registry_429 = "429" in message or "Too Many Requests" in message
    if registry_429:
        underlying_cause = "registry_rate_limited_429"
    elif code == "POD_NEVER_REACHED":
        underlying_cause = "pod_never_reached"
    elif code == "DATABASE_ERROR":
        underlying_cause = "rli_database_error"
    elif code:
        underlying_cause = str(code).lower()
    elif category:
        underlying_cause = str(category).lower()
    else:
        underlying_cause = "rli_error"

    result: dict[str, Any] = {"underlying_cause": underlying_cause}
    if code:
        result["rli_error_code"] = code
    if category:
        result["rli_error_category"] = category
    if message:
        result["rli_error_message"] = message
    retry_strategy = error.get("retry_strategy")
    if retry_strategy:
        result["rli_retry_strategy"] = retry_strategy
    return result


def summarize_trace_item(item: dict[str, Any], snippet_chars: int) -> dict[str, Any]:
    body_payload = response_body_payload(item)
    body_meta = body_payload.get("meta") if isinstance(body_payload, dict) else None
    body_error = body_payload.get("error") if isinstance(body_payload, dict) else None
    event: dict[str, Any] = {
        "ts": item.get("ts"),
        "stage": item.get("stage"),
        "success": item.get("success"),
        "session_id": session_id_from_trace_item(item),
        "request_id": (
            get_nested(item, ("response", "meta", "request_id"))
            or (body_meta.get("request_id") if isinstance(body_meta, dict) else None)
        ),
        "status": get_nested(item, ("response", "status")),
        "status_code": get_nested(item, ("response", "status_code")),
        "error": get_nested(item, ("response", "error")) or body_error,
        "environment": environment_from_trace_item(item),
        "container_names": container_names_from_trace_item(item),
    }

    rli_error = underlying_rli_error(item)
    if rli_error:
        event.update(rli_error)
        message = (
            body_error.get("message_en") if isinstance(body_error, dict) else None
        ) or rli_error.get("rli_error_message")
        if isinstance(message, str) and message:
            event["rli_error_message"] = trim_text(message, snippet_chars)

    command = command_from_trace_item(item)
    if command:
        event["command"] = trim_text(command, snippet_chars)

    observation_output = get_nested(
        item, ("response", "result", "observation", "output")
    )
    if observation_output is not None:
        event["observation_output"] = trim_text(observation_output, snippet_chars)

    event_data = get_nested(item, ("response", "result", "events"))
    if not isinstance(event_data, list):
        event_data = get_nested(item, ("response", "result", "observation", "events"))
    if isinstance(event_data, list) and event_data:
        exit_codes: list[int] = []
        event_errors: list[str] = []
        incomplete = False
        durations: list[int] = []
        for raw_event in event_data:
            if not isinstance(raw_event, dict):
                continue
            exit_code = raw_event.get("exit_code")
            if isinstance(exit_code, int):
                exit_codes.append(exit_code)
            if raw_event.get("error"):
                event_errors.append(str(raw_event["error"]))
            if raw_event.get("incomplete"):
                incomplete = True
            duration_ms = raw_event.get("duration_ms")
            if isinstance(duration_ms, int):
                durations.append(duration_ms)

        first_event = event_data[0]
        if isinstance(first_event, dict):
            event["exit_code"] = first_event.get("exit_code")
            event["duration_ms"] = first_event.get("duration_ms")
            event["event_error"] = first_event.get("error")
            event["incomplete"] = first_event.get("incomplete")
            event["stdout"] = trim_text(first_event.get("stdout_ref"), snippet_chars)
            event["stderr"] = trim_text(first_event.get("stderr_ref"), snippet_chars)

        if exit_codes:
            event["exit_codes"] = exit_codes
        if event_errors:
            event["event_errors"] = event_errors
        if durations:
            event["duration_ms_max"] = max(durations)
        if incomplete:
            event["has_incomplete_event"] = True

    return {key: value for key, value in event.items() if value is not None}


def is_problem_trace_event(event: dict[str, Any]) -> bool:
    if event.get("success") is False:
        return True
    if event.get("status") not in (None, "ok"):
        return True

    exit_codes = event.get("exit_codes")
    if isinstance(exit_codes, list) and any(code != 0 for code in exit_codes):
        return True
    exit_code = event.get("exit_code")
    if isinstance(exit_code, int) and exit_code != 0:
        return True

    if event.get("event_error") or event.get("event_errors"):
        return True
    if event.get("has_incomplete_event"):
        return True

    haystack = "\n".join(
        str(event.get(key, ""))
        for key in (
            "command",
            "stdout",
            "stderr",
            "error",
            "event_error",
            "event_errors",
            "observation_output",
        )
    ).lower()
    return any(
        marker in haystack
        for marker in (
            "no output",
            "exec-proxy connection lost",
            "failed to send non-blocking keys",
            "tmux is not installed",
        )
    )


def classify_root_cause_text(
    message: str | None,
    exception_type: str | None = None,
    extra_text: str | None = None,
) -> dict[str, str | None]:
    text = message or ""
    haystack = f"{text}\n{extra_text or ''}"
    if exception_type == "AgentTimeoutError":
        return {
            "category": "agent_execution_timeout",
            "phase": "agent_execution",
            "summary": text.splitlines()[0] if text else "Agent execution timed out",
        }
    if exception_type == "RewardFileNotFoundError":
        return {
            "category": "reward_file_missing",
            "phase": "verifier",
            "summary": text.splitlines()[0]
            if text
            else "Verifier reward file was not found",
        }
    if exception_type == "DownloadVerifierDirError":
        return {
            "category": "verifier_download_failed",
            "phase": "verifier",
            "summary": text.splitlines()[0]
            if text
            else "Verifier directory download failed",
        }
    if exception_type == "HTTPStatusError":
        return {
            "category": "http_status_error",
            "phase": "environment_setup",
            "summary": text.splitlines()[0]
            if text
            else "HTTPStatusError without message",
        }
    if exception_type == "InternalServerError":
        return {
            "category": "internal_server_error",
            "phase": "environment_setup",
            "summary": text.splitlines()[0]
            if text
            else "InternalServerError without message",
        }
    if exception_type == "RLIExecProxyConnectionLostError":
        return {
            "category": "exec_proxy_connection_lost",
            "phase": "agent_execution",
            "summary": "RLI exec-proxy connection was lost while running an agent command",
        }
    if exception_type == "AgentSetupTimeoutError":
        return {
            "category": "agent_setup_timeout",
            "phase": "agent_setup",
            "summary": text.splitlines()[0] if text else "Agent setup timed out",
        }
    if "Failed to obtain R2DBC Connection" in haystack or "DATABASE_ERROR" in haystack:
        return {
            "category": "rli_database_error",
            "phase": "agent_execution",
            "summary": "RLI backend could not reach its database (DATABASE_ERROR)",
        }
    if "Mini-swe-agent trajectory file" in haystack and "does not exist" in haystack:
        return {
            "category": "agent_child_exit_no_trajectory",
            "phase": "agent_execution",
            "summary": "mini-swe-agent exited before writing trajectory.json",
        }
    if "mini-swe-agent child process exited with code" in text:
        return {
            "category": "agent_child_process_failed",
            "phase": "agent_execution",
            "summary": text.splitlines()[0],
        }
    if "Failed probe warmup command" in haystack:
        return {
            "category": "probe_warmup_timeout",
            "phase": "environment_setup",
            "summary": "probe warmup command timed out before the environment became usable",
        }
    if "tmux is not installed" in haystack:
        return {
            "category": "tmux_missing",
            "phase": "agent_setup",
            "summary": "Terminus agent setup could not find/install tmux in the RLI session",
        }
    if "failed to send non-blocking keys" in haystack:
        return {
            "category": "tmux_send_keys_timeout",
            "phase": "agent_execution",
            "summary": "tmux send-keys command timed out while sending agent input",
        }
    return {
        "category": "runtime_error",
        "phase": None,
        "summary": text or "RuntimeError without message",
    }


UNDERLYING_CAUSE_ROOT_CAUSE = {
    "registry_rate_limited_429": {
        "category": "registry_rate_limited",
        "phase": "environment_setup",
        "summary": (
            "Docker registry (Nexus) returned 429 Too Many Requests while pulling "
            "the environment image; RLI surfaced it as POD_NEVER_REACHED / 503"
        ),
    },
    "pod_never_reached": {
        "category": "pod_never_reached",
        "phase": "environment_setup",
        "summary": "RLI could not reach the environment pod (POD_NEVER_REACHED)",
    },
    "rli_database_error": {
        "category": "rli_database_error",
        "phase": "agent_execution",
        "summary": "RLI backend could not reach its database (DATABASE_ERROR)",
    },
    "rli_gateway_unavailable": {
        "category": "rli_gateway_unavailable",
        "phase": "environment_setup",
        "summary": "RLI gateway returned 503 with no server available to handle the request",
    },
    "rli_internal_server_error": {
        "category": "rli_internal_server_error",
        "phase": "environment_setup",
        "summary": "RLI gateway returned 500 Internal Server Error",
    },
}

INFRA_EXCEPTION_TYPES = (
    "HTTPStatusError",
    "InternalServerError",
    "RuntimeError",
)


def root_cause_from_trace_events(report: "TrialReport") -> dict[str, str | None] | None:
    events = report.failed_trace_events or report.problem_trace_events
    causes = Counter(
        event["underlying_cause"] for event in events if event.get("underlying_cause")
    )
    if not causes:
        return None
    underlying_cause, _ = causes.most_common(1)[0]
    root_cause = UNDERLYING_CAUSE_ROOT_CAUSE.get(underlying_cause)
    if root_cause:
        return dict(root_cause)
    return {
        "category": underlying_cause,
        "phase": None,
        "summary": f"RLI reported {underlying_cause}",
    }


def classify_root_cause(
    report: "TrialReport",
) -> dict[str, str | None]:
    if report.exception_type in INFRA_EXCEPTION_TYPES:
        trace_root_cause = root_cause_from_trace_events(report)
        if trace_root_cause:
            return trace_root_cause
    return classify_root_cause_text(
        report.exception_message,
        report.exception_type,
        "\n".join(
            value for value in (report.exception_text, report.trial_log_tail) if value
        ),
    )


def nearest_event(
    events: list[dict[str, Any]],
    timestamp: str | None,
) -> dict[str, Any] | None:
    target = parse_ts(timestamp)
    if not events:
        return None
    if target is None:
        return events[0]

    def distance(event: dict[str, Any]) -> float:
        event_ts = parse_ts(event.get("ts"))
        if event_ts is None:
            return float("inf")
        return abs((event_ts - target).total_seconds())

    return min(events, key=distance)


def compact_trace_event_for_summary(
    event: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not event:
        return None
    return {
        key: value
        for key, value in {
            "ts": event.get("ts"),
            "stage": event.get("stage"),
            "request_id": event.get("request_id"),
            "command": trim_text(event.get("command"), 500),
            "exit_code": event.get("exit_code"),
            "duration_ms": event.get("duration_ms"),
            "error": trace_error_for_summary(event),
            "incomplete": event.get("incomplete") or event.get("has_incomplete_event"),
        }.items()
        if value is not None
    }


def session_relation_to_trial(
    report: TrialReport,
    first_event: dict[str, Any] | None,
    last_event: dict[str, Any] | None,
) -> str:
    trial_started_at = parse_ts(report.started_at)
    trial_finished_at = parse_ts(report.finished_at)
    first_ts = parse_ts(first_event.get("ts") if first_event else None)
    last_ts = parse_ts(last_event.get("ts") if last_event else None)

    if not trial_started_at or not first_ts or not last_ts:
        return "unknown"
    if trial_finished_at is None:
        trial_finished_at = trial_started_at + timedelta(days=365)
    if last_ts < trial_started_at:
        return "before_trial_window"
    if first_ts > trial_finished_at:
        return "after_trial_window"
    return "overlaps_trial_window"


def phase_for_session_event(
    event: dict[str, Any] | None,
    is_exception_session: bool,
    task_root_cause: dict[str, str | None],
) -> str | None:
    if is_exception_session:
        return task_root_cause.get("phase")
    if not event:
        return None

    command = str(event.get("command") or "").lower()
    if "tmux send-keys" in command or "tmux capture-pane" in command:
        return "agent_execution"
    if "tmux has-session" in command:
        return "agent_execution"
    if "tmux -v" in command or "command -v apt-get" in command:
        return "agent_setup"
    if "apt-get update" in command or "apt-get install" in command:
        return "agent_setup"
    if "pwd" in command:
        return "environment_setup"
    if "apply_patch" in command:
        return "environment_setup"
    if "mkdir -p /logs" in command:
        return "environment_setup"
    if "astral.sh/uv" in command:
        return "environment_setup"
    if "tar -c" in command and "/logs" in command:
        return "artifact_collection"
    return None


def root_cause_for_session_event(
    event: dict[str, Any] | None,
    is_exception_session: bool,
    task_root_cause: dict[str, str | None],
) -> dict[str, str | None]:
    if is_exception_session:
        return task_root_cause
    if not event:
        return {
            "category": "no_trace_problem",
            "phase": None,
            "summary": "no problem trace event was found for this session",
        }

    error = str(trace_error_for_summary(event) or "").lower()
    exit_code = event.get("exit_code")
    phase = phase_for_session_event(event, False, task_root_cause)

    if "exec-proxy connection lost" in error:
        return {
            "category": "exec_proxy_connection_lost",
            "phase": phase,
            "summary": "exec-proxy connection was lost while running the command",
        }
    if isinstance(exit_code, int) and exit_code == 124:
        return {
            "category": "command_timeout",
            "phase": phase,
            "summary": "command timed out in the RLI session",
        }
    if event.get("incomplete") or event.get("has_incomplete_event"):
        return {
            "category": "incomplete_trace_event",
            "phase": phase,
            "summary": "RLI returned an incomplete command event",
        }
    if isinstance(exit_code, int) and exit_code != 0:
        return {
            "category": "nonzero_exit",
            "phase": phase,
            "summary": "command returned a non-zero exit code",
        }
    return {
        "category": "rli_trace_problem",
        "phase": phase,
        "summary": "RLI trace event was classified as problematic",
    }


def is_cleanup_problem_event(event: dict[str, Any]) -> bool:
    command = str(event.get("command") or "")
    return "tar -C /logs" in command or "base64 -w0" in command


def select_session_breaking_event(
    report: TrialReport,
    session_id: str,
    root_cause_trace: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if root_cause_trace and root_cause_trace.get("session_id") == session_id:
        return root_cause_trace

    problem_events = report.session_problem_trace_events.get(session_id, [])
    for event in problem_events:
        if not is_cleanup_problem_event(event):
            return event
    return problem_events[0] if problem_events else None


def session_summary_entry(
    report: TrialReport,
    session_id: str,
    root_cause_trace: dict[str, Any] | None,
) -> dict[str, Any]:
    problem_events = report.session_problem_trace_events.get(session_id, [])
    breaking_event = select_session_breaking_event(report, session_id, root_cause_trace)
    is_root_cause_session = bool(
        root_cause_trace and root_cause_trace.get("session_id") == session_id
    )
    root_cause = root_cause_for_session_event(
        breaking_event,
        is_root_cause_session,
        classify_root_cause(report),
    )
    error_text = trace_error_for_summary(breaking_event)
    is_no_trace_problem = root_cause.get("category") == "no_trace_problem"
    payload = {
        "session_id": session_id,
        "is_root_cause_session": is_root_cause_session,
        "created_at": format_report_ts(report.session_created_at.get(session_id)),
        "occurred_at": format_report_ts(
            report.exception_occurred_at
            if is_root_cause_session
            else (breaking_event.get("ts") if breaking_event else None)
        ),
        "closed_at": format_report_ts(report.session_closed_at.get(session_id)),
        "failure": None if is_no_trace_problem else root_cause.get("category"),
        "failure_summary": None if is_no_trace_problem else root_cause.get("summary"),
        "command": trim_text(breaking_event.get("command"), 1_200)
        if breaking_event
        else None,
        "exit_code": breaking_event.get("exit_code") if breaking_event else None,
        "error": trim_text(error_text, 2_000),
        "trace_ts": breaking_event.get("ts") if breaking_event else None,
        "trace_request_id": breaking_event.get("request_id")
        if breaking_event
        else None,
        "incomplete": (
            breaking_event.get("incomplete")
            or breaking_event.get("has_incomplete_event")
        )
        if breaking_event
        else None,
        "problem_trace_event_count": len(problem_events),
        "trace_event_count": report.session_trace_event_counts.get(session_id, 0),
    }
    return {key: value for key, value in payload.items() if value is not None}


def has_clear_session_error(session: dict[str, Any]) -> bool:
    error = session.get("error")
    return bool(error) and not is_shell_prompt_text(error)


def build_session_summaries(
    report: TrialReport,
    root_cause_trace: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    sessions = [
        session_summary_entry(report, session_id, root_cause_trace)
        for session_id in sorted(report.session_ids)
    ]
    sessions = [session for session in sessions if has_clear_session_error(session)]
    return sorted(
        sessions,
        key=lambda session: (
            session.get("created_at") or session.get("trace_ts") or "",
            session.get("session_id") or "",
        ),
    )


def trace_event_search_text(event: dict[str, Any]) -> str:
    return "\n".join(
        str(event.get(key, ""))
        for key in (
            "command",
            "stdout",
            "stderr",
            "error",
            "event_error",
            "event_errors",
            "observation_output",
        )
    ).lower()


def nearest_matching_event(
    events: list[dict[str, Any]],
    timestamp: str | None,
    marker_groups: list[tuple[str, ...]],
) -> dict[str, Any] | None:
    for markers in marker_groups:
        normalized_markers = tuple(marker.lower() for marker in markers)
        matches = [
            event
            for event in events
            if all(
                marker in trace_event_search_text(event)
                for marker in normalized_markers
            )
        ]
        if matches:
            return nearest_event(matches, timestamp)
    return None


def select_root_cause_trace_event(report: TrialReport) -> dict[str, Any] | None:
    if not report.problem_trace_events:
        return None

    root_cause = classify_root_cause(report)
    category = root_cause.get("category")
    if category == "exec_proxy_connection_lost":
        markers = [("exec-proxy connection lost",)]
    elif category == "tmux_send_keys_timeout":
        markers = [("tmux send-keys",)]
    elif category == "probe_warmup_timeout":
        markers = [(" pwd",), ("&& pwd",), ("pwd",)]
    elif category == "tmux_missing":
        markers = [
            ("tmux -v",),
            ("apt-get install", "tmux"),
            ("apt-get update",),
            ("command -v apt-get",),
        ]
    else:
        markers = []

    matched = nearest_matching_event(
        report.problem_trace_events,
        report.exception_occurred_at,
        markers,
    )
    return matched or nearest_event(
        report.problem_trace_events, report.exception_occurred_at
    )


def correlate_rli_trace(
    rli_trace_path: Path,
    reports: dict[str, TrialReport],
    snippet_chars: int,
    last_events: int,
) -> None:
    task_to_trials: dict[str, list[TrialReport]] = defaultdict(list)
    session_to_trials: dict[str, list[TrialReport]] = defaultdict(list)
    target_task_ids = {report.task_id for report in reports.values()}

    for report in reports.values():
        report.last_trace_events = deque(maxlen=last_events)
        task_to_trials[report.task_id].append(report)

    with rli_trace_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on {rli_trace_path}:{line_no}: {exc}"
                ) from exc

            event_ts = parse_ts(item.get("ts"))
            create_task_id = task_id_from_create_item(item)
            session_id = session_id_from_trace_item(item)

            if create_task_id in target_task_ids and session_id:
                matched_reports = assign_trial_for_event(
                    create_task_id,
                    event_ts,
                    task_to_trials,
                )
                for report in matched_reports:
                    report.session_ids.add(session_id)
                    session_to_trials[session_id].append(report)

            matched_reports = (
                session_to_trials.get(session_id, []) if session_id else []
            )
            if not matched_reports:
                if (
                    not session_id
                    and item.get("success") is False
                    and create_task_id in target_task_ids
                ):
                    matched_reports = assign_trial_for_event(
                        create_task_id,
                        event_ts,
                        task_to_trials,
                    )
                if not matched_reports:
                    continue

            summary = summarize_trace_item(item, snippet_chars)
            for report in matched_reports:
                stage = item.get("stage") or "unknown"
                summary_session_id = summary.get("session_id")
                report.trace_event_count += 1
                report.trace_stage_counts[stage] += 1
                report.last_trace_events.append(summary)

                if isinstance(summary_session_id, str):
                    report.session_trace_event_counts[summary_session_id] += 1
                    report.session_stage_counts[summary_session_id][stage] += 1
                    report.session_first_trace_events.setdefault(
                        summary_session_id, summary
                    )
                    report.session_last_trace_events[summary_session_id] = summary
                    if stage == "create_session":
                        report.session_created_at.setdefault(
                            summary_session_id,
                            summary.get("ts"),
                        )
                    elif stage == "close_session":
                        report.session_closed_at[summary_session_id] = summary.get("ts")

                if item.get("success") is False:
                    report.failed_trace_events.append(summary)
                if is_problem_trace_event(summary):
                    report.problem_trace_events.append(summary)
                    if isinstance(summary_session_id, str):
                        report.session_problem_trace_events[summary_session_id].append(
                            summary
                        )


def report_to_dict(report: TrialReport) -> dict[str, Any]:
    root_cause = classify_root_cause(report)
    first_problem = (
        report.problem_trace_events[0] if report.problem_trace_events else None
    )
    nearest_problem = nearest_event(
        report.problem_trace_events, report.exception_occurred_at
    )
    root_cause_trace = select_root_cause_trace_event(report)
    sessions = build_session_summaries(report, root_cause_trace)
    return {
        "trial_name": report.trial_name,
        "task_id": report.task_id,
        "trial_dir": str(report.trial_dir),
        "exception_type": report.exception_type,
        "exception_message": report.exception_message,
        "exception_occurred_at": report.exception_occurred_at,
        "exception_traceback": report.exception_traceback,
        "root_cause": root_cause,
        "started_at": report.started_at,
        "finished_at": report.finished_at,
        "environment_setup": {
            "started_at": report.environment_setup_started_at,
            "finished_at": report.environment_setup_finished_at,
        },
        "agent_setup": {
            "started_at": report.agent_setup_started_at,
            "finished_at": report.agent_setup_finished_at,
        },
        "session_ids": sorted(report.session_ids),
        "sessions": sessions,
        "trace_event_count": report.trace_event_count,
        "trace_stage_counts": dict(sorted(report.trace_stage_counts.items())),
        "problem_trace_event_count": len(report.problem_trace_events),
        "first_problem_trace_event": first_problem,
        "nearest_problem_trace_event": nearest_problem,
        "root_cause_trace_event": root_cause_trace,
        "problem_trace_events": report.problem_trace_events,
        "failed_trace_events": report.failed_trace_events,
        "last_trace_events": list(report.last_trace_events),
        "exception_text": report.exception_text,
        "trial_log_tail": report.trial_log_tail,
    }


def normalize_exception_message(message: str | None) -> str:
    if not message:
        return "Unknown"
    first_line = message.splitlines()[0]
    prefix = "RLI exec-proxy connection lost while executing an agent command:"
    if first_line.startswith(prefix):
        return prefix
    return first_line


def build_report(
    metrics_dir: Path,
    run_dir: Path,
    rli_trace_path: Path,
    exception_types: list[str],
    reports: dict[str, TrialReport],
) -> dict[str, Any]:
    tasks = [report_to_dict(report) for report in reports.values()]
    message_counts = Counter(
        normalize_exception_message(report.exception_message)
        for report in reports.values()
    )
    category_counts = Counter(task["root_cause"]["category"] for task in tasks)
    exception_type_counts = Counter(
        report.exception_type for report in reports.values()
    )
    return {
        "metrics_dir": str(metrics_dir),
        "run_dir": str(run_dir),
        "rli_trace": str(rli_trace_path),
        "exception_type": exception_types[0]
        if len(exception_types) == 1
        else "multiple",
        "exception_types": exception_types,
        "exception_type_counts": dict(exception_type_counts.most_common()),
        "total": len(tasks),
        "message_counts": dict(message_counts.most_common()),
        "root_cause_counts": dict(category_counts.most_common()),
        "tasks": tasks,
    }


def safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def is_shell_prompt_text(value: Any) -> bool:
    text = str(value or "").strip()
    return text.startswith("(") and "root@rli-" in text and text.endswith("#")


def trace_error_for_summary(event: dict[str, Any] | None) -> str | None:
    if not event:
        return None

    rli_error_message = event.get("rli_error_message")
    if rli_error_message:
        code = event.get("rli_error_code")
        return f"{code}: {rli_error_message}" if code else str(rli_error_message)

    event_errors = event.get("event_errors")
    if isinstance(event_errors, list) and event_errors:
        useful_errors = [
            str(error) for error in event_errors if not is_shell_prompt_text(error)
        ]
        if useful_errors:
            return "; ".join(useful_errors)

    response_error = event.get("error")
    if response_error and not is_shell_prompt_text(response_error):
        return str(response_error)

    event_error = event.get("event_error")
    if event_error and not is_shell_prompt_text(event_error):
        return str(event_error)

    observation_output = event.get("observation_output")
    if observation_output:
        return str(observation_output)

    stderr = event.get("stderr")
    if stderr and not is_shell_prompt_text(stderr):
        return str(stderr)

    stdout = event.get("stdout")
    if stdout:
        return str(stdout)

    if event_error:
        return str(event_error)
    if response_error:
        return str(response_error)
    return None


def trace_event_for_summary(task: dict[str, Any]) -> dict[str, Any] | None:
    return (
        task.get("root_cause_trace_event")
        or task.get("nearest_problem_trace_event")
        or task.get("first_problem_trace_event")
    )


def task_summary_entry(
    task: dict[str, Any],
    output_dir: Path,
    traces_dir: Path,
) -> dict[str, Any]:
    stem = safe_filename(task["task_id"])
    return {
        "task_id": task["task_id"],
        "trial_name": task["trial_name"],
        "exception_type": task["exception_type"],
        "exception_message": task.get("exception_message"),
        "exception_occurred_at": task.get("exception_occurred_at"),
        "root_cause": task.get("root_cause"),
        "session_ids": task.get("session_ids", []),
        "sessions": task.get("sessions", []),
        "report_path": str(output_dir / f"{stem}.json"),
        "problem_trace_path": str(traces_dir / f"{stem}.problem_events.jsonl"),
        "last_trace_path": str(traces_dir / f"{stem}.last_events.jsonl"),
    }


def task_summary_sort_key(task: dict[str, Any]) -> tuple[str, str]:
    sessions = task.get("sessions") or []
    created_values = [
        sortable_report_ts(session.get("created_at"))
        for session in sessions
        if session.get("created_at")
    ]
    return (
        min(created_values) if created_values else "",
        task.get("task_id") or "",
    )


def summary_report_for_output(
    report: dict[str, Any],
    output_dir: Path,
    traces_dir: Path,
) -> dict[str, Any]:
    task_summaries = [
        task_summary_entry(task, output_dir, traces_dir)
        for task in report["tasks"]
        if task.get("sessions") or task.get("failed_trace_events")
    ]
    return {
        "metrics_dir": report["metrics_dir"],
        "run_dir": report["run_dir"],
        "rli_trace": report["rli_trace"],
        "exception_type": report["exception_type"],
        "exception_types": report.get("exception_types", [report["exception_type"]]),
        "exception_type_counts": report.get("exception_type_counts", {}),
        "total": report["total"],
        "root_cause_counts": report["root_cause_counts"],
        "message_counts": report["message_counts"],
        "tasks": sorted(task_summaries, key=task_summary_sort_key),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for event in events:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")


def write_reports(
    output_dir: Path,
    traces_dir: Path,
    summary_path: Path,
    report: dict[str, Any],
) -> tuple[list[Path], list[Path]]:
    summary = summary_report_for_output(report, output_dir, traces_dir)
    write_json(summary_path, summary)
    report_paths: list[Path] = []
    trace_paths: list[Path] = []

    base_metadata = {
        "metrics_dir": report["metrics_dir"],
        "run_dir": report["run_dir"],
        "rli_trace": report["rli_trace"],
        "exception_type": report["exception_type"],
        "exception_types": report.get("exception_types", [report["exception_type"]]),
        "exception_type_counts": report.get("exception_type_counts", {}),
    }

    for task in report["tasks"]:
        stem = safe_filename(task["task_id"])
        task_report_path = output_dir / f"{stem}.json"
        write_json(task_report_path, {**base_metadata, "task": task})
        report_paths.append(task_report_path)

        problem_path = traces_dir / f"{stem}.problem_events.jsonl"
        last_path = traces_dir / f"{stem}.last_events.jsonl"
        write_jsonl(problem_path, task["problem_trace_events"])
        write_jsonl(last_path, task["last_trace_events"])
        trace_paths.extend([problem_path, last_path])

    return report_paths, trace_paths


def command_preview(event: dict[str, Any] | None) -> str:
    if not event:
        return "-"
    command = event.get("command") or ""
    first_line = str(command).splitlines()[0] if command else "-"
    if len(first_line) > 140:
        return first_line[:137] + "..."
    return first_line


def print_summary(report: dict[str, Any], max_print_trials: int | None = None) -> None:
    print(f"metrics_dir: {report['metrics_dir']}")
    print(f"run_dir: {report['run_dir']}")
    print(f"rli_trace: {report['rli_trace']}")
    print(
        "exception_types: "
        + ", ".join(report.get("exception_types", [report["exception_type"]]))
    )
    print(f"trials: {report['total']}")
    print()

    print("Exception Types:")
    for exception_type, count in report.get("exception_type_counts", {}).items():
        print(f"  {count}x {exception_type}")
    print()

    print("Root Causes:")
    for category, count in report["root_cause_counts"].items():
        print(f"  {count}x {category}")
    print()

    print("Messages:")
    for message, count in report["message_counts"].items():
        print(f"  {count}x {message}")
    print()

    print("Trials:")
    tasks_to_print = report["tasks"]
    if max_print_trials is not None:
        tasks_to_print = tasks_to_print[:max_print_trials]
    for task in tasks_to_print:
        sessions = ", ".join(task["session_ids"]) or "-"
        stage_counts = (
            ", ".join(
                f"{stage}={count}"
                for stage, count in task["trace_stage_counts"].items()
            )
            or "-"
        )
        nearest_problem = task["nearest_problem_trace_event"]
        root_cause_problem = task.get("root_cause_trace_event")
        root_cause = task["root_cause"]
        print(f"  - {task['trial_name']} [{task['exception_type']}]")
        print(f"    occurred_at: {task['exception_occurred_at']}")
        print(f"    phase: {root_cause.get('phase')}")
        print(f"    cause: {root_cause.get('category')} - {root_cause.get('summary')}")
        print(f"    message: {task['exception_message']}")
        print(f"    sessions: {sessions}")
        print(
            "    trace_events: "
            f"{task['trace_event_count']} ({stage_counts}); "
            f"problem_events={task['problem_trace_event_count']}"
        )
        if root_cause_problem:
            print(
                "    root_cause_trace: "
                f"ts={root_cause_problem.get('ts')} "
                f"stage={root_cause_problem.get('stage')} "
                f"exit_code={root_cause_problem.get('exit_code')} "
                f"error={trace_error_for_summary(root_cause_problem)}"
            )
            print(f"    root_cause_command: {command_preview(root_cause_problem)}")
        if nearest_problem and nearest_problem != root_cause_problem:
            print(
                "    nearest_problem_trace: "
                f"ts={nearest_problem.get('ts')} "
                f"stage={nearest_problem.get('stage')} "
                f"exit_code={nearest_problem.get('exit_code')} "
                f"error={trace_error_for_summary(nearest_problem)}"
            )
            print(f"    nearest_problem_command: {command_preview(nearest_problem)}")
    if len(tasks_to_print) < len(report["tasks"]):
        remaining = len(report["tasks"]) - len(tasks_to_print)
        print(f"  ... {remaining} more trials omitted from stdout")


def main() -> None:
    args = parse_args()
    metrics_dir = args.metrics_dir
    run_dir = discover_run_dir(metrics_dir, args.run_dir)
    rli_trace_path = discover_rli_trace(metrics_dir, run_dir, args.rli_trace)

    exception_stats = exception_stats_for_eval(run_dir, args.eval_key)
    exception_types = resolve_exception_types(exception_stats, args.exception_type)
    reports = load_target_trials(
        run_dir=run_dir,
        exception_stats=exception_stats,
        exception_types=exception_types,
        snippet_chars=args.snippet_chars,
    )
    correlate_rli_trace(
        rli_trace_path=rli_trace_path,
        reports=reports,
        snippet_chars=args.snippet_chars,
        last_events=args.last_events,
    )

    report = build_report(
        metrics_dir=metrics_dir,
        run_dir=run_dir,
        rli_trace_path=rli_trace_path,
        exception_types=exception_types,
        reports=reports,
    )

    print_summary(report, max_print_trials=args.max_print_trials)

    if args.no_write:
        return

    output_dir = args.output_dir
    traces_dir = args.dump_traces_dir or output_dir / "traces"
    summary_path = args.summary_output or output_dir / "_summary.json"
    report_paths, trace_paths = write_reports(
        output_dir, traces_dir, summary_path, report
    )
    print()
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote task reports: {len(report_paths)} files under {output_dir}")
    print(f"Wrote trace snippets: {len(trace_paths)} files under {traces_dir}")


if __name__ == "__main__":
    main()
