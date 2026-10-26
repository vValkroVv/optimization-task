from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from job_submitter.core.job_templates.region_configs import (
    TRACKING_AUTO_TRACK_ERRORS_COMMAND_TYPE,
    TRACKING_DAEMON_INBOX_S3_KEY_MAP,
    TRACKING_DAEMON_INBOX_S3_KEY_TEMPLATE,
    TRACKING_LAST_CHECKED_RETRY_DEFAULT,
    TRACKING_WORKSPACE_MAP,
)
from job_submitter.core.verl_tracking import resolve_verl_logs_path

logger = logging.getLogger(__name__)

SUBMISSION_TYPE_CLOUD = "cloud"
SUBMISSION_TYPE_SLURM = "slurm"


def _get_nested(data: Any, dotted_path: str, default: Any = None) -> Any:
    """Traverse a dict by dot-separated key path."""
    value = data
    for key in dotted_path.split("."):
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return value if value is not None else default


def _resolve_tracking_config(config: dict) -> dict:
    """Build a plain dict with OmegaConf interpolation resolved from the config root."""
    from omegaconf import OmegaConf
    from omegaconf.errors import InterpolationKeyError, InterpolationResolutionError

    try:
        cfg = OmegaConf.create(config)
        return OmegaConf.to_container(cfg, resolve=True)
    except (InterpolationKeyError, InterpolationResolutionError) as exc:
        raise ValueError(f"Config interpolation error: {exc}") from exc


def extract_logs_path(config: dict, run_name: Optional[str] = None) -> str:
    """Resolve config with OmegaConf and derive the TensorBoard log path."""
    resolved = _resolve_tracking_config(config)

    nested_dir = _get_nested(resolved, "loggers.tensorboard.log_dir")
    top_path = _get_nested(resolved, "tensorboard_path")
    run = run_name or _get_nested(resolved, "run_name", "llm")

    if nested_dir and str(nested_dir).strip():
        path = f"{str(nested_dir).strip()}/{run}".replace("\\", "/")
    elif top_path and str(top_path).strip():
        path = str(top_path).strip().replace("\\", "/")
    else:
        raise ValueError(
            "Config missing tensorboard log path: set 'loggers.tensorboard.log_dir' "
            "or top-level 'tensorboard_path'"
        )

    if len(path) > 2 and path[1] == ":":
        path = path[2:]
    return path


def extract_logs_path_from_file(
    config_path: str, run_name: Optional[str] = None
) -> str:
    """Load YAML or VERL shell config and extract the monitoring path."""
    path = Path(config_path)
    if path.suffix.lower() == ".sh":
        return resolve_verl_logs_path(path)

    with open(config_path, "r", encoding="utf-8") as handle:
        return extract_logs_path(yaml.safe_load(handle), run_name)


def _strip_cluster_prefix(logs_path: str, workspace: str, cluster: str) -> str:
    """Remove ``workspace:cluster:`` prefix from logs path if present."""
    own_prefix = f"{workspace}:{cluster}:"
    while logs_path.startswith(own_prefix):
        logs_path = logs_path[len(own_prefix) :]

    if ":" in logs_path:
        parts = logs_path.split(":", 2)
        if len(parts) == 3 and parts[0] and parts[1]:
            logger.warning("Stripping foreign cluster prefix: %s", logs_path)
            return parts[2]

    return logs_path


def _resolve_daemon_inbox_s3_key(region: str, workspace: str) -> str:
    if workspace != TRACKING_WORKSPACE_MAP.get(region):
        return TRACKING_DAEMON_INBOX_S3_KEY_TEMPLATE.format(
            workspace=workspace,
            cluster=region.lower(),
        )
    return TRACKING_DAEMON_INBOX_S3_KEY_MAP[region]


def _job_metadata(
    submission_type: Optional[str], job_id: Optional[str], job_title: Optional[str]
) -> dict[str, str]:
    """Job identity for the tracker's "started" card, known before the daemon
    resolves the job itself."""
    values = {
        "submission_type": (submission_type or "").strip().lower(),
        "job_id": (job_id or "").strip(),
        "job_title": (job_title or "").strip(),
    }
    return {key: value for key, value in values.items() if value}


def _build_auto_track_errors_command_data(
    *,
    submission_type: Optional[str],
    job_id: Optional[str],
    logs_path: str,
    job_title: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    if not submission_type:
        return None

    normalized = submission_type.strip().lower()
    if normalized not in {SUBMISSION_TYPE_CLOUD, SUBMISSION_TYPE_SLURM}:
        logger.warning(
            "Skipping '%s' command: unsupported submission_type=%s",
            TRACKING_AUTO_TRACK_ERRORS_COMMAND_TYPE,
            submission_type,
        )
        return None

    logs_info: dict[str, Any] = {
        "submission_type": normalized,
        "last_checked_retry": TRACKING_LAST_CHECKED_RETRY_DEFAULT,
    }
    normalized_job_id = (job_id or "").strip()
    if normalized_job_id:
        logs_info["job_id"] = normalized_job_id
    normalized_job_title = (job_title or "").strip()
    if normalized_job_title:
        logs_info["job_title"] = normalized_job_title

    if normalized == SUBMISSION_TYPE_CLOUD:
        logs_info["mlspace_logs_path"] = logs_path
    else:
        logs_info["slurm_log_path"] = logs_path

    return {"enabled": True, "logs_info": logs_info}


def auto_track(
    region: str,
    user_login: str,
    config_path: Optional[str],
    merged_config: Optional[dict] = None,
    env: Optional[dict[str, str]] = None,
    submission_type: Optional[str] = None,
    job_id: Optional[str] = None,
    script_path: Optional[str] = None,
    auto_track_users: Optional[list[str]] = None,
    enable_auto_share: bool = False,
    logs_path: Optional[str] = None,
    extra_metadata: Optional[dict[str, str | int | float | bool]] = None,
    job_title: Optional[str] = None,
) -> bool:
    """Register experiment for auto-tracking in daemon inbox SQLite on S3.

    ``job_title`` is the scheduler-visible job name (Slurm job name, cloud job
    description): the tracker's cards lead with it instead of the logs path.

    This is intentionally non-blocking for job submission: all errors are logged and
    converted to ``False`` so a broken tracker cannot prevent the actual job submit.
    """
    try:
        # Submitting a job must not require the S3 tracking dependencies
        # (sqlite_s3_query, boto3), which slim runtime images omit.
        from job_submitter.core.tracking_storage import (
            _resolve_s3_env,
            _should_skip_auto_track_due_missing_secret_file,
            _write_auto_track_request,
        )

        if _should_skip_auto_track_due_missing_secret_file(region, env):
            return False

        workspace = TRACKING_WORKSPACE_MAP.get(region)
        if not workspace:
            logger.warning("No tracking config for region '%s' - skipping", region)
            return False
        if (
            region in {"SR004", "SR008"}
            and (env or {}).get("MLS_NTB_WORKSPACE_NAME", "").strip().lower() == "tts"
        ):
            workspace = "gigatts"

        logs_path = _resolve_logs_path(
            workspace=workspace,
            region=region,
            config_path=config_path,
            merged_config=merged_config,
            logs_path=logs_path,
        )
        payload = _build_auto_track_payload(
            user_login=user_login,
            auto_track_users=auto_track_users,
            enable_auto_share=enable_auto_share,
            config_path=config_path,
            logs_path=logs_path,
            script_path=script_path,
            extra_metadata={
                **_job_metadata(submission_type, job_id, job_title),
                **(extra_metadata or {}),
            },
        )
        command_data = _build_auto_track_errors_command_data(
            submission_type=submission_type,
            job_id=job_id,
            logs_path=logs_path,
            job_title=job_title,
        )
        return _write_auto_track_request(
            region=region,
            s3_key=_resolve_daemon_inbox_s3_key(region, workspace),
            payload=payload,
            s3_env=_resolve_s3_env(region, env),
            command_data=command_data,
        )

    except Exception:
        logger.exception(
            "Auto-tracking failed (non-blocking): region=%s user=%s config=%s "
            "submission_type=%s job_id=%s",
            region,
            user_login,
            config_path,
            submission_type,
            job_id,
        )
        return False


def _resolve_logs_path(
    *,
    workspace: str,
    region: str,
    config_path: Optional[str],
    merged_config: Optional[dict],
    logs_path: Optional[str] = None,
) -> str:
    if logs_path and logs_path.strip():
        resolved_logs_path = logs_path.strip()
    elif merged_config:
        resolved_logs_path = extract_logs_path(merged_config)
    elif config_path:
        resolved_logs_path = extract_logs_path_from_file(config_path)
    else:
        raise ValueError("Auto-tracking requires config_path or logs_path")
    return _strip_cluster_prefix(resolved_logs_path, workspace, region.lower())


def _sanitize_extra_metadata(
    extra_metadata: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Drop values that cannot be JSON-serialized, logging a warning for each."""
    if not extra_metadata:
        return {}

    sanitized: dict[str, Any] = {}
    for key, value in extra_metadata.items():
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            logger.warning(
                "Dropping non-JSON-serializable auto-track metadata key '%s'", key
            )
            continue
        sanitized[key] = value
    return sanitized


def _build_auto_track_payload(
    *,
    user_login: str,
    auto_track_users: Optional[list[str]],
    enable_auto_share: bool,
    config_path: Optional[str],
    logs_path: str,
    script_path: Optional[str],
    extra_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    resolved_script_path = script_path
    if not resolved_script_path and config_path and config_path.lower().endswith(".sh"):
        resolved_script_path = config_path

    users: list[str] = []
    seen_users: set[str] = set()
    for raw_username in [user_login, *(auto_track_users or [])]:
        username = str(raw_username or "").strip().lstrip("@")
        key = username.lower()
        if not username or key in seen_users:
            continue
        users.append(username)
        seen_users.add(key)

    return {
        "users": users,
        "logs_path": logs_path,
        "yaml_path": os.path.abspath(config_path) if config_path else None,
        "script_path": (
            os.path.abspath(resolved_script_path) if resolved_script_path else None
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            **_sanitize_extra_metadata(extra_metadata),
            "auto_share": bool(enable_auto_share),
        },
    }
