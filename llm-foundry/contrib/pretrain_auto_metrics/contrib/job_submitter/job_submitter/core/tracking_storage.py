from __future__ import annotations

import json
import logging
import os
import sqlite3
import tempfile
import threading
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from job_submitter.core.sqlite_s3_import import raise_sqlite_s3_query_import_error

try:
    from sqlite_s3_query import DatabaseType, ensure_schema
except ModuleNotFoundError as exc:
    raise_sqlite_s3_query_import_error(exc)

from job_submitter.core.job_templates.region_configs import (
    DEFAULT_REGION_ENVS,
    TRACKING_AUTO_TRACK_DB_LOCAL_DIR_ENV,
    TRACKING_AUTO_TRACK_DB_LOCAL_DIR_PREFIX,
    TRACKING_AUTO_TRACK_ERRORS_COMMAND_TYPE,
    TRACKING_DEFAULT_S3_BUCKET,
    TRACKING_DEFAULT_S3_ENDPOINT_URL,
    TRACKING_DEFAULT_S3_KEY_PREFIX,
    TRACKING_DEFAULT_S3_REGION,
    TRACKING_REQUIRED_SECRET_PATH_MARKER,
    TRACKING_S3_ENV_FILE_ENV,
    TRACKING_S3_FORCE_PATH_STYLE_ENV,
    TRACKING_S3_SECRET_ENV_FILE_KEY,
    TRACKING_S3_VERIFY_SSL_ENV,
)
from job_submitter.infrastructure.s3.bus import S3SQLiteBus, build_s3_client
from job_submitter.infrastructure.retry import RetryPolicy
from job_submitter.infrastructure.s3.config import S3StorageConfig
from job_submitter.infrastructure.s3.constants import S3_ADDRESSING_STYLE

_FALSE_FLAG_VALUES = frozenset({"0", "false", "no", "off", ""})
_S3_PATH_ADDRESSING_STYLE = "path"

logger = logging.getLogger(__name__)

S3_SECRET_ENV_FILE_OVERRIDE_KEYS = (
    TRACKING_S3_ENV_FILE_ENV,
    TRACKING_S3_SECRET_ENV_FILE_KEY,
)
AUTO_TRACK_REQUEST_INSERT_SQL = (
    "INSERT INTO auto_track_requests "
    "(users, logs_path, yaml_path, script_path, timestamp, metadata) "
    "VALUES (?, ?, ?, ?, ?, ?)"
)
AUTO_TRACK_COMMAND_INSERT_SQL = (
    "INSERT INTO commands "
    "(command_type, experiment_path, hash, timestamp, data) "
    "VALUES (?, ?, ?, ?, ?)"
)

_AUTO_TRACK_LOCKS_GUARD = threading.Lock()
_AUTO_TRACK_S3_LOCKS: dict[str, threading.Lock] = {}


@dataclass(slots=True, frozen=True)
class AutoTrackLocalDatabase:
    directory: Path
    path: Path


@dataclass(slots=True, frozen=True)
class AutoTrackInboxWriter:
    region: str
    s3_key: str
    payload: dict[str, Any]
    s3_env: dict[str, str]
    command_data: Optional[dict[str, Any]] = None

    def write(self) -> bool:
        storage_config = _build_s3_storage_config(self.s3_env, self.s3_key)
        local_db = _resolve_auto_track_local_database(self.s3_key)
        lock = _lock_for_s3_key(storage_config.key)
        bus = S3SQLiteBus(
            s3_client=build_s3_client(storage_config),
            config=storage_config,
        )

        with lock:
            remote_state = bus.download_or_initialize(local_db.path)
            _write_local_inbox_database(
                local_db_path=local_db.path,
                region=self.region,
                payload=self.payload,
                command_data=self.command_data,
            )
            bus.upload(local_db.path, expected_state=remote_state)

        logger.info(
            "Auto-track request uploaded: region=%s logs=%s inbox=%s downloaded=%s",
            self.region,
            self.payload["logs_path"],
            self.s3_key,
            remote_state is not None,
        )
        return True


def _resolve_auto_track_local_dir() -> Path:
    """Return the per-user local dir used for temporary auto-track SQLite files."""
    override = os.getenv(TRACKING_AUTO_TRACK_DB_LOCAL_DIR_ENV, "").strip()
    if override:
        local_dir = Path(os.path.expandvars(os.path.expanduser(override)))
    else:
        user_scope = os.getenv("USER") or os.getenv("USERNAME") or "unknown"
        local_dir = (
            Path(tempfile.gettempdir())
            / f"{TRACKING_AUTO_TRACK_DB_LOCAL_DIR_PREFIX}{user_scope}"
        )

    local_dir.mkdir(parents=True, exist_ok=True)
    if not local_dir.is_dir():
        raise NotADirectoryError(
            f"Auto-track local path is not a directory: {local_dir}"
        )
    return local_dir


def _resolve_auto_track_local_database(s3_key: str) -> AutoTrackLocalDatabase:
    local_dir = _resolve_auto_track_local_dir()
    return AutoTrackLocalDatabase(
        directory=local_dir,
        path=local_dir / _safe_local_sqlite_name(s3_key),
    )


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse S3 credentials file (``KEY=value`` lines) into ``dict[str, str]``."""
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        values[key] = value
    return values


def _resolve_secret_env_path(
    region: str, job_env: Optional[dict[str, str]] = None
) -> Optional[Path]:
    env_map = job_env or {}
    for key in S3_SECRET_ENV_FILE_OVERRIDE_KEYS:
        env_value = env_map.get(key)
        if env_value:
            return Path(os.path.expandvars(os.path.expanduser(env_value)))

    region_env = DEFAULT_REGION_ENVS.get(region, {})
    env_path_raw = region_env.get(TRACKING_S3_SECRET_ENV_FILE_KEY)
    if not env_path_raw:
        return None
    return Path(os.path.expandvars(os.path.expanduser(env_path_raw)))


def _resolve_s3_env(
    region: str, job_env: Optional[dict[str, str]] = None
) -> dict[str, str]:
    """Merge job env with optional S3 credential file; explicit job env wins."""
    base = dict(job_env or {})
    env_path = _resolve_secret_env_path(region, base)
    if not env_path or not env_path.exists():
        if env_path and not env_path.exists():
            logger.warning(
                "S3 credentials file not found for region '%s': %s", region, env_path
            )
        return base

    try:
        creds = _parse_env_file(env_path)
    except (OSError, TypeError, ValueError):
        logger.exception(
            "Failed to parse S3 credentials file for region '%s': %s",
            region,
            env_path,
        )
        return base

    return {**creds, **base}


def _should_skip_auto_track_due_missing_secret_file(
    region: str, job_env: Optional[dict[str, str]] = None
) -> bool:
    env_path = _resolve_secret_env_path(region, job_env)
    if not env_path:
        return False
    if env_path.exists():
        return False
    if TRACKING_REQUIRED_SECRET_PATH_MARKER not in env_path.as_posix().lower():
        return False
    logger.warning(
        "Auto-track skipped for region '%s': missing secret env file: %s",
        region,
        env_path,
    )
    return True


def _write_auto_track_request(
    *,
    region: str,
    s3_key: str,
    payload: dict[str, Any],
    s3_env: dict[str, str],
    command_data: Optional[dict[str, Any]] = None,
) -> bool:
    from job_submitter.infrastructure.s3.exceptions import ConcurrentWriteError

    writer = AutoTrackInboxWriter(
        region=region,
        s3_key=s3_key,
        payload=payload,
        s3_env=s3_env,
        command_data=command_data,
    )

    # The tracker daemon rewrites the same inbox object, so optimistic-
    # concurrency conflicts are expected; each retry re-downloads fresh state.
    return RetryPolicy(
        exceptions=(ConcurrentWriteError,),
        delay=lambda attempt, error: 0.5 * attempt,
    ).run(writer.write)


def _lock_for_s3_key(s3_key: str) -> threading.Lock:
    with _AUTO_TRACK_LOCKS_GUARD:
        lock = _AUTO_TRACK_S3_LOCKS.get(s3_key)
        if lock is None:
            lock = threading.Lock()
            _AUTO_TRACK_S3_LOCKS[s3_key] = lock
        return lock


def _safe_local_sqlite_name(s3_key: str) -> str:
    return s3_key.replace("/", "_").replace("\\", "_")


def _env_flag(raw: Optional[str], *, default: bool) -> bool:
    """Interpret a textual env flag (``true``/``false``/``1``/``0`` ...)."""
    if raw is None:
        return default
    return raw.strip().lower() not in _FALSE_FLAG_VALUES


def _build_s3_storage_config(s3_env: dict[str, str], s3_key: str) -> S3StorageConfig:
    endpoint = s3_env.get("S3_ENDPOINT_URL", TRACKING_DEFAULT_S3_ENDPOINT_URL)
    region = s3_env.get("AWS_REGION", TRACKING_DEFAULT_S3_REGION)
    bucket = s3_env.get("S3_BUCKET", TRACKING_DEFAULT_S3_BUCKET)
    key_prefix = s3_env.get("S3_KEY_PREFIX", TRACKING_DEFAULT_S3_KEY_PREFIX)
    full_key = f"{key_prefix}{s3_key}" if key_prefix else s3_key
    force_path_style = _env_flag(
        s3_env.get(TRACKING_S3_FORCE_PATH_STYLE_ENV), default=False
    )
    return S3StorageConfig(
        endpoint_url=endpoint,
        region=region,
        bucket=bucket,
        key=full_key,
        aws_access_key_id=s3_env.get("AWS_ACCESS_KEY_ID", ""),
        aws_secret_access_key=s3_env.get("AWS_SECRET_ACCESS_KEY", ""),
        aws_session_token=s3_env.get("AWS_SESSION_TOKEN", ""),
        verify_ssl=_env_flag(s3_env.get(TRACKING_S3_VERIFY_SSL_ENV), default=True),
        addressing_style=(
            _S3_PATH_ADDRESSING_STYLE if force_path_style else S3_ADDRESSING_STYLE
        ),
    )


def _write_local_inbox_database(
    *,
    local_db_path: Path,
    region: str,
    payload: dict[str, Any],
    command_data: Optional[dict[str, Any]],
) -> None:
    with closing(sqlite3.connect(local_db_path)) as conn, conn:
        ensure_schema(conn, DatabaseType.DAEMON_INBOX)
        _insert_auto_track_request(conn, payload)
        if command_data:
            _insert_auto_track_errors_command(
                conn=conn,
                region=region,
                logs_path=payload["logs_path"],
                timestamp=payload["timestamp"],
                command_data=command_data,
            )


def _insert_auto_track_request(
    conn: sqlite3.Connection,
    payload: dict[str, Any],
) -> None:
    conn.execute(
        AUTO_TRACK_REQUEST_INSERT_SQL,
        (
            json.dumps(payload["users"]),
            payload["logs_path"],
            payload["yaml_path"],
            payload.get("script_path"),
            payload["timestamp"],
            json.dumps(payload.get("metadata") or {}),
        ),
    )


def _insert_auto_track_errors_command(
    *,
    conn: sqlite3.Connection,
    region: str,
    logs_path: str,
    timestamp: str,
    command_data: dict[str, Any],
) -> None:
    try:
        conn.execute(
            AUTO_TRACK_COMMAND_INSERT_SQL,
            (
                TRACKING_AUTO_TRACK_ERRORS_COMMAND_TYPE,
                logs_path,
                None,
                timestamp,
                json.dumps(command_data),
            ),
        )
    except Exception:
        # Command enqueue is best-effort; the auto-track request itself must survive.
        logger.exception(
            "Failed to enqueue '%s' command for logs=%s (region=%s)",
            TRACKING_AUTO_TRACK_ERRORS_COMMAND_TYPE,
            logs_path,
            region,
        )
