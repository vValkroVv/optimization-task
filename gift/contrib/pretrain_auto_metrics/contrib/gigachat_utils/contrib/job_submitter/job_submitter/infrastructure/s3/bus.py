from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from job_submitter.core.sqlite_s3_import import raise_sqlite_s3_query_import_error

try:
    from sqlite_s3_query.sqlite_file import normalize_sqlite_file
except ModuleNotFoundError as exc:
    raise_sqlite_s3_query_import_error(exc)

from job_submitter.infrastructure.s3.exceptions import (
    ConcurrentWriteError,
    DatabaseSyncError,
)
from job_submitter.infrastructure.s3.config import S3StorageConfig
from job_submitter.infrastructure.s3.constants import (
    S3_CHECKSUM_OPTIONS_MIN_BOTOCORE_VERSION,
    S3_NOT_FOUND_ERROR_CODES,
    S3_PAYLOAD_SIGNING_ENABLED,
    S3_SERVICE_NAME,
    S3_SIGNATURE_VERSION,
)
from job_submitter.infrastructure.s3.sync_state import RemoteObjectState


class S3LikeClient(Protocol):
    def download_file(self, bucket: str, key: str, filename: str) -> None: ...

    def upload_file(self, filename: str, bucket: str, key: str) -> None: ...

    def head_object(self, *, Bucket: str, Key: str): ...


class S3SQLiteBus:
    def __init__(self, *, s3_client: S3LikeClient, config: S3StorageConfig) -> None:
        self._s3_client = s3_client
        self._config = config

    def download_or_initialize(self, local_db_path: Path) -> RemoteObjectState | None:
        try:
            remote_state = self._current_state()
        except Exception as exc:
            error_code = _extract_error_code(exc)
            if error_code in S3_NOT_FOUND_ERROR_CODES:
                if local_db_path.exists():
                    local_db_path.unlink()
                return None
            raise DatabaseSyncError(
                f"Failed to read remote metadata from s3://{self._config.bucket}/{self._config.key}"
            ) from exc

        try:
            self._s3_client.download_file(
                self._config.bucket,
                self._config.key,
                str(local_db_path),
            )
            _ensure_sqlite_file(
                local_db_path,
                remote_uri=f"s3://{self._config.bucket}/{self._config.key}",
                operation="download",
            )
            return remote_state
        except Exception as exc:
            if isinstance(exc, DatabaseSyncError):
                raise
            error_code = _extract_error_code(exc)
            if error_code in S3_NOT_FOUND_ERROR_CODES:
                if local_db_path.exists():
                    local_db_path.unlink()
                return None
            raise DatabaseSyncError(
                f"Failed to download from s3://{self._config.bucket}/{self._config.key}"
            ) from exc

    def upload(
        self,
        local_db_path: Path,
        *,
        expected_state: RemoteObjectState | None = None,
    ) -> RemoteObjectState | None:
        if not local_db_path.exists():
            raise DatabaseSyncError(
                f"Local database file does not exist: {local_db_path}"
            )
        _ensure_sqlite_file(
            local_db_path,
            remote_uri=f"s3://{self._config.bucket}/{self._config.key}",
            operation="upload",
        )

        current_state = self._safe_current_state()
        if not _states_equal(current_state, expected_state):
            raise ConcurrentWriteError(
                "Remote sqlite changed during local transaction; retry is required"
            )

        try:
            self._s3_client.upload_file(
                str(local_db_path),
                self._config.bucket,
                self._config.key,
            )
        except Exception as exc:
            raise DatabaseSyncError(
                f"Failed to upload to s3://{self._config.bucket}/{self._config.key}"
            ) from exc

        return self._safe_current_state()

    def _current_state(self) -> RemoteObjectState:
        head = self._s3_client.head_object(
            Bucket=self._config.bucket, Key=self._config.key
        )

        etag_raw = head.get("ETag") if isinstance(head, dict) else None
        etag = str(etag_raw).strip('"') if etag_raw else None
        version_raw = head.get("VersionId") if isinstance(head, dict) else None
        version_id = str(version_raw) if version_raw else None

        return RemoteObjectState(etag=etag, version_id=version_id)

    def _safe_current_state(self) -> RemoteObjectState | None:
        try:
            return self._current_state()
        except Exception as exc:
            error_code = _extract_error_code(exc)
            if error_code in S3_NOT_FOUND_ERROR_CODES:
                return None
            raise DatabaseSyncError(
                f"Failed to read remote metadata from s3://{self._config.bucket}/{self._config.key}"
            ) from exc


def build_s3_client(config: S3StorageConfig) -> Any:
    import boto3
    import botocore
    from botocore.config import Config as BotoConfig

    kwargs: dict[str, Any] = {
        "service_name": S3_SERVICE_NAME,
        "region_name": config.region,
        "endpoint_url": config.endpoint_url,
        "verify": config.verify_ssl,
        "config": BotoConfig(
            signature_version=S3_SIGNATURE_VERSION,
            s3={
                "addressing_style": config.addressing_style,
                "payload_signing_enabled": S3_PAYLOAD_SIGNING_ENABLED,
            },
            **_checksum_options(botocore.__version__),
        ),
    }

    if config.aws_access_key_id and config.aws_secret_access_key:
        kwargs["aws_access_key_id"] = config.aws_access_key_id
        kwargs["aws_secret_access_key"] = config.aws_secret_access_key
        if config.aws_session_token:
            kwargs["aws_session_token"] = config.aws_session_token

    return boto3.client(**kwargs)


def _checksum_options(botocore_version: str) -> dict[str, str]:
    # botocore >= 1.36 adds CRC checksums to uploads by default and the island
    # S3 proxies reject them (XAmzContentSHA256Mismatch). Older botocore, e.g.
    # the system python on island front nodes, sends no such checksums and
    # rejects these options with TypeError.
    major, minor = (int(part) for part in botocore_version.split(".")[:2])
    if (major, minor) < S3_CHECKSUM_OPTIONS_MIN_BOTOCORE_VERSION:
        return {}
    return {
        "request_checksum_calculation": "when_required",
        "response_checksum_validation": "when_required",
    }


def _states_equal(
    left: RemoteObjectState | None,
    right: RemoteObjectState | None,
) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False

    if left.version_id is not None or right.version_id is not None:
        return left.version_id == right.version_id

    return left.etag == right.etag


def _ensure_sqlite_file(
    local_db_path: Path,
    *,
    remote_uri: str,
    operation: str,
) -> None:
    try:
        normalize_sqlite_file(local_db_path)
    except ValueError as exc:
        if operation == "download" and local_db_path.exists():
            local_db_path.unlink()
        raise DatabaseSyncError(
            f"Refusing to {operation} non-SQLite database file for {remote_uri}: {exc}"
        ) from exc


def _extract_error_code(exc: Exception) -> str | None:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return None

    error = response.get("Error")
    if not isinstance(error, dict):
        return None

    code = error.get("Code")
    return str(code) if code is not None else None
