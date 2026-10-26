from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Callable

from job_submitter.core.sqlite_s3_import import raise_sqlite_s3_query_import_error

try:
    from sqlite_s3_query.query import sqlite_s3_query
except ModuleNotFoundError as exc:
    raise_sqlite_s3_query_import_error(exc)
from job_submitter.core.storage.exceptions import RemoteQueryError, UnsafeQueryError
from job_submitter.core.storage.query import QueryResult
from job_submitter.infrastructure.s3.config import S3StorageConfig, build_db_url

_SAFE_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_REMOTE_QUERY_ERROR = "Failed to query remote sqlite in s3://{bucket}/{key}"
_UNSAFE_SQL_IDENTIFIER_ERROR = "Unsafe SQL identifier: {value!r}"


class RemoteSQLiteS3Reader:
    def __init__(
        self,
        *,
        config: S3StorageConfig,
        query_context_factory: Callable[..., Any] = sqlite_s3_query,
    ) -> None:
        self._config = config
        self._db_url = build_db_url(config)
        self._query_context_factory = query_context_factory

    def query(
        self,
        sql: str,
        params: Sequence[Any] = (),
        named_params: Sequence[tuple[str, Any]] = (),
    ) -> QueryResult:
        try:
            with self._query_context_factory(url=self._db_url) as remote_query:
                with remote_query(
                    sql,
                    params=tuple(params),
                    named_params=tuple(named_params),
                ) as (columns, rows):
                    return QueryResult(
                        columns=tuple(columns),
                        rows=[tuple(row) for row in rows],
                    )
        except Exception as exc:
            raise RemoteQueryError(
                _REMOTE_QUERY_ERROR.format(
                    bucket=self._config.bucket,
                    key=self._config.key,
                )
            ) from exc


def validate_identifier(value: str) -> str:
    if not _SAFE_IDENTIFIER_RE.fullmatch(value):
        raise UnsafeQueryError(_UNSAFE_SQL_IDENTIFIER_ERROR.format(value=value))
    return value
