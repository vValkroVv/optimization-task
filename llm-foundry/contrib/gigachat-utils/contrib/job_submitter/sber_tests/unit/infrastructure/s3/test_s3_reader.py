from __future__ import annotations

import builtins
import importlib.util
from contextlib import contextmanager
from pathlib import Path

import pytest

from job_submitter.core.storage.exceptions import RemoteQueryError, UnsafeQueryError
from job_submitter.infrastructure.s3.config import S3StorageConfig
from job_submitter.infrastructure.s3.reader import (
    RemoteSQLiteS3Reader,
    validate_identifier,
)


pytestmark = pytest.mark.unit


@contextmanager
def _ok_query_factory(url: str):
    assert url.startswith("https://")

    @contextmanager
    def _query(sql: str, params=(), named_params=()):
        assert sql == "SELECT value FROM test WHERE id = ?"
        assert params == (7,)
        assert named_params == (("unused", "x"),)

        yield ("value",), iter([(123,)])

    yield _query


@contextmanager
def _boom_query_factory(url: str):
    del url
    raise RuntimeError("boom")
    yield


def _config() -> S3StorageConfig:
    return S3StorageConfig(
        endpoint_url="https://obs.example.local",
        region="ru-x",
        bucket="bucket",
        key="db.sqlite",
    )


def test_validate_identifier_success_and_failure() -> None:
    assert validate_identifier("slurm_job_metrics") == "slurm_job_metrics"
    with pytest.raises(UnsafeQueryError):
        validate_identifier("drop table x")


def test_remote_reader_query_returns_query_result() -> None:
    reader = RemoteSQLiteS3Reader(
        config=_config(), query_context_factory=_ok_query_factory
    )

    result = reader.query(
        "SELECT value FROM test WHERE id = ?",
        params=(7,),
        named_params=(("unused", "x"),),
    )

    assert result.columns == ("value",)
    assert result.rows == [(123,)]


def test_remote_reader_wraps_unexpected_errors() -> None:
    reader = RemoteSQLiteS3Reader(
        config=_config(), query_context_factory=_boom_query_factory
    )

    with pytest.raises(RemoteQueryError, match="Failed to query remote sqlite"):
        reader.query("SELECT 1")


def test_reader_import_requires_sqlite_s3_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    reader_path = (
        Path(__file__).resolve().parents[4]
        / "job_submitter"
        / "infrastructure"
        / "s3"
        / "reader.py"
    )
    real_import = builtins.__import__
    module_name = "job_submitter.infrastructure.s3.reader_import_error_test"

    for cached_name in list(sys.modules):
        if cached_name == "sqlite_s3_query" or cached_name.startswith(
            "sqlite_s3_query."
        ):
            monkeypatch.delitem(sys.modules, cached_name, raising=False)

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sqlite_s3_query" or name.startswith("sqlite_s3_query."):
            raise ModuleNotFoundError("sqlite_s3_query")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    spec = importlib.util.spec_from_file_location(module_name, reader_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader

    with pytest.raises(ModuleNotFoundError) as exc_info:
        spec.loader.exec_module(module)
    assert "sqlite_s3_query is not available" in str(exc_info.value)
    assert "contrib/job_submitter/contrib/sqlite_s3_query" in str(exc_info.value)
