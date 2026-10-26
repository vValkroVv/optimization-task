from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from unittest.mock import patch

import pytest

import sqlite_s3_query.query as query_module
from sqlite_s3_query.query import SQLiteError, _download_sqlite
from sqlite_s3_query.sqlite_file import SQLITE_MAGIC_HEADER


def _make_sqlite_payload(tmp_path):
    db_path = tmp_path / "seed.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE q (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()
    return db_path.read_bytes()


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=8192):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]

    def close(self):
        return None


class _FakeSession:
    def __init__(self, body: bytes):
        self._body = body

    def get(self, *args, **kwargs):
        return _FakeResponse(self._body)


def test_download_sqlite_repairs_chunked_payload(tmp_path):
    payload = _make_sqlite_payload(tmp_path)
    chunked = f"{len(payload):X}\r\n".encode("ascii") + payload + b"\r\n0\r\n\r\n"
    session = _FakeSession(chunked)

    path = _download_sqlite(
        "https://bucket.example.com/tracker/daemons/pretrain_sr004_state.sqlite",
        session,
        get_credentials=None,
    )
    try:
        data = path.read_bytes()
        assert data.startswith(SQLITE_MAGIC_HEADER)
    finally:
        path.unlink(missing_ok=True)


def test_download_sqlite_raises_for_invalid_payload():
    session = _FakeSession(b"this is not sqlite")
    with pytest.raises(SQLiteError, match="not a SQLite database"):
        _download_sqlite(
            "https://bucket.example.com/tracker/daemons/pretrain_sr004_state.sqlite",
            session,
            get_credentials=None,
        )


def test_sqlite_s3_query_multi_uses_ctypes_backend_by_default(monkeypatch):
    monkeypatch.delenv("SQLITE_S3_QUERY_READ_BACKEND", raising=False)
    called = {"ctypes": False}

    @contextmanager
    def _fake_ctypes(**kwargs):
        called["ctypes"] = True
        yield lambda *_args, **_kwargs: iter(())

    @contextmanager
    def _fake_download(**kwargs):
        raise AssertionError("download backend must not be used by default")
        yield

    with (
        patch("sqlite_s3_query.query._sqlite_s3_query_multi_ctypes", _fake_ctypes),
        patch(
            "sqlite_s3_query.query._sqlite_s3_query_multi_download",
            _fake_download,
        ),
    ):
        with query_module.sqlite_s3_query_multi(
            "https://bucket.example.com/tracker/daemons/pretrain_sr004_state.sqlite",
            get_credentials=None,
        ):
            pass

    assert called["ctypes"] is True


def test_sqlite_s3_query_multi_supports_download_backend_via_env(monkeypatch):
    monkeypatch.setenv("SQLITE_S3_QUERY_READ_BACKEND", "download")
    called = {"download": False}

    @contextmanager
    def _fake_ctypes(**kwargs):
        raise AssertionError("ctypes backend must not be used in download mode")
        yield

    @contextmanager
    def _fake_download(**kwargs):
        called["download"] = True
        yield lambda *_args, **_kwargs: iter(())

    with (
        patch("sqlite_s3_query.query._sqlite_s3_query_multi_ctypes", _fake_ctypes),
        patch(
            "sqlite_s3_query.query._sqlite_s3_query_multi_download",
            _fake_download,
        ),
    ):
        with query_module.sqlite_s3_query_multi(
            "https://bucket.example.com/tracker/daemons/pretrain_sr004_state.sqlite",
            get_credentials=None,
        ):
            pass

    assert called["download"] is True


def test_sqlite_s3_query_multi_falls_back_to_download_on_non_sqlite(monkeypatch):
    monkeypatch.delenv("SQLITE_S3_QUERY_READ_BACKEND", raising=False)
    called = {"download": False}

    @contextmanager
    def _fake_ctypes(**kwargs):
        def _query(*_args, **_kwargs):
            raise SQLiteError("file is not a database")
            yield  # pragma: no cover

        yield _query

    @contextmanager
    def _fake_download(**kwargs):
        called["download"] = True

        def _query(*_args, **_kwargs):
            yield ("value",), iter(((1,),))

        yield _query

    with (
        patch("sqlite_s3_query.query._sqlite_s3_query_multi_ctypes", _fake_ctypes),
        patch(
            "sqlite_s3_query.query._sqlite_s3_query_multi_download",
            _fake_download,
        ),
    ):
        with query_module.sqlite_s3_query_multi(
            "https://bucket.example.com/tracker/daemons/pretrain_sr004_state.sqlite",
            get_credentials=None,
        ) as query_fn:
            rows = []
            for _columns, result_rows in query_fn("SELECT 1"):
                rows = list(result_rows)

    assert called["download"] is True
    assert rows == [(1,)]
