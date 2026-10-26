from __future__ import annotations

import sqlite3
import sys
import types
from unittest.mock import patch

import pytest

from sqlite_s3_query import DatabaseType, S3DatabaseConfig, SqliteS3Database
from sqlite_s3_query.config import silence_insecure_request_warnings
from sqlite_s3_query import connection as connection_module
from sqlite_s3_query.connection import (
    _build_s3_client,
    _configure_journal_mode,
    _is_filesystem_locking_error,
)
from sqlite_s3_query.sqlite_file import SQLITE_MAGIC_HEADER


def _make_local_sqlite(path):
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO t(value) VALUES ('x')")
    conn.commit()
    conn.close()
    return path.read_bytes()


def _config(**overrides):
    params = {
        "endpoint_url": "https://s3.example.com",
        "region": "us-east-1",
        "bucket": "test-bucket",
        "key_prefix": "tracker/",
    }
    params.update(overrides)
    return S3DatabaseConfig(**params)


def _capture_build_s3_client_kwargs(config):
    calls: list[dict] = []

    class _FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def _fake_client(**kwargs):
        calls.append(kwargs)
        return object()

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = _fake_client
    fake_botocore_config = types.ModuleType("botocore.config")
    fake_botocore_config.Config = _FakeConfig

    with patch.dict(
        sys.modules,
        {"boto3": fake_boto3, "botocore.config": fake_botocore_config},
    ):
        _build_s3_client(config)

    assert calls, "boto3.client must be called"
    return calls[0]


def test_build_s3_client_uses_secure_virtual_defaults(monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

    kwargs = _capture_build_s3_client_kwargs(_config())

    assert kwargs["verify"] is True
    assert kwargs["config"].kwargs["s3"]["addressing_style"] == "virtual"


def test_silence_insecure_request_warnings_only_when_disabled():
    import urllib3

    calls: list = []
    original = urllib3.disable_warnings
    urllib3.disable_warnings = lambda *a, **k: calls.append((a, k))
    try:
        silence_insecure_request_warnings(True)
        assert calls == []
        silence_insecure_request_warnings(False)
        assert len(calls) == 1
    finally:
        urllib3.disable_warnings = original


def test_build_s3_client_supports_on_prem_gateway(monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

    kwargs = _capture_build_s3_client_kwargs(
        _config(verify_ssl=False, force_path_style=True)
    )

    assert kwargs["verify"] is False
    assert kwargs["config"].kwargs["s3"]["addressing_style"] == "path"


class _SnapshotCapturingClient:
    """Fake S3 client that records the uploaded path and its bytes at call time."""

    def __init__(self):
        self.args = None
        self.uploaded_bytes = None

    def upload_file(self, local_path, bucket, key):
        self.args = (local_path, bucket, key)
        with open(local_path, "rb") as handle:
            self.uploaded_bytes = handle.read()


def test_upload_streams_snapshot_copy_not_live_file(tmp_path):
    db = SqliteS3Database(
        config=_config(),
        s3_key="daemons/pretrain_sr004_state.sqlite",
        db_type=DatabaseType.DAEMON_STATE,
        local_dir=tmp_path,
    )
    payload = _make_local_sqlite(db.local_path)

    client = _SnapshotCapturingClient()
    db._s3_client = client
    db.upload()

    assert client.args is not None
    local_path, bucket, key = client.args
    assert bucket == "test-bucket"
    assert key == "tracker/daemons/pretrain_sr004_state.sqlite"
    # The upload must stream an immutable snapshot, never the live file a
    # concurrent writer could mutate mid-upload (S3 BadDigest).
    assert local_path == str(db.local_path) + ".upload"
    assert not db.local_path.with_name(db.local_path.name + ".upload").exists()
    assert client.uploaded_bytes == payload
    assert db.local_path.read_bytes() == payload
    assert client.uploaded_bytes.startswith(SQLITE_MAGIC_HEADER)


def test_upload_snapshots_live_connection_via_backup(tmp_path):
    db = SqliteS3Database(
        config=_config(),
        s3_key="daemons/pretrain_sr004_state.sqlite",
        db_type=DatabaseType.DAEMON_STATE,
        local_dir=tmp_path,
    )
    _make_local_sqlite(db.local_path)
    conn = db.connect()
    conn.execute("INSERT INTO t(value) VALUES ('live-row')")
    conn.commit()

    client = _SnapshotCapturingClient()
    db._s3_client = client
    db.upload()

    assert client.uploaded_bytes.startswith(SQLITE_MAGIC_HEADER)
    snapshot_file = tmp_path / "probe.sqlite"
    snapshot_file.write_bytes(client.uploaded_bytes)
    probe = sqlite3.connect(str(snapshot_file))
    try:
        values = {row[0] for row in probe.execute("SELECT value FROM t")}
    finally:
        probe.close()
    assert "live-row" in values
    db.close()


def test_upload_requires_local_file_or_connection(tmp_path):
    db = SqliteS3Database(
        config=_config(),
        s3_key="daemons/pretrain_sr004_state.sqlite",
        db_type=DatabaseType.DAEMON_STATE,
        local_dir=tmp_path,
    )
    with pytest.raises(FileNotFoundError):
        db.upload()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX-only: Windows locks open database files",
)
def test_connection_property_heals_after_file_replacement(tmp_path):
    import os

    db = SqliteS3Database(
        config=_config(),
        s3_key="daemons/pretrain_sr004_state.sqlite",
        db_type=DatabaseType.DAEMON_STATE,
        local_dir=tmp_path,
    )
    _make_local_sqlite(db.local_path)
    stale = db.connect()
    stale.execute("INSERT INTO t(value) VALUES ('before')")
    stale.commit()

    # Replace the file under the live handle, as an S3 download (tmp+rename)
    # or a /tmp cleaner would. Depending on journal mode the stale handle
    # either turns readonly (DELETE journal) or keeps writing into the
    # unlinked inode where data silently vanishes (WAL) — the inode check
    # heals both.
    replacement = _make_local_sqlite(tmp_path / "replacement.sqlite")
    os.replace(tmp_path / "replacement.sqlite", db.local_path)

    healed = db.connection
    assert healed is not stale
    healed.execute("INSERT INTO t(value) VALUES ('after-heal')")
    healed.commit()
    values = {row[0] for row in healed.execute("SELECT value FROM t")}
    assert "after-heal" in values
    assert replacement.startswith(SQLITE_MAGIC_HEADER)
    db.close()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX-only: Windows locks open database files",
)
def test_connection_property_heals_after_file_deletion(tmp_path):
    db = SqliteS3Database(
        config=_config(),
        s3_key="daemons/pretrain_sr004_state.sqlite",
        db_type=DatabaseType.DAEMON_STATE,
        local_dir=tmp_path,
    )
    _make_local_sqlite(db.local_path)
    stale = db.connect()
    db.local_path.unlink()

    # The heal recreates a fresh file with the daemon schema; ad-hoc tables
    # from the deleted file are gone, so write into a schema table.
    healed = db.connection
    assert healed is not stale
    healed.execute(
        "INSERT OR REPLACE INTO daemon_meta (key, value) VALUES ('probe', '1')"
    )
    healed.commit()
    db.close()


def test_connection_property_heals_on_captured_inode_mismatch(tmp_path):
    # Cross-platform simulation of a replaced file: forge the inode captured
    # at connect time so the next property access sees a mismatch.
    db = SqliteS3Database(
        config=_config(),
        s3_key="daemons/pretrain_sr004_state.sqlite",
        db_type=DatabaseType.DAEMON_STATE,
        local_dir=tmp_path,
    )
    _make_local_sqlite(db.local_path)
    stale = db.connect()
    dev, ino = db._conn_stat
    db._conn_stat = (dev, ino + 1)

    healed = db.connection
    assert healed is not stale
    healed.execute("INSERT INTO t(value) VALUES ('after-heal')")
    healed.commit()
    db.close()


def test_connection_property_keeps_connection_when_file_unchanged(tmp_path):
    db = SqliteS3Database(
        config=_config(),
        s3_key="daemons/pretrain_sr004_state.sqlite",
        db_type=DatabaseType.DAEMON_STATE,
        local_dir=tmp_path,
    )
    _make_local_sqlite(db.local_path)
    conn = db.connect()
    assert db.connection is conn
    db.close()
    with pytest.raises(RuntimeError, match="not connected"):
        _ = db.connection


def test_download_closes_live_connection_before_replacing_file(tmp_path):
    db = SqliteS3Database(
        config=_config(),
        s3_key="daemons/pretrain_sr004_state.sqlite",
        db_type=DatabaseType.DAEMON_STATE,
        local_dir=tmp_path,
    )
    _make_local_sqlite(db.local_path)
    db.connect()
    replacement_bytes = _make_local_sqlite(tmp_path / "fresh.sqlite")

    class _Client:
        def download_file(self, bucket, key, local_path):
            import os

            os.replace(tmp_path / "fresh.sqlite", local_path)

    db._s3_client = _Client()
    assert db.download() is True
    assert db._conn is None

    conn = db.connect()
    conn.execute("INSERT INTO t(value) VALUES ('post-download')")
    conn.commit()
    assert replacement_bytes.startswith(SQLITE_MAGIC_HEADER)
    db.close()


def test_connect_repairs_chunked_sqlite_before_open(tmp_path):
    db = SqliteS3Database(
        config=_config(),
        s3_key="daemons/pretrain_sr008_state.sqlite",
        db_type=DatabaseType.DAEMON_STATE,
        local_dir=tmp_path,
    )
    payload = _make_local_sqlite(tmp_path / "seed.sqlite")
    chunked = f"{len(payload):X}\r\n".encode("ascii") + payload + b"\r\n0\r\n\r\n"
    db.local_path.write_bytes(chunked)

    conn = db.connect()
    assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    assert db.local_path.read_bytes().startswith(SQLITE_MAGIC_HEADER)
    db.close()


class _NoWalConnection:
    """Wraps a real connection but rejects WAL, like an NFS mount without -shm."""

    def __init__(self, real):
        self._real = real

    def execute(self, sql, *args):
        if "journal_mode=WAL" in sql:
            raise sqlite3.OperationalError("locking protocol")
        return self._real.execute(sql, *args)


def test_configure_journal_mode_uses_wal_on_normal_fs(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "wal.db"))
    try:
        _configure_journal_mode(conn)
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        conn.close()


def test_configure_journal_mode_falls_back_when_wal_unavailable(tmp_path):
    real = sqlite3.connect(str(tmp_path / "nfs.db"))
    try:
        _configure_journal_mode(_NoWalConnection(real))
        assert real.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert real.execute("PRAGMA locking_mode").fetchone()[0] == "exclusive"
    finally:
        real.close()


def test_configure_journal_mode_recovers_wal_file(tmp_path):
    path = tmp_path / "stuck.db"
    seed = sqlite3.connect(str(path))
    seed.execute("PRAGMA journal_mode=WAL")
    seed.execute("CREATE TABLE t (x INTEGER)")
    seed.execute("INSERT INTO t VALUES (7)")
    seed.commit()
    seed.close()
    # Header now marks the file as WAL. Reopen where WAL is unavailable (NFS):
    # the fallback must still read the data and convert the file to DELETE.
    real = sqlite3.connect(str(path))
    try:
        _configure_journal_mode(_NoWalConnection(real))
        assert real.execute("SELECT x FROM t").fetchone()[0] == 7
        assert real.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    finally:
        real.close()


def test_is_filesystem_locking_error():
    assert _is_filesystem_locking_error(sqlite3.OperationalError("locking protocol"))
    assert _is_filesystem_locking_error(sqlite3.OperationalError("disk I/O error"))
    assert not _is_filesystem_locking_error(
        sqlite3.OperationalError("no such table: t")
    )


def test_connect_falls_back_to_nolock_on_locking_error(tmp_path, monkeypatch):
    db = SqliteS3Database(
        config=_config(),
        s3_key="daemons/nolock_state.sqlite",
        db_type=DatabaseType.DAEMON_STATE,
        local_dir=tmp_path,
    )

    def raise_locking(_conn):
        raise sqlite3.OperationalError("locking protocol")

    # Simulate a filesystem where normal journal setup fails on file locks.
    monkeypatch.setattr(connection_module, "_configure_journal_mode", raise_locking)

    conn = db.connect()
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert "experiments" in tables  # schema created on the nolock connection
    finally:
        db.close()


def _raise_locking(_conn):
    raise sqlite3.OperationalError("locking protocol")


def test_connect_nolock_fallback_recovers_wal_file(tmp_path, monkeypatch):
    db = SqliteS3Database(
        config=_config(),
        s3_key="daemons/walnolock_state.sqlite",
        db_type=DatabaseType.DAEMON_STATE,
        local_dir=tmp_path,
    )
    # Seed a WAL-mode file with data at the exact local path connect() will use.
    seed = sqlite3.connect(str(db.local_path))
    seed.execute("PRAGMA journal_mode=WAL")
    seed.execute(
        "CREATE TABLE experiments (logs_path TEXT PRIMARY KEY, data TEXT NOT NULL)"
    )
    seed.execute("INSERT INTO experiments VALUES ('p', '{}')")
    seed.commit()
    seed.close()

    # Emulate a filesystem without working locks.
    monkeypatch.setattr(connection_module, "_configure_journal_mode", _raise_locking)

    conn = db.connect()
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        row = conn.execute(
            "SELECT data FROM experiments WHERE logs_path = 'p'"
        ).fetchone()
        assert row[0] == "{}"  # data preserved through the WAL-header downgrade
    finally:
        db.close()


def test_connect_reraises_non_locking_operational_error(tmp_path, monkeypatch):
    db = SqliteS3Database(
        config=_config(),
        s3_key="daemons/other_state.sqlite",
        db_type=DatabaseType.DAEMON_STATE,
        local_dir=tmp_path,
    )

    def raise_other(_conn):
        raise sqlite3.OperationalError("no such table: whatever")

    monkeypatch.setattr(connection_module, "_configure_journal_mode", raise_other)
    with pytest.raises(sqlite3.OperationalError):
        db.connect()


def test_build_s3_client_falls_back_for_legacy_botocore(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ak")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "sk")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)

    class _FakeConfig:
        def __init__(self, **kwargs):
            if "request_checksum_calculation" in kwargs:
                raise TypeError(
                    "Got unexpected keyword argument 'request_checksum_calculation'"
                )
            self.kwargs = kwargs

    calls: list[dict] = []

    def _fake_client(**kwargs):
        calls.append(kwargs)
        return object()

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = _fake_client

    fake_botocore_config = types.ModuleType("botocore.config")
    fake_botocore_config.Config = _FakeConfig

    with patch.dict(
        sys.modules,
        {
            "boto3": fake_boto3,
            "botocore.config": fake_botocore_config,
        },
    ):
        _build_s3_client(_config())

    assert calls, "boto3.client must be called"
    cfg = calls[0]["config"]
    assert "request_checksum_calculation" not in cfg.kwargs
    assert "response_checksum_validation" not in cfg.kwargs
    assert cfg.kwargs["signature_version"] == "s3v4"
