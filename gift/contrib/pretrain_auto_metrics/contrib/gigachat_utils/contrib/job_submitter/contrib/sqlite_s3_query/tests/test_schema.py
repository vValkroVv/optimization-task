from __future__ import annotations

import sqlite3

import pytest

from sqlite_s3_query import DatabaseType, ensure_schema


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    return {row[0] for row in rows}


def _column_names(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]


@pytest.mark.parametrize(
    ("db_type", "expected_tables"),
    [
        (DatabaseType.TRACKER_BOT, {"users", "event_cursors"}),
        (
            DatabaseType.DAEMON_STATE,
            {"experiments", "events", "memory", "daemon_users", "daemon_meta"},
        ),
        (DatabaseType.DAEMON_INBOX, {"commands", "auto_track_requests"}),
        (DatabaseType.QUEUE_STATE, {"queue_jobs", "queue_nodes", "queue_meta"}),
    ],
)
def test_ensure_schema_creates_expected_tables(db_type, expected_tables):
    conn = sqlite3.connect(":memory:")
    try:
        ensure_schema(conn, db_type)
        assert expected_tables <= _table_names(conn)
    finally:
        conn.close()


def test_queue_state_columns():
    conn = sqlite3.connect(":memory:")
    try:
        ensure_schema(conn, DatabaseType.QUEUE_STATE)
        assert _column_names(conn, "queue_jobs") == [
            "job_id",
            "master_job_id",
            "name",
            "username",
            "partition",
            "state",
            "nodes",
            "gpus",
            "time_used",
            "submit_time",
        ]
        assert _column_names(conn, "queue_nodes") == [
            "node_name",
            "partition",
            "state",
            "gpus_total",
        ]
        assert _column_names(conn, "queue_meta") == ["key", "value"]
    finally:
        conn.close()


def test_daemon_inbox_auto_track_metadata_column():
    conn = sqlite3.connect(":memory:")
    try:
        ensure_schema(conn, DatabaseType.DAEMON_INBOX)
        assert _column_names(conn, "auto_track_requests") == [
            "id",
            "users",
            "logs_path",
            "yaml_path",
            "script_path",
            "timestamp",
            "metadata",
        ]
    finally:
        conn.close()


def test_daemon_inbox_migrates_legacy_auto_track_table():
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(
            """
            CREATE TABLE auto_track_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                users TEXT NOT NULL,
                logs_path TEXT NOT NULL,
                yaml_path TEXT,
                script_path TEXT,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO auto_track_requests "
            "(users, logs_path, timestamp) VALUES ('[]', '/legacy', 'now')"
        )

        ensure_schema(conn, DatabaseType.DAEMON_INBOX)

        row = conn.execute(
            "SELECT logs_path, metadata FROM auto_track_requests"
        ).fetchone()
        assert row == ("/legacy", "{}")
    finally:
        conn.close()


def test_ensure_schema_is_idempotent():
    conn = sqlite3.connect(":memory:")
    try:
        ensure_schema(conn, DatabaseType.QUEUE_STATE)
        ensure_schema(conn, DatabaseType.QUEUE_STATE)
        conn.execute("INSERT INTO queue_meta (key, value) VALUES ('cluster', 'o1')")
        conn.commit()
        assert (
            conn.execute(
                "SELECT value FROM queue_meta WHERE key = 'cluster'"
            ).fetchone()[0]
            == "o1"
        )
    finally:
        conn.close()
