from __future__ import annotations

import sqlite3
from enum import Enum


class DatabaseType(str, Enum):
    TRACKER_BOT = "tracker_bot"
    DAEMON_STATE = "daemon_state"
    DAEMON_INBOX = "daemon_inbox"
    QUEUE_STATE = "queue_state"


_TRACKER_BOT_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        in_whitelist INTEGER NOT NULL DEFAULT 1,
        mattermost_user_id TEXT,
        subscribed_projects TEXT NOT NULL DEFAULT '[]'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS event_cursors (
        daemon_name TEXT PRIMARY KEY,
        last_event_id INTEGER NOT NULL DEFAULT 0
    )
    """,
]

_DAEMON_STATE_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS experiments (
        logs_path TEXT PRIMARY KEY,
        data TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT NOT NULL,
        experiment_path TEXT,
        message TEXT NOT NULL,
        usernames TEXT NOT NULL DEFAULT '[]',
        timestamp TEXT NOT NULL,
        cluster_name TEXT NOT NULL DEFAULT '',
        metadata TEXT NOT NULL DEFAULT '{}'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memory (
        key TEXT PRIMARY KEY DEFAULT 'latest',
        data TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS daemon_users (
        username TEXT PRIMARY KEY,
        in_whitelist INTEGER NOT NULL DEFAULT 1,
        mattermost_user_id TEXT,
        subscribed_projects TEXT NOT NULL DEFAULT '[]'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS daemon_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
]

_DAEMON_INBOX_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS commands (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        command_type TEXT NOT NULL,
        experiment_path TEXT,
        hash TEXT,
        timestamp TEXT NOT NULL,
        data TEXT NOT NULL DEFAULT '{}'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS auto_track_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        users TEXT NOT NULL,
        logs_path TEXT NOT NULL,
        yaml_path TEXT,
        script_path TEXT,
        timestamp TEXT NOT NULL,
        metadata TEXT NOT NULL DEFAULT '{}'
    )
    """,
]

_QUEUE_STATE_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS queue_jobs (
        job_id TEXT NOT NULL,
        master_job_id TEXT NOT NULL,
        name TEXT NOT NULL,
        username TEXT NOT NULL,
        partition TEXT NOT NULL,
        state TEXT NOT NULL,
        nodes INTEGER NOT NULL,
        gpus INTEGER NOT NULL,
        time_used TEXT NOT NULL,
        submit_time TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS queue_nodes (
        node_name TEXT NOT NULL,
        partition TEXT NOT NULL,
        state TEXT NOT NULL,
        gpus_total INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS queue_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
]

_SCHEMAS: dict[DatabaseType, list[str]] = {
    DatabaseType.TRACKER_BOT: _TRACKER_BOT_SCHEMA,
    DatabaseType.DAEMON_STATE: _DAEMON_STATE_SCHEMA,
    DatabaseType.DAEMON_INBOX: _DAEMON_INBOX_SCHEMA,
    DatabaseType.QUEUE_STATE: _QUEUE_STATE_SCHEMA,
}


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _migrate_daemon_inbox(conn: sqlite3.Connection) -> None:
    if "metadata" not in _column_names(conn, "auto_track_requests"):
        conn.execute(
            "ALTER TABLE auto_track_requests "
            "ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'"
        )


def ensure_schema(conn: sqlite3.Connection, db_type: DatabaseType) -> None:
    """Create tables if they don't exist."""
    for statement in _SCHEMAS[db_type]:
        conn.execute(statement)
    if db_type == DatabaseType.DAEMON_INBOX:
        _migrate_daemon_inbox(conn)
    conn.commit()
