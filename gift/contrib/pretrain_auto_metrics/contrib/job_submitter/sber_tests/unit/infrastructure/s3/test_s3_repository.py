import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from job_submitter.infrastructure.s3.repository import SQLiteRepository


pytestmark = pytest.mark.unit


@contextmanager
def _memory_connection() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(":memory:")
    try:
        yield conn
    finally:
        conn.close()


def test_query_returns_columns_and_rows() -> None:
    with _memory_connection() as conn:
        repo = SQLiteRepository(conn)
        repo.execute("CREATE TABLE test (id INTEGER, name TEXT)")
        repo.execute_many(
            "INSERT INTO test (id, name) VALUES (?, ?)",
            [(1, "a"), (2, "b")],
        )
        repo.commit()

        result = repo.query("SELECT id, name FROM test ORDER BY id")

    assert result.columns == ("id", "name")
    assert result.rows == [(1, "a"), (2, "b")]


def test_execute_returns_rowcount_for_insert_and_update() -> None:
    with _memory_connection() as conn:
        repo = SQLiteRepository(conn)
        repo.execute("CREATE TABLE test (id INTEGER, name TEXT)")
        insert_count = repo.execute(
            "INSERT INTO test (id, name) VALUES (?, ?)", (1, "x")
        )
        update_count = repo.execute("UPDATE test SET name = ? WHERE id = ?", ("y", 1))

    assert insert_count == 1
    assert update_count == 1


def test_execute_many_returns_rowcount() -> None:
    with _memory_connection() as conn:
        repo = SQLiteRepository(conn)
        repo.execute("CREATE TABLE test (id INTEGER)")

        count = repo.execute_many(
            "INSERT INTO test (id) VALUES (?)", [(1,), (2,), (3,)]
        )

    assert count == 3


def test_execute_script_applies_statements() -> None:
    with _memory_connection() as conn:
        repo = SQLiteRepository(conn)
        repo.execute_script(
            """
            CREATE TABLE t (id INTEGER PRIMARY KEY, value TEXT);
            INSERT INTO t (id, value) VALUES (1, 'a');
            """
        )
        result = repo.query("SELECT id, value FROM t")

    assert result.rows == [(1, "a")]


def test_manual_rollback_discards_uncommitted_dml() -> None:
    with _memory_connection() as conn:
        repo = SQLiteRepository(conn)
        repo.execute("CREATE TABLE t (id INTEGER)")
        repo.execute("INSERT INTO t (id) VALUES (1)")
        repo.rollback()

        result = repo.query("SELECT COUNT(*) FROM t")

    assert result.rows == [(0,)]
