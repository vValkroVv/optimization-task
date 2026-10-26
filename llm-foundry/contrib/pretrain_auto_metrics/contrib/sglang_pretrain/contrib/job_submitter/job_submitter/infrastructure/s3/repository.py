from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from typing import Any

from job_submitter.core.storage.query import QueryResult


class SQLiteRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def query(self, sql: str, params: Sequence[Any] = ()) -> QueryResult:
        cursor = self._connection.execute(sql, tuple(params))
        columns = tuple(desc[0] for desc in (cursor.description or ()))
        rows = [tuple(row) for row in cursor.fetchall()]
        return QueryResult(columns=columns, rows=rows)

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        cursor = self._connection.execute(sql, tuple(params))
        return int(cursor.rowcount)

    def execute_many(self, sql: str, params_list: Sequence[Sequence[Any]]) -> int:
        cursor = self._connection.executemany(
            sql,
            [tuple(params) for params in params_list],
        )
        return int(cursor.rowcount)

    def execute_script(self, script: str) -> None:
        self._connection.executescript(script)

    def commit(self) -> None:
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()
