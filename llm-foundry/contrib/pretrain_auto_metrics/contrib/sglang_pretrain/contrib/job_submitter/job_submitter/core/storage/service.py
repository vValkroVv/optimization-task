from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from job_submitter.core.storage.interfaces import Database
from job_submitter.core.storage.query import QueryResult


class StorageService:
    def __init__(self, db: Database) -> None:
        self._db = db

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        return self._db.execute(sql, params)

    def query(self, sql: str, params: Sequence[Any] = ()) -> QueryResult:
        return self._db.query(sql, params)

    def execute_many(self, sql: str, params_list: Sequence[Sequence[Any]]) -> int:
        return self._db.execute_many(sql, params_list)

    def select(self, sql: str, params: Sequence[Any] = ()) -> QueryResult:
        return self._db.select(sql, params)

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> StorageService:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
