from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from job_submitter.core.storage.query import QueryResult


class Database(ABC):
    @abstractmethod
    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        """Write operation (INSERT/UPDATE/DELETE/DDL). Returns affected row count."""
        ...

    @abstractmethod
    def query(self, sql: str, params: Sequence[Any] = ()) -> QueryResult:
        """Run arbitrary SQL and return results."""
        ...

    @abstractmethod
    def execute_many(self, sql: str, params_list: Sequence[Sequence[Any]]) -> int:
        """Batch write. Returns total affected row count."""
        ...

    @abstractmethod
    def select(self, sql: str, params: Sequence[Any] = ()) -> QueryResult:
        """Read-only SELECT query."""
        ...

    def close(self) -> None:
        pass
