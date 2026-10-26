import pytest

from job_submitter.core.storage.interfaces import Database
from job_submitter.core.storage.query import QueryResult
from job_submitter.core.storage.service import StorageService


pytestmark = pytest.mark.unit


class _FakeDatabase(Database):
    def __init__(self) -> None:
        self.closed = False

    def execute(self, sql, params=()):
        return 1

    def execute_many(self, sql, params_list=()):
        return len(params_list)

    def query(self, sql, params=()):
        return QueryResult(columns=("x",), rows=[(1,)])

    def select(self, sql, params=()):
        return QueryResult(columns=("x",), rows=[(2,)])

    def close(self):
        self.closed = True


def test_storage_service_proxies_execute() -> None:
    db = _FakeDatabase()
    svc = StorageService(db)

    assert svc.execute("INSERT INTO t VALUES (1)") == 1


def test_storage_service_proxies_query() -> None:
    db = _FakeDatabase()
    svc = StorageService(db)

    result = svc.query("SELECT 1")

    assert result.columns == ("x",)
    assert result.rows == [(1,)]


def test_storage_service_proxies_select() -> None:
    db = _FakeDatabase()
    svc = StorageService(db)

    result = svc.select("SELECT 1")

    assert result.rows == [(2,)]


def test_storage_service_proxies_execute_many() -> None:
    db = _FakeDatabase()
    svc = StorageService(db)

    assert svc.execute_many("INSERT INTO t VALUES (?)", [(1,), (2,)]) == 2


def test_database_close_default_is_noop() -> None:
    class _MinimalDB(Database):
        def execute(self, sql, params=()):
            return 0

        def execute_many(self, sql, params_list=()):
            return 0

        def query(self, sql, params=()):
            return QueryResult(columns=(), rows=[])

        def select(self, sql, params=()):
            return QueryResult(columns=(), rows=[])

    db = _MinimalDB()
    db.close()


def test_storage_service_context_manager_calls_close() -> None:
    db = _FakeDatabase()

    with StorageService(db) as svc:
        svc.execute("INSERT INTO t VALUES (1)")

    assert db.closed
