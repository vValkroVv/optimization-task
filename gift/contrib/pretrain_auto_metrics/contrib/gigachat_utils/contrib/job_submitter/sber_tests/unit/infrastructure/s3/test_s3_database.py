from __future__ import annotations

import shutil
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue

import pytest

from job_submitter.infrastructure.s3.exceptions import (
    ConcurrentWriteError,
    DatabaseSyncError,
)
from job_submitter.infrastructure.s3.config import S3StorageConfig
from job_submitter.infrastructure.s3.database import S3SQLiteDatabase, _WriteTask
from job_submitter.infrastructure.s3.sync_state import RemoteObjectState


pytestmark = pytest.mark.unit


@dataclass
class _FakeBus:
    storage_file: Path
    downloads: list[Path] = field(default_factory=list)
    uploads: list[Path] = field(default_factory=list)
    remote_state: RemoteObjectState | None = None
    conflict_once: bool = False
    always_conflict: bool = False
    conflict_count: int = 0

    def download_or_initialize(self, local_db_path: Path) -> RemoteObjectState | None:
        self.downloads.append(local_db_path)
        if self.storage_file.exists():
            shutil.copy2(self.storage_file, local_db_path)
        return self.remote_state

    def upload(
        self,
        local_db_path: Path,
        *,
        expected_state: RemoteObjectState | None = None,
    ) -> RemoteObjectState | None:
        self.uploads.append(local_db_path)

        if self.always_conflict:
            self.conflict_count += 1
            raise ConcurrentWriteError("conflict")

        if self.conflict_once and self.conflict_count == 0:
            self.conflict_count += 1
            raise ConcurrentWriteError("conflict")

        if expected_state != self.remote_state:
            raise ConcurrentWriteError("conflict")

        shutil.copy2(local_db_path, self.storage_file)
        self.remote_state = RemoteObjectState(etag=f"v{len(self.uploads)}")
        return self.remote_state


def _make_db(
    tmp_path: Path,
    bus: _FakeBus | None = None,
    **kwargs,
) -> S3SQLiteDatabase:
    if bus is None:
        bus = _FakeBus(storage_file=tmp_path / "remote.sqlite")
    config = S3StorageConfig(
        endpoint_url="https://obs.example.local",
        region="r1",
        bucket="b",
        key="k",
    )
    return S3SQLiteDatabase(bus=bus, config=config, **kwargs)


def test_execute_and_query_persist_between_calls(tmp_path: Path) -> None:
    bus = _FakeBus(storage_file=tmp_path / "remote.sqlite")
    db = _make_db(tmp_path, bus=bus)

    with db:
        db.execute_script(
            """
            CREATE TABLE events (id INTEGER PRIMARY KEY, payload TEXT);
            """
        )
        db.execute("INSERT INTO events (id, payload) VALUES (?, ?)", (1, "hello"))
        result = db.query("SELECT id, payload FROM events")

    assert result.columns == ("id", "payload")
    assert result.rows == [(1, "hello")]
    assert len(bus.uploads) == 2


def test_execute_many_returns_count(tmp_path: Path) -> None:
    db = _make_db(tmp_path)

    with db:
        db.execute_script("CREATE TABLE events (id INTEGER)")
        count = db.execute_many(
            "INSERT INTO events (id) VALUES (?)",
            [(1,), (2,), (3,)],
        )

    assert count == 3


def test_query_returns_results_after_write(tmp_path: Path) -> None:
    db = _make_db(tmp_path)

    with db:
        db.execute_script("CREATE TABLE events (id INTEGER)")
        db.execute("INSERT INTO events (id) VALUES (1)")
        result = db.query("SELECT COUNT(*) FROM events")

    assert result.rows == [(1,)]


def test_select_falls_back_to_query_when_no_reader(tmp_path: Path) -> None:
    db = _make_db(tmp_path, reader=None)

    with db:
        db.execute_script("CREATE TABLE events (id INTEGER)")
        db.execute("INSERT INTO events (id) VALUES (1)")
        result = db.select("SELECT COUNT(*) FROM events")

    assert result.rows == [(1,)]


def test_select_uses_reader_when_available(tmp_path: Path) -> None:
    from job_submitter.core.storage.query import QueryResult

    class _FakeReader:
        def query(self, sql, params=(), named_params=()):
            return QueryResult(columns=("v",), rows=[(42,)])

    db = _make_db(tmp_path, reader=_FakeReader())

    with db:
        result = db.select("SELECT 1")

    assert result.rows == [(42,)]


def test_async_submit_and_query_flush(tmp_path: Path) -> None:
    db = _make_db(tmp_path, enable_async_writer=True)

    with db:
        db.execute_script("CREATE TABLE events (id INTEGER)")

        futures = [
            db.submit_execute("INSERT INTO events (id) VALUES (?)", (i,))
            for i in (1, 2, 3)
        ]
        result = db.query("SELECT COUNT(*) FROM events")

        assert [f.result() for f in futures] == [1, 1, 1]

    assert result.rows == [(3,)]


def test_sync_mode_executes_without_queue(tmp_path: Path) -> None:
    db = _make_db(tmp_path, enable_async_writer=False)

    db.execute_script("CREATE TABLE events (id INTEGER)")
    db.execute("INSERT INTO events (id) VALUES (1)")
    result = db.query("SELECT COUNT(*) FROM events")
    db.close()

    assert result.rows == [(1,)]


def test_writer_future_gets_exception_from_worker(tmp_path: Path) -> None:
    db = _make_db(tmp_path)

    with db:
        future = db.submit_execute("BROKEN SQL")
        with pytest.raises(Exception):
            future.result()


def test_retry_on_concurrent_write(tmp_path: Path) -> None:
    bus = _FakeBus(storage_file=tmp_path / "remote.sqlite", conflict_once=True)
    db = _make_db(
        tmp_path,
        bus=bus,
        max_publish_attempts=2,
        publish_retry_backoff_seconds=0,
    )

    with db:
        db.execute_script("CREATE TABLE events (id INTEGER)")
        db.execute("INSERT INTO events (id) VALUES (1)")
        result = db.query("SELECT COUNT(*) FROM events")

    assert bus.conflict_count == 1
    assert result.rows == [(1,)]


def test_retry_calls_sleep_when_backoff_positive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = _FakeBus(storage_file=tmp_path / "remote.sqlite", conflict_once=True)
    slept: list[float] = []

    monkeypatch.setattr(
        "job_submitter.infrastructure.retry.time.sleep",
        lambda seconds: slept.append(seconds),
    )

    db = _make_db(
        tmp_path,
        bus=bus,
        max_publish_attempts=2,
        publish_retry_backoff_seconds=0.5,
    )

    with db:
        db.execute_script("CREATE TABLE events (id INTEGER)")

    assert slept == [0.5]


@pytest.mark.parametrize("attempts", [1, 2, 3, 5])
def test_retry_exhaustion_raises_database_sync_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attempts: int
) -> None:
    bus = _FakeBus(storage_file=tmp_path / "remote.sqlite", always_conflict=True)
    slept: list[float] = []
    monkeypatch.setattr("job_submitter.infrastructure.retry.time.sleep", slept.append)
    db = _make_db(
        tmp_path,
        bus=bus,
        max_publish_attempts=attempts,
        publish_retry_backoff_seconds=0.5,
    )

    with db:
        with pytest.raises(DatabaseSyncError, match="concurrent updates") as caught:
            db.execute_script("CREATE TABLE events (id INTEGER)")

    assert bus.conflict_count == attempts
    assert slept == [0.5 * attempt for attempt in range(1, attempts)]
    assert isinstance(caught.value.__cause__, ConcurrentWriteError)


def test_retry_reloads_database_without_duplicating_insert(tmp_path: Path) -> None:
    bus = _FakeBus(storage_file=tmp_path / "remote.sqlite")
    db = _make_db(tmp_path, bus=bus, publish_retry_backoff_seconds=0)

    with db:
        db.execute_script("CREATE TABLE events (id INTEGER PRIMARY KEY)")
        bus.conflict_once = True
        db.execute("INSERT INTO events (id) VALUES (?)", (1,))
        result = db.query("SELECT id FROM events")

    assert bus.conflict_count == 1
    assert result.rows == [(1,)]
    assert len(bus.uploads) == 3


def test_manual_close_is_idempotent(tmp_path: Path) -> None:
    db = _make_db(tmp_path)

    db.execute_script("CREATE TABLE events (id INTEGER)")
    db.close()
    db.close()


def test_sync_submit_sets_exception_on_failure(tmp_path: Path) -> None:
    db = _make_db(tmp_path, enable_async_writer=False)

    future = db.submit_execute("BROKEN SQL")

    with pytest.raises(Exception):
        future.result()

    db.close()


def test_close_skips_join_when_writer_thread_is_missing(tmp_path: Path) -> None:
    db = _make_db(tmp_path, enable_async_writer=True)
    db._writer_thread = None

    db.close()


def test_async_submit_execute_many_persists_rows(tmp_path: Path) -> None:
    db = _make_db(tmp_path, enable_async_writer=True)

    with db:
        db.execute_script("CREATE TABLE events (id INTEGER)")

        future = db.submit_execute_many(
            "INSERT INTO events (id) VALUES (?)",
            [(1,), (2,), (3,)],
        )

        assert future.result() == 3

        result = db.query("SELECT COUNT(*) FROM events")

    assert result.rows == [(3,)]


def test_async_submit_execute_script_creates_table(tmp_path: Path) -> None:
    db = _make_db(tmp_path, enable_async_writer=True)

    with db:
        future = db.submit_execute_script(
            """
            CREATE TABLE events (id INTEGER);
            INSERT INTO events (id) VALUES (1);
            INSERT INTO events (id) VALUES (2);
            """
        )

        assert future.result() is None

        result = db.query("SELECT COUNT(*) FROM events")

    assert result.rows == [(2,)]


def test_writer_loop_sets_result_on_success(tmp_path: Path) -> None:
    db = _make_db(tmp_path, enable_async_writer=False)

    future: Future[int] = Future()
    task = _WriteTask(
        operation=lambda repo: 123,
        future=future,
    )

    db._write_queue = Queue()
    db._write_queue.put(task)
    db._write_queue.put(None)

    db._run_write_with_retries = lambda operation: operation(None)  # type: ignore[arg-type,method-assign]

    db._writer_loop()

    assert future.result() == 123


def test_sync_submit_sets_result_on_success(tmp_path: Path) -> None:
    db = _make_db(tmp_path, enable_async_writer=False)

    db.execute_script("CREATE TABLE events (id INTEGER)")

    future = db.submit_execute("INSERT INTO events (id) VALUES (?)", (1,))

    assert future.result() == 1

    result = db.query("SELECT COUNT(*) FROM events")
    assert result.rows == [(1,)]

    db.close()
