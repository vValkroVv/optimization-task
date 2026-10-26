from __future__ import annotations

import sqlite3
import tempfile
import threading
from collections.abc import Iterator, Sequence
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Generic, TypeVar, cast

from job_submitter.infrastructure.s3.exceptions import (
    ConcurrentWriteError,
    DatabaseSyncError,
)
from job_submitter.infrastructure.retry import RetryPolicy
from job_submitter.core.storage.interfaces import Database
from job_submitter.core.storage.query import QueryResult
from job_submitter.infrastructure.s3.bus import S3SQLiteBus
from job_submitter.infrastructure.s3.config import S3StorageConfig
from job_submitter.infrastructure.s3.constants import LOCAL_DB_FILENAME, TEMP_DIR_PREFIX
from job_submitter.infrastructure.s3.reader import RemoteSQLiteS3Reader
from job_submitter.infrastructure.s3.repository import SQLiteRepository
from job_submitter.infrastructure.s3.sync_state import RemoteObjectState

_T = TypeVar("_T")


@dataclass(slots=True)
class _WriteTask(Generic[_T]):
    operation: Callable[[SQLiteRepository], _T]
    future: Future[_T]


class S3SQLiteDatabase(Database):
    def __init__(
        self,
        *,
        bus: S3SQLiteBus,
        config: S3StorageConfig,
        reader: RemoteSQLiteS3Reader | None = None,
        connect: Callable[[str], sqlite3.Connection] = sqlite3.connect,
        enable_async_writer: bool = True,
        max_pending_writes: int = 1024,
        max_publish_attempts: int = 3,
        publish_retry_backoff_seconds: float = 0.1,
    ) -> None:
        self._bus = bus
        self._config = config
        self._reader = reader
        self._connect = connect
        self._max_publish_attempts = max(1, max_publish_attempts)
        self._publish_retry_backoff_seconds = max(0.0, publish_retry_backoff_seconds)

        self._pending_futures: set[Future[Any]] = set()
        self._pending_lock = threading.Lock()

        self._write_queue: Queue[_WriteTask[Any] | None] | None = None
        self._writer_thread: threading.Thread | None = None
        self._closed = False

        if enable_async_writer:
            queue_size = max(1, max_pending_writes)
            self._write_queue = Queue(maxsize=queue_size)
            self._writer_thread = threading.Thread(
                target=self._writer_loop,
                name="s3-sqlite-writer",
                daemon=True,
            )
            self._writer_thread.start()

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        return self._run_write_with_retries(lambda repo: repo.execute(sql, params))

    def query(self, sql: str, params: Sequence[Any] = ()) -> QueryResult:
        self._flush_writes()
        with self._workdir() as (local_db_path, _):
            conn = self._connect(str(local_db_path))
            try:
                repo = SQLiteRepository(conn)
                return repo.query(sql, params)
            finally:
                conn.close()

    def select(self, sql: str, params: Sequence[Any] = ()) -> QueryResult:
        self._flush_writes()
        if self._reader is not None:
            return self._reader.query(sql, params)
        return self.query(sql, params)

    def execute_many(self, sql: str, params_list: Sequence[Sequence[Any]]) -> int:
        return self._run_write_with_retries(
            lambda repo: repo.execute_many(sql, params_list)
        )

    def execute_script(self, script: str) -> None:
        self._run_write_with_retries(lambda repo: repo.execute_script(script))

    def submit_execute(self, sql: str, params: Sequence[Any] = ()) -> Future[int]:
        return self._submit_write(lambda repo: repo.execute(sql, params))

    def submit_execute_many(
        self,
        sql: str,
        params_list: Sequence[Sequence[Any]],
    ) -> Future[int]:
        return self._submit_write(lambda repo: repo.execute_many(sql, params_list))

    def submit_execute_script(self, script: str) -> Future[None]:
        return self._submit_write(lambda repo: repo.execute_script(script))

    def __enter__(self) -> S3SQLiteDatabase:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._flush_writes()

        if self._write_queue is None:
            return

        self._write_queue.put(None)

        if self._writer_thread is not None:
            self._writer_thread.join()

    def _flush_writes(self) -> None:
        while True:
            with self._pending_lock:
                if not self._pending_futures:
                    return
                pending_snapshot = list(self._pending_futures)

            for future in pending_snapshot:
                future.result()

    def _submit_write(self, operation: Callable[[SQLiteRepository], _T]) -> Future[_T]:
        future: Future[_T] = Future()

        if self._write_queue is None:
            try:
                result = self._run_write_with_retries(operation)
            except Exception as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)
            return future

        write_queue = cast(Queue[_WriteTask[Any] | None], self._write_queue)
        task: _WriteTask[_T] = _WriteTask(operation=operation, future=future)

        with self._pending_lock:
            self._pending_futures.add(cast(Future[Any], future))

        future.add_done_callback(self._remove_pending_future)
        write_queue.put(cast(_WriteTask[Any], task))
        return future

    def _writer_loop(self) -> None:
        write_queue = cast(Queue[_WriteTask[Any] | None], self._write_queue)

        while True:
            item = write_queue.get()
            if item is None:
                return

            task = cast(_WriteTask[Any], item)
            try:
                result = self._run_write_with_retries(task.operation)
            except Exception as exc:
                task.future.set_exception(exc)
            else:
                task.future.set_result(result)

    def _run_write_with_retries(
        self,
        operation: Callable[[SQLiteRepository], _T],
    ) -> _T:
        policy = RetryPolicy(
            exceptions=(ConcurrentWriteError,),
            attempts=self._max_publish_attempts,
            delay=lambda attempt, error: self._publish_retry_backoff_seconds * attempt,
        )
        try:
            return policy.run(self._run_single_write, operation)
        except ConcurrentWriteError as error:
            raise DatabaseSyncError(
                "Failed to persist changes due to concurrent updates"
            ) from error

    def _run_single_write(self, operation: Callable[[SQLiteRepository], _T]) -> _T:
        with self._workdir() as (local_db_path, remote_state):
            conn = self._connect(str(local_db_path))
            repo = SQLiteRepository(conn)

            try:
                result = operation(repo)
                repo.commit()
            except Exception:
                repo.rollback()
                raise
            finally:
                conn.close()

            self._bus.upload(local_db_path, expected_state=remote_state)
            return result

    def _remove_pending_future(self, future: Future[Any]) -> None:
        with self._pending_lock:
            self._pending_futures.discard(future)

    @contextmanager
    def _workdir(self) -> Iterator[tuple[Path, RemoteObjectState | None]]:
        with tempfile.TemporaryDirectory(prefix=TEMP_DIR_PREFIX) as temp_dir:
            local_db_path = Path(temp_dir) / LOCAL_DB_FILENAME
            remote_state = self._bus.download_or_initialize(local_db_path)
            yield local_db_path, remote_state
