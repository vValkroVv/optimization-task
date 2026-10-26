from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from metric_sampler.contract.raw_layout import (
    LOGGED_DATA_NAME,
    LOGGED_DIR_NAME,
    MANIFEST_NAME,
    window_name,
    write_atomically,
)

_MANIFEST_SCHEMA_VERSION = 2
# Sized for a per-parameter weight or optimizer monitor, which emits thousands of scalars in one batch.
_QUEUE_SIZE = 65536
_POLL_INTERVAL_S = 0.1

_PARQUET_SCHEMA = pa.schema([
    ("step", pa.uint32()), ("name", pa.string()), ("value", pa.float64()), ("source_step", pa.uint32()),
])


@dataclass(frozen=True)
class _Row:
    step: int
    name: str
    value: float
    source_step: int | None


class LoggedWriter:
    """Writes scalars logged by training monitors to `attempt_dir/logged/w<a>-<b>/`.

    `add` queues without waiting for disk and raises if the queue is full or the writer failed.
    Queued rows are only durable after their window manifest is written."""

    def __init__(self, attempt_dir: Path, window_steps: int) -> None:
        self._attempt_dir = attempt_dir
        self._window_steps = window_steps
        self._queue: queue.Queue[_Row] = queue.Queue(maxsize=_QUEUE_SIZE)
        self._error: BaseException | None = None
        self._remaining_rows: dict[int, list[_Row]] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def add(self, step: int, name: str, value: float, source_step: int | None = None) -> None:
        if self._stop.is_set():
            raise BufferError("logged metric writer is closed")
        if self._error is not None:
            raise BufferError("logged metric background writer failed") from self._error
        try:
            self._queue.put_nowait(_Row(step, name, value, source_step))
        except queue.Full as error:
            raise BufferError("logged metric queue is full; row was not accepted") from error

    def close(self) -> None:
        self._stop.set()
        self._thread.join()
        while True:
            try:
                row = self._queue.get_nowait()
            except queue.Empty:
                break
            self._remaining_rows.setdefault(row.step, []).append(row)
        try:
            while self._remaining_rows:
                self._write_window(self._remaining_rows)
        except (OSError, pa.ArrowException) as error:
            raise BufferError("logged metric window could not be persisted") from error
        if self._error is not None:
            failure, self._error = self._error, None
            raise BufferError("logged metric background writer failed") from failure

    def _run(self) -> None:
        rows_by_step: dict[int, list[_Row]] = {}
        try:
            while True:
                try:
                    row = self._queue.get(timeout=_POLL_INTERVAL_S)
                except queue.Empty:
                    if self._stop.is_set():
                        break
                    continue
                rows_by_step.setdefault(row.step, []).append(row)
                if len(rows_by_step) > self._window_steps:
                    self._write_window(rows_by_step)
        except BaseException as error:  # reported by add() and close(), never swallowed
            self._error = error
        self._remaining_rows = rows_by_step

    def _write_window(self, rows_by_step: dict[int, list[_Row]]) -> None:
        steps = sorted(rows_by_step)[:self._window_steps]
        rows: list[_Row] = []
        for step in steps:
            rows.extend(rows_by_step[step])

        window_dir = self._window_dir(steps[0], steps[-1])
        window_dir.mkdir(parents=True, exist_ok=True)
        self._write_parquet(window_dir / LOGGED_DATA_NAME, rows)
        self._write_manifest(window_dir / MANIFEST_NAME, steps[0], steps[-1], len(rows), dropped_rows=0)
        for step in steps:
            del rows_by_step[step]

    @staticmethod
    def _write_parquet(path: Path, rows: list[_Row]) -> None:
        table = pa.table(
            {
                "step": [row.step for row in rows],
                "name": [row.name for row in rows],
                "value": [row.value for row in rows],
                "source_step": [row.source_step for row in rows],
            },
            schema=_PARQUET_SCHEMA,
        )
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink, compression="zstd")
        write_atomically(path, sink.getvalue().to_pybytes())

    @staticmethod
    def _write_manifest(path: Path, first_step: int, last_step: int, row_count: int, dropped_rows: int) -> None:
        manifest = {
            "schema_version": _MANIFEST_SCHEMA_VERSION,
            "first_step": first_step,
            "last_step": last_step,
            "rows": row_count,
            "dropped_rows": dropped_rows,
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
        write_atomically(path, json.dumps(manifest, indent=2).encode("utf-8"))

    def _window_dir(self, first_step: int, last_step: int) -> Path:
        return self._attempt_dir / LOGGED_DIR_NAME / window_name(first_step, last_step)
