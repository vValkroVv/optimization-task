from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import TextIO

from metric_sampler.cloud.summaries.clickhouse import ClickHouseError
from metric_sampler.cloud.summaries.clock import Clock


class Log:
    """Timestamped lines and raw output on the standard streams, the way the scripts wrote them."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._lock = threading.Lock()

    def line(self, message: str) -> None:
        self._write_out(f"{self._clock.timestamp()} {message}\n")

    def error_line(self, message: str) -> None:
        self._write_error(f"{self._clock.timestamp()} {message}\n")

    def output(self, text: str) -> None:
        self._write_out(text)

    def error_output(self, text: str) -> None:
        self._write_error(text)

    def report(self, error: ClickHouseError) -> None:
        """Writes what the failed statement answered, its body first and then its error."""
        self.output(error.output)
        self.error_output(error.error_output)

    def _write_out(self, text: str) -> None:
        self._write(sys.stdout, text)

    def _write_error(self, text: str) -> None:
        self._write(sys.stderr, text)

    def _write(self, stream: TextIO, text: str) -> None:
        with self._lock:
            stream.write(text)
            stream.flush()


class FileLog(Log):
    """Appends both streams to one file, opening it for every write so that a rotated file is simply recreated."""

    def __init__(self, path: Path, clock: Clock) -> None:
        super().__init__(clock)
        self._path = path

    def _write_out(self, text: str) -> None:
        self._append(text)

    def _write_error(self, text: str) -> None:
        self._append(text)

    def _append(self, text: str) -> None:
        with self._lock, self._path.open("a", encoding="utf-8") as file:
            file.write(text)
