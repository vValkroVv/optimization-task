from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from pathlib import Path

from metric_sampler.cloud.summaries.clickhouse import ClickHouse, ClickHouseError
from metric_sampler.cloud.summaries.clock import TIMESTAMP_FORMAT, Clock

START_TIME = 1_700_000_000
START_TIMESTAMP = "2023-11-14T22:13:20Z"
CURL_ERROR = "curl: (22) The requested URL returned error: 500\n"
MEMORY_ERROR = "Code: 241. DB::Exception: Memory limit (total) exceeded. (MEMORY_LIMIT_EXCEEDED)\n"


def stamp(seconds_after_start: int = 0) -> str:
    return time.strftime(TIMESTAMP_FORMAT, time.gmtime(START_TIME + seconds_after_start))


def failure(output: str = "", error_output: str = CURL_ERROR, status: int = 22) -> ClickHouseError:
    return ClickHouseError(status, output, error_output)


class FakeClock(Clock):
    """A clock that moves only when a loop sleeps, and stops the service after `stop_after` sleeps."""

    def __init__(self, stop_after: int | None = None, on_sleep: Callable[[int], None] | None = None) -> None:
        super().__init__()
        self.now = START_TIME
        self.sleeps: list[float] = []
        self._stop_after = stop_after
        self.on_sleep = on_sleep

    def seconds(self) -> int:
        return self.now

    def timestamp(self) -> str:
        return time.strftime(TIMESTAMP_FORMAT, time.gmtime(self.now))

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += int(seconds)
        if self.on_sleep is not None:
            self.on_sleep(len(self.sleeps))
        if self._stop_after is not None and len(self.sleeps) >= self._stop_after:
            self.stopped.set()


class FakeClickHouse(ClickHouse):
    """Records every statement and answers the first rule whose pattern the statement matches.

    A rule's answers are used in turn, the last one repeating: a string is the response body, an exception is raised.
    A statement that no rule matches is answered with an empty body. `advance` moves the clock by that many seconds
    after the statements that match a pattern.
    """

    def __init__(self, rules: dict[str, list[str | ClickHouseError]] | None = None) -> None:
        super().__init__()
        self.statements: list[str] = []
        self._rules = {re.compile(pattern, re.S): answers for pattern, answers in (rules or {}).items()}
        self._calls = dict.fromkeys(self._rules, 0)
        self._lock = threading.Lock()
        self.on_execute: Callable[[str], None] | None = None

    def execute(self, statement: str) -> str:
        with self._lock:
            self.statements.append(statement)
            answer = self._answer(statement)
        if self.on_execute is not None:
            self.on_execute(statement)
        if isinstance(answer, ClickHouseError):
            raise answer
        return answer

    def matching(self, pattern: str) -> list[str]:
        return [statement for statement in self.statements if re.search(pattern, statement, re.S)]

    def _answer(self, statement: str) -> str | ClickHouseError:
        for pattern, answers in self._rules.items():
            if pattern.search(statement):
                call = self._calls[pattern]
                self._calls[pattern] += 1
                return answers[min(call, len(answers) - 1)]
        return ""


def computed_files(directory: Path) -> dict[str, str]:
    return {path.name: path.read_text(encoding="utf-8") for path in sorted(directory.glob("summaries-*.computed"))}


def compact(statement: str) -> str:
    return " ".join(statement.split())
