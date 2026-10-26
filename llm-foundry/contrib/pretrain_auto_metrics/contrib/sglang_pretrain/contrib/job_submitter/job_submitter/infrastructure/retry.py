from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from functools import wraps
from typing import Callable, ParamSpec, TypeVar

logger = logging.getLogger(__name__)

_P = ParamSpec("_P")
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True, kw_only=True)
class RetryPolicy:
    exceptions: tuple[type[Exception], ...]
    delay: Callable[[int, Exception], float]
    attempts: int = 3
    should_retry: Callable[[Exception], bool] | None = None
    jitter_seconds: float = 0
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("attempts must be >= 1")
        if self.jitter_seconds < 0:
            raise ValueError("jitter_seconds must be >= 0")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    def run(
        self, operation: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs
    ) -> _T:
        started_at = time.monotonic()
        attempt = 1
        while True:
            try:
                return operation(*args, **kwargs)
            except self.exceptions as error:
                if attempt == self.attempts:
                    raise
                if self.should_retry is not None and not self.should_retry(error):
                    raise
                wait_seconds = self.delay(attempt, error)
                if wait_seconds > 0 and self.jitter_seconds > 0:
                    wait_seconds += random.uniform(0, self.jitter_seconds)
                if (
                    self.timeout_seconds is not None
                    and time.monotonic() - started_at + wait_seconds
                    >= self.timeout_seconds
                ):
                    raise
                logger.warning(
                    "%s attempt %s/%s failed (%s), retrying in %ss",
                    operation.__qualname__,
                    attempt,
                    self.attempts,
                    error,
                    wait_seconds,
                )
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
                if (
                    self.timeout_seconds is not None
                    and time.monotonic() - started_at >= self.timeout_seconds
                ):
                    raise
                attempt += 1

    def __call__(self, operation: Callable[_P, _T]) -> Callable[_P, _T]:
        @wraps(operation)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            return self.run(operation, *args, **kwargs)

        return wrapped
