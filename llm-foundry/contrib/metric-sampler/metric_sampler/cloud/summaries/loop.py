from __future__ import annotations

from abc import ABC, abstractmethod

from metric_sampler.cloud.summaries.clock import Clock
from metric_sampler.cloud.summaries.log import Log

RETRY_SECONDS = 60


class Loop(ABC):
    """A job of the summaries service that repeats its cycle until the service stops."""

    def __init__(self, name: str, clock: Clock, log: Log) -> None:
        self.name = name
        self._clock = clock
        self._log = log

    def run(self) -> None:
        while not self._clock.stopped.is_set():
            self.cycle()

    @abstractmethod
    def cycle(self) -> None:
        """One pass of the job, including the wait before the next one."""
