from __future__ import annotations

import threading
from pathlib import Path

from metric_sampler.cloud.summaries.catalog import SUMMARIES
from metric_sampler.cloud.summaries.clickhouse import ClickHouse
from metric_sampler.cloud.summaries.clock import Clock
from metric_sampler.cloud.summaries.cold_mover import ColdMover
from metric_sampler.cloud.summaries.cold_tier import ColdTier
from metric_sampler.cloud.summaries.health import Health
from metric_sampler.cloud.summaries.hist_latest import HistLatest
from metric_sampler.cloud.summaries.io_guard import IoGuard
from metric_sampler.cloud.summaries.log import FileLog, Log
from metric_sampler.cloud.summaries.loop import RETRY_SECONDS, Loop
from metric_sampler.cloud.summaries.pressure import IoPressure
from metric_sampler.cloud.summaries.slots import STATE_DIRECTORY, InsertSlots
from metric_sampler.cloud.summaries.summary_loop import SummaryLoop
from metric_sampler.cloud.summaries.top_cards import TopCards


class Service:
    """Runs every loop of the summaries in one process, one thread per loop, each logging to its own file.

    A loop that raises is logged in one line and restarted after the retry delay. The service ends when the clock is
    stopped.
    """

    def __init__(
        self,
        client: ClickHouse,
        clock: Clock,
        log: Log,
        log_directory: Path,
        state_directory: Path = STATE_DIRECTORY,
    ) -> None:
        self._client = client
        self._clock = clock
        self._log = log
        self._log_directory = log_directory
        self._state_directory = state_directory

    def run(self) -> None:
        self._log_directory.mkdir(parents=True, exist_ok=True)
        for loop in self.create_loops():
            self._log.line(f"started {loop.name}")
            threading.Thread(target=self.supervise, args=(loop,), name=loop.name, daemon=True).start()
        self._clock.stopped.wait()

    def create_loops(self) -> list[Loop]:
        slots = InsertSlots(self._client, self._clock, self._state_directory)
        cold_log = self._file_log(ColdTier.NAME)
        mover = ColdMover(self._client, self._clock, cold_log, IoPressure())
        loops: list[Loop] = [
            SummaryLoop(
                summary, self._client, slots, self._clock, self._file_log(summary.name), self._state_directory
            )
            for summary in SUMMARIES
        ]
        loops += [
            HistLatest(self._client, self._clock, self._file_log(HistLatest.NAME)),
            TopCards(self._client, self._clock, self._file_log(TopCards.NAME)),
            Health(self._client, self._clock, self._file_log(Health.NAME)),
            IoGuard(self._client, self._clock, self._file_log(IoGuard.NAME)),
            ColdTier(self._client, self._clock, cold_log, mover),
        ]
        return loops

    def supervise(self, loop: Loop) -> None:
        while not self._clock.stopped.is_set():
            try:
                loop.run()
                return
            except Exception as error:
                self._log.error_line(f"{loop.name} failed: {type(error).__name__}: {' '.join(str(error).split())}")
                self._clock.sleep(RETRY_SECONDS)
                if not self._clock.stopped.is_set():
                    self._log.line(f"started {loop.name}")

    def _file_log(self, name: str) -> Log:
        return FileLog(self._log_directory / f"{name}.log", self._clock)
