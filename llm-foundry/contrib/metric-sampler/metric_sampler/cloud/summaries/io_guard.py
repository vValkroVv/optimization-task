from __future__ import annotations

from pathlib import Path

from metric_sampler.cloud.summaries.clickhouse import ClickHouse, ClickHouseError
from metric_sampler.cloud.summaries.clock import Clock
from metric_sampler.cloud.summaries.log import Log
from metric_sampler.cloud.summaries.loop import Loop
from metric_sampler.cloud.summaries.pressure import IoPressure

MEMINFO_PATH = Path("/proc/meminfo")
# Dirty page cache in kB, and the share of the last minute in which every task waited for IO ("full" avg60), %.
HIGH_DIRTY_KB = 40_000_000
LOW_DIRTY_KB = 10_000_000
HIGH_IO_FULL = 50
LOW_IO_FULL = 30
MAX_PAUSE_SECONDS = 1800
CHECK_SECONDS = 15
START_MERGES = "SYSTEM START MERGES"
STOP_MERGES = "SYSTEM STOP MERGES"


class IoGuard(Loop):
    """Stops ClickHouse merges while the data disk falls behind its writes and starts them again once it has caught up.

    The merges also start after MAX_PAUSE_SECONDS so that parts never pile up for long: merges are the writes that can
    wait. Merges are started when the guard starts too, so a guard that died while they were stopped leaves them
    stopped only until the service restarts it; stop merges by hand only while the guard is not running.
    """

    NAME = "io_guard"

    def __init__(
        self,
        client: ClickHouse,
        clock: Clock,
        log: Log,
        pressure: IoPressure | None = None,
        meminfo_path: Path = MEMINFO_PATH,
    ) -> None:
        super().__init__(self.NAME, clock, log)
        self._client = client
        self._pressure = IoPressure() if pressure is None else pressure
        self._meminfo_path = meminfo_path
        self._paused_at: int | None = None

    def run(self) -> None:
        self._paused_at = None
        self._change_merges(START_MERGES)
        super().run()

    def cycle(self) -> None:
        dirty_kb = self._dirty_kb()
        io_full = self._pressure.full_avg60()
        now = self._clock.seconds()
        if self._paused_at is None:
            self._pause_if_behind(dirty_kb, io_full, now)
        else:
            self._resume_if_caught_up(dirty_kb, io_full, now - self._paused_at)
        self._clock.sleep(CHECK_SECONDS)

    def _pause_if_behind(self, dirty_kb: int, io_full: int, now: int) -> None:
        behind = dirty_kb > HIGH_DIRTY_KB or io_full > HIGH_IO_FULL
        if behind and self._change_merges(STOP_MERGES):
            self._paused_at = now
            self._log.line(f"merges stopped: dirty {dirty_kb} kB, io full {io_full}%")

    def _resume_if_caught_up(self, dirty_kb: int, io_full: int, paused_seconds: int) -> None:
        caught_up = dirty_kb < LOW_DIRTY_KB and io_full < LOW_IO_FULL
        if (caught_up or paused_seconds >= MAX_PAUSE_SECONDS) and self._change_merges(START_MERGES):
            self._paused_at = None
            self._log.line(f"merges started: dirty {dirty_kb} kB, io full {io_full}%")

    def _dirty_kb(self) -> int:
        for line in self._meminfo_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("Dirty:"):
                return int(line.split()[1])
        raise ValueError(f"{self._meminfo_path} has no Dirty line")

    def _change_merges(self, statement: str) -> bool:
        try:
            self._client.execute(statement)
        except ClickHouseError as error:
            self._log.report(error)
            return False
        return True
