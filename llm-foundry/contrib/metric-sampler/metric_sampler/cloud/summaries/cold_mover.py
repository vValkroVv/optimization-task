from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from metric_sampler.cloud.summaries.clickhouse import ClickHouse, ClickHouseError
from metric_sampler.cloud.summaries.clock import Clock
from metric_sampler.cloud.summaries.log import Log
from metric_sampler.cloud.summaries.pressure import IoPressure

PARALLEL = 6
# "full" avg10 of the IO pressure above this pauses the moves (they read the data disk while ingest writes it).
MAX_IO_FULL = 50
PRESSURE_WAIT_SECONDS = 30
LOCAL_DISK = "default"
COLD_DISK = "obs_cold_cached"

_ROWS_BY_DISK_STATEMENT = (
    "SELECT table, disk_name, count(), sum(rows) FROM system.parts WHERE database = 'metric_sampler'\n"
    "    AND table = '{table}' AND active AND partition = '{partition}' GROUP BY table, disk_name "
    "ORDER BY disk_name FORMAT TSV"
)
_PARTS_STATEMENT = (
    "SELECT name FROM system.parts WHERE database = 'metric_sampler' AND table = '{table}' AND active\n"
    "    AND partition = '{partition}' AND disk_name = '{disk}' AND modification_time <= now() - "
    "toIntervalHour({min_age_hours})\n"
    "    AND bytes_on_disk >= {min_bytes} ORDER BY bytes_on_disk DESC FORMAT TSV"
)
_MOVE_STATEMENT = "ALTER TABLE metric_sampler.{table} MOVE PART '{part}' TO VOLUME 'cold'"
_CHECK_STATEMENT = (
    "SELECT * FROM metric_sampler.{table} WHERE _part = '{part}' FORMAT Null\n"
    "    SETTINGS checksum_on_read = 1, enable_filesystem_cache = 0, max_threads = 2"
)


class ColdMover:
    """Moves the parts of one run (one partition) of the given tables to the cold volume of the 'tiered' policy.

    PARALLEL parts move at a time: ClickHouse moves the parts of a table one after another, while parallel MOVE PART
    queries add up. The mover waits while the data disk is under write pressure, then reads every column of every part
    it moved back from the cold volume, which checks each compressed block against its checksum, and logs the run's
    rows on each volume before and after, which must add up. The read bypasses the local cache, which would otherwise
    write all of it to the data disk; CHECK TABLE ... PART would verify whole files but always fills the cache.
    """

    def __init__(
        self,
        client: ClickHouse,
        clock: Clock,
        log: Log,
        pressure: IoPressure,
        parallel: int = PARALLEL,
        max_io_full: int = MAX_IO_FULL,
    ) -> None:
        self._client = client
        self._clock = clock
        self._log = log
        self._pressure = pressure
        self._parallel = parallel
        self._max_io_full = max_io_full

    def move_run(
        self, exp: str, run: str, tables: Sequence[str], min_age_hours: int = 0, min_bytes: int = 0
    ) -> None:
        """Moves the parts written at least `min_age_hours` ago and of at least `min_bytes` (all parts by default)."""
        # system.parts shows the partition as ('<exp>','<run>'); doubled quotes put that in a string literal.
        partition = f"(''{exp}'',''{run}'')"
        for table in tables:
            self._log.line(f"{table} before:")
            self._log_rows_by_disk(table, partition)
            listed = self._parts(table, partition, LOCAL_DISK, min_age_hours, min_bytes)
            self._move_all(table, listed)
            self._log.line(f"{table} checking")
            on_cold_disk = self._parts(table, partition, COLD_DISK, 0, 0)
            moved = [part for part in on_cold_disk if part in listed]
            self._check_all(table, moved)
            self._log.line(f"{table} after:")
            self._log_rows_by_disk(table, partition)

    def _log_rows_by_disk(self, table: str, partition: str) -> None:
        try:
            self._log.output(self._client.execute(_ROWS_BY_DISK_STATEMENT.format(table=table, partition=partition)))
        except ClickHouseError as error:
            self._log.report(error)

    def _parts(self, table: str, partition: str, disk: str, min_age_hours: int, min_bytes: int) -> list[str]:
        statement = _PARTS_STATEMENT.format(
            table=table, partition=partition, disk=disk, min_age_hours=min_age_hours, min_bytes=min_bytes
        )
        try:
            return self._client.execute(statement).split()
        except ClickHouseError as error:
            self._log.report(error)
            return []

    def _move_all(self, table: str, parts: list[str]) -> None:
        for batch in self._batches(parts):
            self._wait_for_disk()
            self._run_parallel(partial(self._move_part, table), batch)

    def _check_all(self, table: str, parts: list[str]) -> None:
        for batch in self._batches(parts):
            self._run_parallel(partial(self._check_part, table), batch)

    def _batches(self, parts: list[str]) -> list[list[str]]:
        # Parts of similar size are together, as the lists are ordered by size. xargs runs its command once even
        # when the list is empty, so an empty list is one empty batch.
        return [parts[start:start + self._parallel] for start in range(0, len(parts), self._parallel)] or [[]]

    def _run_parallel(self, action: Callable[[str], None], batch: list[str]) -> None:
        if batch:
            with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                list(pool.map(action, batch))

    def _wait_for_disk(self) -> None:
        while self._pressure.full_avg10() > self._max_io_full:
            self._clock.sleep(PRESSURE_WAIT_SECONDS)

    def _move_part(self, table: str, part: str) -> None:
        try:
            self._client.execute(_MOVE_STATEMENT.format(table=table, part=part))
        except ClickHouseError as error:
            self._log.report(error)
            self._log.error_line(f"{table} {part} move failed")

    def _check_part(self, table: str, part: str) -> None:
        try:
            self._client.execute(_CHECK_STATEMENT.format(table=table, part=part))
        except ClickHouseError as error:
            self._log.report(error)
            self._log.error_line(f"{table} {part} check failed")
