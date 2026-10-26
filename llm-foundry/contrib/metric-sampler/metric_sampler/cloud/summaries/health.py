from __future__ import annotations

from dataclasses import dataclass

from metric_sampler.cloud.summaries.clickhouse import ClickHouse, ClickHouseError
from metric_sampler.cloud.summaries.clock import Clock
from metric_sampler.cloud.summaries.log import Log
from metric_sampler.cloud.summaries.loop import RETRY_SECONDS, Loop

IDLE_SECONDS = 30

PENDING_STATEMENT = (
    "SELECT exp, run, attempt, batch, window_first, window_last FROM metric_sampler.health_pending_windows\n"
    "    ORDER BY collected_at DESC, exp, run, attempt, batch, window_first DESC LIMIT 1 FORMAT TSV"
)
GENERATION_STATEMENT = "SELECT generateUUIDv4()"
# The sort key's granules overshoot the window: without optimize_move_to_prewhere_if_final, FINAL merges the rows
# around it too and drops them afterwards.
SNAPSHOT_STATEMENT = (
    "INSERT INTO metric_sampler.health_snapshots\n"
    "      (exp, run, attempt, batch, window_first, generation, step, nodes_present, nodes_expected,\n"
    "       contributors_total, expected_contributors_total, identical_violations)\n"
    "    SELECT exp, run, attempt, '{batch}', {first}, toUUID('{generation}'), step, nodes_present, nodes_expected,\n"
    "      contributors_total, expected_contributors_total, identical_violations\n"
    "    FROM metric_sampler.health_per_step_window(exp_filter = '{exp}', run_filter = '{run}', "
    "attempt_filter = '{attempt}',\n"
    "      step_from = {first}, step_to = {last})\n"
    "    SETTINGS optimize_aggregation_in_order = 0, optimize_move_to_prewhere_if_final = 1, max_threads = 4,\n"
    "      max_memory_usage = 34359738368, max_execution_time = 900, log_comment = 'health-refresh'"
)
# The timestamp is assigned after the complete snapshot insert, never from the source batch.
REFRESH_STATEMENT = (
    "INSERT INTO metric_sampler.health_refreshes\n"
    "      (exp, run, attempt, batch, window_first, window_last, generation, steps, completed_at)\n"
    "    SELECT '{exp}', '{run}', '{attempt}', '{batch}', {first}, {last}, toUUID('{generation}'),\n"
    "      (SELECT count() FROM metric_sampler.health_snapshots\n"
    "       WHERE exp = '{exp}' AND run = '{run}' AND attempt = '{attempt}' AND batch = '{batch}' "
    "AND window_first = {first} AND generation = toUUID('{generation}')),\n"
    "      greatest(now64(6), (SELECT max(completed_at) + toIntervalMicrosecond(1) "
    "FROM metric_sampler.health_refreshes))"
)


@dataclass(frozen=True)
class PendingWindow:
    exp: str
    run: str
    attempt: str
    batch: str
    first: str
    last: str

    @classmethod
    def parse(cls, row: str) -> PendingWindow:
        exp, run, attempt, batch, first, last = row.split("\t")
        return cls(exp, run, attempt, batch, first, last)

    @property
    def description(self) -> str:
        return f"{self.exp} {self.run} {self.attempt} {self.batch} {self.first}-{self.last}"


class Health(Loop):
    """The single worker that refreshes the health snapshots, one pending window at a time, newest batches first.

    The same queue backfills old batches. It runs apart from the ingest pass, which the refresh would slow on a large
    run. A window's snapshot is published by its health_refreshes row, inserted only after the complete snapshot is
    stored, so retries and overlapping batches replace complete per-step values and never add totals.
    """

    NAME = "health"

    def __init__(self, client: ClickHouse, clock: Clock, log: Log) -> None:
        super().__init__(self.NAME, clock, log)
        self._client = client

    def cycle(self) -> None:
        try:
            row = self._client.value(PENDING_STATEMENT)
            if not row:
                self._clock.sleep(IDLE_SECONDS)
                return
            generation = self._client.value(GENERATION_STATEMENT)
        except ClickHouseError as error:
            self._log.error_output(error.error_output)
            self._clock.sleep(RETRY_SECONDS)
            return
        window = PendingWindow.parse(row)
        for statement, failure in ((SNAPSHOT_STATEMENT, "snapshot"), (REFRESH_STATEMENT, "refresh")):
            try:
                self._client.execute(self._format(statement, window, generation))
            except ClickHouseError as error:
                self._log.report(error)
                self._log.error_line(f"{window.description} {failure} failed")
                self._clock.sleep(RETRY_SECONDS)
                return

    def _format(self, statement: str, window: PendingWindow, generation: str) -> str:
        return statement.format(
            exp=window.exp, run=window.run, attempt=window.attempt, batch=window.batch, first=window.first,
            last=window.last, generation=generation,
        )
