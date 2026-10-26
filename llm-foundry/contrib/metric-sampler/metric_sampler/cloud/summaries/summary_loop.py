from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from metric_sampler.cloud.summaries.catalog import Summary
from metric_sampler.cloud.summaries.clickhouse import ClickHouse, ClickHouseError
from metric_sampler.cloud.summaries.clock import Clock
from metric_sampler.cloud.summaries.log import Log
from metric_sampler.cloud.summaries.loop import RETRY_SECONDS, Loop
from metric_sampler.cloud.summaries.slots import STATE_DIRECTORY, InsertSlots

# A chunk's steps are sized for runs of up to CHUNK_RANKS ranks. A chunk's memory grows with its rows, so a run with
# more ranks takes proportionally fewer steps.
CHUNK_RANKS = 256
# A cycle ends after CYCLE_SECONDS so the freshest runs are revisited while older runs are backfilled.
CYCLE_SECONDS = 300
IDLE_SECONDS = 60
# RECENT_DAYS limits the loop to runs whose batches arrived within that many days (0: every run).
RECENT_DAYS = 0
INSERT_THREADS = 4
INSERT_LOG_COMMENT = "summary"

_FRONTIER = "SELECT exp, run, complete_through, last_ingested FROM metric_sampler.complete_steps_latest"
_DERIVED_FRONTIER = (
    "SELECT c.exp AS exp, c.run AS run, least(c.complete_through, p.finished) AS complete_through, "
    "c.last_ingested AS last_ingested\n"
    "    FROM metric_sampler.complete_steps_latest AS c\n"
    "    INNER JOIN (SELECT exp, run, max(complete_through) AS finished FROM metric_sampler.summary_progress "
    "WHERE summary = '{after}' GROUP BY exp, run) AS p ON p.exp = c.exp AND p.run = c.run"
)
_PENDING_RUNS_STATEMENT = (
    "\n"
    "    WITH frontier AS ({frontier}),\n"
    "    done AS (SELECT exp, run, min(done) AS done FROM (SELECT exp, run, summary, max(complete_through) AS done\n"
    "      FROM metric_sampler.summary_progress WHERE summary IN ({progress_names}) GROUP BY exp, run, summary)\n"
    "      GROUP BY exp, run HAVING count() = {progress_count})\n"
    "    SELECT si.exp, si.run, si.max_step, done.done, frontier.complete_through, si.ingested, si.rank_count\n"
    "    FROM metric_sampler.run_steps_latest AS si\n"
    "    LEFT JOIN done ON done.exp = si.exp AND done.run = si.run\n"
    "    LEFT JOIN frontier ON frontier.exp = si.exp AND frontier.run = si.run\n"
    "    WHERE si.max_step > done.done AND ({recent_days} = 0 OR frontier.last_ingested >= "
    "now() - toIntervalDay({recent_days}))\n"
    "    ORDER BY frontier.last_ingested DESC, si.max_step DESC\n"
    "    SETTINGS join_use_nulls = 0"
)
_STEP_STATEMENT = (
    "SELECT step FROM metric_sampler.step_index WHERE exp = '{exp}' AND run = '{run}' AND step >= {first} "
    "GROUP BY step ORDER BY step LIMIT 1 OFFSET {offset}"
)
_INGESTED_STATEMENT = (
    "SELECT max(max_ingested_at) FROM metric_sampler.step_index "
    "WHERE exp = '{exp}' AND run = '{run}' AND step BETWEEN {first} AND {last}"
)


@dataclass(frozen=True)
class PendingRun:
    """A run with ingested steps beyond a summary's progress."""

    exp: str
    run: str
    max_step: int
    finished: int
    frontier: int
    ingested: str
    ranks: int

    @classmethod
    def parse(cls, cells: list[str]) -> PendingRun:
        exp, run, max_step, finished, frontier, ingested, ranks = cells
        return cls(exp, run, int(max_step), int(finished), int(frontier), ingested, int(ranks))

    def chunk_steps(self, summary_chunk_steps: int) -> int:
        return max(1, summary_chunk_steps * CHUNK_RANKS // max(self.ranks, CHUNK_RANKS))


class SummaryLoop(Loop):
    """Keeps one summary table filled behind the ingest.

    For every run with ingested steps beyond the summary's progress it recomputes those steps in chunks. A chunk counts
    as final (the progress moves) only up to the run's complete_steps frontier, where every node's windows have arrived
    without gaps and every family of their batches is ingested; the newest steps are recomputed until they are
    complete.

    Steps summarized over the same ingested rows give the same result, so a chunk is computed again only when rows
    were ingested since: the last computation of a run is kept in a `computed` file of the state directory, and
    deleting it forces a recompute (after emptying the summary table, say). A source bounded to the chunk's steps
    changes only with rows of those steps, so its chunk waits for them. A source over the whole run (gradient and
    timing norms) changes with any row of the run.
    """

    def __init__(
        self,
        summary: Summary,
        client: ClickHouse,
        slots: InsertSlots,
        clock: Clock,
        log: Log,
        state_directory: Path = STATE_DIRECTORY,
    ) -> None:
        super().__init__(summary.name, clock, log)
        self._summary = summary
        self._client = client
        self._slots = slots
        self._state_directory = state_directory

    def cycle(self) -> None:
        try:
            rows = self._client.rows(self._pending_runs_statement)
        except ClickHouseError as error:
            self._log.error_output(error.error_output)
            self._clock.sleep(RETRY_SECONDS)
            return
        cycle_start = self._clock.seconds()
        for cells in rows:
            if self._cycle_over(cycle_start):
                break
            self._catch_up(PendingRun.parse(cells), cycle_start)
        self._clock.sleep(IDLE_SECONDS)

    @cached_property
    def _pending_runs_statement(self) -> str:
        after = self._summary.after
        progress_names = self._summary.progress_names
        return _PENDING_RUNS_STATEMENT.format(
            frontier=_FRONTIER if after is None else _DERIVED_FRONTIER.format(after=after),
            progress_names=", ".join(f"'{name}'" for name in progress_names),
            progress_count=len(progress_names),
            recent_days=RECENT_DAYS,
        )

    def _cycle_over(self, cycle_start: int) -> bool:
        return self._clock.seconds() - cycle_start >= CYCLE_SECONDS

    def _catch_up(self, run: PendingRun, cycle_start: int) -> None:
        steps = run.chunk_steps(self._summary.chunk_steps)
        first = run.finished + 1
        while first <= run.max_step:
            if self._summary.after is not None and first > run.frontier:
                break
            last = self._chunk_end(run, first, steps)
            if not self._summarize(run, first, last) or last > run.frontier:
                break
            first = last + 1
            if self._cycle_over(cycle_start):
                break

    def _chunk_end(self, run: PendingRun, first: int, steps: int) -> int:
        # The steps-th ingested step from here: step numbers can be sparse.
        answer = self._client.value(
            _STEP_STATEMENT.format(exp=run.exp, run=run.run, first=first, offset=steps - 1)
        )
        last = int(answer) if answer else run.max_step
        # A chunk that the frontier cuts ends at the frontier, so progress reaches every complete step; the steps
        # beyond it are recomputed by the next chunk until all their nodes have delivered.
        if run.frontier >= first and last > run.frontier:
            return run.frontier
        return last

    def _summarize(self, run: PendingRun, first: int, last: int) -> bool:
        computed = f"{first}-{last} {self._ingested_rows(run, first, last)}"
        marker = self._state_directory / f"summaries-{self.name}.{run.exp}.{run.run}.computed"
        if self._read_marker(marker) != computed:
            statement = self._summary.chunk_insert(
                run.exp, run.run, first, last, max_threads=INSERT_THREADS, log_comment=INSERT_LOG_COMMENT
            )
            try:
                self._slots.insert(statement)
            except ClickHouseError as error:
                self._log.report(error)
                self._log.error_line(f"{self.name} {run.exp} {run.run} {first}-{last} failed")
                return False
            marker.write_text(f"{computed}\n", encoding="utf-8")
        if last <= run.frontier:
            self._record_progress(run, last)
        self._log.line(f"{self.name} {run.exp} {run.run} {first}-{last} (complete through {run.frontier})")
        return True

    def _ingested_rows(self, run: PendingRun, first: int, last: int) -> str:
        if self._summary.bounded_to_chunk:
            return self._client.value(
                _INGESTED_STATEMENT.format(exp=run.exp, run=run.run, first=first, last=last)
            )
        return run.ingested

    def _read_marker(self, marker: Path) -> str:
        try:
            return marker.read_text(encoding="utf-8").rstrip("\n")
        except FileNotFoundError:
            return ""

    def _record_progress(self, run: PendingRun, last: int) -> None:
        values = ", ".join(
            f"('{name}', '{run.exp}', '{run.run}', {last}, now64(3))" for name in self._summary.progress_names
        )
        try:
            self._client.execute(f"INSERT INTO metric_sampler.summary_progress VALUES {values}")
        except ClickHouseError as error:
            self._log.report(error)
