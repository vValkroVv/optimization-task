from __future__ import annotations

from metric_sampler.cloud.summaries.catalog import DEFAULT_CHUNK_STEPS, Summary
from metric_sampler.cloud.summaries.clickhouse import ClickHouseError
from metric_sampler.cloud.summaries.log import Log
from metric_sampler.cloud.summaries.slots import InsertSlots

INSERT_THREADS = 6
INSERT_LOG_COMMENT = "recompute"


class Recompute:
    """Recomputes a step range of one summary for one run in place, in chunks through the summary loops' slot pool.

    summary_progress is left as it is, so the dashboards keep their routes while the rows are replaced (summary tables
    keep the newest refreshed_at per key). A chunk is computed exactly as the summary's loop computes it. A summary
    derived from another one (`after` in the catalog) is recomputed after that one.
    """

    def __init__(self, slots: InsertSlots, log: Log, chunk_steps: int = DEFAULT_CHUNK_STEPS) -> None:
        self._slots = slots
        self._log = log
        self._chunk_steps = chunk_steps

    def run(self, summary: Summary, exp: str, run: str, first_step: int, last_step: int) -> bool:
        """Whether every chunk was recomputed; the first failed chunk stops the run."""
        first = first_step
        while first <= last_step:
            last = min(first + self._chunk_steps - 1, last_step)
            statement = summary.chunk_insert(
                exp, run, first, last, max_threads=INSERT_THREADS, log_comment=INSERT_LOG_COMMENT
            )
            try:
                self._slots.insert(statement)
            except ClickHouseError as error:
                self._log.report(error)
                self._log.error_line(f"{summary.name} {first}-{last} failed")
                return False
            self._log.line(f"{summary.name} {first}-{last}")
            first = last + 1
        self._log.line(f"{summary.name} done {first_step}-{last_step}")
        return True
