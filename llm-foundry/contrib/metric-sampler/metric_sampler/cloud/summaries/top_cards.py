from __future__ import annotations

from metric_sampler.cloud.summaries.clickhouse import ClickHouse, ClickHouseError
from metric_sampler.cloud.summaries.clock import Clock
from metric_sampler.cloud.summaries.log import Log
from metric_sampler.cloud.summaries.loop import RETRY_SECONDS, Loop

INTERVAL_SECONDS = 600
RUN_SUMMARIES = ("grad_per_rank", "timing_per_rank", "activation_amax")
MICROBATCH_SUMMARIES = ("activation_amax_mb",)

_QUERY_SETTINGS = (
    "SETTINGS max_threads = 8, priority = 10, max_execution_time = 1800, log_comment = 'summary-top-cards'"
)
_DUE_RUNS_STATEMENT = (
    "\n"
    "    WITH changed AS (SELECT exp, run, max(refreshed_at) AS changed FROM metric_sampler.summary_progress\n"
    "                     WHERE summary IN ('grad_per_rank', 'timing_per_rank', 'activation_amax', "
    "'activation_amax_mb') GROUP BY exp, run),\n"
    "    cards AS (SELECT exp, run, max(refreshed_at) AS refreshed FROM metric_sampler.summary_top_cards "
    "GROUP BY exp, run)\n"
    "    SELECT c.exp, c.run FROM changed AS c LEFT JOIN cards AS t ON t.exp = c.exp AND t.run = c.run\n"
    "    LEFT JOIN metric_sampler.run_steps_latest AS i ON i.exp = c.exp AND i.run = c.run\n"
    "    WHERE t.refreshed < c.changed OR (t.refreshed < now() - INTERVAL 1 HOUR AND i.ingested > t.refreshed)\n"
    "    ORDER BY c.changed DESC\n"
    "    SETTINGS join_use_nulls = 0"
)
_COVERED_STATEMENT = (
    "SELECT count() = {count} AND min(done) + 50 >= (SELECT max(step) FROM metric_sampler.step_index "
    "WHERE exp = '{exp}' AND run = '{run}')\n"
    "          FROM (SELECT summary, max(complete_through) AS done FROM metric_sampler.summary_progress\n"
    "                WHERE exp = '{exp}' AND run = '{run}' AND summary IN ({summaries}) GROUP BY summary)"
)
_RUN_INSERT_STATEMENT = (
    "INSERT INTO metric_sampler.summary_top_cards\n"
    "        SELECT now64(3), exp, run, -2, 'grad', node, rank, max(abs(robust_z)) "
    "FROM metric_sampler.grad_per_rank_fast(exp_filter = '{exp}', run_filter = '{run}')\n"
    "        WHERE exp = '{exp}' AND run = '{run}' GROUP BY exp, run, node, rank\n"
    "        UNION ALL\n"
    "        SELECT now64(3), exp, run, -2, 'timing', node, rank, max(abs(robust_z)) "
    "FROM metric_sampler.timing_per_rank_fast(exp_filter = '{exp}', run_filter = '{run}')\n"
    "        WHERE exp = '{exp}' AND run = '{run}' GROUP BY exp, run, node, rank\n"
    "        UNION ALL\n"
    "        SELECT now64(3), exp, run, -2, 'activation', node, rank, max_abs_z\n"
    "        FROM metric_sampler.activation_per_rank_peak(exp_filter = '{exp}', run_filter = '{run}')\n"
    "        " + _QUERY_SETTINGS
)
_MICROBATCHES_STATEMENT = (
    "SELECT DISTINCT microbatch FROM metric_sampler.summary_activation_amax_mb "
    "WHERE exp = '{exp}' AND run = '{run}' ORDER BY microbatch"
)
_MICROBATCH_INSERT_STATEMENT = (
    "INSERT INTO metric_sampler.summary_top_cards\n"
    "              SELECT now64(3), exp, run, {microbatch}, 'activation', node, rank, max_abs_z\n"
    "              FROM metric_sampler.activation_per_rank_mbpeak(exp_filter = '{exp}', run_filter = '{run}', "
    "microbatch_filter = {microbatch})\n"
    "              " + _QUERY_SETTINGS
)


class TopCards(Loop):
    """Recomputes the node-rank "Top 20 cards" rows of the runs that its summaries cover and that changed.

    A run is due when its gradient, timing or activation-amax summaries changed since its rows were computed, or an
    hour ago for a run with rows ingested since. The rows are the largest |robust z| of every card for gradient norms,
    GPU timing and activation amax, over all microbatches and, when the per-microbatch amax summary covers the run,
    for each microbatch, exactly as the panel computes them live.
    """

    NAME = "top_cards"

    def __init__(self, client: ClickHouse, clock: Clock, log: Log) -> None:
        super().__init__(self.NAME, clock, log)
        self._client = client

    def cycle(self) -> None:
        try:
            runs = self._client.rows(_DUE_RUNS_STATEMENT)
        except ClickHouseError as error:
            self._log.error_output(error.error_output)
            self._clock.sleep(RETRY_SECONDS)
            return
        for exp, run in runs:
            if self._covered(exp, run, RUN_SUMMARIES):
                self._refresh(exp, run)
        self._clock.sleep(INTERVAL_SECONDS)

    def _covered(self, exp: str, run: str, summaries: tuple[str, ...]) -> bool:
        """Whether all the summaries reach the run's newest ingested step (less 50), as the dashboard routes require."""
        statement = _COVERED_STATEMENT.format(
            count=len(summaries), exp=exp, run=run, summaries=", ".join(f"'{summary}'" for summary in summaries)
        )
        try:
            return self._client.value(statement) == "1"
        except ClickHouseError as error:
            self._log.error_output(error.error_output)
            return False

    def _refresh(self, exp: str, run: str) -> None:
        try:
            self._client.execute(_RUN_INSERT_STATEMENT.format(exp=exp, run=run))
        except ClickHouseError as error:
            self._log.report(error)
            self._log.error_line(f"{exp} {run} failed")
            return
        if self._covered(exp, run, MICROBATCH_SUMMARIES):
            for microbatch in self._client.value(_MICROBATCHES_STATEMENT.format(exp=exp, run=run)).split():
                self._refresh_microbatch(exp, run, microbatch)
        self._log.line(f"{exp} {run}")

    def _refresh_microbatch(self, exp: str, run: str, microbatch: str) -> None:
        try:
            self._client.execute(_MICROBATCH_INSERT_STATEMENT.format(exp=exp, run=run, microbatch=microbatch))
        except ClickHouseError as error:
            self._log.report(error)
            self._log.error_line(f"{exp} {run} microbatch {microbatch} failed")
