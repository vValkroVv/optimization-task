from __future__ import annotations

from metric_sampler.cloud.summaries.clickhouse import ClickHouse, ClickHouseError
from metric_sampler.cloud.summaries.clock import Clock
from metric_sampler.cloud.summaries.log import Log
from metric_sampler.cloud.summaries.loop import Loop

INTERVAL_SECONDS = 120

# The columns each split groups by and the suffix of its latest-step table.
SPLITS = (("rank", "rank"), ("microbatch", "mb"), ("rank, microbatch", "rank_mb"))

_UNSUMMARIZED_STEPS_STATEMENT = (
    "\n"
    "    SELECT exp, run, complete_through FROM metric_sampler.complete_steps_latest\n"
    "    WHERE complete_through > 0 AND last_ingested > now() - INTERVAL 1 DAY\n"
    "      AND (exp, run, complete_through) NOT IN (SELECT exp, run, step FROM "
    "metric_sampler.summary_activation_hist_latest)"
)
_INSERT_STATEMENT = (
    "INSERT INTO metric_sampler.summary_activation_hist_latest\n"
    "        SELECT now64(3), exp, run, step, layer, point, histogram, any(bins_start), any(bins_width), "
    "any(bins_count),\n"
    "            sumForEach(exact_hist), sumForEach(legacy_hist), min(counts_exact)\n"
    "        FROM metric_sampler.activation_histograms_step(exp_filter = '{exp}', run_filter = '{run}', "
    "rank_filter = -2, microbatch_filter = -2, step_filter = {step})\n"
    "        GROUP BY exp, run, step, layer, point, histogram\n"
    "        SETTINGS max_threads = 8, priority = 5, max_execution_time = 1800, "
    "optimize_move_to_prewhere_if_final = 1, log_comment = 'summary-hist-latest'"
)
_SPLIT_INSERT_STATEMENT = (
    "INSERT INTO metric_sampler.summary_activation_hist_latest_{table}\n"
    "            SELECT now64(3), exp, run, step, {split}, layer, point, histogram, any(bins_start), "
    "any(bins_width), any(bins_count),\n"
    "                sumForEach(exact_hist), sumForEach(legacy_hist), min(counts_exact)\n"
    "            FROM metric_sampler.activation_histograms_step(exp_filter = '{exp}', run_filter = '{run}', "
    "rank_filter = -2, microbatch_filter = -2, step_filter = {step})\n"
    "            GROUP BY exp, run, step, {split}, layer, point, histogram\n"
    "            SETTINGS max_threads = 8, priority = 5, max_execution_time = 1800, "
    "optimize_move_to_prewhere_if_final = 1, log_comment = 'summary-hist-latest'"
)


class HistLatest(Loop):
    """Sums the activation histograms of each recent run's newest complete step that is not summarized yet.

    The sums over every rank, and per rank, per microbatch and per rank and microbatch, go into the latest-step
    tables that the rank and microbatch selections read.
    """

    NAME = "hist_latest"

    def __init__(self, client: ClickHouse, clock: Clock, log: Log) -> None:
        super().__init__(self.NAME, clock, log)
        self._client = client

    def cycle(self) -> None:
        for exp, run, step in self._client.rows(_UNSUMMARIZED_STEPS_STATEMENT):
            self._summarize(exp, run, step)
        self._clock.sleep(INTERVAL_SECONDS)

    def _summarize(self, exp: str, run: str, step: str) -> None:
        try:
            self._client.execute(_INSERT_STATEMENT.format(exp=exp, run=run, step=step))
        except ClickHouseError as error:
            self._log.report(error)
            self._log.error_line(f"{exp} {run} step {step} failed")
            return
        self._log.line(f"{exp} {run} step {step}")
        for split, table in SPLITS:
            try:
                statement = _SPLIT_INSERT_STATEMENT.format(table=table, split=split, exp=exp, run=run, step=step)
                self._client.execute(statement)
            except ClickHouseError as error:
                self._log.report(error)
                self._log.error_line(f"{exp} {run} step {step} per-{split} failed")
