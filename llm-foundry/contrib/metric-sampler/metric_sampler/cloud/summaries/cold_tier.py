from __future__ import annotations

from metric_sampler.cloud.summaries.clickhouse import ClickHouse
from metric_sampler.cloud.summaries.clock import Clock
from metric_sampler.cloud.summaries.cold_mover import ColdMover
from metric_sampler.cloud.summaries.log import Log
from metric_sampler.cloud.summaries.loop import Loop

HOT_HOURS = 24
SUMMARY_HOT_HOURS = 168
MIN_BYTES = 1073741824
INTERVAL_SECONDS = 3600
RAW_TABLES = ("scalars", "expert_counts", "router_vectors")
# The summaries partitioned by run, which the dashboards read. The line breaks are part of the statement that lists
# the tables.
_SUMMARIES = """summary_activation_rank_mb summary_router_rank summary_activation_amax_mb summary_router_balance_rank_mb
  summary_activation_exp_quantiles_rank_mb summary_activation_massive_rank_mb summary_activation_rank
  summary_activation_rank_by_layer"""
SUMMARY_TABLES = tuple(_SUMMARIES.split())

_RUNS_STATEMENT = (
    "SELECT keys[1], keys[2], max(modification_time) <= now() - toIntervalHour({hot_hours}),\n"
    "      max(modification_time) <= now() - toIntervalHour({summary_hot_hours})\n"
    "    FROM (SELECT extractAll(partition, '''([^'']*)''') AS keys, modification_time, bytes_on_disk "
    "FROM system.parts\n"
    "          WHERE database = 'metric_sampler' AND active AND has(splitByWhitespace('{tables}'), table)\n"
    "            AND disk_name = 'default')\n"
    "    GROUP BY keys\n"
    "    HAVING max(modification_time) <= now() - toIntervalHour({hot_hours})\n"
    "      OR countIf(modification_time <= now() - toIntervalHour({hot_hours}) AND bytes_on_disk >= {min_bytes}) > 0\n"
    "    FORMAT TSV"
)


class ColdTier(Loop):
    """Moves the raw parts that have stayed HOT_HOURS on the data disk to the cold volume.

    A run that is still being written keeps its parts under MIN_BYTES local, as parts on the cold volume are not
    merged again; a run idle for HOT_HOURS moves them all. The summaries partitioned by run move after
    SUMMARY_HOT_HOURS of idleness, because a summary read from the cold volume is slow until it is cached.
    """

    NAME = "cold"

    def __init__(self, client: ClickHouse, clock: Clock, log: Log, mover: ColdMover) -> None:
        super().__init__(self.NAME, clock, log)
        self._client = client
        self._mover = mover

    def cycle(self) -> None:
        statement = _RUNS_STATEMENT.format(
            hot_hours=HOT_HOURS,
            summary_hot_hours=SUMMARY_HOT_HOURS,
            tables=" ".join(RAW_TABLES) + " " + _SUMMARIES,
            min_bytes=MIN_BYTES,
        )
        for exp, run, idle, summaries_idle in self._client.rows(statement):
            if summaries_idle == "1":
                self._mover.move_run(exp, run, (*RAW_TABLES, *SUMMARY_TABLES), HOT_HOURS, 0)
            elif idle == "1":
                self._mover.move_run(exp, run, RAW_TABLES, HOT_HOURS, 0)
            else:
                self._mover.move_run(exp, run, RAW_TABLES, HOT_HOURS, MIN_BYTES)
        self._clock.sleep(INTERVAL_SECONDS)
