from __future__ import annotations

import re

import pytest

from metric_sampler.cloud.summaries.cold_mover import ColdMover
from metric_sampler.cloud.summaries.cold_tier import RAW_TABLES, SUMMARY_TABLES, ColdTier
from metric_sampler.cloud.summaries.log import FileLog
from metric_sampler.cloud.summaries.pressure import IoPressure
from sber_tests.unit.cloud.summaries.fakes import START_TIMESTAMP, FakeClickHouse, FakeClock

pytestmark = pytest.mark.unit

RUNS = r"^SELECT keys\[1\]"
LOCAL_PARTS = r"^SELECT name FROM system.parts .*disk_name = 'default'"
RAW = ["scalars", "expert_counts", "router_vectors"]
SUMMARIES = [
    "summary_activation_rank_mb", "summary_router_rank", "summary_activation_amax_mb",
    "summary_router_balance_rank_mb", "summary_activation_exp_quantiles_rank_mb", "summary_activation_massive_rank_mb",
    "summary_activation_rank", "summary_activation_rank_by_layer",
]
ONE_GIB = 1073741824


class Tier:
    def __init__(self, tmp_path, rows: str, stop_after=None) -> None:
        pressure_path = tmp_path / "pressure"
        pressure_path.write_text("full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n", encoding="utf-8")
        self.clock = FakeClock(stop_after)
        self.client = FakeClickHouse({RUNS: [rows], LOCAL_PARTS: ["p1\n"]})
        self.log_path = tmp_path / "cold.log"
        log = FileLog(self.log_path, self.clock)
        mover = ColdMover(self.client, self.clock, log, IoPressure(pressure_path))
        self.loop = ColdTier(self.client, self.clock, log, mover)

    def moves(self) -> list[tuple[str, str, int, int]]:
        """Per local-disk listing: the table, the run's partition, the minimum age and the minimum size."""
        return [
            (
                re.search(r"table = '(\w+)'", listing)[1],
                re.search(r"partition = '\(''(\w+)'',''(\w+)''\)'", listing)[2],
                int(re.search(r"toIntervalHour\((\d+)\)", listing)[1]),
                int(re.search(r"bytes_on_disk >= (\d+)", listing)[1]),
            )
            for listing in self.client.matching(LOCAL_PARTS)
        ]


def test_the_raw_tables_and_the_summaries_partitioned_by_run_are_the_ones_the_dashboards_read() -> None:
    assert tuple(RAW) == RAW_TABLES
    assert tuple(SUMMARIES) == SUMMARY_TABLES


def test_cold_tier_asks_for_the_runs_whose_raw_parts_have_stayed_a_day_on_the_data_disk(tmp_path) -> None:
    tier = Tier(tmp_path, "")

    tier.loop.cycle()

    assert tier.client.statements == [
        "SELECT keys[1], keys[2], max(modification_time) <= now() - toIntervalHour(24),\n"
        "      max(modification_time) <= now() - toIntervalHour(168)\n"
        "    FROM (SELECT extractAll(partition, '''([^'']*)''') AS keys, modification_time, bytes_on_disk "
        "FROM system.parts\n"
        "          WHERE database = 'metric_sampler' AND active AND has(splitByWhitespace('scalars expert_counts "
        "router_vectors summary_activation_rank_mb summary_router_rank summary_activation_amax_mb "
        "summary_router_balance_rank_mb\n"
        "  summary_activation_exp_quantiles_rank_mb summary_activation_massive_rank_mb summary_activation_rank\n"
        "  summary_activation_rank_by_layer'), table)\n"
        "            AND disk_name = 'default')\n"
        "    GROUP BY keys\n"
        "    HAVING max(modification_time) <= now() - toIntervalHour(24)\n"
        f"      OR countIf(modification_time <= now() - toIntervalHour(24) AND bytes_on_disk >= {ONE_GIB}) > 0\n"
        "    FORMAT TSV"
    ]
    assert tier.clock.sleeps == [3600]


def test_cold_tier_moves_raw_parts_by_how_long_a_run_is_idle_and_its_summaries_after_a_longer_idle_time(
    tmp_path,
) -> None:
    tier = Tier(tmp_path, "e\tweek\t1\t1\ne\tday\t1\t0\ne\twritten\t0\t0\n")

    tier.loop.cycle()

    assert tier.moves() == [
        *((table, "week", 24, 0) for table in [*RAW, *SUMMARIES]),
        *((table, "day", 24, 0) for table in RAW),
        *((table, "written", 24, ONE_GIB) for table in RAW),
    ]


def test_cold_tier_logs_the_moves_of_every_run_to_its_own_log(tmp_path) -> None:
    tier = Tier(tmp_path, "e\tday\t1\t0\n")

    tier.loop.cycle()

    assert [line for line in tier.log_path.read_text(encoding="utf-8").splitlines() if "before:" in line] == [
        f"{START_TIMESTAMP} {table} before:" for table in RAW
    ]


def test_cold_tier_moves_nothing_while_no_run_has_idle_parts(tmp_path) -> None:
    tier = Tier(tmp_path, "")

    tier.loop.cycle()

    assert tier.moves() == []


def test_cold_tier_looks_again_every_hour_until_the_service_stops(tmp_path) -> None:
    tier = Tier(tmp_path, "e\tday\t1\t0\n", stop_after=2)

    tier.loop.run()

    assert tier.clock.sleeps == [3600, 3600]
    assert len(tier.client.matching(RUNS)) == 2
