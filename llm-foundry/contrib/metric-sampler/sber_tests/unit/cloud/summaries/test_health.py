from __future__ import annotations

import pytest

from metric_sampler.cloud.summaries import health
from metric_sampler.cloud.summaries.health import Health
from metric_sampler.cloud.summaries.log import FileLog
from sber_tests.unit.cloud.summaries.fakes import (
    CURL_ERROR,
    MEMORY_ERROR,
    START_TIMESTAMP,
    FakeClickHouse,
    FakeClock,
    compact,
    failure,
)

pytestmark = pytest.mark.unit

PENDING = r"health_pending_windows"
GENERATION = r"^SELECT generateUUIDv4\(\)"
SNAPSHOT = r"^INSERT INTO metric_sampler\.health_snapshots"
REFRESH = r"^INSERT INTO metric_sampler\.health_refreshes"
UUID = "00000000-0000-4000-8000-000000000001\n"
WINDOW = "e\tr\ta\tb\t10\t19\n"
GENERATION_LITERAL = "toUUID('00000000-0000-4000-8000-000000000001')"


class Loop:
    def __init__(self, tmp_path, rules, stop_after=None) -> None:
        self.clock = FakeClock(stop_after)
        self.client = FakeClickHouse({GENERATION: [UUID], **rules})
        self.log_path = tmp_path / "health.log"
        self.loop = Health(self.client, self.clock, FileLog(self.log_path, self.clock))

    def log_lines(self) -> list[str]:
        return self.log_path.read_text(encoding="utf-8").splitlines() if self.log_path.exists() else []

    def inserts(self) -> list[str]:
        return [compact(statement) for statement in self.client.matching(r"^INSERT INTO")]


def test_health_publishes_a_window_only_after_its_complete_snapshot(tmp_path) -> None:
    loop = Loop(tmp_path, {PENDING: [WINDOW]})

    loop.loop.cycle()

    snapshot, refresh = loop.inserts()
    assert snapshot.startswith("INSERT INTO metric_sampler.health_snapshots")
    assert f"SELECT exp, run, attempt, 'b', 10, {GENERATION_LITERAL}, step" in snapshot
    assert (
        "FROM metric_sampler.health_per_step_window(exp_filter = 'e', run_filter = 'r', attempt_filter = 'a', "
        "step_from = 10, step_to = 19)"
    ) in snapshot
    assert refresh.startswith("INSERT INTO metric_sampler.health_refreshes")
    assert (
        f"SELECT 'e', 'r', 'a', 'b', 10, 19, {GENERATION_LITERAL}, "
        "(SELECT count() FROM metric_sampler.health_snapshots "
        f"WHERE exp = 'e' AND run = 'r' AND attempt = 'a' AND batch = 'b' AND window_first = 10 "
        f"AND generation = {GENERATION_LITERAL})"
    ) in refresh
    assert refresh.endswith(
        "greatest(now64(6), (SELECT max(completed_at) + toIntervalMicrosecond(1) "
        "FROM metric_sampler.health_refreshes))"
    )
    assert loop.log_lines() == []
    assert loop.clock.sleeps == []


def test_health_asks_for_the_newest_pending_window_first(tmp_path) -> None:
    loop = Loop(tmp_path, {PENDING: [""]})

    loop.loop.cycle()

    assert loop.client.statements == [
        "SELECT exp, run, attempt, batch, window_first, window_last FROM metric_sampler.health_pending_windows\n"
        "    ORDER BY collected_at DESC, exp, run, attempt, batch, window_first DESC LIMIT 1 FORMAT TSV"
    ]


def test_health_waits_half_a_minute_when_no_window_is_pending(tmp_path) -> None:
    loop = Loop(tmp_path, {PENDING: [""]})

    loop.loop.cycle()

    assert loop.clock.sleeps == [30]
    assert loop.inserts() == []


def test_health_snapshots_use_the_limits_that_keep_the_refresh_off_the_ingest_and_drop_rows_around_the_window(
    tmp_path,
) -> None:
    loop = Loop(tmp_path, {PENDING: [WINDOW]})

    loop.loop.cycle()

    snapshot = loop.inserts()[0]
    assert snapshot.endswith(
        "SETTINGS optimize_aggregation_in_order = 0, optimize_move_to_prewhere_if_final = 1, max_threads = 4, "
        "max_memory_usage = 34359738368, max_execution_time = 900, log_comment = 'health-refresh'"
    )
    assert "ORDER BY collected_at DESC, exp, run, attempt, batch, window_first DESC" in health.PENDING_STATEMENT
    assert "collected_at" not in health.REFRESH_STATEMENT
    assert "DELETE" not in health.SNAPSHOT_STATEMENT + health.REFRESH_STATEMENT


def test_health_goes_straight_on_to_the_next_window_after_a_published_one(tmp_path) -> None:
    loop = Loop(tmp_path, {PENDING: [WINDOW, "e\tr\ta\tc\t20\t29\n", ""]}, stop_after=1)

    loop.loop.run()

    assert [statement.split()[2] for statement in loop.inserts()] == [
        "metric_sampler.health_snapshots", "metric_sampler.health_refreshes"
    ] * 2
    assert loop.clock.sleeps == [30]


@pytest.mark.parametrize(
    ("pattern", "what"), [(SNAPSHOT, "snapshot"), (REFRESH, "refresh")]
)
def test_health_logs_a_failed_insert_and_retries_after_a_minute(tmp_path, pattern, what) -> None:
    loop = Loop(tmp_path, {PENDING: [WINDOW], pattern: [failure(MEMORY_ERROR)]})

    loop.loop.cycle()

    assert loop.log_lines() == [
        MEMORY_ERROR.strip(), CURL_ERROR.strip(), f"{START_TIMESTAMP} e r a b 10-19 {what} failed"
    ]
    assert loop.clock.sleeps == [60]


def test_health_does_not_publish_a_window_whose_snapshot_failed(tmp_path) -> None:
    loop = Loop(tmp_path, {PENDING: [WINDOW], SNAPSHOT: [failure()]})

    loop.loop.cycle()

    assert not loop.client.matching(REFRESH)


@pytest.mark.parametrize("failing", [PENDING, GENERATION])
def test_health_retries_after_a_minute_when_it_cannot_ask_for_a_window_or_a_generation(tmp_path, failing) -> None:
    loop = Loop(tmp_path, {PENDING: [WINDOW], failing: [failure(MEMORY_ERROR)]})

    loop.loop.cycle()

    assert loop.log_lines() == [CURL_ERROR.strip()]
    assert loop.clock.sleeps == [60]
    assert loop.inserts() == []
