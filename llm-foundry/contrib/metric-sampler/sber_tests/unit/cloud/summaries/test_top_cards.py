from __future__ import annotations

import pytest

from metric_sampler.cloud.summaries.log import FileLog
from metric_sampler.cloud.summaries.top_cards import MICROBATCH_SUMMARIES, RUN_SUMMARIES, TopCards
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

DUE_RUNS = r"WITH changed AS"
COVERED_RUN = r"^SELECT count\(\) = 3 "
COVERED_MICROBATCH = r"^SELECT count\(\) = 1 "
MICROBATCHES = r"^SELECT DISTINCT microbatch"
RUN_INSERT = r"summary_top_cards\n\s+SELECT now64\(3\), exp, run, -2, 'grad'"
MICROBATCH_INSERT = r"activation_per_rank_mbpeak"
SETTINGS = "SETTINGS max_threads = 8, priority = 10, max_execution_time = 1800, log_comment = 'summary-top-cards'"


class Loop:
    def __init__(self, tmp_path, rules, stop_after=None) -> None:
        self.clock = FakeClock(stop_after)
        self.client = FakeClickHouse(rules)
        self.log_path = tmp_path / "top_cards.log"
        self.loop = TopCards(self.client, self.clock, FileLog(self.log_path, self.clock))

    def log_lines(self) -> list[str]:
        return self.log_path.read_text(encoding="utf-8").splitlines() if self.log_path.exists() else []

    def inserts(self) -> list[str]:
        return [compact(statement) for statement in self.client.matching(r"^INSERT INTO")]


def covered_run(microbatches: str = "0\n1\n") -> dict:
    return {DUE_RUNS: ["e\tr\n"], COVERED_RUN: ["1\n"], COVERED_MICROBATCH: ["1\n"], MICROBATCHES: [microbatches]}


def test_top_cards_asks_for_the_runs_whose_summaries_changed_since_their_cards_or_that_took_rows_for_an_hour(
    tmp_path,
) -> None:
    loop = Loop(tmp_path, {})

    loop.loop.cycle()

    assert loop.client.statements == [
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
    ]
    assert loop.clock.sleeps == [600]


def test_top_cards_inserts_the_cards_of_the_run_and_of_each_microbatch_from_the_peak_views(tmp_path) -> None:
    loop = Loop(tmp_path, covered_run())

    loop.loop.cycle()

    run_cards, *microbatch_cards = loop.inserts()
    assert run_cards == (
        "INSERT INTO metric_sampler.summary_top_cards "
        "SELECT now64(3), exp, run, -2, 'grad', node, rank, max(abs(robust_z)) "
        "FROM metric_sampler.grad_per_rank_fast(exp_filter = 'e', run_filter = 'r') "
        "WHERE exp = 'e' AND run = 'r' GROUP BY exp, run, node, rank UNION ALL "
        "SELECT now64(3), exp, run, -2, 'timing', node, rank, max(abs(robust_z)) "
        "FROM metric_sampler.timing_per_rank_fast(exp_filter = 'e', run_filter = 'r') "
        "WHERE exp = 'e' AND run = 'r' GROUP BY exp, run, node, rank UNION ALL "
        "SELECT now64(3), exp, run, -2, 'activation', node, rank, max_abs_z "
        f"FROM metric_sampler.activation_per_rank_peak(exp_filter = 'e', run_filter = 'r') {SETTINGS}"
    )
    assert microbatch_cards == [
        f"INSERT INTO metric_sampler.summary_top_cards SELECT now64(3), exp, run, {microbatch}, 'activation', node, "
        "rank, max_abs_z FROM metric_sampler.activation_per_rank_mbpeak(exp_filter = 'e', run_filter = 'r', "
        f"microbatch_filter = {microbatch}) {SETTINGS}"
        for microbatch in (0, 1)
    ]
    assert loop.log_lines() == [f"{START_TIMESTAMP} e r"]
    assert loop.clock.sleeps == [600]


def test_top_cards_checks_that_the_summaries_reach_the_runs_newest_step_less_fifty_as_the_dashboards_do(
    tmp_path,
) -> None:
    loop = Loop(tmp_path, covered_run())

    loop.loop.cycle()

    assert loop.client.matching(r"^SELECT count\(\)") == [
        "SELECT count() = 3 AND min(done) + 50 >= (SELECT max(step) FROM metric_sampler.step_index "
        "WHERE exp = 'e' AND run = 'r')\n"
        "          FROM (SELECT summary, max(complete_through) AS done FROM metric_sampler.summary_progress\n"
        "                WHERE exp = 'e' AND run = 'r' AND summary IN ('grad_per_rank', 'timing_per_rank', "
        "'activation_amax') GROUP BY summary)",
        "SELECT count() = 1 AND min(done) + 50 >= (SELECT max(step) FROM metric_sampler.step_index "
        "WHERE exp = 'e' AND run = 'r')\n"
        "          FROM (SELECT summary, max(complete_through) AS done FROM metric_sampler.summary_progress\n"
        "                WHERE exp = 'e' AND run = 'r' AND summary IN ('activation_amax_mb') GROUP BY summary)",
    ]
    assert RUN_SUMMARIES == ("grad_per_rank", "timing_per_rank", "activation_amax")
    assert MICROBATCH_SUMMARIES == ("activation_amax_mb",)


def test_top_cards_lists_the_microbatches_of_the_amax_summary_in_order(tmp_path) -> None:
    loop = Loop(tmp_path, covered_run("0\n1\n2\n"))

    loop.loop.cycle()

    assert loop.client.matching(MICROBATCHES) == [
        "SELECT DISTINCT microbatch FROM metric_sampler.summary_activation_amax_mb "
        "WHERE exp = 'e' AND run = 'r' ORDER BY microbatch"
    ]
    assert len(loop.inserts()) == 4


@pytest.mark.parametrize("answer", ["0\n", "2\n", ""])
def test_top_cards_leaves_a_run_alone_until_every_summary_covers_it(tmp_path, answer) -> None:
    loop = Loop(tmp_path, {DUE_RUNS: ["e\tr\n"], COVERED_RUN: [answer]})

    loop.loop.cycle()

    assert loop.inserts() == []
    assert loop.log_lines() == []


def test_top_cards_leaves_a_run_alone_when_its_coverage_cannot_be_asked_and_logs_what_curl_said(tmp_path) -> None:
    loop = Loop(tmp_path, {DUE_RUNS: ["e\tr\n"], COVERED_RUN: [failure(MEMORY_ERROR)]})

    loop.loop.cycle()

    assert loop.inserts() == []
    assert loop.log_lines() == [CURL_ERROR.strip()]


def test_top_cards_goes_on_with_the_next_run_after_one_that_is_not_covered(tmp_path) -> None:
    rules = {**covered_run("5\n"), DUE_RUNS: ["e\tr\ne\tq\n"], COVERED_RUN: ["0\n", "1\n"]}
    loop = Loop(tmp_path, rules)

    loop.loop.cycle()

    assert loop.log_lines() == [f"{START_TIMESTAMP} e q"]
    assert len(loop.inserts()) == 2


@pytest.mark.parametrize("answer", ["0\n", failure(MEMORY_ERROR)])
def test_top_cards_inserts_no_microbatch_cards_until_the_microbatch_summary_covers_the_run(tmp_path, answer) -> None:
    loop = Loop(tmp_path, {**covered_run(), COVERED_MICROBATCH: [answer]})

    loop.loop.cycle()

    assert len(loop.inserts()) == 1
    assert not loop.client.matching(MICROBATCHES)
    assert loop.log_lines()[-1] == f"{START_TIMESTAMP} e r"


def test_top_cards_inserts_no_microbatch_cards_when_the_summary_has_no_microbatch(tmp_path) -> None:
    loop = Loop(tmp_path, covered_run(""))

    loop.loop.cycle()

    assert len(loop.inserts()) == 1
    assert loop.log_lines() == [f"{START_TIMESTAMP} e r"]


def test_top_cards_reports_a_failed_run_insert_and_skips_its_microbatches(tmp_path) -> None:
    loop = Loop(tmp_path, {**covered_run(), RUN_INSERT: [failure(MEMORY_ERROR)]})

    loop.loop.cycle()

    assert len(loop.inserts()) == 1
    assert loop.log_lines() == [MEMORY_ERROR.strip(), CURL_ERROR.strip(), f"{START_TIMESTAMP} e r failed"]


def test_top_cards_reports_a_failed_microbatch_and_goes_on_with_the_others(tmp_path) -> None:
    loop = Loop(tmp_path, {**covered_run(), MICROBATCH_INSERT: [failure(MEMORY_ERROR), ""]})

    loop.loop.cycle()

    assert len(loop.inserts()) == 3
    assert loop.log_lines() == [
        MEMORY_ERROR.strip(), CURL_ERROR.strip(),
        f"{START_TIMESTAMP} e r microbatch 0 failed",
        f"{START_TIMESTAMP} e r",
    ]


def test_top_cards_retries_after_a_minute_when_the_due_runs_cannot_be_listed(tmp_path) -> None:
    loop = Loop(tmp_path, {**covered_run(), DUE_RUNS: [failure(MEMORY_ERROR), "e\tr\n"]})

    loop.loop.cycle()

    assert loop.inserts() == []
    assert loop.log_lines() == [CURL_ERROR.strip()]
    assert loop.clock.sleeps == [60]


def test_top_cards_runs_cycle_after_cycle_every_ten_minutes_until_the_service_stops(tmp_path) -> None:
    loop = Loop(tmp_path, {DUE_RUNS: [""]}, stop_after=2)

    loop.loop.run()

    assert loop.clock.sleeps == [600, 600]
    assert len(loop.client.matching(DUE_RUNS)) == 2
