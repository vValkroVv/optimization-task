from __future__ import annotations

import pytest

from metric_sampler.cloud.summaries.hist_latest import HistLatest
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

UNSUMMARIZED_STEPS = r"FROM metric_sampler\.complete_steps_latest"
LATEST = r"^INSERT INTO metric_sampler\.summary_activation_hist_latest\n"
RANK = r"^INSERT INTO metric_sampler\.summary_activation_hist_latest_rank\n"
MICROBATCH = r"^INSERT INTO metric_sampler\.summary_activation_hist_latest_mb\n"
RANK_MICROBATCH = r"^INSERT INTO metric_sampler\.summary_activation_hist_latest_rank_mb\n"
INSERTS = r"^INSERT INTO metric_sampler\.summary_activation_hist_latest"
SETTINGS = (
    "SETTINGS max_threads = 8, priority = 5, max_execution_time = 1800, optimize_move_to_prewhere_if_final = 1, "
    "log_comment = 'summary-hist-latest'"
)


class Loop:
    def __init__(self, tmp_path, rules, stop_after=None) -> None:
        self.clock = FakeClock(stop_after)
        self.client = FakeClickHouse(rules)
        self.log_path = tmp_path / "hist_latest.log"
        self.loop = HistLatest(self.client, self.clock, FileLog(self.log_path, self.clock))

    def log_lines(self) -> list[str]:
        return self.log_path.read_text(encoding="utf-8").splitlines() if self.log_path.exists() else []


def test_hist_latest_asks_for_the_newest_complete_step_of_each_run_ingested_in_the_last_day_and_not_summarized(
    tmp_path,
) -> None:
    loop = Loop(tmp_path, {})

    loop.loop.cycle()

    assert loop.client.statements == [
        "\n"
        "    SELECT exp, run, complete_through FROM metric_sampler.complete_steps_latest\n"
        "    WHERE complete_through > 0 AND last_ingested > now() - INTERVAL 1 DAY\n"
        "      AND (exp, run, complete_through) NOT IN (SELECT exp, run, step FROM "
        "metric_sampler.summary_activation_hist_latest)"
    ]
    assert loop.clock.sleeps == [120]


def test_hist_latest_sums_a_step_over_every_rank_and_then_per_rank_per_microbatch_and_per_both(tmp_path) -> None:
    loop = Loop(tmp_path, {UNSUMMARIZED_STEPS: ["e\ta\t120\n"]})

    loop.loop.cycle()

    source = (
        "FROM metric_sampler.activation_histograms_step(exp_filter = 'e', run_filter = 'a', rank_filter = -2, "
        "microbatch_filter = -2, step_filter = 120)"
    )
    assert [compact(statement) for statement in loop.client.matching(INSERTS)] == [
        "INSERT INTO metric_sampler.summary_activation_hist_latest SELECT now64(3), exp, run, step, layer, point, "
        "histogram, any(bins_start), any(bins_width), any(bins_count), sumForEach(exact_hist), "
        f"sumForEach(legacy_hist), min(counts_exact) {source} GROUP BY exp, run, step, layer, point, histogram "
        f"{SETTINGS}",
        *(
            f"INSERT INTO metric_sampler.summary_activation_hist_latest_{table} SELECT now64(3), exp, run, step, "
            f"{split}, layer, point, histogram, any(bins_start), any(bins_width), any(bins_count), "
            f"sumForEach(exact_hist), sumForEach(legacy_hist), min(counts_exact) {source} "
            f"GROUP BY exp, run, step, {split}, layer, point, histogram {SETTINGS}"
            for split, table in (("rank", "rank"), ("microbatch", "mb"), ("rank, microbatch", "rank_mb"))
        ),
    ]
    assert loop.log_lines() == [f"{START_TIMESTAMP} e a step 120"]


def test_hist_latest_keeps_the_statement_layout_of_each_insert(tmp_path) -> None:
    loop = Loop(tmp_path, {UNSUMMARIZED_STEPS: ["e\ta\t120\n"]})

    loop.loop.cycle()

    assert loop.client.matching(LATEST)[0].splitlines()[:3] == [
        "INSERT INTO metric_sampler.summary_activation_hist_latest",
        "        SELECT now64(3), exp, run, step, layer, point, histogram, any(bins_start), any(bins_width), "
        "any(bins_count),",
        "            sumForEach(exact_hist), sumForEach(legacy_hist), min(counts_exact)",
    ]
    assert loop.client.matching(RANK_MICROBATCH)[0].splitlines()[1] == (
        "            SELECT now64(3), exp, run, step, rank, microbatch, layer, point, histogram, any(bins_start), "
        "any(bins_width), any(bins_count),"
    )


def test_hist_latest_summarizes_every_due_run_in_turn_and_waits_two_minutes_before_asking_again(tmp_path) -> None:
    loop = Loop(tmp_path, {UNSUMMARIZED_STEPS: ["e\ta\t120\ne\tb\t45\n", ""]}, stop_after=2)

    loop.loop.run()

    assert [statement.split("\n")[0] for statement in loop.client.statements if "INSERT" in statement] == [
        "INSERT INTO metric_sampler.summary_activation_hist_latest",
        "INSERT INTO metric_sampler.summary_activation_hist_latest_rank",
        "INSERT INTO metric_sampler.summary_activation_hist_latest_mb",
        "INSERT INTO metric_sampler.summary_activation_hist_latest_rank_mb",
    ] * 2
    assert loop.log_lines() == [f"{START_TIMESTAMP} e a step 120", f"{START_TIMESTAMP} e b step 45"]
    assert loop.clock.sleeps == [120, 120]


def test_hist_latest_skips_the_splits_of_a_step_whose_total_failed_and_goes_on_with_the_next_run(tmp_path) -> None:
    loop = Loop(tmp_path, {UNSUMMARIZED_STEPS: ["e\ta\t120\ne\tb\t45\n"], LATEST: [failure(MEMORY_ERROR), ""]})

    loop.loop.cycle()

    assert loop.log_lines() == [
        MEMORY_ERROR.strip(), CURL_ERROR.strip(),
        f"{START_TIMESTAMP} e a step 120 failed",
        f"{START_TIMESTAMP} e b step 45",
    ]
    assert len(loop.client.matching(INSERTS)) == 5


@pytest.mark.parametrize(
    ("pattern", "split"),
    [(RANK, "rank"), (MICROBATCH, "microbatch"), (RANK_MICROBATCH, "rank, microbatch")],
)
def test_hist_latest_logs_the_split_that_failed_and_still_does_the_others(tmp_path, pattern, split) -> None:
    loop = Loop(tmp_path, {UNSUMMARIZED_STEPS: ["e\ta\t120\n"], pattern: [failure(MEMORY_ERROR)]})

    loop.loop.cycle()

    assert loop.log_lines() == [
        f"{START_TIMESTAMP} e a step 120",
        MEMORY_ERROR.strip(), CURL_ERROR.strip(),
        f"{START_TIMESTAMP} e a step 120 per-{split} failed",
    ]
    assert len(loop.client.matching(INSERTS)) == 4
