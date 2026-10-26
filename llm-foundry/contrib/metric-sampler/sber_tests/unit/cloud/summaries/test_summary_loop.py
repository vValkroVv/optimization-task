from __future__ import annotations

import re
from pathlib import Path

import pytest

from metric_sampler.cloud.summaries.catalog import Summary
from metric_sampler.cloud.summaries.log import FileLog
from metric_sampler.cloud.summaries.slots import InsertSlots
from metric_sampler.cloud.summaries.summary_loop import SummaryLoop
from sber_tests.unit.cloud.summaries.fakes import (
    CURL_ERROR,
    MEMORY_ERROR,
    START_TIMESTAMP,
    FakeClickHouse,
    FakeClock,
    compact,
    computed_files,
    failure,
)

pytestmark = pytest.mark.unit

PENDING_RUNS = r"WITH frontier AS"
STEP_LOOKUP = r"^SELECT step FROM"
ROWS_LOOKUP = r"^SELECT max\(max_ingested_at\)"
CHUNK_INSERT = r"^INSERT INTO metric_sampler\.summary_test "
PROGRESS_INSERT = r"^INSERT INTO metric_sampler\.summary_progress"
SOURCE = "SELECT * FROM v(step_from = __FROM__) WHERE __WHERE__"
BOUNDED_SOURCE = "SELECT * FROM v(step_to = __TO__) WHERE __WHERE__"
_COMPUTED = re.compile(r"exp = '(\w+)' AND run = '(\w+)' AND step BETWEEN (\d+) AND (\d+)\) SETTINGS")
_PROGRESS = re.compile(r"summary_progress VALUES \('\w+', '(\w+)', '(\w+)', (\d+),")


def run_row(run="a", max_step=120, finished=99, frontier=0, ingested="T1", ranks=16) -> str:
    return f"e\t{run}\t{max_step}\t{finished}\t{frontier}\t{ingested}\t{ranks}\n"


class Loop:
    """A summary loop on a fake ClickHouse, with its state files and log under `tmp_path`."""

    def __init__(
        self, tmp_path: Path, rules: dict, summary: Summary | None = None, stop_after: int | None = None
    ) -> None:
        self.clock = FakeClock(stop_after)
        self.client = FakeClickHouse(rules)
        self.state = tmp_path / "state"
        self.state.mkdir(exist_ok=True)
        self.log_path = tmp_path / "test.log"
        self.summary = summary or Summary("test", SOURCE)
        self.loop = SummaryLoop(
            self.summary,
            self.client,
            InsertSlots(self.client, self.clock, self.state),
            self.clock,
            FileLog(self.log_path, self.clock),
            self.state,
        )

    def cycles(self, count: int) -> list[str]:
        for _ in range(count):
            self.loop.cycle()
        return self.work()

    def work(self) -> list[str]:
        work = []
        for statement in self.client.statements:
            if progress := _PROGRESS.search(statement):
                work.append(f"progress {progress[1]}/{progress[2]} {progress[3]}")
            elif computed := _COMPUTED.search(statement):
                work.append(f"{computed[1]}/{computed[2]} {computed[3]}-{computed[4]}")
        return work

    def advance_on_insert(self, seconds: int) -> None:
        def advance(statement: str) -> None:
            if re.match(CHUNK_INSERT, statement):
                self.clock.now += seconds

        self.client.on_execute = advance

    def log_lines(self) -> list[str]:
        return self.log_path.read_text(encoding="utf-8").splitlines() if self.log_path.exists() else []


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        pytest.param(
            [run_row(ingested="T1")] * 3, ["e/a 100-120"],
            id="a chunk waiting for its nodes is not recomputed while no rows arrive",
        ),
        pytest.param(
            [run_row(ingested=stamp) for stamp in ("T1", "T1", "T2")], ["e/a 100-120", "e/a 100-120"],
            id="rows ingested since the last computation recompute the chunk",
        ),
        pytest.param(
            [run_row(frontier=0), run_row(frontier=120)], ["e/a 100-120", "progress e/a 120"],
            id="a computed chunk the frontier reaches is final without a recomputation",
        ),
        pytest.param(
            [run_row() + run_row(run="b", max_step=45, finished=39)] * 2
            + [run_row() + run_row(run="b", max_step=45, finished=39, ingested="T2")],
            ["e/a 100-120", "e/b 40-45", "e/b 40-45"],
            id="every run keeps its own computation",
        ),
        pytest.param(
            [run_row(frontier=0), run_row(frontier=109)],
            ["e/a 100-120", "e/a 100-109", "progress e/a 109", "e/a 110-120"],
            id="a chunk the frontier cuts is computed again",
        ),
    ],
)
def test_summary_loop_computes_a_chunk_again_only_when_rows_were_ingested(tmp_path, answers, expected) -> None:
    loop = Loop(tmp_path, {PENDING_RUNS: answers})

    assert loop.cycles(len(answers)) == expected


def test_summary_loop_keeps_the_last_computation_of_a_run_in_a_state_file_that_an_operator_can_delete(tmp_path) -> None:
    loop = Loop(tmp_path, {PENDING_RUNS: [run_row(ingested="2026-10-17 10:00:00.123")]})

    loop.cycles(1)

    assert computed_files(loop.state) == {"summaries-test.e.a.computed": "100-120 2026-10-17 10:00:00.123\n"}
    (loop.state / "summaries-test.e.a.computed").unlink()
    assert loop.cycles(1) == ["e/a 100-120", "e/a 100-120"]


def test_summary_loop_continues_from_the_computation_that_the_shell_loops_left_in_the_state_file(tmp_path) -> None:
    loop = Loop(tmp_path, {PENDING_RUNS: [run_row(ingested="T1", frontier=120)]})
    (loop.state / "summaries-test.e.a.computed").write_bytes(b"100-120 T1\n")

    assert loop.cycles(1) == ["progress e/a 120"]


def test_summary_loop_resumes_several_summaries_from_their_lowest_progress_and_moves_them_all(tmp_path) -> None:
    loop = Loop(
        tmp_path, {PENDING_RUNS: [run_row(frontier=120)]}, Summary("test", SOURCE, progress=("first", "second"))
    )

    assert loop.cycles(1) == ["e/a 100-120", "progress e/a 120"]
    statement = compact(loop.client.matching(PENDING_RUNS)[0])
    assert (
        "done AS (SELECT exp, run, min(done) AS done FROM (SELECT exp, run, summary, max(complete_through) AS done "
        "FROM metric_sampler.summary_progress WHERE summary IN ('first', 'second') GROUP BY exp, run, summary) "
        "GROUP BY exp, run HAVING count() = 2)"
    ) in statement
    assert loop.client.matching(PROGRESS_INSERT) == [
        "INSERT INTO metric_sampler.summary_progress VALUES ('first', 'e', 'a', 120, now64(3)), "
        "('second', 'e', 'a', 120, now64(3))"
    ]


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        pytest.param(["W1", "W1", "W1"], ["e/a 100-120"], id="rows of other steps leave the chunk as it is"),
        pytest.param(["W1", "W1", "W2"], ["e/a 100-120", "e/a 100-120"], id="rows of its steps recompute the chunk"),
    ],
)
def test_summary_loop_over_a_chunk_bounded_source_recomputes_a_chunk_only_for_rows_of_its_steps(
    tmp_path, rows, expected,
) -> None:
    answers = [run_row(ingested=stamp) for stamp in ("T1", "T2", "T3")]
    loop = Loop(tmp_path, {PENDING_RUNS: answers, ROWS_LOOKUP: rows}, Summary("test", BOUNDED_SOURCE))

    assert loop.cycles(3) == expected
    assert loop.client.matching(ROWS_LOOKUP)[0] == (
        "SELECT max(max_ingested_at) FROM metric_sampler.step_index "
        "WHERE exp = 'e' AND run = 'a' AND step BETWEEN 100 AND 120"
    )
    assert computed_files(loop.state) == {"summaries-test.e.a.computed": f"100-120 {rows[-1]}\n"}


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        pytest.param(
            [run_row(frontier=0)] * 2, [],
            id="nothing is computed while the summary it derives from has finished no step",
        ),
        pytest.param(
            [run_row(frontier=109)], ["e/a 100-109", "progress e/a 109"],
            id="the chunk ends at the steps the other summary finished and nothing beyond them is computed",
        ),
        pytest.param(
            [run_row(frontier=120)], ["e/a 100-120", "progress e/a 120"],
            id="a chunk the other summary finished is computed once and final",
        ),
    ],
)
def test_derived_summary_loop_computes_only_the_steps_its_source_summary_finished(tmp_path, answers, expected) -> None:
    loop = Loop(tmp_path, {PENDING_RUNS: answers}, Summary("test", SOURCE, after="source_summary"))

    assert loop.cycles(len(answers)) == expected


@pytest.mark.parametrize(("after", "derived"), [(None, False), ("source_summary", True)])
def test_summary_loop_limits_the_frontier_to_the_progress_of_the_summary_it_derives_from(
    tmp_path, after, derived,
) -> None:
    loop = Loop(tmp_path, {PENDING_RUNS: [run_row()]}, Summary("test", SOURCE, after=after))

    loop.cycles(1)

    statement = loop.client.matching(PENDING_RUNS)[0]
    assert ("least(c.complete_through, p.finished)" in statement) is derived
    assert ("summary = 'source_summary'" in statement) is derived
    if not derived:
        assert "WITH frontier AS (SELECT exp, run, complete_through, last_ingested FROM" in statement


def test_summary_loop_asks_for_the_runs_with_steps_beyond_the_progress_newest_first_and_every_run_by_default(
    tmp_path,
) -> None:
    loop = Loop(tmp_path, {PENDING_RUNS: [""]})

    loop.cycles(1)

    assert loop.client.statements[0] == (
        "\n"
        "    WITH frontier AS (SELECT exp, run, complete_through, last_ingested FROM "
        "metric_sampler.complete_steps_latest),\n"
        "    done AS (SELECT exp, run, min(done) AS done FROM "
        "(SELECT exp, run, summary, max(complete_through) AS done\n"
        "      FROM metric_sampler.summary_progress WHERE summary IN ('test') GROUP BY exp, run, summary)\n"
        "      GROUP BY exp, run HAVING count() = 1)\n"
        "    SELECT si.exp, si.run, si.max_step, done.done, frontier.complete_through, si.ingested, si.rank_count\n"
        "    FROM metric_sampler.run_steps_latest AS si\n"
        "    LEFT JOIN done ON done.exp = si.exp AND done.run = si.run\n"
        "    LEFT JOIN frontier ON frontier.exp = si.exp AND frontier.run = si.run\n"
        "    WHERE si.max_step > done.done AND (0 = 0 OR frontier.last_ingested >= now() - toIntervalDay(0))\n"
        "    ORDER BY frontier.last_ingested DESC, si.max_step DESC\n"
        "    SETTINGS join_use_nulls = 0"
    )


@pytest.mark.parametrize(
    ("ranks", "offset"),
    [
        pytest.param(0, 9, id="a run without manifests"),
        pytest.param(16, 9, id="16 ranks"),
        pytest.param(256, 9, id="256 ranks"),
        pytest.param(1024, 1, id="1024 ranks"),
        pytest.param(4096, 0, id="4096 ranks"),
    ],
)
def test_summary_loop_takes_proportionally_fewer_steps_per_chunk_of_a_run_with_more_ranks(
    tmp_path, ranks, offset,
) -> None:
    loop = Loop(tmp_path, {PENDING_RUNS: [run_row(ranks=ranks)]}, Summary("test", SOURCE, chunk_steps=10))

    assert loop.cycles(1) == ["e/a 100-120"]
    assert loop.client.matching(STEP_LOOKUP) == [
        "SELECT step FROM metric_sampler.step_index WHERE exp = 'e' AND run = 'a' AND step >= 100 "
        f"GROUP BY step ORDER BY step LIMIT 1 OFFSET {offset}"
    ]


def test_summary_loop_ends_a_chunk_at_the_step_that_is_the_chunk_size_ingested_steps_further_on(tmp_path) -> None:
    loop = Loop(
        tmp_path, {PENDING_RUNS: [run_row(frontier=120)], STEP_LOOKUP: ["105\n", "112\n", ""]}
    )

    assert loop.cycles(1) == [
        "e/a 100-105", "progress e/a 105", "e/a 106-112", "progress e/a 112", "e/a 113-120", "progress e/a 120",
    ]
    assert [re.search(r"step >= (\d+)", lookup)[1] for lookup in loop.client.matching(STEP_LOOKUP)] == [
        "100", "106", "113",
    ]


def test_summary_loop_cuts_a_chunk_at_the_frontier_so_that_progress_reaches_every_complete_step(tmp_path) -> None:
    loop = Loop(
        tmp_path, {PENDING_RUNS: [run_row(frontier=108)], STEP_LOOKUP: ["105\n", "112\n"]}
    )

    assert loop.cycles(1) == ["e/a 100-105", "progress e/a 105", "e/a 106-108", "progress e/a 108", "e/a 109-112"]


def test_summary_loop_inserts_each_chunk_with_the_limits_that_keep_its_memory_bounded(tmp_path) -> None:
    loop = Loop(tmp_path, {PENDING_RUNS: [run_row()]})

    loop.cycles(1)

    assert loop.client.matching(CHUNK_INSERT) == [
        "INSERT INTO metric_sampler.summary_test SELECT now64(3), * FROM "
        "(SELECT * FROM v(step_from = 100) WHERE exp = 'e' AND run = 'a' AND step BETWEEN 100 AND 120) "
        "SETTINGS max_threads = 4, priority = 10, max_execution_time = 1800, max_memory_usage = 30000000000, "
        "max_bytes_before_external_group_by = 10000000000, optimize_move_to_prewhere_if_final = 1, "
        "log_comment = 'summary-test'"
    ]


def test_summary_loop_inserts_chunks_through_the_slots_all_the_loops_share(tmp_path) -> None:
    loop = Loop(tmp_path, {PENDING_RUNS: [run_row()]})

    loop.cycles(1)

    assert sorted(path.name for path in loop.state.glob("summaries-slot-*.lock")) == ["summaries-slot-1.lock"]


def test_summary_loop_logs_each_chunk_and_waits_before_asking_again(tmp_path) -> None:
    loop = Loop(tmp_path, {PENDING_RUNS: [run_row(frontier=120) + run_row(run="b", max_step=45, finished=39)]})

    loop.cycles(1)

    assert loop.log_lines() == [
        f"{START_TIMESTAMP} test e a 100-120 (complete through 120)",
        f"{START_TIMESTAMP} test e b 40-45 (complete through 0)",
    ]
    assert loop.clock.sleeps == [60]


def test_summary_loop_reports_a_failed_chunk_and_leaves_its_run_for_the_next_cycle(tmp_path) -> None:
    rules = {
        PENDING_RUNS: [run_row(frontier=120) + run_row(run="b", max_step=45, finished=39)],
        CHUNK_INSERT: [failure(MEMORY_ERROR), ""],
    }
    loop = Loop(tmp_path, rules)

    assert loop.cycles(1) == ["e/a 100-120", "e/b 40-45"]
    assert loop.log_lines() == [
        MEMORY_ERROR.strip(), CURL_ERROR.strip(),
        f"{START_TIMESTAMP} test e a 100-120 failed",
        f"{START_TIMESTAMP} test e b 40-45 (complete through 0)",
    ]
    assert computed_files(loop.state) == {"summaries-test.e.b.computed": "40-45 T1\n"}


def test_summary_loop_stops_a_runs_chunks_at_the_first_failed_chunk(tmp_path) -> None:
    rules = {PENDING_RUNS: [run_row(frontier=120)], STEP_LOOKUP: ["105\n", ""], CHUNK_INSERT: ["", failure()]}
    loop = Loop(tmp_path, rules)

    assert loop.cycles(1) == ["e/a 100-105", "progress e/a 105", "e/a 106-120"]
    assert loop.client.matching(PROGRESS_INSERT) == [
        "INSERT INTO metric_sampler.summary_progress VALUES ('test', 'e', 'a', 105, now64(3))"
    ]


def test_summary_loop_keeps_going_when_the_progress_could_not_be_recorded(tmp_path) -> None:
    loop = Loop(tmp_path, {PENDING_RUNS: [run_row(frontier=120)], PROGRESS_INSERT: [failure(MEMORY_ERROR)]})

    loop.cycles(1)

    assert loop.log_lines() == [
        MEMORY_ERROR.strip(), CURL_ERROR.strip(), f"{START_TIMESTAMP} test e a 100-120 (complete through 120)"
    ]


def test_summary_loop_retries_after_a_minute_when_the_runs_cannot_be_listed(tmp_path) -> None:
    loop = Loop(tmp_path, {PENDING_RUNS: [failure(MEMORY_ERROR, CURL_ERROR), run_row()]})

    assert loop.cycles(1) == []
    assert loop.log_lines() == [CURL_ERROR.strip()]
    assert loop.clock.sleeps == [60]
    assert loop.cycles(1) == ["e/a 100-120"]


def test_summary_loop_ends_a_cycle_when_it_has_run_for_five_minutes_so_the_freshest_runs_are_revisited(
    tmp_path,
) -> None:
    loop = Loop(tmp_path, {PENDING_RUNS: [run_row(run="a") + run_row(run="b") + run_row(run="c")]})
    loop.advance_on_insert(400)

    assert loop.cycles(1) == ["e/a 100-120"]


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (299, ["e/a 100-105", "progress e/a 105", "e/a 106-112", "progress e/a 112"]),
        (300, ["e/a 100-105", "progress e/a 105"]),
    ],
)
def test_summary_loop_ends_a_cycle_inside_a_run_when_five_minutes_have_passed(tmp_path, seconds, expected) -> None:
    rules = {PENDING_RUNS: [run_row(frontier=120) + run_row(run="b")], STEP_LOOKUP: ["105\n", "112\n", ""]}
    loop = Loop(tmp_path, rules)
    loop.advance_on_insert(seconds)

    assert loop.cycles(1) == expected


def test_summary_loop_does_not_wait_for_anything_when_a_run_has_nothing_beyond_its_progress(tmp_path) -> None:
    loop = Loop(tmp_path, {PENDING_RUNS: [run_row(max_step=100, finished=100)]})

    assert loop.cycles(1) == []
    assert not loop.client.matching(STEP_LOOKUP)


def test_summary_loop_runs_cycle_after_cycle_until_the_service_stops(tmp_path) -> None:
    loop = Loop(tmp_path, {PENDING_RUNS: [""]}, stop_after=3)

    loop.loop.run()

    assert len(loop.client.matching(PENDING_RUNS)) == 3
    assert loop.clock.sleeps == [60, 60, 60]


@pytest.mark.parametrize(
    ("frontier", "expected"),
    [
        pytest.param(-1, ["e/a 100-120"], id="a run whose first step is not complete is computed but not final"),
        pytest.param(50, ["e/a 100-120"], id="a frontier before the chunk leaves it as it is"),
    ],
)
def test_summary_loop_does_not_cut_a_chunk_at_a_frontier_before_its_first_step(tmp_path, frontier, expected) -> None:
    loop = Loop(tmp_path, {PENDING_RUNS: [run_row(frontier=frontier)]})

    assert loop.cycles(1) == expected
