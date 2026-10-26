from __future__ import annotations

import pytest

from metric_sampler.cloud.summaries.catalog import Summary
from metric_sampler.cloud.summaries.log import Log
from metric_sampler.cloud.summaries.recompute import Recompute
from metric_sampler.cloud.summaries.slots import InsertSlots
from sber_tests.unit.cloud.summaries.fakes import (
    CURL_ERROR,
    MEMORY_ERROR,
    START_TIMESTAMP,
    FakeClickHouse,
    FakeClock,
    failure,
)

pytestmark = pytest.mark.unit

SUMMARY = Summary("test", "SELECT * FROM v(step_from = __FROM__) WHERE __WHERE__")
INSERT = r"^INSERT INTO metric_sampler\.summary_test "


class Job:
    def __init__(self, tmp_path, rules=None, chunk_steps=None) -> None:
        self.clock = FakeClock()
        self.client = FakeClickHouse(rules)
        slots = InsertSlots(self.client, self.clock, tmp_path)
        log = Log(self.clock)
        self.recompute = Recompute(slots, log) if chunk_steps is None else Recompute(slots, log, chunk_steps)

    def chunks(self) -> list[str]:
        return [statement.split("BETWEEN ")[1].split(")")[0] for statement in self.client.matching(INSERT)]


def test_recompute_replaces_the_rows_of_a_range_in_chunks_of_ten_steps(tmp_path, capsys) -> None:
    job = Job(tmp_path)

    assert job.recompute.run(SUMMARY, "e", "a", 1, 25)

    assert job.chunks() == ["1 AND 10", "11 AND 20", "21 AND 25"]
    assert capsys.readouterr().out.splitlines() == [
        f"{START_TIMESTAMP} test 1-10",
        f"{START_TIMESTAMP} test 11-20",
        f"{START_TIMESTAMP} test 21-25",
        f"{START_TIMESTAMP} test done 1-25",
    ]


def test_recompute_inserts_a_chunk_as_the_summary_loop_does_with_more_threads_and_its_own_log_comment(tmp_path) -> None:
    job = Job(tmp_path)

    job.recompute.run(SUMMARY, "e", "a", 1, 5)

    assert job.client.matching(INSERT) == [
        "INSERT INTO metric_sampler.summary_test SELECT now64(3), * FROM "
        "(SELECT * FROM v(step_from = 1) WHERE exp = 'e' AND run = 'a' AND step BETWEEN 1 AND 5) "
        "SETTINGS max_threads = 6, priority = 10, max_execution_time = 1800, max_memory_usage = 30000000000, "
        "max_bytes_before_external_group_by = 10000000000, optimize_move_to_prewhere_if_final = 1, "
        "log_comment = 'recompute-test'"
    ]


def test_recompute_leaves_the_progress_of_the_summary_alone(tmp_path) -> None:
    job = Job(tmp_path)

    job.recompute.run(SUMMARY, "e", "a", 1, 25)

    assert not job.client.matching("summary_progress")


@pytest.mark.parametrize(
    ("first", "last", "chunk_steps", "expected"),
    [
        (7, 7, None, ["7 AND 7"]),
        (11, 30, None, ["11 AND 20", "21 AND 30"]),
        (1, 25, 7, ["1 AND 7", "8 AND 14", "15 AND 21", "22 AND 25"]),
        (5, 9, 100, ["5 AND 9"]),
        (8, 7, None, []),
    ],
)
def test_recompute_cuts_the_range_into_chunks_of_the_size_it_was_given(
    tmp_path, first, last, chunk_steps, expected,
) -> None:
    job = Job(tmp_path, chunk_steps=chunk_steps)

    assert job.recompute.run(SUMMARY, "e", "a", first, last)

    assert job.chunks() == expected


def test_recompute_stops_at_the_first_chunk_that_fails_and_says_so(tmp_path, capsys) -> None:
    job = Job(tmp_path, {INSERT: ["", failure(MEMORY_ERROR)]})

    assert not job.recompute.run(SUMMARY, "e", "a", 1, 25)

    assert job.chunks() == ["1 AND 10", "11 AND 20"]
    captured = capsys.readouterr()
    assert captured.out.splitlines() == [f"{START_TIMESTAMP} test 1-10", MEMORY_ERROR.strip()]
    assert captured.err.splitlines() == [CURL_ERROR.strip(), f"{START_TIMESTAMP} test 11-20 failed"]
