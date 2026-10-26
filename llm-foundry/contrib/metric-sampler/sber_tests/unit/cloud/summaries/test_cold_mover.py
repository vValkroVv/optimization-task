from __future__ import annotations

import re
import threading

import pytest

from metric_sampler.cloud.summaries.cold_mover import COLD_DISK, LOCAL_DISK, MAX_IO_FULL, PARALLEL, ColdMover
from metric_sampler.cloud.summaries.log import FileLog
from metric_sampler.cloud.summaries.pressure import IoPressure
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

ROWS = r"^SELECT table, disk_name, count\(\)"
LOCAL_PARTS = r"disk_name = 'default'"
COLD_PARTS = r"disk_name = 'obs_cold_cached'"
MOVE = r"^ALTER TABLE"
CHECK = r"^SELECT \* FROM metric_sampler\.\w+ WHERE _part"
ROWS_TEXT = "scalars\tdefault\t3\t1500\nscalars\tobs_cold_cached\t1\t500\n"
BARRIER_TIMEOUT_SECONDS = 10


def pressure(avg10: float) -> str:
    return (
        "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"
        f"full avg10={avg10:.2f} avg60=99.00 avg300=0.00 total=0\n"
    )


class Mover:
    def __init__(self, tmp_path, rules, parallel=PARALLEL, max_io_full=MAX_IO_FULL, avg10=(1.0,)) -> None:
        self.pressure_path = tmp_path / "pressure"
        self.pressure_path.write_text(pressure(avg10[0]), encoding="utf-8")
        self.clock = FakeClock(
            on_sleep=lambda count: self.pressure_path.write_text(
                pressure(avg10[min(count, len(avg10) - 1)]), encoding="utf-8"
            )
        )
        self.client = FakeClickHouse(rules)
        self.log_path = tmp_path / "cold.log"
        self.mover = ColdMover(
            self.client, self.clock, FileLog(self.log_path, self.clock), IoPressure(self.pressure_path), parallel,
            max_io_full,
        )

    def log_lines(self) -> list[str]:
        return self.log_path.read_text(encoding="utf-8").splitlines() if self.log_path.exists() else []

    def moved(self) -> list[str]:
        return [re.search(r"PART '(\w+)'", statement)[1] for statement in self.client.matching(MOVE)]

    def checked(self) -> list[str]:
        return [re.search(r"_part = '(\w+)'", statement)[1] for statement in self.client.matching(CHECK)]


def base_rules(local: str = "p1\np2\np3\n", cold: str = "earlier\np1\np3\n") -> dict:
    return {ROWS: [ROWS_TEXT], LOCAL_PARTS: [local], COLD_PARTS: [cold]}


def test_defaults_are_six_parts_at_a_time_and_a_pause_above_half_of_the_time_waiting_for_io() -> None:
    assert (PARALLEL, MAX_IO_FULL, LOCAL_DISK, COLD_DISK) == (6, 50, "default", "obs_cold_cached")


def test_cold_mover_moves_the_listed_parts_and_reads_back_the_ones_it_moved(tmp_path) -> None:
    mover = Mover(tmp_path, {**base_rules(), r"MOVE PART 'p2'": [failure(MEMORY_ERROR)]})

    mover.mover.move_run("e", "r1", ["router_vectors"])

    assert sorted(mover.moved()) == ["p1", "p2", "p3"]
    assert sorted(mover.checked()) == ["p1", "p3"]
    assert mover.client.matching(MOVE)[0] == "ALTER TABLE metric_sampler.router_vectors MOVE PART 'p1' TO VOLUME 'cold'"
    assert mover.client.matching(CHECK)[0] == (
        "SELECT * FROM metric_sampler.router_vectors WHERE _part = 'p1' FORMAT Null\n"
        "    SETTINGS checksum_on_read = 1, enable_filesystem_cache = 0, max_threads = 2"
    )
    assert mover.log_lines() == [
        f"{START_TIMESTAMP} router_vectors before:",
        *ROWS_TEXT.splitlines(),
        MEMORY_ERROR.strip(), CURL_ERROR.strip(),
        f"{START_TIMESTAMP} router_vectors p2 move failed",
        f"{START_TIMESTAMP} router_vectors checking",
        f"{START_TIMESTAMP} router_vectors after:",
        *ROWS_TEXT.splitlines(),
    ]


def test_cold_mover_lists_the_parts_of_the_run_largest_first_on_each_disk_with_the_age_and_size_it_was_given(
    tmp_path,
) -> None:
    mover = Mover(tmp_path, base_rules())

    mover.mover.move_run("e", "r1", ["router_vectors"], min_age_hours=24, min_bytes=1073741824)

    assert mover.client.matching(LOCAL_PARTS) == [
        "SELECT name FROM system.parts WHERE database = 'metric_sampler' AND table = 'router_vectors' AND active\n"
        "    AND partition = '(''e'',''r1'')' AND disk_name = 'default' AND modification_time <= now() - "
        "toIntervalHour(24)\n"
        "    AND bytes_on_disk >= 1073741824 ORDER BY bytes_on_disk DESC FORMAT TSV"
    ]
    assert mover.client.matching(COLD_PARTS) == [
        "SELECT name FROM system.parts WHERE database = 'metric_sampler' AND table = 'router_vectors' AND active\n"
        "    AND partition = '(''e'',''r1'')' AND disk_name = 'obs_cold_cached' AND modification_time <= now() - "
        "toIntervalHour(0)\n"
        "    AND bytes_on_disk >= 0 ORDER BY bytes_on_disk DESC FORMAT TSV"
    ]


def test_cold_mover_moves_parts_of_any_age_and_size_by_default(tmp_path) -> None:
    mover = Mover(tmp_path, base_rules())

    mover.mover.move_run("e", "r1", ["router_vectors"])

    assert "toIntervalHour(0)\n    AND bytes_on_disk >= 0 ORDER BY" in mover.client.matching(LOCAL_PARTS)[0]


def test_cold_mover_counts_the_rows_of_the_run_on_each_disk_before_and_after(tmp_path) -> None:
    mover = Mover(tmp_path, base_rules())

    mover.mover.move_run("e", "r1", ["scalars"])

    assert mover.client.matching(ROWS) == [
        "SELECT table, disk_name, count(), sum(rows) FROM system.parts WHERE database = 'metric_sampler'\n"
        "    AND table = 'scalars' AND active AND partition = '(''e'',''r1'')' GROUP BY table, disk_name "
        "ORDER BY disk_name FORMAT TSV"
    ] * 2


def test_cold_mover_goes_through_the_tables_one_after_the_other(tmp_path) -> None:
    mover = Mover(tmp_path, base_rules("a\nb\n", "b\nc\n"))

    mover.mover.move_run("e", "r1", ["scalars", "expert_counts"])

    assert [line for line in mover.log_lines() if "before:" in line or "after:" in line] == [
        f"{START_TIMESTAMP} scalars before:", f"{START_TIMESTAMP} scalars after:",
        f"{START_TIMESTAMP} expert_counts before:", f"{START_TIMESTAMP} expert_counts after:",
    ]
    expert_counts_moves = mover.client.matching(r"^ALTER TABLE metric_sampler.expert_counts")
    assert [compact(statement) for statement in expert_counts_moves] == [
        "ALTER TABLE metric_sampler.expert_counts MOVE PART 'a' TO VOLUME 'cold'",
        "ALTER TABLE metric_sampler.expert_counts MOVE PART 'b' TO VOLUME 'cold'",
    ]
    assert mover.checked() == ["b", "b"]


@pytest.mark.parametrize(
    ("cold", "expected"),
    [("z\nc\nb\ny\n", ["c", "b"]), ("z\n", []), ("", [])],
)
def test_cold_mover_reads_back_only_the_listed_parts_that_the_cold_volume_now_holds(tmp_path, cold, expected) -> None:
    mover = Mover(tmp_path, base_rules("a\nb\nc\n", cold))

    mover.mover.move_run("e", "r1", ["scalars"])

    assert mover.checked() == expected


def test_cold_mover_moves_parts_in_batches_of_the_given_size_waiting_for_each_batch(tmp_path) -> None:
    parts = "".join(f"p{number}\n" for number in range(1, 7))
    mover = Mover(tmp_path, base_rules(parts, ""), parallel=3)
    barrier = threading.Barrier(3, timeout=BARRIER_TIMEOUT_SECONDS)
    mover.client.on_execute = lambda statement: barrier.wait() if statement.startswith("ALTER") else None

    mover.mover.move_run("e", "r1", ["scalars"])

    assert [set(mover.moved()[:3]), set(mover.moved()[3:])] == [{"p1", "p2", "p3"}, {"p4", "p5", "p6"}]


def test_cold_mover_reads_back_in_batches_too(tmp_path) -> None:
    parts = "".join(f"p{number}\n" for number in range(1, 5))
    mover = Mover(tmp_path, base_rules(parts, parts), parallel=2)
    barrier = threading.Barrier(2, timeout=BARRIER_TIMEOUT_SECONDS)
    mover.client.on_execute = lambda statement: barrier.wait() if statement.startswith("SELECT *") else None

    mover.mover.move_run("e", "r1", ["scalars"])

    assert [set(mover.checked()[:2]), set(mover.checked()[2:])] == [{"p1", "p2"}, {"p3", "p4"}]


def test_cold_mover_moves_the_last_short_batch_of_a_list(tmp_path) -> None:
    mover = Mover(tmp_path, base_rules("p1\np2\np3\np4\np5\n", ""), parallel=2)

    mover.mover.move_run("e", "r1", ["scalars"])

    assert sorted(mover.moved()) == ["p1", "p2", "p3", "p4", "p5"]


def test_cold_mover_waits_for_the_disk_before_each_batch_while_io_is_above_the_limit(tmp_path) -> None:
    mover = Mover(tmp_path, base_rules("p1\np2\n", ""), parallel=1)
    events = []

    def moves_load_the_disk(statement: str) -> None:
        if statement.startswith("ALTER"):
            events.append("move")
            mover.pressure_path.write_text(pressure(90.0), encoding="utf-8")

    def waiting_lets_it_ease(count: int) -> None:
        events.append("wait")
        mover.pressure_path.write_text(pressure(50.0), encoding="utf-8")

    mover.pressure_path.write_text(pressure(90.0), encoding="utf-8")
    mover.client.on_execute = moves_load_the_disk
    mover.clock.on_sleep = waiting_lets_it_ease

    mover.mover.move_run("e", "r1", ["scalars"])

    assert events == ["wait", "move", "wait", "move"]
    assert mover.clock.sleeps == [30, 30]


def test_cold_mover_waits_for_the_disk_even_when_there_is_nothing_to_move_as_xargs_runs_once_on_an_empty_list(
    tmp_path,
) -> None:
    mover = Mover(tmp_path, base_rules("", "x1\n"), avg10=(80.0, 10.0))

    mover.mover.move_run("e", "r1", ["scalars"])

    assert mover.clock.sleeps == [30]
    assert mover.moved() == []
    assert mover.checked() == []


def test_cold_mover_does_not_wait_while_io_is_at_the_limit_or_when_only_the_minute_average_is_high(tmp_path) -> None:
    mover = Mover(tmp_path, base_rules("p1\n", ""), avg10=(50.0,))

    mover.mover.move_run("e", "r1", ["scalars"])

    assert mover.clock.sleeps == []


def test_cold_mover_pauses_above_the_io_limit_it_was_given(tmp_path) -> None:
    mover = Mover(tmp_path, base_rules("p1\n", ""), max_io_full=20, avg10=(30.0, 20.0))

    mover.mover.move_run("e", "r1", ["scalars"])

    assert mover.clock.sleeps == [30]


def test_cold_mover_logs_the_parts_that_failed_the_read_back_and_the_ones_that_failed_to_move(tmp_path) -> None:
    rules = {**base_rules("p1\np2\n", "p1\np2\n"), r"_part = 'p2'": [failure(MEMORY_ERROR)]}
    mover = Mover(tmp_path, rules)

    mover.mover.move_run("e", "r1", ["scalars"])

    assert mover.log_lines()[4:7] == [
        MEMORY_ERROR.strip(), CURL_ERROR.strip(), f"{START_TIMESTAMP} scalars p2 check failed"
    ]


def test_cold_mover_goes_on_when_the_rows_per_disk_cannot_be_counted(tmp_path) -> None:
    mover = Mover(tmp_path, {**base_rules(), ROWS: [failure(MEMORY_ERROR)]})

    mover.mover.move_run("e", "r1", ["scalars"])

    assert mover.log_lines() == [
        f"{START_TIMESTAMP} scalars before:", MEMORY_ERROR.strip(), CURL_ERROR.strip(),
        f"{START_TIMESTAMP} scalars checking",
        f"{START_TIMESTAMP} scalars after:", MEMORY_ERROR.strip(), CURL_ERROR.strip(),
    ]
    assert sorted(mover.moved()) == ["p1", "p2", "p3"]


def test_cold_mover_moves_nothing_of_a_table_whose_parts_cannot_be_listed(tmp_path) -> None:
    mover = Mover(tmp_path, {**base_rules(), LOCAL_PARTS: [failure(MEMORY_ERROR)]})

    mover.mover.move_run("e", "r1", ["scalars"])

    assert mover.moved() == []
    assert mover.checked() == []
    assert mover.log_lines()[3:5] == [MEMORY_ERROR.strip(), CURL_ERROR.strip()]


def test_cold_mover_does_nothing_without_tables(tmp_path) -> None:
    mover = Mover(tmp_path, base_rules())

    mover.mover.move_run("e", "r1", [])

    assert mover.client.statements == []
    assert mover.log_lines() == []
