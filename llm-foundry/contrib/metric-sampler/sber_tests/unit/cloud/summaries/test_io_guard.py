from __future__ import annotations

import pytest

from metric_sampler.cloud.summaries import io_guard
from metric_sampler.cloud.summaries.io_guard import IoGuard
from metric_sampler.cloud.summaries.log import FileLog
from metric_sampler.cloud.summaries.pressure import IoPressure
from sber_tests.unit.cloud.summaries.fakes import (
    CURL_ERROR,
    MEMORY_ERROR,
    FakeClickHouse,
    FakeClock,
    failure,
    stamp,
)

pytestmark = pytest.mark.unit

START = "SYSTEM START MERGES"
STOP = "SYSTEM STOP MERGES"
CALM, DIRTY, BUSY, EASING = (1_000, 10), (50_000_000, 10), (1_000, 70), (1_000, 40)


def meminfo(dirty_kb: int) -> str:
    return f"MemTotal: 377000000 kB\nDirty: {dirty_kb} kB\n"


def pressure(io_full: float) -> str:
    return (
        "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"
        f"full avg10=99.00 avg60={io_full:.2f} avg300=0.00 total=0\n"
    )


class Guard:
    """An IO guard that finds the n-th (dirty kB, IO full %) state of the disk before its n-th check."""

    def __init__(self, tmp_path, states, rules=None) -> None:
        self.meminfo_path = tmp_path / "meminfo"
        self.pressure_path = tmp_path / "pressure"
        self.states = states
        self.write_state(0)
        self.clock = FakeClock(len(states), on_sleep=self.write_state)
        self.client = FakeClickHouse(rules)
        self.log_path = tmp_path / "io_guard.log"
        self.loop = IoGuard(
            self.client, self.clock, FileLog(self.log_path, self.clock), IoPressure(self.pressure_path),
            self.meminfo_path,
        )

    def write_state(self, index: int) -> None:
        if index < len(self.states):
            dirty_kb, io_full = self.states[index]
            self.meminfo_path.write_text(meminfo(dirty_kb), encoding="utf-8")
            self.pressure_path.write_text(pressure(io_full), encoding="utf-8")

    def log_lines(self) -> list[str]:
        return self.log_path.read_text(encoding="utf-8").splitlines() if self.log_path.exists() else []


@pytest.mark.parametrize(
    ("states", "expected"),
    [
        pytest.param([CALM, CALM], [START], id="merges are started on start and left alone while calm"),
        pytest.param([DIRTY, CALM], [START, STOP, START], id="dirty pages stop merges until they drain"),
        pytest.param(
            [BUSY, EASING, CALM], [START, STOP, START],
            id="io pressure stops merges, which stay stopped until it falls below the low mark",
        ),
        pytest.param([BUSY] * 3, [START, STOP], id="merges stay stopped while the disk is busy"),
        pytest.param(
            [DIRTY, (9_999_999, 40), (10_000_000, 10)], [START, STOP], id="dirty pages have to fall below the low mark"
        ),
        pytest.param(
            [BUSY, (1_000, 30), (1_000, 29)], [START, STOP, START], id="io pressure has to fall below the low mark"
        ),
        pytest.param(
            [(40_000_000, 10), (40_000_001, 10)], [START, STOP], id="dirty pages have to exceed the high mark"
        ),
        pytest.param([(1_000, 50), (1_000, 51)], [START, STOP], id="io pressure has to exceed the high mark"),
        pytest.param([DIRTY, CALM, DIRTY, CALM], [START, STOP, START, STOP, START], id="merges can be stopped again"),
    ],
)
def test_io_guard_stops_merges_while_the_disk_is_behind(tmp_path, states, expected) -> None:
    guard = Guard(tmp_path, states)

    guard.loop.run()

    assert guard.client.statements == expected


def test_io_guard_drops_the_fraction_of_the_io_percentage_before_comparing(tmp_path) -> None:
    guard = Guard(tmp_path, [(1_000, 50)])
    guard.pressure_path.write_text(pressure(50.99), encoding="utf-8")

    guard.loop.run()

    assert guard.client.statements == [START]


def test_io_guard_starts_merges_after_the_longest_pause_even_when_the_disk_is_still_behind(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setattr(io_guard, "MAX_PAUSE_SECONDS", 30)
    guard = Guard(tmp_path, [BUSY] * 4)

    guard.loop.run()

    assert guard.client.statements == [START, STOP, START, STOP]
    assert io_guard.CHECK_SECONDS == 15


def test_io_guard_pauses_merges_for_at_most_the_default_of_half_an_hour() -> None:
    assert io_guard.MAX_PAUSE_SECONDS == 1800


def test_io_guard_logs_what_made_it_stop_and_start_merges(tmp_path) -> None:
    guard = Guard(tmp_path, [(50_000_000, 80), CALM])

    guard.loop.run()

    assert guard.log_lines() == [
        f"{stamp(0)} merges stopped: dirty 50000000 kB, io full 80%",
        f"{stamp(15)} merges started: dirty 1000 kB, io full 10%",
    ]


def test_io_guard_checks_every_fifteen_seconds(tmp_path) -> None:
    guard = Guard(tmp_path, [CALM] * 3)

    guard.loop.run()

    assert guard.clock.sleeps == [15, 15, 15]


def test_io_guard_tries_again_at_the_next_check_when_stopping_merges_failed(tmp_path) -> None:
    guard = Guard(tmp_path, [BUSY] * 3, {STOP: [failure(MEMORY_ERROR), failure(MEMORY_ERROR), ""]})

    guard.loop.run()

    assert guard.client.statements == [START, STOP, STOP, STOP]
    assert guard.log_lines() == [
        MEMORY_ERROR.strip(), CURL_ERROR.strip(),
        MEMORY_ERROR.strip(), CURL_ERROR.strip(),
        f"{stamp(30)} merges stopped: dirty 1000 kB, io full 70%",
    ]


def test_io_guard_keeps_merges_marked_stopped_when_starting_them_again_failed(tmp_path) -> None:
    guard = Guard(tmp_path, [DIRTY, CALM, CALM], {START: ["", failure(MEMORY_ERROR), ""]})

    guard.loop.run()

    assert guard.client.statements == [START, STOP, START, START]
    assert guard.log_lines() == [
        f"{stamp(0)} merges stopped: dirty 50000000 kB, io full 10%",
        MEMORY_ERROR.strip(), CURL_ERROR.strip(),
        f"{stamp(30)} merges started: dirty 1000 kB, io full 10%",
    ]


def test_io_guard_goes_on_when_merges_could_not_be_started_on_start(tmp_path) -> None:
    guard = Guard(tmp_path, [CALM], {START: [failure(MEMORY_ERROR)]})

    guard.loop.run()

    assert guard.client.statements == [START]
    assert guard.log_lines() == [MEMORY_ERROR.strip(), CURL_ERROR.strip()]


def test_a_restarted_io_guard_starts_merges_first_and_forgets_that_it_stopped_them(tmp_path) -> None:
    guard = Guard(tmp_path, [BUSY, BUSY, BUSY])
    guard.clock = FakeClock(1, on_sleep=guard.write_state)
    guard.loop = IoGuard(
        guard.client, guard.clock, FileLog(guard.log_path, guard.clock), IoPressure(guard.pressure_path),
        guard.meminfo_path,
    )

    guard.loop.run()
    guard.clock.stopped.clear()
    guard.loop.run()

    assert guard.client.statements == [START, STOP, START, STOP]


def test_io_guard_without_a_dirty_line_in_meminfo_is_an_error(tmp_path) -> None:
    guard = Guard(tmp_path, [CALM])
    guard.meminfo_path.write_text("MemTotal: 377000000 kB\n", encoding="utf-8")

    with pytest.raises(ValueError, match="has no Dirty line"):
        guard.loop.cycle()
