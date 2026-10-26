from __future__ import annotations

import threading
from pathlib import Path

import pytest

from metric_sampler.cloud.summaries.clickhouse import ClickHouseError
from metric_sampler.cloud.summaries.slots import SLOT_WAIT_SECONDS, SLOTS, InsertSlots
from sber_tests.unit.cloud.summaries.fakes import FakeClickHouse, FakeClock, failure

pytestmark = pytest.mark.unit

HOLD_TIMEOUT_SECONDS = 10


class HeldSlots:
    """Inserts in progress in other threads, each holding a slot until `release` is called."""

    def __init__(self, directory: Path, count: int, slots: int) -> None:
        self._release = threading.Event()
        self._entered = threading.Semaphore(0)
        client = FakeClickHouse()
        client.on_execute = self._hold
        insert_slots = InsertSlots(client, FakeClock(), directory, slots)
        self._threads = [threading.Thread(target=insert_slots.insert, args=("held",)) for _ in range(count)]
        for thread in self._threads:
            thread.start()
        for _ in self._threads:
            assert self._entered.acquire(timeout=HOLD_TIMEOUT_SECONDS)

    def release(self) -> None:
        self._release.set()
        for thread in self._threads:
            thread.join(HOLD_TIMEOUT_SECONDS)

    def _hold(self, statement: str) -> None:
        self._entered.release()
        assert self._release.wait(HOLD_TIMEOUT_SECONDS)


def slot_files(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.glob("summaries-slot-*.lock"))


def test_there_are_eight_slots_and_a_busy_one_is_waited_for_five_seconds() -> None:
    assert (SLOTS, SLOT_WAIT_SECONDS) == (8, 5)


def test_an_insert_runs_in_the_first_slot_and_leaves_it_free_afterwards(tmp_path) -> None:
    client = FakeClickHouse()
    slots = InsertSlots(client, FakeClock(), tmp_path)

    slots.insert("INSERT 1")
    slots.insert("INSERT 2")

    assert client.statements == ["INSERT 1", "INSERT 2"]
    assert slot_files(tmp_path) == ["summaries-slot-1.lock"]


def test_an_insert_takes_the_next_slot_while_an_earlier_one_is_busy(tmp_path) -> None:
    held = HeldSlots(tmp_path, count=2, slots=SLOTS)
    client = FakeClickHouse()

    InsertSlots(client, FakeClock(), tmp_path).insert("INSERT 3")
    held.release()

    assert client.statements == ["INSERT 3"]
    assert slot_files(tmp_path) == ["summaries-slot-1.lock", "summaries-slot-2.lock", "summaries-slot-3.lock"]


def test_an_insert_waits_five_seconds_at_a_time_while_every_slot_is_busy(tmp_path) -> None:
    held = HeldSlots(tmp_path, count=2, slots=2)
    client = FakeClickHouse()
    clock = FakeClock(on_sleep=lambda count: held.release() if count == 2 else None)

    InsertSlots(client, clock, tmp_path, slots=2).insert("INSERT 3")

    assert clock.sleeps == [5, 5]
    assert client.statements == ["INSERT 3"]
    assert slot_files(tmp_path) == ["summaries-slot-1.lock", "summaries-slot-2.lock"]


def test_a_failed_insert_gives_its_slot_back_and_raises_the_error(tmp_path) -> None:
    client = FakeClickHouse({"INSERT": [failure(), ""]})
    slots = InsertSlots(client, FakeClock(), tmp_path)

    with pytest.raises(ClickHouseError):
        slots.insert("INSERT 1")
    slots.insert("INSERT 2")

    assert client.statements == ["INSERT 1", "INSERT 2"]
    assert slot_files(tmp_path) == ["summaries-slot-1.lock"]
