from __future__ import annotations

import time

import pytest

from metric_sampler.cloud.summaries.clock import Clock
from sber_tests.unit.cloud.summaries.fakes import START_TIME, START_TIMESTAMP

pytestmark = pytest.mark.unit


def test_a_timestamp_is_the_utc_time_as_date_prints_it_with_u_and_the_f_and_t_formats(monkeypatch) -> None:
    gmtime = time.gmtime
    monkeypatch.setattr(time, "gmtime", lambda: gmtime(START_TIME))

    assert Clock().timestamp() == START_TIMESTAMP


def test_the_seconds_are_the_whole_seconds_of_the_epoch(monkeypatch) -> None:
    monkeypatch.setattr(time, "time", lambda: START_TIME + 0.9)

    assert Clock().seconds() == START_TIME


def test_sleeping_waits_for_the_given_seconds(monkeypatch) -> None:
    slept = []
    monkeypatch.setattr(time, "sleep", slept.append)

    Clock().sleep(15)

    assert slept == [15]


def test_a_new_clock_is_not_stopped_until_the_service_stops_it() -> None:
    clock = Clock()

    assert not clock.stopped.is_set()
    clock.stopped.set()
    assert clock.stopped.is_set()
