from __future__ import annotations

import threading

import pytest

from metric_sampler.cloud.summaries.catalog import SUMMARIES
from metric_sampler.cloud.summaries.cold_tier import ColdTier
from metric_sampler.cloud.summaries.health import Health
from metric_sampler.cloud.summaries.hist_latest import HistLatest
from metric_sampler.cloud.summaries.io_guard import IoGuard
from metric_sampler.cloud.summaries.log import Log
from metric_sampler.cloud.summaries.loop import Loop
from metric_sampler.cloud.summaries.service import Service
from metric_sampler.cloud.summaries.summary_loop import SummaryLoop
from metric_sampler.cloud.summaries.top_cards import TopCards
from sber_tests.unit.cloud.summaries.fakes import (
    CURL_ERROR,
    START_TIMESTAMP,
    FakeClickHouse,
    FakeClock,
    failure,
    stamp,
)

pytestmark = pytest.mark.unit

THREAD_TIMEOUT_SECONDS = 10
LOOP_NAMES = [summary.name for summary in SUMMARIES] + ["hist_latest", "top_cards", "health", "io_guard", "cold"]


class ScriptedLoop(Loop):
    """A loop whose runs raise or return as scripted."""

    def __init__(self, clock: FakeClock, log: Log, outcomes: list[Exception | None]) -> None:
        super().__init__("scripted", clock, log)
        self.runs = 0
        self._outcomes = outcomes

    def run(self) -> None:
        outcome = self._outcomes[min(self.runs, len(self._outcomes) - 1)]
        self.runs += 1
        if outcome is not None:
            raise outcome
        self._clock.stopped.set()

    def cycle(self) -> None:
        raise AssertionError("the service runs whole loops")


class RecordingLoop(Loop):
    """A loop that records the thread it ran in."""

    def __init__(self, clock: FakeClock, log: Log) -> None:
        super().__init__("recording", clock, log)
        self.threads: list[threading.Thread] = []
        self.ran = threading.Event()

    def run(self) -> None:
        self.threads.append(threading.current_thread())
        self.ran.set()

    def cycle(self) -> None:
        raise AssertionError("the service runs whole loops")


class OneLoopService(Service):
    def __init__(self, loop: Loop, clock: FakeClock, tmp_path) -> None:
        super().__init__(FakeClickHouse(), clock, Log(clock), tmp_path / "logs", tmp_path / "state")
        self._loop = loop

    def create_loops(self) -> list[Loop]:
        return [self._loop]


def make_service(tmp_path, clock: FakeClock | None = None, client: FakeClickHouse | None = None) -> Service:
    clock = clock or FakeClock()
    return Service(client or FakeClickHouse(), clock, Log(clock), tmp_path / "logs", tmp_path / "state")


def test_the_service_runs_one_loop_per_summary_in_the_catalog_order_and_then_the_five_others(tmp_path) -> None:
    loops = make_service(tmp_path).create_loops()

    assert [loop.name for loop in loops] == LOOP_NAMES
    assert [type(loop) for loop in loops] == [
        *[SummaryLoop] * len(SUMMARIES), HistLatest, TopCards, Health, IoGuard, ColdTier,
    ]


@pytest.mark.parametrize(
    ("index", "name", "rules", "content"),
    [
        pytest.param(0, "layer_load", {"": [failure("", CURL_ERROR)]}, CURL_ERROR, id="a summary loop"),
        pytest.param(-4, "top_cards", {"": [failure("", CURL_ERROR)]}, CURL_ERROR, id="the top cards loop"),
        pytest.param(-3, "health", {"": [failure("", CURL_ERROR)]}, CURL_ERROR, id="the health loop"),
        pytest.param(
            -5, "hist_latest", {r"^\s+SELECT exp, run": ["e\ta\t5\n"], "": [failure("", CURL_ERROR)]},
            f"{CURL_ERROR}{START_TIMESTAMP} e a step 5 failed\n", id="the latest-step loop",
        ),
    ],
)
def test_each_loop_logs_to_its_own_file_in_the_log_directory(tmp_path, index, name, rules, content) -> None:
    service = make_service(tmp_path, client=FakeClickHouse(rules))
    (tmp_path / "logs").mkdir()

    service.create_loops()[index].cycle()

    assert [path.name for path in (tmp_path / "logs").iterdir()] == [f"{name}.log"]
    assert (tmp_path / "logs" / f"{name}.log").read_text(encoding="utf-8") == content


def test_run_starts_every_loop_and_logs_each_start_until_the_service_stops(tmp_path, capsys) -> None:
    clock = FakeClock()
    clock.stopped.set()

    make_service(tmp_path, clock).run()

    assert capsys.readouterr().out.splitlines() == [f"{START_TIMESTAMP} started {name}" for name in LOOP_NAMES]
    assert (tmp_path / "logs").is_dir()


def test_run_runs_each_loop_in_a_thread_of_its_own_and_returns_when_the_service_stops(tmp_path) -> None:
    clock = FakeClock()
    loop = RecordingLoop(clock, Log(clock))
    service = OneLoopService(loop, clock, tmp_path)
    runner = threading.Thread(target=service.run)

    runner.start()
    assert loop.ran.wait(THREAD_TIMEOUT_SECONDS)
    clock.stopped.set()
    runner.join(THREAD_TIMEOUT_SECONDS)

    assert not runner.is_alive()
    assert len(loop.threads) == 1
    assert loop.threads[0] is not runner
    assert loop.threads[0].daemon
    assert loop.threads[0].name == "recording"


def test_a_loop_that_raises_is_logged_in_one_line_and_started_again_after_the_retry_delay(tmp_path, capsys) -> None:
    clock = FakeClock()
    loop = ScriptedLoop(clock, Log(clock), [RuntimeError("boom\nsecond line"), None])

    make_service(tmp_path, clock).supervise(loop)

    captured = capsys.readouterr()
    assert captured.err == f"{START_TIMESTAMP} scripted failed: RuntimeError: boom second line\n"
    assert captured.out == f"{stamp(60)} started scripted\n"
    assert clock.sleeps == [60]
    assert loop.runs == 2


def test_a_loop_that_keeps_raising_is_started_again_every_minute(tmp_path, capsys) -> None:
    clock = FakeClock(stop_after=3)
    loop = ScriptedLoop(clock, Log(clock), [ValueError("no")])

    make_service(tmp_path, clock).supervise(loop)

    assert capsys.readouterr().err.splitlines() == [
        f"{stamp(0)} scripted failed: ValueError: no",
        f"{stamp(60)} scripted failed: ValueError: no",
        f"{stamp(120)} scripted failed: ValueError: no",
    ]
    assert clock.sleeps == [60, 60, 60]


def test_a_loop_is_not_started_again_when_the_service_stopped_while_it_waited(tmp_path, capsys) -> None:
    clock = FakeClock(stop_after=1)
    loop = ScriptedLoop(clock, Log(clock), [KeyError("gone")])

    make_service(tmp_path, clock).supervise(loop)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"{START_TIMESTAMP} scripted failed: KeyError: 'gone'\n"
    assert loop.runs == 1


def test_a_failed_statement_is_logged_with_what_curl_said(tmp_path, capsys) -> None:
    clock = FakeClock()
    loop = ScriptedLoop(clock, Log(clock), [failure("", CURL_ERROR), None])

    make_service(tmp_path, clock).supervise(loop)

    assert capsys.readouterr().err == (
        f"{START_TIMESTAMP} scripted failed: ClickHouseError: curl: (22) The requested URL returned error: 500\n"
    )


def test_a_loop_that_ends_because_the_service_stopped_is_not_started_again(tmp_path, capsys) -> None:
    clock = FakeClock()
    loop = ScriptedLoop(clock, Log(clock), [None])

    make_service(tmp_path, clock).supervise(loop)

    captured = capsys.readouterr()
    assert loop.runs == 1
    assert (captured.out, captured.err) == ("", "")
