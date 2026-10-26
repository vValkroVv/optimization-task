import threading

import pytest
import requests

import node_runner.src.trace.metrics.scraper as scraper_module
from node_runner.src.trace.metrics.model import MetricSample, ScrapeTarget
from node_runner.src.trace.metrics.parser import MetricsParser
from node_runner.src.trace.metrics.scraper import MetricsScraper

pytestmark = pytest.mark.unit

ENGINE = ScrapeTarget(source="engine", endpoint=":30001", url="http://engine/metrics")
ROUTER = ScrapeTarget(source="router", endpoint=":29000", url="http://router/metrics")


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _StuckThread:
    """A scraper thread that ignores the stop event."""

    def join(self, timeout=None) -> None:
        return None

    def is_alive(self) -> bool:
        return True


def _scraper(*targets: ScrapeTarget) -> MetricsScraper:
    return MetricsScraper(
        log_profile="evaluated", model_name="gigachat", targets=list(targets)
    )


def _sample_for(samples: list[MetricSample], metric: str) -> MetricSample:
    return next(sample for sample in samples if sample.metric == metric)


@pytest.fixture
def logged(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    records: dict[str, list[str]] = {"info": [], "warning": [], "error": []}
    for level in records:
        monkeypatch.setattr(scraper_module.Logger, level, records[level].append)
    return records


@pytest.fixture
def written(monkeypatch: pytest.MonkeyPatch) -> list[list[dict]]:
    batches: list[list[dict]] = []
    monkeypatch.setattr(scraper_module.MetricsStream, "write", batches.append)
    return batches


class TestFetch:
    def test_an_unreachable_target_is_reported_once(
        self, monkeypatch: pytest.MonkeyPatch, logged: dict[str, list[str]]
    ):
        def _refuse(url, timeout):
            raise requests.exceptions.ConnectionError("refused")

        monkeypatch.setattr(scraper_module.requests, "get", _refuse)
        scraper = _scraper(ROUTER)

        assert scraper._fetch(ROUTER) is None
        assert scraper._fetch(ROUTER) is None
        assert len(logged["warning"]) == 1
        assert ROUTER.key in logged["warning"][0]

    def test_recovery_is_reported_once_the_target_answers(
        self, monkeypatch: pytest.MonkeyPatch, logged: dict[str, list[str]]
    ):
        responses = [requests.exceptions.ConnectionError("refused"), "smg_x 1.0"]

        def _flaky(url, timeout):
            answer = responses.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return _FakeResponse(answer)

        monkeypatch.setattr(scraper_module.requests, "get", _flaky)
        scraper = _scraper(ROUTER)

        assert scraper._fetch(ROUTER) is None
        assert scraper._fetch(ROUTER) is not None
        assert len(logged["warning"]) == 1
        assert len(logged["info"]) == 1
        assert "reachable again" in logged["info"][0]


class TestSamples:
    def test_an_engine_sample_carries_the_scrape_context(self):
        scraper = _scraper(ENGINE)
        parser = MetricsParser(
            'sglang:token_usage{model_name="m",tp_rank="0"} 0.25\n'
            'sglang:process_cpu_seconds_total{tp_rank="0"} 4.5'
        )

        samples = scraper._samples(ENGINE, parser)

        assert {
            "log_profile": "evaluated",
            "model_name": "gigachat",
            "source": "engine",
            "endpoint": ENGINE.endpoint,
            "metric": "sglang:token_usage",
            "kind": "gauge",
            "labels": {"model_name": "m", "tp_rank": "0"},
            "value": 0.25,
        }.items() <= _sample_for(samples, "sglang:token_usage").model_dump().items()

    def test_router_samples_keep_their_labels(self):
        scraper = _scraper(ROUTER)
        parser = MetricsParser('smg_worker_health{worker="a"} 1.0')

        sample = _sample_for(scraper._samples(ROUTER, parser), "smg_worker_health")

        assert sample.source == "router"
        assert sample.labels == {"worker": "a"}
        assert sample.value == 1.0

    def test_a_histogram_bucket_keeps_the_exposition_name(self):
        scraper = _scraper(ENGINE)
        parser = MetricsParser(
            'sglang:e2e_request_latency_seconds_bucket{le="0.5",tp_rank="0"} 3\n'
            'sglang:e2e_request_latency_seconds_sum{tp_rank="0"} 12.5\n'
            'sglang:e2e_request_latency_seconds_count{tp_rank="0"} 5'
        )

        samples = scraper._samples(ENGINE, parser)

        bucket = _sample_for(samples, "sglang:e2e_request_latency_seconds_bucket")
        assert bucket.kind == "histogram"
        assert bucket.labels == {"le": "0.5", "tp_rank": "0"}
        assert bucket.value == 3.0
        assert {sample.metric for sample in samples} == {
            "sglang:e2e_request_latency_seconds_bucket",
            "sglang:e2e_request_latency_seconds_sum",
            "sglang:e2e_request_latency_seconds_count",
        }

    def test_every_sample_of_one_scrape_shares_a_timestamp(self):
        scraper = _scraper(ENGINE)
        parser = MetricsParser(
            'sglang:token_usage{tp_rank="0"} 0.25\n'
            'sglang:num_running_reqs{tp_rank="0"} 3'
        )

        samples = scraper._samples(ENGINE, parser)

        assert len({s.ts for s in samples}) == 1


class TestTick:
    def test_only_reachable_targets_contribute_samples(
        self, monkeypatch: pytest.MonkeyPatch, logged, written: list[list[dict]]
    ):
        def _answer(url, timeout):
            if url == ROUTER.url:
                raise requests.exceptions.ConnectionError("refused")
            return _FakeResponse('sglang:token_usage{tp_rank="0"} 0.25')

        monkeypatch.setattr(scraper_module.requests, "get", _answer)
        scraper = _scraper(ENGINE, ROUTER)

        scraper._tick()

        assert [s.source for s in written[0]] == ["engine"]


class TestThreadLifecycle:
    def test_a_scraper_without_targets_never_starts_a_thread(self):
        scraper = _scraper()

        scraper.start()

        assert scraper._thread is None

    def test_starting_twice_keeps_one_thread(self, monkeypatch: pytest.MonkeyPatch):
        scraper = _scraper(ENGINE)
        monkeypatch.setattr(scraper, "_tick", lambda: None)
        monkeypatch.setattr(scraper_module, "SCRAPE_INTERVAL_SECONDS", 0.01)

        scraper.start()
        thread = scraper._thread
        scraper.start()

        assert scraper._thread is thread
        scraper.stop()

    def test_stopping_a_scraper_that_never_started_is_a_noop(self):
        _scraper(ENGINE).stop()  # must not raise

    def test_the_loop_ticks_until_it_is_stopped(self, monkeypatch: pytest.MonkeyPatch):
        scraper = _scraper(ENGINE)
        ticked = threading.Event()
        monkeypatch.setattr(scraper, "_tick", ticked.set)
        monkeypatch.setattr(scraper_module, "SCRAPE_INTERVAL_SECONDS", 0.01)

        scraper.start()
        assert ticked.wait(timeout=5)
        scraper.stop()

        assert scraper._thread.is_alive() is False

    def test_a_failing_tick_does_not_kill_the_loop(
        self, monkeypatch: pytest.MonkeyPatch, logged: dict[str, list[str]]
    ):
        scraper = _scraper(ENGINE)
        ticks = []
        survived = threading.Event()

        def _tick():
            ticks.append(None)
            if len(ticks) == 1:
                raise RuntimeError("boom")
            survived.set()

        monkeypatch.setattr(scraper, "_tick", _tick)
        monkeypatch.setattr(scraper_module, "SCRAPE_INTERVAL_SECONDS", 0.01)

        scraper.start()
        assert survived.wait(timeout=5)
        scraper.stop()

        assert "boom" in logged["warning"][0]

    def test_a_thread_that_outlives_stop_is_reported(
        self, logged: dict[str, list[str]]
    ):
        scraper = _scraper(ENGINE)
        scraper._thread = _StuckThread()

        scraper.stop()

        assert "did not stop in time" in logged["error"][0]
