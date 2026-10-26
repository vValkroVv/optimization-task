import asyncio
import json

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

import pytest

from node_runner.src.obm_proxy.proxy.middleware import (
    MetricsMiddleware,
    MetricsReporter,
    ProxyMetrics,
)

pytestmark = pytest.mark.unit


def _app_with_metrics(metrics: ProxyMetrics) -> FastAPI:
    app = FastAPI()
    app.add_middleware(MetricsMiddleware, metrics=metrics)

    @app.get("/work")
    async def work() -> JSONResponse:
        return JSONResponse(content={"ok": True})

    @app.get("/boom")
    async def boom() -> JSONResponse:
        raise RuntimeError("boom")

    @app.get("/ready")
    async def ready() -> JSONResponse:
        return JSONResponse(content={"ready": True})

    return app


class TestMetricsMiddleware:
    def test_records_success_request_stats(self):
        metrics = ProxyMetrics()
        client = TestClient(_app_with_metrics(metrics))

        response = client.get("/work")
        assert response.status_code == 200

        snap = metrics.snapshot()
        assert snap["total"] == 1
        assert snap["errors"] == 0
        assert snap["active"] == 0
        assert snap["peak"] == 1
        assert snap["routes"]["GET /work"]["ok"] == 1
        assert snap["routes"]["GET /work"]["4xx"] == 0
        assert snap["routes"]["GET /work"]["5xx"] == 0

    def test_skips_health_paths(self):
        metrics = ProxyMetrics()
        client = TestClient(_app_with_metrics(metrics))

        response = client.get("/ready")
        assert response.status_code == 200

        snap = metrics.snapshot()
        assert snap["total"] == 0
        assert snap["routes"] == {}

    def test_records_exception_as_5xx_and_error_type(self):
        metrics = ProxyMetrics()
        client = TestClient(_app_with_metrics(metrics), raise_server_exceptions=False)

        response = client.get("/boom")
        assert response.status_code == 500

        snap = metrics.snapshot()
        assert snap["total"] == 1
        assert snap["errors"] == 1
        assert snap["routes"]["GET /boom"]["5xx"] == 1
        assert snap["error_types"]["RuntimeError"] == 1

    def test_peak_does_not_grow_when_active_not_exceeding_previous_peak(self):
        metrics = ProxyMetrics()
        metrics.on_start()
        metrics.on_finish(
            route="GET /work",
            status=200,
            latency_ms=1.0,
            exc_name=None,
        )
        metrics.on_start()
        snap = metrics.snapshot()
        assert snap["active"] == 1
        assert snap["peak"] == 1


class TestMetricsReporter:
    def test_run_emits_periodic_report_and_log_final(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        metrics = ProxyMetrics()
        metrics.on_start()
        metrics.on_finish(
            route="GET /work",
            status=200,
            latency_ms=12.3,
            exc_name="RuntimeError",
        )
        reporter = MetricsReporter(metrics)

        sleep_calls = {"count": 0}

        async def _fake_sleep(_interval_sec: int):
            sleep_calls["count"] += 1
            if sleep_calls["count"] > 1:
                raise asyncio.CancelledError()

        now = {"value": 100.0}

        def _fake_monotonic() -> float:
            now["value"] += 1.0
            return now["value"]

        monkeypatch.setattr(
            "node_runner.src.obm_proxy.proxy.middleware.asyncio.sleep",
            _fake_sleep,
        )
        monkeypatch.setattr(
            "node_runner.src.obm_proxy.proxy.middleware.time.monotonic",
            _fake_monotonic,
        )

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(reporter.run(interval_sec=1))

        reporter.log_final()
        output = capsys.readouterr().out.strip().splitlines()
        periodic = [
            line
            for line in output
            if "[obm_proxy][perf]" in line and '"kind": "periodic"' in line
        ]
        final = [
            line
            for line in output
            if "[obm_proxy][perf]" in line and '"kind": "final"' in line
        ]
        assert periodic
        assert final

        payload = json.loads(periodic[0].split(" ", 1)[1])
        assert payload["total"] == 1
        assert payload["error_types"]["RuntimeError"] == 1

    def test_run_omits_error_types_when_empty(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        metrics = ProxyMetrics()
        metrics.on_start()
        metrics.on_finish(
            route="GET /work",
            status=200,
            latency_ms=10.0,
            exc_name=None,
        )
        reporter = MetricsReporter(metrics)

        sleep_calls = {"count": 0}

        async def _fake_sleep(_interval_sec: int):
            sleep_calls["count"] += 1
            if sleep_calls["count"] > 1:
                raise asyncio.CancelledError()

        now = {"value": 100.0}

        def _fake_monotonic() -> float:
            now["value"] += 1.0
            return now["value"]

        monkeypatch.setattr(
            "node_runner.src.obm_proxy.proxy.middleware.asyncio.sleep",
            _fake_sleep,
        )
        monkeypatch.setattr(
            "node_runner.src.obm_proxy.proxy.middleware.time.monotonic",
            _fake_monotonic,
        )

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(reporter.run(interval_sec=1))

        output = capsys.readouterr().out.strip().splitlines()
        periodic = [
            line
            for line in output
            if "[obm_proxy][perf]" in line and '"kind": "periodic"' in line
        ]
        assert periodic
        payload = json.loads(periodic[0].split(" ", 1)[1])
        assert payload["total"] == 1
        assert "error_types" not in payload
