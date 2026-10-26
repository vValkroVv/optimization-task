import asyncio
import importlib
import json
import sys
import types
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx

import pytest

from node_runner.src.obm_proxy.proxy.config import ProxySettings

pytestmark = pytest.mark.unit


def _ensure_transformers_stub() -> None:
    if "transformers" in sys.modules:
        return
    module = types.ModuleType("transformers")
    module.AutoTokenizer = MagicMock()
    sys.modules["transformers"] = module


def _settings(
    *, stats_enabled: bool, chat_handlers_mode: str = "pretrain"
) -> ProxySettings:
    return ProxySettings(
        sglang_port=30000,
        tokenizer_path="/tmp/tokenizer",
        reasoning_token_budget=0,
        template_version_override=None,
        chat_template_kwargs={},
        chat_template_path=None,
        stats_enabled=stats_enabled,
        stats_interval_sec=5,
        chat_handlers_mode=chat_handlers_mode,
        logging_level="info",
    )


class _DummyService:
    def __init__(self):
        self.preload_called = 0

    def preload(self):
        self.preload_called += 1


class _DummyClient:
    def __init__(self):
        self.closed = 0

    async def aclose(self):
        self.closed += 1


class _DummyReporter:
    def __init__(self):
        self.periodic_interval = None
        self.final_calls = 0

    async def run(self, interval_sec: int):
        self.periodic_interval = interval_sec
        await asyncio.Future()

    def log_final(self):
        self.final_calls += 1


class _DummyMetrics:
    def __init__(self, payload: dict):
        self.payload = payload

    def snapshot(self):
        return dict(self.payload)


def _load_main_module(monkeypatch: pytest.MonkeyPatch):
    _ensure_transformers_stub()
    monkeypatch.setenv("OBM_PROXY_TOKENIZER_PATH", "/tmp/tokenizer")
    return importlib.import_module("node_runner.src.obm_proxy.proxy.main")


class TestMainLifespan:
    def test_lifespan_without_stats_starts_and_closes_http_client(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        module = _load_main_module(monkeypatch)
        dummy_service = _DummyService()
        dummy_reporter = _DummyReporter()
        dummy_client = _DummyClient()

        monkeypatch.setattr(module, "service", dummy_service)
        monkeypatch.setattr(module, "metrics_reporter", dummy_reporter)
        monkeypatch.setattr(module, "settings", _settings(stats_enabled=False))
        client_kwargs = {}

        def _async_client(**kwargs):
            client_kwargs.update(kwargs)
            return dummy_client

        monkeypatch.setattr(module.httpx, "AsyncClient", _async_client)

        app = FastAPI()

        async def _run():
            async with module.lifespan(app):
                assert app.state.http_client is dummy_client
                assert app.state.metrics_task is None
                assert dummy_service.preload_called == 1

        asyncio.run(_run())
        assert dummy_client.closed == 1
        assert dummy_reporter.final_calls == 0
        # httpx defaults to 100 pooled connections per worker, which throttled OBM concurrency
        assert client_kwargs["limits"].max_connections is None
        assert client_kwargs["limits"].max_keepalive_connections is None

    def test_lifespan_with_stats_runs_periodic_task_and_logs_final(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        module = _load_main_module(monkeypatch)
        dummy_service = _DummyService()
        dummy_reporter = _DummyReporter()
        dummy_client = _DummyClient()

        monkeypatch.setattr(module, "service", dummy_service)
        monkeypatch.setattr(module, "metrics_reporter", dummy_reporter)
        monkeypatch.setattr(module, "settings", _settings(stats_enabled=True))
        monkeypatch.setattr(module.httpx, "AsyncClient", lambda **_: dummy_client)

        app = FastAPI()

        async def _run():
            async with module.lifespan(app):
                assert app.state.metrics_task is not None
                await asyncio.sleep(0)

        asyncio.run(_run())
        assert dummy_client.closed == 1
        assert dummy_reporter.periodic_interval == 5
        assert dummy_reporter.final_calls == 1

    def test_module_level_app_contains_expected_routes(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        module = _load_main_module(monkeypatch)
        route_paths = {route.path for route in module.app.routes}
        assert "/v1/chat/completions" in route_paths
        assert "/v1/completions" in route_paths
        assert "/generate" in route_paths
        assert "/ready" in route_paths

    def test_latency_forwards_routing_headers_while_bulk_keeps_previous_behavior(
        self, monkeypatch
    ):
        module = _load_main_module(monkeypatch)
        sent_headers = []

        async def proxy_to_sglang(**kwargs):
            sent_headers.append(kwargs["headers"])
            return httpx.Response(200, json={"text": "answer"})

        monkeypatch.setattr(module.sglang_gateway, "proxy_to_sglang", proxy_to_sglang)
        monkeypatch.setattr(module.app.state, "http_client", object(), raising=False)
        client = TestClient(module.app)
        for path in ("/generate", "/latency/generate"):
            response = client.post(
                path,
                json={"input_ids": [1, 2]},
                headers={"x-smg-routing-key": "dialog"},
            )
            assert response.status_code == 200
        assert sent_headers == [{}, {"x-smg-routing-key": "dialog"}]

    def test_get_metrics_returns_snapshot_payload(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        module = _load_main_module(monkeypatch)
        monkeypatch.setattr(module, "metrics", _DummyMetrics({"total": 7}))

        response = asyncio.run(module.get_metrics())
        payload = json.loads(response.body.decode("utf-8"))
        assert payload == {"total": 7}
