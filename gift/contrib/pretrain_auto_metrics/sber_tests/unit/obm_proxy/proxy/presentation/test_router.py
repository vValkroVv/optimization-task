from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.testclient import TestClient

import pytest

from node_runner.src.obm_proxy.proxy.config import ProxySettings
from node_runner.src.obm_proxy.proxy.presentation.router import build_router

pytestmark = pytest.mark.unit


def _settings(mode: str) -> ProxySettings:
    return ProxySettings(
        sglang_port=30000,
        tokenizer_path="/tmp/tokenizer",
        reasoning_token_budget=0,
        template_version_override=None,
        chat_template_kwargs={},
        chat_template_path=None,
        stats_enabled=False,
        stats_interval_sec=60,
        chat_handlers_mode=mode,
        logging_level="info",
    )


class _DummyService:
    def __init__(self):
        self.calls = {
            "custom": 0,
            "v2": 0,
            "opensource": 0,
            "completions": 0,
            "generate": 0,
        }

    async def handle_chat_completions_custom(self, request: Request, body):
        self.calls["custom"] += 1
        return JSONResponse(
            {"mode": "custom", "path": request.url.path, "model": body.model}
        )

    async def handle_chat_completions_v2(self, request: Request, body):
        self.calls["v2"] += 1
        return JSONResponse({"mode": "v2", "path": request.url.path})

    async def handle_chat_completions_opensource(self, request: Request, body):
        self.calls["opensource"] += 1
        return Response(
            content=f"opensource:{request.url.path}:{body.model}".encode(),
            status_code=207,
        )

    async def handle_completions(self, request: Request):
        self.calls["completions"] += 1
        return Response(
            content=f"completions:{request.url.path}".encode(), status_code=201
        )

    async def handle_generate(self, request: Request):
        self.calls["generate"] += 1
        return Response(
            content=f"generate:{request.url.path}".encode(), status_code=202
        )


def _make_app(service: _DummyService, mode: str) -> FastAPI:
    app = FastAPI()
    app.include_router(build_router(service, _settings(mode)))
    return app


class TestRouterChatModes:
    def test_custom_mode_routes_chat_to_custom_handler(self):
        service = _DummyService()
        app = _make_app(service, "custom")
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert response.status_code == 200
        assert response.json()["mode"] == "custom"
        assert service.calls["custom"] == 1
        assert service.calls["opensource"] == 0

    def test_custom_mode_routes_v2_chat_to_v2_handler(self):
        service = _DummyService()
        app = _make_app(service, "custom")
        with TestClient(app) as client:
            response = client.post(
                "/v2/chat/completions",
                json={"messages": [{"role": "user", "content": [{"text": "hi"}]}]},
            )
        assert response.status_code == 200
        assert response.json()["mode"] == "v2"
        assert service.calls["v2"] == 1
        assert service.calls["custom"] == 0

    def test_opensource_mode_routes_chat_to_opensource_handler(self):
        service = _DummyService()
        app = _make_app(service, "opensource")
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert response.status_code == 207
        assert response.content.decode() == "opensource:/v1/chat/completions:m"
        assert service.calls["opensource"] == 1
        assert service.calls["custom"] == 0

    def test_pretrain_mode_returns_404_without_chat_template(self):
        service = _DummyService()
        app = _make_app(service, "pretrain")
        with TestClient(app) as client:
            response = client.post("/v1/chat/completions", json={"anything": 1})
        assert response.status_code == 404
        assert response.json() == {"detail": "No chat-template-provided"}
        assert service.calls["custom"] == 0
        assert service.calls["opensource"] == 0


class TestRouterSharedEndpoints:
    def test_completions_and_generate_delegate_to_service(self):
        service = _DummyService()
        app = _make_app(service, "pretrain")
        with TestClient(app) as client:
            completions = client.post("/v1/completions", json={"prompt": "x"})
            generate = client.post("/generate", json={"input_ids": [1, 2]})
        assert completions.status_code == 201
        assert completions.content.decode() == "completions:/v1/completions"
        assert generate.status_code == 202
        assert generate.content.decode() == "generate:/generate"
        assert service.calls["completions"] == 1
        assert service.calls["generate"] == 1

    @pytest.mark.parametrize("path", ["/ready", "/health", "/ping"])
    def test_health_endpoints_return_ready(self, path: str):
        service = _DummyService()
        app = _make_app(service, "pretrain")
        with TestClient(app) as client:
            response = client.get(path)
        assert response.status_code == 200
        assert response.json() == {"ready": True}
