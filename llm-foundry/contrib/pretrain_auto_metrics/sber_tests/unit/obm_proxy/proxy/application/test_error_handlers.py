import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from node_runner.src.obm_proxy.proxy.application.error_handlers import (
    register_error_handlers,
)
from node_runner.src.obm_proxy.proxy.infrastructure.errors import (
    SGLangNonRetryableStatusError,
    SGLangRetryableRequestError,
    SGLangRetryableStatusError,
)

pytestmark = pytest.mark.unit


class TestErrorHandlers:
    def test_request_validation_error_handler_logs_request(
        self, caplog: pytest.LogCaptureFixture
    ):
        app = FastAPI()
        register_error_handlers(app, sglang_port=30000)

        class _Payload(BaseModel):
            x: int

        @app.post("/validate")
        def validate(payload: _Payload):
            return {"x": payload.x}

        with caplog.at_level(logging.WARNING, logger="obm_proxy"):
            client = TestClient(app)
            response = client.post("/validate", json={"x": "bad"})

        assert response.status_code == 422
        assert "detail" in response.json()
        assert "422 validation failed: POST /validate" in caplog.text

    def test_retryable_status_error_handler(self):
        app = FastAPI()
        register_error_handlers(app, sglang_port=30000)

        @app.get("/boom")
        def boom():
            raise SGLangRetryableStatusError(
                status_code=503, response_text="overloaded"
            )

        client = TestClient(app)
        response = client.get("/boom")
        assert response.status_code == 429
        assert "returned 503" in response.json()["detail"]

    def test_retryable_request_error_handler(self):
        app = FastAPI()
        register_error_handlers(app, sglang_port=31111)

        @app.get("/boom")
        def boom():
            raise SGLangRetryableRequestError("connection refused")

        client = TestClient(app)
        response = client.get("/boom")
        assert response.status_code == 502
        assert "port 31111" in response.json()["detail"]

    def test_non_retryable_status_error_handler(self):
        app = FastAPI()
        register_error_handlers(app, sglang_port=30000)

        @app.get("/boom")
        def boom():
            raise SGLangNonRetryableStatusError(
                status_code=400, response_text="bad input"
            )

        client = TestClient(app)
        response = client.get("/boom")
        assert response.status_code == 400
        assert "returned 400" in response.json()["detail"]
