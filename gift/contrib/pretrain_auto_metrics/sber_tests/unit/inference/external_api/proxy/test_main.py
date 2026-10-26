import importlib
import asyncio
import json
import runpy
import sys
import types

import httpx
import pytest
from fastapi.testclient import TestClient

from node_runner.src.pam_proxies.models import SGLangChatCompletionRequest

pytestmark = pytest.mark.unit


def _load_main_module(
    monkeypatch: pytest.MonkeyPatch,
    logging_level: str = "debug",
    formatting: str = "openai",
    request_customization: str | None = None,
    user_pass: str | None = None,
):
    module_name = "node_runner.src.inference.external_api.proxy.main"
    sys.modules.pop(module_name, None)
    argv = [
        "external-api-proxy",
        "--host",
        "0.0.0.0",
        "--port",
        "30000",
        "--formatting",
        formatting,
        "--base-url",
        "https://api.example.test",
        "--model",
        "judge-model",
        "--timeout",
        "10",
        "--logging-level",
        logging_level,
        "--max-retries",
        "0",
        "--retry-wait-sec",
        "1",
        "--max-concurrency",
        "1",
    ]
    if request_customization is not None:
        argv.extend(["--request-customization", request_customization])
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("EXTERNAL_API_PROXY_API_KEY", "key")
    if user_pass is None:
        monkeypatch.delenv("EXTERNAL_API_PROXY_USER_PASS", raising=False)
    else:
        monkeypatch.setenv("EXTERNAL_API_PROXY_USER_PASS", user_pass)
    return importlib.import_module(module_name)


def _request(path: str = "/v1/chat/completions"):
    return types.SimpleNamespace(method="POST", url=types.SimpleNamespace(path=path))


def _body() -> SGLangChatCompletionRequest:
    return SGLangChatCompletionRequest(
        model="local-model",
        messages=[{"role": "user", "content": "hi"}],
    )


class _AsyncPostClient:
    def __init__(self, response: httpx.Response):
        self.response = response
        self.calls = []

    async def post(self, url, *, json=None, headers=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return self.response


def test_mapper_is_not_configured_for_invalid_formatting_from_settings(monkeypatch):
    from node_runner.src.inference.external_api.proxy import config

    module_name = "node_runner.src.inference.external_api.proxy.main"
    sys.modules.pop(module_name, None)
    monkeypatch.setattr(
        config,
        "load_settings",
        lambda: config.ExternalApiProxySettings(
            host="0.0.0.0",
            port=30000,
            formatting="unknown",
            base_url="https://api.example.test",
            model="judge-model",
            api_key="key",
            user_pass=None,
            timeout=10,
            logging_level="info",
            max_retries=0,
            retry_wait_sec=1,
            max_concurrency=1,
            request_customization={},
        ),
    )

    module = importlib.import_module(module_name)

    assert not hasattr(module, "mapper")


def test_load_settings_rejects_non_object_request_customization(monkeypatch):
    from node_runner.src.inference.external_api.proxy.config import load_settings

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "external-api-proxy",
            "--host",
            "0.0.0.0",
            "--port",
            "30000",
            "--formatting",
            "openai",
            "--base-url",
            "https://api.example.test",
            "--model",
            "judge-model",
            "--timeout",
            "10",
            "--max-retries",
            "0",
            "--retry-wait-sec",
            "1",
            "--max-concurrency",
            "1",
            "--request-customization",
            "[]",
        ],
    )

    with pytest.raises(RuntimeError, match="must be a JSON object"):
        load_settings()


def test_chat_completions_maps_customized_request_and_response(monkeypatch):
    module = _load_main_module(
        monkeypatch,
        request_customization=json.dumps(
            {
                "metadata": {"source": "secondary-eval"},
                "temperature": 0.2,
            }
        ),
    )
    upstream_calls = []

    async def fake_post_upstream(*, client, payload):
        upstream_calls.append({"client": client, "payload": payload})
        return httpx.Response(
            200,
            json={
                "object": "chat.completion",
                "model": "judge-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3,
                },
            },
        )

    monkeypatch.setattr(module, "post_upstream", fake_post_upstream)

    with TestClient(module.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "local-model",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is in the image?"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,aW1hZ2U=",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                "metadata": {"request_id": "req-1"},
            },
        )
        ready = client.get("/ready")
        unsupported = client.post("/generate")
        completions = client.post("/v1/completions")

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "ok"
    assert upstream_calls[0]["payload"]["model"] == "judge-model"
    assert upstream_calls[0]["payload"]["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in the image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,aW1hZ2U=",
                        "detail": "high",
                    },
                },
            ],
        }
    ]
    assert upstream_calls[0]["payload"]["metadata"] == {
        "request_id": "req-1",
        "source": "secondary-eval",
    }
    assert upstream_calls[0]["payload"]["temperature"] == 0.2
    assert ready.json() == {"ready": True}
    assert unsupported.status_code == 501
    assert unsupported.json() == {
        "detail": "/generate is not supported by External API proxy"
    }
    assert completions.status_code == 501
    assert completions.json() == {
        "detail": "/v1/completions is not supported by External API proxy"
    }


@pytest.mark.parametrize(
    ("upstream", "expected_status"),
    [
        (
            httpx.Response(
                400,
                headers={"content-type": "application/problem+json"},
                content=b'{"error":"bad"}',
            ),
            400,
        ),
        (
            httpx.Response(
                503,
                headers={"content-type": "application/problem+json"},
                content=b'{"error":"busy"}',
            ),
            503,
        ),
    ],
)
def test_chat_completions_forwards_upstream_errors(
    monkeypatch, upstream, expected_status
):
    module = _load_main_module(monkeypatch)
    errors = []
    monkeypatch.setattr(
        module, "log_upstream_error", lambda **kwargs: errors.append(kwargs)
    )

    async def fake_post_upstream(*, client, payload):
        if upstream.status_code == 503:
            raise module.ExternalApiRetryableStatusError(upstream)
        return upstream

    monkeypatch.setattr(module, "post_upstream", fake_post_upstream)

    with TestClient(module.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "local-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == expected_status
    assert response.content == upstream.content
    assert response.headers["content-type"] == "application/problem+json"
    assert errors[0]["upstream_response"] is upstream


def test_chat_completions_returns_502_for_unexpected_upstream_response(monkeypatch):
    module = _load_main_module(monkeypatch)

    async def fake_post_upstream(*, client, payload):
        return httpx.Response(200, content=b"not-json")

    monkeypatch.setattr(module, "post_upstream", fake_post_upstream)

    with TestClient(module.app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "local-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 502
    assert response.json()["detail"].startswith(
        "External API response has unexpected format"
    )


def test_chat_completions_returns_422_when_request_cannot_be_mapped(monkeypatch):
    module = _load_main_module(monkeypatch)
    monkeypatch.setattr(
        module.mapper,
        "map_request",
        lambda body, model: (_ for _ in ()).throw(ValueError("bad request")),
    )

    with TestClient(module.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "local-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "OpenAI request cannot be converted to upstream format: bad request"
    )


def test_post_upstream_sends_auth_header_and_marks_retryable_statuses(monkeypatch):
    module = _load_main_module(monkeypatch)
    client = _AsyncPostClient(httpx.Response(200, json={"ok": True}))

    response = asyncio.run(module.post_upstream(client=client, payload={"p": 1}))

    assert response.status_code == 200
    assert client.calls == [
        {
            "url": "https://api.example.test/v1/chat/completions",
            "json": {"p": 1},
            "headers": {
                "Content-Type": "application/json",
                "Authorization": "Bearer key",
            },
        }
    ]

    for status_code in (429, 500):
        retryable_client = _AsyncPostClient(httpx.Response(status_code))
        with pytest.raises(module.ExternalApiRetryableStatusError):
            asyncio.run(module.post_upstream(client=retryable_client, payload={"p": 1}))


def test_get_api_key_uses_ift_token_provider(monkeypatch):
    module = _load_main_module(monkeypatch, formatting="ift", user_pass="user:pass")

    class TokenProvider:
        async def get_token(self, client):
            return "ift-token"

    module.ift_token_provider = TokenProvider()

    assert asyncio.run(module.get_api_key(object())) == "ift-token"


def test_exception_handlers_return_proxy_errors(monkeypatch):
    module = _load_main_module(monkeypatch)
    logged = []
    request = types.SimpleNamespace(method="POST", url="https://proxy.test/v1/chat")
    monkeypatch.setattr(
        module,
        "log_exception",
        lambda context, exc: logged.append((context, type(exc))),
    )

    request_error = asyncio.run(
        module.handle_request_error(request, httpx.RequestError("boom"))
    )
    unexpected_response = asyncio.run(
        module.handle_unexpected_response_error(
            request, module.ExternalApiUnexpectedResponseError("bad response")
        )
    )
    token_error = asyncio.run(
        module.handle_ift_token_refresh_error(
            request, module.IftTokenRefreshError("bad token")
        )
    )
    unexpected_error = asyncio.run(
        module.handle_unexpected_error(request, RuntimeError("boom"))
    )

    assert request_error.status_code == 502
    assert json.loads(request_error.body)["detail"].startswith(
        "External API request failed"
    )
    assert unexpected_response.status_code == 502
    assert json.loads(unexpected_response.body) == {"detail": "bad response"}
    assert token_error.status_code == 502
    assert json.loads(token_error.body) == {"detail": "bad token"}
    assert unexpected_error.status_code == 500
    assert json.loads(unexpected_error.body) == {
        "detail": "Unexpected External API proxy error"
    }
    assert logged == [
        ("External API transport error", httpx.RequestError),
        ("Unexpected External API response", module.ExternalApiUnexpectedResponseError),
        ("IFT token refresh error", module.IftTokenRefreshError),
        ("Unhandled error for POST https://proxy.test/v1/chat", RuntimeError),
    ]


def test_main_runs_uvicorn_with_loaded_settings(monkeypatch):
    module = _load_main_module(monkeypatch, logging_level="info")
    calls = []
    monkeypatch.setattr(
        module.uvicorn,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    module.main()

    assert calls == [
        (
            (module.app,),
            {"host": "0.0.0.0", "port": 30000, "log_level": "info"},
        )
    ]


def test_module_entrypoint_runs_main(monkeypatch):
    module_name = "node_runner.src.inference.external_api.proxy.main"
    sys.modules.pop(module_name, None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "external-api-proxy",
            "--host",
            "127.0.0.1",
            "--port",
            "30001",
            "--formatting",
            "openai",
            "--base-url",
            "https://api.example.test",
            "--model",
            "judge-model",
            "--timeout",
            "10",
            "--max-retries",
            "0",
            "--retry-wait-sec",
            "1",
            "--max-concurrency",
            "1",
        ],
    )
    calls = []
    monkeypatch.setattr(
        "uvicorn.run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    runpy.run_module(module_name, run_name="__main__")

    assert len(calls) == 1
    assert calls[0][1] == {
        "host": "127.0.0.1",
        "port": 30001,
        "log_level": "info",
    }


def test_log_upstream_error_writes_detailed_json_in_debug(monkeypatch):
    module = _load_main_module(monkeypatch, logging_level="debug")
    messages = []
    monkeypatch.setattr(
        module.external_api_logger, "error", lambda *args: messages.append(args)
    )

    module.log_upstream_error(
        request=_request(),
        incoming_body=_body(),
        upstream_payload={
            "model": "judge-model",
            "messages": [{"role": "user", "content": "hi"}],
        },
        upstream_response=httpx.Response(
            400,
            headers={"x-request-id": "req-1"},
            content=b'{"error":"bad request"}',
        ),
    )

    assert len(messages) == 1
    assert messages[0][0] == "[external_api_proxy][upstream_error] %s"
    payload = json.loads(messages[0][1])
    assert payload["route"] == {"method": "POST", "path": "/v1/chat/completions"}
    assert payload["incoming_request"]["model"] == "local-model"
    assert (
        payload["upstream_request"]["url"]
        == "https://api.example.test/v1/chat/completions"
    )
    assert payload["upstream_request"]["body"]["model"] == "judge-model"
    assert payload["upstream_response"]["status_code"] == 400
    assert payload["upstream_response"]["body"] == '{"error":"bad request"}'


def test_log_debug_exchange_writes_all_four_states(monkeypatch):
    module = _load_main_module(monkeypatch, logging_level="debug")
    messages = []
    monkeypatch.setattr(
        module.external_api_logger, "debug", lambda *args: messages.append(args)
    )

    module.log_debug_exchange(
        request=_request(),
        incoming_body=_body(),
        upstream_payload={"model": "judge-model"},
        upstream_response=httpx.Response(200, content=b'{"answer":"hi"}'),
        mapped_response={"choices": [{"message": {"content": "hi"}}]},
    )

    assert len(messages) == 1
    assert messages[0][0] == "[external_api_proxy][exchange] %s"
    payload = json.loads(messages[0][1])
    assert payload["incoming_request"]["model"] == "local-model"
    assert payload["upstream_request"]["body"] == {"model": "judge-model"}
    assert payload["upstream_response"]["body"] == {"answer": "hi"}
    assert payload["outgoing_response"]["choices"][0]["message"]["content"] == "hi"


def test_log_debug_exchange_falls_back_to_text_for_non_json_upstream(monkeypatch):
    module = _load_main_module(monkeypatch, logging_level="debug")
    messages = []
    monkeypatch.setattr(
        module.external_api_logger, "debug", lambda *args: messages.append(args)
    )

    module.log_debug_exchange(
        request=_request(),
        incoming_body=_body(),
        upstream_payload={"model": "judge-model"},
        upstream_response=httpx.Response(200, content=b"plain text, not json"),
        mapped_response={},
    )

    payload = json.loads(messages[0][1])
    assert payload["upstream_response"]["body"] == "plain text, not json"


def test_log_debug_exchange_silent_without_debug_level(monkeypatch):
    module = _load_main_module(monkeypatch, logging_level="info")
    messages = []
    monkeypatch.setattr(
        module.external_api_logger, "debug", lambda *args: messages.append(args)
    )

    module.log_debug_exchange(
        request=_request(),
        incoming_body=_body(),
        upstream_payload={"model": "judge-model"},
        upstream_response=httpx.Response(200, content=b"{}"),
        mapped_response={},
    )

    assert messages == []


def test_log_upstream_error_writes_details_for_503(monkeypatch):
    module = _load_main_module(monkeypatch, logging_level="debug")
    messages = []
    monkeypatch.setattr(
        module.external_api_logger, "error", lambda *args: messages.append(args)
    )

    module.log_upstream_error(
        request=_request(),
        incoming_body=_body(),
        upstream_payload={"model": "judge-model"},
        upstream_response=httpx.Response(503, content=b'{"error":"busy"}'),
    )

    assert len(messages) == 1
    assert messages[0][0] == "[external_api_proxy][upstream_error] %s"
    payload = json.loads(messages[0][1])
    assert payload["upstream_response"]["status_code"] == 503
    assert payload["upstream_response"]["body"] == '{"error":"busy"}'


def test_log_upstream_error_skips_success_status_in_debug(monkeypatch):
    module = _load_main_module(monkeypatch, logging_level="debug")
    messages = []
    monkeypatch.setattr(
        module.external_api_logger, "error", lambda *args: messages.append(args)
    )

    module.log_upstream_error(
        request=_request(),
        incoming_body=_body(),
        upstream_payload={"model": "judge-model"},
        upstream_response=httpx.Response(200, content=b'{"ok":true}'),
    )

    assert messages == []
