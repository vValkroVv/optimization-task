import asyncio
import json
import logging
import sys
import types
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.responses import Response

sys.modules.setdefault("transformers", MagicMock())

from node_runner.src.obm_proxy.proxy.application.service import (  # noqa: E402
    ProxyService,
    _maybe_parse_json_args,
)
from node_runner.src.obm_proxy.proxy.config import ProxySettings  # noqa: E402
from node_runner.src.pam_proxies.models import (  # noqa: E402
    DevicesV2ChatCompletionRequest,
    SGLangChatCompletionRequest,
)
from node_runner.src.obm_proxy.proxy.infrastructure.errors import (  # noqa: E402
    SGLangNonRetryableStatusError,
    SGLangRetryableRequestError,
    SGLangRetryableStatusError,
)
from node_runner.src.obm_proxy.proxy.infrastructure.sglang_gateway import (  # noqa: E402
    SGLangGateway,
)
from tenacity import stop_after_attempt, wait_fixed  # noqa: E402

pytestmark = pytest.mark.unit


class _DummyTokenizerGateway:
    def __init__(self):
        self.calls = []

    def get(self):
        return object()

    def apply_chat_template(
        self, messages, tools, add_generation_prompt, tokenize, **kwargs
    ):
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "add_generation_prompt": add_generation_prompt,
                "tokenize": tokenize,
                "kwargs": kwargs,
            }
        )
        return [1, 2, 3]


class _FailingTokenizerGateway(_DummyTokenizerGateway):
    def apply_chat_template(
        self, messages, tools, add_generation_prompt, tokenize, **kwargs
    ):
        raise RuntimeError("template error")


class _DummyResponse:
    def __init__(self, *, status_code=200, content=b"{}", text=None, headers=None):
        self.status_code = status_code
        self.content = content
        self.text = (
            text if text is not None else content.decode("utf-8", errors="ignore")
        )
        self.headers = headers or {}

    def json(self):
        return json.loads(self.text)


class _DummySGLangGateway:
    def __init__(self, *, generate_response=None, proxy_response=None):
        self.generate_response = generate_response
        self.proxy_response = proxy_response
        self.generate_calls = []
        self.proxy_calls = []

    async def generate(self, *, client, url, payload, headers=None):
        self.generate_calls.append({"client": client, "url": url, "payload": payload})
        self.headers = headers
        return self.generate_response

    async def proxy_to_sglang(self, *, client, url, body, headers=None):
        self.proxy_calls.append({"client": client, "url": url, "body": body})
        self.headers = headers
        return self.proxy_response


class _DummyRequest:
    def __init__(self, *, body=None, path="/v1/chat/completions", headers=None):
        self._body = body
        self.headers = headers or {}
        self.url = types.SimpleNamespace(path=path)
        self.app = types.SimpleNamespace(
            state=types.SimpleNamespace(http_client=object())
        )

    async def json(self):
        return self._body


def _settings(**overrides) -> ProxySettings:
    base = dict(
        sglang_port=30000,
        tokenizer_path="/tmp/tokenizer",
        reasoning_token_budget=7,
        template_version_override=None,
        chat_template_kwargs={"use_devsystem": True},
        chat_template_path=None,
        stats_enabled=False,
        stats_interval_sec=60,
        chat_handlers_mode="custom",
        logging_level="info",
    )
    base.update(overrides)
    return ProxySettings(**base)


def _service(
    *, settings=None, tokenizer_gateway=None, sglang_gateway=None, dialog_affinity=False
) -> ProxyService:
    return ProxyService(
        settings=settings or _settings(),
        tokenizer_gateway=tokenizer_gateway or _DummyTokenizerGateway(),
        sglang_gateway=sglang_gateway or _DummySGLangGateway(),
        dialog_affinity=dialog_affinity,
    )


def _generate_payload(
    *, text, finish_reason="stop", prompt_tokens=2, completion_tokens=3
):
    return json.dumps(
        {
            "text": text,
            "meta_info": {
                "id": "req-1",
                "finish_reason": {"type": finish_reason},
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }
    ).encode("utf-8")


def _chat_completions_payload(
    *,
    content=None,
    reasoning_content=None,
    model="opensource-model",
    finish_reason="stop",
) -> bytes:
    return json.dumps(
        {
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": reasoning_content,
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }
    ).encode("utf-8")


class TestDialogAffinity:
    @pytest.mark.parametrize(
        "key,expected",
        [
            (
                "dialog-a",
                "30f4e4ce037a686854465e6903e96c4ee2caa18961c427a8c59277954a2165eb",
            ),
            (
                "диалог-1",
                "33791a751282ddbfac8e7ee076e0df2cfbf37f335ead924628480ee1152132e6",
            ),
        ],
    )
    def test_v2_explicit_body_key_overrides_header_and_automatic_affinity(
        self, key, expected
    ):
        gateway = _DummySGLangGateway(
            generate_response=_DummyResponse(content=_generate_payload(text="answer"))
        )
        service = _service(sglang_gateway=gateway, dialog_affinity=True)
        body = DevicesV2ChatCompletionRequest(
            model="37b",
            messages=[{"role": "user", "content": [{"text": "Begin"}]}],
            routing_key=key,
        )
        request = _DummyRequest(headers={"x-smg-routing-key": "header-key"})

        asyncio.run(service.handle_chat_completions_v2(request, body))

        assert gateway.headers == {"x-smg-routing-key": expected}

    @pytest.mark.parametrize("api", ["custom", "opensource"])
    @pytest.mark.parametrize("field", ["tools", "functions"])
    def test_empty_and_omitted_tools_keep_the_same_affinity(self, api, field):
        gateway = _DummySGLangGateway(
            generate_response=_DummyResponse(content=_generate_payload(text="answer")),
            proxy_response=_DummyResponse(
                content=_chat_completions_payload(content="answer")
            ),
        )
        tokenizer = _DummyTokenizerGateway()
        service = _service(
            sglang_gateway=gateway,
            tokenizer_gateway=tokenizer,
            dialog_affinity=True,
        )
        handler = (
            service.handle_chat_completions_custom
            if api == "custom"
            else service.handle_chat_completions_opensource
        )
        keys = []
        for extra in ({}, {field: []}):
            body = SGLangChatCompletionRequest(
                model="37b", messages=[{"role": "user", "content": "Begin"}], **extra
            )
            asyncio.run(handler(_DummyRequest(), body))
            keys.append(gateway.headers["x-smg-routing-key"])

        assert keys[0] == keys[1]
        if api == "custom":
            assert tokenizer.calls[0] == tokenizer.calls[1]

    @pytest.mark.parametrize(
        "key,header",
        [
            (
                "диалог-1",
                "33791a751282ddbfac8e7ee076e0df2cfbf37f335ead924628480ee1152132e6",
            ),
            (
                "line\nbreak",
                "697affca60d5d0dc96f00aac4b2d833e3adc0708a4efd8b657acc4229f4449ef",
            ),
            (
                " space ",
                "b36423973d2147c12f2131c952d188dca26c032ef3e8bc62ce2aad1ebb9e4ade",
            ),
        ],
    )
    def test_json_routing_keys_are_safe_as_http_headers_without_rewriting_body(
        self, key, header
    ):
        seen = []

        def upstream(request):
            seen.append(
                (request.headers["x-smg-routing-key"], json.loads(request.content))
            )
            return httpx.Response(200, json={"text": "answer"})

        async def send():
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(upstream)
            ) as client:
                service = _service(
                    settings=_settings(generate_url="http://sglang/generate"),
                    sglang_gateway=SGLangGateway(),
                    dialog_affinity=True,
                )
                body = {"input_ids": [1, 2], "routing_key": key}
                request = _DummyRequest(body=body)
                request.app.state.http_client = client
                response = await service.handle_generate(request)
                assert response.status_code == 200
                assert seen == [(header, body)]

        asyncio.run(send())

    @pytest.mark.parametrize("api", ["custom", "opensource", "v2"])
    def test_followup_keeps_first_turn_affinity_without_client_headers(self, api):
        messages = [
            {"role": "system", "content": "Use tools"},
            {"role": "user", "content": "Fix task A"},
        ]
        gateway = _DummySGLangGateway(
            generate_response=_DummyResponse(content=_generate_payload(text="answer")),
            proxy_response=_DummyResponse(
                content=_chat_completions_payload(content="answer")
            ),
        )
        keys = []
        for history in (
            messages,
            messages
            + [
                {"role": "assistant", "content": "checking"},
                {"role": "user", "content": "continue"},
            ],
        ):
            service = _service(sglang_gateway=gateway, dialog_affinity=True)
            request = _DummyRequest()
            body = SGLangChatCompletionRequest(model="37b", messages=history)
            if api == "custom":
                asyncio.run(service.handle_chat_completions_custom(request, body))
            elif api == "opensource":
                asyncio.run(service.handle_chat_completions_opensource(request, body))
            else:
                v2_body = DevicesV2ChatCompletionRequest(
                    model="37b",
                    messages=[
                        {
                            "role": item["role"],
                            "content": [{"type": "text", "text": item["content"]}],
                        }
                        for item in history
                    ],
                )
                asyncio.run(service.handle_chat_completions_v2(request, v2_body))
            keys.append(gateway.headers["x-smg-routing-key"])
        assert keys[0] == keys[1]
        if api == "opensource":
            assert "routing_key" not in gateway.proxy_calls[-1]["body"]
        else:
            assert "routing_key" not in gateway.generate_calls[-1]["payload"]

    @pytest.mark.parametrize(
        "field,value",
        [
            ("messages", [{"role": "user", "content": "Fix task B"}]),
            ("model", "another-model"),
            ("tools", [{"type": "function", "function": {"name": "new_tool"}}]),
        ],
    )
    def test_different_first_turn_or_model_context_has_another_key(self, field, value):
        service = _service(dialog_affinity=True)
        body = {"model": "37b", "messages": [{"role": "user", "content": "Fix task A"}]}
        first = service._routing_headers(_DummyRequest(), body, chat=True)
        assert first != service._routing_headers(
            _DummyRequest(), {**body, field: value}, chat=True
        )

    def test_sampling_and_dictionary_order_do_not_change_affinity(self):
        service = _service(dialog_affinity=True)
        first = service._routing_headers(
            _DummyRequest(),
            {
                "model": "37b",
                "messages": [{"role": "user", "content": "task"}],
                "temperature": 0.2,
            },
            chat=True,
        )
        second = service._routing_headers(
            _DummyRequest(),
            {
                "messages": [{"content": "task", "role": "user"}],
                "model": "37b",
                "temperature": 0.8,
                "max_tokens": 500,
            },
            chat=True,
        )
        assert first == second

    @pytest.mark.parametrize(
        "payload",
        [
            {"input_ids": [1, 2, 3]},
            {"input_ids": [1, 2, 3], "routing_key": ""},
            {"input_ids": [[1, 2], [3, 4]]},
            {"prompt": "a raw completion"},
            {"messages": [{"role": "system", "content": "system only"}]},
        ],
    )
    def test_does_not_guess_dialog_identity_from_raw_prompts(self, payload):
        service = _service(dialog_affinity=True)
        assert service._routing_headers(_DummyRequest(), payload) == {}
        assert (
            service._routing_headers(_DummyRequest(), {"messages": []}, chat=True) == {}
        )

    def test_bulk_service_does_not_add_affinity(self):
        body = {"messages": [{"role": "user", "content": "task"}]}
        assert _service()._routing_headers(_DummyRequest(), body) == {}

    def test_existing_explicit_trial_header_keeps_its_routing_key(self):
        request = _DummyRequest(
            headers={
                "x-pam-session-id": "trial-123",
                "x-pam-shooter": "terminal_bench",
            }
        )
        actual = _service(dialog_affinity=True)._routing_headers(request, {}, chat=True)
        assert actual == {"x-smg-routing-key": "42b35789b7185ec6828c9fe6b5a40f6b"}

    @pytest.mark.parametrize("api", ["generate", "completions"])
    def test_explicit_header_survives_raw_protocol_without_body_changes(self, api):
        gateway = _DummySGLangGateway(proxy_response=_DummyResponse())
        service = _service(dialog_affinity=True, sglang_gateway=gateway)
        payload = {"input_ids": [1, 2]} if api == "generate" else {"prompt": "task"}
        request = _DummyRequest(body=payload, headers={"x-smg-routing-key": "explicit"})
        handler = (
            service.handle_generate if api == "generate" else service.handle_completions
        )
        asyncio.run(handler(request))
        assert gateway.headers == {"x-smg-routing-key": "explicit"}
        assert gateway.proxy_calls[0]["body"] == payload

    @pytest.mark.parametrize(
        "headers,body,expected",
        [
            (
                {"x-smg-routing-key": "header"},
                {"routing_key": "body"},
                "230d8358dc8e8890b4c58deeb62912ee2f20357ae92a5cc861b98e68fe31acb5",
            ),
            ({"x-smg-routing-key": "header"}, {}, "header"),
        ],
    )
    def test_explicit_routing_key_takes_precedence_over_automatic_affinity(
        self, headers, body, expected
    ):
        payload = {"messages": [{"role": "user", "content": "task"}], **body}
        actual = _service(dialog_affinity=True)._routing_headers(
            _DummyRequest(headers=headers), payload, chat=True
        )
        assert actual == {"x-smg-routing-key": expected}


class TestProxyServicePriority:
    @pytest.mark.parametrize(
        "api", ["custom", "v2", "opensource", "generate", "completions"]
    )
    @pytest.mark.parametrize(
        "default_priority,request_priority,expected_priority",
        [(None, None, None), (1, None, 1), (1, 0, 0), (1, 2, 2)],
    )
    def test_default_priority_applies_only_without_explicit_request_priority(
        self, api, default_priority, request_priority, expected_priority
    ):
        gateway = _DummySGLangGateway(
            generate_response=_DummyResponse(content=_generate_payload(text="ok")),
            proxy_response=_DummyResponse(
                content=_chat_completions_payload(content="ok")
            ),
        )
        service = _service(
            settings=_settings(priority=default_priority), sglang_gateway=gateway
        )
        payload = {"input_ids": [1, 2, 3]} if api == "generate" else {"prompt": "hi"}
        if request_priority is not None:
            payload["priority"] = request_priority
        request = _DummyRequest(body=payload)
        body = SGLangChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}], priority=request_priority
        )

        if api == "custom":
            response = asyncio.run(
                service.handle_chat_completions_custom(request, body)
            )
        elif api == "v2":
            v2_body = DevicesV2ChatCompletionRequest(
                messages=[{"role": "user", "content": [{"text": "hi"}]}],
                priority=request_priority,
            )
            response = asyncio.run(service.handle_chat_completions_v2(request, v2_body))
        elif api == "opensource":
            response = asyncio.run(
                service.handle_chat_completions_opensource(request, body)
            )
        elif api == "generate":
            response = asyncio.run(service.handle_generate(request))
        else:
            response = asyncio.run(service.handle_completions(request))

        assert response.status_code == 200
        sent = (
            gateway.generate_calls[0]["payload"]
            if gateway.generate_calls
            else gateway.proxy_calls[0]["body"]
        )
        if expected_priority is None:
            assert "priority" not in sent
        else:
            assert sent["priority"] == expected_priority


class TestProxyServiceCustom:
    def test_custom_parses_function_call_and_sets_finish_reason(self):
        tokenizer_gateway = _DummyTokenizerGateway()
        sglang_gateway = _DummySGLangGateway(
            generate_response=_DummyResponse(
                status_code=200,
                content=_generate_payload(
                    text=(
                        "function call<|role_sep|>\n"
                        '{"name":"weather","arguments":{"city":"Moscow"}}'
                    )
                ),
            )
        )
        service = _service(
            tokenizer_gateway=tokenizer_gateway, sglang_gateway=sglang_gateway
        )

        request = _DummyRequest()
        body = SGLangChatCompletionRequest(
            model="custom-model",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
        )

        response = asyncio.run(service.handle_chat_completions_custom(request, body))
        data = json.loads(response.body.decode("utf-8"))

        assert data["choices"][0]["finish_reason"] == "function_call"
        assert (
            data["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
            == "weather"
        )
        assert data["usage"]["prompt_tokens"] == 2
        assert data["usage"]["completion_tokens"] == 3
        assert data["usage"]["total_tokens"] == 5

        assert tokenizer_gateway.calls[0]["messages"] == [
            {"role": "user", "content": "hi"}
        ]
        assert tokenizer_gateway.calls[0]["add_generation_prompt"] is True
        assert tokenizer_gateway.calls[0]["tokenize"] is True

    def test_v2_parses_function_call_and_builds_devices_v2_response(self):
        sglang_gateway = _DummySGLangGateway(
            generate_response=_DummyResponse(
                status_code=200,
                content=_generate_payload(
                    text=(
                        "function call<|role_sep|>\n"
                        '{"name":"weather","arguments":{"city":"Moscow"}}'
                    )
                ),
            )
        )
        service = _service(sglang_gateway=sglang_gateway)

        request = _DummyRequest(path="/v2/chat/completions")
        body = DevicesV2ChatCompletionRequest.model_validate(
            {
                "model": "custom-model",
                "messages": [{"role": "user", "content": [{"text": "hi"}]}],
                "max_tokens": 5,
            }
        )

        response = asyncio.run(service.handle_chat_completions_v2(request, body))
        data = json.loads(response.body.decode("utf-8"))

        assert data["model"] == "custom-model"
        assert data["finish_reason"] == "function_call"
        parts = data["messages"][0]["content"]
        assert parts[0] == {"text": ""}
        assert parts[1]["function_call"]["name"] == "weather"
        assert parts[1]["function_call"]["arguments"] == {"city": "Moscow"}
        assert data["usage"]["input_tokens"] == 2
        assert data["usage"]["output_tokens"] == 3
        assert data["usage"]["total_tokens"] == 5

    def test_custom_logs_sglang_error_details_in_debug_mode(self, caplog):
        service = _service(
            settings=_settings(logging_level="debug"),
            sglang_gateway=_DummySGLangGateway(
                generate_response=_DummyResponse(
                    status_code=400,
                    content=b'{"error":"bad request"}',
                    text='{"error":"bad request"}',
                )
            ),
        )
        request = _DummyRequest()
        body = SGLangChatCompletionRequest(
            model="custom-model",
            messages=[{"role": "user", "content": "hi"}],
        )

        with caplog.at_level(logging.ERROR, logger="obm_proxy"):
            with pytest.raises(SGLangNonRetryableStatusError) as exc:
                asyncio.run(service.handle_chat_completions_custom(request, body))

        assert exc.value.status_code == 400
        record = next(
            record
            for record in caplog.records
            if "[obm_proxy][sglang_error]" in record.message
        )
        payload = json.loads(record.message[record.message.index("{") :])
        assert payload["route"]["path"] == "/v1/chat/completions"
        assert payload["incoming_request"]["model"] == "custom-model"
        assert payload["incoming_request"]["messages"] == [
            {"role": "user", "content": "hi"}
        ]
        assert payload["sglang_request"]["url"] == service._settings.generate_url
        assert payload["sglang_request"]["body"]["input_ids"] == [1, 2, 3]
        assert payload["sglang_response"]["status_code"] == 400
        assert payload["sglang_response"]["body"] == '{"error":"bad request"}'

    def test_custom_logs_full_exchange_and_response_in_debug_mode(self, caplog):
        service = _service(
            sglang_gateway=_DummySGLangGateway(
                generate_response=_DummyResponse(
                    content=_generate_payload(text="hello"),
                )
            ),
        )
        request = _DummyRequest()
        body = SGLangChatCompletionRequest(
            model="custom-model",
            messages=[{"role": "user", "content": "hi"}],
        )

        with caplog.at_level(logging.DEBUG, logger="obm_proxy"):
            asyncio.run(service.handle_chat_completions_custom(request, body))

        exchange = next(
            record
            for record in caplog.records
            if "[obm_proxy][exchange]" in record.message
        )
        payload = json.loads(exchange.message[exchange.message.index("{") :])
        assert payload["incoming_request"]["model"] == "custom-model"
        assert payload["sglang_request"]["body"]["input_ids"] == [1, 2, 3]
        assert payload["sglang_response"]["text"] == "hello"
        assert (
            payload["outgoing_response"]["choices"][0]["message"]["content"] == "hello"
        )

        caplog.clear()
        with caplog.at_level(logging.INFO, logger="obm_proxy"):
            asyncio.run(service.handle_chat_completions_custom(request, body))
        assert "[obm_proxy][exchange]" not in caplog.text

    def test_custom_raises_non_retryable_status_error_for_4xx(self):
        service = _service(
            sglang_gateway=_DummySGLangGateway(
                generate_response=_DummyResponse(
                    status_code=400,
                    content=b'{"error":"bad request"}',
                    text='{"error":"bad request"}',
                )
            )
        )
        request = _DummyRequest()
        body = SGLangChatCompletionRequest(
            model="custom-model",
            messages=[{"role": "user", "content": "hi"}],
        )

        with pytest.raises(SGLangNonRetryableStatusError) as exc:
            asyncio.run(service.handle_chat_completions_custom(request, body))
        assert exc.value.status_code == 400

    def test_logs_sglang_error_without_incoming_body(self, caplog):
        service = _service()

        with caplog.at_level(logging.ERROR, logger="obm_proxy"):
            service._log_sglang_error(
                request=_DummyRequest(),
                incoming_body=None,
                sglang_url=service._settings.generate_url,
                sglang_payload={"input_ids": [1, 2, 3]},
                response=_DummyResponse(
                    status_code=400,
                    content=b'{"error":"bad request"}',
                ),
            )

        record = next(
            record
            for record in caplog.records
            if "[obm_proxy][sglang_error]" in record.message
        )
        payload = json.loads(record.message[record.message.index("{") :])
        assert payload["incoming_request"] is None
        assert payload["sglang_request"]["body"]["input_ids"] == [1, 2, 3]


class TestProxyServiceOpenSourceAndProxying:
    def test_opensource_merges_chat_template_kwargs_and_applies_reasoning_budget(self):
        sglang_gateway = _DummySGLangGateway(
            proxy_response=_DummyResponse(
                status_code=200,
                content=_chat_completions_payload(content=None, reasoning_content=None),
                headers={},
            )
        )
        service = _service(sglang_gateway=sglang_gateway)

        request = _DummyRequest(path="/v1/chat/completions")
        body = SGLangChatCompletionRequest(
            model="opensource-model",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
            chat_template_kwargs={"reasoning": True},
        )

        response = asyncio.run(
            service.handle_chat_completions_opensource(request, body)
        )
        assert response.status_code == 200

        sent = sglang_gateway.proxy_calls[0]["body"]
        assert sent["chat_template_kwargs"] == {
            "use_devsystem": True,
            "reasoning": True,
        }
        assert sent["max_completion_tokens"] == 12
        assert "max_tokens" not in sent

    def test_opensource_passes_legacy_function_message_through(self):
        sglang_gateway = _DummySGLangGateway(
            proxy_response=_DummyResponse(
                status_code=200,
                content=_chat_completions_payload(content=None, reasoning_content=None),
                headers={},
            )
        )
        service = _service(sglang_gateway=sglang_gateway)

        request = _DummyRequest(path="/v1/chat/completions")
        body = SGLangChatCompletionRequest(
            model="opensource-model",
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "function", "name": "legacy_fn", "content": "{}"},
            ],
        )

        response = asyncio.run(
            service.handle_chat_completions_opensource(request, body)
        )
        assert response.status_code == 200

        sent_messages = sglang_gateway.proxy_calls[0]["body"]["messages"]
        assert sent_messages[1]["role"] == "function"
        assert sent_messages[1]["name"] == "legacy_fn"
        assert sent_messages[1]["content"] == "{}"

    def test_opensource_maps_sglang_503_to_429(self, caplog):
        sglang_gateway = _DummySGLangGateway(
            proxy_response=_DummyResponse(
                status_code=503,
                content=b'{"error":"busy"}',
                headers={"content-type": "application/problem+json"},
            )
        )
        service = _service(
            settings=_settings(logging_level="debug"),
            sglang_gateway=sglang_gateway,
        )

        request = _DummyRequest(path="/v1/chat/completions")
        body = SGLangChatCompletionRequest(
            model="opensource-model",
            messages=[{"role": "user", "content": "hi"}],
        )

        with caplog.at_level(logging.ERROR, logger="obm_proxy"):
            response = asyncio.run(
                service.handle_chat_completions_opensource(request, body)
            )

        assert response.status_code == 429
        assert response.body == b'{"error":"busy"}'
        assert response.headers["content-type"] == "application/problem+json"
        assert "[obm_proxy][sglang_error]" not in caplog.text

    def test_handle_generate_returns_status_and_content_only(self):
        sglang_gateway = _DummySGLangGateway(
            proxy_response=_DummyResponse(
                status_code=201,
                content=b'{"ok":1}',
                headers={
                    "content-type": "application/json",
                    "x-request-id": "abc-123",
                    "content-length": "999",
                    "content-encoding": "gzip",
                },
            )
        )
        service = _service(sglang_gateway=sglang_gateway)
        request = _DummyRequest(body={"p": 1}, path="/generate")

        response = asyncio.run(service.handle_generate(request))

        assert response.status_code == 201
        assert response.body == b'{"ok":1}'
        assert response.headers["content-type"] == "application/json"
        assert "x-request-id" not in response.headers

    def test_handle_generate_relabels_octet_stream_as_json(self):
        sglang_gateway = _DummySGLangGateway(
            proxy_response=_DummyResponse(
                content=b'{"text":"","meta_info":{}}',
                headers={"content-type": "application/octet-stream"},
            )
        )
        service = _service(sglang_gateway=sglang_gateway)
        request = _DummyRequest(body={"return_logprob": True}, path="/generate")

        response = asyncio.run(service.handle_generate(request))

        assert response.status_code == 200
        assert response.body == b'{"text":"","meta_info":{}}'
        assert response.headers["content-type"] == "application/json"

    def test_handle_completions_proxies_to_expected_path(self):
        sglang_gateway = _DummySGLangGateway(
            proxy_response=_DummyResponse(status_code=202, content=b'{"ok":2}')
        )
        service = _service(sglang_gateway=sglang_gateway)
        request = _DummyRequest(body={"prompt": "x"}, path="/v1/completions")

        response = asyncio.run(service.handle_completions(request))

        assert response.status_code == 202
        assert response.body == b'{"ok":2}'
        sent = sglang_gateway.proxy_calls[0]
        assert sent["url"] == service._settings.completions_url
        assert sent["body"] == {"prompt": "x"}


class TestProxyServiceEdgeCases:
    def test_preload_calls_tokenizer_get(self):
        tokenizer_gateway = _DummyTokenizerGateway()
        service = _service(tokenizer_gateway=tokenizer_gateway)
        service.preload()

    def test_preload_noops_for_non_custom_handlers(self):
        tokenizer_gateway = _DummyTokenizerGateway()
        service = _service(
            settings=_settings(chat_handlers_mode="opensource"),
            tokenizer_gateway=tokenizer_gateway,
        )

        service.preload()

        assert tokenizer_gateway.calls == []

    def test_build_input_ids_raises_http_500_when_template_application_fails(self):
        service = _service(tokenizer_gateway=_FailingTokenizerGateway())
        body = SGLangChatCompletionRequest(
            model="custom-model",
            messages=[{"role": "user", "content": "hi"}],
        )

        with pytest.raises(Exception) as exc:
            service._build_input_ids(body=body)
        assert getattr(exc.value, "status_code", None) == 500
        assert "Failed to apply chat template" in str(exc.value.detail)

    def test_build_input_ids_normalizes_legacy_function_message_to_tool(self):
        tokenizer_gateway = _DummyTokenizerGateway()
        service = _service(tokenizer_gateway=tokenizer_gateway)
        body = SGLangChatCompletionRequest(
            model="custom-model",
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "function", "name": "legacy_fn", "content": "{}"},
            ],
        )

        service._build_input_ids(body=body)

        sent_messages = tokenizer_gateway.calls[0]["messages"]
        assert sent_messages[1]["role"] == "tool"
        assert sent_messages[1]["tool_call_id"] == "legacy::legacy_fn"
        assert sent_messages[1]["content"] == "{}"

    def test_build_input_ids_requests_return_dict_false(self):
        # transformers 5.x defaults return_dict=True (BatchEncoding, not JSON
        # serializable); return_dict=False keeps a plain input_ids list for the
        # sglang payload. Assert we pin it so the 502 regression can't return.
        tokenizer_gateway = _DummyTokenizerGateway()
        service = _service(tokenizer_gateway=tokenizer_gateway)
        body = SGLangChatCompletionRequest(
            model="custom-model",
            messages=[{"role": "user", "content": "hi"}],
        )

        result = service._build_input_ids(body=body)

        assert result == [1, 2, 3]
        assert tokenizer_gateway.calls[0]["kwargs"]["return_dict"] is False

    def test_request_sglang_wraps_transport_error_as_retryable_request(self):
        class _RaisingGateway(_DummySGLangGateway):
            async def generate(self, *, client, url, payload):
                raise RuntimeError("connection reset")

        service = _service(sglang_gateway=_RaisingGateway())
        params = service._build_sampling_params(
            body=SGLangChatCompletionRequest(
                model="custom-model",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=10,
            )
        )
        service._request_sglang.retry.stop = stop_after_attempt(1)
        service._request_sglang.retry.wait = wait_fixed(0)

        with pytest.raises(SGLangRetryableRequestError):
            asyncio.run(
                service._request_sglang(
                    request=_DummyRequest(),
                    incoming_body=SGLangChatCompletionRequest(
                        messages=[{"role": "user", "content": "hi"}]
                    ),
                    input_ids=[1, 2, 3],
                    sampling_params=params,
                )
            )

    @pytest.mark.parametrize("status_code", [429, 500])
    def test_request_sglang_raises_retryable_status_for_429_and_5xx(
        self, status_code: int
    ):
        service = _service(
            sglang_gateway=_DummySGLangGateway(
                generate_response=_DummyResponse(
                    status_code=status_code, content=b"{}", text="{}"
                )
            )
        )
        params = service._build_sampling_params(
            body=SGLangChatCompletionRequest(
                model="custom-model",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=10,
            )
        )
        service._request_sglang.retry.stop = stop_after_attempt(1)
        service._request_sglang.retry.wait = wait_fixed(0)

        with pytest.raises(SGLangRetryableStatusError):
            asyncio.run(
                service._request_sglang(
                    request=_DummyRequest(),
                    incoming_body=SGLangChatCompletionRequest(
                        messages=[{"role": "user", "content": "hi"}]
                    ),
                    input_ids=[1, 2, 3],
                    sampling_params=params,
                )
            )

    @pytest.mark.parametrize("media_field", ["image_data", "audio_data"])
    def test_request_sglang_includes_media_data_in_payload_when_provided(
        self, media_field
    ):
        payload = {
            "text": "ok",
            "meta_info": {
                "id": "req-1",
                "finish_reason": {"type": "stop"},
            },
        }
        gateway = _DummySGLangGateway(
            generate_response=_DummyResponse(
                status_code=200, content=json.dumps(payload).encode("utf-8")
            )
        )
        service = _service(sglang_gateway=gateway)
        params = service._build_sampling_params(
            body=SGLangChatCompletionRequest(
                model="custom-model",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=10,
            )
        )
        service._request_sglang.retry.stop = stop_after_attempt(1)
        service._request_sglang.retry.wait = wait_fixed(0)

        media_data = {"b64": "abc"}
        result = asyncio.run(
            service._request_sglang(
                request=_DummyRequest(),
                incoming_body=SGLangChatCompletionRequest(
                    messages=[{"role": "user", "content": "hi"}]
                ),
                input_ids=[1, 2, 3],
                **{media_field: media_data},
                sampling_params=params,
            )
        )

        assert result.response.text == "ok"
        assert result.payload[media_field] == media_data
        assert len(gateway.generate_calls) == 1
        sent = gateway.generate_calls[0]["payload"]
        assert sent[media_field] == media_data

    def test_build_sampling_params_copies_optional_sampling_fields(self):
        service = _service()
        body = SGLangChatCompletionRequest(
            model="custom-model",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=11,
            temperature=0.3,
            repetition_penalty=1.1,
            top_p=0.9,
            top_k=20,
            stop=["</s>"],
        )
        sampling = service._build_sampling_params(body=body)
        assert sampling.max_new_tokens == 11
        assert sampling.temperature == 0.3
        assert sampling.repetition_penalty == 1.1
        assert sampling.top_p == 0.9
        assert sampling.top_k == 20
        assert sampling.stop == ["</s>"]

    def test_resolve_base_max_tokens_uses_max_completion_tokens_when_max_tokens_absent(
        self,
    ):
        service = _service()
        body = SGLangChatCompletionRequest(
            model="custom-model",
            messages=[{"role": "user", "content": "hi"}],
            max_completion_tokens=77,
        )
        assert service._resolve_base_max_tokens(body) == 77

    def test_inject_max_tokens_for_reasoning_uses_base_when_reasoning_keys_not_set(
        self,
    ):
        service = _service()
        body = SGLangChatCompletionRequest(
            model="opensource-model",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=33,
            chat_template_kwargs={"some_other_flag": True},
        )
        updated = service._inject_max_tokens_for_reasoning(body)
        assert updated.max_completion_tokens == 33
        assert updated.max_tokens is None

    def test_build_sampling_params_does_not_add_reasoning_budget_when_inactive(self):
        service = _service()
        body = SGLangChatCompletionRequest(
            model="custom-model",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=11,
        )
        sampling = service._build_sampling_params(body=body)
        assert sampling.max_new_tokens == 11

    def test_inject_max_tokens_for_reasoning_effort_activates_reasoning_budget(self):
        service = _service()
        body = SGLangChatCompletionRequest(
            model="opensource-model",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
            chat_template_kwargs={"reasoning_effort": "low"},
        )
        updated = service._inject_max_tokens_for_reasoning(body)
        assert updated.max_completion_tokens == 17
        assert updated.max_tokens is None


class TestLoggableResponse:
    def test_httpx_response_json_is_used(self):
        response = httpx.Response(200, json={"a": 1})
        assert ProxyService._loggable_response(response) == {"a": 1}

    def test_httpx_response_invalid_json_falls_back_to_text(self):
        response = httpx.Response(200, content=b"plain text")
        assert ProxyService._loggable_response(response) == "plain text"

    def test_starlette_response_body_parsed_as_json(self):
        assert ProxyService._loggable_response(Response(content=b'{"b": 2}')) == {
            "b": 2
        }

    def test_starlette_response_invalid_json_decoded_as_text(self):
        assert (
            ProxyService._loggable_response(Response(content=b"plain text"))
            == "plain text"
        )


class TestMaybeParseJsonArgs:
    def test_returns_non_string_unchanged(self):
        args = {"city": "Moscow"}
        assert _maybe_parse_json_args(args) is args

    def test_parses_json_object_string_to_dict(self):
        assert _maybe_parse_json_args('{"city": "Moscow"}') == {"city": "Moscow"}

    def test_keeps_raw_string_when_not_valid_json(self):
        assert _maybe_parse_json_args("{not json") == "{not json"

    def test_keeps_raw_string_when_json_is_not_an_object(self):
        # ``123`` is valid JSON but not a dict, so the legacy string is kept
        # rather than coercing it into a scalar.
        assert _maybe_parse_json_args("123") == "123"


class TestNormalizeAssistantMessageForTemplate:
    def test_parses_json_string_tool_call_and_function_call_arguments(self):
        # devices-v1 shooter compatibility: legacy clients send tool-call and
        # function_call arguments as JSON *strings*; the template expects dicts.
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "calc", "arguments": '{"x": 1}'},
                }
            ],
            "function_call": {"name": "calc", "arguments": '{"x": 1}'},
        }

        normalized = ProxyService._normalize_chat_message_for_template(message)

        assert normalized["tool_calls"][0]["function"]["arguments"] == {"x": 1}
        assert normalized["function_call"]["arguments"] == {"x": 1}

    def test_leaves_non_dict_tool_calls_and_argless_functions_untouched(self):
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                "not-a-dict",
                {"id": "call_2", "type": "function", "function": {"name": "ping"}},
            ],
        }

        normalized = ProxyService._normalize_chat_message_for_template(message)

        assert normalized["tool_calls"][0] == "not-a-dict"
        assert normalized["tool_calls"][1]["function"] == {"name": "ping"}

    def test_assistant_message_without_tool_or_function_calls_unchanged(self):
        message = {"role": "assistant", "content": "plain answer"}
        assert ProxyService._normalize_chat_message_for_template(message) == message


class TestSamplingOverrides:
    @pytest.mark.parametrize(
        "params",
        [
            {},
            {"sampling_params": None},
            {"sampling_params": {"temperature": 0.9, "max_new_tokens": 12}},
            {
                "sampling_params": [
                    {"temperature": 0.9, "max_new_tokens": 12},
                    {"temperature": 0.8, "max_new_tokens": 24},
                ]
            },
        ],
    )
    @pytest.mark.parametrize("temperature", [0.3, 0.0, None])
    def test_generate_overrides_each_sample_without_mutating_input(
        self, params, temperature
    ):
        overrides = {"temperature": temperature, "ignore_eos": False, "stop": []}
        body = {"text": "hi", **params}
        original = json.loads(json.dumps(body))
        gateway = _DummySGLangGateway(proxy_response=_DummyResponse())
        service = _service(
            settings=_settings(sampling_params=overrides), sglang_gateway=gateway
        )

        response = asyncio.run(service.handle_generate(_DummyRequest(body=body)))

        samples = params.get("sampling_params") or {}
        expected = (
            [{**sample, **overrides} for sample in samples]
            if isinstance(samples, list)
            else {**samples, **overrides}
        )
        assert response.status_code == 200
        assert gateway.proxy_calls[0]["body"] == {
            "text": "hi",
            "sampling_params": expected,
        }
        assert body == original
        assert service._settings.sampling_params == overrides

    def test_completions_overrides_top_level_and_preserves_other_fields(self):
        body = {"prompt": "hi", "temperature": 0.9, "max_tokens": 12}
        gateway = _DummySGLangGateway(proxy_response=_DummyResponse())
        service = _service(
            settings=_settings(sampling_params={"temperature": 0.3, "top_k": 40}),
            sglang_gateway=gateway,
        )

        response = asyncio.run(service.handle_completions(_DummyRequest(body=body)))

        assert response.status_code == 200
        assert gateway.proxy_calls[0]["body"] == {
            "prompt": "hi",
            "temperature": 0.3,
            "max_tokens": 12,
            "top_k": 40,
        }
        assert body == {"prompt": "hi", "temperature": 0.9, "max_tokens": 12}
