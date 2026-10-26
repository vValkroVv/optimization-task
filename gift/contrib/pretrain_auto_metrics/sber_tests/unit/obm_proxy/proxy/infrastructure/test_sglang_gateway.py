import asyncio

import pytest

from node_runner.src.pam_proxies.models import SGLangChatCompletionRequest
from node_runner.src.obm_proxy.proxy.infrastructure.sglang_gateway import SGLangGateway

pytestmark = pytest.mark.unit


class _DummyClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def post(self, url: str, json: dict, headers=None):
        self.calls.append({"url": url, "json": json})
        self.headers = headers
        return self.response


class TestSGLangGateway:
    @pytest.mark.parametrize("api", ["generate", "proxy_to_sglang"])
    def test_affinity_header_reaches_upstream_without_changing_json(self, api):
        client = _DummyClient(object())
        gateway = SGLangGateway()
        headers = {"x-smg-routing-key": "dialog"}
        payload = {"input_ids": [1, 2, 3]}
        if api == "generate":
            asyncio.run(
                gateway.generate(
                    client=client, url="/generate", payload=payload, headers=headers
                )
            )
        else:
            asyncio.run(
                gateway.proxy_to_sglang(
                    client=client, url="/generate", body=payload, headers=headers
                )
            )
        assert client.headers == headers
        assert client.calls[0]["json"] == payload

    def test_generate_posts_to_generate_endpoint(self):
        response = object()
        client = _DummyClient(response)
        gateway = SGLangGateway()

        out = asyncio.run(
            gateway.generate(
                client=client,
                url="http://localhost:31001/generate",
                payload={"input_ids": [1, 2, 3]},
            )
        )

        assert out is response
        assert client.calls == [
            {
                "url": "http://localhost:31001/generate",
                "json": {"input_ids": [1, 2, 3]},
            }
        ]

    def test_proxy_to_sglang_accepts_dict_body_without_conversion(self):
        response = object()
        client = _DummyClient(response)
        gateway = SGLangGateway()
        body = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}

        out = asyncio.run(
            gateway.proxy_to_sglang(
                client=client,
                url="http://localhost:30000/v1/chat/completions",
                body=body,
            )
        )

        assert out is response
        assert client.calls[0]["json"] == body
        assert client.calls[0]["url"] == "http://localhost:30000/v1/chat/completions"

    def test_proxy_to_sglang_serializes_pydantic_request_excluding_none(self):
        response = object()
        client = _DummyClient(response)
        gateway = SGLangGateway()
        body = SGLangChatCompletionRequest(
            model="sglang-model",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=12,
        )

        asyncio.run(
            gateway.proxy_to_sglang(
                client=client,
                url="http://localhost:30000/v1/chat/completions",
                body=body,
            )
        )

        sent = client.calls[0]["json"]
        assert sent["model"] == "sglang-model"
        assert sent["messages"][0]["content"] == "hello"
        assert sent["max_tokens"] == 12
        assert "audio" not in sent
