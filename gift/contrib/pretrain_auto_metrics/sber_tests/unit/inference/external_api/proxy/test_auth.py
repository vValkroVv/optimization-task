import asyncio

import httpx
import pytest

from node_runner.src.config_model import ExternalApiInferenceConfig
from node_runner.src.inference.external_api.proxy.auth import (
    IftTokenProvider,
    IftTokenRefreshError,
)

pytestmark = pytest.mark.unit


def test_ift_token_provider_requests_and_reuses_token():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"tok": f"token-{len(requests)}"})

    async def run() -> tuple[str, str]:
        cfg = ExternalApiInferenceConfig.model_validate(
            {
                "formatting": "ift",
                "base_url": "https://gigachat.ift.sberdevices.ru/v1",
                "model": "model",
                "user_pass": "USER:PASS",
            }
        )
        assert cfg.user_pass is not None
        provider = IftTokenProvider(
            cfg.user_pass,
            cfg.base_url,
        )
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        ) as client:
            return await provider.get_token(client), await provider.get_token(client)

    first, second = asyncio.run(run())

    assert first == "token-1"
    assert second == "token-1"
    assert len(requests) == 1
    assert str(requests[0].url) == "https://gigachat.ift.sberdevices.ru/v1/token"
    assert requests[0].headers["Authorization"] == "Basic VVNFUjpQQVNT"
    assert requests[0].headers["Content-Type"] == "application/json"


def test_ift_token_provider_refreshes_stale_token():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"tok": f"token-{len(requests)}"})

    async def run() -> tuple[str, str]:
        provider = IftTokenProvider(
            "USER:PASS",
            "https://gigachat.ift.sberdevices.ru",
        )
        provider.refresh_interval_sec = 0.0
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        ) as client:
            return await provider.get_token(client), await provider.get_token(client)

    first, second = asyncio.run(run())

    assert first == "token-1"
    assert second == "token-2"
    assert len(requests) == 2


def test_ift_token_provider_reuses_token_created_while_waiting_for_lock():
    async def run() -> str:
        provider = IftTokenProvider("USER:PASS", "https://gigachat.ift.sberdevices.ru")

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async with provider._lock:
                task = asyncio.create_task(provider.get_token(client))
                await asyncio.sleep(0)
                provider._token = "token-from-other-task"
                provider._refreshed_at = 10**12
            return await task

    assert asyncio.run(run()) == "token-from-other-task"


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (httpx.Response(500, json={"tok": "x"}), "status 500"),
        (httpx.Response(200, content=b"not-json"), "not valid JSON"),
        (httpx.Response(200, json=["token"]), "must be a JSON object"),
        (httpx.Response(200, json={"tok": ""}), "does not contain tok"),
        (httpx.Response(200, json={"tok": 123}), "does not contain tok"),
    ],
)
def test_ift_token_provider_rejects_bad_token_responses(response, match):
    async def run() -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return response

        provider = IftTokenProvider("USER:PASS", "https://gigachat.ift.sberdevices.ru")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(IftTokenRefreshError, match=match):
                await IftTokenProvider._request_token.__wrapped__(provider, client)

    asyncio.run(run())
