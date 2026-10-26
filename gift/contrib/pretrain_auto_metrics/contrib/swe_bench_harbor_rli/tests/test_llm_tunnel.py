from __future__ import annotations

import gzip
import importlib
import json
import shutil
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from swe_bench_harbor_rli.src.errors import (
    EndpointUnavailableError,
    RLIExecProxyConnectionLostError,
)


def _load_tunnel_module():
    harbor_module = types.ModuleType("harbor")
    environments_module = types.ModuleType("harbor.environments")
    base_module = types.ModuleType("harbor.environments.base")
    base_module.BaseEnvironment = object

    sys.modules["harbor"] = harbor_module
    sys.modules["harbor.environments"] = environments_module
    sys.modules["harbor.environments.base"] = base_module
    sys.modules.pop(
        "swe_bench_harbor_rli.agents.llm_tunnel.tunnel",
        None,
    )
    return importlib.import_module("swe_bench_harbor_rli.agents.llm_tunnel.tunnel")


def _result(
    *,
    return_code: int = 0,
    stdout: str = "",
    stderr: str = "",
):
    return type(
        "ExecResult",
        (),
        {
            "return_code": return_code,
            "stdout": stdout,
            "stderr": stderr,
        },
    )()


@pytest.mark.asyncio
async def test_tunnel_forwards_path_auth_and_served_model(tmp_path: Path) -> None:
    tunnel_module = _load_tunnel_module()
    captured: dict[str, object] = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": []})

    tunnel = tunnel_module.RLILLMTunnel(
        object(),
        bridge_path=tmp_path / "bridge.py",
        llm_base_url="http://model-server:8000/v1",
        api_key="host-secret",
        served_model="served-model",
    )
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
            status, body = await tunnel._call_upstream(
                client,
                {
                    "method": "POST",
                    "path": "/v1/chat/completions",
                    "body": {"model": "agent-model", "messages": []},
                },
            )
    finally:
        tunnel._cache_dir.cleanup()

    assert status == 200
    assert body == {"choices": []}
    assert captured == {
        "url": "http://model-server:8000/v1/chat/completions",
        "authorization": "Bearer host-secret",
        "body": {"model": "served-model", "messages": []},
    }


@pytest.mark.asyncio
async def test_tunnel_reuses_response_after_delivery_failure(tmp_path: Path) -> None:
    tunnel_module = _load_tunnel_module()
    request_id = "request-1"
    request_template = tmp_path / "request.req.json.gz"
    with gzip.open(request_template, "wt", encoding="utf-8") as stream:
        json.dump(
            {
                "id": request_id,
                "method": "POST",
                "path": "/v1/chat/completions",
                "body": {"model": "agent-model", "messages": []},
            },
            stream,
        )

    class FakeEnvironment:
        async def download_file(self, source_path, target_path):
            _ = source_path
            shutil.copyfile(request_template, target_path)

    tunnel = tunnel_module.RLILLMTunnel(
        FakeEnvironment(),
        bridge_path=tmp_path / "bridge.py",
        llm_base_url="http://model-server:8000/v1",
        api_key=None,
    )
    tunnel._call_upstream = AsyncMock(return_value=(200, {"choices": []}))
    tunnel._deliver_response = AsyncMock(
        side_effect=[RuntimeError("commit failed"), None]
    )
    request_path = f"/opt/llmtun/outbox/{request_id}.req.json.gz"

    try:
        with pytest.raises(RuntimeError, match="commit failed"):
            await tunnel._process_request(object(), request_path)
        await tunnel._process_request(object(), request_path)
    finally:
        tunnel._cache_dir.cleanup()

    tunnel._call_upstream.assert_awaited_once()
    assert tunnel._deliver_response.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 500])
async def test_tunnel_preserves_non_availability_errors(
    tmp_path: Path,
    status: int,
) -> None:
    tunnel_module = _load_tunnel_module()

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            request=request,
            json={"error": {"message": "upstream error"}},
        )

    tunnel = tunnel_module.RLILLMTunnel(
        object(),
        bridge_path=tmp_path / "bridge.py",
        llm_base_url="http://model-server:8000/v1",
        api_key=None,
    )
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
            actual_status, body = await tunnel._call_upstream(
                client,
                {
                    "method": "POST",
                    "path": "/v1/chat/completions",
                    "body": {"model": "agent-model", "messages": []},
                },
            )
    finally:
        tunnel._cache_dir.cleanup()

    assert actual_status == status
    assert body == {"error": {"message": "upstream error"}}


@pytest.mark.asyncio
async def test_tunnel_raises_after_consecutive_unavailable_responses(
    tmp_path: Path,
) -> None:
    tunnel_module = _load_tunnel_module()

    tunnel = tunnel_module.RLILLMTunnel(
        object(),
        bridge_path=tmp_path / "bridge.py",
        llm_base_url="http://model-server:8000/v1",
        api_key=None,
        unavailable_threshold=2,
    )
    tunnel._request_upstream = AsyncMock(
        return_value=httpx.Response(503, json={"error": {}})
    )
    envelope = {
        "method": "POST",
        "path": "/v1/chat/completions",
        "body": {"model": "agent-model", "messages": []},
    }
    try:
        status, _ = await tunnel._call_upstream(object(), envelope)
        with pytest.raises(EndpointUnavailableError):
            await tunnel._call_upstream(object(), envelope)
    finally:
        tunnel._cache_dir.cleanup()

    assert status == 503
    assert tunnel._request_upstream.await_count == 2


@pytest.mark.asyncio
async def test_tunnel_maps_upstream_timeout_to_504(tmp_path: Path) -> None:
    tunnel_module = _load_tunnel_module()

    tunnel = tunnel_module.RLILLMTunnel(
        object(),
        bridge_path=tmp_path / "bridge.py",
        llm_base_url="http://model-server:8000/v1",
        api_key=None,
    )
    tunnel._request_upstream = AsyncMock(
        side_effect=httpx.ReadTimeout("timed out")
    )
    try:
        status, body = await tunnel._call_upstream(
            object(),
            {
                "method": "POST",
                "path": "/v1/chat/completions",
                "body": {"model": "agent-model", "messages": []},
            },
        )
    finally:
        tunnel._cache_dir.cleanup()

    assert status == 504
    assert body["error"]["code"] == "model_request_timeout"
    tunnel._request_upstream.assert_awaited_once()


@pytest.mark.parametrize(
    "result",
    [
        _result(return_code=-1),
        _result(stderr="exec-proxy connection lost"),
    ],
)
def test_tunnel_classifies_rli_connection_loss(result) -> None:
    tunnel_module = _load_tunnel_module()

    with pytest.raises(RLIExecProxyConnectionLostError):
        tunnel_module.RLILLMTunnel._raise_if_connection_lost(
            result,
            "polling",
        )


@pytest.mark.asyncio
async def test_tunnel_poll_ignores_exec_proxy_prefix(tmp_path: Path) -> None:
    tunnel_module = _load_tunnel_module()
    tunnel = tunnel_module.RLILLMTunnel(
        object(),
        bridge_path=tmp_path / "bridge.py",
        llm_base_url="http://model-server:8000/v1",
        api_key=None,
    )
    tunnel._exec_idempotent = AsyncMock(
        return_value=_result(
            stdout=(
                "\"; printf '%s' '<base64>'\n"
                "__EXEC_PROXY_START_deadbeef__\n"
                "PID:alive\n"
                "EXIT:\n"
            )
        )
    )
    try:
        state = await tunnel._poll()
    finally:
        tunnel._cache_dir.cleanup()

    assert state.pid_state == "alive"
    assert state.exit_code is None
    assert state.request_path is None
