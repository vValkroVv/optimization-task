from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from uuid import UUID
from unittest.mock import AsyncMock

import httpx
import pytest

from gigarli.client import RLIClient, RLIClientError
from gigarli.client_session import RLIClientSession
from gigarli.schemas.common import ErrorResponse
from gigarli.schemas.env import (
    CloseSessionRequest,
    CreateSessionRequest,
    CreateSessionResult,
    GetSessionResult,
    SessionEnv,
    StepAction,
    StepActionArgs,
    StepRequest,
    StepResult,
)


def _make_client() -> RLIClient:
    return RLIClient(base_url="https://rli.sberdevices.ru")


def test_uses_rli_api_base_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RLI_API_BASE", "https://gym.static.space")

    assert RLIClient().base_url == "https://gym.static.space"


def test_ignores_slurm_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RLI_API_BASE", raising=False)
    monkeypatch.setenv("SLURMRESTD_URLS", "http://slurm.example")
    monkeypatch.setenv("SLURM_JOB_ID", "123")

    assert RLIClient().base_url == "https://rli.sberdevices.ru"


@pytest.mark.asyncio
async def test_create_session_posts_expected_payload_and_returns_result_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client()
    send = AsyncMock(
        return_value={
            "status": "ok",
            "result": {
                "session_id": "sess-123",
                "created_at": "2026-05-30T12:17:21.317239741Z",
                "backend_ref": "https://example.invalid/backend/sess-123",
                "backend_type": "SWE",
                "shell_mode": "current",
                "tool_schemas": [],
                "terminal_url": "https://example.invalid/backend/sess-123/terminal",
                "workspace_path": "/",
            },
            "meta": {
                "version": "rli-api-gateway@2.1.1",
                "request_id": "req-123",
                "runtime_ms": 12,
                "schema_version": "1.0",
            },
        }
    )
    monkeypatch.setattr(client, "_send", send)

    result = await client.create_session(
        CreateSessionRequest(
            env=SessionEnv(
                name="terminal-bench-modernize-scientific-stack",
                version="1.0.0",
                task_id="modernize-scientific-stack",
            ),
            ttl_ms=3_600_000,
            use_pip_proxy=True,
        )
    )

    assert isinstance(result, CreateSessionResult)
    assert result.session_id == "sess-123"
    assert "session_id" not in send.await_args.kwargs["json"]
    assert UUID(send.await_args.kwargs["headers"]["Idempotency-Key"])
    send.assert_awaited_once_with(
        "POST",
        "/v1/sessions",
        json={
            "env": {
                "name": "terminal-bench-modernize-scientific-stack",
                "version": "1.0.0",
                "task_id": "modernize-scientific-stack",
            },
            "ttl_ms": 3_600_000,
            "use_pip_proxy": True,
            "allow_internet_access": None,
            "tenantId": "bench",
            "attribution": None,
            "project_id": None,
        },
        headers={"Idempotency-Key": send.await_args.kwargs["headers"]["Idempotency-Key"]},
    )


@pytest.mark.asyncio
async def test_create_session_uses_provided_idempotency_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client()
    send = AsyncMock(
        return_value={
            "status": "ok",
            "result": {
                "session_id": "sess-123",
                "created_at": "2026-05-30T12:17:21.317239741Z",
                "backend_ref": "https://example.invalid/backend/sess-123",
                "backend_type": "SWE",
                "shell_mode": "current",
                "tool_schemas": [],
                "terminal_url": "https://example.invalid/backend/sess-123/terminal",
                "workspace_path": "/",
            },
            "meta": {
                "version": "rli-api-gateway@2.1.1",
                "request_id": "req-123",
                "runtime_ms": 12,
                "schema_version": "1.0",
            },
        }
    )
    monkeypatch.setattr(client, "_send", send)

    await client.create_session(
        CreateSessionRequest(
            env=SessionEnv(
                name="terminal-bench-modernize-scientific-stack",
                version="1.0.0",
                task_id="modernize-scientific-stack",
            ),
            ttl_ms=3_600_000,
            use_pip_proxy=True,
        ),
        idempotency_key="fixed-idempotency-key",
    )

    assert send.await_args.kwargs["headers"] == {
        "Idempotency-Key": "fixed-idempotency-key"
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("client_type", [RLIClient, RLIClientSession])
@pytest.mark.parametrize("recreate_client", [False, True])
async def test_create_session_fallback_reuses_session_after_lost_response(
    monkeypatch: pytest.MonkeyPatch,
    client_type: type[RLIClient],
    recreate_client: bool,
) -> None:
    sessions: dict[str, str] = {}
    lose_response = True

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal lose_response
        tenant = json.loads(request.content)["tenantId"]
        if tenant == "bench":
            return httpx.Response(
                400,
                json={"error": {"diagnostic": {"reason": "ARI_TENANT_NOT_FOUND"}}},
            )
        assert tenant is None
        key = request.headers["Idempotency-Key"]
        assert key != "original-key"
        if key not in sessions:
            sessions[key] = f"session-{len(sessions) + 1}"
        if lose_response:
            lose_response = False
            raise httpx.ReadTimeout("Session created; response lost", request=request)
        return httpx.Response(
            201,
            json={
                "status": "ok",
                "result": {
                    "session_id": sessions[key],
                    "created_at": "2026-05-30T12:17:21Z",
                    "backend_ref": "backend",
                    "backend_type": "SWE",
                    "shell_mode": "current",
                    "tool_schemas": [],
                    "terminal_url": "https://example.invalid/terminal",
                    "workspace_path": "/",
                },
            },
        )

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        partial(httpx.AsyncClient, transport=httpx.MockTransport(handle)),
    )
    request = CreateSessionRequest(
        env=SessionEnv(name="test-env", version="1.0.0", task_id="test-task"),
    )
    client = client_type(base_url="https://example.invalid")
    try:
        with pytest.raises(httpx.ReadTimeout, match="response lost"):
            await client.create_session(request, idempotency_key="original-key")
        assert list(sessions.values()) == ["session-1"]

        if recreate_client:
            if isinstance(client, RLIClientSession):
                await client.aclose()
            client = client_type(base_url="https://example.invalid")
        result = await client.create_session(request, idempotency_key="original-key")

        assert result.session_id == "session-1"
        assert list(sessions.values()) == ["session-1"]
    finally:
        if isinstance(client, RLIClientSession):
            await client.aclose()


@pytest.mark.asyncio
async def test_create_session_traces_http_status_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "rli_trace.jsonl"
    client = RLIClient(
        base_url="https://rli.sberdevices.ru",
        trace_path=trace_path,
    )
    request = CreateSessionRequest(
        env=SessionEnv(
            name="terminal-bench-modernize-scientific-stack",
            version="1.0.0",
            task_id="modernize-scientific-stack",
        ),
        ttl_ms=3_600_000,
        use_pip_proxy=False,
    )
    send = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "Server error '503 Service Unavailable' for url 'https://rli.sberdevices.ru/v1/sessions'",
            request=httpx.Request("POST", "https://rli.sberdevices.ru/v1/sessions"),
            response=httpx.Response(
                503,
                request=httpx.Request(
                    "POST",
                    "https://rli.sberdevices.ru/v1/sessions",
                ),
            ),
        )
    )
    monkeypatch.setattr(client, "_send", send)

    with pytest.raises(httpx.HTTPStatusError):
        await client.create_session(request, idempotency_key="fixed-idempotency-key")

    events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert events == [
        {
            "ts": events[0]["ts"],
            "stage": "create_session",
            "success": False,
            "request": {
                "json": {
                    "env": {
                        "name": "terminal-bench-modernize-scientific-stack",
                        "version": "1.0.0",
                        "task_id": "modernize-scientific-stack",
                    },
                    "ttl_ms": 3_600_000,
                    "use_pip_proxy": False,
                    "allow_internet_access": None,
                    "tenantId": "bench",
                    "attribution": None,
                    "project_id": None,
                },
                "headers": {"Idempotency-Key": "fixed-idempotency-key"},
            },
            "response": {
                "status_code": 503,
                "body": None,
            },
            "error": (
                "Server error '503 Service Unavailable' for url "
                "'https://rli.sberdevices.ru/v1/sessions'"
            ),
        }
    ]


@pytest.mark.asyncio
async def test_get_session_fetches_expected_path_and_returns_result_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client()
    send = AsyncMock(
        return_value={
            "status": "ok",
            "result": {
                "status": "RUNNING",
                "session_id": "sess-123",
                "env_name": "terminal-bench-modernize-scientific-stack",
                "env_version": "1.0.0",
                "task_id": "modernize-scientific-stack",
                "ttl_ms": 3_600_000,
                "backend_type": "SWE",
                "backend_ref": "https://example.invalid/backend/sess-123",
                "shell_mode": "current",
                "created_at": "2026-05-30T12:17:21.317240Z",
                "last_seen": "2026-05-30T12:17:21.317240Z",
            },
            "meta": {
                "version": "rli-api-gateway@2.1.1",
                "request_id": "req-123",
                "session_id": "sess-123",
                "runtime_ms": 1,
                "schema_version": "1.0",
            },
        }
    )
    monkeypatch.setattr(client, "_send", send)

    result = await client.get_session("sess-123")

    assert isinstance(result, GetSessionResult)
    assert result.status == "RUNNING"
    send.assert_awaited_once_with("GET", "/v1/sessions/sess-123")


@pytest.mark.asyncio
async def test_get_session_traces_http_status_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "rli_trace.jsonl"
    client = RLIClient(
        base_url="https://rli.sberdevices.ru",
        trace_path=trace_path,
    )
    send = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "Server error '503 Service Unavailable' for url 'https://rli.sberdevices.ru/v1/sessions/sess-123'",
            request=httpx.Request("GET", "https://rli.sberdevices.ru/v1/sessions/sess-123"),
            response=httpx.Response(
                503,
                request=httpx.Request(
                    "GET",
                    "https://rli.sberdevices.ru/v1/sessions/sess-123",
                ),
            ),
        )
    )
    monkeypatch.setattr(client, "_send", send)

    with pytest.raises(httpx.HTTPStatusError):
        await client.get_session("sess-123")

    events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert events == [
        {
            "ts": events[0]["ts"],
            "stage": "get_session",
            "success": False,
            "request": {"session_id": "sess-123"},
            "response": {
                "status_code": 503,
                "body": None,
            },
            "error": (
                "Server error '503 Service Unavailable' for url "
                "'https://rli.sberdevices.ru/v1/sessions/sess-123'"
            ),
        }
    ]


@pytest.mark.asyncio
async def test_step_posts_expected_payload_and_returns_result_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client()
    send = AsyncMock(
        return_value={
            "status": "ok",
            "result": {
                "observation": {"output": "stdout:\\nhello\\n"},
                "events": [
                    {
                        "action_index": 0,
                        "stdout_lines": ["hello"],
                        "stderr_lines": [],
                        "stdout_meta": {
                            "total_lines": 1,
                            "returned_lines": 1,
                            "total_bytes": 5,
                        },
                        "stderr_meta": {
                            "total_lines": 0,
                            "returned_lines": 0,
                            "total_bytes": 0,
                        },
                        "stdout_ref": "hello",
                        "stderr_ref": "",
                        "exit_code": 0,
                        "duration_ms": 511,
                    }
                ],
                "terminated": False,
                "agent_id": "default",
                "applied_limits": {"deadline_ms": 60_000},
            },
            "meta": {
                "version": "rli-api-gateway@2.1.1",
                "request_id": "req-123",
                "session_id": "sess-123",
                "runtime_ms": 514,
                "schema_version": "1.0",
            },
        }
    )
    monkeypatch.setattr(client, "_send", send)

    result = await client.step(
        StepRequest(
            session_id="sess-123",
            mode="single",
            actions=[
                StepAction(
                    type="env_action",
                    name="bash",
                    args=StepActionArgs(cmd="set -x; pwd"),
                )
            ],
        )
    )

    assert isinstance(result, StepResult)
    assert result.terminated is False
    send.assert_awaited_once_with(
        "POST",
        "/v1/sessions/step",
        json={
            "session_id": "sess-123",
            "mode": "single",
            "actions": [
                {
                    "type": "env_action",
                    "name": "bash",
                    "args": {"cmd": "set -x; pwd"},
                }
            ],
            "limits": None,
        },
    )


@pytest.mark.asyncio
async def test_step_traces_http_status_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "rli_trace.jsonl"
    client = RLIClient(
        base_url="https://rli.sberdevices.ru",
        trace_path=trace_path,
    )
    request = StepRequest(
        session_id="sess-123",
        mode="single",
        actions=[
            StepAction(
                type="env_action",
                name="bash",
                args=StepActionArgs(cmd="set -x; pwd"),
            )
        ],
    )
    send = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "Server error '500 Internal Server Error' for url 'https://rli.sberdevices.ru/v1/sessions/step'",
            request=httpx.Request("POST", "https://rli.sberdevices.ru/v1/sessions/step"),
            response=httpx.Response(
                500,
                request=httpx.Request(
                    "POST",
                    "https://rli.sberdevices.ru/v1/sessions/step",
                ),
            ),
        )
    )
    monkeypatch.setattr(client, "_send", send)

    with pytest.raises(httpx.HTTPStatusError):
        await client.step(request)

    events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert events == [
        {
            "ts": events[0]["ts"],
            "stage": "step",
            "success": False,
            "request": {
                "json": {
                    "session_id": "sess-123",
                    "mode": "single",
                    "actions": [
                        {
                            "type": "env_action",
                            "name": "bash",
                            "args": {"cmd": "set -x; pwd"},
                        }
                    ],
                    "limits": None,
                }
            },
            "response": {
                "status_code": 500,
                "body": None,
            },
            "error": (
                "Server error '500 Internal Server Error' for url "
                "'https://rli.sberdevices.ru/v1/sessions/step'"
            ),
        }
    ]


@pytest.mark.asyncio
async def test_close_session_posts_expected_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client()
    send = AsyncMock(
        return_value={
            "status": "ok",
            "meta": {
                "version": "rli-api-gateway@2.1.1",
                "request_id": "req-123",
                "session_id": "sess-123",
                "runtime_ms": 0,
                "schema_version": "1.0",
            },
        }
    )
    monkeypatch.setattr(client, "_send", send)

    await client.close_session(CloseSessionRequest(session_id="sess-123"))

    send.assert_awaited_once_with(
        "POST",
        "/v1/sessions/close",
        json={"session_id": "sess-123"},
    )


@pytest.mark.asyncio
async def test_close_session_traces_http_status_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "rli_trace.jsonl"
    client = RLIClient(
        base_url="https://rli.sberdevices.ru",
        trace_path=trace_path,
    )
    send = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "Server error '503 Service Unavailable' for url 'https://rli.sberdevices.ru/v1/sessions/close'",
            request=httpx.Request("POST", "https://rli.sberdevices.ru/v1/sessions/close"),
            response=httpx.Response(
                503,
                request=httpx.Request(
                    "POST",
                    "https://rli.sberdevices.ru/v1/sessions/close",
                ),
            ),
        )
    )
    monkeypatch.setattr(client, "_send", send)

    with pytest.raises(httpx.HTTPStatusError):
        await client.close_session(CloseSessionRequest(session_id="sess-123"))

    events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert events == [
        {
            "ts": events[0]["ts"],
            "stage": "close_session",
            "success": False,
            "request": {"json": {"session_id": "sess-123"}},
            "response": {
                "status_code": 503,
                "body": None,
            },
            "error": (
                "Server error '503 Service Unavailable' for url "
                "'https://rli.sberdevices.ru/v1/sessions/close'"
            ),
        }
    ]


@pytest.mark.asyncio
async def test_close_session_accepts_success_response_without_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client()
    send = AsyncMock(
        return_value={
            "status": "ok",
            "meta": {
                "version": "rli-api-gateway@2.1.1",
                "request_id": "req-123",
                "session_id": "sess-123",
                "runtime_ms": 0,
                "schema_version": "1.0",
            },
        }
    )
    monkeypatch.setattr(client, "_send", send)

    result = await client.close_session(CloseSessionRequest(session_id="sess-123"))

    assert result is None


@pytest.mark.asyncio
async def test_error_envelope_raises_rli_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client()
    send = AsyncMock(
        return_value={
            "status": "error",
            "error": {
                "id": "err-123",
                "code": "INVALID_ARGUMENT",
                "category": "client_error",
                "message": "Invalid request body",
                "retryable": False,
            },
            "meta": {
                "version": "rli-api-gateway@2.1.1",
                "request_id": "req-123",
                "runtime_ms": 0,
                "schema_version": "1.0",
            },
        }
    )
    monkeypatch.setattr(client, "_send", send)

    with pytest.raises(RLIClientError) as exc_info:
        await client.get_session("sess-123")

    err = exc_info.value
    assert err.error.code == "INVALID_ARGUMENT"
    assert err.error.message == "Invalid request body"
    assert isinstance(err.response, ErrorResponse)


@pytest.mark.asyncio
async def test_error_envelope_preserves_extended_error_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client()
    send = AsyncMock(
        return_value={
            "status": "error",
            "error": {
                "id": "err-123",
                "code": "SESSION_CLOSED",
                "category": "client_error",
                "message": "Session is not running: FAILED.",
                "retryable": False,
                "title_ru": "Сессия закрыта",
                "title_en": "Session closed",
                "message_ru": "Сессия была закрыта и больше не принимает команды.",
                "message_en": "Session has been closed and no longer accepts commands.",
                "user_action_ru": "Создайте новую сессию.",
                "user_action_en": "Create a new session.",
                "retry_strategy": "create_new_session",
                "docs_url": "https://docs.gigarli/errors/SESSION_CLOSED",
            },
            "meta": {
                "version": "rli-api-gateway@2.1.1",
                "request_id": "req-123",
                "runtime_ms": 0,
                "schema_version": "1.0",
            },
        }
    )
    monkeypatch.setattr(client, "_send", send)

    with pytest.raises(RLIClientError) as exc_info:
        await client.get_session("sess-123")

    err = exc_info.value
    assert err.error.title_en == "Session closed"
    assert err.error.title_ru == "Сессия закрыта"
    assert err.error.retry_strategy == "create_new_session"
    assert err.error.docs_url == "https://docs.gigarli/errors/SESSION_CLOSED"
    assert err.response.error.user_action_en == "Create a new session."


@pytest.mark.asyncio
async def test_step_raises_rli_client_error_on_error_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client()
    send = AsyncMock(
        return_value={
            "status": "error",
            "error": {
                "id": "err-123",
                "code": "RESOURCE_EXHAUSTED",
                "category": "client_error",
                "message": "Backend has no free resources",
                "retryable": True,
                "retry_strategy": "wait_and_retry_step",
            },
            "meta": {
                "version": "rli-api-gateway@2.1.1",
                "request_id": "req-123",
                "session_id": "sess-123",
                "runtime_ms": 0,
                "schema_version": "1.0",
            },
        }
    )
    monkeypatch.setattr(client, "_send", send)

    with pytest.raises(RLIClientError) as exc_info:
        await client.step(
            StepRequest(
                session_id="sess-123",
                mode="single",
                actions=[
                    StepAction(
                        type="env_action",
                        name="bash",
                        args=StepActionArgs(cmd="echo hi"),
                    )
                ],
            )
        )

    assert exc_info.value.error.code == "RESOURCE_EXHAUSTED"
    assert exc_info.value.error.retryable is True


@pytest.mark.asyncio
async def test_step_writes_trace_event_on_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pytest.TempPathFactory,
) -> None:
    trace_path = tmp_path / "rli_trace.jsonl"
    client = RLIClient(
        base_url="https://rli.sberdevices.ru",
        trace_path=trace_path,
    )
    send = AsyncMock(
        return_value={
            "status": "ok",
            "result": {
                "observation": {"output": "stdout:\\nhello\\n"},
                "events": [
                    {
                        "action_index": 0,
                        "stdout_lines": ["hello"],
                        "stderr_lines": [],
                        "stdout_meta": {
                            "total_lines": 1,
                            "returned_lines": 1,
                            "total_bytes": 5,
                        },
                        "stderr_meta": {
                            "total_lines": 0,
                            "returned_lines": 0,
                            "total_bytes": 0,
                        },
                        "stdout_ref": "hello",
                        "stderr_ref": "",
                        "exit_code": 0,
                        "duration_ms": 511,
                    }
                ],
                "terminated": False,
                "agent_id": "default",
                "applied_limits": {"deadline_ms": 60_000},
            },
            "meta": {
                "version": "rli-api-gateway@2.1.1",
                "request_id": "req-123",
                "session_id": "sess-123",
                "runtime_ms": 514,
                "schema_version": "1.0",
            },
        }
    )
    monkeypatch.setattr(client, "_send", send)

    await client.step(
        StepRequest(
            session_id="sess-123",
            mode="single",
            actions=[
                StepAction(
                    type="env_action",
                    name="bash",
                    args=StepActionArgs(cmd="set -x; pwd"),
                )
            ],
        )
    )

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    event = json.loads(lines[0])
    assert event["stage"] == "step"
    assert event["success"] is True
    assert event["request"]["json"]["session_id"] == "sess-123"
    assert event["request"]["json"]["actions"][0]["args"]["cmd"] == "set -x; pwd"
    assert event["response"]["result"]["events"][0]["exit_code"] == 0


@pytest.mark.asyncio
async def test_get_session_writes_trace_event_on_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pytest.TempPathFactory,
) -> None:
    trace_path = tmp_path / "rli_trace.jsonl"
    client = RLIClient(
        base_url="https://rli.sberdevices.ru",
        trace_path=trace_path,
    )
    send = AsyncMock(
        return_value={
            "status": "error",
            "error": {
                "id": "err-123",
                "code": "SESSION_CLOSED",
                "category": "client_error",
                "message": "Session is already closed",
                "retryable": False,
                "title_ru": "Сессия закрыта",
                "title_en": "Session closed",
                "message_ru": "Сессия уже закрыта.",
                "message_en": "The session is already closed.",
                "user_action_ru": "Создайте новую сессию.",
                "user_action_en": "Create a new session.",
                "retry_strategy": "fix_request",
                "docs_url": "https://docs.gigarli/errors/SESSION_CLOSED",
            },
            "meta": {
                "version": "rli-api-gateway@2.1.1",
                "request_id": "req-123",
                "session_id": "sess-123",
                "runtime_ms": 1,
                "schema_version": "1.0",
            },
        }
    )
    monkeypatch.setattr(client, "_send", send)

    with pytest.raises(RLIClientError):
        await client.get_session("sess-123")

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    event = json.loads(lines[0])
    assert event["stage"] == "get_session"
    assert event["success"] is False
    assert event["request"]["session_id"] == "sess-123"
    assert event["response"]["error"]["code"] == "SESSION_CLOSED"


@pytest.mark.asyncio
async def test_client_does_not_store_current_session_between_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client()
    send = AsyncMock(
        side_effect=[
            {
                "status": "ok",
                "result": {
                    "status": "RUNNING",
                    "session_id": "sess-a",
                    "env_name": "demo",
                    "env_version": "1.0.0",
                    "task_id": "task-a",
                    "ttl_ms": 3_600_000,
                    "backend_type": "SWE",
                    "backend_ref": "https://example.invalid/backend/sess-a",
                    "shell_mode": "current",
                    "created_at": "2026-05-30T12:17:21.317240Z",
                    "last_seen": "2026-05-30T12:17:21.317240Z",
                },
                "meta": {
                    "version": "rli-api-gateway@2.1.1",
                    "request_id": "req-a",
                    "session_id": "sess-a",
                    "runtime_ms": 1,
                    "schema_version": "1.0",
                },
            },
            {
                "status": "ok",
                "result": {
                    "status": "RUNNING",
                    "session_id": "sess-b",
                    "env_name": "demo",
                    "env_version": "1.0.0",
                    "task_id": "task-b",
                    "ttl_ms": 3_600_000,
                    "backend_type": "SWE",
                    "backend_ref": "https://example.invalid/backend/sess-b",
                    "shell_mode": "current",
                    "created_at": "2026-05-30T12:17:21.317240Z",
                    "last_seen": "2026-05-30T12:17:21.317240Z",
                },
                "meta": {
                    "version": "rli-api-gateway@2.1.1",
                    "request_id": "req-b",
                    "session_id": "sess-b",
                    "runtime_ms": 1,
                    "schema_version": "1.0",
                },
            },
        ]
    )
    monkeypatch.setattr(client, "_send", send)

    first = await client.get_session("sess-a")
    second = await client.get_session("sess-b")

    assert first.session_id == "sess-a"
    assert second.session_id == "sess-b"
    assert send.await_args_list[0].args == ("GET", "/v1/sessions/sess-a")
    assert send.await_args_list[1].args == ("GET", "/v1/sessions/sess-b")
