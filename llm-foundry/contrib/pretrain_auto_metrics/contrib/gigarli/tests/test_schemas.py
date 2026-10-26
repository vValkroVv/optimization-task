from __future__ import annotations

import pytest
from pydantic import ValidationError

from gigarli.schemas.common import (
    ErrorDetails,
    ErrorResponse,
    RequestLimits,
    ResponseMeta,
    SuccessResponse,
)
from gigarli.schemas.env import (
    CreateSessionRequest,
    CreateSessionResult,
    SessionEnv,
    CloseSessionRequest,
    GetSessionResult,
    StepAction,
    StepActionArgs,
    StepEvent,
    StepObservation,
    StepRequest,
    StepResult,
    StepStreamMeta,
)


def test_response_meta_requires_core_fields() -> None:
    with pytest.raises(ValidationError):
        ResponseMeta()

    meta = ResponseMeta(
        request_id="req-123",
        runtime_ms=12,
        version="rli-api-gateway@2.1.1",
        schema_version="1.0",
    )

    assert meta.request_id == "req-123"
    assert meta.runtime_ms == 12
    assert meta.version == "rli-api-gateway@2.1.1"
    assert meta.schema_version == "1.0"
    assert meta.session_id is None


def test_response_meta_accepts_optional_session_id() -> None:
    meta = ResponseMeta(
        request_id="req-123",
        session_id="sess-123",
        runtime_ms=12,
        version="rli-api-gateway@2.1.1",
        schema_version="1.0",
    )

    assert meta.session_id == "sess-123"


def test_error_details_requires_contract_fields() -> None:
    with pytest.raises(ValidationError):
        ErrorDetails()

    error = ErrorDetails(
        id="err-123",
        code="INVALID_ARGUMENT",
        category="client_error",
        message="Invalid request body",
        retryable=False,
    )

    assert error.id == "err-123"
    assert error.code == "INVALID_ARGUMENT"
    assert error.category == "client_error"
    assert error.message == "Invalid request body"
    assert error.retryable is False


def test_error_details_accepts_extended_documented_fields() -> None:
    error = ErrorDetails(
        id="err-123",
        code="SESSION_CLOSED",
        category="client_error",
        message="Session is not running: FAILED.",
        retryable=False,
        title_ru="Сессия закрыта",
        title_en="Session closed",
        message_ru="Сессия была закрыта и больше не принимает команды.",
        message_en="Session has been closed and no longer accepts commands.",
        user_action_ru="Создайте новую сессию.",
        user_action_en="Create a new session.",
        retry_strategy="create_new_session",
        docs_url="https://docs.gigarli/errors/SESSION_CLOSED",
    )

    dumped = error.model_dump()
    assert dumped["title_ru"] == "Сессия закрыта"
    assert dumped["title_en"] == "Session closed"
    assert dumped["message_ru"] == "Сессия была закрыта и больше не принимает команды."
    assert dumped["message_en"] == "Session has been closed and no longer accepts commands."
    assert dumped["user_action_ru"] == "Создайте новую сессию."
    assert dumped["user_action_en"] == "Create a new session."
    assert dumped["retry_strategy"] == "create_new_session"
    assert dumped["docs_url"] == "https://docs.gigarli/errors/SESSION_CLOSED"


def test_request_limits_allows_deadline_ms() -> None:
    limits = RequestLimits(deadline_ms=60_000)
    assert limits.deadline_ms == 60_000


def test_success_response_defaults_status_to_ok() -> None:
    meta = ResponseMeta(
        request_id="req-123",
        runtime_ms=12,
        version="rli-api-gateway@2.1.1",
        schema_version="1.0",
    )

    response = SuccessResponse[dict](
        result={"session_id": "sess-123"},
        meta=meta,
    )

    dumped = response.model_dump()
    assert dumped["status"] == "ok"
    assert dumped["result"] == {"session_id": "sess-123"}
    assert dumped["meta"]["request_id"] == "req-123"


def test_error_response_defaults_status_to_error() -> None:
    meta = ResponseMeta(
        request_id="req-123",
        runtime_ms=12,
        version="rli-api-gateway@2.1.1",
        schema_version="1.0",
    )
    error = ErrorDetails(
        id="err-123",
        code="INVALID_ARGUMENT",
        category="client_error",
        message="Invalid request body",
        retryable=False,
    )

    response = ErrorResponse(error=error, meta=meta)

    dumped = response.model_dump()
    assert dumped["status"] == "error"
    assert dumped["error"]["code"] == "INVALID_ARGUMENT"
    assert dumped["meta"]["request_id"] == "req-123"


def test_session_env_requires_name_version_and_task_id() -> None:
    with pytest.raises(ValidationError):
        SessionEnv()

    env = SessionEnv(
        name="terminal-bench-modernize-scientific-stack",
        version="1.0.0",
        task_id="modernize-scientific-stack",
    )

    assert env.name == "terminal-bench-modernize-scientific-stack"
    assert env.version == "1.0.0"
    assert env.task_id == "modernize-scientific-stack"


def test_create_session_request_allows_observed_optional_fields() -> None:
    req = CreateSessionRequest(
        env=SessionEnv(
            name="terminal-bench-modernize-scientific-stack",
            version="1.0.0",
            task_id="modernize-scientific-stack",
        ),
        ttl_ms=3_600_000,
        use_pip_proxy=True,
    )

    dumped = req.model_dump()
    assert dumped["env"]["name"] == "terminal-bench-modernize-scientific-stack"
    assert dumped["ttl_ms"] == 3_600_000
    assert dumped["use_pip_proxy"] is True


def test_create_session_result_requires_core_fields() -> None:
    with pytest.raises(ValidationError):
        CreateSessionResult()

    result = CreateSessionResult(
        session_id="sess-123",
        created_at="2026-05-30T12:17:21.317239741Z",
        backend_ref="https://example.invalid/backend/sess-123",
        backend_type="SWE",
        shell_mode="current",
        tool_schemas=[],
        terminal_url="https://example.invalid/backend/sess-123/terminal",
        workspace_path="/",
    )

    assert result.session_id == "sess-123"
    assert result.backend_type == "SWE"
    assert result.workspace_path == "/"


def test_get_session_result_accepts_running_and_closed_statuses() -> None:
    running = GetSessionResult(
        status="RUNNING",
        session_id="sess-123",
        env_name="terminal-bench-modernize-scientific-stack",
        env_version="1.0.0",
        task_id="modernize-scientific-stack",
        ttl_ms=3_600_000,
        backend_type="SWE",
        backend_ref="https://example.invalid/backend/sess-123",
        shell_mode="current",
        created_at="2026-05-30T12:17:21.317240Z",
        last_seen="2026-05-30T12:17:21.317240Z",
    )
    closed = GetSessionResult(
        status="CLOSED",
        session_id="sess-123",
        env_name="terminal-bench-modernize-scientific-stack",
        env_version="1.0.0",
        task_id="modernize-scientific-stack",
        ttl_ms=3_600_000,
        backend_type="SWE",
        backend_ref="https://example.invalid/backend/sess-123",
        shell_mode="current",
        created_at="2026-05-30T12:17:21.317240Z",
        last_seen="2026-05-30T12:17:21.317240Z",
        closed_at="2026-05-30T12:18:21.317240Z",
    )

    assert running.status == "RUNNING"
    assert closed.status == "CLOSED"
    assert closed.closed_at == "2026-05-30T12:18:21.317240Z"


def test_step_action_requires_observed_working_fields() -> None:
    with pytest.raises(ValidationError):
        StepAction()

    action = StepAction(
        type="env_action",
        name="bash",
        args=StepActionArgs(cmd="set -x; pwd"),
    )

    dumped = action.model_dump()
    assert dumped["type"] == "env_action"
    assert dumped["name"] == "bash"
    assert dumped["args"]["cmd"] == "set -x; pwd"


def test_step_request_requires_session_mode_and_actions() -> None:
    with pytest.raises(ValidationError):
        StepRequest()

    req = StepRequest(
        session_id="sess-123",
        mode="single",
        actions=[
            StepAction(
                type="env_action",
                name="bash",
                args=StepActionArgs(cmd="set -x; pwd"),
            )
        ],
        limits=RequestLimits(deadline_ms=60_000),
    )

    dumped = req.model_dump()
    assert dumped["session_id"] == "sess-123"
    assert dumped["mode"] == "single"
    assert dumped["limits"]["deadline_ms"] == 60_000


def test_step_result_supports_observed_output_structure() -> None:
    result = StepResult(
        observation=StepObservation(output="stdout:\\nhello\\n"),
        events=[
            StepEvent(
                action_index=0,
                stdout_lines=["hello"],
                stderr_lines=[],
                stdout_meta=StepStreamMeta(
                    total_lines=1,
                    returned_lines=1,
                    total_bytes=5,
                ),
                stderr_meta=StepStreamMeta(
                    total_lines=0,
                    returned_lines=0,
                    total_bytes=0,
                ),
                stdout_ref="hello",
                stderr_ref="",
                exit_code=0,
                duration_ms=511,
            )
        ],
        terminated=False,
        agent_id="default",
        applied_limits=RequestLimits(deadline_ms=60_000),
    )

    dumped = result.model_dump()
    assert dumped["observation"]["output"] == "stdout:\\nhello\\n"
    assert dumped["events"][0]["exit_code"] == 0
    assert dumped["applied_limits"]["deadline_ms"] == 60_000


def test_close_session_request_requires_session_id() -> None:
    with pytest.raises(ValidationError):
        CloseSessionRequest()

    req = CloseSessionRequest(session_id="sess-123")
    assert req.session_id == "sess-123"
