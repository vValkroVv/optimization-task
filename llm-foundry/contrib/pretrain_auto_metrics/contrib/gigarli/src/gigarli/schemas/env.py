from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .common import RequestLimits


class SessionEnv(BaseModel):
    """Environment descriptor used during session creation."""

    name: str
    version: str
    task_id: str


class SessionAttribution(BaseModel):
    username: str | None = None
    stream: str | None = None
    release_type: str | None = None


class CreateSessionRequest(BaseModel):
    """Request model for creating a new session."""

    env: SessionEnv
    tenant_id: str | None = Field(default=None, serialization_alias="tenantId")
    attribution: SessionAttribution | None = None
    project_id: str | None = Field(default=None, max_length=128)
    ttl_ms: int | None = None
    use_pip_proxy: bool | None = None
    allow_internet_access: bool | None = None


class CreateSessionResult(BaseModel):
    """Observed payload returned after successful session creation."""

    session_id: str
    created_at: str
    backend_ref: str
    backend_type: str
    shell_mode: str
    tool_schemas: list[Any]
    terminal_url: str
    workspace_path: str


class GetSessionResult(BaseModel):
    """Observed payload returned by get-session requests."""

    status: Literal["RUNNING", "CLOSED"]
    session_id: str
    env_name: str
    env_version: str
    task_id: str
    ttl_ms: int
    backend_type: str
    backend_ref: str
    shell_mode: str
    created_at: str
    last_seen: str
    closed_at: str | None = None


class StepActionArgs(BaseModel):
    """Arguments for a single step action."""

    cmd: str


class StepAction(BaseModel):
    """Single action executed as part of a step request."""

    type: str
    name: str
    args: StepActionArgs


class StepRequest(BaseModel):
    """Request model for executing actions inside an existing session."""

    session_id: str
    mode: str
    actions: list[StepAction]
    limits: RequestLimits | None = None


class StepObservation(BaseModel):
    """Top-level observation returned by a step."""

    output: str


class StepStreamMeta(BaseModel):
    """Line and byte counters for a single output stream."""

    total_lines: int
    returned_lines: int
    total_bytes: int


class StepEvent(BaseModel):
    """Observed per-action event returned by a step request."""

    error: str | None = Field(default=None, exclude=True)
    action_index: int
    stdout_lines: list[str]
    stderr_lines: list[str]
    stdout_meta: StepStreamMeta
    stderr_meta: StepStreamMeta
    stdout_ref: str | None = None
    stderr_ref: str | None = None
    exit_code: int
    duration_ms: int


class StepResult(BaseModel):
    """Observed payload returned after step execution."""

    observation: StepObservation
    events: list[StepEvent]
    terminated: bool
    agent_id: str | None = None
    applied_limits: RequestLimits | None = None


class CloseSessionRequest(BaseModel):
    """Request model for closing an existing session."""

    session_id: str
