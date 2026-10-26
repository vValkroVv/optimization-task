"""Common schemas for the minimal GigaRLI sessions contract."""

from __future__ import annotations

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel


# ------------------------------------------------------------------------------
# Response meta
# ------------------------------------------------------------------------------


class ResponseMeta(BaseModel):
    """Response metadata."""

    request_id: str
    """Request identifier for tracing."""

    runtime_ms: int
    """Request runtime in milliseconds."""

    version: str
    """Service version string."""

    schema_version: str
    """Schema version string."""

    session_id: str | None = None
    """Session identifier when the response is tied to a session."""


# ------------------------------------------------------------------------------
# Error models
# ------------------------------------------------------------------------------


class ErrorDetails(BaseModel):
    """Error payload."""

    id: str
    """Error instance identifier."""

    code: str
    """Machine-readable error code."""

    category: str
    """High-level error category."""

    message: str
    """Human-readable error message."""

    retryable: bool
    """Whether the request can be retried safely."""

    title_ru: str | None = None
    """Short Russian error title."""

    title_en: str | None = None
    """Short English error title."""

    message_ru: str | None = None
    """Localized Russian error message."""

    message_en: str | None = None
    """Localized English error message."""

    user_action_ru: str | None = None
    """Suggested user action in Russian."""

    user_action_en: str | None = None
    """Suggested user action in English."""

    retry_strategy: str | None = None
    """Platform-provided retry strategy hint."""

    docs_url: str | None = None
    """Documentation URL for this error code."""


# ------------------------------------------------------------------------------
# Response envelope
# ------------------------------------------------------------------------------


T = TypeVar("T")


class SuccessResponse(BaseModel, Generic[T]):
    """Successful response envelope."""

    status: Literal["ok"] = "ok"
    """Response status marker for successful requests."""

    result: T
    """Successful result payload."""

    meta: ResponseMeta
    """Response metadata."""


class ErrorResponse(BaseModel):
    """Error response envelope."""

    status: Literal["error"] = "error"
    """Response status marker for failed requests."""

    error: ErrorDetails
    """Error payload."""

    meta: ResponseMeta
    """Response metadata."""


# ------------------------------------------------------------------------------
# Request limits
# ------------------------------------------------------------------------------


class RequestLimits(BaseModel):
    """Execution limits for a request."""

    deadline_ms: int | None = None
    """Execution deadline in milliseconds."""
