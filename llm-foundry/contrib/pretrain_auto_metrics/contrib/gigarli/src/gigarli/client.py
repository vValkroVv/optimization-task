from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

import httpx
from pydantic import ValidationError

from .schemas.common import ErrorResponse
from .schemas.env import (
    CloseSessionRequest,
    CreateSessionRequest,
    CreateSessionResult,
    GetSessionResult,
    StepRequest,
    StepResult,
)


DEFAULT_RLI_BASE_URL = "https://rli.sberdevices.ru"


class RLIClientError(Exception):
    """Raised when GigaRLI returns an error response envelope."""

    def __init__(self, response: ErrorResponse):
        self.response = response
        self.error = response.error
        super().__init__(self.error.message)


class RLIClient:
    """Minimal async client for the GigaRLI sessions API."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        verify: bool = True,
        timeout: float = 30.0,
        trace_path: str | Path | None = None,
    ) -> None:
        if base_url is None:
            base_url = os.getenv("RLI_API_BASE", DEFAULT_RLI_BASE_URL)
        self.base_url = base_url.rstrip("/")
        self.verify = verify
        self.timeout = timeout
        self.trace_path = Path(trace_path) if trace_path is not None else None

    @asynccontextmanager
    async def _http_client(self) -> AsyncIterator[httpx.AsyncClient]:
        async with httpx.AsyncClient(
            base_url=self.base_url,
            verify=self.verify,
            timeout=self.timeout,
        ) as client:
            yield client

    async def create_session(
        self,
        request: CreateSessionRequest,
        *,
        idempotency_key: str | None = None,
    ) -> CreateSessionResult:
        idempotency_key = idempotency_key or str(uuid4())
        if request.tenant_id is None:
            request = request.model_copy(update={"tenant_id": "bench"})
        try:
            return await self._create_session(request, idempotency_key=idempotency_key)
        except httpx.HTTPStatusError as exc:
            if request.tenant_id != "bench" or exc.response.status_code != 400:
                raise
            try:
                payload = exc.response.json()
            except ValueError:
                raise exc
            error = payload.get("error") if isinstance(payload, dict) else None
            diagnostic = error.get("diagnostic") if isinstance(error, dict) else None
            if (
                not isinstance(diagnostic, dict)
                or diagnostic.get("reason") != "ARI_TENANT_NOT_FOUND"
            ):
                raise
        return await self._create_session(
            request.model_copy(update={"tenant_id": None}),
            idempotency_key=str(
                uuid5(NAMESPACE_URL, f"gigarli:tenant-none:{idempotency_key}")
            ),
        )

    async def _create_session(
        self,
        request: CreateSessionRequest,
        *,
        idempotency_key: str | None = None,
    ) -> CreateSessionResult:
        request_payload = request.model_dump(by_alias=True)
        request_headers = {"Idempotency-Key": idempotency_key or str(uuid4())}
        try:
            payload = await self._send(
                "POST",
                "/v1/sessions",
                json=request_payload,
                headers=request_headers,
            )
            self._raise_for_error(payload)
            result = self._parse_result(payload, CreateSessionResult)
        except RLIClientError as exc:
            self._append_trace_event(
                stage="create_session",
                success=False,
                request={"json": request_payload, "headers": request_headers},
                response=exc.response.model_dump(),
                error=str(exc),
            )
            raise
        except httpx.HTTPStatusError as exc:
            self._append_trace_event(
                stage="create_session",
                success=False,
                request={"json": request_payload, "headers": request_headers},
                response=self._http_status_error_response(exc),
                error=str(exc),
            )
            raise
        except httpx.TransportError as exc:
            self._append_transport_error(
                stage="create_session",
                request={"json": request_payload, "headers": request_headers},
                exc=exc,
            )
            raise

        self._append_trace_event(
            stage="create_session",
            success=True,
            request={"json": request_payload, "headers": request_headers},
            response=payload,
        )
        return result

    async def get_session(self, session_id: str) -> GetSessionResult:
        request_payload = {"session_id": session_id}
        try:
            payload = await self._send(
                "GET",
                f"/v1/sessions/{session_id}",
            )
            self._raise_for_error(payload)
            result = self._parse_result(payload, GetSessionResult)
        except RLIClientError as exc:
            self._append_trace_event(
                stage="get_session",
                success=False,
                request=request_payload,
                response=exc.response.model_dump(),
                error=str(exc),
            )
            raise
        except httpx.HTTPStatusError as exc:
            self._append_trace_event(
                stage="get_session",
                success=False,
                request=request_payload,
                response=self._http_status_error_response(exc),
                error=str(exc),
            )
            raise
        except httpx.TransportError as exc:
            self._append_transport_error(
                stage="get_session",
                request=request_payload,
                exc=exc,
            )
            raise

        self._append_trace_event(
            stage="get_session",
            success=True,
            request=request_payload,
            response=payload,
        )
        return result

    async def step(self, request: StepRequest) -> StepResult:
        request_payload = request.model_dump()
        try:
            payload = await self._send(
                "POST",
                "/v1/sessions/step",
                json=request_payload,
            )
            self._raise_for_error(payload)
            result = self._parse_result(payload, StepResult)
        except ValidationError as exc:
            self._append_trace_event(
                stage="step",
                success=False,
                request={"json": request_payload},
                response=payload,
                error=str(exc),
            )
            raise
        except RLIClientError as exc:
            self._append_trace_event(
                stage="step",
                success=False,
                request={"json": request_payload},
                response=exc.response.model_dump(),
                error=str(exc),
            )
            raise
        except httpx.HTTPStatusError as exc:
            self._append_trace_event(
                stage="step",
                success=False,
                request={"json": request_payload},
                response=self._http_status_error_response(exc),
                error=str(exc),
            )
            raise
        except httpx.TransportError as exc:
            self._append_transport_error(
                stage="step",
                request={"json": request_payload},
                exc=exc,
            )
            raise

        self._append_trace_event(
            stage="step",
            success=True,
            request={"json": request_payload},
            response=payload,
        )
        return result

    async def close_session(self, request: CloseSessionRequest) -> None:
        request_payload = request.model_dump()
        try:
            payload = await self._send(
                "POST",
                "/v1/sessions/close",
                json=request_payload,
            )
            self._raise_for_error(payload)
        except RLIClientError as exc:
            self._append_trace_event(
                stage="close_session",
                success=False,
                request={"json": request_payload},
                response=exc.response.model_dump(),
                error=str(exc),
            )
            raise
        except httpx.HTTPStatusError as exc:
            self._append_trace_event(
                stage="close_session",
                success=False,
                request={"json": request_payload},
                response=self._http_status_error_response(exc),
                error=str(exc),
            )
            raise
        except httpx.TransportError as exc:
            self._append_transport_error(
                stage="close_session",
                request={"json": request_payload},
                exc=exc,
            )
            raise

        self._append_trace_event(
            stage="close_session",
            success=True,
            request={"json": request_payload},
            response=payload,
        )
        return None

    async def _send(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        async with self._http_client() as client:
            response = await client.request(method, path, json=json, headers=headers)
            response.raise_for_status()
            try:
                payload = response.json()
            except json.JSONDecodeError as exc:
                content_type = response.headers.get("content-type")
                raise httpx.RemoteProtocolError(
                    "RLI returned an invalid JSON response: "
                    f"status_code={response.status_code}, "
                    f"content_type={content_type!r}, "
                    f"content_length={len(response.content)}",
                    request=response.request,
                ) from exc

        return payload

    @staticmethod
    def _parse_result(payload: dict[str, Any], result_model: type[Any]) -> Any:
        return result_model.model_validate(payload["result"])

    @staticmethod
    def _raise_for_error(payload: dict[str, Any]) -> None:
        if payload.get("status") == "error":
            raise RLIClientError(ErrorResponse.model_validate(payload))

    @staticmethod
    def _http_status_error_response(exc: httpx.HTTPStatusError) -> dict[str, Any]:
        body: str | None
        try:
            body = exc.response.text
        except Exception:
            body = None
        if body == "":
            body = None
        return {
            "status_code": exc.response.status_code,
            "body": body,
        }

    def _append_transport_error(
        self,
        *,
        stage: str,
        request: dict[str, Any],
        exc: httpx.TransportError,
    ) -> None:
        self._append_trace_event(
            stage=stage,
            success=False,
            request=request,
            response={},
            error=f"{type(exc).__name__}: {exc}",
        )

    def _append_trace_event(
        self,
        *,
        stage: str,
        success: bool,
        request: dict[str, Any],
        response: dict[str, Any],
        error: str | None = None,
    ) -> None:
        if self.trace_path is None:
            return

        event = {
            "ts": datetime.now(UTC).isoformat(),
            "stage": stage,
            "success": success,
            "request": request,
            "response": response,
        }
        if error is not None:
            event["error"] = error

        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
