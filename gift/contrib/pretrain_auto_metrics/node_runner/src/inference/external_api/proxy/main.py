from __future__ import annotations

import asyncio
import json
import logging
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Mapping

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from ....logger import create_stdout_logger
from ....pam_proxies.models import SGLangChatCompletionRequest
from .auth import IftTokenProvider, IftTokenRefreshError
from .config import load_settings, log_settings
from .mappers import IftMapper, OpenAiMapper

settings = load_settings()
external_api_logger = create_stdout_logger(
    "external_api_proxy", level=settings.logging_level.upper()
)
log_settings(settings, external_api_logger)
upstream_semaphore = asyncio.Semaphore(settings.max_concurrency)

if settings.formatting == "openai":
    mapper = OpenAiMapper()
elif settings.formatting == "ift":
    mapper = IftMapper()

ift_token_provider = (
    IftTokenProvider(settings.user_pass, settings.base_url)
    if settings.user_pass is not None
    else None
)


@dataclass(frozen=True, slots=True)
class ExternalApiRetryableStatusError(Exception):
    response: httpx.Response


@dataclass(frozen=True, slots=True)
class ExternalApiUnexpectedResponseError(Exception):
    detail: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.timeout),
        verify=False,
    )
    try:
        yield
    finally:
        client: httpx.AsyncClient = app.state.http_client
        await client.aclose()


app = FastAPI(title="External API proxy", lifespan=lifespan)


def log_exception(context: str, exc: Exception) -> None:
    formatted_traceback = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    )
    external_api_logger.error("%s: %r\n%s", context, exc, formatted_traceback)


@app.exception_handler(httpx.RequestError)
async def handle_request_error(_: Request, exc: httpx.RequestError) -> JSONResponse:
    log_exception("External API transport error", exc)
    return JSONResponse(
        status_code=502,
        content={"detail": f"External API request failed: {exc}"},
    )


@app.exception_handler(ExternalApiUnexpectedResponseError)
async def handle_unexpected_response_error(
    _: Request, exc: ExternalApiUnexpectedResponseError
) -> JSONResponse:
    log_exception("Unexpected External API response", exc)
    return JSONResponse(status_code=502, content={"detail": exc.detail})


@app.exception_handler(IftTokenRefreshError)
async def handle_ift_token_refresh_error(
    _: Request, exc: IftTokenRefreshError
) -> JSONResponse:
    log_exception("IFT token refresh error", exc)
    return JSONResponse(status_code=502, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    log_exception(f"Unhandled error for {request.method} {request.url}", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Unexpected External API proxy error"},
    )


@retry(
    stop=stop_after_attempt(max(1, settings.max_retries + 1)),
    wait=wait_fixed(settings.retry_wait_sec),
    reraise=True,
    retry=retry_if_exception_type(
        (httpx.RequestError, ExternalApiRetryableStatusError)
    ),
)
async def post_upstream(
    *,
    client: httpx.AsyncClient,
    payload: dict[str, Any],
) -> httpx.Response:
    api_key = await get_api_key(client)
    async with upstream_semaphore:
        response = await client.post(
            f"{settings.base_url}/{mapper.upstream_path}",
            json=payload,
            headers=mapper.request_headers(api_key),
        )
    if response.status_code == 429:
        raise ExternalApiRetryableStatusError(response)
    if response.status_code >= 500:
        raise ExternalApiRetryableStatusError(response)
    return response


async def get_api_key(client: httpx.AsyncClient) -> str | None:
    if ift_token_provider is not None:
        return await ift_token_provider.get_token(client)
    return settings.api_key


def forward_response(upstream: httpx.Response) -> Response:
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


def merge_request_customization(
    payload: dict[str, Any], customization: Mapping[str, Any]
) -> dict[str, Any]:
    if not customization:
        return payload

    merged = dict(payload)
    for key, value in customization.items():
        current_value = merged.get(key)
        if isinstance(current_value, dict) and isinstance(value, dict):
            merged[key] = merge_request_customization(current_value, value)
        else:
            merged[key] = value
    return merged


def unsupported_response(route: str) -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={"detail": f"{route} is not supported by External API proxy"},
    )


def map_upstream_response(
    *, body: SGLangChatCompletionRequest, upstream: httpx.Response
) -> dict[str, Any]:
    try:
        payload = upstream.json()
        prefer_tool_calls = body.prefer_tool_calls_response()
        return mapper.map_response(
            payload,
            upstream.headers,
            prefer_tool_calls=prefer_tool_calls,
        )
    except (ValueError, ValidationError) as exc:
        raise ExternalApiUnexpectedResponseError(
            f"External API response has unexpected format: {exc}"
        ) from exc


def log_upstream_error(
    *,
    request: Request,
    incoming_body: SGLangChatCompletionRequest,
    upstream_payload: dict[str, Any],
    upstream_response: httpx.Response,
) -> None:
    if upstream_response.status_code < 400:
        return

    event = {
        "event": "upstream_error",
        "route": {
            "method": request.method,
            "path": request.url.path,
        },
        "incoming_request": incoming_body.model_dump(exclude_none=True),
        "upstream_request": {
            "url": f"{settings.base_url}/{mapper.upstream_path}",
            "body": upstream_payload,
        },
        "upstream_response": {
            "status_code": upstream_response.status_code,
            "headers": dict(upstream_response.headers),
            "body": upstream_response.text,
        },
    }
    external_api_logger.error(
        "[external_api_proxy][upstream_error] %s",
        json.dumps(event, ensure_ascii=False, default=str, indent=4),
    )


def log_debug_exchange(
    *,
    request: Request,
    incoming_body: SGLangChatCompletionRequest,
    upstream_payload: dict[str, Any],
    upstream_response: httpx.Response,
    mapped_response: dict[str, Any],
) -> None:
    if not external_api_logger.isEnabledFor(logging.DEBUG):
        return
    try:
        upstream_body: Any = upstream_response.json()
    except ValueError:
        upstream_body = upstream_response.text
    event = {
        "event": "exchange",
        "route": {
            "method": request.method,
            "path": request.url.path,
        },
        "incoming_request": incoming_body.model_dump(exclude_none=True),
        "upstream_request": {
            "url": f"{settings.base_url}/{mapper.upstream_path}",
            "body": upstream_payload,
        },
        "upstream_response": {
            "status_code": upstream_response.status_code,
            "body": upstream_body,
        },
        "outgoing_response": mapped_response,
    }
    external_api_logger.debug(
        "[external_api_proxy][exchange] %s",
        json.dumps(event, ensure_ascii=False, default=str, indent=4),
    )


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request, body: SGLangChatCompletionRequest
) -> Response:
    try:
        upstream_payload = mapper.map_request(body, settings.model)
        upstream_payload = merge_request_customization(
            upstream_payload,
            settings.request_customization,
        )
    except (ValueError, ValidationError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"OpenAI request cannot be converted to upstream format: {exc}",
        ) from exc

    try:
        upstream_response = await post_upstream(
            client=request.app.state.http_client,
            payload=upstream_payload,
        )
    except ExternalApiRetryableStatusError as exc:
        log_upstream_error(
            request=request,
            incoming_body=body,
            upstream_payload=upstream_payload,
            upstream_response=exc.response,
        )
        return forward_response(exc.response)

    if upstream_response.status_code != 200:
        log_upstream_error(
            request=request,
            incoming_body=body,
            upstream_payload=upstream_payload,
            upstream_response=upstream_response,
        )
        return forward_response(upstream_response)

    mapped_response = map_upstream_response(body=body, upstream=upstream_response)
    log_debug_exchange(
        request=request,
        incoming_body=body,
        upstream_payload=upstream_payload,
        upstream_response=upstream_response,
        mapped_response=mapped_response,
    )

    return JSONResponse(
        content=mapped_response,
    )


@app.post("/v1/completions")
async def completions(_: Request) -> JSONResponse:
    return unsupported_response("/v1/completions")


@app.post("/generate")
async def generate(_: Request) -> JSONResponse:
    return unsupported_response("/generate")


@app.get("/ready")
@app.get("/health")
@app.get("/ping")
async def ready() -> JSONResponse:
    return JSONResponse(content={"ready": True})


def main() -> None:
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.logging_level,
    )


if __name__ == "__main__":
    main()
