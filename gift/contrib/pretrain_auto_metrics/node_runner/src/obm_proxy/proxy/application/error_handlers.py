from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..infrastructure.errors import (
    SGLangNonRetryableStatusError,
    SGLangRetryableRequestError,
    SGLangRetryableStatusError,
    client_status_code_for_sglang,
)

logger = logging.getLogger("obm_proxy")


def register_error_handlers(app: FastAPI, *, sglang_port: int) -> None:
    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        errors = exc.errors()

        logger.warning(
            "422 validation failed: %s %s errors=%s",
            request.method,
            request.url.path,
            errors,
        )

        return JSONResponse(
            status_code=422,
            content={"detail": errors},
        )

    @app.exception_handler(SGLangRetryableStatusError)
    async def handle_sglang_retryable_status_error(
        _: Request,
        exc: SGLangRetryableStatusError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=client_status_code_for_sglang(exc.status_code),
            content={
                "detail": (
                    f"sglang /generate returned {exc.status_code}: {exc.response_text}"
                )
            },
        )

    @app.exception_handler(SGLangRetryableRequestError)
    async def handle_sglang_retryable_request_error(
        _: Request,
        exc: SGLangRetryableRequestError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=502,
            content={
                "detail": f"Failed to reach sglang server on port {sglang_port}: {exc}"
            },
        )

    @app.exception_handler(SGLangNonRetryableStatusError)
    async def handle_sglang_non_retryable_status_error(
        _: Request,
        exc: SGLangNonRetryableStatusError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=client_status_code_for_sglang(exc.status_code),
            content={
                "detail": (
                    f"sglang /generate returned {exc.status_code}: {exc.response_text}"
                )
            },
        )
