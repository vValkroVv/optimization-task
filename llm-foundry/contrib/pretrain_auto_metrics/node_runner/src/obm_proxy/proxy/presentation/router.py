from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from pam_proxies.models import (
    DevicesV2ChatCompletionRequest,
    SGLangChatCompletionRequest,
    TemplateChatCompletionRequest,
)

from ..application.service import ProxyService
from ..config import ProxySettings


def build_router(service: ProxyService, settings: ProxySettings) -> APIRouter:
    router = APIRouter()

    if settings.chat_handlers_mode == "custom":

        @router.post("/v1/chat/completions")
        async def chat_completions_custom(
            request: Request,
            body: SGLangChatCompletionRequest | TemplateChatCompletionRequest,
        ) -> JSONResponse:
            return await service.handle_chat_completions_custom(request, body)

        @router.post("/v2/chat/completions")
        async def chat_completions_v2(
            request: Request,
            body: DevicesV2ChatCompletionRequest,
        ) -> JSONResponse:
            return await service.handle_chat_completions_v2(request, body)
    elif settings.chat_handlers_mode == "opensource":

        @router.post("/v1/chat/completions")
        async def chat_completions_opensource(
            request: Request, body: SGLangChatCompletionRequest
        ) -> Response:
            return await service.handle_chat_completions_opensource(request, body)
    else:

        @router.post("/v1/chat/completions")
        async def chat_completions_pretrain(_: Request) -> JSONResponse:
            return JSONResponse(
                status_code=404,
                content={"detail": "No chat-template-provided"},
            )

    @router.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await service.handle_completions(request)

    @router.post("/generate")
    async def generate(request: Request) -> Response:
        return await service.handle_generate(request)

    @router.get("/ready")
    @router.get("/health")
    @router.get("/ping")
    async def ready(_: Request) -> JSONResponse:
        return JSONResponse(content={"ready": True})

    return router
