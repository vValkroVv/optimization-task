from __future__ import annotations

from typing import Any

import httpx

from pam_proxies.models import SGLangChatCompletionRequest


class SGLangGateway:
    async def generate(
        self,
        *,
        client: httpx.AsyncClient,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        responce = await client.post(
            url,
            json=payload,
            headers=headers,
        )
        return responce

    async def proxy_to_sglang(
        self,
        *,
        client: httpx.AsyncClient,
        url: str,
        body: dict[str, Any] | SGLangChatCompletionRequest,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        payload = self._serialize_body(body)
        responce = await client.post(
            url,
            json=payload,
            headers=headers,
        )
        return responce

    @staticmethod
    def _serialize_body(
        body: dict[str, Any] | SGLangChatCompletionRequest,
    ) -> dict[str, Any]:
        if isinstance(body, dict):
            return body
        return body.model_dump(exclude_none=True)
