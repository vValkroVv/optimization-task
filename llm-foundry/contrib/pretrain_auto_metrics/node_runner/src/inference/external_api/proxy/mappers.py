from __future__ import annotations

from typing import Any, Mapping

from ....pam_proxies.models import (
    IFTChatCompletionRequest,
    IFTChatCompletionResponse,
    OpenAIChatCompletionResponse,
    SGLangChatCompletionRequest,
)


class OpenAiMapper:
    upstream_path = "v1/chat/completions"

    def request_headers(self, api_key: str | None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = (
                api_key if api_key.startswith("Bearer ") else f"Bearer {api_key}"
            )
        return headers

    def map_request(
        self, request: SGLangChatCompletionRequest, model: str
    ) -> dict[str, Any]:
        return request.to_openai_request_dict(model=model)

    def map_response(
        self,
        payload: dict[str, Any],
        headers: Mapping[str, str],
        prefer_tool_calls: bool,
    ) -> dict[str, Any]:
        return (
            OpenAIChatCompletionResponse.model_validate(payload)
            .to_openai(prefer_tool_calls=prefer_tool_calls)
            .model_dump(exclude_none=True)
        )


class IftMapper:
    upstream_path = "v2/chat/completions"

    def request_headers(self, api_key: str | None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = (
                api_key if api_key.startswith("Bearer ") else f"Bearer {api_key}"
            )
        return headers

    def map_request(
        self, request: SGLangChatCompletionRequest, model: str
    ) -> dict[str, Any]:
        return IFTChatCompletionRequest.from_sglang(
            request, model=model
        ).to_ift_request_dict()

    def map_response(
        self,
        payload: dict[str, Any],
        headers: Mapping[str, str],
        prefer_tool_calls: bool,
    ) -> dict[str, Any]:
        return (
            IFTChatCompletionResponse.model_validate(payload)
            .to_openai(
                headers=headers,
                prefer_tool_calls=prefer_tool_calls,
            )
            .model_dump(exclude_none=True)
        )
