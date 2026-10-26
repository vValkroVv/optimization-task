from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from typing import Any, List, NamedTuple, Union

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from pam_proxies.models import (
    DevicesV2ChatCompletionRequest,
    OpenAIChatCompletionChoice,
    OpenAIChatCompletionResponse,
    OpenAIChatCompletionUsage,
    SamplingParams,
    SGLangChatCompletionRequest,
    SGLangGenerateResponse,
    TemplateChatCompletionRequest,
)
from ..config import ProxySettings
from ..domain.completion_parsing import parse_function_and_content
from ..domain.devices_v2 import build_v2_response, convert_v2_request_to_openai
from ..domain.reasoning_active import is_reasoning_active
from ..infrastructure.errors import (
    SGLangNonRetryableStatusError,
    SGLangRetryableRequestError,
    SGLangRetryableStatusError,
    client_status_code_for_sglang,
)
from ..infrastructure.sglang_gateway import SGLangGateway
from ..infrastructure.tokenizer_gateway import TokenizerGateway

STOP_TOKEN_IDS = [2]

SGLANG_REQUEST_MAX_ATTEMPTS = 3
SGLANG_REQUEST_RETRY_WAIT_SEC = 1.0
logger = logging.getLogger("obm_proxy")


def _maybe_parse_json_args(arguments: Any) -> Any:
    if not isinstance(arguments, str):
        return arguments
    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, ValueError):
        return arguments
    return parsed if isinstance(parsed, dict) else arguments


class _SGLangExchange(NamedTuple):
    payload: dict[str, Any]
    response: SGLangGenerateResponse


class _CustomGeneration(NamedTuple):
    assistant_message: Any
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    meta_info: dict[str, Any]
    sglang_payload: dict[str, Any]
    sglang_response: SGLangGenerateResponse


class ProxyService:
    def __init__(
        self,
        *,
        settings: ProxySettings,
        tokenizer_gateway: TokenizerGateway,
        sglang_gateway: SGLangGateway,
        dialog_affinity: bool = False,
    ):
        self._settings = settings
        self._tokenizer_gateway = tokenizer_gateway
        self._sglang_gateway = sglang_gateway
        self._dialog_affinity = dialog_affinity

    def preload(self) -> None:
        if self._settings.chat_handlers_mode != "custom":
            return
        self._tokenizer_gateway.get()

    async def handle_completions(self, request: Request) -> Response:
        payload = self._prepare_sglang_payload(await request.json())
        sglang_response = await self._sglang_gateway.proxy_to_sglang(
            client=request.app.state.http_client,
            url=self._settings.completions_url,
            body=payload,
            headers=self._routing_headers(request, payload),
        )
        self._log_sglang_error(
            request=request,
            incoming_body=payload,
            sglang_url=self._settings.completions_url,
            sglang_payload=payload,
            response=sglang_response,
        )
        self._log_exchange_debug(
            request=request,
            incoming_body=payload,
            sglang_url=self._settings.completions_url,
            sglang_payload=payload,
            sglang_response=sglang_response,
            outgoing_response=sglang_response,
        )
        return self._forward_response(sglang_response)

    async def handle_generate(self, request: Request) -> Response:
        payload = self._prepare_sglang_payload(await request.json(), generate=True)
        sglang_response = await self._sglang_gateway.proxy_to_sglang(
            client=request.app.state.http_client,
            url=self._settings.generate_url,
            body=payload,
            headers=self._routing_headers(request, payload),
        )
        self._log_sglang_error(
            request=request,
            incoming_body=payload,
            sglang_url=self._settings.generate_url,
            sglang_payload=payload,
            response=sglang_response,
        )
        self._log_exchange_debug(
            request=request,
            incoming_body=payload,
            sglang_url=self._settings.generate_url,
            sglang_payload=payload,
            sglang_response=sglang_response,
            outgoing_response=sglang_response,
        )
        return self._forward_response(sglang_response)

    async def handle_chat_completions_opensource(
        self, request: Request, body: SGLangChatCompletionRequest
    ) -> Response:
        prepared_body = self._inject_chat_template_kwargs(body)
        prepared_body = self._inject_max_tokens_for_reasoning(prepared_body)
        payload = self._prepare_sglang_payload(
            prepared_body.model_dump(exclude_none=True)
        )
        sglang_response = await self._sglang_gateway.proxy_to_sglang(
            client=request.app.state.http_client,
            url=self._settings.chat_completions_url,
            body=payload,
            headers=self._routing_headers(request, payload, chat=True),
        )
        self._log_sglang_error(
            request=request,
            incoming_body=body,
            sglang_url=self._settings.chat_completions_url,
            sglang_payload=payload,
            response=sglang_response,
        )
        response = self._forward_chat_completions_response(sglang_response)
        self._log_exchange_debug(
            request=request,
            incoming_body=body,
            sglang_url=self._settings.chat_completions_url,
            sglang_payload=payload,
            sglang_response=sglang_response,
            outgoing_response=response,
        )
        return response

    async def handle_chat_completions_custom(
        self,
        request: Request,
        body: SGLangChatCompletionRequest | TemplateChatCompletionRequest,
    ) -> JSONResponse:
        prepared_body = self._inject_chat_template_kwargs(body)
        prepared_body = self._inject_max_tokens_for_reasoning(prepared_body)

        generation = await self._run_custom(
            request=request, prepared_body=prepared_body
        )

        completion_response = OpenAIChatCompletionResponse(
            model=body.model,
            choices=[
                OpenAIChatCompletionChoice(
                    index=0,
                    message=generation.assistant_message,
                    finish_reason=generation.finish_reason,
                )
            ],
            usage=OpenAIChatCompletionUsage(
                prompt_tokens=generation.prompt_tokens,
                completion_tokens=generation.completion_tokens,
                total_tokens=generation.prompt_tokens + generation.completion_tokens,
            ),
            metadata=generation.meta_info,
        )

        content = completion_response.model_dump()
        self._log_exchange_debug(
            request=request,
            incoming_body=body,
            sglang_url=self._settings.generate_url,
            sglang_payload=generation.sglang_payload,
            sglang_response=generation.sglang_response,
            outgoing_response=content,
        )
        return JSONResponse(content=content)

    async def handle_chat_completions_v2(
        self,
        request: Request,
        body: DevicesV2ChatCompletionRequest,
    ) -> JSONResponse:
        openai_body = convert_v2_request_to_openai(body)
        prepared_body = self._inject_chat_template_kwargs(openai_body)
        prepared_body = self._inject_max_tokens_for_reasoning(prepared_body)

        generation = await self._run_custom(
            request=request, prepared_body=prepared_body
        )

        v2_response = build_v2_response(
            generation.assistant_message,
            model=body.model,
            finish_reason=generation.finish_reason,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            created_at=int(time.time()),
        )

        self._log_exchange_debug(
            request=request,
            incoming_body=body,
            sglang_url=self._settings.generate_url,
            sglang_payload=generation.sglang_payload,
            sglang_response=generation.sglang_response,
            outgoing_response=v2_response,
        )
        return JSONResponse(content=v2_response)

    async def _run_custom(
        self,
        *,
        request: Request,
        prepared_body: Union[
            SGLangChatCompletionRequest, TemplateChatCompletionRequest
        ],
    ) -> _CustomGeneration:
        input_ids = self._build_input_ids(body=prepared_body)
        image_data = self._get_image_data(prepared_body)
        audio_data = self._get_audio_data(prepared_body)

        sampling_params = self._build_sampling_params(body=prepared_body)

        exchange = await self._request_sglang(
            request=request,
            incoming_body=prepared_body,
            input_ids=input_ids,
            image_data=image_data,
            audio_data=audio_data,
            sampling_params=sampling_params,
        )
        response = exchange.response

        assistant_message = parse_function_and_content(
            response.text,
            template_version_override=self._settings.template_version_override,
            reasoning_active=is_reasoning_active(
                prepared_body.chat_template_kwargs or {}
            ),
        )
        finish_reason = (
            "function_call"
            if assistant_message.function_call or assistant_message.tool_calls
            else response.meta_info.finish_reason.type
        )

        return _CustomGeneration(
            assistant_message=assistant_message,
            finish_reason=finish_reason,
            prompt_tokens=response.meta_info.prompt_tokens or 0,
            completion_tokens=response.meta_info.completion_tokens or 0,
            meta_info=response.meta_info.model_dump(),
            sglang_payload=exchange.payload,
            sglang_response=response,
        )

    def _build_input_ids(
        self,
        *,
        body: Union[SGLangChatCompletionRequest, TemplateChatCompletionRequest],
    ) -> Union[List[int], List[List[int]]]:
        try:
            messages = [
                self._normalize_chat_message_for_template(
                    message.model_dump(exclude_none=True)
                )
                for message in body.messages
            ]
            tools = (
                [function.model_dump(exclude_none=True) for function in body.functions]
                if body.functions
                else [tool.model_dump(exclude_none=True) for tool in body.tools]
                if body.tools
                else []
            )
            return self._tokenizer_gateway.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
                **(body.chat_template_kwargs or {}),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Failed to apply chat template: {exc}"
            ) from exc

    @classmethod
    def _normalize_chat_message_for_template(
        cls,
        message: dict[str, Any],
    ) -> dict[str, Any]:
        if message.get("role") == "function":
            function_name = str(message.get("name") or "function")
            return {
                "role": "tool",
                "tool_call_id": f"legacy::{function_name}",
                "content": message.get("content", ""),
            }
        if message.get("role") == "assistant":
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                normalized_calls: List[dict[str, Any]] = []
                for call in tool_calls:
                    if not isinstance(call, dict):
                        normalized_calls.append(call)
                        continue
                    fn = call.get("function")
                    if isinstance(fn, dict) and "arguments" in fn:
                        new_fn = {
                            **fn,
                            "arguments": _maybe_parse_json_args(fn["arguments"]),
                        }
                        normalized_calls.append({**call, "function": new_fn})
                    else:
                        normalized_calls.append(call)
                message = {**message, "tool_calls": normalized_calls}
            function_call = message.get("function_call")
            if isinstance(function_call, dict) and "arguments" in function_call:
                message = {
                    **message,
                    "function_call": {
                        **function_call,
                        "arguments": _maybe_parse_json_args(function_call["arguments"]),
                    },
                }
        return message

    def _build_sampling_params(
        self,
        *,
        body: Union[SGLangChatCompletionRequest, TemplateChatCompletionRequest],
    ) -> SamplingParams:
        sampling_params = SamplingParams(
            max_new_tokens=body.max_completion_tokens or body.max_tokens,
            stop_token_ids=body.stop_token_ids
            if body.stop_token_ids is not None
            else STOP_TOKEN_IDS,
            skip_special_tokens=False,
        )
        if body.temperature is not None:
            sampling_params.temperature = float(body.temperature)
        if body.repetition_penalty is not None:
            sampling_params.repetition_penalty = float(body.repetition_penalty)
        if body.top_p is not None:
            sampling_params.top_p = float(body.top_p)
        if body.top_k is not None:
            sampling_params.top_k = int(body.top_k)
        if body.stop is not None:
            sampling_params.stop = body.stop
        return sampling_params

    @staticmethod
    def _resolve_base_max_tokens(
        body: Union[SGLangChatCompletionRequest, TemplateChatCompletionRequest],
    ) -> int:
        if body.max_tokens is not None:
            return body.max_tokens
        if body.max_completion_tokens is not None:
            return body.max_completion_tokens
        return 4096

    @staticmethod
    def _get_image_data(
        body: Union[SGLangChatCompletionRequest, TemplateChatCompletionRequest],
    ) -> Any:
        return getattr(body, "image_data", None)

    @staticmethod
    def _get_audio_data(
        body: Union[SGLangChatCompletionRequest, TemplateChatCompletionRequest],
    ) -> Any:
        return getattr(body, "audio_data", None)

    @retry(
        stop=stop_after_attempt(SGLANG_REQUEST_MAX_ATTEMPTS),
        wait=wait_fixed(SGLANG_REQUEST_RETRY_WAIT_SEC),
        reraise=True,
        retry=retry_if_exception_type(
            (SGLangRetryableRequestError, SGLangRetryableStatusError)
        ),
    )
    async def _request_sglang(
        self,
        *,
        request: Request,
        incoming_body: Union[
            SGLangChatCompletionRequest, TemplateChatCompletionRequest
        ],
        input_ids: Union[List[int], List[List[int]]],
        image_data: Any = None,
        audio_data: Any = None,
        sampling_params: SamplingParams,
    ) -> _SGLangExchange:
        try:
            payload = {
                "input_ids": input_ids,
                "sampling_params": sampling_params.model_dump(
                    exclude_none=True, exclude_unset=True
                ),
            }
            if incoming_body.priority is not None:
                payload["priority"] = incoming_body.priority
            payload = self._prepare_sglang_payload(payload, generate=True)
            if image_data is not None:
                payload["image_data"] = image_data
            if audio_data is not None:
                payload["audio_data"] = audio_data
            response = await self._sglang_gateway.generate(
                client=request.app.state.http_client,
                url=self._settings.generate_url,
                payload=payload,
                headers=self._routing_headers(
                    request, incoming_body.model_dump(exclude_none=True), chat=True
                ),
            )
            self._log_sglang_error(
                request=request,
                incoming_body=incoming_body,
                sglang_url=self._settings.generate_url,
                sglang_payload=payload,
                response=response,
            )
        except Exception as exc:
            raise SGLangRetryableRequestError(str(exc)) from exc

        if response.status_code == 200:
            return _SGLangExchange(
                payload=payload,
                response=SGLangGenerateResponse.model_validate_json(response.content),
            )
        if response.status_code == 429 or response.status_code >= 500:
            raise SGLangRetryableStatusError(
                status_code=response.status_code,
                response_text=response.text,
            )
        raise SGLangNonRetryableStatusError(
            status_code=response.status_code,
            response_text=response.text,
        )

    def _routing_headers(
        self, request: Request, body: dict[str, Any], *, chat: bool = False
    ) -> dict[str, str]:
        if not self._dialog_affinity:
            return {}
        key = body.get("routing_key")
        if isinstance(key, str) and key:
            key = hashlib.sha256(key.encode("utf-8")).hexdigest()
        else:
            key = request.headers.get("x-smg-routing-key")
        if not key:
            session_id = request.headers.get("x-pam-session-id")
            if session_id:
                shooter = request.headers.get("x-pam-shooter", "unknown")
                key = uuid.uuid5(
                    uuid.NAMESPACE_URL, f"pam\0{shooter}\0{session_id}"
                ).hex
        if not key:
            if not chat:
                return {}
            messages = []
            for message in body.get("messages", []):
                messages.append(message)
                if message["role"] == "user":
                    break
            else:
                return {}
            anchor = {
                name: body[name]
                for name in (
                    "model",
                    "tools",
                    "functions",
                    "lora_path",
                    "chat_template_kwargs",
                    "image_data",
                    "audio_data",
                )
                if name in body
            }
            for name in ("tools", "functions"):
                if not anchor.get(name):
                    anchor.pop(name, None)
            anchor["messages"] = messages
            canonical = json.dumps(
                anchor, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
            key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return {"x-smg-routing-key": key}

    @staticmethod
    def _forward_response(upstream: "httpx.Response") -> Response:
        media_type = upstream.headers.get("content-type", "application/json")
        if media_type.split(";")[0].strip() == "application/octet-stream":
            media_type = "application/json"
        return Response(
            content=upstream.content,
            status_code=client_status_code_for_sglang(upstream.status_code),
            media_type=media_type,
        )

    def _log_sglang_error(
        self,
        *,
        request: Request,
        incoming_body: Any,
        sglang_url: str,
        sglang_payload: Any,
        response: "httpx.Response",
    ) -> None:
        if response.status_code < 400 or response.status_code == 503:
            return

        event = {
            "event": "sglang_error",
            "route": self._route_for_log(request),
            "incoming_request": self._to_loggable(incoming_body),
            "sglang_request": {
                "url": sglang_url,
                "body": self._to_loggable(sglang_payload),
            },
            "sglang_response": {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response.text,
            },
        }
        logger.error(
            "[obm_proxy][sglang_error] %s",
            json.dumps(event, ensure_ascii=False, default=str, indent=4),
        )

    def _log_exchange_debug(
        self,
        *,
        request: Request,
        incoming_body: Any,
        sglang_url: str,
        sglang_payload: Any,
        sglang_response: Any,
        outgoing_response: Any,
    ) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return

        event = {
            "event": "exchange",
            "route": self._route_for_log(request),
            "incoming_request": self._to_loggable(incoming_body),
            "sglang_request": {
                "url": sglang_url,
                "body": self._to_loggable(sglang_payload),
            },
            "sglang_response": self._loggable_response(sglang_response),
            "outgoing_response": self._loggable_response(outgoing_response),
        }
        logger.debug(
            "[obm_proxy][exchange] %s",
            json.dumps(event, ensure_ascii=False, default=str, indent=4),
        )

    @staticmethod
    def _route_for_log(request: Request) -> dict[str, Any]:
        return {
            "method": getattr(request, "method", None),
            "path": getattr(getattr(request, "url", None), "path", None),
        }

    @classmethod
    def _loggable_response(cls, value: Any) -> Any:
        if isinstance(value, SGLangGenerateResponse):
            return value.model_dump(exclude_none=True)
        if isinstance(value, httpx.Response):
            try:
                return value.json()
            except ValueError:
                return value.text
        if isinstance(value, Response):
            try:
                return json.loads(value.body)
            except ValueError:
                return value.body.decode("utf-8", errors="replace")
        return value

    @staticmethod
    def _to_loggable(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            return value.model_dump(exclude_none=True)
        return value

    @classmethod
    def _forward_chat_completions_response(cls, upstream: "httpx.Response") -> Response:
        if upstream.status_code != 200:
            return cls._forward_response(upstream)

        payload = upstream.json()
        normalized = OpenAIChatCompletionResponse.model_validate(payload)

        return JSONResponse(
            content=normalized.model_dump(), status_code=upstream.status_code
        )

    def _prepare_sglang_payload(
        self, payload: dict[str, Any], *, generate: bool = False
    ) -> dict[str, Any]:
        if self._settings.priority is not None and payload.get("priority") is None:
            payload = {**payload, "priority": self._settings.priority}
        return self._inject_sampling_params(payload, generate=generate)

    def _inject_sampling_params(
        self, payload: dict[str, Any], *, generate: bool = False
    ) -> dict[str, Any]:
        overrides = self._settings.sampling_params
        if not overrides:
            return payload

        if generate:
            params = payload.get("sampling_params")
            return {
                **payload,
                "sampling_params": [dict(item, **overrides) for item in params]
                if isinstance(params, list)
                else {**(params or {}), **overrides},
            }
        return {**payload, **overrides}

    def _inject_chat_template_kwargs(
        self, body: Union[SGLangChatCompletionRequest, TemplateChatCompletionRequest]
    ) -> Union[SGLangChatCompletionRequest, TemplateChatCompletionRequest]:
        request_kwargs = body.chat_template_kwargs or {}
        merged_kwargs = {
            **self._settings.chat_template_kwargs,
            **request_kwargs,
        }
        return body.model_copy(update={"chat_template_kwargs": merged_kwargs})

    def _inject_max_tokens_for_reasoning(
        self, body: Union[SGLangChatCompletionRequest, TemplateChatCompletionRequest]
    ) -> Union[SGLangChatCompletionRequest, TemplateChatCompletionRequest]:
        chat_template_kwargs = body.chat_template_kwargs or {}

        base_max_tokens = self._resolve_base_max_tokens(body)

        if is_reasoning_active(chat_template_kwargs):
            return body.model_copy(
                update={
                    "max_completion_tokens": self._settings.reasoning_token_budget
                    + base_max_tokens,
                    "max_tokens": None,
                }
            )

        return body.model_copy(
            update={"max_completion_tokens": base_max_tokens, "max_tokens": None}
        )
