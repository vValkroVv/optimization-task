from __future__ import annotations

import json
import time
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# =========================
# Common helpers
# =========================


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="allow")


# =========================
# Message content parts
# =========================


class TextPart(StrictModel):
    type: Literal["text"]
    text: str


class ImageURL(StrictModel):
    url: str
    detail: Optional[Literal["auto", "low", "high"]] = None


class ImagePart(StrictModel):
    type: Literal["image_url"]
    image_url: ImageURL


class InputAudioData(StrictModel):
    data: str
    format: Literal["wav", "mp3"]


class InputAudioPart(StrictModel):
    type: Literal["input_audio"]
    input_audio: InputAudioData


class RefusalPart(StrictModel):
    type: Literal["refusal"]
    refusal: str


ContentPart = Annotated[
    Union[TextPart, ImagePart, InputAudioPart, RefusalPart],
    Field(discriminator="type"),
]


# =========================
# Messages
# =========================

AssistantContent = Optional[Union[str, List[Union[TextPart, RefusalPart]]]]
UserContent = Union[str, List[Union[TextPart, ImagePart, InputAudioPart]]]
DeveloperSystemContent = Union[str, List[TextPart]]
ToolMessageContent = Union[str, List[TextPart]]


class SystemMessage(StrictModel):
    role: Literal["system"]
    content: DeveloperSystemContent
    name: Optional[str] = None


class DeveloperMessage(StrictModel):
    role: Literal["developer"]
    content: DeveloperSystemContent
    name: Optional[str] = None


class UserMessage(StrictModel):
    role: Literal["user"]
    content: UserContent
    name: Optional[str] = None


class AssistantAudioRef(StrictModel):
    id: str


class AssistantMessage(StrictModel):
    role: Literal["assistant"]
    content: AssistantContent = None
    reasoning_content: Optional[str] = None
    name: Optional[str] = None
    refusal: Optional[str] = None
    audio: Optional[AssistantAudioRef] = None
    tool_calls: Optional[List[dict[str, Any]]] = None

    # deprecated OpenAI field, still accepted by many compat layers
    function_call: Optional[dict[str, Any]] = None

    @model_validator(mode="after")
    def validate_content_or_tools(self) -> "AssistantMessage":
        if (
            self.content is None
            and self.tool_calls is None
            and self.function_call is None
            and self.reasoning_content is None
        ):
            raise ValueError(
                "assistant message must have content, tool_calls, or function_call"
            )
        if self.content is None:
            self.content = ""
        return self


class ToolMessage(StrictModel):
    role: Literal["tool"]
    tool_call_id: Optional[str] = None
    content: ToolMessageContent

    @model_validator(mode="after")
    def ensure_tool_call_id(self) -> "ToolMessage":
        if not self.tool_call_id:
            self.tool_call_id = "legacy::tool"
        return self


class FunctionMessage(StrictModel):
    role: Literal["function"]
    content: ToolMessageContent
    name: Optional[str] = None


ChatMessage = Annotated[
    Union[
        SystemMessage,
        DeveloperMessage,
        UserMessage,
        AssistantMessage,
        FunctionMessage,
        ToolMessage,
    ],
    Field(discriminator="role"),
]


# =========================
# Tools
# =========================


class JSONSchemaObject(StrictModel):
    # deliberately permissive: JSON Schema can be arbitrarily deep
    model_config = ConfigDict(extra="allow")


class FunctionDefinition(StrictModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    strict: Optional[bool] = None


class FunctionTool(StrictModel):
    type: Literal["function"]
    function: FunctionDefinition


class CustomToolDefinition(StrictModel):
    name: str
    description: Optional[str] = None
    format: Optional[Dict[str, Any]] = None


class CustomTool(StrictModel):
    type: Literal["custom"]
    custom: CustomToolDefinition


ToolDef = Annotated[Union[FunctionTool, CustomTool], Field(discriminator="type")]


class NamedFunctionToolChoice(StrictModel):
    type: Literal["function"]
    function: Dict[str, str]  # {"name": "my_function"}


class NamedCustomToolChoice(StrictModel):
    type: Literal["custom"]
    custom: Dict[str, str]  # {"name": "my_custom_tool"}


ToolChoice = Union[
    Literal["none", "auto", "required"],
    NamedFunctionToolChoice,
    NamedCustomToolChoice,
]


# deprecated OpenAI function-calling fields
class DeprecatedFunctionChoiceByName(StrictModel):
    name: str


DeprecatedFunctionCall = Union[Literal["none", "auto"], DeprecatedFunctionChoiceByName]


class DeprecatedFunction(StrictModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


# =========================
# Response format
# =========================


class JSONSchemaSpec(StrictModel):
    name: str
    schema_: Dict[str, Any] = Field(alias="schema")
    strict: Optional[bool] = None


class ResponseFormatText(StrictModel):
    type: Literal["text"]


class ResponseFormatJSONObject(StrictModel):
    type: Literal["json_object"]


class ResponseFormatJSONSchema(StrictModel):
    type: Literal["json_schema"]
    json_schema: JSONSchemaSpec


ResponseFormat = Union[
    ResponseFormatText,
    ResponseFormatJSONObject,
    ResponseFormatJSONSchema,
]


# =========================
# Audio / streaming / web search
# =========================


class AudioOutput(StrictModel):
    format: Literal["wav", "aac", "mp3", "flac", "opus", "pcm16"]
    voice: Union[
        str,
        Literal[
            "alloy",
            "ash",
            "ballad",
            "coral",
            "echo",
            "fable",
            "nova",
            "onyx",
            "sage",
            "shimmer",
            "marin",
            "cedar",
            "verse",
        ],
        Dict[str, str],  # {"id": "voice_1234"}
    ]


class StreamOptions(StrictModel):
    include_usage: Optional[bool] = None


class WebSearchLocationApproximate(StrictModel):
    city: Optional[str] = None
    country: Optional[str] = None
    region: Optional[str] = None
    timezone: Optional[str] = None


class WebSearchLocation(StrictModel):
    type: Literal["approximate"]
    approximate: WebSearchLocationApproximate


class WebSearchOptions(StrictModel):
    user_location: Optional[WebSearchLocation] = None
    search_context_size: Optional[Literal["low", "medium", "high"]] = None


# =========================
# OpenAI request
# =========================


class OpenAIChatCompletionRequest(StrictModel):
    messages: List[ChatMessage]
    model: Optional[str] = None

    # optional standard fields
    audio: Optional[AudioOutput] = None
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, int]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = Field(default=None, ge=0, le=20)

    max_completion_tokens: Optional[int] = Field(default=None, ge=1)
    max_tokens: Optional[int] = Field(default=None, ge=1)  # deprecated

    metadata: Optional[Dict[str, str]] = None
    modalities: Optional[List[Literal["text", "audio"]]] = None
    n: Optional[int] = Field(default=None, ge=1)
    parallel_tool_calls: Optional[bool] = None
    prediction: Optional[Dict[str, Any]] = None
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)

    prompt_cache_key: Optional[str] = None
    prompt_cache_retention: Optional[Literal["in-memory", "24h"]] = None

    reasoning_effort: Optional[
        Literal["none", "minimal", "low", "medium", "high", "xhigh"]
    ] = None

    response_format: Optional[ResponseFormat] = None
    safety_identifier: Optional[str] = Field(default=None, max_length=64)
    seed: Optional[int] = None
    service_tier: Optional[Literal["auto", "default", "flex", "scale", "priority"]] = (
        None
    )

    stop: Optional[Union[str, List[str]]] = None
    store: Optional[bool] = None
    stream: Optional[bool] = None
    stream_options: Optional[StreamOptions] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)

    tool_choice: Optional[ToolChoice] = None
    tools: Optional[List[ToolDef]] = None

    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    user: Optional[str] = None
    web_search_options: Optional[WebSearchOptions] = None

    # deprecated
    function_call: Optional[DeprecatedFunctionCall] = None
    functions: Optional[List[DeprecatedFunction]] = None

    @model_validator(mode="after")
    def validate_request(self) -> "OpenAIChatCompletionRequest":
        if not self.messages:
            raise ValueError("messages must not be empty")

        if self.audio is not None:
            if not self.modalities or "audio" not in self.modalities:
                raise ValueError(
                    "audio is allowed only when modalities contains 'audio'"
                )

        if self.top_logprobs is not None and self.logprobs is not True:
            raise ValueError("top_logprobs requires logprobs=true")

        if self.stream_options is not None and self.stream is not True:
            raise ValueError("stream_options is allowed only when stream=true")

        if self.metadata is not None and len(self.metadata) > 16:
            raise ValueError("metadata supports at most 16 key-value pairs")

        if self.tools is not None and self.functions is not None:
            raise ValueError("do not send both tools and deprecated functions")

        if self.tool_choice is not None and self.function_call is not None:
            raise ValueError(
                "do not send both tool_choice and deprecated function_call"
            )

        if self.temperature is not None and self.top_p is not None:
            # OpenAI says usually change one or the other, not both.
            # Not a protocol error, so we allow it.
            pass

        return self


# =========================
# SGLang extensions
# =========================


class SGLangChatCompletionRequest(OpenAIChatCompletionRequest):
    # OpenAI-compatible + SGLang extras
    return_hidden_states: Optional[bool] = None
    return_routed_experts: Optional[bool] = None
    return_cached_tokens_details: Optional[bool] = None

    # SRT backend-only generation params
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    min_tokens: Optional[int] = Field(default=None, ge=0)
    regex: Optional[str] = None
    ebnf: Optional[str] = None
    repetition_penalty: Optional[float] = None
    stop_token_ids: Optional[List[int]] = None
    stop_regex: Optional[Union[str, List[str]]] = None
    no_stop_trim: Optional[bool] = None
    ignore_eos: Optional[bool] = None
    continue_final_message: Optional[bool] = None
    skip_special_tokens: Optional[bool] = None

    lora_path: Optional[Union[str, List[Optional[str]]]] = None
    session_params: Optional[Dict[str, Any]] = None

    separate_reasoning: Optional[bool] = None
    stream_reasoning: Optional[bool] = None

    chat_template_kwargs: Optional[Dict[str, Any]] = None

    # multimodal tiling
    max_dynamic_patch: Optional[int] = None
    min_dynamic_patch: Optional[int] = None

    # advanced custom sampling
    custom_logit_processor: Optional[Union[str, List[Optional[str]]]] = None
    custom_params: Optional[Dict[str, Any]] = None

    # request routing / ids / caching / PD/DP internals
    rid: Optional[Union[str, List[str]]] = None
    extra_key: Optional[Union[str, List[str]]] = None
    cache_salt: Optional[Union[str, List[str]]] = None
    priority: Optional[int] = None

    bootstrap_host: Optional[Union[str, List[str]]] = None
    bootstrap_port: Optional[Union[int, List[Optional[int]]]] = None
    bootstrap_room: Optional[Union[int, List[int]]] = None

    routed_dp_rank: Optional[int] = None
    disagg_prefill_dp_rank: Optional[int] = None
    data_parallel_rank: Optional[int] = None  # deprecated in SGLang

    # SGLang also normalizes a non-standard nested "reasoning" object
    reasoning: Optional[Dict[str, Any]] = None

    def prefer_tool_calls_response(self) -> bool:
        return bool(self.tools and not self.functions)

    def to_openai_request_dict(self, *, model: str) -> Dict[str, Any]:
        mapped = self.model_dump(
            include=set(OpenAIChatCompletionRequest.model_fields) - {"messages"},
            exclude_none=True,
            by_alias=True,
        )
        mapped["model"] = model
        mapped["messages"] = self._normalize_legacy_history(
            [self._to_openai_message_dict(message) for message in self.messages]
        )
        if mapped.get("tools") is None and mapped.get("functions") is not None:
            mapped["tools"] = [
                {
                    "type": "function",
                    "function": function,
                }
                for function in mapped["functions"]
            ]
        if (
            mapped.get("tool_choice") is None
            and mapped.get("function_call") is not None
        ):
            function_call = mapped["function_call"]
            if isinstance(function_call, str):
                mapped["tool_choice"] = function_call
            else:
                mapped["tool_choice"] = {
                    "type": "function",
                    "function": {"name": function_call.get("name")},
                }
        mapped.pop("functions", None)
        mapped.pop("function_call", None)
        return mapped

    @staticmethod
    def _normalize_legacy_history(
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        pending_call_ids: List[str] = []
        call_counter = 0

        for message in messages:
            role = message.get("role")
            if role == "assistant" and message.get("function_call") is not None:
                message = dict(message)
                function_call = message.pop("function_call")
                if message.get("tool_calls"):
                    normalized.append(message)
                    continue
                arguments = function_call.get("arguments", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                call_id = f"call_{call_counter}"
                call_counter += 1
                pending_call_ids.append(call_id)
                message["tool_calls"] = [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": function_call.get("name"),
                            "arguments": arguments,
                        },
                    }
                ]
            elif role == "function":
                if pending_call_ids:
                    call_id = pending_call_ids.pop(0)
                else:
                    call_id = f"legacy::{message.get('name') or 'function'}"
                message = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": message.get("content", ""),
                }
            normalized.append(message)
        return normalized

    @staticmethod
    def _to_openai_message_dict(message: ChatMessage) -> Dict[str, Any]:
        return message.model_dump(
            include=set(type(message).model_fields),
            exclude_none=True,
            by_alias=True,
        )


class TemplateChatMessage(StrictModel):
    role: str
    content: Optional[Any] = ""


class TemplateToolPayload(StrictModel):
    pass


class TemplateChatCompletionRequest(SGLangChatCompletionRequest):
    messages: List[TemplateChatMessage]

    tools: Optional[List[TemplateToolPayload]] = None
    functions: Optional[List[TemplateToolPayload]] = None

    image_data: Optional[
        Union[
            str,
            Dict[str, Any],
            List[Union[str, Dict[str, Any]]],
            List[List[Union[str, Dict[str, Any]]]],
        ]
    ] = None


class OpenAILegacyFunctionCall(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    arguments: str

    @field_validator("arguments", mode="before")
    @classmethod
    def normalize_arguments(cls, value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)


class OpenAIToolCallFunction(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    arguments: str


class OpenAIToolCall(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    type: Literal["function"] = "function"
    function: OpenAIToolCallFunction


class OpenAIChatCompletionMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None

    function_call: Optional[OpenAILegacyFunctionCall] = None

    @model_validator(mode="after")
    def normalize_nullable_text_fields(self) -> "OpenAIChatCompletionMessage":
        if self.content is None:
            self.content = ""
        if self.reasoning_content is None:
            self.reasoning_content = ""
        return self

    def to_tool_calls_message(self) -> "OpenAIChatCompletionMessage":
        if self.tool_calls is not None:
            return self.model_copy(update={"function_call": None})
        if self.function_call is None:
            return self
        return self.model_copy(
            update={
                "tool_calls": [
                    OpenAIToolCall(
                        function=OpenAIToolCallFunction(
                            name=self.function_call.name,
                            arguments=self.function_call.arguments,
                        )
                    )
                ],
                "function_call": None,
            }
        )

    def to_legacy_function_call_message(self) -> "OpenAIChatCompletionMessage":
        if self.function_call is not None:
            return self.model_copy(update={"tool_calls": None})
        if not self.tool_calls:
            return self
        tool_call = self.tool_calls[0]
        if tool_call.type != "function":
            return self
        return self.model_copy(
            update={
                "tool_calls": None,
                "function_call": OpenAILegacyFunctionCall(
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                ),
            }
        )


class OpenAIChatCompletionChoice(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int
    message: OpenAIChatCompletionMessage
    finish_reason: Optional[str] = None
    matched_stop: Optional[Union[str, int]] = None
    logprobs: Optional[Any] = None


class OpenAIChatCompletionUsage(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    prompt_tokens_details: Optional[Any] = None


class OpenAIChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: Optional[str] = None
    choices: List[OpenAIChatCompletionChoice]
    usage: OpenAIChatCompletionUsage

    metadata: Optional[Dict[str, Any]] = None

    def to_openai(self, *, prefer_tool_calls: bool) -> "OpenAIChatCompletionResponse":
        choices: list[OpenAIChatCompletionChoice] = []
        for choice in self.choices:
            if prefer_tool_calls:
                message = choice.message.to_tool_calls_message()
            else:
                message = choice.message.to_legacy_function_call_message()
            choices.append(
                choice.model_copy(
                    update={
                        "message": message,
                        "finish_reason": self.openai_finish_reason(
                            choice=choice,
                            message=message,
                            prefer_tool_calls=prefer_tool_calls,
                        ),
                    }
                )
            )
        return self.model_copy(update={"choices": choices})

    def openai_finish_reason(
        self,
        *,
        choice: OpenAIChatCompletionChoice,
        message: OpenAIChatCompletionMessage,
        prefer_tool_calls: bool,
    ) -> str | None:
        if message.tool_calls is None and message.function_call is None:
            return choice.finish_reason
        if prefer_tool_calls and choice.finish_reason == "function_call":
            return "tool_calls"
        if not prefer_tool_calls and choice.finish_reason == "tool_calls":
            return "function_call"
        return choice.finish_reason
