from __future__ import annotations

import json
import uuid
from typing import Any, Literal, Mapping, TypeAlias, cast

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from .openai import (
    DeprecatedFunctionChoiceByName,
    FunctionTool,
    NamedFunctionToolChoice,
    OpenAIChatCompletionResponse,
    ResponseFormat,
    ResponseFormatJSONSchema,
    SGLangChatCompletionRequest,
)

JsonValue: TypeAlias = Any
JsonObject: TypeAlias = dict[str, Any]


class IFTBaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class IFTContentPart(IFTBaseModel):
    text: str | None = None
    inline_data: JsonObject | None = None
    files: list[JsonObject] | None = None
    function_call: IFTFunctionCall | None = None
    function_result: IFTFunctionResult | None = None

    @classmethod
    def from_openai_content(cls, content: object) -> list["IFTContentPart"]:
        if content is None:
            return []
        if isinstance(content, str):
            return [cls(text=content)]
        if not isinstance(content, list):
            return []

        parts: list[IFTContentPart] = []
        for part in content:
            data = cast(JsonObject, part)
            if data.get("type") == "text":
                parts.append(cls(text=cast(str, data.get("text", ""))))
            if data.get("type") == "image_url":
                parts.append(cls(inline_data={"image_url": cls.openai_image_url(data)}))
        return parts

    @staticmethod
    def openai_image_url(content_part: JsonObject) -> str:
        image_url = content_part.get("image_url")
        if isinstance(image_url, dict):
            return str(image_url.get("url", ""))
        return str(image_url or "")


class IFTFunctionCall(IFTBaseModel):
    id: str | None = Field(default=None, exclude=True)
    name: str
    arguments: JsonObject = Field(default_factory=dict)

    @field_validator("arguments", mode="before")
    @classmethod
    def parse_arguments(cls, value: object) -> JsonObject:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("Function arguments must be valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("Function arguments must be a JSON object")
        return cast(JsonObject, value)

    @classmethod
    def from_openai(
        cls, value: object, *, call_id: str | None = None
    ) -> "IFTFunctionCall | None":
        if not isinstance(value, dict) or not value.get("name"):
            return None
        return cls(
            id=call_id,
            name=str(value["name"]),
            arguments=value.get("arguments", {}),
        )

    def to_openai(self) -> JsonObject:
        return {
            "name": self.name,
            "arguments": self.arguments_json,
        }

    @property
    def arguments_json(self) -> str:
        return json.dumps(self.arguments, ensure_ascii=False)

    def to_openai_tool_call(self, *, call_id: str | None = None) -> JsonObject:
        return {
            "id": self.id or call_id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments_json,
            },
        }


class IFTFunctionResult(IFTBaseModel):
    name: str
    result: JsonValue

    @classmethod
    def from_tool_message(
        cls, data: JsonObject, *, function_name: str | None = None
    ) -> "IFTFunctionResult | None":
        if data.get("role") not in ("tool", "function"):
            return None
        name = function_name or data.get("name")
        if not name:
            raise ValueError("Tool message does not reference a known function")
        return cls(name=str(name), result=cls.result_json(data.get("content")))

    @staticmethod
    def result_json(value: object) -> str:
        if isinstance(value, str):
            try:
                json.loads(value)
            except json.JSONDecodeError:
                return json.dumps(value, ensure_ascii=False)
            return value
        return json.dumps(value, ensure_ascii=False)


class IFTMessage(IFTBaseModel):
    message_id: str | None = None
    role: str
    content: list[IFTContentPart] = Field(default_factory=list)
    tools_state_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("tools_state_id", "tool_state_id"),
    )
    function_call: IFTFunctionCall | None = None
    function_result: IFTFunctionResult | None = None
    logprobs: list[JsonObject] | None = None

    @classmethod
    def from_openai_message(
        cls, message: BaseModel, *, function_name: str | None = None
    ) -> "IFTMessage":
        data = cast(JsonObject, message.model_dump(exclude_none=True))
        role = str(data["role"])
        if role in ("tool", "function"):
            function_result = IFTFunctionResult.from_tool_message(
                data, function_name=function_name
            )
            return cls(
                role="tool",
                content=[IFTContentPart(function_result=function_result)]
                if function_result
                else [],
            )
        content = IFTContentPart.from_openai_content(data.get("content"))
        function_calls = cls.openai_function_calls(data)
        if function_calls:
            content = [part for part in content if not cls.is_empty_text_part(part)]
            content.extend(
                IFTContentPart(function_call=function_call)
                for function_call in function_calls
            )

        return cls(
            role=role,
            content=content,
            tools_state_id=cast(str | None, data.get("tools_state_id")),
        )

    @classmethod
    def openai_function_calls(cls, message: JsonObject) -> list[IFTFunctionCall]:
        function_call = IFTFunctionCall.from_openai(message.get("function_call"))
        if function_call is not None:
            return [function_call]

        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            return []

        function_calls: list[IFTFunctionCall] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function_call = IFTFunctionCall.from_openai(
                tool_call.get("function"),
                call_id=(
                    str(tool_call["id"]) if tool_call.get("id") is not None else None
                ),
            )
            if function_call is not None:
                function_calls.append(function_call)
        return function_calls

    @staticmethod
    def is_empty_text_part(part: IFTContentPart) -> bool:
        return (
            part.text == ""
            and part.inline_data is None
            and part.files is None
            and part.function_call is None
            and part.function_result is None
        )

    def text_content(self) -> str:
        return "".join(part.text or "" for part in self.content)

    def to_openai_assistant_message(self, *, prefer_tool_calls: bool) -> JsonObject:
        function_calls = self.function_calls()
        message: JsonObject = {
            "role": "assistant",
            "content": self.text_content(),
        }
        if not function_calls:
            return message
        if prefer_tool_calls:
            message["tool_calls"] = [
                function_call.to_openai_tool_call(call_id=self.tool_call_id(index))
                for index, function_call in enumerate(function_calls)
            ]
        else:
            message["function_call"] = function_calls[0].to_openai()
        return message

    def content_function_call(self) -> IFTFunctionCall | None:
        function_calls = self.content_function_calls()
        return function_calls[0] if function_calls else None

    def content_function_calls(self) -> list[IFTFunctionCall]:
        function_calls: list[IFTFunctionCall] = []
        for part in self.content:
            if part.function_call is not None:
                function_calls.append(part.function_call)
        return function_calls

    def function_calls(self) -> list[IFTFunctionCall]:
        function_calls = self.content_function_calls()
        if self.function_call is not None:
            return [self.function_call, *function_calls]
        return function_calls

    def tool_call_id(self, index: int) -> str:
        if self.message_id:
            return f"call_{self.message_id}_{index}"
        return f"call_{uuid.uuid4().hex}_{index}"


class IFTResponseFormat(IFTBaseModel):
    type: str
    schema_: str | None = Field(default=None, alias="schema")
    strict: bool | None = None

    @classmethod
    def from_openai(cls, value: ResponseFormat | None) -> "IFTResponseFormat | None":
        if value is None:
            return None
        if isinstance(value, ResponseFormatJSONSchema):
            return cls(
                type=value.type,
                schema=json.dumps(value.json_schema.schema_, ensure_ascii=False),
                strict=value.json_schema.strict,
            )
        return cls(type=value.type)


class IFTReasoningOptions(IFTBaseModel):
    effort: str


class IFTModelOptions(IFTBaseModel):
    preset: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    repetition_penalty: float | None = None
    top_logprobs: int | None = None
    reasoning: IFTReasoningOptions | None = None
    response_format: IFTResponseFormat | None = None

    @classmethod
    def from_sglang(
        cls, request: SGLangChatCompletionRequest
    ) -> "IFTModelOptions | None":
        options = cls(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_completion_tokens or request.max_tokens,
            repetition_penalty=request.repetition_penalty,
            top_logprobs=request.top_logprobs,
            reasoning=IFTReasoningOptions(effort=request.reasoning_effort)
            if request.reasoning_effort
            else None,
            response_format=IFTResponseFormat.from_openai(request.response_format),
        )
        return options if options.model_dump(exclude_none=True) else None


class IFTFunctions(IFTBaseModel):
    specifications: list[JsonObject]

    @classmethod
    def from_sglang(cls, request: SGLangChatCompletionRequest) -> "IFTFunctions | None":
        specifications: list[JsonObject] = []
        if request.tools:
            specifications.extend(
                cast(JsonObject, tool.function.model_dump(exclude_none=True))
                for tool in request.tools
                if isinstance(tool, FunctionTool)
            )
        elif request.functions:
            specifications.extend(
                cast(JsonObject, function.model_dump(exclude_none=True))
                for function in request.functions
            )
        return cls(specifications=specifications) if specifications else None


class IFTTool(IFTBaseModel):
    functions: IFTFunctions | None = None

    @classmethod
    def from_sglang(cls, request: SGLangChatCompletionRequest) -> "IFTTool | None":
        functions = IFTFunctions.from_sglang(request)
        return cls(functions=functions) if functions else None


class IFTToolConfig(IFTBaseModel):
    mode: Literal["auto", "none", "forced"]
    tool_name: str | None = None
    function_name: str | None = None

    @classmethod
    def from_sglang(
        cls, request: SGLangChatCompletionRequest
    ) -> "IFTToolConfig | None":
        if request.tool_choice in ("auto", "none"):
            return cls(mode=request.tool_choice)
        if request.tool_choice == "required":
            return cls(mode="forced")
        if isinstance(request.tool_choice, NamedFunctionToolChoice):
            return cls(
                mode="forced", function_name=request.tool_choice.function["name"]
            )
        if isinstance(request.function_call, DeprecatedFunctionChoiceByName):
            return cls(mode="forced", function_name=request.function_call.name)
        return None


class IFTChatCompletionRequest(IFTBaseModel):
    model: str
    messages: list[IFTMessage]
    model_options: IFTModelOptions | None = None
    tool_config: IFTToolConfig | None = None
    stream: bool | None = None
    update_interval: float | None = None
    unnormalized_history: bool | None = None
    disable_filter: bool = True
    filter_config: JsonObject | None = None
    flags: list[str] | None = None
    storage: JsonObject | bool | None = None
    limit: int | None = None
    thread_id: str | None = None
    metadata: JsonObject | None = None
    ranker_options: JsonObject | None = None
    tools: list[IFTTool] | None = None

    @classmethod
    def from_sglang(
        cls, request: SGLangChatCompletionRequest, *, model: str
    ) -> "IFTChatCompletionRequest":
        return cls(
            model=model,
            messages=cls.messages_from_sglang(request),
            model_options=IFTModelOptions.from_sglang(request),
            tool_config=IFTToolConfig.from_sglang(request),
            stream=request.stream,
            metadata=cast(JsonObject | None, request.metadata),
            tools=cls.tools_from_sglang(request),
        )

    def to_ift_request_dict(self) -> JsonObject:
        return cast(
            JsonObject,
            self.model_dump(exclude_none=True, by_alias=True),
        )

    @staticmethod
    def messages_from_sglang(request: SGLangChatCompletionRequest) -> list[IFTMessage]:
        tool_call_names: dict[str, str] = {}
        pending_calls: list[tuple[str | None, str]] = []
        messages: list[IFTMessage] = []

        for message in request.messages:
            data = cast(JsonObject, message.model_dump(exclude_none=True))
            role = str(data["role"])
            function_name: str | None = None

            if role == "assistant":
                IFTChatCompletionRequest.collect_tool_call_names(data, tool_call_names)
                IFTChatCompletionRequest.collect_pending_calls(data, pending_calls)
            elif role == "tool":
                tool_call_id = data.get("tool_call_id")
                if tool_call_id is not None:
                    function_name = tool_call_names.get(str(tool_call_id))
                if function_name is None:
                    # Legacy histories carry tool results without a resolvable
                    # id: treat them as answering the oldest unanswered call.
                    function_name = IFTChatCompletionRequest.pop_pending_call(
                        pending_calls
                    )
                else:
                    IFTChatCompletionRequest.consume_pending_call(
                        pending_calls, call_id=str(tool_call_id)
                    )
            elif role == "function":
                name = data.get("name")
                function_name = str(name) if name is not None else None
                if function_name is None:
                    function_name = IFTChatCompletionRequest.pop_pending_call(
                        pending_calls
                    )
                else:
                    IFTChatCompletionRequest.consume_pending_call(
                        pending_calls, name=function_name
                    )

            messages.append(
                IFTMessage.from_openai_message(
                    message,
                    function_name=function_name,
                )
            )

        return messages

    @staticmethod
    def collect_tool_call_names(
        message: JsonObject, tool_call_names: dict[str, str]
    ) -> None:
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            return
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_call_id = tool_call.get("id")
            function = tool_call.get("function")
            if tool_call_id is None or not isinstance(function, dict):
                continue
            name = function.get("name")
            if name:
                tool_call_names[str(tool_call_id)] = str(name)

    @staticmethod
    def collect_pending_calls(
        message: JsonObject, pending_calls: list[tuple[str | None, str]]
    ) -> None:
        function_call = message.get("function_call")
        if isinstance(function_call, dict) and function_call.get("name"):
            pending_calls.append((None, str(function_call["name"])))
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            return
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict) or not function.get("name"):
                continue
            tool_call_id = tool_call.get("id")
            pending_calls.append(
                (
                    str(tool_call_id) if tool_call_id is not None else None,
                    str(function["name"]),
                )
            )

    @staticmethod
    def pop_pending_call(
        pending_calls: list[tuple[str | None, str]],
    ) -> str | None:
        if pending_calls:
            return pending_calls.pop(0)[1]
        return None

    @staticmethod
    def consume_pending_call(
        pending_calls: list[tuple[str | None, str]],
        *,
        call_id: str | None = None,
        name: str | None = None,
    ) -> None:
        for index, (pending_id, pending_name) in enumerate(pending_calls):
            if (call_id is not None and pending_id == call_id) or (
                call_id is None and pending_name == name
            ):
                del pending_calls[index]
                return

    @staticmethod
    def tools_from_sglang(request: SGLangChatCompletionRequest) -> list[IFTTool] | None:
        tool = IFTTool.from_sglang(request)
        return [tool] if tool else None


class IFTUsage(IFTBaseModel):
    input_tokens: int | None = None
    input_tokens_details: JsonObject | None = None
    output_tokens: int | None = None
    output_tokens_details: JsonObject | None = None
    total_tokens: int | None = None


class IFTChatCompletionResponse(IFTBaseModel):
    model: str | None = None
    thread_id: str | None = None
    created_at: int | None = None
    messages: list[IFTMessage] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: IFTUsage | None = None
    additional_data: JsonObject | None = None

    def to_openai(
        self, *, headers: Mapping[str, str], prefer_tool_calls: bool
    ) -> OpenAIChatCompletionResponse:
        message = self.assistant_message()

        return OpenAIChatCompletionResponse.model_validate(
            {
                "model": self.model,
                "choices": [
                    {
                        "index": 0,
                        "message": message.to_openai_assistant_message(
                            prefer_tool_calls=prefer_tool_calls,
                        ),
                        "finish_reason": self.openai_finish_reason(
                            message=message,
                            prefer_tool_calls=prefer_tool_calls,
                        ),
                        "logprobs": message.logprobs,
                    }
                ],
                "usage": self.openai_usage(),
                "metadata": self.openai_metadata(headers=headers, message=message),
            }
        )

    def assistant_message(self) -> IFTMessage:
        for message in reversed(self.messages):
            if message.role == "assistant":
                return message
        raise ValueError("IFT response does not contain an assistant message")

    def openai_finish_reason(
        self, *, message: IFTMessage, prefer_tool_calls: bool
    ) -> str | None:
        function_call = message.function_call or message.content_function_call()
        if function_call is None:
            return self.finish_reason
        if prefer_tool_calls and self.finish_reason == "function_call":
            return "tool_calls"
        if not prefer_tool_calls and self.finish_reason == "tool_calls":
            return "function_call"
        return self.finish_reason

    def openai_usage(self) -> JsonObject:
        usage = self.usage
        return {
            "prompt_tokens": usage.input_tokens if usage else None,
            "completion_tokens": usage.output_tokens if usage else None,
            "total_tokens": usage.total_tokens if usage else None,
            "prompt_tokens_details": usage.input_tokens_details if usage else None,
            "completion_tokens_details": usage.output_tokens_details if usage else None,
        }

    def openai_metadata(
        self, *, headers: Mapping[str, str], message: IFTMessage
    ) -> JsonObject:
        content_parts = message.model_dump(exclude_none=True).get("content")
        return {
            key: value
            for key, value in {
                "thread_id": self.thread_id,
                "message_id": message.message_id,
                "tools_state_id": message.tools_state_id,
                "ift_content": content_parts,
                "additional_data": self.additional_data,
                "x_request_id": headers.get("X-Request-ID")
                or headers.get("x-request-id"),
                "x_session_id": headers.get("X-Session-ID")
                or headers.get("x-session-id"),
            }.items()
            if value is not None
        }
