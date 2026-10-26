import pytest

from node_runner.src.pam_proxies.models.ift import (
    IFTChatCompletionRequest,
    IFTChatCompletionResponse,
    IFTContentPart,
    IFTFunctionCall,
    IFTFunctionResult,
    IFTFunctions,
    IFTMessage,
    IFTModelOptions,
    IFTResponseFormat,
    IFTTool,
    IFTToolConfig,
)
from node_runner.src.pam_proxies.models.openai import (
    AssistantMessage,
    FunctionMessage,
    ResponseFormatJSONSchema,
    ResponseFormatText,
    SGLangChatCompletionRequest,
    ToolMessage,
)

pytestmark = pytest.mark.unit


def _request(**overrides) -> SGLangChatCompletionRequest:
    payload = {
        "model": "local-model",
        "messages": [{"role": "user", "content": "hi"}],
    }
    payload.update(overrides)
    return SGLangChatCompletionRequest.model_validate(payload)


class TestIFTContentParts:
    @pytest.mark.parametrize("content", [None, {"type": "text", "text": "ignored"}])
    def test_from_openai_content_returns_empty_for_absent_or_unknown_shape(
        self, content
    ):
        assert IFTContentPart.from_openai_content(content) == []

    def test_from_openai_content_maps_text_and_image_parts(self):
        parts = IFTContentPart.from_openai_content(
            [
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
                {"type": "image_url", "image_url": "https://example.test/image.png"},
            ]
        )

        assert [part.model_dump(exclude_none=True) for part in parts] == [
            {"text": "hello"},
            {"inline_data": {"image_url": "data:image/png;base64,x"}},
            {"inline_data": {"image_url": "https://example.test/image.png"}},
        ]

    def test_from_openai_content_maps_string_content(self):
        assert IFTContentPart.from_openai_content("hello") == [
            IFTContentPart(text="hello")
        ]


class TestIFTFunctionCall:
    def test_parses_arguments_and_serializes_openai_shapes(self):
        call = IFTFunctionCall.model_validate(
            {"id": "call_1", "name": "lookup", "arguments": '{"id": 1}'}
        )

        assert call.arguments == {"id": 1}
        assert call.to_openai() == {
            "name": "lookup",
            "arguments": '{"id": 1}',
        }
        assert call.to_openai_tool_call() == {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"id": 1}'},
        }
        assert (
            IFTFunctionCall(name="fallback").to_openai_tool_call(
                call_id="call_fallback"
            )["id"]
            == "call_fallback"
        )

    @pytest.mark.parametrize("arguments", ["{broken", "[]", 1])
    def test_rejects_non_object_arguments(self, arguments):
        with pytest.raises(ValueError, match="Function arguments"):
            IFTFunctionCall.model_validate({"name": "lookup", "arguments": arguments})

    @pytest.mark.parametrize("value", [None, {}, {"arguments": {}}])
    def test_from_openai_ignores_values_without_function_name(self, value):
        assert IFTFunctionCall.from_openai(value) is None

    def test_from_openai_preserves_call_id(self):
        call = IFTFunctionCall.from_openai(
            {"name": "lookup", "arguments": {"id": 1}},
            call_id="call_1",
        )

        assert call == IFTFunctionCall(id="call_1", name="lookup", arguments={"id": 1})


class TestIFTFunctionResult:
    def test_from_tool_message_maps_tool_and_function_roles(self):
        tool_result = IFTFunctionResult.from_tool_message(
            {"role": "tool", "content": {"ok": True}},
            function_name="lookup",
        )
        function_result = IFTFunctionResult.from_tool_message(
            {"role": "function", "name": "legacy", "content": '{"ok": true}'}
        )

        assert tool_result == IFTFunctionResult(name="lookup", result='{"ok": true}')
        assert function_result == IFTFunctionResult(
            name="legacy",
            result='{"ok": true}',
        )
        assert IFTFunctionResult.from_tool_message({"role": "user"}) is None

    def test_from_tool_message_requires_function_name(self):
        with pytest.raises(ValueError, match="known function"):
            IFTFunctionResult.from_tool_message({"role": "tool", "content": "{}"})

    def test_result_json_wraps_plain_strings(self):
        assert IFTFunctionResult.result_json("plain") == '"plain"'
        assert IFTFunctionResult.result_json(["a"]) == '["a"]'


class TestIFTMessage:
    def test_from_openai_message_maps_assistant_tool_calls_and_drops_empty_text(self):
        message = AssistantMessage(
            role="assistant",
            content=[{"type": "text", "text": ""}],
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"id": 1}'},
                }
            ],
            tools_state_id="state-1",
        )

        mapped = IFTMessage.from_openai_message(message)

        assert mapped.role == "assistant"
        assert mapped.tools_state_id == "state-1"
        assert [part.function_call for part in mapped.content] == [
            IFTFunctionCall(id="call_1", name="lookup", arguments={"id": 1})
        ]

    def test_openai_function_calls_ignores_malformed_tool_calls(self):
        assert IFTMessage.openai_function_calls(
            {
                "tool_calls": [
                    "not-a-dict",
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"arguments": "{}"},
                    },
                    {
                        "id": "call_3",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    },
                ]
            }
        ) == [IFTFunctionCall(id="call_3", name="lookup", arguments={})]

    def test_from_openai_message_maps_legacy_function_call(self):
        mapped = IFTMessage.from_openai_message(
            AssistantMessage(
                role="assistant",
                content="call lookup",
                function_call={"name": "lookup", "arguments": '{"id": 1}'},
            )
        )

        assert mapped.text_content() == "call lookup"
        assert mapped.content_function_call() == IFTFunctionCall(
            name="lookup",
            arguments={"id": 1},
        )

    def test_from_openai_message_maps_tool_and_function_results(self):
        tool = IFTMessage.from_openai_message(
            ToolMessage(role="tool", tool_call_id="call_1", content='{"ok": true}'),
            function_name="lookup",
        )
        function = IFTMessage.from_openai_message(
            FunctionMessage(role="function", name="legacy", content="plain")
        )

        assert tool.role == "tool"
        assert tool.content[0].function_result == IFTFunctionResult(
            name="lookup",
            result='{"ok": true}',
        )
        assert function.content[0].function_result == IFTFunctionResult(
            name="legacy",
            result='"plain"',
        )

    def test_to_openai_assistant_message_supports_tool_and_legacy_shapes(self):
        message = IFTMessage(
            message_id="message-1",
            role="assistant",
            content=[
                IFTContentPart(text="answer"),
                IFTContentPart(
                    function_call=IFTFunctionCall(
                        name="lookup",
                        arguments={"id": 1},
                    )
                ),
            ],
            function_call=IFTFunctionCall(name="primary", arguments={}),
        )

        tool_message = message.to_openai_assistant_message(prefer_tool_calls=True)
        legacy_message = message.to_openai_assistant_message(prefer_tool_calls=False)

        assert tool_message["content"] == "answer"
        assert [call["function"]["name"] for call in tool_message["tool_calls"]] == [
            "primary",
            "lookup",
        ]
        assert tool_message["tool_calls"][0]["id"] == "call_message-1_0"
        assert legacy_message["function_call"] == {
            "name": "primary",
            "arguments": "{}",
        }

    def test_to_openai_assistant_message_without_function_calls_returns_content(self):
        assert IFTMessage(
            role="assistant", content=[IFTContentPart(text="ok")]
        ).to_openai_assistant_message(prefer_tool_calls=True) == {
            "role": "assistant",
            "content": "ok",
        }

    def test_tool_call_id_without_message_id_is_generated(self):
        assert IFTMessage(role="assistant").tool_call_id(0).startswith("call_")


class TestIFTRequestMapping:
    def test_model_options_maps_sampling_reasoning_and_schema(self):
        request = _request(
            temperature=0.2,
            top_p=0.9,
            max_completion_tokens=11,
            max_tokens=5,
            repetition_penalty=1.1,
            logprobs=True,
            top_logprobs=2,
            reasoning_effort="low",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {"type": "object"},
                    "strict": True,
                },
            },
        )

        options = IFTModelOptions.from_sglang(request)

        assert options is not None
        assert options.max_tokens == 11
        assert options.reasoning is not None
        assert options.reasoning.effort == "low"
        assert options.response_format is not None
        assert options.response_format.type == "json_schema"
        assert options.response_format.schema_ == '{"type": "object"}'
        assert options.response_format.strict is True
        assert IFTResponseFormat.from_openai(ResponseFormatText(type="text")) == (
            IFTResponseFormat(type="text")
        )
        assert IFTResponseFormat.from_openai(None) is None
        assert IFTModelOptions.from_sglang(_request()) is None

    def test_functions_and_tool_config_map_tools_functions_and_choices(self):
        tools_request = _request(
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Lookup",
                        "parameters": {"type": "object"},
                    },
                },
                {
                    "type": "custom",
                    "custom": {"name": "ignored"},
                },
            ],
            tool_choice={"type": "function", "function": {"name": "lookup"}},
        )
        functions_request = _request(
            functions=[{"name": "legacy", "parameters": {"type": "object"}}],
            function_call={"name": "legacy"},
        )

        assert IFTFunctions.from_sglang(tools_request).specifications == [
            {
                "name": "lookup",
                "description": "Lookup",
                "parameters": {"type": "object"},
            }
        ]
        assert IFTFunctions.from_sglang(functions_request).specifications == [
            {"name": "legacy", "parameters": {"type": "object"}}
        ]
        assert IFTTool.from_sglang(_request()) is None
        assert IFTTool.from_sglang(tools_request) == IFTTool(
            functions=IFTFunctions(
                specifications=[
                    {
                        "name": "lookup",
                        "description": "Lookup",
                        "parameters": {"type": "object"},
                    }
                ]
            )
        )
        assert IFTToolConfig.from_sglang(_request(tool_choice="auto")) == (
            IFTToolConfig(mode="auto")
        )
        assert IFTToolConfig.from_sglang(_request(tool_choice="none")) == (
            IFTToolConfig(mode="none")
        )
        assert IFTToolConfig.from_sglang(_request(tool_choice="required")) == (
            IFTToolConfig(mode="forced")
        )
        assert IFTToolConfig.from_sglang(tools_request) == IFTToolConfig(
            mode="forced",
            function_name="lookup",
        )
        assert IFTToolConfig.from_sglang(functions_request) == IFTToolConfig(
            mode="forced",
            function_name="legacy",
        )
        assert IFTToolConfig.from_sglang(_request()) is None

    def test_chat_completion_request_maps_messages_and_tool_results(self):
        request = _request(
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"id": 1}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'},
                {"role": "function", "name": "legacy", "content": "plain"},
            ],
            stream=True,
            metadata={"trace": "1"},
            tools=[{"type": "function", "function": {"name": "lookup"}}],
        )

        mapped = IFTChatCompletionRequest.from_sglang(request, model="ift-model")
        dumped = mapped.to_ift_request_dict()

        assert dumped["model"] == "ift-model"
        assert dumped["stream"] is True
        assert dumped["metadata"] == {"trace": "1"}
        assert dumped["messages"][2]["content"][0]["function_result"] == {
            "name": "lookup",
            "result": '{"ok": true}',
        }
        assert dumped["messages"][3]["content"][0]["function_result"] == {
            "name": "legacy",
            "result": '"plain"',
        }
        assert dumped["tools"][0]["functions"]["specifications"] == [{"name": "lookup"}]

    def test_chat_completion_request_maps_tool_without_known_function_name(self):
        request = _request(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "tool", "content": '{"ok": true}'},
            ]
        )

        with pytest.raises(ValueError, match="known function"):
            IFTChatCompletionRequest.messages_from_sglang(request)

    def test_chat_completion_request_handles_tool_message_without_call_id_branch(self):
        request = SGLangChatCompletionRequest.model_construct(
            messages=[
                ToolMessage.model_construct(
                    role="tool",
                    content="{}",
                    tool_call_id=None,
                )
            ]
        )

        with pytest.raises(ValueError, match="known function"):
            IFTChatCompletionRequest.messages_from_sglang(request)

    def test_tool_without_id_resolves_via_legacy_function_call(self):
        request = _request(
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "",
                    "function_call": {"name": "lookup", "arguments": "{}"},
                },
                {"role": "tool", "content": '{"ok": true}'},
            ]
        )

        messages = IFTChatCompletionRequest.messages_from_sglang(request)
        assert messages[2].content[0].function_result.name == "lookup"

    def test_tool_with_unknown_id_resolves_positionally(self):
        request = _request(
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "stale_id", "content": "{}"},
            ]
        )

        messages = IFTChatCompletionRequest.messages_from_sglang(request)
        assert messages[2].content[0].function_result.name == "lookup"

    def test_tool_without_id_skips_calls_already_answered_by_id(self):
        request = _request(
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "first", "arguments": "{}"},
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "second", "arguments": "{}"},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
                {"role": "tool", "content": "{}"},
            ]
        )

        messages = IFTChatCompletionRequest.messages_from_sglang(request)
        assert messages[2].content[0].function_result.name == "first"
        assert messages[3].content[0].function_result.name == "second"

    def test_function_without_name_resolves_via_pending_call(self):
        request = _request(
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "",
                    "function_call": {"name": "lookup", "arguments": "{}"},
                },
                {"role": "function", "content": "plain"},
            ]
        )

        messages = IFTChatCompletionRequest.messages_from_sglang(request)
        assert messages[2].content[0].function_result.name == "lookup"

    def test_named_function_reply_consumes_its_pending_call(self):
        request = _request(
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "",
                    "function_call": {"name": "first", "arguments": "{}"},
                },
                {"role": "function", "name": "first", "content": "plain"},
                {
                    "role": "assistant",
                    "content": "",
                    "function_call": {"name": "second", "arguments": "{}"},
                },
                {"role": "tool", "content": "{}"},
            ]
        )

        messages = IFTChatCompletionRequest.messages_from_sglang(request)
        assert messages[4].content[0].function_result.name == "second"

    def test_collect_tool_call_names_ignores_malformed_entries(self):
        names = {}
        IFTChatCompletionRequest.collect_tool_call_names(
            {
                "tool_calls": [
                    "not-a-dict",
                    {"id": "missing-function"},
                    {"id": "call_1", "function": "not-a-dict"},
                    {"id": "call_2", "function": {"name": "lookup"}},
                ]
            },
            names,
        )

        assert names == {"call_2": "lookup"}

    def test_collect_tool_call_names_ignores_missing_name(self):
        names = {}
        IFTChatCompletionRequest.collect_tool_call_names(
            {"tool_calls": [{"id": "call_1", "function": {}}]},
            names,
        )

        assert names == {}

    def test_collect_tool_call_names_ignores_non_list_tool_calls(self):
        names = {}
        IFTChatCompletionRequest.collect_tool_call_names({"tool_calls": {}}, names)
        assert names == {}

    def test_collect_pending_calls_ignores_malformed_entries(self):
        pending = []
        IFTChatCompletionRequest.collect_pending_calls(
            {
                "function_call": {"arguments": "{}"},
                "tool_calls": [
                    "not-a-dict",
                    {"id": "missing-function"},
                    {"id": "call_1", "function": {}},
                    {"type": "function", "function": {"name": "lookup"}},
                ],
            },
            pending,
        )

        assert pending == [(None, "lookup")]

    def test_consume_pending_call_without_match_keeps_queue(self):
        pending = [("call_1", "lookup")]
        IFTChatCompletionRequest.consume_pending_call(pending, call_id="other")
        assert pending == [("call_1", "lookup")]


class TestIFTResponseMapping:
    def test_response_maps_assistant_message_usage_and_metadata(self):
        response = IFTChatCompletionResponse.model_validate(
            {
                "model": "ift-model",
                "thread_id": "thread-1",
                "messages": [
                    {"role": "user", "content": [{"text": "hi"}]},
                    {
                        "role": "assistant",
                        "message_id": "msg-1",
                        "tools_state_id": "state-1",
                        "content": [
                            {"text": "answer"},
                            {
                                "function_call": {
                                    "name": "lookup",
                                    "arguments": {"id": 1},
                                }
                            },
                        ],
                    },
                ],
                "finish_reason": "function_call",
                "usage": {
                    "input_tokens": 3,
                    "input_tokens_details": {"cached_tokens": 1},
                    "output_tokens": 4,
                    "output_tokens_details": {"reasoning_tokens": 2},
                    "total_tokens": 7,
                },
                "additional_data": {"trace": "1"},
            }
        )

        mapped = response.to_openai(
            headers={"x-request-id": "req-1", "X-Session-ID": "session-1"},
            prefer_tool_calls=True,
        )
        dumped = mapped.model_dump(exclude_none=True)

        assert dumped["choices"][0]["finish_reason"] == "tool_calls"
        assert dumped["choices"][0]["message"]["content"] == "answer"
        assert dumped["choices"][0]["message"]["tool_calls"][0]["function"] == {
            "name": "lookup",
            "arguments": '{"id": 1}',
        }
        assert dumped["usage"] == {
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "total_tokens": 7,
            "prompt_tokens_details": {"cached_tokens": 1},
            "completion_tokens_details": {"reasoning_tokens": 2},
        }
        assert dumped["metadata"]["thread_id"] == "thread-1"
        assert dumped["metadata"]["message_id"] == "msg-1"
        assert dumped["metadata"]["tools_state_id"] == "state-1"
        assert dumped["metadata"]["x_request_id"] == "req-1"
        assert dumped["metadata"]["x_session_id"] == "session-1"
        assert dumped["metadata"]["additional_data"] == {"trace": "1"}

    def test_response_maps_tool_finish_reason_to_legacy_function_call(self):
        response = IFTChatCompletionResponse(
            finish_reason="tool_calls",
            messages=[
                IFTMessage(
                    role="assistant",
                    function_call=IFTFunctionCall(name="lookup", arguments={}),
                )
            ],
        )

        mapped = response.to_openai(headers={}, prefer_tool_calls=False)

        assert mapped.choices[0].finish_reason == "function_call"
        assert mapped.choices[0].message.function_call is not None

    def test_response_without_usage_or_assistant_message(self):
        response = IFTChatCompletionResponse(messages=[IFTMessage(role="user")])

        assert response.openai_usage() == {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "prompt_tokens_details": None,
            "completion_tokens_details": None,
        }
        with pytest.raises(ValueError, match="assistant message"):
            response.assistant_message()

    def test_response_finish_reason_without_function_call_is_preserved(self):
        response = IFTChatCompletionResponse(
            finish_reason="stop",
            messages=[
                IFTMessage(role="assistant", content=[IFTContentPart(text="ok")])
            ],
        )

        assert (
            response.to_openai(headers={}, prefer_tool_calls=True)
            .choices[0]
            .finish_reason
            == "stop"
        )


def test_json_schema_response_format_model_validates_alias():
    fmt = ResponseFormatJSONSchema.model_validate(
        {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {"type": "object"},
                "strict": False,
            },
        }
    )

    assert fmt.json_schema.schema_ == {"type": "object"}
