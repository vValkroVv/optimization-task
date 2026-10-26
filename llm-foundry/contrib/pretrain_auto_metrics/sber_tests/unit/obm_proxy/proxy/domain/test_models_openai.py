import json

import pytest

from node_runner.src.pam_proxies.models.openai import (
    AssistantMessage,
    FunctionMessage,
    OpenAIChatCompletionMessage,
    OpenAIChatCompletionResponse,
    OpenAILegacyFunctionCall,
    OpenAIChatCompletionRequest,
    SGLangChatCompletionRequest,
)

pytestmark = pytest.mark.unit


class TestLegacyFunctionCallSerialization:
    def test_arguments_serialize_as_json_string(self):
        fc = OpenAILegacyFunctionCall(
            name="find_user_id_by_name_zip",
            arguments={"first_name": "Aarav", "zip": "19031"},
        )
        dumped = fc.model_dump()
        assert isinstance(dumped["arguments"], str)
        assert json.loads(dumped["arguments"]) == {
            "first_name": "Aarav",
            "zip": "19031",
        }
        assert isinstance(fc.arguments, str)

    def test_message_dump_keeps_tool_calls_and_function_call_as_strings(self):
        msg = OpenAIChatCompletionMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "f", "arguments": '{"a": 1}'},
                }
            ],
            function_call=OpenAILegacyFunctionCall(name="f", arguments='{"a": 1}'),
        )
        dumped = msg.model_dump()
        assert isinstance(dumped["function_call"]["arguments"], str)
        assert isinstance(dumped["tool_calls"][0]["function"]["arguments"], str)


def _base_request() -> dict:
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
    }


class TestAssistantMessageValidation:
    def test_assistant_message_requires_content_or_tools_or_function_call(self):
        with pytest.raises(
            ValueError,
            match="assistant message must have content, tool_calls, or function_call",
        ):
            AssistantMessage(role="assistant")

    def test_assistant_message_accepts_reasoning_only(self):
        msg = AssistantMessage(role="assistant", reasoning_content="thinking")
        assert msg.reasoning_content == "thinking"
        assert msg.content == ""

    def test_assistant_message_with_tool_calls_defaults_content_to_empty(self):
        msg = AssistantMessage(
            role="assistant",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "ls", "arguments": "{}"},
                }
            ],
        )
        assert msg.content == ""

    def test_assistant_message_preserves_explicit_content(self):
        msg = AssistantMessage(role="assistant", content="hello")
        assert msg.content == "hello"

    def test_function_message_is_accepted(self):
        msg = FunctionMessage(role="function", name="legacy_fn", content="ok")
        assert msg.role == "function"
        assert msg.name == "legacy_fn"
        assert msg.content == "ok"


class TestOpenAIChatCompletionRequestValidation:
    def test_messages_must_not_be_empty(self):
        payload = _base_request()
        payload["messages"] = []
        with pytest.raises(ValueError, match="messages must not be empty"):
            OpenAIChatCompletionRequest.model_validate(payload)

    def test_audio_requires_audio_modality(self):
        payload = _base_request()
        payload["audio"] = {"format": "wav", "voice": "alloy"}
        payload["modalities"] = ["text"]
        with pytest.raises(
            ValueError,
            match="audio is allowed only when modalities contains 'audio'",
        ):
            OpenAIChatCompletionRequest.model_validate(payload)

    def test_audio_is_allowed_when_modalities_include_audio(self):
        payload = _base_request()
        payload["audio"] = {"format": "wav", "voice": "alloy"}
        payload["modalities"] = ["text", "audio"]
        parsed = OpenAIChatCompletionRequest.model_validate(payload)
        assert parsed.audio is not None
        assert parsed.modalities == ["text", "audio"]

    def test_top_logprobs_requires_logprobs_true(self):
        payload = _base_request()
        payload["top_logprobs"] = 3
        payload["logprobs"] = False
        with pytest.raises(ValueError, match="top_logprobs requires logprobs=true"):
            OpenAIChatCompletionRequest.model_validate(payload)

    def test_stream_options_requires_stream_true(self):
        payload = _base_request()
        payload["stream_options"] = {"include_usage": True}
        payload["stream"] = False
        with pytest.raises(
            ValueError, match="stream_options is allowed only when stream=true"
        ):
            OpenAIChatCompletionRequest.model_validate(payload)

    def test_metadata_limit_is_enforced(self):
        payload = _base_request()
        payload["metadata"] = {f"k{i}": "v" for i in range(17)}
        with pytest.raises(
            ValueError, match="metadata supports at most 16 key-value pairs"
        ):
            OpenAIChatCompletionRequest.model_validate(payload)

    def test_tools_and_functions_cannot_be_mixed(self):
        payload = _base_request()
        payload["tools"] = [{"type": "function", "function": {"name": "f"}}]
        payload["functions"] = [{"name": "legacy"}]
        with pytest.raises(
            ValueError, match="do not send both tools and deprecated functions"
        ):
            OpenAIChatCompletionRequest.model_validate(payload)

    def test_tool_choice_and_function_call_cannot_be_mixed(self):
        payload = _base_request()
        payload["tool_choice"] = "auto"
        payload["function_call"] = "auto"
        with pytest.raises(
            ValueError,
            match="do not send both tool_choice and deprecated function_call",
        ):
            OpenAIChatCompletionRequest.model_validate(payload)

    def test_temperature_and_top_p_together_are_allowed(self):
        payload = _base_request()
        payload["temperature"] = 0.2
        payload["top_p"] = 0.9
        parsed = OpenAIChatCompletionRequest.model_validate(payload)
        assert parsed.temperature == 0.2
        assert parsed.top_p == 0.9

    def test_legacy_function_role_message_is_accepted(self):
        payload = _base_request()
        payload["messages"] = [
            {"role": "user", "content": "hi"},
            {"role": "function", "name": "legacy_fn", "content": "{}"},
        ]
        parsed = OpenAIChatCompletionRequest.model_validate(payload)
        assert parsed.messages[1].role == "function"

    def test_tool_message_without_tool_call_id_gets_default(self):
        payload = _base_request()
        payload["messages"] = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": "ok"},
        ]
        parsed = OpenAIChatCompletionRequest.model_validate(payload)
        assert parsed.messages[1].role == "tool"
        assert parsed.messages[1].tool_call_id == "legacy::tool"

    def test_tool_message_with_tool_call_id_preserves_value(self):
        payload = _base_request()
        payload["messages"] = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ]
        parsed = OpenAIChatCompletionRequest.model_validate(payload)
        assert parsed.messages[1].role == "tool"
        assert parsed.messages[1].tool_call_id == "call_1"


class TestSGLangChatCompletionRequestMapping:
    def test_to_openai_request_dict_maps_legacy_functions_to_tools(self):
        request = SGLangChatCompletionRequest.model_validate(
            {
                "model": "ignored",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "developer", "content": [{"type": "text", "text": "dev"}]},
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": "",
                        "function_call": {"name": "lookup", "arguments": "{}"},
                    },
                    {"role": "function", "name": "lookup", "content": '{"ok": true}'},
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": [{"type": "text", "text": "done"}],
                    },
                ],
                "functions": [{"name": "lookup", "parameters": {"type": "object"}}],
                "function_call": {"name": "lookup"},
                "temperature": 0.1,
                "top_k": 20,
            }
        )

        mapped = request.to_openai_request_dict(model="upstream-model")

        assert mapped["model"] == "upstream-model"
        assert mapped["tools"] == [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}},
            }
        ]
        assert mapped["tool_choice"] == {
            "type": "function",
            "function": {"name": "lookup"},
        }
        assert "functions" not in mapped
        assert "function_call" not in mapped
        assert "top_k" not in mapped
        assert [message["role"] for message in mapped["messages"]] == [
            "system",
            "developer",
            "user",
            "assistant",
            "tool",
            "tool",
        ]
        assistant = mapped["messages"][3]
        assert "function_call" not in assistant
        assert assistant["tool_calls"] == [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
        ]
        legacy_result = mapped["messages"][4]
        assert legacy_result == {
            "role": "tool",
            "tool_call_id": "call_0",
            "content": '{"ok": true}',
        }
        native_tool = mapped["messages"][5]
        assert native_tool["tool_call_id"] == "call_1"

    def test_normalize_legacy_history_fifo_matches_parallel_results(self):
        request = SGLangChatCompletionRequest.model_validate(
            {
                "model": "m",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": "",
                        "function_call": {
                            "name": "first",
                            "arguments": {"x": 1},
                        },
                    },
                    {"role": "function", "name": "first", "content": "r1"},
                    {
                        "role": "assistant",
                        "content": "",
                        "function_call": {"name": "second", "arguments": "{}"},
                    },
                    {"role": "function", "name": "second", "content": "r2"},
                ],
                "functions": [{"name": "first"}, {"name": "second"}],
            }
        )

        messages = request.to_openai_request_dict(model="m")["messages"]

        first_call = messages[1]["tool_calls"][0]
        second_call = messages[3]["tool_calls"][0]
        assert first_call["id"] != second_call["id"]
        # dict arguments coming from a legacy shooter are dumped to a string
        assert first_call["function"]["arguments"] == '{"x": 1}'
        assert messages[2] == {
            "role": "tool",
            "tool_call_id": first_call["id"],
            "content": "r1",
        }
        assert messages[4] == {
            "role": "tool",
            "tool_call_id": second_call["id"],
            "content": "r2",
        }

    def test_normalize_legacy_history_unmatched_result_gets_legacy_id(self):
        request = SGLangChatCompletionRequest.model_validate(
            {
                "model": "m",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "function", "name": "orphan", "content": "r"},
                ],
            }
        )

        messages = request.to_openai_request_dict(model="m")["messages"]

        assert messages[1] == {
            "role": "tool",
            "tool_call_id": "legacy::orphan",
            "content": "r",
        }

    def test_normalize_legacy_history_keeps_modern_messages_untouched(self):
        tool_calls = [
            {
                "id": "call_native",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
        ]
        request = SGLangChatCompletionRequest.model_validate(
            {
                "model": "m",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "", "tool_calls": tool_calls},
                    {"role": "tool", "tool_call_id": "call_native", "content": "ok"},
                ],
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
            }
        )

        messages = request.to_openai_request_dict(model="m")["messages"]

        assert messages[1]["tool_calls"] == tool_calls
        assert "function_call" not in messages[1]
        assert messages[2]["tool_call_id"] == "call_native"

    def test_normalize_legacy_history_drops_duplicate_function_call(self):
        request = SGLangChatCompletionRequest.model_validate(
            {
                "model": "m",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_native",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ],
                        "function_call": {"name": "lookup", "arguments": "{}"},
                    },
                ],
            }
        )

        messages = request.to_openai_request_dict(model="m")["messages"]

        assert "function_call" not in messages[1]
        assert messages[1]["tool_calls"][0]["id"] == "call_native"

    def test_to_openai_request_dict_preserves_string_function_call_choice(self):
        request = SGLangChatCompletionRequest.model_validate(
            {
                "model": "ignored",
                "messages": [{"role": "user", "content": "hi"}],
                "function_call": "auto",
            }
        )

        mapped = request.to_openai_request_dict(model="model")

        assert mapped["tool_choice"] == "auto"
        assert "function_call" not in mapped

    def test_prefer_tool_calls_response_uses_tools_without_legacy_functions(self):
        with_tools = SGLangChatCompletionRequest.model_validate(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "f"}}],
            }
        )
        with_functions = SGLangChatCompletionRequest.model_validate(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "functions": [{"name": "f"}],
            }
        )

        assert with_tools.prefer_tool_calls_response() is True
        assert with_functions.prefer_tool_calls_response() is False

    def test_to_openai_request_dict_keeps_native_tools_and_tool_choice(self):
        request = SGLangChatCompletionRequest.model_validate(
            {
                "model": "ignored",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
                "tool_choice": "required",
            }
        )

        mapped = request.to_openai_request_dict(model="upstream-model")

        assert mapped["tools"] == [{"type": "function", "function": {"name": "lookup"}}]
        assert mapped["tool_choice"] == "required"


class TestOpenAIChatCompletionResponseConversion:
    def test_message_converts_legacy_function_call_to_tool_call(self):
        message = OpenAIChatCompletionMessage(
            content=None,
            reasoning_content=None,
            function_call=OpenAILegacyFunctionCall(
                name="lookup",
                arguments='{"id": 1}',
            ),
        )

        converted = message.to_tool_calls_message()

        assert converted.function_call is None
        assert converted.tool_calls is not None
        assert converted.tool_calls[0].function.name == "lookup"
        assert converted.tool_calls[0].function.arguments == '{"id": 1}'
        assert converted.content == ""
        assert converted.reasoning_content == ""

    def test_message_converts_tool_call_to_legacy_function_call(self):
        message = OpenAIChatCompletionMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"id": 1}'},
                }
            ],
        )

        converted = message.to_legacy_function_call_message()

        assert converted.tool_calls is None
        assert converted.function_call is not None
        assert converted.function_call.name == "lookup"
        assert converted.function_call.arguments == '{"id": 1}'

    def test_message_conversion_keeps_existing_protocol_shape(self):
        tool_message = OpenAIChatCompletionMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        )
        legacy_message = OpenAIChatCompletionMessage(
            content="",
            function_call=OpenAILegacyFunctionCall(name="lookup", arguments="{}"),
        )
        plain_message = OpenAIChatCompletionMessage(content="plain")

        assert tool_message.to_tool_calls_message().function_call is None
        assert legacy_message.to_legacy_function_call_message().tool_calls is None
        assert plain_message.to_tool_calls_message().content == "plain"
        assert plain_message.to_legacy_function_call_message().content == "plain"

    def test_legacy_conversion_ignores_non_function_tool_call(self):
        message = OpenAIChatCompletionMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        )
        message.tool_calls[0].type = "custom"

        converted = message.to_legacy_function_call_message()

        assert converted.function_call is None
        assert converted.tool_calls is not None

    def test_response_normalizes_finish_reason_for_requested_protocol(self):
        response = OpenAIChatCompletionResponse.model_validate(
            {
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "function_call": {
                                "name": "lookup",
                                "arguments": '{"id": 1}',
                            },
                        },
                        "finish_reason": "function_call",
                    },
                    {
                        "index": 1,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"id": 2}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    },
                    {
                        "index": 2,
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    },
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            }
        )

        tool_response = response.to_openai(prefer_tool_calls=True)
        legacy_response = response.to_openai(prefer_tool_calls=False)

        assert tool_response.choices[0].finish_reason == "tool_calls"
        assert tool_response.choices[0].message.tool_calls is not None
        assert legacy_response.choices[1].finish_reason == "function_call"
        assert legacy_response.choices[1].message.function_call is not None
        assert tool_response.choices[2].finish_reason == "stop"
