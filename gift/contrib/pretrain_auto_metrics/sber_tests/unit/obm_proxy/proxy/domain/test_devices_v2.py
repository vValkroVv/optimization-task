import json
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from node_runner.src.obm_proxy.proxy.domain.devices_v2 import (
    build_v2_response,
    convert_v2_request_to_openai,
)
from node_runner.src.pam_proxies.models import (
    DevicesV2ChatCompletionRequest,
    OpenAIChatCompletionMessage,
    OpenAIToolCall,
    OpenAIToolCallFunction,
)

pytestmark = pytest.mark.unit


class TextPart(BaseModel):
    text: str


class FunctionCall(BaseModel):
    name: str
    arguments: Any


class FunctionCallPart(BaseModel):
    function_call: FunctionCall


class _Message(BaseModel):
    content: list[TextPart | FunctionCallPart]


class _ChatCompletionV2(BaseModel):
    messages: list[_Message]


def _shooters_completion_module():
    try:
        from shooters.models.devices import completion
    except ModuleNotFoundError:
        return SimpleNamespace(ChatCompletionV2=_ChatCompletionV2)
    return completion


def _dump(messages):
    return [m.model_dump(exclude_none=True) for m in messages]


class TestConvertRequest:
    def test_plain_user_system_text_parts_joined(self):
        body = DevicesV2ChatCompletionRequest.model_validate(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": [{"text": "you are"}, {"text": " bot"}],
                    },
                    {"role": "user", "content": [{"text": "hi"}]},
                ]
            }
        )
        msgs = _dump(convert_v2_request_to_openai(body).messages)
        assert msgs[0] == {"role": "system", "content": "you are bot"}
        assert msgs[1] == {"role": "user", "content": "hi"}

    def test_single_function_call_history(self):
        body = DevicesV2ChatCompletionRequest.model_validate(
            {
                "messages": [
                    {"role": "user", "content": [{"text": "weather?"}]},
                    {
                        "role": "assistant",
                        "content": [
                            {"text": "ok"},
                            {
                                "function_call": {
                                    "name": "wx",
                                    "arguments": {"city": "P"},
                                }
                            },
                        ],
                    },
                    {
                        "role": "tool",
                        "content": [
                            {"function_result": {"name": "wx", "result": {"t": 9}}}
                        ],
                    },
                ]
            }
        )
        msgs = _dump(convert_v2_request_to_openai(body).messages)
        assistant = msgs[1]
        assert assistant["content"] == "ok"
        assert len(assistant["tool_calls"]) == 1
        call = assistant["tool_calls"][0]
        assert call["function"] == {"name": "wx", "arguments": {"city": "P"}}
        tool = msgs[2]
        assert tool["role"] == "tool"
        assert tool["tool_call_id"] == call["id"]
        assert json.loads(tool["content"]) == {"t": 9}

    def test_parallel_calls_fifo_matched_with_duplicate_names(self):
        body = DevicesV2ChatCompletionRequest.model_validate(
            {
                "messages": [
                    {"role": "user", "content": [{"text": "two cities"}]},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "function_call": {
                                    "name": "wx",
                                    "arguments": {"c": "Paris"},
                                }
                            },
                            {
                                "function_call": {
                                    "name": "wx",
                                    "arguments": {"c": "London"},
                                }
                            },
                        ],
                    },
                    {
                        "role": "tool",
                        "content": [
                            {"function_result": {"name": "wx", "result": {"t": 20}}},
                            {"function_result": {"name": "wx", "result": {"t": 10}}},
                        ],
                    },
                ]
            }
        )
        msgs = _dump(convert_v2_request_to_openai(body).messages)
        assistant = msgs[1]
        ids = [c["id"] for c in assistant["tool_calls"]]
        assert len(ids) == len(set(ids)) == 2
        # a k-result tool message expands into k tool messages,
        # FIFO-matched positionally (disambiguates duplicate names).
        tool_msgs = msgs[2:]
        assert len(tool_msgs) == 2
        assert [t["tool_call_id"] for t in tool_msgs] == ids
        assert json.loads(tool_msgs[0]["content"]) == {"t": 20}
        assert json.loads(tool_msgs[1]["content"]) == {"t": 10}

    def test_tool_only_assistant_keeps_empty_content(self):
        body = DevicesV2ChatCompletionRequest.model_validate(
            {
                "messages": [
                    {"role": "user", "content": [{"text": "go"}]},
                    {
                        "role": "assistant",
                        "content": [{"function_call": {"name": "f", "arguments": {}}}],
                    },
                ]
            }
        )
        assistant = _dump(convert_v2_request_to_openai(body).messages)[1]
        assert assistant["content"] == ""
        assert len(assistant["tool_calls"]) == 1

    def test_unmatched_function_result_falls_back_to_legacy_id(self):
        body = DevicesV2ChatCompletionRequest.model_validate(
            {
                "messages": [
                    {"role": "user", "content": [{"text": "x"}]},
                    {
                        "role": "tool",
                        "content": [
                            {"function_result": {"name": "orphan", "result": "r"}}
                        ],
                    },
                ]
            }
        )
        tool = _dump(convert_v2_request_to_openai(body).messages)[1]
        assert tool["tool_call_id"] == "legacy::orphan"
        assert tool["content"] == "r"

    def test_functions_kept_flat_with_return_params_and_few_shots(self):
        body = DevicesV2ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": [{"text": "x"}]}],
                "functions": [
                    {
                        "role": "assistant",
                        "name": "f",
                        "description": "d",
                        "parameters": {"type": "object"},
                        "return_parameters": {"type": "object"},
                        "few_shot_examples": [{"q": 1}],
                    }
                ],
            }
        )
        out = convert_v2_request_to_openai(body)
        # tools must stay None so the OpenAI validator (forbids tools+functions)
        # is satisfied; flat function dicts go under `functions`.
        assert out.tools is None
        fn = out.functions[0].model_dump(exclude_none=True)
        assert "role" not in fn
        assert fn["name"] == "f"
        assert fn["return_parameters"] == {"type": "object"}
        assert fn["few_shot_examples"] == [{"q": 1}]

    def test_assistant_text_only_keeps_no_tool_calls(self):
        body = DevicesV2ChatCompletionRequest.model_validate(
            {
                "messages": [
                    {"role": "user", "content": [{"text": "hi"}]},
                    {"role": "assistant", "content": [{"text": "hello"}]},
                ]
            }
        )
        assistant = _dump(convert_v2_request_to_openai(body).messages)[1]
        assert assistant["content"] == "hello"
        assert "tool_calls" not in assistant

    def test_tool_message_without_function_result_part_skipped(self):
        body = DevicesV2ChatCompletionRequest.model_validate(
            {
                "messages": [
                    {"role": "user", "content": [{"text": "x"}]},
                    {"role": "tool", "content": [{"text": "noise"}]},
                ]
            }
        )
        msgs = _dump(convert_v2_request_to_openai(body).messages)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_non_dict_parts_are_ignored(self):
        body = DevicesV2ChatCompletionRequest.model_validate(
            {
                "messages": [
                    {"role": "user", "content": [{"text": "x"}]},
                    {
                        "role": "assistant",
                        "content": [{"function_call": {"name": "f", "arguments": {}}}],
                    },
                    {
                        "role": "tool",
                        "content": [{"function_result": {"name": "f", "result": "r"}}],
                    },
                ]
            }
        )
        # Inject non-dict parts the wire schema would never validate, to exercise
        # the defensive `continue` guards in the adapter.
        body.messages[1].content.insert(0, "junk-assistant-part")
        body.messages[2].content.insert(0, "junk-tool-part")
        msgs = _dump(convert_v2_request_to_openai(body).messages)
        assert len(msgs[1]["tool_calls"]) == 1
        assert msgs[2]["role"] == "tool"
        assert msgs[2]["content"] == "r"

    def test_sampling_fields_carried_through(self):
        body = DevicesV2ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": [{"text": "x"}]}],
                "temperature": 0.5,
                "top_p": 0.9,
                "max_tokens": 128,
                "chat_template_kwargs": {"functions_mode": "parallel"},
            }
        )
        out = convert_v2_request_to_openai(body)
        assert out.temperature == 0.5
        assert out.top_p == 0.9
        assert out.max_tokens == 128
        assert out.chat_template_kwargs == {"functions_mode": "parallel"}


class TestBuildResponse:
    def _msg(self, content="", tool_calls=None):
        return OpenAIChatCompletionMessage(content=content, tool_calls=tool_calls)

    def test_tool_only_response_still_has_text_part(self):
        msg = self._msg(
            content="",
            tool_calls=[
                OpenAIToolCall(
                    id="c0",
                    function=OpenAIToolCallFunction(name="f", arguments='{"a": 1}'),
                )
            ],
        )
        resp = build_v2_response(
            msg,
            model="m",
            finish_reason="function_call",
            prompt_tokens=3,
            completion_tokens=4,
            created_at=7,
        )
        parts = resp["messages"][0]["content"]
        assert parts[0] == {"text": ""}
        assert parts[1] == {"function_call": {"name": "f", "arguments": {"a": 1}}}

    def test_parallel_calls_produce_n_function_call_parts(self):
        msg = self._msg(
            content="sure",
            tool_calls=[
                OpenAIToolCall(
                    id="c0", function=OpenAIToolCallFunction(name="f", arguments="{}")
                ),
                OpenAIToolCall(
                    id="c1", function=OpenAIToolCallFunction(name="g", arguments="{}")
                ),
            ],
        )
        resp = build_v2_response(
            msg,
            model="m",
            finish_reason="function_call",
            prompt_tokens=1,
            completion_tokens=2,
            created_at=0,
        )
        parts = resp["messages"][0]["content"]
        assert parts[0] == {"text": "sure"}
        assert [p["function_call"]["name"] for p in parts[1:]] == ["f", "g"]

    def test_non_json_arguments_pass_through_as_string(self):
        msg = self._msg(
            tool_calls=[
                OpenAIToolCall(
                    id="c0",
                    function=OpenAIToolCallFunction(name="f", arguments="not json"),
                )
            ],
        )
        resp = build_v2_response(
            msg,
            model="m",
            finish_reason="function_call",
            prompt_tokens=0,
            completion_tokens=0,
            created_at=0,
        )
        assert (
            resp["messages"][0]["content"][1]["function_call"]["arguments"]
            == "not json"
        )

    def test_usage_mapping(self):
        resp = build_v2_response(
            self._msg(content="hi"),
            model="m",
            finish_reason="stop",
            prompt_tokens=11,
            completion_tokens=22,
            created_at=99,
        )
        assert resp["created_at"] == 99
        assert resp["finish_reason"] == "stop"
        assert resp["usage"] == {
            "input_tokens": 11,
            "input_tokens_details": {"prompt_tokens": 11, "cached_tokens": 0},
            "output_tokens": 22,
            "total_tokens": 33,
        }

    def test_reasoning_content_is_preserved(self):
        msg = OpenAIChatCompletionMessage(
            content="final answer",
            reasoning_content="private reasoning trace",
        )
        resp = build_v2_response(
            msg,
            model="m",
            finish_reason="stop",
            prompt_tokens=2,
            completion_tokens=3,
            created_at=1,
        )

        assert resp["messages"][0] == {
            "role": "assistant",
            "content": [{"text": "final answer"}],
            "reasoning_content": "private reasoning trace",
        }

    def test_reasoning_content_is_omitted_when_missing(self):
        msg = OpenAIChatCompletionMessage(content="final answer")
        resp = build_v2_response(
            msg,
            model="m",
            finish_reason="stop",
            prompt_tokens=2,
            completion_tokens=3,
            created_at=1,
        )

        assert "reasoning_content" not in resp["messages"][0]


class TestContractWithWheel:
    def test_response_validates_against_chat_completion_v2(self):
        shooters_completion = _shooters_completion_module()
        msg = OpenAIChatCompletionMessage(
            content="",
            tool_calls=[
                OpenAIToolCall(
                    id="c0",
                    function=OpenAIToolCallFunction(name="f", arguments='{"a": 1}'),
                ),
                OpenAIToolCall(
                    id="c1",
                    function=OpenAIToolCallFunction(name="g", arguments="raw"),
                ),
            ],
        )
        resp = build_v2_response(
            msg,
            model="m",
            finish_reason="function_call",
            prompt_tokens=5,
            completion_tokens=7,
            created_at=123,
        )
        validated = shooters_completion.ChatCompletionV2.model_validate(resp)
        assert [type(p).__name__ for p in validated.messages[0].content] == [
            "TextPart",
            "FunctionCallPart",
            "FunctionCallPart",
        ]
