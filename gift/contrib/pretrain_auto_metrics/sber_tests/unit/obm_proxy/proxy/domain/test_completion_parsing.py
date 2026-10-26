import json
import re

import pytest
from node_runner.src.obm_proxy.proxy.domain.completion_parsing import (
    PARSERS,
    V4CompletionParser,
    V5CompletionParser,
    V41CompletionParser,
    _new_tool_call_id,
    _parse_gcml_invoke_body,
    _parse_v41_fc_json_payload,
    _tool_calls_from_function_call,
    _tool_calls_from_function_calls,
    _try_parse_fc_json,
    detect_template_version,
    parse_function_and_content,
)
from node_runner.src.pam_proxies.models import OpenAILegacyFunctionCall
from node_runner.src.pam_proxies.models.openai import (
    OpenAIChatCompletionMessage,
)

pytestmark = pytest.mark.unit


def _assert_tool_call(
    message: OpenAIChatCompletionMessage,
    index: int = 0,
    *,
    name: str,
    arguments: dict,
):
    # devices-v1 shooter compatibility: every parser mirrors the first tool
    # call into the legacy ``function_call`` field so clients that only read
    # ``function_call`` still see the call.
    assert message.tool_calls
    first = message.tool_calls[0]
    assert message.function_call is not None
    assert message.function_call.name == first.function.name
    assert json.loads(message.function_call.arguments) == json.loads(
        first.function.arguments
    )
    assert message.tool_calls is not None
    call = message.tool_calls[index]
    assert call.type == "function"
    assert call.function.name == name
    assert json.loads(call.function.arguments) == arguments
    assert isinstance(call.id, str) and call.id.startswith("call_")


class TestHelpers:
    def test_try_parse_fc_json_valid(self):
        parsed = _try_parse_fc_json('{"name":"weather","arguments":{"city":"Moscow"}}')
        assert parsed is not None
        assert parsed.name == "weather"
        assert json.loads(parsed.arguments) == {"city": "Moscow"}

    @pytest.mark.parametrize(
        "raw",
        [
            None,
            "",
            "not json",
            "[]",
            '{"name":"f","arguments":1}',
            '{"name":"f"}',
            '{"arguments":{"x":1}}',
        ],
    )
    def test_try_parse_fc_json_invalid(self, raw):
        assert _try_parse_fc_json(raw) is None

    def test_tool_calls_from_function_call_none(self):
        assert _tool_calls_from_function_call(None) is None

    def test_tool_calls_from_function_call_serializes_arguments(self):
        function_call = OpenAILegacyFunctionCall(
            name="weather",
            arguments='{"city": "Москва"}',
        )
        tool_calls = _tool_calls_from_function_call(function_call)
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert json.loads(tool_calls[0].function.arguments) == {"city": "Москва"}
        # v4 / v4.1 single-call path must also produce an OpenAI-compliant id;
        # the openai shooter's Pydantic model rejects ``None``.
        assert tool_calls[0].id and tool_calls[0].id.startswith("call_")

    def test_tool_calls_from_function_calls_none_on_empty(self):
        # v5 helper used by the parser when no ``<｜GCML｜invoke>`` was found.
        assert _tool_calls_from_function_calls([]) is None

    def test_tool_calls_from_function_calls_assigns_unique_ids(self):
        # Each entry must get a distinct ``call_<uuid>`` id so the OBM openai
        # shooter can correlate subsequent ``role: tool`` results.
        calls = [
            OpenAILegacyFunctionCall(name="a", arguments='{"x": 1}'),
            OpenAILegacyFunctionCall(name="b", arguments='{"y": 2}'),
        ]
        result = _tool_calls_from_function_calls(calls)
        assert result is not None and len(result) == 2
        assert result[0].id and result[0].id.startswith("call_")
        assert result[1].id and result[1].id.startswith("call_")
        assert result[0].id != result[1].id
        assert json.loads(result[0].function.arguments) == {"x": 1}
        assert json.loads(result[1].function.arguments) == {"y": 2}

    def test_new_tool_call_id_format_and_uniqueness(self):
        ids = {_new_tool_call_id() for _ in range(64)}
        assert len(ids) == 64
        for cid in ids:
            assert cid.startswith("call_")
            assert len(cid) == len("call_") + 24

    @pytest.mark.parametrize(
        "raw",
        [
            "   ",
            '{"name":"f","arguments":{}} trailing',
        ],
    )
    def test_parse_v41_fc_json_payload_returns_none_on_garbage(self, raw):
        # Whitespace-only payload and JSON with trailing junk both fail the
        # strict ``raw_decode`` + remainder check in ``_parse_v41_fc_json_payload``.
        assert _parse_v41_fc_json_payload(raw) is None

    @pytest.mark.parametrize(
        "raw",
        [
            "[]",
            '[{"name":"f","arguments":{}}]',
            '[{"name":"f","arguments":{}},{"name":"f","arguments":{}}]',
        ],
    )
    def test_parse_v41_fc_json_payload_rejects_json_arrays(self, raw):
        # v4.1 stays single-call: the v4.1 models aren't trained to emit
        # parallel tool calls, so any array-shaped payload is treated as
        # malformed regardless of element validity (parallel is v5-only).
        assert _parse_v41_fc_json_payload(raw) is None

    def test_parse_gcml_invoke_body_keeps_raw_for_malformed_non_string_param(self):
        # ``string="false"`` means we *expect* JSON; on JSONDecodeError we keep
        # the raw text instead of dropping the parameter so downstream metrics
        # can still see the model's intent.
        body = (
            '<\uff5cGCML\uff5cparameter name="x" string="false">{not json'
            "</\uff5cGCML\uff5cparameter>"
        )
        assert _parse_gcml_invoke_body(body) == {"x": "{not json"}


class TestTemplateDetection:
    def test_detect_v41(self):
        assert detect_template_version("x<|function_call|>{}") == "v4.1"

    def test_detect_v4(self):
        assert detect_template_version("function call<|role_sep|>\n{}") == "v4"

    def test_detect_default_v5(self):
        assert detect_template_version("plain text") == "v5"

    def test_v41_marker_has_priority_when_both_markers_present(self):
        completion = "function call<|role_sep|>\n{}<|function_call|>{}"
        assert detect_template_version(completion) == "v4.1"

    @pytest.mark.parametrize(
        "completion",
        [
            "thinking\n<|message_sep|>\n\nassistant<|role_sep|>\n9901",
            "thinking<|message_sep|>\nassistant<|role_sep|>\nanswer<|message_sep|>",
            "thinking<|message_sep|>\nassistant<|role_sep|>\nanswer</s>",
            "dangling separator only<|message_sep|>",
            "dangling role sep only<|role_sep|>",
        ],
        ids=[
            "reasoning-answer-no-terminator",
            "reasoning-answer-message-sep",
            "reasoning-answer-eos",
            "message-sep-only",
            "role-sep-only",
        ],
    )
    def test_detect_v41_for_reasoning_output_without_function_call(self, completion):
        assert detect_template_version(completion) == "v4.1"

    def test_v4_function_call_marker_has_priority_over_bare_separators(self):
        completion = 'function call<|role_sep|>\n{"name":"f","arguments":{}}'
        assert detect_template_version(completion) == "v4"

    # v5 completions also end with ``<|message_sep|>`` / ``</s>`` — the v5
    # markers must keep priority over the bare-separator v4.1 fallback.
    def test_v5_think_marker_has_priority_over_bare_separators(self):
        completion = "thinking</think>answer<|message_sep|>"
        assert detect_template_version(completion) == "v5"

    def test_v5_gcml_marker_has_priority_over_bare_separators(self):
        completion = (
            "<｜GCML｜tool_calls>\n"
            '<｜GCML｜invoke name="f">'
            "</｜GCML｜invoke>\n"
            "</｜GCML｜tool_calls><|message_sep|>"
        )
        assert detect_template_version(completion) == "v5"


class TestV4Parser:
    def setup_method(self):
        self.parser = V4CompletionParser()

    def test_content_only_with_end_of_sequence(self):
        msg = self.parser.parse("Hello</s>")
        assert msg.role == "assistant"
        assert msg.content == "Hello"
        assert msg.reasoning_content == ""
        assert msg.function_call is None
        assert msg.tool_calls is None

    def test_content_without_terminator_returns_raw_content(self):
        msg = self.parser.parse("plain text")
        assert msg.content == "plain text"
        assert msg.reasoning_content == ""
        assert msg.function_call is None
        assert msg.tool_calls is None

    def test_function_call_only(self):
        completion = 'function call<|role_sep|>\n{"name":"weather","arguments":{"city":"Moscow"}}'
        msg = self.parser.parse(completion)
        assert msg.content == ""
        _assert_tool_call(msg, name="weather", arguments={"city": "Moscow"})

    def test_function_call_with_content_prefix_uses_re_content_branch(self):
        completion = (
            'hello</s>function call<|role_sep|>\n{"name":"calc","arguments":{"x":1}}'
        )
        msg = self.parser.parse(completion)
        assert msg.content == "hello"
        _assert_tool_call(msg, name="calc", arguments={"x": 1})

    def test_invalid_function_call_returns_raw_content(self):
        completion = "function call<|role_sep|>\n{broken"
        msg = self.parser.parse(completion)
        assert msg.function_call is None
        assert msg.tool_calls is None
        assert msg.content == completion

    def test_invalid_function_call_schema_returns_raw_content(self):
        completion = 'function call<|role_sep|>\n{"name":"calc","arguments":1}'
        msg = self.parser.parse(completion)
        assert msg.function_call is None
        assert msg.tool_calls is None
        assert msg.content == completion

    def test_reasoning_assistant_content(self):
        completion = (
            "thinking\n"
            "<|message_sep|>\n"
            "assistant<|role_sep|>\n"
            "final answer\n"
            "<|message_sep|>"
        )
        msg = self.parser.parse(completion)
        assert msg.reasoning_content == "thinking"
        assert msg.content == "final answer"
        assert msg.function_call is None
        assert msg.tool_calls is None

    def test_reasoning_assistant_function_call(self):
        completion = (
            "thinking\n"
            "<|message_sep|>\n"
            "assistant<|role_sep|>\n"
            "need a tool\n"
            'function call<|role_sep|>\n{"name":"calc","arguments":{"x":1}}\n'
            "<|message_sep|>"
        )
        msg = self.parser.parse(completion)
        assert msg.reasoning_content == "thinking"
        assert msg.content.strip() == "need a tool"
        _assert_tool_call(msg, name="calc", arguments={"x": 1})

    def test_reasoning_assistant_invalid_function_call_returns_assistant_raw(self):
        completion = (
            "thinking\n"
            "<|message_sep|>\n"
            "assistant<|role_sep|>\n"
            "need a tool\n"
            "function call<|role_sep|>\n"
            "{broken\n"
            "<|message_sep|>"
        )
        msg = self.parser.parse(completion)
        assert msg.reasoning_content == "thinking"
        assert msg.function_call is None
        assert msg.tool_calls is None
        assert msg.content == "need a tool\nfunction call<|role_sep|>\n{broken"

    def test_reasoning_direct_function_call_when_reasoning_active(self):
        completion = (
            "I should call calc\n"
            "<|message_sep|>\n\n"
            'function call<|role_sep|>\n{"name":"calc","arguments":{"x":1}}'
        )
        msg = self.parser.parse(completion, reasoning_active=True)
        assert msg.reasoning_content == "I should call calc"
        assert msg.content == ""
        _assert_tool_call(msg, name="calc", arguments={"x": 1})

    def test_reasoning_active_is_ignored_without_message_separator(self):
        completion = (
            'prefix text function call<|role_sep|>\n{"name":"calc","arguments":{"x":1}}'
        )
        msg = self.parser.parse(completion, reasoning_active=True)
        assert msg.reasoning_content == ""
        assert msg.content == "prefix text "
        _assert_tool_call(msg, name="calc", arguments={"x": 1})


class TestV41Parser:
    def setup_method(self):
        self.parser = V41CompletionParser()

    def test_content_only(self):
        msg = self.parser.parse("hello")
        assert msg.content == "hello"
        assert msg.reasoning_content == ""
        assert msg.function_call is None
        assert msg.tool_calls is None

    def test_content_with_function_call(self):
        completion = (
            "Generating image<|function_call|>"
            '{"name":"text2image","arguments":{"query":"cat"}}'
        )
        msg = self.parser.parse(completion)
        assert msg.content == "Generating image"
        _assert_tool_call(
            msg,
            name="text2image",
            arguments={"query": "cat"},
        )

    def test_reasoning_assistant_function_call(self):
        completion = (
            "Need a tool\n"
            "<|message_sep|>\n"
            "assistant<|role_sep|>\n"
            '<|function_call|>{"name":"calc","arguments":{"x":1}}'
        )
        msg = self.parser.parse(completion)
        assert msg.reasoning_content == "Need a tool"
        assert msg.content == ""
        _assert_tool_call(msg, name="calc", arguments={"x": 1})

    def test_reasoning_assistant_content_only(self):
        completion = (
            "Need to answer\n<|message_sep|>\nassistant<|role_sep|>\nFinal text"
        )
        msg = self.parser.parse(completion)
        assert msg.reasoning_content == "Need to answer"
        assert msg.content == "Final text"
        assert msg.function_call is None
        assert msg.tool_calls is None

    def test_reasoning_assistant_invalid_function_call_returns_assistant_raw(self):
        completion = (
            "Need a tool\n"
            "<|message_sep|>\n"
            "assistant<|role_sep|>\n"
            '<|function_call|>{"name":"calc","arguments":1}'
        )
        msg = self.parser.parse(completion)
        assert msg.reasoning_content == "Need a tool"
        assert msg.content == '<|function_call|>{"name":"calc","arguments":1}'
        assert msg.function_call is None
        assert msg.tool_calls is None

    def test_reasoning_direct_function_call_when_reasoning_active(self):
        completion = (
            'I need to call calc\n<|function_call|>{"name":"calc","arguments":{"x":1}}'
        )
        msg = self.parser.parse(completion, reasoning_active=True)
        assert msg.reasoning_content == "I need to call calc"
        assert msg.content == ""
        _assert_tool_call(msg, name="calc", arguments={"x": 1})

    def test_reasoning_active_with_whitespace_content_keeps_empty_reasoning_content(
        self,
    ):
        completion = '   <|function_call|>{"name":"calc","arguments":{"x":1}}'
        msg = self.parser.parse(completion, reasoning_active=True)
        assert msg.content == ""
        assert msg.reasoning_content == ""
        _assert_tool_call(msg, name="calc", arguments={"x": 1})

    def test_invalid_function_call_json_returns_raw_content(self):
        completion = "hello<|function_call|>{broken json"
        msg = self.parser.parse(completion)
        assert msg.function_call is None
        assert msg.tool_calls is None
        assert msg.content == completion

    def test_invalid_function_call_schema_returns_raw_content(self):
        completion = 'hello<|function_call|>{"name":"search","arguments":1}'
        msg = self.parser.parse(completion)
        assert msg.function_call is None
        assert msg.tool_calls is None
        assert msg.content == completion

    def test_parse_content_fc_returns_none_when_regex_does_not_match(self):
        self.parser._re = re.compile(r"a^")
        assert self.parser._parse_content_fc("anything") is None


class TestAutoDetectParse:
    def test_parse_auto_v4(self):
        completion = 'function call<|role_sep|>\n{"name":"weather","arguments":{"city":"Moscow"}}'
        msg = parse_function_and_content(completion)
        _assert_tool_call(msg, name="weather", arguments={"city": "Moscow"})

    def test_parse_auto_v41(self):
        completion = (
            'calling<|function_call|>{"name":"search","arguments":{"query":"x"}}'
        )
        msg = parse_function_and_content(completion)
        _assert_tool_call(msg, name="search", arguments={"query": "x"})

    def test_parse_with_override(self):
        completion = 'function call<|role_sep|>\n{"name":"f","arguments":{}}'
        msg = parse_function_and_content(completion, template_version_override="v4")
        _assert_tool_call(msg, name="f", arguments={})

    def test_parse_with_reasoning_active_for_v4(self):
        completion = (
            "Need calculator\n"
            "<|message_sep|>\n"
            'function call<|role_sep|>\n{"name":"calc","arguments":{"x":1}}'
        )
        msg = parse_function_and_content(completion, reasoning_active=True)
        assert msg.reasoning_content == "Need calculator"
        assert msg.content == ""
        _assert_tool_call(msg, name="calc", arguments={"x": 1})

    def test_parse_with_unknown_override_raises(self):
        with pytest.raises(KeyError):
            parse_function_and_content("plain", template_version_override="vX")

    @pytest.mark.parametrize("reasoning_active", [True, False])
    def test_parse_auto_v41_reasoning_answer_without_function_call(
        self, reasoning_active
    ):
        completion = (
            "I need to compute 99^2 + 99 + 1. 99^2 is 9801, "
            "9801 + 99 + 1 = 9901."
            "<|message_sep|>\n\n"
            "assistant<|role_sep|>\n"
            "9901"
        )
        msg = parse_function_and_content(completion, reasoning_active=reasoning_active)
        assert msg.content == "9901"
        assert (
            msg.reasoning_content
            == "I need to compute 99^2 + 99 + 1. 99^2 is 9801, 9801 + 99 + 1 = 9901."
        )
        assert msg.function_call is None
        assert msg.tool_calls is None

    @pytest.mark.parametrize("terminator", ["<|message_sep|>", "</s>"])
    def test_parse_auto_v41_reasoning_answer_with_terminator(self, terminator):
        completion = (
            f"thinking hard<|message_sep|>\nassistant<|role_sep|>\nanswer{terminator}"
        )
        msg = parse_function_and_content(completion, reasoning_active=True)
        assert msg.content == "answer"
        assert msg.reasoning_content == "thinking hard"

    def test_parse_auto_v41_multiline_reasoning_and_answer(self):
        completion = (
            "step 1: expand the square\n"
            "step 2: add the terms\n"
            "<|message_sep|>\n\n"
            "assistant<|role_sep|>\n"
            "The answer is:\n\\boxed{9901}"
        )
        msg = parse_function_and_content(completion, reasoning_active=True)
        assert msg.content == "The answer is:\n\\boxed{9901}"
        assert (
            msg.reasoning_content == "step 1: expand the square\nstep 2: add the terms"
        )

    def test_parse_auto_v41_reasoning_then_function_call(self):
        completion = (
            "I should call calc\n"
            "<|message_sep|>\n"
            "assistant<|role_sep|>\n"
            '<|function_call|>{"name":"calc","arguments":{"x":1}}'
        )
        msg = parse_function_and_content(completion, reasoning_active=True)
        assert msg.reasoning_content == "I should call calc"
        assert msg.content == ""
        _assert_tool_call(msg, name="calc", arguments={"x": 1})

    def test_parse_auto_truncated_reasoning_stays_in_reasoning_content(self):
        completion = "partial reasoning that was cut mid-senten"
        msg = parse_function_and_content(completion, reasoning_active=True)
        assert msg.content == ""
        assert msg.reasoning_content == completion
        assert msg.tool_calls is None

    def test_parse_auto_v41_reasoning_only_dangling_separator(self):
        completion = "reasoning that never reached the answer<|message_sep|>"
        msg = parse_function_and_content(completion, reasoning_active=True)
        assert msg.content == "reasoning that never reached the answer"
        assert msg.reasoning_content == ""
        assert msg.tool_calls is None


GCML_OPEN = "<\uff5cGCML\uff5ctool_calls>"
GCML_CLOSE = "</\uff5cGCML\uff5ctool_calls>"


def _gcml_invoke(name: str, params: list[tuple[str, str, bool]]) -> str:
    lines = [f'<\uff5cGCML\uff5cinvoke name="{name}">']
    for pname, pvalue, is_string in params:
        flag = "true" if is_string else "false"
        lines.append(
            f'<\uff5cGCML\uff5cparameter name="{pname}" string="{flag}">{pvalue}'
            f"</\uff5cGCML\uff5cparameter>"
        )
    lines.append("</\uff5cGCML\uff5cinvoke>")
    return "\n".join(lines)


def _gcml_tool_calls(invokes: list[str]) -> str:
    return GCML_OPEN + "\n" + "\n".join(invokes) + "\n" + GCML_CLOSE


def _think(reasoning: str) -> str:
    return f"{reasoning}</think>"


class TestTemplateDetectionV5:
    def test_detects_v5_via_think_marker(self):
        text = f"{_think('the user wants weather')}\nHere you go.<|message_sep|>"
        assert detect_template_version(text) == "v5"

    def test_detects_v5_with_tool_calls(self):
        invoke = _gcml_invoke("get_weather", [("city", "Moscow", True)])
        text = (
            f"{_think('I should call get_weather for Moscow')}\n"
            f"Sure, checking.\n\n{_gcml_tool_calls([invoke])}<|message_sep|>"
        )
        assert detect_template_version(text) == "v5"

    def test_v5_think_marker_takes_priority_over_gcml(self):
        invoke = _gcml_invoke("f", [])
        text = f"{_think('plan')}\n{_gcml_tool_calls([invoke])}<|message_sep|>"
        assert detect_template_version(text) == "v5"

    def test_detects_v5_via_gcml_without_think(self):
        invoke = _gcml_invoke("f", [])
        text = f"Sure.\n{_gcml_tool_calls([invoke])}<|message_sep|>"
        assert detect_template_version(text) == "v5"


class TestV5Parser:
    def setup_method(self):
        self.parser = V5CompletionParser()

    # ---- instant-mode behaviour (no <think>) -------------------------------

    def test_instant_plain_content_message_sep(self):
        msg = self.parser.parse("Just a friendly answer.<|message_sep|>")
        assert msg.tool_calls is None
        assert msg.content == "Just a friendly answer."
        assert msg.reasoning_content == ""

    def test_instant_plain_content_eos(self):
        # v5 final-turn marker is ``</s>``; the parser must strip it the same
        # way it strips ``<|message_sep|>``.
        msg = self.parser.parse("Final answer.</s>")
        assert msg.tool_calls is None
        assert msg.content == "Final answer."
        assert msg.reasoning_content == ""

    def test_instant_tool_call_no_prefix(self):
        invoke = _gcml_invoke("get_weather", [("city", "Moscow", True)])
        msg = self.parser.parse(f"{_gcml_tool_calls([invoke])}<|message_sep|>")
        assert len(msg.tool_calls) == 1
        _assert_tool_call(
            msg,
            0,
            name="get_weather",
            arguments={"city": "Moscow"},
        )
        assert msg.content == ""
        assert msg.reasoning_content == ""

    def test_instant_tool_call_with_text_prefix(self):
        invoke = _gcml_invoke("calc", [("expr", "2+2", True)])
        text = f"Sure, calculating.\n\n{_gcml_tool_calls([invoke])}<|message_sep|>"
        msg = self.parser.parse(text)
        assert len(msg.tool_calls) == 1
        _assert_tool_call(
            msg,
            0,
            name="calc",
            arguments={"expr": "2+2"},
        )
        assert msg.content == "Sure, calculating."
        assert msg.reasoning_content == ""

    def test_instant_parallel_tool_calls(self):
        # v5 + OBM ``openai`` shooter: parallel ``<｜GCML｜invoke>`` entries
        # are surfaced as a multi-element ``tool_calls`` array, each with a
        # unique ``call_<uuid>`` id.
        c1 = _gcml_invoke("expression_calculator", [("query", "2 плюс 3", True)])
        c2 = _gcml_invoke("expression_calculator", [("query", "10 минус 4", True)])
        text = f"Секунду, посчитаю.\n\n{_gcml_tool_calls([c1, c2])}<|message_sep|>"
        msg = self.parser.parse(text)
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 2
        # parallel calls: function_call mirrors only the *first* tool call.
        _assert_tool_call(
            msg,
            0,
            name="expression_calculator",
            arguments={"query": "2 плюс 3"},
        )
        _assert_tool_call(
            msg,
            1,
            name="expression_calculator",
            arguments={"query": "10 минус 4"},
        )
        assert msg.tool_calls[0].id != msg.tool_calls[1].id
        assert msg.content == "Секунду, посчитаю."
        assert msg.reasoning_content == ""

    # ---- reasoner-mode behaviour (with <think>) ----------------------------

    def test_reasoner_think_only(self):
        text = "I need to recall the capital of France.</think>Paris.<|message_sep|>"
        msg = self.parser.parse(text)
        assert msg.tool_calls is None
        assert msg.reasoning_content == "I need to recall the capital of France."
        assert msg.content == "Paris."

    def test_reasoner_think_with_eos(self):
        msg = self.parser.parse("simple addition</think>2+2=4</s>")
        assert msg.reasoning_content == "simple addition"
        assert msg.content == "2+2=4"
        assert msg.tool_calls is None

    def test_reasoner_think_then_tool_call(self):
        invoke = _gcml_invoke("get_weather", [("city", "Moscow", True)])
        text = (
            "The user asks for Moscow weather, I should call get_weather.</think>"
            f"Sure, looking up.\n\n{_gcml_tool_calls([invoke])}<|message_sep|>"
        )
        msg = self.parser.parse(text)
        assert (
            msg.reasoning_content
            == "The user asks for Moscow weather, I should call get_weather."
        )
        assert msg.content == "Sure, looking up."
        assert len(msg.tool_calls) == 1
        _assert_tool_call(
            msg,
            0,
            name="get_weather",
            arguments={"city": "Moscow"},
        )

    def test_reasoner_think_then_parallel_calls(self):
        # v5 parallel + <think>: both invokes surface as separate
        # ``tool_calls`` entries with distinct ids, and the ``<think>`` block
        # is still extracted into ``reasoning_content``.
        invokes = [
            _gcml_invoke("calc", [("expr", "2+3", True)]),
            _gcml_invoke("calc", [("expr", "10-4", True)]),
        ]
        msg = self.parser.parse(
            "Both expressions are simple, I'll compute them in parallel.</think>"
            f"Computing both.\n\n{_gcml_tool_calls(invokes)}<|message_sep|>"
        )
        assert (
            msg.reasoning_content
            == "Both expressions are simple, I'll compute them in parallel."
        )
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 2
        _assert_tool_call(msg, 0, name="calc", arguments={"expr": "2+3"})
        _assert_tool_call(msg, 1, name="calc", arguments={"expr": "10-4"})
        assert msg.tool_calls[0].id != msg.tool_calls[1].id
        assert msg.content == "Computing both."

    def test_reasoner_think_with_multiline_body(self):
        reasoning = "step 1: parse query\nstep 2: pick tool\nstep 3: emit call"
        invoke = _gcml_invoke("search", [("q", "weather", True)])
        text = (
            f"{reasoning}</think>On it.\n\n{_gcml_tool_calls([invoke])}<|message_sep|>"
        )
        msg = self.parser.parse(text)
        assert msg.reasoning_content == reasoning
        assert len(msg.tool_calls) == 1
        _assert_tool_call(
            msg,
            0,
            name="search",
            arguments={"q": "weather"},
        )
        assert msg.content == "On it."

    def test_reasoner_empty_think_block(self):
        msg = self.parser.parse("</think>Hi.<|message_sep|>")
        assert msg.reasoning_content == ""
        assert msg.content == "Hi."

    def test_reasoner_think_no_following_content(self):
        msg = self.parser.parse("just thinking</think><|message_sep|>")
        assert msg.reasoning_content == "just thinking"
        assert msg.content == ""
        assert msg.tool_calls is None

    # ---- robustness --------------------------------------------------------

    def test_unclosed_think_strips_trailing_markers_from_reasoning(self):
        # Truncated reasoning keeps the partial reasoning payload without stop markers.
        msg = self.parser.parse(
            "partial reasoning never closed<|message_sep|>", reasoning_active=True
        )
        assert msg.reasoning_content == "partial reasoning never closed"
        assert msg.content == ""

    def test_truncated_gcml_block_best_effort_parses_completed_invokes(self):
        # Stream cut after ``<｜GCML｜tool_calls>`` but before its closing tag.
        # We still recover the one fully-emitted invoke (best-effort path).
        invoke = _gcml_invoke("get_weather", [("city", "Moscow", True)])
        text = f"Sure.\n\n{GCML_OPEN}\n{invoke}\n"
        msg = self.parser.parse(text)
        assert msg.content == "Sure."
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        _assert_tool_call(
            msg,
            0,
            name="get_weather",
            arguments={"city": "Moscow"},
        )

    def test_gcml_block_with_no_invokes_falls_back_to_plain_content(self):
        # ``<｜GCML｜tool_calls>`` opened but no ``<｜GCML｜invoke>`` inside —
        # treat the whole completion as plain content (no spurious tool call).
        text = f"Hello.\n{GCML_OPEN}\n{GCML_CLOSE}<|message_sep|>"
        msg = self.parser.parse(text)
        assert msg.tool_calls is None
        assert msg.function_call is None
        # Body is preserved (trailing terminator only is stripped).
        assert msg.content.startswith("Hello.")
        assert GCML_OPEN in msg.content

    def test_literal_marker_in_body_is_preserved(self):
        # Trailing-marker stripping must only trim the dangling terminator;
        # a literal ``</s>`` or ``<|message_sep|>`` that the model legitimately
        # mentions inside the answer body must survive verbatim.
        text = "Token </s> separates turns, and <|message_sep|> also.<|message_sep|>"
        msg = self.parser.parse(text)
        assert msg.tool_calls is None
        assert msg.content == "Token </s> separates turns, and <|message_sep|> also."

    # ---- devices-v1 shooter contract (legacy function_call mirror) ---------

    def test_single_tool_call_mirrors_function_call(self):
        # v5 talks to the OBM ``devices`` shooter, which still reads the legacy
        # ``function_call`` field, so the first tool call must be mirrored there
        # in addition to ``tool_calls``.
        invoke = _gcml_invoke("get_weather", [("city", "Moscow", True)])
        msg = self.parser.parse(f"{_gcml_tool_calls([invoke])}<|message_sep|>")
        assert len(msg.tool_calls) == 1
        _assert_tool_call(
            msg,
            0,
            name="get_weather",
            arguments={"city": "Moscow"},
        )

    def test_single_tool_call_with_reasoning_mirrors_function_call(self):
        invoke = _gcml_invoke("calc", [("expr", "2+2", True)])
        text = f"need calc</think>On it.\n\n{_gcml_tool_calls([invoke])}<|message_sep|>"
        msg = self.parser.parse(text)
        _assert_tool_call(
            msg,
            0,
            name="calc",
            arguments={"expr": "2+2"},
        )
        assert msg.reasoning_content == "need calc"

    def test_every_v5_tool_call_has_unique_id(self):
        # ``tool_call.id`` is the openai-protocol anchor for round-trip
        # ``tool_call_id`` correlation in subsequent ``role: tool`` messages.
        invokes = [
            _gcml_invoke("calc", [("expr", "1+1", True)]),
            _gcml_invoke("calc", [("expr", "2+2", True)]),
            _gcml_invoke("calc", [("expr", "3+3", True)]),
        ]
        msg = self.parser.parse(f"{_gcml_tool_calls(invokes)}<|message_sep|>")
        assert msg.tool_calls is not None and len(msg.tool_calls) == 3
        ids = [call.id for call in msg.tool_calls]
        assert all(isinstance(i, str) and i.startswith("call_") for i in ids)
        assert len(set(ids)) == 3

    def test_plain_content_keeps_function_call_none(self):
        msg = self.parser.parse("Just a friendly answer.<|message_sep|>")
        assert msg.tool_calls is None
        assert msg.function_call is None


class TestV5ExtractCalls:
    def _extract(self, source: str):
        return V5CompletionParser._extract_calls(source)

    def test_no_gcml_open_tag_returns_source_as_content(self):
        calls, content = self._extract("Just a friendly answer.<|message_sep|>")
        assert calls == []
        assert content == "Just a friendly answer."

    def test_single_invoke_no_leading_text(self):
        invoke = _gcml_invoke("get_weather", [("city", "Moscow", True)])
        calls, content = self._extract(f"{_gcml_tool_calls([invoke])}<|message_sep|>")
        assert content == ""
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert json.loads(calls[0].arguments) == {"city": "Moscow"}

    def test_single_invoke_with_leading_text_trims_trailing_newlines(self):
        invoke = _gcml_invoke("calc", [("expr", "2+2", True)])
        calls, content = self._extract(
            f"Sure, calculating.\n\n{_gcml_tool_calls([invoke])}<|message_sep|>"
        )
        assert content == "Sure, calculating."
        assert len(calls) == 1

    def test_parallel_invokes_preserved_in_order(self):
        c1 = _gcml_invoke("a", [("x", "1", False)])
        c2 = _gcml_invoke("b", [("y", "2", False)])
        calls, content = self._extract(
            f"Doing both.\n\n{_gcml_tool_calls([c1, c2])}<|message_sep|>"
        )
        assert content == "Doing both."
        assert [c.name for c in calls] == ["a", "b"]
        assert json.loads(calls[0].arguments) == {"x": 1}
        assert json.loads(calls[1].arguments) == {"y": 2}

    def test_empty_block_falls_back_to_plain_content(self):
        source = f"Hello.\n{GCML_OPEN}\n{GCML_CLOSE}<|message_sep|>"
        calls, content = self._extract(source)
        assert calls == []
        assert content.startswith("Hello.")
        assert GCML_OPEN in content
        assert GCML_CLOSE in content

    def test_truncated_stream_without_close_tag_still_reads_invokes(self):
        invoke = _gcml_invoke("get_weather", [("city", "Moscow", True)])
        calls, content = self._extract(f"Sure.\n\n{GCML_OPEN}\n{invoke}\n")
        assert content == "Sure."
        assert len(calls) == 1
        assert calls[0].name == "get_weather"


class TestAutoDetectParseV5:
    def test_registry_has_v5(self):
        assert "v5" in PARSERS
        assert isinstance(PARSERS["v5"], V5CompletionParser)

    def test_override_v5_routes_to_v5_parser(self):
        invoke = _gcml_invoke("f", [("x", "1", False)])
        text = f"plan</think>ok\n\n{_gcml_tool_calls([invoke])}<|message_sep|>"
        msg = parse_function_and_content(text, template_version_override="v5")
        assert msg.reasoning_content == "plan"
        assert msg.content == "ok"
        assert len(msg.tool_calls) == 1
        _assert_tool_call(msg, 0, name="f", arguments={"x": 1})

    def test_autodetect_picks_v5_when_think_present(self):
        msg = parse_function_and_content("r</think>hi<|message_sep|>")
        assert msg.reasoning_content == "r"
        assert msg.content == "hi"
