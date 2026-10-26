from __future__ import annotations

import json
import re
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pam_proxies.models import (
    OpenAIChatCompletionMessage,
    OpenAILegacyFunctionCall,
    OpenAIToolCall,
    OpenAIToolCallFunction,
)

REGEX_REASONING_SPLIT = re.compile(
    r"^(?P<reasoning>.*?)"
    r"<\|message_sep\|>\s*assistant<\|role_sep\|>\s*"
    r"(?P<assistant>.*?)"
    r"(?:<\|message_sep\|>|</s>|$)",
    re.DOTALL,
)

_GCML_OPEN = "<\uff5cGCML\uff5ctool_calls>"
_GCML_CLOSE = "</\uff5cGCML\uff5ctool_calls>"
_GCML_INVOKE_RE = re.compile(
    r"<\uff5cGCML\uff5cinvoke\s+name=\"(?P<name>[^\"]*)\"\s*>"
    r"(?P<body>.*?)"
    r"</\uff5cGCML\uff5cinvoke>",
    re.DOTALL,
)
_GCML_PARAM_RE = re.compile(
    r"<\uff5cGCML\uff5cparameter\s+name=\"(?P<name>[^\"]*)\"\s+"
    r"string=\"(?P<is_string>true|false)\"\s*>"
    r"(?P<value>.*?)"
    r"</\uff5cGCML\uff5cparameter>",
    re.DOTALL,
)

_V5_THINK_CLOSE = "</think>"
_V5_THINK_RE = re.compile(
    r"^\s*(?P<reasoning>.*?)</think>\s*",
    re.DOTALL,
)


class CompletionParserStrategy(ABC):
    @abstractmethod
    def parse(
        self, completion_str: str, reasoning_active: bool = False
    ) -> OpenAIChatCompletionMessage: ...


def _new_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _tool_calls_from_function_calls(
    calls: List[OpenAILegacyFunctionCall],
) -> Optional[list[OpenAIToolCall]]:
    if not calls:
        return None
    return [
        OpenAIToolCall(
            id=_new_tool_call_id(),
            function=OpenAIToolCallFunction(
                name=fc.name,
                arguments=fc.arguments,
            ),
        )
        for fc in calls
    ]


def _tool_calls_from_function_call(
    function_call: Optional[OpenAILegacyFunctionCall],
) -> Optional[list[OpenAIToolCall]]:
    return _tool_calls_from_function_calls([function_call] if function_call else [])


class V4CompletionParser(CompletionParserStrategy):
    _re_fc = re.compile(
        r"function call<\|role_sep\|>\n(.*?)(?:</s>|<\|message_sep\|>|$)", re.DOTALL
    )
    _re_content = re.compile(r"^(.*?)(?:<\|message_sep\|>|</s>)", re.DOTALL)

    def parse(
        self, completion_str: str, reasoning_active: bool = False
    ) -> OpenAIChatCompletionMessage:
        m_reasoning = REGEX_REASONING_SPLIT.match(completion_str)
        if m_reasoning:
            reasoning = m_reasoning.group("reasoning").strip() or ""
            assistant_text = m_reasoning.group("assistant").strip()
            m_func = self._re_fc.search(assistant_text)
            if m_func:
                function_call = _try_parse_fc_json(m_func.group(1))
                if function_call is None and m_func.group(1):
                    return OpenAIChatCompletionMessage(
                        content=assistant_text,
                        reasoning_content=reasoning,
                        tool_calls=None,
                        function_call=None,
                    )
                return OpenAIChatCompletionMessage(
                    content=assistant_text[: m_func.start()],
                    reasoning_content=reasoning,
                    tool_calls=_tool_calls_from_function_call(function_call),
                    function_call=function_call,
                )

            return OpenAIChatCompletionMessage(
                content=assistant_text, reasoning_content=reasoning, tool_calls=None
            )

        m_func = self._re_fc.search(completion_str)
        if m_func:
            function_call = _try_parse_fc_json(m_func.group(1))
            if function_call is None and m_func.group(1):
                return OpenAIChatCompletionMessage(
                    content=completion_str,
                    reasoning_content="",
                    tool_calls=None,
                    function_call=None,
                )
            text = completion_str[: m_func.start()]
            if reasoning_active and "<|message_sep|>" in text:
                reasoning = text.split("<|message_sep|>")[0].strip() or ""
                return OpenAIChatCompletionMessage(
                    content="",
                    reasoning_content=reasoning,
                    tool_calls=_tool_calls_from_function_call(function_call),
                    function_call=function_call,
                )
            m_content = self._re_content.search(text)
            if m_content:
                return OpenAIChatCompletionMessage(
                    content=m_content.group(1),
                    reasoning_content="",
                    tool_calls=_tool_calls_from_function_call(function_call),
                    function_call=function_call,
                )
            return OpenAIChatCompletionMessage(
                content=text,
                reasoning_content="",
                tool_calls=_tool_calls_from_function_call(function_call),
                function_call=function_call,
            )

        m_content = self._re_content.search(completion_str)
        if m_content:
            return OpenAIChatCompletionMessage(
                content=m_content.group(1),
                reasoning_content="",
                tool_calls=None,
                function_call=None,
            )
        return OpenAIChatCompletionMessage(
            content=completion_str,
            reasoning_content="",
            tool_calls=None,
            function_call=None,
        )


class V41CompletionParser(CompletionParserStrategy):
    _re = re.compile(
        r"^(?P<content>.*?)"
        r"(?:<\|function_call\|>(?P<fc_suffix>.*?))?"
        r"(?:</s>|<\|message_sep\|>)?$",
        re.DOTALL,
    )

    def parse(
        self, completion_str: str, reasoning_active: bool = False
    ) -> OpenAIChatCompletionMessage:
        m_reasoning = REGEX_REASONING_SPLIT.match(completion_str)
        if m_reasoning:
            reasoning = m_reasoning.group("reasoning").strip() or ""
            assistant_text = m_reasoning.group("assistant").strip()
            parsed = self._parse_content_fc(assistant_text)
            if parsed is None:
                return OpenAIChatCompletionMessage(
                    content=assistant_text, reasoning_content=reasoning, tool_calls=None
                )
            fc, content = parsed
            return OpenAIChatCompletionMessage(
                content=content,
                reasoning_content=reasoning,
                tool_calls=_tool_calls_from_function_call(fc),
                function_call=fc,
            )

        parsed = self._parse_content_fc(completion_str)
        if parsed is None:
            return OpenAIChatCompletionMessage(
                content=completion_str, reasoning_content="", tool_calls=None
            )

        fc, content = parsed
        if reasoning_active and fc and content:
            return OpenAIChatCompletionMessage(
                content="",
                reasoning_content=content.strip() or "",
                tool_calls=_tool_calls_from_function_call(fc),
                function_call=fc,
            )

        return OpenAIChatCompletionMessage(
            content=content,
            reasoning_content="",
            tool_calls=_tool_calls_from_function_call(fc),
            function_call=fc,
        )

    def _parse_content_fc(
        self, source: str
    ) -> Optional[
        tuple[
            Optional[
                Optional[OpenAILegacyFunctionCall | list[OpenAILegacyFunctionCall]]
            ],
            str,
        ]
    ]:
        m = self._re.match(source)
        if not m:
            return None
        fc_raw = m.group("fc_suffix")
        fc = (
            _parse_v41_fc_json_payload(fc_raw)
            if fc_raw is not None and fc_raw.strip()
            else None
        )
        if fc_raw is not None and fc_raw.strip() and fc is None:
            return None
        return fc, (m.group("content") or "")


class V5CompletionParser(CompletionParserStrategy):
    def parse(
        self, completion_str: str, reasoning_active: bool = False
    ) -> OpenAIChatCompletionMessage:
        m_think = _V5_THINK_RE.match(completion_str)
        if reasoning_active and not m_think:
            return OpenAIChatCompletionMessage(
                content="",
                reasoning_content=_strip_trailing_markers(completion_str),
                tool_calls=None,
                function_call=None,
            )

        if m_think:
            reasoning = m_think.group("reasoning").strip()
            rest = completion_str[m_think.end() :]
        else:
            reasoning = ""
            rest = completion_str

        calls, content = self._extract_calls(rest)
        return OpenAIChatCompletionMessage(
            content=content,
            reasoning_content=reasoning,
            tool_calls=_tool_calls_from_function_calls(calls),
            function_call=calls[0] if calls else None,
        )

    @staticmethod
    def _extract_calls(
        source: str,
    ) -> tuple[List[OpenAILegacyFunctionCall], str]:
        if _GCML_OPEN not in source:
            return [], _strip_trailing_markers(source)

        leading_content, _, after_open = source.partition(_GCML_OPEN)
        invokes_body, _, _after_close = after_open.partition(_GCML_CLOSE)

        calls = [
            OpenAILegacyFunctionCall(
                name=m.group("name"),
                arguments=_parse_gcml_invoke_body(m.group("body")),
            )
            for m in _GCML_INVOKE_RE.finditer(invokes_body)
        ]

        if not calls:
            return [], _strip_trailing_markers(source)

        return calls, _strip_trailing_markers(leading_content).rstrip("\n")


_TRAILING_MARKER_RE = re.compile(r"(?:<\|message_sep\|>|</s>)+\s*$")


def _strip_trailing_markers(text: str) -> str:
    return _TRAILING_MARKER_RE.sub("", text)


def _parse_gcml_invoke_body(body: str) -> Dict[str, Any]:
    args: Dict[str, Any] = {}
    for m in _GCML_PARAM_RE.finditer(body):
        raw = m.group("value")
        if m.group("is_string") == "true":
            args[m.group("name")] = raw
            continue
        try:
            args[m.group("name")] = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            args[m.group("name")] = raw
    return args


def _parse_v41_fc_json_payload(
    raw: Optional[str],
) -> Optional[OpenAILegacyFunctionCall]:
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        obj, end = json.JSONDecoder().raw_decode(raw)
    except json.JSONDecodeError:
        return None
    trailing = raw[end:].strip()
    if trailing and not trailing.startswith("<|function_call|>"):
        return None
    if not (
        isinstance(obj, dict)
        and "name" in obj
        and "arguments" in obj
        and isinstance(obj["arguments"], dict)
    ):
        return None
    return OpenAILegacyFunctionCall(
        name=obj["name"],
        arguments=json.dumps(obj["arguments"], ensure_ascii=False),
    )


def _try_parse_fc_json(raw: Optional[str]) -> Optional[OpenAILegacyFunctionCall]:
    return _parse_v41_fc_json_payload(raw)


PARSERS = {
    "v4": V4CompletionParser(),
    "v4.1": V41CompletionParser(),
    "v5": V5CompletionParser(),
}


def detect_template_version(completion_str: str) -> str:
    if _V5_THINK_CLOSE in completion_str:
        return "v5"
    if _GCML_OPEN in completion_str:
        return "v5"
    if "<|function_call|>" in completion_str:
        return "v4.1"
    if "function call<|role_sep|>" in completion_str:
        return "v4"
    if "<|message_sep|>" in completion_str or "<|role_sep|>" in completion_str:
        return "v4.1"
    return "v5"


def parse_function_and_content(
    completion_str: str,
    *,
    template_version_override: Optional[str] = None,
    reasoning_active: bool = False,
) -> OpenAIChatCompletionMessage:
    parser: CompletionParserStrategy = (
        PARSERS[template_version_override]
        if template_version_override
        else PARSERS[detect_template_version(completion_str)]
    )
    return parser.parse(completion_str, reasoning_active=reasoning_active)
