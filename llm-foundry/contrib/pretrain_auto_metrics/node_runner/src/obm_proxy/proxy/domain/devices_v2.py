from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pam_proxies.models import (
    DevicesV2ChatCompletionRequest,
    OpenAIChatCompletionMessage,
    TemplateChatCompletionRequest,
)


def _args_to_dict(arguments: Any) -> Any:
    """Return a JSON object for tool-call arguments.

    devices-v2 carries arguments as a dict already; the GCML parser may produce
    a JSON string. Mirror service._maybe_parse_json_args: parse strings into a
    dict when possible, otherwise pass through unchanged (never raise).
    """
    if not isinstance(arguments, str):
        return arguments
    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, ValueError):
        return arguments
    return parsed if isinstance(parsed, dict) else arguments


def _join_text_parts(content: List[Any]) -> str:
    texts: List[str] = []
    for part in content:
        if isinstance(part, dict) and "text" in part:
            texts.append("" if part["text"] is None else str(part["text"]))
    return "".join(texts)


def convert_v2_request_to_openai(
    body: DevicesV2ChatCompletionRequest,
) -> TemplateChatCompletionRequest:
    """Adapt a devices-v2 wire request into the OpenAI-shaped request the custom
    proxy path already knows how to render with the v5 jinja template.

    The conversion is intentionally lossless w.r.t. tool calling:
      * assistant `{function_call}` parts -> OpenAI `tool_calls` (args kept as
        dict; order preserved for N parallel calls),
      * a k-result tool message -> k OpenAI `{role:"tool"}` messages, each
        FIFO-matched to the queued (id, name) of the preceding assistant calls,
      * `functions` -> flat function dicts under `functions` so the v5 template
        sees `return_parameters` / `few_shot_examples` (it would drop them if
        they were wrapped under an OpenAI `{"type","function"}` tool item).
    """
    openai_messages: List[Dict[str, Any]] = []
    pending_calls: List[tuple] = []  # FIFO queue of (call_id, name)
    call_counter = 0

    for msg in body.messages:
        role = msg.role
        parts = msg.content or []

        if role == "assistant":
            text = _join_text_parts(parts)
            tool_calls: List[Dict[str, Any]] = []
            for part in parts:
                if not isinstance(part, dict):
                    continue
                fc = part.get("function_call")
                if not isinstance(fc, dict):
                    continue
                name = fc.get("name")
                args = fc.get("arguments", {})
                call_id = f"call_{call_counter}"
                call_counter += 1
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": _args_to_dict(args)},
                    }
                )
                pending_calls.append((call_id, name))

            out: Dict[str, Any] = {"role": "assistant"}
            if tool_calls:
                out["tool_calls"] = tool_calls
            # Always carry a content string: empty string is fine and keeps the
            # assistant message valid even for a tool-only turn.
            out["content"] = text
            openai_messages.append(out)

        elif role == "tool":
            for part in parts:
                if not isinstance(part, dict):
                    continue
                fr = part.get("function_result")
                if not isinstance(fr, dict):
                    continue
                name = fr.get("name")
                result = fr.get("result")
                if not isinstance(result, str):
                    result = json.dumps(result, ensure_ascii=False)
                if pending_calls:
                    call_id = pending_calls.pop(0)[0]
                else:
                    call_id = f"legacy::{name}"
                openai_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": result,
                    }
                )

        else:
            # user / system / developer / ... -> plain string content.
            # Per-message `function_names` is intentionally ignored: the v5
            # serving template re-emits a global available-functions block from
            # `functions=`, so subsetting is not honored.
            openai_messages.append({"role": role, "content": _join_text_parts(parts)})

    functions: Optional[List[Dict[str, Any]]] = None
    if body.functions:
        functions = []
        for fn in body.functions:
            fd = fn.model_dump(exclude_none=True)
            # `role` is a devices-v2 wrapper artifact; the v5 template wants flat
            # function dicts (name/description/parameters/return_parameters/
            # few_shot_examples).
            fd.pop("role", None)
            functions.append(fd)

    data: Dict[str, Any] = {
        "messages": openai_messages,
        "functions": functions,
        "model": body.model,
        "routing_key": body.routing_key,
        "temperature": body.temperature,
        "top_p": body.top_p,
        "top_k": body.top_k,
        "repetition_penalty": body.repetition_penalty,
        "max_tokens": body.max_tokens,
        "max_completion_tokens": body.max_completion_tokens,
        "stop": body.stop,
        "stop_token_ids": body.stop_token_ids,
        "chat_template_kwargs": body.chat_template_kwargs,
    }
    data["priority"] = body.priority
    data = {key: value for key, value in data.items() if value is not None}
    return TemplateChatCompletionRequest.model_validate(data)


def build_v2_response(
    message: OpenAIChatCompletionMessage,
    *,
    model: Optional[str],
    finish_reason: str,
    prompt_tokens: int,
    completion_tokens: int,
    created_at: int,
) -> Dict[str, Any]:
    """Build a devices-v2 `ChatCompletionV2` payload from the parsed assistant
    message. Content is always `[{text}, {function_call}*]`: the leading text
    part (possibly empty) satisfies OBM's `assert contents` for tool-only turns
    and each parsed tool call becomes a `{function_call}` part (parallel calls
    -> N parts). The permissive
    devices-v2 message contract also carries `reasoning_content` so downstream
    clients can persist reasoning traces separately from the answer.
    """
    content_parts: List[Dict[str, Any]] = [{"text": message.content or ""}]
    for tool_call in message.tool_calls or []:
        content_parts.append(
            {
                "function_call": {
                    "name": tool_call.function.name,
                    "arguments": _args_to_dict(tool_call.function.arguments),
                }
            }
        )

    response_message: Dict[str, Any] = {
        "role": "assistant",
        "content": content_parts,
    }
    if message.reasoning_content:
        response_message["reasoning_content"] = message.reasoning_content

    return {
        "model": model,
        "created_at": created_at,
        "messages": [response_message],
        "finish_reason": finish_reason,
        "usage": {
            "input_tokens": prompt_tokens,
            "input_tokens_details": {
                "prompt_tokens": prompt_tokens,
                "cached_tokens": 0,
            },
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
