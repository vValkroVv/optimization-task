from typing import Any, Dict, List

from bfcl_eval.model_handler.api_inference.openai_completion import (
    OpenAICompletionsHandler,
)
from overrides import override


class GigaChatAPIHandler(OpenAICompletionsHandler):
    """
    GigaChat OpenAI-compatible API can return `reasoning_content` and may still return
    regular `content` alongside `tool_calls`.

    Unlike OpenAICompletionsHandler (DeepSeek/Grok-style), we keep `reasoning_content`
    on assistant turns in chat history, preserve assistant `content` when tools are
    present (including coercing null to ""), and avoid dropping `tool_calls` when
    attaching reasoning in prompting mode.
    """

    @staticmethod
    def _assistant_text_content(message: Any) -> str:
        """Use empty string when the API omits content (null) so the field is still sent."""
        raw = getattr(message, "content", None)
        return "" if raw is None else raw

    @staticmethod
    def _tool_calls_for_history(message: Any) -> List[dict]:
        return [
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in message.tool_calls
        ]

    @override
    def _add_reasoning_content_if_available_prompting(
        self, api_response: Any, response_data: dict
    ) -> None:
        message = api_response.choices[0].message
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning is None:
            return

        response_data["reasoning_content"] = reasoning

        assistant: Dict[str, Any] = {
            "role": "assistant",
            "content": str(response_data["model_responses"]),
            "reasoning_content": reasoning,
        }
        if getattr(message, "tool_calls", None):
            assistant["tool_calls"] = self._tool_calls_for_history(message)

        response_data["model_responses_message_for_chat_history"] = assistant

    @override
    def _add_reasoning_content_if_available_FC(
        self, api_response: Any, response_data: dict
    ) -> None:
        message = api_response.choices[0].message
        content = self._assistant_text_content(message)
        reasoning = getattr(message, "reasoning_content", None)

        if getattr(message, "tool_calls", None):
            assistant_message: Dict[str, Any] = {
                "role": "assistant",
                "content": content,
                "tool_calls": self._tool_calls_for_history(message),
            }
            if reasoning is not None:
                assistant_message["reasoning_content"] = reasoning
            response_data["model_responses_message_for_chat_history"] = assistant_message
        elif reasoning is not None:
            response_data["model_responses_message_for_chat_history"] = {
                "role": "assistant",
                "content": content,
                "reasoning_content": reasoning,
            }

        if reasoning is not None:
            response_data["reasoning_content"] = reasoning

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        response_data = super()._parse_query_response_prompting(api_response)
        self._add_reasoning_content_if_available_prompting(api_response, response_data)
        return response_data

    @override
    def _parse_query_response_FC(self, api_response: Any) -> dict:
        response_data = super()._parse_query_response_FC(api_response)
        self._add_reasoning_content_if_available_FC(api_response, response_data)
        return response_data
