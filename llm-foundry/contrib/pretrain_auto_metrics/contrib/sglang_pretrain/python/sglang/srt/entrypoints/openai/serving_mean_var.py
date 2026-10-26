import json
import uuid
from pathlib import Path

from transformers import PreTrainedTokenizerFast

from sglang.srt.entrypoints.openai.mean_var.decoder import decode_pair
from sglang.srt.entrypoints.openai.mean_var.protocol import (
    SCHEMA,
    load_protocol,
    marker_prefix,
    sha256,
)
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    UsageInfo,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.managers.io_struct import EmbeddingReqInput
from sglang.srt.observability.req_time_stats import monotonic_time


class OpenAIServingMeanVar(OpenAIServingBase):
    """Score structured answer pairs through Chat Completions without decoding tokens."""

    def __init__(self, tokenizer_manager, template_manager):
        super().__init__(tokenizer_manager)
        checkpoint = Path(tokenizer_manager.model_path)
        self.protocol = load_protocol(checkpoint)
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(checkpoint / "tokenizer.json"),
            chat_template=(checkpoint / "chat_template.jinja").read_text(),
        )
        self.hashes = {
            "protocol_sha256": sha256(checkpoint / "mean_var_protocol.json"),
            "export_manifest_sha256": sha256(checkpoint / "export_manifest.json"),
        }
        if tokenizer_manager.is_generation:
            raise ValueError("Mean-Var requires --is-embedding")
        if tokenizer_manager.model_config.context_len < self.protocol["max_length"]:
            raise ValueError(
                "Mean-Var context-length is smaller than protocol.max_length"
            )

    def _request_id_prefix(self):
        return "chatcmpl-"

    def _validate_request(self, request):
        if request.lora_path or self._parse_model_parameter(request.model)[1]:
            return "Mean-Var requires an exported checkpoint, not a request-time LoRA adapter"
        if request.stream or request.n != 1 or request.tools or request.logprobs:
            return "Mean-Var supports non-streaming requests with n=1, no tools or token logprobs"
        if request.reasoning_effort not in (None, "none") or any(
            (request.chat_template_kwargs or {}).get(key)
            for key in ("enable_thinking", "thinking")
        ):
            return "Mean-Var does not support reasoning"
        if request.response_format and request.response_format.type not in (
            "text",
            "json_object",
        ):
            return "Mean-Var returns fixed JSON; response_format must be text or json_object"
        if (
            len(request.messages) != 1
            or request.messages[0].role != "user"
            or not isinstance(request.messages[0].content, str)
        ):
            return "Mean-Var expects one user message containing a JSON answer pair or pairs array"

    def _convert_to_internal_request(self, request, raw_request=None):
        received_time = monotonic_time()
        payload = json.loads(request.messages[0].content)
        if not isinstance(payload, dict) or payload.get("schema", SCHEMA) != SCHEMA:
            raise ValueError("Expected a gemma-mean-var-v1 JSON object")
        is_batch = "pairs" in payload
        pairs = payload["pairs"] if is_batch else [payload]
        if not isinstance(pairs, list) or not pairs:
            raise ValueError("pairs must be a non-empty array")
        results, tokenized = [], []
        for index, pair in enumerate(pairs):
            try:
                if (
                    not isinstance(pair, dict)
                    or any(
                        not isinstance(pair.get(key), str)
                        for key in ("question", "answer_a", "answer_b")
                    )
                    or not isinstance(pair.get("domain", ""), str)
                ):
                    raise ValueError(
                        "Each pair requires text question, answer_a, answer_b and optional domain"
                    )
                ids = [
                    marker_prefix(
                        self.tokenizer,
                        self.protocol,
                        pair["question"],
                        pair[key],
                        pair.get("domain", ""),
                    )
                    for key in ("answer_a", "answer_b")
                ]
            except ValueError as exc:
                if not is_batch:
                    raise
                results.append({"error": str(exc), **self.hashes})
                continue
            results.append(None)
            tokenized.append((index, ids))
        return tokenized, (request, results, is_batch, received_time)

    async def _handle_non_streaming_request(
        self, tokenized, processed_request, raw_request
    ):
        request, results, is_batch, received_time = processed_request
        prompt_tokens = 0
        if tokenized:
            adapted = EmbeddingReqInput(
                input_ids=[ids for _, pair in tokenized for ids in pair],
                priority=request.priority,
                routing_key=self.extract_routing_key(raw_request),
                received_time=received_time,
            )
            rows = await self.tokenizer_manager.generate_request(
                adapted, raw_request
            ).__anext__()
            if not isinstance(rows, list) or len(rows) != 2 * len(tokenized):
                raise RuntimeError("Mean-Var backend returned incomplete embeddings")
            prompt_tokens = sum(
                row["meta_info"].get("prompt_tokens", 0) for row in rows
            )
            for offset, (index, _) in enumerate(tokenized):
                outputs = [
                    row["embedding"] for row in rows[2 * offset : 2 * offset + 2]
                ]
                results[index] = {
                    **decode_pair(outputs, self.protocol),
                    "reverse": decode_pair(outputs[::-1], self.protocol),
                    **self.hashes,
                }
        content = {"results": results} if is_batch else results[0]
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            model=self.tokenizer_manager.server_args.served_model_name,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=json.dumps(
                            content, ensure_ascii=False, allow_nan=False
                        ),
                    ),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                total_tokens=prompt_tokens,
            ),
        )
