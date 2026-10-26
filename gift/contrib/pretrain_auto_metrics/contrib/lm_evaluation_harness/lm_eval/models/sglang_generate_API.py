from typing import Dict, List, Optional, Tuple, Union

from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import LocalCompletionsAPI
from lm_eval.models.utils import handle_stop_sequences


@register_model("sglang-generate")
class SGLANGGENERATEAPI(LocalCompletionsAPI):
    def __init__(
        self,
        base_url=None,
        tokenizer_backend="huggingface",
        **kwargs,
    ):
        super().__init__(
            base_url=base_url, tokenizer_backend=tokenizer_backend, **kwargs
        )

    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        generate=False,
        gen_kwargs: Optional[dict] = None,
        seed: int = 1234,
        eos=None,
        **kwargs,
    ) -> dict:
        is_string = (
            True
            if (isinstance(messages, str) or isinstance(messages[0], str))
            else False
        )
        if generate:
            gen_kwargs.pop("do_sample", False)
            if "max_tokens" in gen_kwargs:
                max_tokens = gen_kwargs.pop("max_tokens")
            else:
                max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
            temperature = gen_kwargs.pop("temperature", 0)
            stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
            request = {
                "sampling_params": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "stop": stop,
                    **gen_kwargs,
                },
            }
            request.update({"text": messages}) if is_string else request.update(
                {"input_ids": messages}
            )
            return request
        else:
            assert not is_string, "Logprobs are only supported for tokenized inputs"
            ctxlens = kwargs.get("ctxlens")
            start_from_ctx = max(min(ctxlens) - 1, 0) if ctxlens else 0
            if isinstance(messages[0], int):
                min_input_len = len(messages)
            else:
                min_input_len = min(len(m) for m in messages)
            max_start = max(min_input_len - 1, 0)
            logprob_start_len = min(start_from_ctx, max_start)
            request = {
                "input_ids": messages,
                "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                "logprob_start_len": logprob_start_len,
                "top_logprobs_num": 1,
                "return_logprob": True,
            }
            return request

    @staticmethod
    def parse_logprobs(
        outputs: Union[Dict, List[Dict]],
        tokens: Union[List[List[int]], List[int]] = None,
        ctxlens: List[int] = None,
        **kwargs,
    ) -> List[Tuple[float, bool]]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for choice, ctxlen in zip(outputs, ctxlens):
            meta = choice["meta_info"]
            assert ctxlen > 0, "Context length must be greater than 0"
            logprobs_list = meta["input_token_logprobs"]
            top_logprobs_list = meta["input_top_logprobs"]
            if tokens is not None:
                tok = tokens[len(res)] if isinstance(tokens[0], list) else tokens
                total_len = len(tok)
                logprob_start = max(total_len - len(logprobs_list), 0)
                offset = max(ctxlen - logprob_start, 0)
            else:
                offset = ctxlen
            offset = min(offset, len(logprobs_list), len(top_logprobs_list))
            logprobs = sum(x[0] for x in logprobs_list[offset:])
            is_greedy = all(
                x[1] != y[0][1]
                for x, y in zip(
                    logprobs_list[offset:],
                    top_logprobs_list[offset:],
                )
            )
            res.append((logprobs, is_greedy))
        return res

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            res.append(out["text"])
        return res

    @property
    def api_key(self):
        return ""
