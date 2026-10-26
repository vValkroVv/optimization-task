import asyncio
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gigaomni, pytest.mark.slow]

_DEFAULT_IMAGE_TOKEN_ID = int(
    os.environ.get("GIGAOMNI_TEST_IMAGE_TOKEN_ID", 128004)
)

_GREEDY = {"temperature": 0.0, "max_new_tokens": 16}
_BATCH_SIZE = 4
_CHUNKED_PREFILL_SIZE = 4096
_LONG_PROMPT_REPEATS = 1024


@dataclass(frozen=True)
class _Request:
    name: str
    input_ids: list[int]
    image_data: Optional[list[str]] = None
    audio_data: Optional[list[str]] = None


def _build_text_dialog(prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "developer system",
            "content": "",
            "tokenize_special_tokens": True,
            "message_index": 0,
        },
        {"role": "system", "content": "ты - Гигачат", "message_index": 1},
        {"role": "user", "content": prompt, "message_index": 2},
    ]


def _build_image_dialog(prompt: str, image_path: str) -> list[dict[str, Any]]:
    image_id = "1c63a4ec-0c3b-4662-b0d2-5ba77f37bfe3"
    return [
        {
            "role": "developer system",
            "content": "",
            "tokenize_special_tokens": True,
            "message_index": 0,
        },
        {"role": "system", "content": "ты - Гигачат", "message_index": 1},
        {
            "role": "added files",
            "content": [{"id": image_id, "name": "image.jpg", "type": "image"}],
            "message_index": 2,
        },
        {
            "role": "user",
            "content": prompt,
            "embed_files": [
                {
                    "id": image_id,
                    "name": "image.jpg",
                    "type": "image",
                    "embed": True,
                    "path": image_path,
                }
            ],
            "message_index": 3,
        },
    ]


def _build_audio_dialog(prompt: str, audio_path: str) -> list[dict[str, Any]]:
    audio_id = "e35ec3c0-69c6-4a23-af2b-43646685ddad"
    return [
        {
            "role": "developer system",
            "content": "",
            "tokenize_special_tokens": True,
            "message_index": 0,
        },
        {"role": "system", "content": "ты - Гигачат", "message_index": 1},
        {
            "role": "added files",
            "content": [{"id": audio_id, "name": "audio.wav", "type": "audio"}],
            "message_index": 2,
        },
        {
            "role": "user",
            "content": prompt,
            "embed_files": [
                {
                    "id": audio_id,
                    "name": "audio.wav",
                    "type": "audio",
                    "embed": True,
                    "path": audio_path,
                }
            ],
            "message_index": 3,
        },
    ]


def _tokenize(tokenizer, dialog: list[dict[str, Any]]) -> list[int]:
    return tokenizer.apply_chat_template(
        dialog,
        tools=[],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )


def _make_text_request(tokenizer, name: str, prompt: str) -> _Request:
    return _Request(name=name, input_ids=_tokenize(tokenizer, _build_text_dialog(prompt)))


def _make_image_request(tokenizer, name: str, prompt: str, image_path: str) -> _Request:
    return _Request(
        name=name,
        input_ids=_tokenize(tokenizer, _build_image_dialog(prompt, image_path)),
        image_data=[image_path],
    )


def _make_audio_request(tokenizer, name: str, prompt: str, audio_path: str) -> _Request:
    return _Request(
        name=name,
        input_ids=_tokenize(tokenizer, _build_audio_dialog(prompt, audio_path)),
        audio_data=[audio_path],
    )


def _find_image_token_index(tokenizer, input_ids: list[int]) -> int:
    candidate_ids = []
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    for token in ("[image_token]", "<image_token>", "<image>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != unk_token_id:
            candidate_ids.append(token_id)
    candidate_ids.append(_DEFAULT_IMAGE_TOKEN_ID)

    for token_id in dict.fromkeys(candidate_ids):
        if token_id in input_ids:
            return input_ids.index(token_id)

    raise ValueError(
        "Cannot find image token in tokenized image request. "
        "Set GIGAOMNI_TEST_IMAGE_TOKEN_ID explicitly."
    )


def _make_text_prefix_of_image_request(tokenizer, image_request: _Request) -> _Request:
    image_token_index = _find_image_token_index(tokenizer, image_request.input_ids)
    assert image_token_index > 0, (
        f"Unexpected image token index: {image_token_index}"
    )
    return _Request(
        name=f"{image_request.name}_text_prefix_only",
        input_ids=image_request.input_ids[:image_token_index],
        image_data=None,
    )


@contextmanager
def _engine(checkpoint, **overrides: Any):
    import sglang as sgl

    engine_args = {
        "model_path": str(checkpoint),
        "tokenizer_path": str(checkpoint / "processors" / "GigaOmniVisionProcessor"),
        "trust_remote_code": True,
        "log_level": "warning",
        "disable_fast_image_processor": True,
        "random_seed": 42,
        "chunked_prefill_size": -1,
        "max_running_requests": 1,
        "prefill_max_requests": 1,
        "disable_overlap_schedule": True,
        "disable_radix_cache": True,
        "enable_deterministic_inference": True,
    }
    engine_args.update(overrides)
    llm = sgl.Engine(**engine_args)
    try:
        yield llm
    finally:
        llm.shutdown()


def _generate_one(llm, request: _Request) -> dict[str, Any]:
    kwargs = {"input_ids": request.input_ids, "sampling_params": dict(_GREEDY)}
    if request.image_data is not None:
        kwargs["image_data"] = request.image_data
    if request.audio_data is not None:
        kwargs["audio_data"] = request.audio_data
    return llm.generate(**kwargs)


async def _async_generate_one(llm, request: _Request) -> dict[str, Any]:
    kwargs = {"input_ids": request.input_ids, "sampling_params": dict(_GREEDY)}
    if request.image_data is not None:
        kwargs["image_data"] = request.image_data
    if request.audio_data is not None:
        kwargs["audio_data"] = request.audio_data
    return await llm.async_generate(**kwargs)


def _generate_concurrent(llm, requests: list[_Request]) -> list[dict[str, Any]]:
    tasks = [_async_generate_one(llm, request) for request in requests]
    outputs = llm.loop.run_until_complete(asyncio.gather(*tasks))
    assert len(outputs) == len(requests)
    return list(outputs)


def _texts(requests: list[_Request], outputs: list[dict[str, Any]]) -> dict[str, str]:
    return {request.name: output["text"] for request, output in zip(requests, outputs)}


def _cached_tokens(output: dict[str, Any]) -> int:
    return int(output.get("meta_info", {}).get("cached_tokens", 0))


@pytest.fixture(scope="module")
def batch_requests(
    gigaomni_tokenizer, gigaomni_image_path, gigaomni_audio_path
) -> list[_Request]:
    image_path = str(gigaomni_image_path)
    audio_path = str(gigaomni_audio_path)
    return [
        _make_image_request(
            gigaomni_tokenizer,
            "image_describe",
            "Опиши картинку одним коротким предложением.",
            image_path,
        ),
        _make_text_request(
            gigaomni_tokenizer,
            "text_math",
            "Сколько будет два плюс два? Ответь одним числом.",
        ),
        _make_image_request(
            gigaomni_tokenizer,
            "image_main_object",
            "Что является главным объектом на картинке? Ответь кратко.",
            image_path,
        ),
        _make_audio_request(
            gigaomni_tokenizer,
            "audio_transcribe",
            "Что сказано в аудиозаписи? Ответь кратко.",
            audio_path,
        ),
        _make_text_request(
            gigaomni_tokenizer,
            "text_role",
            "Кто ты? Ответь очень кратко.",
        ),
    ]


class TestGigaOmniScheduling:
    def test_concurrent_batch_matches_sequential(self, gigaomni_checkpoint, batch_requests):
        requests = batch_requests[:_BATCH_SIZE]
        with _engine(
            gigaomni_checkpoint,
            max_running_requests=max(_BATCH_SIZE, 2),
            prefill_max_requests=max(_BATCH_SIZE, 2),
            disable_radix_cache=True,
        ) as llm:
            sequential = [_generate_one(llm, request) for request in requests]
            concurrent = _generate_concurrent(llm, requests)

        assert _texts(requests, concurrent) == _texts(requests, sequential)

    def test_concurrent_batch_matches_sequential_with_radix_cache(
        self, gigaomni_checkpoint, batch_requests
    ):
        requests = batch_requests[:_BATCH_SIZE]
        with _engine(
            gigaomni_checkpoint,
            max_running_requests=max(_BATCH_SIZE, 2),
            prefill_max_requests=max(_BATCH_SIZE, 2),
            disable_radix_cache=False,
        ) as llm:
            sequential = []
            for request in requests:
                sequential.append(_generate_one(llm, request))
                llm.flush_cache()

            concurrent = _generate_concurrent(llm, requests)

        assert _texts(requests, concurrent) == _texts(requests, sequential)


class TestGigaOmniRadixCache:
    def test_text_cache_hit_preserves_output(self, gigaomni_checkpoint, gigaomni_tokenizer):
        request = _make_text_request(
            gigaomni_tokenizer,
            "radix_text_repeat",
            "Кратко объясни, зачем нужен RadixCache в inference сервере.",
        )
        with _engine(gigaomni_checkpoint, disable_radix_cache=False) as llm:
            _warm = _generate_one(llm, request)
            cached = _generate_one(llm, request)
            llm.flush_cache()
            cold = _generate_one(llm, request)

        assert cached["text"] == cold["text"]

    def test_image_cache_hit_preserves_output(
        self, gigaomni_checkpoint, gigaomni_tokenizer, gigaomni_image_path
    ):
        request = _make_image_request(
            gigaomni_tokenizer,
            "radix_image_repeat",
            "Опиши картинку одним коротким предложением.",
            str(gigaomni_image_path),
        )
        with _engine(gigaomni_checkpoint, disable_radix_cache=False) as llm:
            _warm = _generate_one(llm, request)
            cached = _generate_one(llm, request)
            llm.flush_cache()
            cold = _generate_one(llm, request)

        assert cached["text"] == cold["text"]

    def test_text_prefix_cache_hit_stops_at_image(
        self, gigaomni_checkpoint, gigaomni_tokenizer, gigaomni_image_path
    ):
        image_request = _make_image_request(
            gigaomni_tokenizer,
            "radix_text_prefix_then_image",
            "Опиши картинку одним коротким предложением.",
            str(gigaomni_image_path),
        )
        prefix_request = _make_text_prefix_of_image_request(
            gigaomni_tokenizer, image_request
        )

        with _engine(gigaomni_checkpoint, disable_radix_cache=False) as llm:
            _prefix_warm = _generate_one(llm, prefix_request)
            image_cached = _generate_one(llm, image_request)
            llm.flush_cache()
            image_cold = _generate_one(llm, image_request)

        assert image_cached["text"] == image_cold["text"]

        cached = _cached_tokens(image_cached)
        prefix_len = len(prefix_request.input_ids)
        assert cached > 0, "expected a text-prefix cache hit"
        assert cached <= prefix_len, (
            f"cache hit crossed into image tokens: cached={cached}, prefix={prefix_len}"
        )


class TestGigaOmniChunkedPrefill:
    def test_chunked_prefill_matches_unchunked(
        self, gigaomni_checkpoint, gigaomni_tokenizer, gigaomni_image_path
    ):
        long_prompt = (
            "В конце ответь одним словом: готово. " * _LONG_PROMPT_REPEATS
        )
        request = _make_image_request(
            gigaomni_tokenizer,
            "chunked_prefill_image_long_prompt",
            long_prompt,
            str(gigaomni_image_path),
        )
        assert len(request.input_ids) > 2 * _CHUNKED_PREFILL_SIZE, "prompt too short"

        with _engine(
            gigaomni_checkpoint, chunked_prefill_size=-1, disable_radix_cache=True
        ) as llm:
            reference = _generate_one(llm, request)

        with _engine(
            gigaomni_checkpoint,
            chunked_prefill_size=_CHUNKED_PREFILL_SIZE,
            disable_radix_cache=True,
        ) as llm:
            chunked = _generate_one(llm, request)

        assert chunked["text"] == reference["text"], (
            f"chunked {chunked['text']!r} != reference {reference['text']!r}"
        )
