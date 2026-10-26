import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletion

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.openai.mean_var.protocol import marker_prefix, sha256
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_mean_var import OpenAIServingMeanVar

pytestmark = pytest.mark.unit


@pytest.fixture
def service(bundle):
    calls = []
    version = sha256(bundle / "export_manifest.json")
    manager = SimpleNamespace(
        model_path=str(bundle),
        is_generation=False,
        model_config=SimpleNamespace(context_len=1000),
        server_args=SimpleNamespace(
            weight_version=version, served_model_name="gemma4-meanvar"
        ),
        request_logger=SimpleNamespace(log_requests=False),
    )

    async def generate(request, raw_request):
        calls.append(request.input_ids)
        yield [
            {
                "embedding": [2.0 if i % 2 == 0 else 0.0, 0.4],
                "meta_info": {
                    "weight_version": manager.server_args.weight_version,
                    "prompt_tokens": len(ids),
                },
            }
            for i, ids in enumerate(request.input_ids)
        ]

    manager.generate_request = generate
    handler = OpenAIServingMeanVar(manager, None)
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def score(body: ChatCompletionRequest, request: Request):
        return await handler.handle_request(body, request)

    return TestClient(app), calls, manager, handler


def request_body(payload, **kwargs):
    return {
        "model": "gemma4-meanvar",
        "messages": [{"role": "user", "content": json.dumps(payload)}],
        **kwargs,
    }


def pair():
    return {"question": "question", "answer_a": "answer", "answer_b": "answer"}


def test_chat_completion_scores_both_orders_from_two_inputs(service):
    client, calls, _, handler = service
    response = client.post("/v1/chat/completions", json=request_body(pair()))
    assert response.status_code == 200
    completion = ChatCompletion.model_validate(response.json())
    result = json.loads(completion.choices[0].message.content)
    assert result["verdict"] == "A>B"
    assert result["reverse"]["verdict"] == "B>A"
    assert result["probabilities"]["A>B"] == pytest.approx(
        result["reverse"]["probabilities"]["B>A"]
    )
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    assert result["rewards"] == {"A": [2.0, 0.4], "B": [0.0, 0.4]}
    assert result["export_manifest_sha256"] == handler.hashes["export_manifest_sha256"]
    assert len(calls) == 1 and len(calls[0]) == 2
    assert completion.usage.completion_tokens == 0
    assert completion.choices[0].logprobs is None


def test_batch_submits_all_inputs_to_scheduler(service):
    client, calls, _, _ = service
    response = client.post(
        "/v1/chat/completions",
        json=request_body(
            {
                "pairs": [pair(), pair(), pair()],
            }
        ),
    )
    assert response.status_code == 200
    results = json.loads(response.json()["choices"][0]["message"]["content"])["results"]
    assert len(results) == 3
    assert [len(ids) for ids in calls] == [6]


def test_invalid_pair_does_not_drop_valid_batch_members(service):
    client, calls, _, _ = service
    response = client.post(
        "/v1/chat/completions", json=request_body({"pairs": [pair(), {"question": 5}]})
    )
    assert response.status_code == 200
    results = json.loads(response.json()["choices"][0]["message"]["content"])["results"]
    assert results[0]["verdict"] == "A>B"
    assert "error" in results[1]
    assert len(calls[0]) == 2


@pytest.mark.parametrize(
    "option",
    [
        {"stream": True},
        {"n": 2},
        {"logprobs": True},
        {"reasoning_effort": "high"},
        {"lora_path": "adapter"},
    ],
)
def test_unsupported_generation_options_fail_before_inference(service, option):
    client, calls, _, _ = service
    response = client.post("/v1/chat/completions", json=request_body(pair(), **option))
    assert response.status_code == 400
    assert not calls


def test_weight_version_label_does_not_change_scores_or_identity(service):
    client, _, manager, _ = service
    original = client.post("/v1/chat/completions", json=request_body(pair()))
    assert original.status_code == 200
    manager.server_args.weight_version = "another-label"
    response = client.post("/v1/chat/completions", json=request_body(pair()))
    assert response.status_code == 200
    assert response.json()["choices"] == original.json()["choices"]


@pytest.mark.parametrize("version", ["default", "custom-label"])
def test_constructor_preserves_weight_version_label(service, version):
    _, _, manager, handler = service
    manager.server_args.weight_version = version
    reloaded = OpenAIServingMeanVar(manager, None)
    assert manager.server_args.weight_version == version
    assert reloaded.hashes == handler.hashes


def test_batch_preserves_original_pair_order(service):
    client, calls, manager, _ = service

    async def generate(request, raw_request):
        calls.append(request.input_ids)
        yield [
            {
                "embedding": [float(len(ids)), 0.4],
                "meta_info": {"weight_version": manager.server_args.weight_version},
            }
            for ids in request.input_ids
        ]

    manager.generate_request = generate
    longer = {**pair(), "answer_a": "answer " * 10}
    response = client.post(
        "/v1/chat/completions", json=request_body({"pairs": [longer, pair()]})
    )
    assert response.status_code == 200
    results = json.loads(response.json()["choices"][0]["message"]["content"])["results"]
    assert len(calls[0][0]) > len(calls[0][2])
    assert results[0]["rewards"]["A"][0] == len(calls[0][0])
    assert results[1]["rewards"]["A"][0] == len(calls[0][2])


def test_overlength_pair_is_reported_without_truncation(service):
    client, calls, _, handler = service
    longer = {**pair(), "answer_a": "answer " * handler.protocol["max_length"]}
    response = client.post(
        "/v1/chat/completions", json=request_body({"pairs": [longer, pair()]})
    )
    assert response.status_code == 200
    results = json.loads(response.json()["choices"][0]["message"]["content"])["results"]
    assert "truncation is forbidden" in results[0]["error"]
    assert results[1]["verdict"] == "A>B"
    assert len(calls[0]) == 2


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("is_generation", True, "is-embedding"),
        ("context_len", 1, "context-length"),
    ],
)
def test_constructor_rejects_incompatible_runtime(service, field, value, error):
    _, _, manager, _ = service
    setattr(manager.model_config if field == "context_len" else manager, field, value)
    with pytest.raises(ValueError, match=error):
        OpenAIServingMeanVar(manager, None)


def test_unhashed_tokenizer_config_cannot_change_inputs_or_identity(service, bundle):
    _, _, manager, handler = service
    expected = marker_prefix(handler.tokenizer, handler.protocol, "question", "answer")
    config_path = bundle / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["extra_special_tokens"] = {"pair_token": "question answer"}
    config_path.write_text(json.dumps(config))

    reloaded = OpenAIServingMeanVar(manager, None)

    assert (
        marker_prefix(reloaded.tokenizer, reloaded.protocol, "question", "answer")
        == expected
    )
    assert reloaded.hashes == handler.hashes
