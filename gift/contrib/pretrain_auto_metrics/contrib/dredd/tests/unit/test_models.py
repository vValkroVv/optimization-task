import pytest

from src.models import APIModel, create_model

pytestmark = [pytest.mark.unit]


def test_create_model_requires_api_base_for_openai():
    with pytest.raises(ValueError, match="api_base"):
        create_model("m", {"api_type": "openai"})


def test_create_model_ift_requires_env(monkeypatch):
    monkeypatch.delenv("GIGACHAT_IFT_CLIENT_ID", raising=False)
    monkeypatch.delenv("GIGACHAT_IFT_CLIENT_SECRET", raising=False)

    with pytest.raises(ValueError, match="requires env"):
        create_model("ift-model", {"api_type": "ift"})


def test_create_model_ift_with_env(monkeypatch):
    monkeypatch.setenv("GIGACHAT_IFT_CLIENT_ID", "id")
    monkeypatch.setenv("GIGACHAT_IFT_CLIENT_SECRET", "secret")

    model = create_model("ift-model", {"api_type": "ift"})
    assert isinstance(model, APIModel)
    assert model.api_type == "ift"


def test_request_url_changes_by_api_type():
    openai_model = APIModel("m", {"api_type": "openai", "api_base": "http://api.local/v1"})
    trt_model = APIModel("m", {"api_type": "trt", "api_base": "http://api.local/v1"})
    ift_model = APIModel("m", {"api_type": "ift"})

    assert openai_model._request_url() == "http://api.local/v1/chat/completions"
    assert trt_model._request_url() == "http://api.local/v1/chat"
    assert ift_model._request_url().endswith("/v1/chat/completions")


def test_headers_for_openai_include_api_key():
    model = APIModel("m", {"api_type": "openai", "api_key": "secret", "api_base": "http://api.local/v1"})
    headers = model._headers_for_request()

    assert headers["Authorization"] == "Bearer secret"
    assert headers["Content-Type"] == "application/json"


def test_headers_for_ift_use_runtime_token(monkeypatch):
    monkeypatch.setattr("src.models._get_ift_token", lambda: "ift-token")
    model = APIModel("m", {"api_type": "ift"})

    headers = model._headers_for_request()
    assert headers["Authorization"] == "Bearer ift-token"


def test_generate_answer_uses_configured_num_retries(monkeypatch):
    calls = {"count": 0}

    def _failing_post(*args, **kwargs):
        calls["count"] += 1
        raise RuntimeError("boom")

    monkeypatch.setattr("src.models.requests.post", _failing_post)
    monkeypatch.setattr("src.models.time.sleep", lambda *_args, **_kwargs: None)

    model = APIModel(
        "m",
        {
            "api_type": "openai",
            "api_base": "http://api.local/v1",
            "num_retries": 5,
        },
    )
    with pytest.raises(RuntimeError, match="API failed after 5 retries"):
        model.generate_answer({"question_id": "1", "turns": [{"content": "q"}]})

    assert calls["count"] == 5


def test_num_retries_lower_bound_is_one():
    model = APIModel(
        "m",
        {
            "api_type": "openai",
            "api_base": "http://api.local/v1",
            "num_retries": 0,
        },
    )
    assert model.num_retries == 1
