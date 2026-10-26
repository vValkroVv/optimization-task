import pytest
import responses
import json
import os
from unittest.mock import patch
from src.models import APIModel, create_model

pytestmark = [pytest.mark.integration]

@responses.activate
def test_ift_token_and_chat_flow(monkeypatch):
    """Тест: получение токена и выполнение чат-запроса IFT."""
    monkeypatch.setenv("GIGACHAT_IFT_CLIENT_ID", "user")
    monkeypatch.setenv("GIGACHAT_IFT_CLIENT_SECRET", "secret")
    
    # 1. Мокаем получение токена (Basic Auth)
    responses.add(
        responses.POST,
        "https://gigachat.sberdevices.ru/v1/token",
        json={"access_token": "mock-token-123"},
        status=200
    )
    
    # 2. Мокаем сам чат-запрос (Bearer Auth)
    responses.add(
        responses.POST,
        "https://gigachat.ift.sberdevices.ru/v1/chat/completions",
        json={
            "choices": [{
                "message": {"content": "IFT Answer"},
                "index": 0
            }]
        },
        status=200
    )
    
    config = {
        "name": "ift_model",
        "api_type": "ift",
        "api_base": "http://ignored", # IFT использует свой URL
        "temperature": 0.0
    }
    
    model = create_model("ift_model", config)
    answer = model.generate_answer({"prompt": "Hello"})
    
    assert answer["choices"][0]["turns"][0]["content"] == "IFT Answer"
    # Проверяем, что токен был передан в заголовках второго запроса
    assert responses.calls[1].request.headers["Authorization"] == "Bearer mock-token-123"

@responses.activate
def test_api_resilience_retries():
    """Тест: механизм повторных попыток (retries) при ошибке 500."""
    # Первый запрос - ошибка 500
    responses.add(
        responses.POST,
        "http://api/chat/completions",
        status=500
    )
    # Второй запрос - успех
    responses.add(
        responses.POST,
        "http://api/chat/completions",
        json={"choices": [{"message": {"content": "Finally success"}}]},
        status=200
    )
    
    config = {
        "name": "m",
        "api_base": "http://api",
        "num_retries": 3,
        "temperature": 0.0
    }
    
    model = APIModel("m", config)
    
    # Чтобы тест не шел долго из-за экспоненциального отката, можно подменить time.sleep
    with patch("time.sleep", return_value=None):
        answer = model.generate_answer({"prompt": "..."})
    
    assert answer["choices"][0]["turns"][0]["content"] == "Finally success"
    assert len(responses.calls) == 2

def test_create_model_validation():
    """Тест: валидация при создании модели (покрытие веток create_model)."""
    # Ошибка: openai без api_base
    with pytest.raises(ValueError, match="api_base required"):
        create_model("m", {"api_type": "openai"})
    
    # Ошибка: ift без переменных окружения
    with patch.dict(os.environ, clear=True):
        if "GIGACHAT_IFT_CLIENT_ID" in os.environ:
             del os.environ["GIGACHAT_IFT_CLIENT_ID"]
        with pytest.raises(ValueError, match="api_type=ift requires env"):
            create_model("m", {"api_type": "ift"})
