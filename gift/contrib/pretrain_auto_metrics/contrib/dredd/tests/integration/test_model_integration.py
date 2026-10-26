import pytest
import responses
from pathlib import Path
from unittest.mock import MagicMock
from src.models import APIModel
from src.utils.io import load_prompt

pytestmark = [pytest.mark.integration]

@pytest.fixture
def temp_prompts(tmp_path):
    """Создает временные файлы промтов."""
    p1 = tmp_path / "sys_v1.txt"
    p1.write_text("You are an expert coder.")
    return tmp_path

def test_load_prompt_resolution(temp_prompts):
    """Тест: load_prompt правильно определяет текст vs путь."""
    # 1. Просто текст
    assert load_prompt("Hello") == "Hello"
    
    # 2. Абсолютный путь
    abs_path = temp_prompts / "sys_v1.txt"
    assert load_prompt(str(abs_path)) == "You are an expert coder."
    
    # 3. Относительный путь от корня поиска
    assert load_prompt("sys_v1.txt", temp_prompts) == "You are an expert coder."
    
    # 4. Несуществующий файл -> текст
    assert load_prompt("missing.txt", temp_prompts) == "missing.txt"

@responses.activate
def test_api_model_payload_with_system_prompt(temp_prompts):
    """Тест: APIModel правильно включает системный промт в payload."""
    responses.add(
        responses.POST,
        "http://mock-api/chat/completions",
        json={"choices": [{"message": {"content": "Hello!"}}]},
        status=200
    )
    
    # Случай А: системный промт через путь
    sys_file = temp_prompts / "expert.txt"
    sys_file.write_text("Expert prompt from file.")
    
    model_conf = {
        "name": "test-model",
        "api_base": "http://mock-api",
        "system_prompt": str(sys_file),
        "temperature": 0.0
    }
    
    model = APIModel("test-model", model_conf)
    model.generate_answer({"question_id": "1", "prompt": "Hi"})
    
    # Проверяем, что в запросе был правильный системный промт
    sent_payload = responses.calls[0].request.body.decode()
    assert '"role": "system"' in sent_payload
    assert "Expert prompt from file." in sent_payload

def test_api_model_deprecated_path_to_sys(caplog):
    """Тест: использование path_to_sys вызывает предупреждение, но работает."""
    model_conf = {
        "name": "test-model",
        "path_to_sys": "I am a legacy prompt",
        "api_base": "http://mock-api"
    }
    
    model = APIModel("test-model", model_conf)
    # Проверяем предупреждение при генерации (т.к. load_prompt вызывается там)
    with responses.RequestsMock() as rsps:
        rsps.add(responses.POST, "http://mock-api/chat/completions", json={"choices": [{"message": {"content": "..."}}]}, status=200)
        model.generate_answer({"prompt": "..."})
        
    assert "deprecated" in caplog.text.lower()
    assert "Use 'system_prompt' instead" in caplog.text
