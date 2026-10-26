import pytest
import os
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf
from datetime import datetime

pytestmark = [pytest.mark.integration]

# Импортируем функции напрямую из dredd.py
import dredd

@pytest.fixture
def mock_project_structure(tmp_path):
    """Создает временную структуру проекта для тестов поиска генераций."""
    data_root = tmp_path / "data"
    model_data = data_root / "model_data"
    model_path = tmp_path / "my_model_workdir"
    
    # Создаем папки
    for p in [data_root, model_data, model_path]:
        p.mkdir(parents=True, exist_ok=True)
        
    # Подменяем константы в dredd.py через monkeypatch в тестах
    return {
        "data_root": data_root,
        "model_data": model_data,
        "model_path": model_path,
        "arena": "test_arena"
    }

def test_discovery_priority_default_exists(mock_project_structure, monkeypatch):
    """Тест: если есть default/generation.jsonl, он выбирается ВСЕГДА."""
    arena = mock_project_structure["arena"]
    model_path = mock_project_structure["model_path"]
    
    # Создаем папку default с файлом
    gen_dir = model_path / "artifacts" / "arenas" / arena / "generations"
    default_dir = gen_dir / "default"
    default_dir.mkdir(parents=True, exist_ok=True)
    gen_file = default_dir / "generation.jsonl"
    gen_file.touch()
    
    # Конфиг с другими параметрами
    model_conf = OmegaConf.create({
        "name": "test_model",
        "path": str(model_path),
        "temperature": 0.0  # В default может быть что угодно
    })
    
    # Вызываем поиск
    found = dredd._find_gen_file(model_conf, arena)
    
    assert found == gen_file
    assert "default" in str(found)

def test_discovery_matching_config(mock_project_structure):
    """Тест: если default нет, ищем папку с совпадающим config.yaml."""
    arena = mock_project_structure["arena"]
    model_path = mock_project_structure["model_path"]
    gen_dir = model_path / "artifacts" / "arenas" / arena / "generations"
    
    # Создаем "старый" run с другими параметрами
    old_run = gen_dir / "20240101_000000"
    old_run.mkdir(parents=True, exist_ok=True)
    (old_run / "config.yaml").write_text("temperature: 0.7\nmax_tokens: 100")
    (old_run / "generation.jsonl").touch()
    
    # Создаем "нужный" run
    match_run = gen_dir / "20240102_000000"
    match_run.mkdir(parents=True, exist_ok=True)
    (match_run / "config.yaml").write_text("temperature: 0.0\nmax_tokens: 50")
    (match_run / "generation.jsonl").touch()
    
    model_conf = OmegaConf.create({
        "name": "test_model",
        "path": str(model_path),
        "temperature": 0.0,
        "max_tokens": 50
    })
    
    found = dredd._find_gen_file(model_conf, arena)
    
    assert found == match_run / "generation.jsonl"

def test_discovery_exclusion_logic(mock_project_structure, monkeypatch):
    """Тест: если задан model.path, Dredd НЕ смотрит в model_data."""
    arena = mock_project_structure["arena"]
    model_data = mock_project_structure["model_data"]
    model_path = mock_project_structure["model_path"]
    
    # Кладем генерации в эталонную папку.
    # Reference-раскладка (model_data) НЕ содержит artifacts/ — он есть только в mode model.path.
    ref_dir = model_data / "my_model" / "arenas" / arena / "generations" / "default"
    ref_dir.mkdir(parents=True, exist_ok=True)
    (ref_dir / "generation.jsonl").touch()
    
    # Подменяем DATA_ROOT в dredd.py
    monkeypatch.setattr(dredd, "DATA_ROOT", mock_project_structure["data_root"])
    
    # Случай А: path задан (и пуст) -> не должен найти в референсах
    model_conf_with_path = OmegaConf.create({
        "name": "my_model",
        "path": str(model_path)
    })
    found = dredd._find_gen_file(model_conf_with_path, arena)
    assert found is None
    
    # Случай Б: path НЕ задан -> должен найти в референсах
    model_conf_no_path = OmegaConf.create({
        "name": "my_model"
    })
    found = dredd._find_gen_file(model_conf_no_path, arena)
    assert found is not None
    assert "model_data" in str(found)

def test_generate_logic_read_only_default(mock_project_structure, monkeypatch):
    """Тест: если найден default, _generate пропускает вызов GenerationRunner (read-only)."""
    arena = mock_project_structure["arena"]
    model_path = mock_project_structure["model_path"]
    
    # Создаем default
    gen_file = model_path / "artifacts" / "arenas" / arena / "generations" / "default" / "generation.jsonl"
    gen_file.parent.mkdir(parents=True, exist_ok=True)
    gen_file.touch()
    
    model_conf = OmegaConf.create({
        "name": "test_model",
        "path": str(model_path),
        "api_base": "http://mock"
    })
    
    with patch("dredd.GenerationRunner") as mock_runner:
        dredd._generate(model_conf, arena, Path("questions.jsonl"), overwrite=False)
        # Runner не должен быть вызван, так как default read-only
        mock_runner.assert_not_called()
