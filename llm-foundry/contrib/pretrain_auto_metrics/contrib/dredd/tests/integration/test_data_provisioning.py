import pytest
import os
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import dredd

pytestmark = [pytest.mark.integration]

def test_init_data_root_nested_structure(tmp_path, monkeypatch):
    """Тест: корректная инициализация DATA_ROOT при вложенной структуре."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    
    # Имитируем: задаем DREDD_DATA_ROOT, где внутри есть папка data/arenas
    env_path = tmp_path / "env_root"
    nested_arenas = env_path / "data" / "arenas"
    nested_arenas.mkdir(parents=True)
    
    monkeypatch.setenv("DREDD_DATA_ROOT", str(env_path))
    monkeypatch.setattr(dredd, "PROJECT_ROOT", project_root)
    
    # Вызываем функцию инициализации (в тесте вызываем напрямую)
    new_root = dredd._init_data_root()
    
    # Она должна вернуть путь до вложенной папки data
    assert new_root == env_path / "data"
    assert (new_root / "arenas").exists()

def test_init_data_root_env_override(tmp_path, monkeypatch):
    """Тест: инициализация DATA_ROOT из переменной окружения (прямой путь)."""
    env_data_path = tmp_path / "env_data"
    (env_data_path / "arenas").mkdir(parents=True)
    
    monkeypatch.setenv("DREDD_DATA_ROOT", str(env_data_path))
    monkeypatch.setattr(dredd, "PROJECT_ROOT", tmp_path / "project")
    
    new_root = dredd._init_data_root()
    
    assert new_root == env_data_path
