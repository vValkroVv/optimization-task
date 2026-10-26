import pytest
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
import dredd

pytestmark = [pytest.mark.integration]

def _write_yaml(path, data):
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path

def test_main_complex_merge_and_validation(tmp_path, monkeypatch):
    """Тест: слияние 4 источников конфига и валидация через main()."""
    # 1. Дефолтный конфиг
    def_cfg_path = _write_yaml(tmp_path / "default.yaml", {
        "experiment": {"overwrite_generations": False},
        "model": {"name": "base", "temperature": 0.0},
        "datasets": [{"name": "arena"}]
    })
    
    # 2. Editable конфиг
    edit_cfg_path = _write_yaml(tmp_path / "editable.yaml", {
        "model": {"temperature": 0.7} # Перекрываем дефолт
    })
    
    # Мокаем константы в dredd.py
    monkeypatch.setattr(dredd, "DEFAULT_CONFIG", def_cfg_path)
    
    # Мокаем run(), чтобы проверить, что в него пришло
    captured_config = []
    def mock_run(config, phase):
        captured_config.append(config)
    
    monkeypatch.setattr(dredd, "run", mock_run)
    
    # 3. CLI аргументы: --overrides-json и позиционный аргумент
    overrides_json = json.dumps({"model": {"name": "cli-overridden"}})
    
    test_args = [
        "dredd.py",
        "--editable", str(edit_cfg_path),
        "--overrides-json", overrides_json,
        "model.temperature=1.0", # Финальное слово
        "--phase", "generation"
    ]
    
    monkeypatch.setattr(sys, "argv", test_args)
    
    # Запуск
    dredd.main()
    
    final_cfg = captured_config[0]
    # Проверяем иерархию слияния
    assert final_cfg.model.name == "cli-overridden"   # Из JSON
    assert final_cfg.model.temperature == 1.0        # Из позиционного аргумента (высший приоритет)
    assert final_cfg.experiment.overwrite_generations is False # Из дефолта

def test_main_rejects_invalid_overrides(tmp_path, monkeypatch):
    """Тест: main() выбрасывает ошибку при попытке переопределить несуществующий ключ."""
    def_cfg_path = _write_yaml(tmp_path / "default.yaml", {"model": {"name": "m"}})
    monkeypatch.setattr(dredd, "DEFAULT_CONFIG", def_cfg_path)
    
    # Опечатка в ключе: 'modell' вместо 'model'
    test_args = ["dredd.py", "modell.name=error"]
    monkeypatch.setattr(sys, "argv", test_args)
    
    with pytest.raises(ValueError, match="Invalid CLI parameters"):
        dredd.main()
