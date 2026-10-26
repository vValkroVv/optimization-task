import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf
import dredd

pytestmark = [pytest.mark.integration]

@pytest.fixture
def mock_run_env(tmp_path):
    """Создает окружение для теста функции run()."""
    arena_file = tmp_path / "data" / "arenas" / "test_arena.jsonl"
    arena_file.parent.mkdir(parents=True, exist_ok=True)
    with open(arena_file, "w") as f:
        f.write(json.dumps({"question_id": "1", "prompt": "test"}) + "\n")
    
    # Константы путей
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(dredd, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(dredd, "PROJECT_ROOT", tmp_path)
    
    config = OmegaConf.create({
        "merged_config_path": str(tmp_path / "merged.yaml"),
        "datasets": [
            {
                "name": "test_arena",
                "model": {"name": "m1", "path": str(tmp_path / "m1")},
                "baseline": {"name": "b1", "path": str(tmp_path / "b1")},
                "judge": {"model_name": "j1"}
            }
        ],
        "experiment": {"overwrite_generations": False, "overwrite_judgments": False},
        "judge": {"model_name": "j1", "api_base": "http://mock"},
        "model": {"name": "default_m"},
        "baseline": {"name": "default_b"}
    })
    
    return config, tmp_path

def test_run_phases_generation_only(mock_run_env):
    """Тест: фаза 'generation' вызывает только _generate."""
    config, tmp_path = mock_run_env
    
    with patch("dredd._generate") as mock_gen, \
         patch("dredd.JudgeRunner") as mock_judge, \
         patch("dredd.compute_judgment_summary") as mock_summary:
        
        dredd.run(config, phase="generation")
        
        # _generate должен быть вызван для модели и бейзлайна
        assert mock_gen.call_count == 2
        # JudgeRunner НЕ должен быть вызван
        mock_judge.assert_not_called()

def test_run_phases_judgement_only(mock_run_env):
    """Тест: фаза 'judgement' НЕ вызывает _generate."""
    config, tmp_path = mock_run_env
    
    # Создаем фиктивные файлы генераций, чтобы JudgeRunner не упал
    m1_gen = tmp_path / "m1" / "artifacts" / "arenas" / "test_arena" / "generations" / "default" / "generation.jsonl"
    b1_gen = tmp_path / "b1" / "artifacts" / "arenas" / "test_arena" / "generations" / "default" / "generation.jsonl"
    m1_gen.parent.mkdir(parents=True, exist_ok=True)
    b1_gen.parent.mkdir(parents=True, exist_ok=True)
    m1_gen.touch()
    b1_gen.touch()

    with patch("dredd._generate") as mock_gen, \
         patch("dredd.JudgeRunner") as mock_judge, \
         patch("dredd.compute_judgment_summary") as mock_summary:
        
        mock_judge.return_value.run.return_value = {"succeeded": 1}
        mock_summary.return_value = {"win_rate": 50}
        
        dredd.run(config, phase="judgement")
        
        # _generate НЕ должен быть вызван
        mock_gen.assert_not_called()
        # JudgeRunner должен быть вызван
        mock_judge.assert_called_once()

def test_run_full_cycle(mock_run_env):
    """Тест: фаза 'full' вызывает и генерацию, и судейство."""
    config, tmp_path = mock_run_env
    
    # Создаем фиктивные файлы генераций (имитируем успешную генерацию)
    m1_gen = tmp_path / "m1" / "artifacts" / "arenas" / "test_arena" / "generations" / "default" / "generation.jsonl"
    b1_gen = tmp_path / "b1" / "artifacts" / "arenas" / "test_arena" / "generations" / "default" / "generation.jsonl"
    m1_gen.parent.mkdir(parents=True, exist_ok=True)
    b1_gen.parent.mkdir(parents=True, exist_ok=True)
    m1_gen.touch()
    b1_gen.touch()

    with patch("dredd._generate") as mock_gen, \
         patch("dredd.JudgeRunner") as mock_judge, \
         patch("dredd.compute_judgment_summary") as mock_summary:
        
        mock_judge.return_value.run.return_value = {"succeeded": 1}
        mock_summary.return_value = {"win_rate": 100}
        
        dredd.run(config, phase="full")
        
        # Должны быть вызваны оба блока
        assert mock_gen.call_count == 2
        assert mock_judge.call_count == 1
