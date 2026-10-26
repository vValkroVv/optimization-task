import pytest
import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import dredd
from omegaconf import OmegaConf

@pytest.fixture(autouse=True)
def mock_judge_dependencies(monkeypatch):
    """Мокаем внешние зависимости, которые мешают инициализации JudgeRunner"""
    import types
    import sys
    
    # Мокаем gen_judgment
    gen_mod = types.ModuleType("gen_judgment")
    def mock_judgment(args_dict):
        # Эта функция НЕ должна вызываться в нашем тесте, если пре-чек работает!
        raise RuntimeError("API was called, but automated tie should have prevented it!")
    gen_mod.judgment = mock_judgment
    monkeypatch.setitem(sys.modules, "gen_judgment", gen_mod)
    
    # Мокаем utils.judge_utils
    utils_mod = types.ModuleType("utils")
    utils_mod.__path__ = []
    judge_utils = types.ModuleType("utils.judge_utils")
    judge_utils.JUDGE_SETTINGS = {"test": {"baseline": "base", "system_prompt": "prompt"}}
    monkeypatch.setitem(sys.modules, "utils", utils_mod)
    monkeypatch.setitem(sys.modules, "utils.judge_utils", judge_utils)
    
    # Предотвращаем импорт gigachat
    monkeypatch.setitem(sys.modules, "gigachat", MagicMock())

@pytest.mark.integration
def test_automated_tie_end_to_end_scoring(tmp_path):
    """
    Интеграционный тест: 
    1. Ответы моделей идентичны.
    2. Запускаем судейство через dredd.run.
    3. Проверяем, что API судьи НЕ вызывался.
    4. Проверяем, что итоговый скор ровно 50.0 (паритет).
    """
    project_dir = tmp_path / "tie_project"
    project_dir.mkdir()
    
    # 1. Готовим данные (10 вопросов с идентичными ответами)
    arena_name = "test-arena"
    data_dir = project_dir / "arenas"
    data_dir.mkdir(parents=True)
    q_file = data_dir / f"{arena_name}.jsonl"
    
    questions = [{"uid": f"q{i}", "category": "test", "prompt": f"Q {i}"} for i in range(10)]
    with open(q_file, "w") as f:
        for q in questions: f.write(json.dumps(q) + "\n")
        
    # Пути для моделей
    model_a_path = project_dir / "model_a"
    model_a_path.mkdir()
    model_b_path = project_dir / "model_b"
    model_b_path.mkdir()
    
    # Создаем идентичные генерации
    def create_gen(model_path, model_name):
        gen_dir = model_path / "artifacts" / "arenas" / arena_name / "generations" / "default"
        gen_dir.mkdir(parents=True)
        gen_file = gen_dir / "generation.jsonl"
        with open(gen_file, "w") as f:
            for q in questions:
                f.write(json.dumps({
                    "uid": q["uid"],
                    "model": model_name,
                    "choices": [{"turns": [{"content": "SAME IDENTICAL RESPONSE"}]}]
                }) + "\n")
        return gen_file

    create_gen(model_a_path, "model_a")
    create_gen(model_b_path, "model_b")

    # 2. Конфиг Dredd
    config = OmegaConf.create({
        "datasets": [{
            "name": arena_name,
            "model": {"name": "model_a", "path": str(model_a_path)},
            "baseline": {"name": "model_b", "path": str(model_b_path)},
            "judge": {
                "model_name": "judge_v1",
                "temperature": 0.0,
                "num_retries": 1
            },
            "experiment": {"overwrite_judgments": True}
        }],
        "merged_config_path": str(project_dir / "merged_config.yaml")
    })
    
    # 3. ЗАПУСК
    # Настраиваем мок Judge более детально
    mock_judge_inst = MagicMock()
    mock_judge_inst.config.get.side_effect = lambda k, default=None: {
        "regex_pattern": r"\[\[([AB<>=]+)\]\]",
        "temperature": 0.0,
        "max_tokens": 4096,
        "num_retries": 1
    }.get(k, default)
    mock_judge_inst.model.name = "judge-model"
    mock_judge_inst.model.api_base = "http://localhost"
    mock_judge_inst.model.api_key = "key"
    mock_judge_inst.model.config.get.return_value = "openai"
    mock_judge_inst.system_prompt = "sys"
    mock_judge_inst.prompt_template = "tmpl"

    with patch("dredd.DATA_ROOT", project_dir), \
         patch("dredd.get_questions_path", return_value=q_file), \
         patch("dredd.Judge", return_value=mock_judge_inst), \
         patch("dredd.APIModel"):
        
        # Запускаем полную фазу судейства
        dredd.run(config, phase="judgement")

    # 4. ПРОВЕРКА РЕЗУЛЬТАТОВ
    judgment_file = model_a_path / "artifacts" / "arenas" / arena_name / "judgments" / "model_b" / "judgment.jsonl"
    summary_file = judgment_file.parent / "judgment_summary.jsonl"
    
    assert judgment_file.exists(), "Judgment file was not created!"
    assert summary_file.exists(), "Summary file was not created!"
    
    # Читаем саммари
    with open(summary_file, "r") as f:
        summary = json.loads(f.read())
    
    print(f"\nFinal Stats: {summary}")
    
    assert summary["total"] == 10
    assert summary["ties"] == 10
    assert summary["candidate_wins"] == 0
    assert summary["baseline_wins"] == 0
    assert summary["score_0_100"] == 50.0  # ПАРИТЕТ!
    
    # Также проверим, что в judgment_file записано правильное обоснование
    with open(judgment_file, "r") as f:
        first_record = json.loads(f.readline())
        assert "Automated Tie" in first_record["games"][0]["judgment"]

    print("\nEnd-to-end automated tie test passed perfectly!")
