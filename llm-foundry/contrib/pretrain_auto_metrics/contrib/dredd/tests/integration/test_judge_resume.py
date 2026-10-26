import pytest
import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import dredd
from src.judges.runner import JudgeRunner

@pytest.fixture(autouse=True)
def mock_judge_dependencies(monkeypatch):
    """Мокаем внешние зависимости, которые мешают инициализации JudgeRunner"""
    import types
    import sys
    
    # Мокаем gen_judgment
    gen_mod = types.ModuleType("gen_judgment")
    def mock_judgment(args_dict):
        output_file = Path(args_dict["output_file"])
        qid = args_dict["question"].get("uid")
        with open(output_file, "w") as f:
            f.write(json.dumps({"uid": qid, "games": [{"score": "A>B"}]}) + "\n")
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
@pytest.mark.skip(reason="JudgeRunner currently lacks smart resume. Enable when JudgeRunner is refactored to match GenerationRunner logic.")
def test_judge_resume_logic(tmp_path):
    """
    Проверка дозапуска судейства.
    ОЖИДАНИЕ (БУДУЩЕЕ): Dredd должен уметь дописывать файл судейства.
    РЕАЛЬНОСТЬ (СЕЙЧАС): JudgeRunner скипает выполнение, если файл существует.
    """
    project_dir = tmp_path / "judge_resume"
    project_dir.mkdir()
    
    # 1. Готовим данные
    questions_file = project_dir / "questions.jsonl"
    questions = [{"uid": f"q{i}", "category": "test"} for i in range(3)]
    with open(questions_file, "w") as f:
        for q in questions: f.write(json.dumps(q) + "\n")
        
    cand_file = project_dir / "cand.jsonl"
    with open(cand_file, "w") as f:
        for q in questions: 
            f.write(json.dumps({"uid": q["uid"], "model": "cand", "choices": [{"turns": [{"content": "CANDIDATE ANSWER"}]}]}) + "\n")
            
    base_file = project_dir / "base.jsonl"
    with open(base_file, "w") as f:
        for q in questions: 
            f.write(json.dumps({"uid": q["uid"], "model": "base", "choices": [{"turns": [{"content": "BASELINE ANSWER"}]}]}) + "\n")

    output_file = project_dir / "judgments.jsonl"
    
    # Создаем фейковый объект Judge
    mock_judge = MagicMock()
    mock_judge.config = {"temperature": 0.0, "num_retries": 1}
    mock_judge.model.name = "judge-model"
    mock_judge.model.api_base = "http://localhost"
    mock_judge.model.api_key = "key"
    mock_judge.model.config = {"api_type": "openai"}
    mock_judge.system_prompt = "sys"
    mock_judge.prompt_template = "tmpl"
    
    # 2. ПЕРВЫЙ ЗАПУСК (ПРЕРВАННЫЙ)
    call_count = 0
    import sys
    original_judgment = sys.modules["gen_judgment"].judgment
    
    def interrupted_judgment(args_dict):
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            raise RuntimeError("Stop")
        return original_judgment(args_dict)

    with patch("gen_judgment.judgment", side_effect=interrupted_judgment):
        runner = JudgeRunner(
            judge=mock_judge,
            questions_path=str(questions_file),
            candidate_path=str(cand_file),
            baseline_path=str(base_file),
            output_dir=str(project_dir),
            candidate_model_name="cand",
            baseline_model_name="base",
            overwrite=True,
            output_file=str(output_file)
        )
        runner.external_repo_path = project_dir
        try:
            runner.run()
        except RuntimeError:
            pass

    # Проверяем, что в файле 1 запись
    assert output_file.exists(), "Judgments file should have been created"
    with open(output_file, "r") as f:
        lines = f.readlines()
    assert len(lines) == 1, f"Expected 1 line before resume, got {len(lines)}"

    # 3. ВТОРОЙ ЗАПУСК (ПРОДОЛЖЕНИЕ)
    # overwrite=False должен позволить дозапись
    with patch("gen_judgment.judgment", side_effect=original_judgment):
        runner_resume = JudgeRunner(
            judge=mock_judge,
            questions_path=str(questions_file),
            candidate_path=str(cand_file),
            baseline_path=str(base_file),
            output_dir=str(project_dir),
            candidate_model_name="cand",
            baseline_model_name="base",
            overwrite=False,
            output_file=str(output_file)
        )
        runner_resume.external_repo_path = project_dir
        runner_resume.run()

    # ПРОВЕРКА: В файле должно стать 3 записи
    with open(output_file, "r") as f:
        final_lines = f.readlines()
    
    # ОЖИДАНИЕ: 3 записи.
    assert len(final_lines) == 3, f"Resume failed! Expected 3 lines, got {len(final_lines)}"

@pytest.mark.integration
@pytest.mark.skip(reason="JudgeRunner currently lacks config-based versioning. Enable when judged runs are versioned like generations.")
def test_judge_new_identity_on_param_change(tmp_path):
    """
    Проверка того, что при изменении параметров судьи (температура)
    Dredd создает НОВУЮ директорию для судейства.
    ОЖИДАНИЕ (БУДУЩЕЕ): Разные параметры = разные папки.
    РЕАЛЬНОСТЬ (СЕЙЧАС): Путь судейства захардкожен и не зависит от параметров.
    """
    project_dir = tmp_path / "judge_identity"
    project_dir.mkdir()
    
    # 1. Данные
    q_file = project_dir / "questions.jsonl"
    with open(q_file, "w") as f: f.write(json.dumps({"uid": "q1", "category": "test"}) + "\n")
    c_file = project_dir / "cand.jsonl"
    with open(c_file, "w") as f: f.write(json.dumps({"uid": "q1", "model": "cand", "choices": [{"turns": [{"content": "C"}]}]}) + "\n")
    b_file = project_dir / "base.jsonl"
    with open(b_file, "w") as f: f.write(json.dumps({"uid": "q1", "model": "base", "choices": [{"turns": [{"content": "B"}]}]}) + "\n")

    # Конфиг Dredd (через OmegaConf)
    from omegaconf import OmegaConf
    config = OmegaConf.create({
        "datasets": [{
            "name": "test-arena",
            "model": {"name": "model_a", "path": str(project_dir / "model_a")},
            "baseline": {"name": "model_b", "path": str(project_dir / "model_b")},
            "judge": {
                "model_name": "judge_v1",
                "temperature": 0.0,
                "num_retries": 1
            },
            "experiment": {"overwrite_judgments": False}
        }]
    })
    
    # Мокаем зависимости внутри dredd.run
    with patch("dredd.DATA_ROOT", project_dir), \
         patch("dredd.get_questions_path", return_value=q_file), \
         patch("dredd._find_gen_file", side_effect=[c_file, b_file]), \
         patch("dredd.Judge"), \
         patch("dredd.APIModel"), \
         patch("dredd.JudgeRunner") as MockRunner:
        
        # Мокаем поведение runner.run()
        mock_instance = MockRunner.return_value
        mock_instance.run.return_value = {"succeeded": 1}
        
        # 1. Запуск V1
        dredd.run(config, phase="judgement")
        
        # Смотрим, какой путь передали в JudgeRunner
        args_v1, kwargs_v1 = MockRunner.call_args
        path_v1 = kwargs_v1.get("output_file") or args_v1[8] # 9-й аргумент
        
        # 2. Меняем температуру судьи
        config.datasets[0].judge.temperature = 0.7
        
        # Запуск V2
        dredd.run(config, phase="judgement")
        
        args_v2, kwargs_v2 = MockRunner.call_args
        path_v2 = kwargs_v2.get("output_file") or args_v2[8]
        
        # ПРОВЕРКА: Пути должны быть РАЗНЫМИ
        assert path_v1 != path_v2, f"Expected different paths for different judge params. Got both: {path_v1}"

import sys
