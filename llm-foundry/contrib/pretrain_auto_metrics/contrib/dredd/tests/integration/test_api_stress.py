import pytest
import responses
import sys
import json
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
from src.models import APIModel
from src.judges.runner import JudgeRunner
from src.judges.judge import Judge

pytestmark = [pytest.mark.integration]

@responses.activate
def test_api_retry_count_and_failure():
    """Тест: строгий подсчет попыток. В develop num_retries - это ОБЩЕЕ число попыток."""
    responses.add(
        responses.POST,
        "http://mock-api/v1/chat/completions",
        status=500
    )
    
    num_retries = 2
    model_conf = {
        "name": "failing-model",
        "api_base": "http://mock-api/v1",
        "num_retries": num_retries,
        "temperature": 0.0
    }
    
    model = APIModel("failing-model", model_conf)
    
    with patch("time.sleep", return_value=None):
        # Now generate_answer raises RuntimeError after exhausting retries
        with pytest.raises(RuntimeError, match="API failed after 2 retries"):
            model.generate_answer({"prompt": "Hello"})
            
    # Проверяем количество запросов: в develop это ровно num_retries
    assert len(responses.calls) == num_retries

@responses.activate
def test_judge_runner_fail_on_error_integration(tmp_path):
    """Тест: JudgeRunner должен выбрасывать RuntimeError, если fail_on_error=True."""
    responses.add(responses.POST, "http://api/v1/chat/completions", status=500)
    
    judge_conf = {
        "model_name": "judge-m",
        "api_base": "http://api/v1",
        "num_retries": 1,
        "temperature": 0.0,
        "fail_on_error": True
    }
    from dredd import DATA_ROOT
    judge_obj = Judge(APIModel("judge-m", judge_conf), OmegaConf.create(judge_conf), project_root=DATA_ROOT)
    
    # 1. Создаем файлы вопросов и ответов
    q_file = tmp_path / "q.jsonl"
    cand_file = tmp_path / "cand.jsonl"
    base_file = tmp_path / "base.jsonl"
    
    with open(q_file, "w") as f: f.write(json.dumps({"uid": "1", "prompt": "p"}) + "\n")
    with open(cand_file, "w") as f: f.write(json.dumps({"uid": "1", "model": "cand", "choices": [{"turns": [{"content": "CANDIDATE ANSWER"}]}]}) + "\n")
    with open(base_file, "w") as f: f.write(json.dumps({"uid": "1", "model": "base", "choices": [{"turns": [{"content": "BASELINE ANSWER"}]}]}) + "\n")
    
    with patch.dict(sys.modules, {
        "gen_judgment": MagicMock(),
        "utils.judge_utils": MagicMock()
    }):
        runner = JudgeRunner(
            judge_obj,
            str(q_file),
            str(cand_file), str(base_file), str(tmp_path),
            candidate_model_name="cand_m",
            baseline_model_name="base_m",
            fail_on_error=True
        )
        
        # Мокаем вызов, чтобы он гарантированно упал в процессе выполнения
        with patch.object(runner, "_judge_with_retries", side_effect=Exception("API Death")):
            with patch("time.sleep", return_value=None):
                with pytest.raises(RuntimeError, match="Judging completed with 1 failed task"):
                    runner.run()
