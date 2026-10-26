import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.judges.runner import JudgeRunner

pytestmark = [pytest.mark.unit]

@pytest.mark.parametrize("input_format", ["prompt_template", "structured_sbs_json"])
def test_judge_runner_automated_tie_on_identical_answers(tmp_path, input_format):
    # 1. Setup minimal mocks
    mock_judge = MagicMock()
    mock_judge.config = {"temperature": 0.0, "num_retries": 1, "input_format": input_format}
    mock_judge.model.name = "judge-model"
    mock_judge.model.api_base = "http://localhost"
    mock_judge.model.api_key = "key"
    mock_judge.model.config = {"api_type": "openai"}
    mock_judge.system_prompt = "sys"
    mock_judge.prompt_template = "tmpl"

    # Files
    q_file = tmp_path / "q.jsonl"
    q_file.write_text(json.dumps({"uid": "q1", "category": "test"}) + "\n")
    
    # Identical answers
    ans_data = json.dumps({
        "uid": "q1", 
        "model": "m", 
        "choices": [{"turns": [{"content": "SAME TEXT"}]}]
    }) + "\n"
    c_file = tmp_path / "c.jsonl"
    c_file.write_text(ans_data)
    b_file = tmp_path / "b.jsonl"
    b_file.write_text(ans_data)
    
    out_file = tmp_path / "out.jsonl"

    # 2. Initialize Runner
    with patch("src.judges.runner.load_jsonl", side_effect=[
        [{"uid": "q1", "category": "test"}], # questions
        [json.loads(ans_data)], # cand
        [json.loads(ans_data)]  # base
    ]), patch("src.judges.runner.JudgeRunner.__init__", return_value=None):
        runner = JudgeRunner()
        runner.judge = mock_judge
        runner.questions_path = q_file
        runner.candidate_path = c_file
        runner.baseline_path = b_file
        runner.output_file = out_file
        runner.failed_output_file = tmp_path / "failed_judgments.jsonl"
        runner.overwrite = True
        runner.parallel_api_calls = 1
        runner.candidate_model_name = "cand"
        runner.baseline_model_name = "base"
        runner.category_namespace = ""
        runner.judge_settings = {"test": {"system_prompt": "sys"}}
        runner.fail_on_error = False

        # Mocking internal methods that we don't want to test here
        runner._normalize_question_for_judgment = lambda x: x
        runner._normalize_answer_for_judgment = lambda x: x
        runner._extract_answer_text = lambda x: x["choices"][0]["turns"][0]["content"]
        runner._normalize_answer_text_for_compare = lambda x: x.strip()
        runner._text_hash = lambda x: "hash"
        runner._annotate_game_roles_and_hashes = MagicMock()
        runner._ensure_categories_in_judge_settings = MagicMock()
        
        # 3. RUN
        stats = runner.run()

    # 4. ASSERTIONS
    assert stats["succeeded"] == 1
    assert out_file.exists()
    with open(out_file, "r") as f:
        result = json.loads(f.read())
    
    assert result["uid"] == "q1"
    assert result["games"][0]["score"] == "[[A=B]]"
    assert "Automated Tie" in result["games"][0]["judgment"]
    print("\nAutomated Tie test passed!")
