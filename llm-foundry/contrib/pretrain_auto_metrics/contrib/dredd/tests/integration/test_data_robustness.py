import pytest
import json
from src.utils.judgment_summary import compute_judgment_summary

@pytest.mark.integration
def test_data_robustness_ideal(tmp_path):
    """Случай 1: Идеальные данные."""
    p = tmp_path / "ideal.jsonl"
    data = [{"uid": f"q{i}", "score": "A>B", "games": [{"score": "A>B", "role_a": "candidate", "role_b": "baseline"}]} for i in range(10)]
    with open(p, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
    
    summary = compute_judgment_summary(p)
    assert summary["total"] == 10
    assert summary["candidate_wins"] == 10

def test_data_robustness_broken_json(tmp_path, caplog):
    """Случай 2: Битый JSON. Ожидаем, что Dredd пропустит битую строку и не упадет."""
    p = tmp_path / "broken.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps({"uid": "q1", "score": "A>B", "games": [{"score": "A>B", "role_a": "candidate", "role_b": "baseline"}]}) + "\n")
        f.write('{"uid": "q2", "score": "A>B", "games": [INVALID_JSON]}\n') # Битый JSON
        f.write(json.dumps({"uid": "q3", "score": "A>B", "games": [{"score": "A>B", "role_a": "candidate", "role_b": "baseline"}]}) + "\n")
    
    summary = compute_judgment_summary(p)
    # Ожидаем, что 2 записи из 3 обработаны
    assert summary["total"] == 2
    assert "Error parsing judgment" in caplog.text

def test_data_robustness_empty_lines(tmp_path):
    """Случай 3: Пустые строки. Ожидаем, что Dredd их просто проигнорирует."""
    p = tmp_path / "empty.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps({"uid": "q1", "score": "A>B", "games": [{"score": "A>B", "role_a": "candidate", "role_b": "baseline"}]}) + "\n")
        f.write("\n\n") # Пустые строки
        f.write(json.dumps({"uid": "q2", "score": "A>B", "games": [{"score": "A>B", "role_a": "candidate", "role_b": "baseline"}]}) + "\n")
    
    summary = compute_judgment_summary(p)
    assert summary["total"] == 2

def test_data_robustness_legacy_no_roles(tmp_path, caplog):
    """Случай 4: Старые записи без метаданных ролей. Ожидаем fallback на A=candidate, B=baseline."""
    p = tmp_path / "legacy.jsonl"
    # В этом формате нет role_a/role_b в games. 
    # Dredd должен использовать fallback_roles=(candidate, baseline) по умолчанию
    data = [{"uid": f"q{i}", "games": [{"score": "A>B"}]} for i in range(5)]
    with open(p, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
            
    summary = compute_judgment_summary(p)
    assert summary["total"] == 5
    assert summary["candidate_wins"] == 5
    assert "Role metadata is missing" in caplog.text
