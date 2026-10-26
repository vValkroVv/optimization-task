import pytest
import json
from pathlib import Path
from src.utils.judgment_summary import compute_judgment_summary

pytestmark = [pytest.mark.integration]

def test_summary_advanced_resolving_duplicates(tmp_path):
    """Тест: при наличии нескольких записей для одного uid используется последняя.
    Также проверяем новый классификатор со знаком <."""
    p = tmp_path / "judgment.jsonl"
    
    # 1. Первая запись: победа кандидата через A>B
    r1_win_cand = {
        "uid": "q1",
        "games": [
            {"score": "A>B", "role_a": "candidate", "role_b": "baseline"}, # A(cand) > B(base) -> Cand win
            {"score": "B>A", "role_a": "baseline", "role_b": "candidate"}  # B(cand) > A(base) -> Cand win
        ]
    }
    
    # 2. Вторая запись для того же uid: победа бейзлайна
    r1_win_base = {
        "uid": "q1",
        "games": [
            {"score": "A<B", "role_a": "candidate", "role_b": "baseline"}, # A(cand) < B(base) -> Base win
            {"score": "A>B", "role_a": "baseline", "role_b": "candidate"}  # A(base) > B(cand) -> Base win
        ]
    }
    
    with open(p, "w") as f:
        f.write(json.dumps(r1_win_cand) + "\n")
        f.write(json.dumps(r1_win_base) + "\n")
    
    summary = compute_judgment_summary(p)
    
    # В итоге должна быть 1 победа бейзлайна (из второй записи)
    assert summary["total"] == 1
    assert summary["baseline_wins"] == 1
    assert summary["candidate_wins"] == 0

def test_summary_deprecation_warning(tmp_path, caplog):
    """Тест: при отсутствии поля 'score' выдается предупреждение о депрекации."""
    p = tmp_path / "judgment.jsonl"
    
    r_old = {"uid": "q3", "judgment": "The winner is [[A>B]]"}
    
    with open(p, "w") as f:
        f.write(json.dumps(r_old) + "\n")
        
    import logging
    with caplog.at_level(logging.WARNING):
        summary = compute_judgment_summary(p)
        
    assert "DEPRECATION WARNING" in caplog.text
    assert summary["total"] == 1

def test_summary_fallback_roles(tmp_path, caplog):
    """Тест: при отсутствии role_a/role_b используется fallback-логика."""
    p = tmp_path / "judgment.jsonl"
    
    r_no_roles = {
        "uid": "q2",
        "games": [
            {"score": "B>A"}, # Fallback Game 1: A=base, B=cand -> Cand wins
            {"score": "A>B"}  # Fallback Game 2: A=cand, B=base -> Cand wins
        ]
    }
    
    with open(p, "w") as f:
        f.write(json.dumps(r_no_roles) + "\n")
    
    summary = compute_judgment_summary(p)
    
    assert summary["candidate_wins"] == 1
    assert "Role metadata is missing" in caplog.text

def test_summary_old_single_row_format(tmp_path):
    """Тест: поддержка старого формата (одна строка без списка 'games')."""
    p = tmp_path / "judgment.jsonl"
    
    r_old = {"uid": "q3", "score": "A>B"} 
    r_old_text = {"uid": "q4", "judgment": "The winner is [[B>A]]"}
    
    with open(p, "w") as f:
        f.write(json.dumps(r_old) + "\n")
        f.write(json.dumps(r_old_text) + "\n")
        
    summary = compute_judgment_summary(p)
    assert summary["candidate_wins"] == 1
    assert summary["baseline_wins"] == 1
    assert summary["total"] == 2
