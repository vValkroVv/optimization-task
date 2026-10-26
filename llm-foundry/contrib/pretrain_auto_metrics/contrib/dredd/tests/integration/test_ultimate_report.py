import pytest
import json
from pathlib import Path
from src.utils.judgment_summary import compute_judgment_summary, format_summary_table

pytestmark = [pytest.mark.integration]

def test_ultimate_report_soft_verification(tmp_path, capsys):
    """Тест: проверка итогового отчета (таблица) и расширенных метаданных."""
    p = tmp_path / "judgment.jsonl"
    
    results = [
        {"uid": "q1", "games": [
            {"score": "A>B", "role_a": "candidate", "role_b": "baseline"},
            {"score": "B>A", "role_a": "baseline", "role_b": "candidate"}
        ]},
        {"uid": "q2", "games": [
            {"score": "B>A", "role_a": "baseline", "role_b": "candidate"},
            {"score": "A>B", "role_a": "candidate", "role_b": "baseline"}
        ]}
    ]
    
    with open(p, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    summary = compute_judgment_summary(p)
    
    benchmark_results = {"Test Arena": summary}
    table_text = format_summary_table(benchmark_results)
    print(table_text)
    
    captured = capsys.readouterr().out.lower()
    
    # МЯГКАЯ ПРОВЕРКА (без учета регистра):
    assert "test arena" in captured
    assert "100.0" in captured
    assert "candidate" in captured
    assert "baseline" in captured
    
def test_metadata_preservation_in_summary(tmp_path):
    """Тест: проверка сохранения категорий и кластеров в итоговом JSONL-отчете."""
    p = tmp_path / "judgment.jsonl"
    results = [
        {"uid": "q1", "category": "science", "cluster": "physics", "score": "A>B", "role_a": "candidate", "role_b": "baseline"}
    ]
    with open(p, "w") as f:
        f.write(json.dumps(results[0]) + "\n")
        
    summary = compute_judgment_summary(p)
    assert summary["total"] == 1
