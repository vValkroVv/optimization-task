from pathlib import Path

import pytest

from src.utils.judgment_summary import (
    compute_judgment_summary,
    format_summary_table,
    save_judgment_summary_jsonl,
)

pytestmark = [pytest.mark.unit]


def test_compute_judgment_summary_returns_zeroes_for_missing_file(tmp_path: Path):
    summary = compute_judgment_summary(tmp_path / "missing.jsonl")
    assert summary == {
        "total": 0,
        "candidate_wins": 0,
        "baseline_wins": 0,
        "ties": 0,
        "win_rate": 0.0,
        "loss_rate": 0.0,
        "tie_rate": 0.0,
        "score_0_100": 0.0,
    }


def test_compute_judgment_summary_groups_by_question_and_uses_latest_row(tmp_path: Path, write_jsonl):
    path = write_jsonl(
        tmp_path / "judgment.jsonl",
        [
            {"uid": "q1", "score": "A>B"},  # old row, should be ignored
            {"uid": "q1", "score": "B>A"},  # latest row for q1: baseline win
            {"uid": "q2", "games": [{"score": "B>A"}, {"score": "A>B"}]},  # candidate win by role mapping
            {"uid": "q3", "judgment": "analysis [[A=B]]"},  # tie
        ],
    )

    summary = compute_judgment_summary(path)

    assert summary["total"] == 3
    assert summary["candidate_wins"] == 1
    assert summary["baseline_wins"] == 1
    assert summary["ties"] == 1
    assert summary["score_0_100"] == 50.0


def test_save_summary_jsonl_and_format_table(tmp_path: Path, read_jsonl):
    summary = {
        "total": 10,
        "candidate_wins": 6,
        "baseline_wins": 2,
        "ties": 2,
        "win_rate": 60.0,
        "loss_rate": 20.0,
        "tie_rate": 20.0,
        "score_0_100": 70.0,
    }
    output = tmp_path / "summary.jsonl"

    save_judgment_summary_jsonl(summary, output, arena_name="arena-x", baseline_name="gpt-4")
    rows = read_jsonl(output)

    assert len(rows) == 1
    assert rows[0]["arena"] == "arena-x"
    assert rows[0]["baseline"] == "gpt-4"

    table = format_summary_table({"arena-x": {**summary, "failed_judgments": 1}})
    assert "arena-x" in table
    assert "Judge errors: 1" in table
