from pathlib import Path

import pytest

from src.utils.io import get_existing_ids, load_jsonl, normalize_question

pytestmark = [pytest.mark.unit]


def test_load_jsonl_returns_empty_for_missing_file(tmp_path: Path):
    missing = tmp_path / "missing.jsonl"
    assert load_jsonl(str(missing)) == []


def test_load_jsonl_skips_invalid_lines(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"question_id": 1}\nnot-json\n{"uid":"u-2"}\n', encoding="utf-8")

    rows = load_jsonl(str(path))
    assert rows == [{"question_id": 1}, {"uid": "u-2"}]


def test_get_existing_ids_extracts_question_id_and_uid(tmp_path: Path, write_jsonl):
    path = write_jsonl(
        tmp_path / "rows.jsonl",
        [
            {"question_id": 1},
            {"uid": "u-1"},
            {"question_id": "2"},
            {"no_id": True},
        ],
    )

    assert get_existing_ids(str(path)) == {"1", "u-1", "2"}


def test_normalize_question_accepts_prompt_shape():
    normalized = normalize_question({"uid": "abc", "prompt": "hello", "category": "coding"})

    assert normalized["question_id"] == "abc"
    assert normalized["turns"] == [{"content": "hello", "role": "user"}]
    assert normalized["category"] == "coding"


def test_normalize_question_accepts_turns_string_and_dict():
    normalized = normalize_question(
        {
            "question_id": 7,
            "turns": ["turn-1", {"content": "turn-2", "role": "assistant"}],
            "subcategory": "math",
        }
    )

    assert normalized["question_id"] == "7"
    assert normalized["turns"] == [
        {"content": "turn-1", "role": "user"},
        {"content": "turn-2", "role": "assistant"},
    ]
    assert normalized["subcategory"] == "math"


def test_normalize_question_fails_without_supported_identifier():
    with pytest.raises(ValueError, match="question_id"):
        normalize_question({"prompt": "hello"})


def test_normalize_question_fails_on_empty_turns():
    with pytest.raises(ValueError, match="turns"):
        normalize_question({"question_id": "q-1", "turns": []})
