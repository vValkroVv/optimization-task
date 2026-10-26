from pathlib import Path
from typing import Dict, List

import pytest

from src.generators.runner import GenerationRunner
from src.utils.io import load_jsonl

pytestmark = [pytest.mark.integration]


class FakeModel:
    def __init__(self, name: str = "fake-model"):
        self.name = name
        self.generated_for: List[str] = []

    def generate_answer(self, question: Dict) -> Dict:
        qid = str(question.get("question_id") or question.get("uid"))
        self.generated_for.append(qid)
        key = "question_id" if "question_id" in question else "uid"
        return {
            key: qid,
            "model_id": self.name,
            "choices": [{"index": 0, "turns": [{"content": f"answer-{qid}", "role": "assistant"}]}],
        }


def test_generation_runner_resumes_and_processes_only_missing_ids(tmp_path: Path, write_jsonl):
    questions_path = write_jsonl(
        tmp_path / "questions.jsonl",
        [
            {"question_id": "1", "turns": [{"content": "q1"}]},
            {"question_id": "2", "turns": [{"content": "q2"}]},
            {"uid": "3", "prompt": "q3"},
        ],
    )
    output_file = write_jsonl(
        tmp_path / "out" / "generation.jsonl",
        [{"question_id": "1", "model_id": "fake-model", "choices": [{"turns": [{"content": "old"}]}]}],
    )

    model = FakeModel()
    runner = GenerationRunner(
        model=model,
        questions_path=str(questions_path),
        output_dir=str(output_file.parent),
        overwrite=False,
        output_file=str(output_file),
        parallel_api_calls=1,
    )
    runner.run()

    rows = load_jsonl(str(output_file))
    ids = {str(r.get("question_id") or r.get("uid")) for r in rows}
    assert ids == {"1", "2", "3"}
    assert set(model.generated_for) == {"2", "3"}


def test_generation_runner_overwrite_recreates_output_file(tmp_path: Path, write_jsonl):
    questions_path = write_jsonl(
        tmp_path / "questions.jsonl",
        [{"question_id": "1", "turns": [{"content": "q1"}]}],
    )
    output_file = write_jsonl(
        tmp_path / "out" / "generation.jsonl",
        [{"question_id": "legacy", "choices": [{"turns": [{"content": "legacy"}]}]}],
    )

    model = FakeModel()
    runner = GenerationRunner(
        model=model,
        questions_path=str(questions_path),
        output_dir=str(output_file.parent),
        overwrite=True,
        output_file=str(output_file),
        parallel_api_calls=1,
    )
    runner.run()

    rows = load_jsonl(str(output_file))
    assert len(rows) == 1
    assert rows[0]["question_id"] == "1"
