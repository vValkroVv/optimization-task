import json
from pathlib import Path
from typing import Dict, List

import pytest


@pytest.fixture
def write_jsonl():
    def _write(path: Path, rows: List[Dict]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    return _write


@pytest.fixture
def read_jsonl():
    def _read(path: Path) -> List[Dict]:
        rows: List[Dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    return _read


@pytest.fixture
def structured_runner(tmp_path, write_jsonl):
    from src.judges.judge import Judge
    from src.judges.runner import JudgeRunner
    from src.models import APIModel

    config = {
        "input_format": "structured_sbs_json", "api_base": "http://judge-router/v1",
        "request_timeout": 42, "num_retries": 3, "missing_verdict_retries": 0,
        "system_prompt": "must not be sent", "prompt_template": "{question}",
    }
    model = APIModel("gemma4-meanvar", config)
    question = {"uid": "q1", "category": "test", "prompt": '"quoted"\nВопрос 🐈', "domain": "science"}
    paths = [
        write_jsonl(tmp_path / "questions.jsonl", [question]),
        write_jsonl(tmp_path / "candidate.jsonl", [
            {"uid": "q1", "model": "candidate", "messages": [{"content": "C"}]},
        ]),
        write_jsonl(tmp_path / "baseline.jsonl", [
            {"uid": "q1", "model": "baseline", "messages": [{"content": "B"}]},
        ]),
    ]
    return JudgeRunner(
        Judge(model, config, tmp_path), *map(str, paths), str(tmp_path / "out"),
        "candidate", "baseline", fail_on_error=True,
    )


@pytest.fixture
def structured_result():
    forward = {
        "verdict": "B>A", "probabilities": {"A>B": 0.1, "B>A": 0.8, "A=B": 0.1},
        "rewards": {"A": [1.0, 0.4], "B": [2.0, 0.4]}, "tie_margin": 0.5,
    }
    return {
        **forward, "protocol_sha256": "p", "export_manifest_sha256": "e",
        "reverse": {
            **forward, "verdict": "A>B",
            "probabilities": {"A>B": 0.8, "B>A": 0.1, "A=B": 0.1},
            "rewards": {"A": [2.0, 0.4], "B": [1.0, 0.4]},
        },
    }


@pytest.fixture
def structured_http(mocker, structured_result):
    response = mocker.Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": json.dumps(structured_result)}}],
    }
    mocker.patch("src.models.time.sleep")
    return mocker.patch("src.models.requests.post", return_value=response)
