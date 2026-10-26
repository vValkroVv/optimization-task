import json
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import pytest

from src.judges.runner import JudgeRunner

pytestmark = [pytest.mark.integration]


def _build_fake_judge() -> SimpleNamespace:
    model = SimpleNamespace(
        name="judge-model",
        config={"api_type": "openai"},
        api_base="http://localhost/v1",
        api_key="secret",
        request_timeout=300,
    )
    config = {
        "regex_pattern": r"\[\[([AB<>=]+)\]\]",
        "missing_verdict_retries": 1,
        "fail_on_missing_verdict": True,
        "temperature": 0.0,
        "max_tokens": 256,
        "num_retries": 1,
    }
    return SimpleNamespace(
        model=model,
        config=config,
        system_prompt="judge-system",
        prompt_template="{question}",
    )


def _install_fake_judgment_modules(
    monkeypatch: pytest.MonkeyPatch, behavior
) -> Tuple[List[str], Dict]:
    calls: List[str] = []

    def judgment(args_dict: Dict) -> None:
        question = args_dict["question"]
        row = behavior(question)
        output_file = Path(args_dict["output_file"])
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        calls.append(str(question.get("category", "")))

    gen_mod = types.ModuleType("gen_judgment")
    gen_mod.judgment = judgment

    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = []  # mark as package
    judge_utils_mod = types.ModuleType("utils.judge_utils")
    judge_utils_mod.JUDGE_SETTINGS = {"known": {"baseline": "old-baseline", "system_prompt": "old-prompt"}}

    monkeypatch.setitem(sys.modules, "gen_judgment", gen_mod)
    monkeypatch.setitem(sys.modules, "utils", utils_pkg)
    monkeypatch.setitem(sys.modules, "utils.judge_utils", judge_utils_mod)

    return calls, judge_utils_mod.JUDGE_SETTINGS


@pytest.mark.skip(reason="JudgeRunner.__init__ imports gigachat which fails in this env (pydantic/gigachat); re-enable when fixed")
def test_judge_runner_run_adds_fallback_categories_and_annotations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write_jsonl, read_jsonl
):
    def behavior(question: Dict) -> Dict:
        qid = str(question.get("uid") or question.get("question_id"))
        return {
            "uid": qid,
            "games": [{"score": "B>A"}, {"score": "A>B"}],
        }

    calls, _ = _install_fake_judgment_modules(monkeypatch, behavior)

    questions = write_jsonl(
        tmp_path / "questions.jsonl",
        [{"question_id": "q1", "category": "custom-category", "turns": [{"content": "question"}]}],
    )
    candidate = write_jsonl(
        tmp_path / "candidate.jsonl",
        [
            {
                "question_id": "q1",
                "model_id": "candidate",
                "choices": [{"turns": [{"content": "candidate answer"}]}],
            }
        ],
    )
    baseline = write_jsonl(
        tmp_path / "baseline.jsonl",
        [
            {
                "question_id": "q1",
                "model_id": "baseline",
                "choices": [{"turns": [{"content": "baseline answer"}]}],
            }
        ],
    )
    output_file = tmp_path / "judgments" / "judgment.jsonl"

    runner = JudgeRunner(
        judge=_build_fake_judge(),
        questions_path=str(questions),
        candidate_path=str(candidate),
        baseline_path=str(baseline),
        output_dir=str(output_file.parent),
        candidate_model_name="candidate",
        baseline_model_name="baseline",
        overwrite=True,
        output_file=str(output_file),
        parallel_api_calls=1,
    )
    stats = runner.run()

    assert stats == {"succeeded": 1, "failed": 0, "skipped": False}
    assert calls == ["custom-category"]
    assert "custom-category" in runner.judge_settings

    rows = read_jsonl(output_file)
    games = rows[0]["games"]
    assert games[0]["role_a"] == "baseline"
    assert games[0]["role_b"] == "candidate"
    assert games[1]["role_a"] == "candidate"
    assert games[1]["role_b"] == "baseline"


@pytest.mark.skip(reason="JudgeRunner.__init__ imports gigachat which fails in this env (pydantic/gigachat); re-enable when fixed")
def test_judge_runner_retries_when_verdict_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def behavior(question: Dict) -> Dict:
        qid = str(question.get("uid") or question.get("question_id"))
        if question.get("category") == "__strict_retry__":
            return {
                "uid": qid,
                "games": [{"score": "B>A"}, {"score": "A>B"}],
            }
        return {
            "uid": qid,
            "games": [{"judgment": "no verdict label"}, {"judgment": {"answer": "still no label"}}],
        }

    calls, _ = _install_fake_judgment_modules(monkeypatch, behavior)

    placeholder = tmp_path / "placeholder.jsonl"
    placeholder.write_text("", encoding="utf-8")
    output_file = tmp_path / "judgments" / "judgment.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    runner = JudgeRunner(
        judge=_build_fake_judge(),
        questions_path=str(placeholder),
        candidate_path=str(placeholder),
        baseline_path=str(placeholder),
        output_dir=str(output_file.parent),
        candidate_model_name="candidate",
        baseline_model_name="baseline",
        overwrite=True,
        output_file=str(output_file),
        parallel_api_calls=1,
    )

    pattern = re.compile(r"\[\[([AB<>=]+)\]\]")
    q_norm = {"uid": "q1", "category": "known", "prompt": "question"}
    cand_norm = {"model": "candidate", "messages": [{"role": "assistant", "content": {"answer": "cand"}}]}
    base_norm = {"model": "baseline", "messages": [{"role": "assistant", "content": {"answer": "base"}}]}

    result = runner._judge_with_retries(
        q_norm=q_norm,
        cand_norm=cand_norm,
        base_norm=base_norm,
        configs={},
        endpoint_dict={},
        pattern_compiled=pattern,
    )

    assert calls == ["known", "__strict_retry__"]
    assert runner._missing_verdict_indices(result, pattern) == []


def test_judge_runner_pam_mode_parallel_strict_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def behavior(question: Dict) -> Dict:
        qid = str(question.get("uid") or question.get("question_id"))
        if question.get("category") == "__strict_retry__":
            return {
                "uid": qid,
                "games": [{"score": "B>A"}, {"score": "A>B"}],
            }
        return {
            "uid": qid,
            "games": [{"judgment": "no verdict label"}, {"judgment": {"answer": "still no label"}}],
        }

    calls, _ = _install_fake_judgment_modules(monkeypatch, behavior)

    placeholder = tmp_path / "placeholder.jsonl"
    placeholder.write_text("", encoding="utf-8")
    output_file = tmp_path / "judgments" / "judgment.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    judge = _build_fake_judge()
    judge.config["pam_mode"] = True
    runner = JudgeRunner(
        judge=judge,
        questions_path=str(placeholder),
        candidate_path=str(placeholder),
        baseline_path=str(placeholder),
        output_dir=str(output_file.parent),
        candidate_model_name="candidate",
        baseline_model_name="baseline",
        overwrite=True,
        output_file=str(output_file),
        parallel_api_calls=1,
    )

    pattern = re.compile(r"\[\[([AB<>=]+)\]\]")
    q_norm = {"uid": "q1", "category": "known", "prompt": "question"}
    cand_norm = {"model": "candidate", "messages": [{"role": "assistant", "content": {"answer": "cand"}}]}
    base_norm = {"model": "baseline", "messages": [{"role": "assistant", "content": {"answer": "base"}}]}

    result = runner._judge_with_retries(
        q_norm=q_norm,
        cand_norm=cand_norm,
        base_norm=base_norm,
        configs={},
        endpoint_dict={},
        pattern_compiled=pattern,
    )

    assert runner._missing_verdict_indices(result, pattern) == []
    # First call is the normal attempt; the rest are the parallel strict fan-out
    # (up to 5, background stragglers may still be appending after return).
    assert calls[0] == "known"
    assert set(calls[1:]) == {"__strict_retry__"}
    assert 2 <= len(calls) <= 6
