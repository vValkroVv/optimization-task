"""Tests for judging several datasets through one shared judge pool.

Datasets whose judge configs share (model_name, api_base, api_type, api_key)
are judged by a single ThreadPoolExecutor. Each dataset keeps its own system
prompt via runner-scoped JUDGE_SETTINGS keys (category namespacing), so
same-named categories never collide.
"""
import json
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

import dredd
from src.judges.runner import JudgeRunner, run_judge_runners

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _install_fake_judgment_modules(
    monkeypatch: pytest.MonkeyPatch, behavior: Callable[[Dict], Dict]
) -> Tuple[List[str], Dict]:
    """Fake gen_judgment + utils.judge_utils, mirroring the real contract:
    judgment() writes one row to args['output_file'] and JUDGE_SETTINGS is a
    plain module-level dict."""
    calls: List[str] = []
    lock = threading.Lock()

    judge_utils_mod = types.ModuleType("utils.judge_utils")
    judge_utils_mod.JUDGE_SETTINGS = {
        "known": {"baseline": "old-baseline", "system_prompt": "upstream-prompt"}
    }

    def judgment(args_dict: Dict) -> None:
        question = args_dict["question"]
        row = behavior(question)
        output_file = Path(args_dict["output_file"])
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with lock:
            calls.append(str(question.get("category", "")))

    gen_mod = types.ModuleType("gen_judgment")
    gen_mod.judgment = judgment

    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = []  # mark as package

    monkeypatch.setitem(sys.modules, "gen_judgment", gen_mod)
    monkeypatch.setitem(sys.modules, "utils", utils_pkg)
    monkeypatch.setitem(sys.modules, "utils.judge_utils", judge_utils_mod)
    monkeypatch.setitem(sys.modules, "gigachat", MagicMock())

    return calls, judge_utils_mod.JUDGE_SETTINGS


def _build_fake_judge(system_prompt: str) -> SimpleNamespace:
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
        system_prompt=system_prompt,
        prompt_template="{QUESTION}",
    )


def _make_runner(
    tmp_path: Path,
    name: str,
    judge: SimpleNamespace,
    category: str,
    namespace: str,
    parallel_api_calls: int = 1,
) -> Tuple[JudgeRunner, Path]:
    ds_dir = tmp_path / name
    ds_dir.mkdir(parents=True, exist_ok=True)

    questions = ds_dir / "questions.jsonl"
    questions.write_text(
        json.dumps({"question_id": f"{name}-q1", "category": category, "turns": [{"content": "Q"}]}) + "\n",
        encoding="utf-8",
    )
    candidate = ds_dir / "candidate.jsonl"
    candidate.write_text(
        json.dumps({
            "question_id": f"{name}-q1",
            "model_id": "candidate",
            "choices": [{"turns": [{"content": f"candidate answer {name}"}]}],
        }) + "\n",
        encoding="utf-8",
    )
    baseline = ds_dir / "baseline.jsonl"
    baseline.write_text(
        json.dumps({
            "question_id": f"{name}-q1",
            "model_id": "baseline",
            "choices": [{"turns": [{"content": f"baseline answer {name}"}]}],
        }) + "\n",
        encoding="utf-8",
    )
    output_file = ds_dir / "judgments" / "judgment.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    runner = JudgeRunner(
        judge,
        str(questions),
        str(candidate),
        str(baseline),
        str(output_file.parent),
        candidate_model_name=f"cand-{name}",
        baseline_model_name=f"base-{name}",
        overwrite=True,
        output_file=str(output_file),
        parallel_api_calls=parallel_api_calls,
        category_namespace=namespace,
    )
    return runner, output_file


def _read_jsonl(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Core: same category name, different prompts, one pool
# ---------------------------------------------------------------------------

def test_shared_pool_same_category_different_prompts(tmp_path, monkeypatch):
    """Two datasets share a judge AND a category name but use different system
    prompts. Each battle must be judged with its own dataset's prompt, and the
    persisted records must carry the original (un-namespaced) category."""

    def behavior(question: Dict) -> Dict:
        # Resolve the prompt exactly the way gen_judgment does at judgment time.
        settings = sys.modules["utils.judge_utils"].JUDGE_SETTINGS
        prompt = settings[question["category"]]["system_prompt"]
        qid = str(question.get("uid") or question.get("question_id"))
        return {
            "uid": qid,
            "category": question["category"],
            "games": [
                {"score": "A>B", "judgment": {"answer": f"sys={prompt}"}},
                {"score": "B>A"},
            ],
        }

    calls, judge_settings = _install_fake_judgment_modules(monkeypatch, behavior)

    runner_a, out_a = _make_runner(
        tmp_path, "ds_a", _build_fake_judge("PROMPT-A"), "shared-cat", namespace="ds0"
    )
    runner_b, out_b = _make_runner(
        tmp_path, "ds_b", _build_fake_judge("PROMPT-B"), "shared-cat", namespace="ds1"
    )

    stats = run_judge_runners([runner_a, runner_b])

    assert stats == [
        {"succeeded": 1, "failed": 0, "skipped": False},
        {"succeeded": 1, "failed": 0, "skipped": False},
    ]

    # Each dataset was judged through its own registry key...
    assert set(calls) == {"ds0::shared-cat", "ds1::shared-cat"}
    assert judge_settings["ds0::shared-cat"]["system_prompt"] == "PROMPT-A"
    assert judge_settings["ds1::shared-cat"]["system_prompt"] == "PROMPT-B"
    # ...and the raw category was never registered or touched.
    assert "shared-cat" not in judge_settings

    # Results landed in the right files with prompts matching their dataset.
    rows_a = _read_jsonl(out_a)
    rows_b = _read_jsonl(out_b)
    assert len(rows_a) == 1 and len(rows_b) == 1
    assert rows_a[0]["uid"] == "ds_a-q1"
    assert rows_b[0]["uid"] == "ds_b-q1"
    assert rows_a[0]["games"][0]["judgment"]["answer"] == "sys=PROMPT-A"
    assert rows_b[0]["games"][0]["judgment"]["answer"] == "sys=PROMPT-B"

    # Persisted category is the original one, not the internal registry key.
    assert rows_a[0]["category"] == "shared-cat"
    assert rows_b[0]["category"] == "shared-cat"


def test_shared_pool_runs_datasets_concurrently(tmp_path, monkeypatch):
    """Battles from different datasets must be in flight simultaneously in one
    pool. A 2-party barrier can only be released if both datasets' battles run
    concurrently; sequential execution would break it by timeout."""
    barrier = threading.Barrier(2, timeout=15)

    def behavior(question: Dict) -> Dict:
        barrier.wait()
        qid = str(question.get("uid") or question.get("question_id"))
        return {"uid": qid, "category": question["category"], "games": [{"score": "A=B"}]}

    _install_fake_judgment_modules(monkeypatch, behavior)

    runner_a, _ = _make_runner(
        tmp_path, "ds_a", _build_fake_judge("PA"), "cat-a", namespace="ds0", parallel_api_calls=2
    )
    runner_b, _ = _make_runner(
        tmp_path, "ds_b", _build_fake_judge("PB"), "cat-b", namespace="ds1", parallel_api_calls=1
    )

    stats = run_judge_runners([runner_a, runner_b])

    assert [s["succeeded"] for s in stats] == [1, 1]
    assert [s["failed"] for s in stats] == [0, 0]


def test_strict_retry_uses_runner_scoped_key(tmp_path, monkeypatch):
    """When verdicts are missing, each runner retries through its own strict
    category, carrying its own system prompt; records keep the original
    category."""

    def behavior(question: Dict) -> Dict:
        qid = str(question.get("uid") or question.get("question_id"))
        category = question["category"]
        if "__strict_retry__" in category:
            settings = sys.modules["utils.judge_utils"].JUDGE_SETTINGS
            prompt = settings[category]["system_prompt"]
            return {
                "uid": qid,
                "category": category,
                "games": [{"score": "A>B", "judgment": {"answer": f"strict sys={prompt}"}}],
            }
        return {"uid": qid, "category": category, "games": [{"judgment": "no verdict label"}]}

    calls, judge_settings = _install_fake_judgment_modules(monkeypatch, behavior)

    runner_a, out_a = _make_runner(
        tmp_path, "ds_a", _build_fake_judge("PROMPT-A"), "shared-cat", namespace="ds0"
    )
    runner_b, out_b = _make_runner(
        tmp_path, "ds_b", _build_fake_judge("PROMPT-B"), "shared-cat", namespace="ds1"
    )

    stats = run_judge_runners([runner_a, runner_b])

    assert [s["succeeded"] for s in stats] == [1, 1]
    assert set(calls) == {
        "ds0::shared-cat",
        "ds0::__strict_retry__",
        "ds1::shared-cat",
        "ds1::__strict_retry__",
    }
    assert judge_settings["ds0::__strict_retry__"]["system_prompt"].startswith("PROMPT-A")
    assert judge_settings["ds1::__strict_retry__"]["system_prompt"].startswith("PROMPT-B")

    rows_a = _read_jsonl(out_a)
    rows_b = _read_jsonl(out_b)
    assert rows_a[0]["games"][0]["judgment"]["answer"] == "strict sys=PROMPT-A\n\n" + \
        "IMPORTANT: End your answer with exactly one final verdict label in the form " + \
        "[[A>>B]], [[A>B]], [[A=B]], [[B>A]], or [[B>>A]]."
    assert rows_b[0]["games"][0]["judgment"]["answer"].startswith("strict sys=PROMPT-B")
    # Strict-retried records also keep the original category.
    assert rows_a[0]["category"] == "shared-cat"
    assert rows_b[0]["category"] == "shared-cat"


def test_upstream_prompt_inherited_for_known_category(tmp_path, monkeypatch):
    """Categories shipped by arena-hard-auto keep their upstream judge
    protocol: the runner-scoped key inherits the upstream system prompt, not
    the judge's own one."""

    def behavior(question: Dict) -> Dict:
        settings = sys.modules["utils.judge_utils"].JUDGE_SETTINGS
        prompt = settings[question["category"]]["system_prompt"]
        qid = str(question.get("uid") or question.get("question_id"))
        return {
            "uid": qid,
            "category": question["category"],
            "games": [{"score": "A>B", "judgment": {"answer": f"sys={prompt}"}}],
        }

    _, judge_settings = _install_fake_judgment_modules(monkeypatch, behavior)

    runner, out_file = _make_runner(
        tmp_path, "ds_a", _build_fake_judge("JUDGE-OWN-PROMPT"), "known", namespace="ds0"
    )
    stats = runner.run()

    assert stats == {"succeeded": 1, "failed": 0, "skipped": False}
    assert judge_settings["ds0::known"]["system_prompt"] == "upstream-prompt"
    # Upstream entry itself is untouched.
    assert judge_settings["known"] == {
        "baseline": "old-baseline",
        "system_prompt": "upstream-prompt",
    }
    rows = _read_jsonl(out_file)
    assert rows[0]["games"][0]["judgment"]["answer"] == "sys=upstream-prompt"


def test_pool_never_raises_stats_always_returned(tmp_path, monkeypatch):
    """run_judge_runners must return stats for every runner even when a
    fail_on_error runner has failures — the caller persists summaries first
    and only then honors the flag (runner.run() raises for standalone use)."""

    def behavior(question: Dict) -> Dict:
        qid = str(question.get("uid") or question.get("question_id"))
        if qid.startswith("ds_bad"):
            raise RuntimeError("judge API died")
        return {"uid": qid, "category": question["category"], "games": [{"score": "A>B"}]}

    _install_fake_judgment_modules(monkeypatch, behavior)

    runner_ok, out_ok = _make_runner(
        tmp_path, "ds_ok", _build_fake_judge("PA"), "cat-a", namespace="ds0"
    )
    runner_bad, _ = _make_runner(
        tmp_path, "ds_bad", _build_fake_judge("PB"), "cat-b", namespace="ds1"
    )
    runner_bad.fail_on_error = True

    # The pooled call itself never raises...
    stats = run_judge_runners([runner_ok, runner_bad])
    assert stats[0] == {"succeeded": 1, "failed": 0, "skipped": False}
    assert stats[1]["failed"] == 1
    # ...and the clean sibling's judgments are persisted.
    assert len(_read_jsonl(out_ok)) == 1

    # Standalone runner.run() still raises per the old contract.
    with pytest.raises(RuntimeError, match="Judging completed with 1 failed task"):
        runner_bad.run()


def test_pooled_runners_require_unique_namespaces(tmp_path, monkeypatch):
    _install_fake_judgment_modules(monkeypatch, lambda q: {})
    runner_a, _ = _make_runner(tmp_path, "ds_a", _build_fake_judge("PA"), "c", namespace="ds0")
    runner_b, _ = _make_runner(tmp_path, "ds_b", _build_fake_judge("PB"), "c", namespace="ds0")
    with pytest.raises(ValueError, match="unique non-empty category_namespace"):
        run_judge_runners([runner_a, runner_b])


# ---------------------------------------------------------------------------
# Grouping at the orchestrator level
# ---------------------------------------------------------------------------

def test_judge_group_key_identity():
    same_a = OmegaConf.create({"model_name": "m", "api_base": "http://x", "api_key": "k", "temperature": 0.0})
    same_b = OmegaConf.create({"model_name": "m", "api_base": "http://x", "api_key": "k", "temperature": 0.7,
                               "system_prompt": "other", "max_tokens": 1})
    other_endpoint = OmegaConf.create({"model_name": "m", "api_base": "http://y", "api_key": "k"})
    other_model = OmegaConf.create({"model_name": "m2", "api_base": "http://x", "api_key": "k"})

    # Request-body parameters (temperature, prompts...) do not affect identity.
    assert dredd._judge_group_key(same_a) == dredd._judge_group_key(same_b)
    assert dredd._judge_group_key(same_a) != dredd._judge_group_key(other_endpoint)
    assert dredd._judge_group_key(same_a) != dredd._judge_group_key(other_model)


def test_run_groups_datasets_by_judge_identity(tmp_path, monkeypatch):
    """dredd.run(): datasets sharing a judge go through run_judge_runners as
    one group; a dataset with its own judge runs standalone via runner.run()."""
    q_file = tmp_path / "arena.jsonl"
    q_file.write_text(json.dumps({"question_id": "1", "prompt": "test"}) + "\n", encoding="utf-8")
    gen_file = tmp_path / "generation.jsonl"
    gen_file.write_text("", encoding="utf-8")

    def dataset(name: str, api_base: str) -> dict:
        return {
            "name": name,
            "model": {"name": f"m-{name}", "path": str(tmp_path / name)},
            "baseline": {"name": "b1"},
            "judge": {"model_name": "shared-judge", "api_base": api_base},
            "experiment": {"overwrite_judgments": False},
        }

    config = OmegaConf.create({
        "merged_config_path": str(tmp_path / "merged.yaml"),
        "datasets": [
            dataset("arena-one", "http://shared"),
            dataset("arena-two", "http://shared"),
            dataset("arena-solo", "http://own-endpoint"),
        ],
        "experiment": {"overwrite_generations": False, "overwrite_judgments": False},
        "judge": {"model_name": "shared-judge"},
        "model": {"name": "default_m"},
        "baseline": {"name": "default_b"},
    })

    created_runners: List[MagicMock] = []

    def make_runner_mock(*args, **kwargs):
        m = MagicMock()
        m.parallel_api_calls = kwargs.get("parallel_api_calls", 1)
        m.run.return_value = {"succeeded": 1, "failed": 0, "skipped": False}
        m.category_namespace = kwargs.get("category_namespace", "")
        created_runners.append(m)
        return m

    with patch("dredd.DATA_ROOT", tmp_path), \
         patch("dredd.get_questions_path", return_value=q_file), \
         patch("dredd._find_gen_file", return_value=gen_file), \
         patch("dredd.Judge"), \
         patch("dredd.APIModel"), \
         patch("dredd.JudgeRunner", side_effect=make_runner_mock) as MockRunner, \
         patch("dredd.run_judge_runners") as mock_grouped, \
         patch("dredd.compute_judgment_summary", return_value={"win_rate": 0}):

        mock_grouped.return_value = [
            {"succeeded": 1, "failed": 0, "skipped": False},
            {"succeeded": 1, "failed": 0, "skipped": False},
        ]

        dredd.run(config, phase="judgement")

    # Three runners were constructed, each with a unique namespace.
    namespaces = [call.kwargs["category_namespace"] for call in MockRunner.call_args_list]
    assert namespaces == ["ds0", "ds1", "ds2"]

    # The two shared-judge datasets went through one pooled call...
    mock_grouped.assert_called_once()
    grouped_runners = mock_grouped.call_args.args[0]
    assert len(grouped_runners) == 2
    assert [r.category_namespace for r in grouped_runners] == ["ds0", "ds1"]
    # ...and only the standalone dataset used runner.run().
    assert [m.run.call_count for m in created_runners] == [0, 0, 1]
