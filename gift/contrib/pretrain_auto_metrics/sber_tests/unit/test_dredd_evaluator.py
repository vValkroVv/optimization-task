import json
from pathlib import Path

import pytest
from node_runner.src.job_dispatcher import JD
from node_runner.src.config_model import (
    DreddEvaluatorConfig,
    ModelInfoConfig as ModelInfo,
)
from node_runner.src.metrics.evaluators.dredd import (
    DreddEvaluator,
    get_dredd_submodule_path,
)
from sber_tests.test_support import judge_endpoints

pytestmark = pytest.mark.unit


def _dredd_config(payload: dict) -> DreddEvaluatorConfig:
    return DreddEvaluatorConfig.model_validate(payload)


def _build_evaluator(tmp_path: Path) -> DreddEvaluator:
    evaluator = DreddEvaluator()
    evaluator._dredd_phase = "generation"
    evaluator.hf_checkpoint_path = (
        tmp_path / "model" / "ep1_ba2" / "artifacts" / "hf_checkpoint"
    )
    evaluator.metrics_dir = tmp_path / "artifacts" / "metrics" / "dredd"
    evaluator.metrics_dir.mkdir(parents=True, exist_ok=True)
    evaluator.model_info = ModelInfo(
        model_class="sft",
        size="lite",
        username="test_user",
    )
    evaluator.autometric_runs_config = _dredd_config(
        {"judge": {"model_name": "test-judge"}}
    )
    return evaluator


def _write_complete_summary(
    evaluator,
    tmp_path,
    monkeypatch,
    arena="arena-hard-logs-v3",
    baseline="gpt-4o",
    questions=5,
):
    data_root = tmp_path / "data"
    questions_path = data_root / "arenas" / f"{arena}.jsonl"
    questions_path.parent.mkdir(parents=True, exist_ok=True)
    questions_path.write_text(
        "".join(
            json.dumps({"uid": i, "question": f"Question {i}"}) + "\n"
            for i in range(questions)
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DREDD_DATA_ROOT", str(data_root))
    summary_path = (
        evaluator.metrics_dir
        / "arenas"
        / arena
        / "judgments"
        / baseline
        / "judgment_summary.jsonl"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "total": questions,
        "candidate_wins": 1,
        "baseline_wins": 0,
        "ties": 4,
        "failed_judgments": 0,
        "win_rate": 20.0,
        "loss_rate": 0.0,
        "tie_rate": 80.0,
        "score_0_100": 60.0,
    }
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    return summary_path


@pytest.mark.usefixtures("scalar_ci_upload")
def test_dredd_send_metrics_to_jd_builds_payload(monkeypatch, tmp_path: Path):
    evaluator = _build_evaluator(tmp_path)
    evaluator.autometric_runs_config = _dredd_config(
        {"judge": {"model_name": "minimax-m2.7"}}
    )
    saved_payload: dict[str, object] = {}
    sent_payload: dict[str, object] = {}

    def _fake_save_metrics_json(body: dict, step: int, metrics_dir: Path) -> None:
        saved_payload["body"] = body
        saved_payload["step"] = step
        saved_payload["metrics_dir"] = metrics_dir

    def _fake_make_request_to_jd(body: dict, _artifacts) -> None:
        sent_payload["body"] = body

    monkeypatch.setattr(JD, "save_metrics_json", _fake_save_metrics_json)
    monkeypatch.setattr(JD, "send_metrics", _fake_make_request_to_jd)
    monkeypatch.setattr(JD, "get_shooter_version", lambda path: "deadbeef")

    evaluator._send_metrics_to_jd(
        {
            "dredd/arena-general-v0.1/yandex_gpt_pro/win_rate": 0.75,
            "dredd/arena-general-v0.1/yandex_gpt_pro/score_0_1": 0.77,
            "dredd/arena-hard-ru/gpt-4-0613/score_0_1": 0.52,
        }
    )

    assert "body" in saved_payload
    assert "body" in sent_payload
    assert saved_payload["body"] == sent_payload["body"]

    body = sent_payload["body"]
    assert body["nfs_path"] == str(tmp_path / "model")
    assert body["name"] == "model"
    assert body["shooter_name"] == "dredd"
    assert body["shooter_version"] == "deadbeef"
    assert body["model_class"] == "sft"
    assert body["size"] == "lite"
    assert body["username"] == "test_user"
    assert body["metrics"] == {
        2: {
            "ARENA_GENERAL_V0_1_VS_YANDEX_GPT_PRO_JUDGE_MINIMAX_M2_7": 0.77,
            "ARENA_HARD_RU_VS_GPT_4_0613_JUDGE_MINIMAX_M2_7": 0.52,
        }
    }
    assert saved_payload["step"] == 2
    assert saved_payload["metrics_dir"] == evaluator.metrics_dir


def test_dredd_convert_metrics_to_jd_keys_keeps_only_score_0_1_and_benchmark(
    tmp_path: Path,
):
    evaluator = _build_evaluator(tmp_path)
    evaluator.autometric_runs_config = _dredd_config(
        {"judge": {"model_name": "minimax-m2.7"}}
    )
    processed = {
        "dredd/arena-hard-ru/gpt-4-0613/total": 10.0,
        "dredd/arena-hard-ru/gpt-4-0613/score_0_1": 0.55,
        "dredd/arena-hard-ru/another-baseline/score_0_1": 0.57,
        "dredd/ru-llm-arena/gpt-3.5-turbo-0125/score_0_1": 0.48,
    }
    converted = evaluator._convert_metrics_to_jd_keys(processed)
    assert converted == {
        "ARENA_HARD_RU_VS_GPT_4_0613_JUDGE_MINIMAX_M2_7": 0.55,
        "ARENA_HARD_RU_VS_ANOTHER_BASELINE_JUDGE_MINIMAX_M2_7": 0.57,
        "RU_LLM_ARENA_VS_GPT_3_5_TURBO_0125_JUDGE_MINIMAX_M2_7": 0.48,
    }


def test_dredd_normalize_metric_name_for_jd():
    metric_name = "dredd/arena-general-v0.1/yandex_gpt_pro/win_rate"
    assert (
        DreddEvaluator._normalize_metric_name_for_jd(metric_name)
        == "DREDD_ARENA_GENERAL_V0_1_YANDEX_GPT_PRO_WIN_RATE"
    )


def test_send_metrics_to_jd_skips_without_checkpoint(monkeypatch, tmp_path: Path):
    evaluator = DreddEvaluator()
    evaluator.hf_checkpoint_path = None
    evaluator.metrics_dir = tmp_path / "artifacts" / "metrics" / "dredd"
    evaluator.metrics_dir.mkdir(parents=True, exist_ok=True)
    evaluator.model_info = ModelInfo(
        model_class="sft",
        size="lite",
        username="test_user",
    )

    called = {"save": 0, "send": 0}
    monkeypatch.setattr(
        JD,
        "save_metrics_json",
        lambda *args, **kwargs: called.__setitem__("save", 1),
    )
    monkeypatch.setattr(
        JD,
        "send_metrics",
        lambda *args, **kwargs: called.__setitem__("send", 1),
    )

    evaluator._send_metrics_to_jd({"dredd/arena/general/win_rate": 0.75})

    assert called["save"] == 0
    assert called["send"] == 0


def test_dredd_convert_raw_config_to_bash_includes_phase(tmp_path: Path):
    evaluator = _build_evaluator(tmp_path)
    cmd = evaluator._convert_raw_config_to_bash(
        dredd_overrides=_dredd_config(
            {
                "judge": {"model_name": "j"},
                "experiment": {"max_questions": 5},
                "datasets": [],
            }
        ),
        phase="generation",
    )["dredd"]

    assert cmd[0] == DreddEvaluator.DREDD_PYTHON
    assert "--phase" in cmd
    phase_index = cmd.index("--phase")
    assert cmd[phase_index + 1] == "generation"


def test_dredd_convert_raw_config_to_bash_skips_phase_when_not_set(tmp_path: Path):
    evaluator = _build_evaluator(tmp_path)
    cmd = evaluator._convert_raw_config_to_bash(
        dredd_overrides=_dredd_config(
            {
                "judge": {"model_name": "j"},
                "experiment": {"max_questions": 5},
                "datasets": [],
            }
        ),
        phase=None,
    )["dredd"]

    assert "--phase" not in cmd


def test_dredd_convert_raw_config_to_bash_forces_full_dataset_and_model_name(
    tmp_path: Path,
):
    evaluator = _build_evaluator(tmp_path)
    cmd = evaluator._convert_raw_config_to_bash(
        dredd_overrides=_dredd_config(
            {
                "judge": {"model_name": "j"},
                "experiment": {"max_questions": 5},
                "datasets": [
                    {"name": "arena-hard-ru", "experiment": {"max_questions": 3}}
                ],
                "model": {"name": "deepseek-chat-v3-0324"},
            }
        ),
        phase="generation",
    )["dredd"]

    overrides_json = cmd[cmd.index("--overrides-json") + 1]
    payload = json.loads(overrides_json)

    assert payload["experiment"]["max_questions"] is None
    assert payload["datasets"][0]["experiment"]["max_questions"] is None
    assert payload["model"]["path"] == str(evaluator.metrics_dir)
    assert payload["model"]["name"] == "model"


def _apply_defaults(config) -> None:
    config.judge_endpoints = judge_endpoints(
        {config.judge.model_name: "http://127.0.0.1:31000"},
        {config.judge.model_name: 4},
    )
    config.apply_defaults(
        inference_instances=1,
        base_url="http://localhost:29999",
    )


def test_dredd_apply_defaults_urls_reach_overrides_payload(tmp_path: Path):
    evaluator = _build_evaluator(tmp_path)
    config = _dredd_config(
        {"judge": {"model_name": "openrouter:gpt-5"}, "datasets": []}
    )
    _apply_defaults(config)
    cmd = evaluator._convert_raw_config_to_bash(config, phase="judgement")["dredd"]

    payload = json.loads(cmd[cmd.index("--overrides-json") + 1])
    assert payload["judge"] == {
        "model_name": "openrouter:gpt-5",
        "parallel_api_calls": 512,
        "request_timeout": 21600,
        "max_tokens": 16384,
        "api_base": "http://127.0.0.1:31000/v1",
        "pam_mode": True,
    }
    assert payload["model"]["api_base"] == "http://localhost:29999/v1"
    assert payload["model"]["parallel_api_calls"] == 128


def test_dredd_apply_defaults_keeps_explicit_judge_values():
    config = _dredd_config(
        {
            "judge": {
                "model_name": "test-judge",
                "parallel_api_calls": 32,
                "api_base": "http://judge.example/v1",
            }
        }
    )

    _apply_defaults(config)

    assert config.judge.parallel_api_calls == 32
    assert config.judge.api_base == "http://judge.example/v1"


def test_dredd_convert_raw_config_keeps_explicit_judge_api_base(tmp_path: Path):
    evaluator = _build_evaluator(tmp_path)
    cmd = evaluator._convert_raw_config_to_bash(
        _dredd_config(
            {
                "judge": {"model_name": "j", "api_base": "http://judge.example/v1"},
                "datasets": [],
            }
        ),
        phase="judgement",
    )["dredd"]

    payload = json.loads(cmd[cmd.index("--overrides-json") + 1])
    assert payload["judge"]["api_base"] == "http://judge.example/v1"


def test_dredd_get_metrics_converts_to_fractions(tmp_path: Path, monkeypatch):
    evaluator = _build_evaluator(tmp_path)
    _write_complete_summary(evaluator, tmp_path, monkeypatch)

    assert evaluator.get_metrics() == {}
    evaluator._dredd_phase = "judgement"
    metrics = evaluator.get_metrics()
    prefix = "dredd/arena-hard-logs-v3/gpt-4o"
    assert metrics[f"{prefix}/total"] == 5.0
    assert metrics[f"{prefix}/candidate_wins"] == 1.0
    assert metrics[f"{prefix}/ties"] == 4.0
    assert metrics[f"{prefix}/failed_judgments"] == 0.0
    assert metrics[f"{prefix}/win_rate"] == 0.2
    assert metrics[f"{prefix}/tie_rate"] == 0.8
    assert metrics[f"{prefix}/score_0_1"] == 0.6


@pytest.mark.parametrize(
    "summary_override,chat_allowed",
    [
        ({"failed_judgments": 1}, True),
        ({"total": 4}, True),
        ({"total": "not-a-number"}, False),
        ({"total": None}, False),
        ({"total": 0}, False),
        ({"total": 6}, False),
    ],
    ids=[
        "failed-judgment",
        "incomplete-coverage",
        "invalid-total",
        "null-total",
        "empty",
        "excess",
    ],
)
@pytest.mark.parametrize("input_format", ["text", "structured_sbs_json"])
def test_dredd_get_metrics_rejects_failed_or_incomplete_summary(
    tmp_path, monkeypatch, summary_override, chat_allowed, input_format
):
    evaluator = _build_evaluator(tmp_path)
    evaluator._dredd_phase = "judgement"
    evaluator.autometric_runs_config.judge.input_format = input_format
    summary_path = _write_complete_summary(evaluator, tmp_path, monkeypatch)
    assert evaluator.get_metrics()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(summary_override)
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    metrics = evaluator.get_metrics()
    if chat_allowed and input_format == "text":
        assert metrics["dredd/arena-hard-logs-v3/gpt-4o/score_0_1"] == 0.6
    else:
        assert metrics == {}


@pytest.mark.parametrize(
    "totals,failures,published",
    [
        ((60, 60), (40, 40), True),
        ((60, 59), (40, 41), False),
        ((60, 60), (0, 0), True),
        ((60, 59), (0, 0), False),
        ((100, 100), (40, 41), False),
    ],
    ids=[
        "80-no-double-count",
        "81",
        "80-missing-generations",
        "81-missing-generations",
        "81-reported",
    ],
)
def test_dredd_failure_budget_is_shared_across_arenas(
    tmp_path, monkeypatch, totals, failures, published
):
    evaluator = _build_evaluator(tmp_path)
    evaluator._dredd_phase = "judgement"
    for arena, total, failed in zip(("arena-a", "arena-b"), totals, failures):
        path = _write_complete_summary(
            evaluator, tmp_path, monkeypatch, arena=arena, questions=100
        )
        summary = json.loads(path.read_text())
        summary.update(total=total, failed_judgments=failed)
        path.write_text(json.dumps(summary))

    scores = {
        key: value
        for key, value in evaluator.get_metrics().items()
        if key.endswith("/score_0_1")
    }
    assert scores == (
        {f"dredd/{arena}/gpt-4o/score_0_1": 0.6 for arena in ("arena-a", "arena-b")}
        if published
        else {}
    )


@pytest.mark.parametrize(
    "global_format,arena_format,published",
    [("text", "structured_sbs_json", False), ("structured_sbs_json", "text", True)],
)
def test_dredd_failure_budget_uses_arena_judge_format(
    tmp_path, monkeypatch, global_format, arena_format, published
):
    evaluator = _build_evaluator(tmp_path)
    evaluator._dredd_phase = "judgement"
    evaluator.autometric_runs_config = _dredd_config(
        {
            "judge": {"model_name": "test-judge", "input_format": global_format},
            "datasets": [{"name": "arena-a", "judge": {"input_format": arena_format}}],
        }
    )
    path = _write_complete_summary(evaluator, tmp_path, monkeypatch, arena="arena-a")
    summary = json.loads(path.read_text())
    summary.update(total=4, failed_judgments=1)
    path.write_text(json.dumps(summary))

    assert bool(evaluator.get_metrics()) is published


def test_judge_model_name_validation():
    with pytest.raises(ValueError, match="judge"):
        _dredd_config({})

    with pytest.raises(ValueError, match="model_name"):
        _dredd_config({"judge": {}})


def test_run_fails_without_dredd_root(monkeypatch, tmp_path: Path):
    evaluator = _build_evaluator(tmp_path)
    evaluator.autometric_runs_config = _dredd_config({"judge": {"model_name": "j"}})

    monkeypatch.setattr(
        "node_runner.src.metrics.evaluators.dredd.get_dredd_submodule_path",
        lambda: tmp_path / "no_such_dredd",
    )
    with pytest.raises(RuntimeError, match="Dredd root not found"):
        evaluator.run_phase("generation")


@pytest.mark.parametrize("phase", ["generation", "judgement"])
def test_failed_runs_fail_generation_but_not_scoring(
    monkeypatch, tmp_path: Path, phase
):
    evaluator = _build_evaluator(tmp_path)
    evaluator._dredd_phase = "judgement"
    evaluator.autometric_runs_config = _dredd_config({"judge": {"model_name": "j"}})
    dredd_root = tmp_path / "contrib" / "dredd"
    dredd_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "node_runner.src.metrics.evaluators.dredd.get_dredd_submodule_path",
        lambda: dredd_root,
    )

    created_run_names = []

    class _FailingSM:
        def __init__(self, *args, **kwargs):
            self.exit_code = 42
            created_run_names.append(args[3])

        def start(self):
            pass

        def wait_for_completion(self):
            pass

        def log_error(self):
            pass

        def is_successful(self):
            return False

    monkeypatch.setattr(
        "node_runner.src.metrics.evaluators.dredd.SubprocessManager", _FailingSM
    )
    _write_complete_summary(evaluator, tmp_path, monkeypatch)
    assert evaluator.get_metrics()
    if phase == "generation":
        with pytest.raises(RuntimeError, match="every generation run failed"):
            evaluator.run_phase(phase)
    else:
        evaluator.run_phase(phase)
    assert created_run_names == [f"Dredd dredd {phase}"]
    assert evaluator.get_metrics() == {}


def test_get_metrics_handles_empty_inputs_and_baseline_filter(
    tmp_path: Path, monkeypatch
):
    evaluator = _build_evaluator(tmp_path)
    evaluator._dredd_phase = "judgement"
    evaluator.autometric_runs_config = _dredd_config(
        {
            "judge": {"model_name": "j"},
            "datasets": [{"name": "arena-hard", "baseline": {"name": "gpt-4o"}}],
        }
    )

    _write_complete_summary(
        evaluator, tmp_path, monkeypatch, arena="arena-hard", baseline="stale-baseline"
    )
    _write_complete_summary(evaluator, tmp_path, monkeypatch, arena="arena-hard")

    # A non-dir child in arenas should be ignored
    (evaluator.metrics_dir / "arenas" / "README.txt").write_text("x", encoding="utf-8")

    metrics = evaluator.get_metrics()
    assert "dredd/arena-hard/stale-baseline/score_0_1" not in metrics
    assert metrics["dredd/arena-hard/gpt-4o/win_rate"] == 0.2
    assert metrics["dredd/arena-hard/gpt-4o/score_0_1"] == 0.6


def test_send_metrics_to_jd_skips_when_metrics_dir_missing(tmp_path: Path):
    evaluator = _build_evaluator(tmp_path)
    evaluator.metrics_dir = None
    evaluator._send_metrics_to_jd({"dredd/a/b/score_0_1": 0.7})


def test_convert_metrics_to_jd_keys_filters_non_score_and_non_dredd(tmp_path: Path):
    evaluator = _build_evaluator(tmp_path)
    evaluator.autometric_runs_config = _dredd_config(
        {"judge": {"model_name": "minimax-m2.7"}}
    )
    converted = evaluator._convert_metrics_to_jd_keys(
        {
            "not_dredd_metric": 1.0,
            "dredd/arena/general/total": 2.0,
            "dredd/arena/general/win_rate": 0.6,
            "dredd/arena/general/score_0_1": 0.5,
            "dredd/arena2/baseline2/score_0_1": 0.7,
        }
    )
    assert converted == {
        "ARENA_VS_GENERAL_JUDGE_MINIMAX_M2_7": 0.5,
        "ARENA2_VS_BASELINE2_JUDGE_MINIMAX_M2_7": 0.7,
    }


def test_load_summary_oserror_returns_none(monkeypatch, tmp_path: Path):
    summary_path = tmp_path / "summary.jsonl"
    summary_path.write_text('{"ok": 1}\n', encoding="utf-8")

    def _raise_oserror(*args, **kwargs):
        raise OSError("cannot read")

    monkeypatch.setattr("builtins.open", _raise_oserror)
    from node_runner.src.metrics.evaluators.dredd import _load_summary_from_jsonl

    assert _load_summary_from_jsonl(summary_path) is None


def test_convert_raw_config_keeps_model_name_when_checkpoint_name_empty(
    tmp_path: Path, monkeypatch
):
    evaluator = _build_evaluator(tmp_path)
    monkeypatch.setattr(
        JD,
        "parse_checkpoint_path",
        lambda _path: ("/tmp/model", "", 1, 2),
    )
    cmd = evaluator._convert_raw_config_to_bash(
        _dredd_config(
            {
                "judge": {"model_name": "j"},
                "experiment": {"max_questions": 1},
                "model": {"name": "keep-me"},
            }
        ),
        phase="generation",
    )["dredd"]
    payload = json.loads(cmd[cmd.index("--overrides-json") + 1])
    assert payload["experiment"]["max_questions"] is None
    assert payload["model"]["name"] == "keep-me"


def test_convert_raw_config_dataset_without_max_questions(tmp_path: Path):
    evaluator = _build_evaluator(tmp_path)
    cmd = evaluator._convert_raw_config_to_bash(
        _dredd_config(
            {
                "judge": {"model_name": "j"},
                "experiment": {"max_questions": 10},
                "datasets": [{"name": "arena-hard", "experiment": {}}],
                "model": {},
            }
        ),
        phase="generation",
    )["dredd"]
    payload = json.loads(cmd[cmd.index("--overrides-json") + 1])
    assert payload["datasets"][0]["experiment"]["max_questions"] is None


def test_get_configured_baselines_uses_common_baseline_fallback(tmp_path: Path):
    evaluator = _build_evaluator(tmp_path)
    evaluator.autometric_runs_config = _dredd_config(
        {
            "judge": {"model_name": "j"},
            "baseline": {"name": "common-baseline"},
            "datasets": [
                {"name": "arena-a"},
                {"name": "arena-b", "baseline": {"name": "gpt-4o"}},
                {"name": "arena-c", "baseline": {}},
            ],
        }
    )
    assert evaluator._get_configured_baselines() == {
        "arena-a": "common-baseline",
        "arena-b": "gpt-4o",
    }


def test_get_metrics_early_returns_and_continue_paths(tmp_path: Path):
    evaluator = _build_evaluator(tmp_path)
    evaluator._dredd_phase = "judgement"

    evaluator.metrics_dir = None
    assert evaluator.get_metrics() == {}

    evaluator.metrics_dir = tmp_path / "dredd_metrics2"
    evaluator.metrics_dir.mkdir(parents=True, exist_ok=True)
    assert evaluator.get_metrics() == {}

    arenas = evaluator.metrics_dir / "arenas"
    arenas.mkdir(parents=True, exist_ok=True)
    (arenas / "not_dir.txt").write_text("x", encoding="utf-8")

    arena_without_judgments = arenas / "arena-no-judgments"
    arena_without_judgments.mkdir(parents=True, exist_ok=True)

    arena = arenas / "arena-has-judgments"
    judgments = arena / "judgments"
    judgments.mkdir(parents=True, exist_ok=True)
    (judgments / "not_dir.txt").write_text("x", encoding="utf-8")
    assert evaluator.get_metrics() == {}

    no_summary = judgments / "baseline-without-summary"
    no_summary.mkdir(parents=True, exist_ok=True)

    assert evaluator.get_metrics() == {}


def test_convert_metrics_to_jd_keys_empty_normalized_name(monkeypatch, tmp_path: Path):
    evaluator = _build_evaluator(tmp_path)
    evaluator.autometric_runs_config = _dredd_config(
        {"judge": {"model_name": "minimax-m2.7"}}
    )
    monkeypatch.setattr(
        DreddEvaluator,
        "_normalize_metric_name_for_jd",
        staticmethod(lambda _name: ""),
    )
    converted = evaluator._convert_metrics_to_jd_keys({"dredd/a/b/score_0_1": 0.7})
    assert converted == {}


def test_setup_creates_metrics_dir(tmp_path: Path):
    evaluator = DreddEvaluator()
    config = _dredd_config({"judge": {"model_name": "j"}})
    evaluator.setup(
        autometric_runs_config=config,
        artifacts_path=tmp_path / "artifacts",
        hf_checkpoint_path=tmp_path
        / "model"
        / "ep1_ba2"
        / "artifacts"
        / "hf_checkpoint",
        model_info=ModelInfo(model_class="sft", size="lite", username="u"),
    )
    assert evaluator.metrics_dir.exists()
    assert evaluator.autometric_runs_config is config


def test_run_completes_when_subprocess_is_successful(monkeypatch, tmp_path: Path):
    evaluator = _build_evaluator(tmp_path)
    evaluator.autometric_runs_config = _dredd_config({"judge": {"model_name": "j"}})
    dredd_root = tmp_path / "contrib" / "dredd"
    dredd_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "node_runner.src.metrics.evaluators.dredd.get_dredd_submodule_path",
        lambda: dredd_root,
    )

    class _OkSM:
        def __init__(self, *args, **kwargs):
            self.exit_code = 0

        def start(self):
            pass

        def wait_for_completion(self):
            pass

        def log_error(self):
            pass

        def is_successful(self):
            return True

    monkeypatch.setattr(
        "node_runner.src.metrics.evaluators.dredd.SubprocessManager", _OkSM
    )
    evaluator.run_phase("generation")


def test_after_run_invokes_jd_tar_and_symlink(monkeypatch, tmp_path: Path):
    evaluator = _build_evaluator(tmp_path)
    evaluator.artifacts_path = tmp_path / "artifacts"

    called: dict[str, object] = {}

    def _tar(metrics_dir, original_artifacts_path, _name):
        called["tar"] = (metrics_dir, original_artifacts_path)
        return tmp_path / "tar.zst"

    def _symlink(tar_path, hf_path, _name):
        called["symlink"] = (tar_path, hf_path)

    monkeypatch.setattr(
        evaluator,
        "_send_metrics_to_jd",
        lambda metrics: called.setdefault("send", metrics),
    )
    monkeypatch.setattr(JD, "tar_metrics_dir", _tar)
    monkeypatch.setattr(JD, "symlink_metrics_for_transfer", _symlink)

    evaluator.after_run({"dredd/x/y/score_0_1": 0.42})
    assert called["send"] == {"dredd/x/y/score_0_1": 0.42}
    assert called["tar"] == (
        evaluator.metrics_dir,
        evaluator.artifacts_path,
    )
    assert called["symlink"] == (
        tmp_path / "tar.zst",
        evaluator.hf_checkpoint_path,
    )


def test_convert_metrics_to_jd_keys_duplicate_warning(monkeypatch, tmp_path: Path):
    evaluator = _build_evaluator(tmp_path)
    evaluator.autometric_runs_config = _dredd_config(
        {"judge": {"model_name": "minimax-m2.7"}}
    )
    monkeypatch.setattr(
        DreddEvaluator,
        "_normalize_metric_name_for_jd",
        staticmethod(lambda _name: "SAME_KEY"),
    )
    converted = evaluator._convert_metrics_to_jd_keys(
        {
            "dredd/arena1/b1/score_0_1": 0.5,
            "dredd/arena2/b2/score_0_1": 0.7,
        }
    )
    assert converted == {"SAME_KEY": 0.7}


@pytest.mark.parametrize(
    "case",
    [
        "complete",
        "missing_runtime",
        "export_failed",
        "missing_summary",
        "failed_judgments",
        "wrong_count",
        "malformed",
    ],
)
def test_ci_requires_complete_exported_judgments(tmp_path, monkeypatch, case):
    from unittest.mock import Mock

    evaluator = _build_evaluator(tmp_path)
    (evaluator.metrics_dir / "dredd_merged_config.yaml").write_text("{}")
    manager = Mock()
    manager.is_successful.return_value = case != "export_failed"
    if case == "missing_runtime":
        manager.start.side_effect = OSError("runtime unavailable")
    subprocess = Mock(return_value=manager)
    monkeypatch.setattr(
        "node_runner.src.metrics.evaluators.dredd.SubprocessManager", subprocess
    )
    judgment = evaluator.metrics_dir / "arenas/general/judgments/baseline"
    judgment.mkdir(parents=True)
    summary = {"total": 2, "failed_judgments": 0}
    if case == "failed_judgments":
        summary["failed_judgments"] = 1
    elif case == "wrong_count":
        summary["total"] = 3
    if case != "missing_summary":
        (judgment / "judgment_summary.jsonl").write_text(json.dumps(summary))
    (judgment / "ci_quality.json").write_text(
        "invalid"
        if case == "malformed"
        else json.dumps(
            {"identity": {"judge": "test-judge"}, "samples": [["a", 1], ["b", 0.5]]}
        )
    )
    key = "GENERAL_VS_BASELINE_JUDGE_TEST_JUDGE"
    scores = evaluator._cache_scores({key: 0.75})
    assert subprocess.call_args.args[0] == [
        evaluator.DREDD_PYTHON,
        str(get_dredd_submodule_path() / "export_ci.py"),
        str(evaluator.metrics_dir),
    ]
    assert {
        key: [row["score"] for row in rows] for key, rows in scores.records.items()
    } == ({key: [1, 0.5]} if case == "complete" else {})


def test_dredd_get_metrics_accepts_nested_data_root(tmp_path, monkeypatch):
    evaluator = _build_evaluator(tmp_path)
    evaluator._dredd_phase = "judgement"
    _write_complete_summary(evaluator, tmp_path, monkeypatch)
    monkeypatch.setenv("DREDD_DATA_ROOT", str(tmp_path))

    assert evaluator.get_metrics()["dredd/arena-hard-logs-v3/gpt-4o/score_0_1"] == 0.6


def test_dredd_get_metrics_rejects_unverifiable_coverage(tmp_path, monkeypatch):
    evaluator = _build_evaluator(tmp_path)
    evaluator._dredd_phase = "judgement"
    _write_complete_summary(evaluator, tmp_path, monkeypatch)
    monkeypatch.setenv("DREDD_DATA_ROOT", str(tmp_path / "missing"))
    errors = []
    monkeypatch.setattr(
        "node_runner.src.metrics.evaluators.dredd.Logger.error", errors.append
    )

    assert evaluator.get_metrics() == {}
    assert any("cannot verify coverage" in message for message in errors)


@pytest.mark.parametrize("win_rate", [None, "unavailable"])
def test_dredd_get_metrics_ignores_missing_or_invalid_optional_rate(
    tmp_path, monkeypatch, win_rate
):
    evaluator = _build_evaluator(tmp_path)
    evaluator._dredd_phase = "judgement"
    summary_path = _write_complete_summary(evaluator, tmp_path, monkeypatch)
    summary = json.loads(summary_path.read_text())
    if win_rate is None:
        del summary["win_rate"]
    else:
        summary["win_rate"] = win_rate
    summary_path.write_text(json.dumps(summary))

    metrics = evaluator.get_metrics()
    assert metrics["dredd/arena-hard-logs-v3/gpt-4o/score_0_1"] == 0.6
    assert "dredd/arena-hard-logs-v3/gpt-4o/win_rate" not in metrics
