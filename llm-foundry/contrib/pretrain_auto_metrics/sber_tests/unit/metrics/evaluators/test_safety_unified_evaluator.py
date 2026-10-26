import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pydantic
import pytest
from omegaconf import OmegaConf

from node_runner.src.config_manager import ConfigManager
from node_runner.src.config_model import DEFAULT_REQUEST_TIMEOUT
from node_runner.src.config_model import ExternalApiInferenceConfig
from node_runner.src.config_model import ModelInfoConfig as ModelInfo
from node_runner.src.config_model import SafetyUnifiedEvaluatorConfig
from node_runner.src.config_model import NodeRunnerConfigModel
from node_runner.src.inference.external_api.proxy.mappers import OpenAiMapper
from node_runner.src.inference.inference_manager import InferenceManager
from node_runner.src.job_dispatcher import JD
from node_runner.src.metrics.evaluators.safety_unified import SafetyUnifiedEvaluator
from node_runner.src.metrics.metrics_manager import AutoMetricsManager
from node_runner.src.metrics import tensorboard_saver
from node_runner.src.pam_proxies.models import SGLangChatCompletionRequest
from sber_tests.test_support import judge_endpoints

pytestmark = pytest.mark.unit

_DEFAULT_MODEL_SECTION = {
    **SafetyUnifiedEvaluatorConfig.DEFAULT_MODEL,
    "timeout": DEFAULT_REQUEST_TIMEOUT,
}
_DEFAULT_JUDGE_SECTION = {
    **SafetyUnifiedEvaluatorConfig.DEFAULT_JUDGE,
    "timeout": DEFAULT_REQUEST_TIMEOUT,
    "parallel": 8,
}


def _cfg(judges=None, **kwargs) -> SafetyUnifiedEvaluatorConfig:
    if judges is None:
        judges = {"minimax-m2.7": "all"}
    kwargs.setdefault("dataset_root", "/data/su")
    return SafetyUnifiedEvaluatorConfig(judges=judges, **kwargs)


def _setup(
    tmp_path: Path,
    cfg: SafetyUnifiedEvaluatorConfig,
    *,
    apply: bool = True,
    checkpoint: Path | None = None,
) -> SafetyUnifiedEvaluator:
    evaluator = SafetyUnifiedEvaluator()
    if apply:
        cfg.judge_endpoints = judge_endpoints(
            {name: "http://judge:30000" for name in cfg.judges}
        )
        cfg.apply_defaults(
            inference_instances=1,
            base_url="http://cand:29999",
        )
    artifacts_path = tmp_path / "artifacts"
    artifacts_path.mkdir(parents=True, exist_ok=True)
    evaluator.setup(
        autometric_runs_config=cfg,
        artifacts_path=artifacts_path,
        hf_checkpoint_path=checkpoint
        or tmp_path / "model" / "ep1_ba2" / "artifacts" / "hf_checkpoint",
        model_info=ModelInfo(model_class="sft", size="lite", username="u"),
    )
    return evaluator


class TestConfig:
    @pytest.mark.parametrize(
        "instances,judge_parallel,expected",
        [(1, None, 8), (8, None, 64), (16, None, 128), (16, 5, 5)],
    )
    def test_judge_concurrency_scales_with_primary_instances(
        self, instances, judge_parallel, expected
    ):
        config = _cfg(judge_parallel=judge_parallel)

        config.judge_endpoints = judge_endpoints({"minimax-m2.7": "http://judge"})
        config.apply_defaults(
            inference_instances=instances,
            base_url="http://candidate",
        )

        assert config.judge_parallel == expected
        assert config.parallel == instances * 16

    def test_normalize_scalar_task_to_list(self):
        assert _cfg(judges={"j": "all"}).judges == {"j": ["all"]}

    def test_normalize_keeps_list(self):
        assert _cfg(judges={"j": ["a", "b"]}).judges == {"j": ["a", "b"]}

    def test_normalize_non_dict_passthrough(self):
        with pytest.raises(pydantic.ValidationError):
            SafetyUnifiedEvaluatorConfig.model_validate(
                {"judges": ["a"], "dataset_root": "/d"}
            )

    def test_dataset_files_expands_all(self):
        assert _cfg(judges={"j": "all"}).dataset_files_for("j") == list(
            SafetyUnifiedEvaluatorConfig.ALL_DATASET_FILES
        )

    def test_dataset_files_wraps_single_name(self):
        cfg = _cfg(judges={"j": []}, dataset_file="one.jsonl")
        assert cfg.dataset_files_for("j") == ["one.jsonl"]

    def test_dataset_files_keeps_list(self):
        cfg = _cfg(judges={"j": ["a.jsonl", "b.jsonl"]})
        assert cfg.dataset_files_for("j") == ["a.jsonl", "b.jsonl"]

    def test_multiple_dataset_scope_is_order_independent(self):
        first = _cfg(judges={"j": ["b.jsonl", "a.jsonl"]})
        second = _cfg(judges={"j": ["a.jsonl", "b.jsonl"]})
        assert first.dataset_scope_for("j") == second.dataset_scope_for("j") == "a_b"

    def test_secondary_model_names_lists_judges(self):
        assert _cfg(judges={"a": "all", "b": ["x"]}).secondary_model_names == ["a", "b"]

    def test_empty_judges_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            _cfg(judges={})

    def test_blank_judge_name_raises(self):
        with pytest.raises(ValueError, match="non-empty model names"):
            _cfg(judges={"  ": "all"})

    def test_apply_defaults_sets_endpoints(self):
        cfg = _cfg()
        cfg.judge_endpoints = judge_endpoints({"minimax-m2.7": "http://j:2"})
        cfg.apply_defaults(
            inference_instances=1,
            base_url="http://c:1",
        )
        assert cfg.base_url == "http://c:1/v1"
        assert cfg.require_base_url() == "http://c:1/v1"

    def test_build_overrides_uses_url_of_each_judge(self):
        cfg = _cfg(judges={"judge-a": "all", "judge-b": "all"})
        cfg.judge_endpoints = judge_endpoints(
            {
                "judge-a": "http://a:31000",
                "judge-b": "http://b:31001",
            }
        )
        cfg.apply_defaults(
            inference_instances=1,
            base_url="http://c:1",
        )

        api_bases = {
            judge: cfg.build_overrides(
                judge_model_name=judge,
                out_dir=Path("metrics"),
                default_answers_dir=Path("answers"),
                checkpoint_model_name="candidate",
                checkpoint_path=Path("candidate"),
            )["judge"]["api_base"]
            for judge in ("judge-a", "judge-b")
        }

        assert api_bases == {
            "judge-a": "http://a:31000/v1",
            "judge-b": "http://b:31001/v1",
        }

    def test_apply_defaults_scales_parallelism_with_instances(self):
        cfg = _cfg()
        cfg.judge_endpoints = judge_endpoints({"minimax-m2.7": "http://j:2"})
        cfg.apply_defaults(
            inference_instances=4,
            base_url="http://c:1",
        )
        assert cfg.parallel == 64
        assert cfg.judge_parallel == 32

    def test_apply_defaults_keeps_explicit_parallelism(self):
        cfg = _cfg(parallel=3, judge_parallel=5)
        cfg.judge_endpoints = judge_endpoints({"minimax-m2.7": "http://j:2"})
        cfg.apply_defaults(
            inference_instances=4,
            base_url="http://c:1",
        )
        assert cfg.parallel == 3
        assert cfg.judge_parallel == 5

    @pytest.mark.parametrize("key", ["judge_timeoutt", "python_path"])
    def test_unknown_key_is_rejected(self, key):
        with pytest.raises(pydantic.ValidationError, match=key):
            _cfg(**{key: "value"})

    def test_default_llm_timeout_is_the_shared_request_timeout(self):
        assert _cfg().llm_timeout == DEFAULT_REQUEST_TIMEOUT

    def test_several_judges_on_one_task_are_allowed(self):
        cfg = _cfg(judges={"Qwen3.5-9B": "all", "openrouter:qwen/qwen3.5-9b": "all"})
        assert cfg.secondary_model_names == [
            "Qwen3.5-9B",
            "openrouter:qwen/qwen3.5-9b",
        ]

    def test_default_runner_root_is_image_path(self):
        assert _cfg().runner_root == "/home/user/safety_unified/runner"

    def test_default_judge_prompt_is_production_policy(self):
        assert _cfg().judge_prompt_file == "judge_prompt_production.txt"


class TestBasics:
    def test_judge_name_is_kept_as_configured(self):
        assert SafetyUnifiedEvaluator._judge_key("Qwen3.5-9B") == "Qwen3.5-9B"

    def test_gateway_judge_name_stays_one_path_segment(self, tmp_path):
        judge_model_name = "openrouter:qwen/qwen3-235b-a22b-2507"
        evaluator = _setup(tmp_path, _cfg(judges={judge_model_name: "all"}))
        assert evaluator._judge_out_dir(judge_model_name) == (
            Path(evaluator.metrics_dir) / "openrouter:qwen_qwen3-235b-a22b-2507"
        )
        assert evaluator._judge_metric_prefix(judge_model_name) == (
            "safety_unified/judge_openrouter:qwen_qwen3-235b-a22b-2507/"
        )

    def test_setup_state(self, tmp_path):
        cfg = _cfg(judges={"a": "all", "b": "all"})
        evaluator = _setup(tmp_path, cfg)
        assert evaluator.autometric_runs_config is cfg
        assert evaluator.metrics_dir.name == "safety_unified"
        assert evaluator._judge_tasks == cfg.judges
        assert evaluator._runner_root == Path("/home/user/safety_unified/runner")


class TestBuildOverrides:
    def test_default_judge_request_matches_published_comparison_protocol(
        self, tmp_path
    ):
        judge_name = "Qwen3-235B-A22B-Instruct-2507-FP8"
        evaluator = _setup(tmp_path, _cfg(judges={judge_name: "all"}))

        judge = evaluator._build_overrides(judge_name)["judge"]

        assert {
            key: judge.get(key) for key in ("max_tokens", "seed", "temperature")
        } == {
            "max_tokens": 1536,
            "seed": 12345,
            "temperature": 0.0,
        }

    def test_parallelism_must_come_from_apply_defaults(self, tmp_path, monkeypatch):
        cfg = _cfg()
        cfg.base_url = "http://cand:29999/v1"
        evaluator = _setup(tmp_path, cfg, apply=False)
        monkeypatch.setattr(JD, "parse_checkpoint_path", lambda p: ("nfs", "m", 1, 2))

        with pytest.raises(ValueError, match="must be set via apply_defaults"):
            evaluator._build_overrides("minimax-m2.7")

    def test_all_tasks(self, tmp_path, monkeypatch):
        evaluator = _setup(tmp_path, _cfg(judges={"minimax-m2.7": "all"}))
        monkeypatch.setattr(
            JD, "parse_checkpoint_path", lambda p: ("nfs", "ckpt-model", 1, 2)
        )
        overrides = evaluator._build_overrides("minimax-m2.7")
        assert overrides["dataset_file"] == "all"
        assert overrides["model"] == {
            **_DEFAULT_MODEL_SECTION,
            "api_base": "http://cand:29999/v1",
            "name": "ckpt-model",
            "checkpoint": str(
                tmp_path / "model" / "ep1_ba2" / "artifacts" / "hf_checkpoint"
            ),
        }
        assert overrides["judge"] == {
            **_DEFAULT_JUDGE_SECTION,
            "model_name": "minimax-m2.7",
            "api_base": "http://judge:30000/v1",
        }
        assert overrides["dataset_root"] == "/data/su"
        assert overrides["parallel"] == 16 and "limit" not in overrides

    def test_same_model_checkpoints_reach_generation_as_distinct_candidates(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            JD, "parse_checkpoint_path", lambda p: ("nfs", "same-model", 1, 10)
        )
        generations = []

        def phase(self, name, overrides):
            if name == "generation":
                generations.append(overrides["model"])
            return True

        monkeypatch.setattr(SafetyUnifiedEvaluator, "_run_phase", phase)
        checkpoints = [
            tmp_path / "same-model" / "ep1_ba10" / "artifacts" / "hf_checkpoint",
            tmp_path / "same-model" / "ep1_ba20" / "artifacts" / "hf_checkpoint",
        ]
        for checkpoint in checkpoints:
            evaluator = _setup(tmp_path, _cfg(), checkpoint=checkpoint)
            evaluator.run_on_model_name("minimax-m2.7")

        assert [model["name"] for model in generations] == ["same-model", "same-model"]
        assert [model["checkpoint"] for model in generations] == [
            str(path) for path in checkpoints
        ]

    def test_default_production_schema_reaches_the_runner(self, tmp_path):
        evaluator = _setup(tmp_path, _cfg())

        assert (
            evaluator._build_overrides("minimax-m2.7")["judge_schema"] == "production"
        )

    def test_default_dataset_root_is_image_path(self, tmp_path, monkeypatch):
        cfg = SafetyUnifiedEvaluatorConfig(judges={"minimax-m2.7": "all"})
        evaluator = _setup(tmp_path, cfg)
        monkeypatch.setattr(JD, "parse_checkpoint_path", lambda p: ("nfs", "m", 1, 2))
        overrides = evaluator._build_overrides("minimax-m2.7")
        assert overrides["dataset_root"] == "/home/user/safety_unified/data"

    def test_yaml_overrides_win_over_code_defaults(self, tmp_path, monkeypatch):
        evaluator = _setup(
            tmp_path,
            _cfg(
                model={"name": "served", "timeout": 60},
                judge={"max_tokens": 32, "seed": 7},
            ),
        )
        monkeypatch.setattr(JD, "parse_checkpoint_path", lambda p: ("nfs", "m", 1, 2))
        overrides = evaluator._build_overrides("minimax-m2.7")
        assert overrides["model"]["name"] == "served"
        assert overrides["model"]["timeout"] == 60
        assert (
            overrides["model"]["max_tokens"]
            == SafetyUnifiedEvaluatorConfig.DEFAULT_MODEL["max_tokens"]
        )
        assert overrides["judge"]["max_tokens"] == 32
        assert overrides["judge"]["seed"] == 7
        assert overrides["judge"]["timeout"] == DEFAULT_REQUEST_TIMEOUT

    def test_effective_timeouts_match_runner_defaults(self, tmp_path, monkeypatch):
        evaluator = _setup(tmp_path, _cfg())
        monkeypatch.setattr(JD, "parse_checkpoint_path", lambda p: ("nfs", "m", 1, 2))
        overrides = evaluator._build_overrides("minimax-m2.7")
        assert overrides["model"]["timeout"] == DEFAULT_REQUEST_TIMEOUT
        assert overrides["judge"]["timeout"] == DEFAULT_REQUEST_TIMEOUT

    def test_llm_timeout_drives_both_sections(self, tmp_path, monkeypatch):
        evaluator = _setup(tmp_path, _cfg(llm_timeout=120))
        monkeypatch.setattr(JD, "parse_checkpoint_path", lambda p: ("nfs", "m", 1, 2))
        overrides = evaluator._build_overrides("minimax-m2.7")
        assert overrides["model"]["timeout"] == 120
        assert overrides["judge"]["timeout"] == 120

    def test_judge_parallel_and_explicit_judge_timeout(self, tmp_path, monkeypatch):
        evaluator = _setup(tmp_path, _cfg(judge_parallel=8, judge={"timeout": 300}))
        monkeypatch.setattr(JD, "parse_checkpoint_path", lambda p: ("nfs", "m", 1, 2))
        judge = evaluator._build_overrides("minimax-m2.7")["judge"]
        assert judge["parallel"] == 8
        assert judge["timeout"] == 300

    def test_answers_dir_override(self, tmp_path, monkeypatch):
        evaluator = _setup(tmp_path, _cfg(answers_dir="/cache/answers"))
        monkeypatch.setattr(JD, "parse_checkpoint_path", lambda p: ("nfs", "m", 1, 2))
        assert (
            evaluator._build_overrides("minimax-m2.7")["answers_dir"]
            == "/cache/answers"
        )

    def test_explicit_task_list(self, tmp_path, monkeypatch):
        evaluator = _setup(tmp_path, _cfg(judges={"j": ["red.jsonl", "green.jsonl"]}))
        monkeypatch.setattr(JD, "parse_checkpoint_path", lambda p: ("nfs", "m", 1, 2))
        overrides = evaluator._build_overrides("j")
        assert overrides["dataset_file"] == ["red.jsonl", "green.jsonl"]

    def test_empty_tasks_uses_dataset_file(self, tmp_path, monkeypatch):
        evaluator = _setup(
            tmp_path, _cfg(judges={"j": []}, dataset_file="fallback.jsonl")
        )
        monkeypatch.setattr(JD, "parse_checkpoint_path", lambda p: ("nfs", "m", 1, 2))
        assert evaluator._build_overrides("j")["dataset_file"] == "fallback.jsonl"

    def test_empty_tasks_defaults_to_all(self, tmp_path, monkeypatch):
        evaluator = _setup(tmp_path, _cfg(judges={"j": []}))
        monkeypatch.setattr(JD, "parse_checkpoint_path", lambda p: ("nfs", "m", 1, 2))
        assert evaluator._build_overrides("j")["dataset_file"] == "all"

    def test_checkpoint_name_fallback_model(self, tmp_path, monkeypatch):
        evaluator = _setup(tmp_path, _cfg())
        monkeypatch.setattr(JD, "parse_checkpoint_path", lambda p: ("nfs", "", 1, 2))
        assert evaluator._build_overrides("minimax-m2.7")["model"]["name"] == "model"

    def test_options_passed_through(self, tmp_path, monkeypatch):
        evaluator = _setup(
            tmp_path,
            _cfg(
                parallel=4,
                limit=10,
                mutate=True,
                mutation_seed=7,
                color_weights={"red": 1.0},
                domain_weights={"d": 2.0},
                judge_prompt_file="judge_prompt_qwen.txt",
            ),
        )
        monkeypatch.setattr(JD, "parse_checkpoint_path", lambda p: ("nfs", "m", 1, 2))
        overrides = evaluator._build_overrides("minimax-m2.7")
        assert overrides["parallel"] == 4 and overrides["limit"] == 10
        assert overrides["mutate"] is True and overrides["mutation_seed"] == 7
        assert overrides["color_weights"] == {"red": 1.0}
        assert overrides["domain_weights"] == {"d": 2.0}
        assert overrides["judge_prompt_file"] == "judge_prompt_qwen.txt"

    def test_judge_prompt_files_per_judge(self, tmp_path, monkeypatch):
        evaluator = _setup(
            tmp_path,
            _cfg(
                judges={"minimax-m2.7": "all", "Qwen3.5-9B": "all"},
                judge_prompt_file="judge_prompt.txt",
                judge_prompt_files={"Qwen3.5-9B": "judge_prompt_qwen.txt"},
            ),
        )
        monkeypatch.setattr(JD, "parse_checkpoint_path", lambda p: ("nfs", "m", 1, 2))
        assert (
            evaluator._build_overrides("minimax-m2.7")["judge_prompt_file"]
            == "judge_prompt.txt"
        )
        assert (
            evaluator._build_overrides("Qwen3.5-9B")["judge_prompt_file"]
            == "judge_prompt_qwen.txt"
        )

    def test_judge_pool_reaches_the_runner(self, tmp_path, monkeypatch):
        evaluator = _setup(
            tmp_path,
            _cfg(
                judges={"minimax-m2.7": "all"},
                judge_pool=["a.txt", {"prompt": "b.txt", "name": "b"}],
                judge_min_agreement=0.6,
            ),
        )
        monkeypatch.setattr(JD, "parse_checkpoint_path", lambda p: ("nfs", "m", 1, 2))
        overrides = evaluator._build_overrides("minimax-m2.7")
        assert overrides["judge_pool"] == ["a.txt", {"prompt": "b.txt", "name": "b"}]
        assert overrides["judge_min_agreement"] == 0.6

    def test_judge_pool_defaults_to_the_single_prompt(self, tmp_path, monkeypatch):
        evaluator = _setup(tmp_path, _cfg(judges={"minimax-m2.7": "all"}))
        monkeypatch.setattr(JD, "parse_checkpoint_path", lambda p: ("nfs", "m", 1, 2))
        overrides = evaluator._build_overrides("minimax-m2.7")
        assert overrides["judge_pool"] == []
        assert overrides["judge_min_agreement"] == 0.75

    def test_judge_min_agreement_must_be_a_share(self):
        with pytest.raises(pydantic.ValidationError):
            _cfg(judges={"minimax-m2.7": "all"}, judge_min_agreement=1.5)

    def test_judge_chat_template_kwargs_per_judge(self, tmp_path, monkeypatch):
        evaluator = _setup(
            tmp_path,
            _cfg(
                judges={"minimax-m2.7": "all", "Qwen3.5-9B": "all"},
                judge_chat_template_kwargs={"Qwen3.5-9B": {"enable_thinking": False}},
            ),
        )
        monkeypatch.setattr(JD, "parse_checkpoint_path", lambda p: ("nfs", "m", 1, 2))
        assert evaluator._build_overrides("Qwen3.5-9B")["judge"][
            "chat_template_kwargs"
        ] == {"enable_thinking": False}
        assert (
            "chat_template_kwargs"
            not in evaluator._build_overrides("minimax-m2.7")["judge"]
        )

    def test_judge_extra_keys_passed_through(self, tmp_path, monkeypatch):
        evaluator = _setup(tmp_path, _cfg(judge={"seed": 12345}))
        monkeypatch.setattr(JD, "parse_checkpoint_path", lambda p: ("nfs", "m", 1, 2))
        judge = evaluator._build_overrides("minimax-m2.7")["judge"]
        assert judge["seed"] == 12345
        assert judge["model_name"] == "minimax-m2.7"
        assert judge["api_base"] == "http://judge:30000/v1"

    def test_missing_base_url_raises(self, tmp_path, monkeypatch):
        cfg = _cfg()
        cfg.judge_endpoints = judge_endpoints({"minimax-m2.7": "http://judge:30000"})
        cfg.parallel = 4
        cfg.judge_parallel = 8
        evaluator = _setup(tmp_path, cfg, apply=False)
        monkeypatch.setattr(JD, "parse_checkpoint_path", lambda p: ("nfs", "m", 1, 2))
        with pytest.raises(ValueError, match="base_url must be set"):
            evaluator._build_overrides("minimax-m2.7")


@pytest.fixture
def runner_process(monkeypatch):
    calls = {"exit_code": 0, "commands": [], "logs": [], "started": 0, "waited": 0}

    class Process:
        def __init__(self, cmd, log_file, cwd):
            calls["commands"].append(cmd)

        def start(self):
            calls["started"] += 1

        def wait_for_completion(self):
            calls["waited"] += 1

        def log_error(self):
            calls["logs"].append(calls["exit_code"])

        def is_successful(self):
            return calls["exit_code"] == 0

    monkeypatch.setattr(
        "node_runner.src.metrics.evaluators.safety_unified.SubprocessManager", Process
    )
    return calls


def _runner(tmp_path):
    path = tmp_path / "runner"
    path.mkdir(exist_ok=True)
    (path / "run_safety_unified.py").write_text("")
    return path


def _dataset(tmp_path, rows):
    root = tmp_path / "data"
    root.mkdir(exist_ok=True)
    (root / "ds.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return root


def _answer(root, item_id="a", **overrides):
    root.mkdir(exist_ok=True, parents=True)
    (root / f"{item_id}.json").write_text(
        json.dumps({"generated": "answer", "ok": True, **overrides})
    )


def _metrics(evaluator, judge="j", **overrides):
    root = Path(evaluator.metrics_dir).parent.parent / "dataset"
    root.mkdir(exist_ok=True)
    for index, filename in enumerate(
        evaluator.autometric_runs_config.dataset_files_for(judge)
    ):
        (root / filename).write_text('{"id":"a"}\n' if index == 0 else "")
    evaluator.autometric_runs_config.dataset_root = str(root)
    out = evaluator._judge_out_dir(judge)
    out.mkdir(exist_ok=True)
    values = {
        "safety_unified/overall/n_total": 1,
        "safety_unified/overall/n_judged": 1,
        "safety_unified/overall/coverage": 1.0,
        "safety_unified/overall/n_parse_fail": 0,
        "safety_unified/overall/n_no_answer": 0,
        "safety_unified/v2/score": 0.6,
        "safety_unified/v2/red_fail_rate": 0.1,
        "safety_unified/v2/content_safe_rate": None,
        **overrides,
    }
    path = out / "safety_unified_metrics.json"
    path.write_text(json.dumps(values))
    manifest = {
        "runner_version": "0.7.0",
        "metric_version": 3,
        "metric_identity": "a" * 64,
        "n_expected": 1,
        "n_scored": 1,
        "metrics_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (out / "safety_unified_complete.json").write_text(json.dumps(manifest))
    evaluator._completed_judges.add(judge)
    return path


class TestRunPhase:
    @pytest.mark.parametrize("exit_code, expected", [(0, True), (1, False)])
    def test_subprocess_status_and_log(
        self, tmp_path, runner_process, exit_code, expected
    ):
        evaluator = _setup(tmp_path, _cfg())
        evaluator._runner_root = _runner(tmp_path)
        runner_process["exit_code"] = exit_code
        assert evaluator._run_phase("generation", {"parallel": 2}) is expected
        assert runner_process["started"] == 1
        assert runner_process["logs"] == [exit_code]
        assert runner_process["commands"][0] == [
            sys.executable,
            str(evaluator._runner_root / "run_safety_unified.py"),
            "--phase",
            "generation",
            "--overrides-json",
            '{"parallel": 2}',
        ]

    def test_missing_runner_stops_without_raise(self, tmp_path, runner_process):
        evaluator = _setup(tmp_path, _cfg())
        evaluator._runner_root = tmp_path / "missing_runner"
        assert evaluator._run_phase("generation", {}) is False
        assert runner_process["started"] == 0

    def test_overrides_are_not_logged(self, tmp_path, monkeypatch, runner_process):
        evaluator = _setup(tmp_path, _cfg())
        evaluator._runner_root = _runner(tmp_path)
        logs = []
        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.safety_unified.Logger.info", logs.append
        )
        evaluator._run_phase("generation", {"model": {"api_key_env": "WORKLOAD_KEY"}})
        assert logs == ["SafetyUnified run phase=generation"]


class TestRunOnModelName:
    @pytest.mark.parametrize("answers_path_type", ["missing", "file"])
    def test_invalid_answers_directory_stops_before_any_phase(
        self, tmp_path, monkeypatch, answers_path_type
    ):
        root = _dataset(tmp_path, [{"id": "a"}])
        answers = tmp_path / "answers"
        if answers_path_type == "file":
            answers.write_text("not a directory")
        evaluator = _setup(
            tmp_path,
            _cfg(
                judges={"j": ["ds.jsonl"]},
                dataset_root=str(root),
                answers_dir=str(answers),
            ),
        )
        phases = []
        monkeypatch.setattr(
            evaluator, "_run_phase", lambda *args: phases.append(args) or True
        )
        evaluator.run_on_model_name("j")
        assert phases == []
        assert evaluator.get_metrics() == {}

    @pytest.mark.parametrize("payload", [None, b"", b"\xff", b"{\n"])
    def test_unreadable_or_empty_dataset_cannot_reuse_answers(
        self, tmp_path, monkeypatch, payload
    ):
        root = _dataset(tmp_path, [{"id": "a"}])
        dataset = root / "ds.jsonl"
        if payload is None:
            dataset.unlink()
        else:
            dataset.write_bytes(payload)
        answers = tmp_path / "answers"
        _answer(answers)
        evaluator = _setup(
            tmp_path,
            _cfg(
                judges={"j": ["ds.jsonl"]},
                dataset_root=str(root),
                answers_dir=str(answers),
            ),
        )
        phases = []
        monkeypatch.setattr(
            evaluator, "_run_phase", lambda *args: phases.append(args) or True
        )
        evaluator.run_on_model_name("j")
        assert phases == []
        assert evaluator.get_metrics() == {}

    def test_blank_dataset_lines_preserve_fallback_id_line_numbers(self, tmp_path):
        root = _dataset(tmp_path, [])
        (root / "ds.jsonl").write_text('\n{}\n \n{"id":"a"}\n')
        evaluator = _setup(
            tmp_path, _cfg(judges={"j": ["ds.jsonl"]}, dataset_root=str(root))
        )
        assert evaluator._dataset_item_ids("j") == ["ds_1", "a"]

    @pytest.mark.parametrize("failed_phase", [None, "generation", "evaluation"])
    def test_phases_stop_at_failure_and_reset_success(
        self, tmp_path, monkeypatch, failed_phase
    ):
        evaluator = _setup(tmp_path, _cfg(judges={"j": "all"}))
        evaluator._completed_judges.add("j")
        calls = []

        def run(phase, overrides):
            calls.append(phase)
            return phase != failed_phase

        monkeypatch.setattr(evaluator, "_run_phase", run)
        evaluator.run_on_model_name("j")
        assert calls == (
            ["generation"]
            if failed_phase == "generation"
            else ["generation", "evaluation"]
        )
        assert evaluator._completed_judges == ({"j"} if failed_phase is None else set())

    def test_reusable_answers_skip_generation(self, tmp_path, monkeypatch):
        root = _dataset(tmp_path, [{"id": "a"}])
        answers = tmp_path / "answers"
        _answer(answers)
        evaluator = _setup(
            tmp_path,
            _cfg(
                judges={"j": ["ds.jsonl"]},
                dataset_root=str(root),
                answers_dir=str(answers),
            ),
        )
        calls = []
        monkeypatch.setattr(
            evaluator, "_run_phase", lambda phase, ov: calls.append(phase) or True
        )
        evaluator.run_on_model_name("j")
        assert calls == ["evaluation"]

    @pytest.mark.parametrize(
        "payload",
        [
            None,
            "{",
            "[]",
            '{"generated":"","ok":true}',
            '{"generated":"   ","ok":true}',
            '{"generated":"text","ok":false}',
            '{"generated":"text","ok":true,"mutate":true}',
            '{"generated":"text","ok":true,"mutation_seed":8}',
        ],
    )
    def test_invalid_answers_never_start_evaluation(
        self, tmp_path, monkeypatch, payload
    ):
        root = _dataset(tmp_path, [{"id": "a"}])
        answers = tmp_path / "answers"
        answers.mkdir()
        if payload is not None:
            (answers / "a.json").write_text(payload)
        evaluator = _setup(
            tmp_path,
            _cfg(
                judges={"j": ["ds.jsonl"]},
                dataset_root=str(root),
                answers_dir=str(answers),
            ),
        )
        calls = []
        monkeypatch.setattr(
            evaluator, "_run_phase", lambda *args: calls.append(args) or True
        )
        evaluator.run_on_model_name("j")
        assert calls == []
        assert evaluator.get_metrics() == {}

    def test_multiple_dataset_limit_matches_runner_interleave(self, tmp_path):
        root = _dataset(tmp_path, [{"id": "a1"}, {"id": "a2"}])
        (root / "other.jsonl").write_text('{"id":"b1"}\n{"id":"b2"}\n')
        evaluator = _setup(
            tmp_path,
            _cfg(
                judges={"j": ["ds.jsonl", "other.jsonl"]},
                dataset_root=str(root),
                limit=2,
            ),
        )
        assert evaluator._dataset_item_ids("j") == ["a1", "b1"]

    @pytest.mark.parametrize(
        "rows", [[{"id": "a"}, {"id": "a"}], [{"id": "../a"}], [["invalid"]]]
    )
    def test_invalid_dataset_has_no_reusable_items(self, tmp_path, rows):
        root = _dataset(tmp_path, rows)
        evaluator = _setup(
            tmp_path, _cfg(judges={"j": ["ds.jsonl"]}, dataset_root=str(root))
        )
        assert evaluator._dataset_item_ids("j") == []


class TestMetrics:
    def test_unconfigured_metrics_directory_returns_no_metrics(self, tmp_path):
        evaluator = _setup(tmp_path, _cfg(judges={"j": "all"}))
        evaluator.metrics_dir = None
        assert evaluator.get_metrics() == {}

    @pytest.mark.parametrize(
        "filename", ["safety_unified_metrics.json", "safety_unified_complete.json"]
    )
    def test_failed_artifact_invalidation_stops_rerun_and_publication(
        self, tmp_path, monkeypatch, filename
    ):
        evaluator = _setup(tmp_path, _cfg(judges={"j": "all"}))
        path = _metrics(evaluator)
        assert evaluator.get_metrics()
        blocked = path.parent / filename
        unlink = Path.unlink
        phases = []

        def invalidate(path, *args, **kwargs):
            if path == blocked:
                raise PermissionError("artifact is read only")
            return unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", invalidate)
        monkeypatch.setattr(
            evaluator, "_run_phase", lambda *args: phases.append(args) or True
        )
        evaluator.run_on_model_name("j")
        assert phases == []
        assert evaluator._completed_judges == set()
        assert evaluator._metric_identities == {}
        assert evaluator.get_metrics() == {}

    def test_complete_run_keeps_null_slices_out_of_numeric_metrics(self, tmp_path):
        evaluator = _setup(tmp_path, _cfg(judges={"j": "all"}))
        _metrics(evaluator)
        assert evaluator.get_metrics() == {
            "safety_unified/judge_j/overall/n_total": 1.0,
            "safety_unified/judge_j/overall/n_judged": 1.0,
            "safety_unified/judge_j/overall/coverage": 1.0,
            "safety_unified/judge_j/overall/n_parse_fail": 0.0,
            "safety_unified/judge_j/overall/n_no_answer": 0.0,
            "safety_unified/judge_j/v2/score": 0.6,
            "safety_unified/judge_j/v2/red_fail_rate": 0.1,
        }

    @pytest.mark.parametrize(
        "field,value",
        [
            ("n_judged", 0),
            ("n_total", 2),
            ("coverage", 0.95),
            ("n_no_answer", 1),
            ("n_parse_fail", 1),
        ],
    )
    def test_incomplete_metrics_not_published(self, tmp_path, field, value):
        evaluator = _setup(tmp_path, _cfg(judges={"j": "all"}))
        _metrics(evaluator, **{f"safety_unified/overall/{field}": value})
        assert evaluator.get_metrics() == {}

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), "0.99", True])
    def test_invalid_metric_values_not_published(self, tmp_path, value):
        evaluator = _setup(tmp_path, _cfg(judges={"j": "all"}))
        _metrics(evaluator, **{"safety_unified/v2/score": value})
        assert evaluator.get_metrics() == {}

    @pytest.mark.parametrize(
        "change",
        [
            "absent",
            "invalid_json",
            "changed_metrics",
            "wrong_version",
            "wrong_count",
            "wrong_identity",
        ],
    )
    def test_invalid_completion_not_published(self, tmp_path, change):
        evaluator = _setup(tmp_path, _cfg(judges={"j": "all"}))
        path = _metrics(evaluator)
        manifest = path.parent / "safety_unified_complete.json"
        if change == "absent":
            manifest.unlink()
        elif change == "invalid_json":
            manifest.write_text("{")
        elif change == "changed_metrics":
            path.write_text(path.read_text() + " ")
        else:
            payload = json.loads(manifest.read_text())
            key, value = {
                "wrong_version": ("metric_version", 2),
                "wrong_count": ("n_scored", 0),
                "wrong_identity": ("metric_identity", "unknown"),
            }[change]
            payload[key] = value
            manifest.write_text(json.dumps(payload))
        assert evaluator.get_metrics() == {}

    def test_failed_rerun_cannot_publish_old_metrics(self, tmp_path, monkeypatch):
        evaluator = _setup(tmp_path, _cfg(judges={"j": "all"}))
        _metrics(evaluator)
        assert evaluator.get_metrics()
        monkeypatch.setattr(evaluator, "_run_phase", lambda *args: False)
        evaluator.run_on_model_name("j")
        assert evaluator.get_metrics() == {}
        sent = []
        monkeypatch.setattr(evaluator, "_send_metrics_to_jd", sent.append)
        evaluator.after_run({"stale": 1.0})
        assert sent == []

    def test_success_exit_without_fresh_outputs_cannot_reuse_old_metrics(
        self, tmp_path, monkeypatch
    ):
        evaluator = _setup(tmp_path, _cfg(judges={"j": "all"}))
        path = _metrics(evaluator)
        monkeypatch.setattr(evaluator, "_run_phase", lambda *args: True)
        evaluator.run_on_model_name("j")
        assert not path.exists()
        assert not (path.parent / "safety_unified_complete.json").exists()
        assert evaluator.get_metrics() == {}

    @pytest.mark.parametrize(
        "filename", ["safety_unified_metrics.json", "safety_unified_complete.json"]
    )
    def test_invalid_utf8_does_not_escape_metrics_collection(self, tmp_path, filename):
        evaluator = _setup(tmp_path, _cfg(judges={"j": "all"}))
        path = _metrics(evaluator)
        (path.parent / filename).write_bytes(b"\xff")
        assert evaluator.get_metrics() == {}

    def test_one_failed_judge_blocks_entire_result(self, tmp_path):
        evaluator = _setup(tmp_path, _cfg(judges={"j": "all", "other": "all"}))
        _metrics(evaluator)
        assert evaluator.get_metrics() == {}


@pytest.fixture
def jd_publication(monkeypatch):
    calls = {"allowed": True, "sent": []}
    monkeypatch.setattr(JD, "can_send_metrics", lambda **kwargs: calls["allowed"])
    monkeypatch.setattr(
        JD, "send_evaluator_metrics", lambda **kwargs: calls["sent"].append(kwargs)
    )
    return calls


class TestJdMetrics:
    def test_jd_permission_denial_does_not_publish(self, tmp_path, jd_publication):
        evaluator = _setup(tmp_path, _cfg(judges={"j": "all"}))
        _metrics(evaluator)
        jd_publication["allowed"] = False
        evaluator._send_metrics_to_jd(evaluator.get_metrics())
        assert jd_publication["sent"] == []

    def test_unverified_identity_blocks_entire_jd_publication(
        self, tmp_path, jd_publication
    ):
        evaluator = _setup(tmp_path, _cfg(judges={"j": "all", "other": "all"}))
        _metrics(evaluator, "j")
        _metrics(evaluator, "other")
        metrics = evaluator.get_metrics()
        del evaluator._metric_identities["other"]
        evaluator._send_metrics_to_jd(metrics)
        assert jd_publication["sent"] == []

    def test_multiple_judges_publish_only_their_own_metric_prefix(
        self, tmp_path, jd_publication
    ):
        evaluator = _setup(tmp_path, _cfg(judges={"j": "all", "other": "all"}))
        _metrics(evaluator, "j", **{"safety_unified/v2/score": 0.2})
        _metrics(evaluator, "other", **{"safety_unified/v2/score": 0.8})
        evaluator._send_metrics_to_jd(evaluator.get_metrics())
        assert len(jd_publication["sent"]) == 1
        assert jd_publication["sent"][0]["jd_metrics"] == {
            evaluator._jd_metric_name("j", "v2/score"): 0.2,
            evaluator._jd_metric_name("j", "v2/red_fail_rate"): 0.1,
            evaluator._jd_metric_name("other", "v2/score"): 0.8,
            evaluator._jd_metric_name("other", "v2/red_fail_rate"): 0.1,
        }

    def test_colliding_version_aliases_block_jd_publication(
        self, tmp_path, jd_publication
    ):
        evaluator = _setup(tmp_path, _cfg(judges={"j": "all"}))
        _metrics(evaluator)
        metrics = evaluator.get_metrics()
        metrics["safety_unified/judge_j/v3/score"] = 0.9
        evaluator._send_metrics_to_jd(metrics)
        assert jd_publication["sent"] == []

    def test_all_and_diagnostic_slices_are_distinct(self, tmp_path, monkeypatch):
        evaluator = _setup(tmp_path, _cfg(judges={"j": "all"}))
        _metrics(
            evaluator,
            **{
                "safety_unified/ds_a/v2/score": 0.2,
                "safety_unified/confident/v2/score": 0.99,
                "safety_unified/stub_safe/v2/score": 0.8,
            },
        )
        sent = []
        monkeypatch.setattr(
            JD, "send_metrics", lambda body, artifacts_path: sent.append(body)
        )
        evaluator._send_metrics_to_jd(evaluator.get_metrics())
        values = sent[0]["metrics"][2]
        assert len(values) == 5
        assert sorted(values.values()) == [0.1, 0.2, 0.6, 0.8, 0.99]
        assert any(
            "V3_WGT_SCORE" in name and value == 0.6 for name, value in values.items()
        )
        assert any(
            "CONFIDENT_SCORE" in name and value == 0.99
            for name, value in values.items()
        )
        assert sent[0]["name"] == "model"
        assert sent[0]["shooter_version"] == "0.7.0"

    def test_names_include_policy_and_exact_slice_not_candidate(self, tmp_path):
        names = []
        for index, candidate in enumerate(["model-a", "model-b"]):
            evaluator = _setup(
                tmp_path / str(index),
                _cfg(judges={"j": "all"}, model={"name": candidate}),
            )
            evaluator._metric_identities["j"] = "a" * 64
            names.append(evaluator._jd_metric_name("j", "v2/content_safe_rate"))
        assert names[0] == names[1]
        evaluator._metric_identities["j"] = "b" * 64
        assert evaluator._jd_metric_name("j", "v2/content_safe_rate") != names[0]
        assert evaluator._jd_metric_name(
            "j", "ds_a-b/v2/score"
        ) != evaluator._jd_metric_name("j", "ds_a_b/v2/score")

    @pytest.mark.parametrize(
        "judge", ["Qwen3.5-9B", "openrouter:qwen/qwen3-235b-a22b-2507"]
    )
    def test_names_preserve_full_judge_and_slice(self, tmp_path, judge):
        evaluator = _setup(tmp_path, _cfg(judges={judge: "all"}))
        evaluator._metric_identities[judge] = "a" * 64
        for tail in [
            "v2/score",
            "confident/ds_very_long_dataset_name/v2/content_safe_rate",
            "stub_safe/v2/red_fail_rate",
        ]:
            name = evaluator._jd_metric_name(judge, tail)
            assert judge.replace("/", "_").upper() in name
            assert tail.split("/")[-1].upper() in name
            if "ds_very_long_dataset_name" in tail:
                assert "DS_VERY_LONG_DATASET_NAME" in name

    def test_after_run_transfers_complete_artifacts(self, tmp_path, monkeypatch):
        evaluator = _setup(tmp_path, _cfg(judges={"j": "all"}))
        _metrics(evaluator)
        transferred = []
        monkeypatch.setattr(JD, "send_metrics", lambda body, artifacts_path: None)
        monkeypatch.setattr(
            JD, "tar_metrics_dir", lambda *args: tmp_path / "archive.tar.zst"
        )
        monkeypatch.setattr(
            JD,
            "symlink_metrics_for_transfer",
            lambda *args: transferred.append(args),
        )
        evaluator.after_run(evaluator.get_metrics())
        assert transferred == [
            (
                tmp_path / "archive.tar.zst",
                evaluator.hf_checkpoint_path,
                "SafetyUnifiedEvaluator",
            )
        ]


class TestConfigValidation:
    def test_production_schema_reaches_runner(self, tmp_path):
        pool = [
            {
                "name": "policy",
                "prompt": "judge_prompt_production.txt",
                "schema": "production",
            }
        ]
        evaluator = _setup(tmp_path, _cfg(judge_pool=pool))
        assert evaluator._build_overrides("minimax-m2.7")["judge_pool"] == pool

    @pytest.mark.parametrize(
        "settings",
        [
            {"limit": 0},
            {"limit": -1},
        ],
    )
    def test_invalid_inputs_rejected_at_config_load(self, settings):
        with pytest.raises(pydantic.ValidationError):
            _cfg(**settings)

    def test_workload_environment_name_and_strict_coverage_forwarded(self, tmp_path):
        evaluator = _setup(tmp_path, _cfg(judge={"api_key_env": "SAFETY_JUDGE_KEY"}))
        overrides = evaluator._build_overrides("minimax-m2.7")
        assert overrides["judge"]["api_key_env"] == "SAFETY_JUDGE_KEY"
        assert "judges" not in overrides


@pytest.fixture
def selected_evaluator(tmp_path, monkeypatch):
    judges = ["Qwen3-235B-A22B-Instruct-2507-FP8", "Qwen3.5-397B-A17B-FP8"]
    dataset = _dataset(
        tmp_path, [{"id": "a", "messages": [{"role": "user", "content": "question"}]}]
    )
    manifest = {
        "mode": "joint",
        "source_order": [
            model + ":" + role
            for model in ("B235", "B397")
            for role in ("main", "critic", "content")
        ],
        "sources": {"B235:main": {"prompt": "main.txt"}},
    }
    for name in ("policy.txt", "main.txt"):
        (dataset / name).write_text(name)
    (dataset / "selection.json").write_text(json.dumps(manifest))
    evaluator = _setup(
        tmp_path,
        _cfg(
            judges={name: ["ds.jsonl"] for name in judges},
            dataset_root=str(dataset),
            selection_manifest="selection.json",
        ),
    )
    evaluator._runner_root = tmp_path / "packaged_runner"
    evaluator._runner_root.mkdir()
    (evaluator._runner_root / "run_safety_unified.py").write_text(
        'RUNNER_VERSION = "0.7.0"\n'
    )
    monkeypatch.setattr(
        JD, "parse_checkpoint_path", lambda p: ("nfs", "candidate", 1, 2)
    )
    calls = []

    def run(phase, overrides):
        calls.append((phase, overrides))
        root = Path(overrides["out_dir"])
        root.mkdir(parents=True, exist_ok=True)
        if phase == "generation":
            _answer(Path(overrides["answers_dir"]))
        if phase == "evaluation":
            (root / "judgments").mkdir(exist_ok=True)
            for name in (
                "safety_unified_complete.json",
                "safety_unified_metrics.json",
                "judgments/a.json",
            ):
                (root / name).write_text("{}")
            (root / "safety_source_complete.json").write_text(
                json.dumps(
                    {
                        "model": overrides["judge"]["model_name"],
                        "judge_key": overrides["judge"]["model_name"],
                        "runner_version": "0.7.0",
                        "runner_sha256": hashlib.sha256(
                            (
                                evaluator._runner_root / "run_safety_unified.py"
                            ).read_bytes()
                        ).hexdigest(),
                        "format": "safety-source-pass-v2",
                        "completion_sha256": hashlib.sha256(
                            (root / "safety_unified_complete.json").read_bytes()
                        ).hexdigest(),
                        "metrics_sha256": hashlib.sha256(
                            (root / "safety_unified_metrics.json").read_bytes()
                        ).hexdigest(),
                        "records": {
                            "a": {
                                "judgment_sha256": hashlib.sha256(
                                    (root / "judgments/a.json").read_bytes()
                                ).hexdigest()
                            }
                        },
                    }
                )
            )
        if phase == "metrics":
            values = {
                "safety_unified/overall/n_total": 1,
                "safety_unified/overall/n_judged": 1,
                "safety_unified/overall/coverage": 1.0,
                "safety_unified/overall/n_parse_fail": 0,
                "safety_unified/overall/n_no_answer": 0,
                "safety_unified/v2/score": 0.6,
            }
            (root / "safety_unified_metrics.json").write_text(json.dumps(values))
            (root / "selected_verdicts.json").write_text(
                '{"a":{"verdict":{"unsafe":false}}}'
            )
            complete = {
                "runner_version": "0.7.0",
                "metric_version": 3,
                "metric_identity": "a" * 64,
                "n_expected": 1,
                "n_scored": 1,
                "mode": "joint",
                "source_order": manifest["source_order"],
                "metrics_sha256": hashlib.sha256(
                    (root / "safety_unified_metrics.json").read_bytes()
                ).hexdigest(),
                "selection_manifest_sha256": hashlib.sha256(
                    (dataset / "selection.json").read_bytes()
                ).hexdigest(),
                "aggregation_cache_key": "b" * 64,
                "source_proof_hashes": {
                    name: hashlib.sha256(
                        (
                            evaluator._judge_out_dir(name)
                            / "safety_source_complete.json"
                        ).read_bytes()
                    ).hexdigest()
                    for name in judges
                },
                "verdicts_sha256": hashlib.sha256(
                    (root / "selected_verdicts.json").read_bytes()
                ).hexdigest(),
                "dataset_file_hashes": {
                    "ds.jsonl": hashlib.sha256(
                        (dataset / "ds.jsonl").read_bytes()
                    ).hexdigest()
                },
                "answer_file_hashes": {
                    "a": hashlib.sha256(
                        (Path(overrides["answers_dir"]) / "a.json").read_bytes()
                    ).hexdigest()
                },
                "source_asset_hashes": {
                    name: hashlib.sha256((dataset / name).read_bytes()).hexdigest()
                    for name in ("policy.txt", "main.txt")
                },
            }
            (root / "safety_unified_complete.json").write_text(json.dumps(complete))
        return True

    monkeypatch.setattr(evaluator, "_run_phase", run)
    return evaluator, judges, calls, run


def test_selection_finalizes_only_after_both_passes_and_reads_without_execution(
    selected_evaluator, monkeypatch
):
    evaluator, judges, calls, _ = selected_evaluator
    evaluator.run_on_model_name(judges[0])
    assert evaluator.get_metrics() == {}
    assert [phase for phase, _ in calls] == ["generation", "evaluation"]
    evaluator.run_on_model_name(judges[1])
    assert [phase for phase, _ in calls] == [
        "generation",
        "evaluation",
        "generation",
        "evaluation",
        "metrics",
    ]
    overrides = calls[-1][1]
    assert overrides["source_runs"] == [
        {
            "model": model,
            "judge_key": model,
            "root": str(evaluator._judge_out_dir(model)),
        }
        for model in judges
    ]
    assert overrides["selection_manifest"] == "selection.json"
    assert Path(overrides["out_dir"]) == evaluator._judge_out_dir("selected")
    monkeypatch.setattr(
        evaluator,
        "_run_phase",
        lambda *args: pytest.fail("get_metrics started execution"),
    )
    first = evaluator.get_metrics()
    assert first["safety_unified/selected/v2/score"] == 0.6
    assert all(key.startswith("safety_unified/selected/") for key in first)
    assert evaluator.get_metrics() == first
    assert "SELECTED" in evaluator._jd_metric_name("selected", "v2/score")


@pytest.mark.parametrize("failed_phase", ["evaluation", "metrics"])
def test_selection_failure_invalidates_old_final_result(
    selected_evaluator, monkeypatch, failed_phase
):
    evaluator, judges, calls, successful = selected_evaluator
    for model in judges:
        evaluator.run_on_model_name(model)
    assert evaluator.get_metrics()
    calls.clear()
    monkeypatch.setattr(
        evaluator,
        "_run_phase",
        lambda phase, overrides: (
            False if phase == failed_phase else successful(phase, overrides)
        ),
    )
    evaluator.run_on_model_name(judges[-1])
    assert evaluator.get_metrics() == {}
    root = evaluator._judge_out_dir("selected")
    assert not (root / "safety_unified_complete.json").exists()
    assert not (root / "safety_unified_metrics.json").exists()


@pytest.mark.parametrize(
    "changed",
    [
        "manifest",
        "dataset",
        "answer",
        "source_proof",
        "verdict",
        "completion",
        "raw",
        "source_complete",
        "prompt",
        "policy",
    ],
)
def test_selection_reader_rejects_changed_current_artifacts(
    selected_evaluator, changed
):
    evaluator, judges, _, _ = selected_evaluator
    for model in judges:
        evaluator.run_on_model_name(model)
    assert evaluator.get_metrics()
    config = evaluator.autometric_runs_config
    paths = {
        "manifest": Path(config.dataset_root) / "selection.json",
        "dataset": Path(config.dataset_root) / "ds.jsonl",
        "answer": Path(evaluator.metrics_dir) / "answers/a.json",
        "source_proof": evaluator._judge_out_dir(judges[0])
        / "safety_source_complete.json",
        "verdict": evaluator._judge_out_dir("selected") / "selected_verdicts.json",
        "completion": evaluator._judge_out_dir("selected")
        / "safety_unified_complete.json",
        "raw": evaluator._judge_out_dir(judges[0]) / "judgments/a.json",
        "source_complete": evaluator._judge_out_dir(judges[0])
        / "safety_unified_complete.json",
        "prompt": Path(config.dataset_root) / "main.txt",
        "policy": Path(config.dataset_root) / "policy.txt",
    }
    paths[changed].write_text(paths[changed].read_text() + " ")
    if changed == "completion":
        path = paths[changed]
        data = json.loads(path.read_text())
        data["source_proof_hashes"] = {}
        path.write_text(json.dumps(data))
    assert evaluator.get_metrics() == {}


def test_selection_config_rejects_different_dataset_scopes():
    with pytest.raises(pydantic.ValidationError, match="identical dataset files"):
        _cfg(
            selection_manifest="selection.json",
            judges={"a": ["a.jsonl"], "b": ["b.jsonl"]},
        )


def test_selected_metrics_publish_only_selected_jd_fields(
    selected_evaluator, jd_publication
):
    evaluator, judges, _, _ = selected_evaluator
    for model in judges:
        evaluator.run_on_model_name(model)
    evaluator._send_metrics_to_jd(evaluator.get_metrics())
    assert len(jd_publication["sent"]) == 1
    assert jd_publication["sent"][0]["jd_metrics"] == {
        evaluator._jd_metric_name("selected", "v2/score"): 0.6
    }


def test_selection_finalize_preparation_failure_does_not_escape(
    selected_evaluator, monkeypatch
):
    evaluator, judges, _, _ = selected_evaluator
    evaluator.run_on_model_name(judges[0])
    original = Path.mkdir

    def mkdir(path, *args, **kwargs):
        if path == evaluator._judge_out_dir("selected"):
            raise OSError("fixture selected output unavailable")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir)
    evaluator.run_on_model_name(judges[1])
    assert evaluator.get_metrics() == {}


def test_external_selection_maps_canonical_identity_to_owned_prefixed_roots(tmp_path):
    models = ["Qwen3-235B-A22B-Instruct-2507-FP8", "Qwen3.5-397B-A17B-FP8"]
    evaluator = _setup(
        tmp_path,
        _cfg(
            judges={"pam_smoke:" + name: "all" for name in models},
            selection_manifest="selection.json",
        ),
    )
    overrides = evaluator.autometric_runs_config.build_selection_overrides(
        metrics_dir=Path(evaluator.metrics_dir),
        checkpoint_model_name="candidate",
        checkpoint_path=Path("/weights/candidate"),
    )
    assert overrides["source_runs"] == [
        {
            "model": model,
            "judge_key": "pam_smoke:" + model,
            "root": str(evaluator._judge_out_dir("pam_smoke:" + model)),
        }
        for model in models
    ]


@pytest.mark.parametrize(
    "customization",
    [
        {},
        {"temperature": 1.0},
        {"chat_template_kwargs": {"enable_thinking": True}},
        {"chat_template_kwargs": {"enable_thinking": False}, "top_p": 0.2},
    ],
)
def test_external_selection_rejects_unreproducible_request_before_calls(
    tmp_path, monkeypatch, customization
):
    model = "pam_smoke:Qwen3-235B-A22B-Instruct-2507-FP8"
    evaluator = _setup(
        tmp_path, _cfg(judges={model: "all"}, selection_manifest="selection.json")
    )
    monkeypatch.setattr(
        InferenceManager,
        "_secondary_inference",
        {
            "pam_smoke": {
                "base_url": "http://actual-upstream:30000",
                "api_key": "fixture-local-only",
                "request_customization": customization,
            }
        },
        raising=False,
    )
    monkeypatch.setattr(
        evaluator,
        "_run_phase",
        lambda *args: pytest.fail("unverified source started model calls"),
    )
    evaluator.run_on_model_name(model)
    assert evaluator.get_metrics() == {}


def test_external_selection_uses_actual_upstream_and_preserves_mapper_protocol(
    tmp_path, monkeypatch
):
    model = "Qwen3-235B-A22B-Instruct-2507-FP8"
    judge = "pam_smoke:" + model
    evaluator = _setup(
        tmp_path,
        _cfg(
            judges={judge: "all"},
            selection_manifest="selection.json",
            judge={"seed": 12345, "temperature": 0.0, "max_tokens": 1536},
        ),
    )
    customization = {"chat_template_kwargs": {"enable_thinking": False}}
    monkeypatch.setattr(
        InferenceManager,
        "_secondary_inference",
        {
            "pam_smoke": {
                "base_url": "http://actual-upstream:30000/v1",
                "api_key": "fixture-local-only",
                "request_customization": customization,
            }
        },
        raising=False,
    )
    calls = []
    monkeypatch.setattr(
        evaluator,
        "_run_phase",
        lambda phase, overrides: calls.append((phase, overrides)) or False,
    )
    evaluator.run_on_model_name(judge)
    assert len(calls) == 1
    assert "selection_serving_api_base" not in calls[0][1]
    assert calls[0][1]["judge"]["api_base"] == "http://judge:30000/v1"
    assert "fixture-local-only" not in json.dumps(calls)
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "verdict",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"unsafe": {"type": "boolean"}},
                "required": ["unsafe"],
            },
        },
    }
    payload = {
        "model": judge,
        "messages": [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "complete history and answer"},
        ],
        "seed": 12345,
        "temperature": 0.0,
        "max_tokens": 1536,
        "response_format": schema,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    mapped = OpenAiMapper().map_request(
        SGLangChatCompletionRequest.model_validate(payload), model
    )
    assert "chat_template_kwargs" not in mapped
    mapped.update(customization)
    for field in (
        "messages",
        "seed",
        "temperature",
        "max_tokens",
        "response_format",
        "chat_template_kwargs",
    ):
        assert mapped[field] == payload[field]
    assert mapped["model"] == model


@pytest.mark.parametrize(
    "field,value",
    [
        ("runner_sha256", None),
        ("runner_sha256", "not-a-hash"),
        ("runner_sha256", "0" * 64),
        ("runner_version", "0.4.0"),
        ("format", "safety-source-pass-v1"),
    ],
)
def test_publication_rejects_changed_source_stamp_even_when_proof_hash_matches(
    selected_evaluator, field, value
):
    evaluator, judges, _, _ = selected_evaluator
    for judge in judges:
        evaluator.run_on_model_name(judge)
    assert evaluator.get_metrics()
    proof_path = evaluator._judge_out_dir(judges[0]) / "safety_source_complete.json"
    proof = json.loads(proof_path.read_text())
    proof[field] = value
    proof_path.write_text(json.dumps(proof))
    complete_path = (
        evaluator._judge_out_dir("selected") / "safety_unified_complete.json"
    )
    complete = json.loads(complete_path.read_text())
    complete["source_proof_hashes"][judges[0]] = hashlib.sha256(
        proof_path.read_bytes()
    ).hexdigest()
    complete_path.write_text(json.dumps(complete))
    assert evaluator.get_metrics() == {}


def test_publication_rejects_changed_packaged_runner(selected_evaluator):
    evaluator, judges, _, _ = selected_evaluator
    for judge in judges:
        evaluator.run_on_model_name(judge)
    assert evaluator.get_metrics()
    (evaluator._runner_root / "run_safety_unified.py").write_text(
        'RUNNER_VERSION = "0.7.0"\nCHANGED = True\n'
    )
    assert evaluator.get_metrics() == {}


def test_private_publication_preserves_local_metrics_without_any_jd_transfer(
    tmp_path, monkeypatch
):
    evaluator = _setup(tmp_path, _cfg(judges={"j": "all"}, publish_metrics=False))
    path = _metrics(evaluator)
    before = path.read_bytes()
    monkeypatch.setattr(JD, "internet_is_available", lambda: True)
    for name in (
        "send_evaluator_metrics",
        "send_metrics",
        "tar_metrics_dir",
        "symlink_metrics_for_transfer",
    ):
        monkeypatch.setattr(
            JD,
            name,
            lambda *args, **kwargs: pytest.fail(
                "private run attempted publication or transfer"
            ),
        )
    metrics = evaluator.get_metrics()
    assert metrics["safety_unified/judge_j/v2/score"] == 0.6
    evaluator.after_run(metrics)
    assert evaluator.get_metrics() == metrics
    assert path.read_bytes() == before
    assert "publish_metrics" not in evaluator._build_overrides("j")
    assert _cfg().publish_metrics is True


def test_private_external_pipeline_keeps_all_publication_boundaries_closed(
    tmp_path, monkeypatch
):
    evaluator = _setup(tmp_path, _cfg(judges={"j": "all"}, publish_metrics=False))
    path = _metrics(evaluator)
    before = path.read_bytes()
    manager = AutoMetricsManager()
    manager.checkpoint_path = evaluator.hf_checkpoint_path
    manager.run_name = "private"
    manager.tensorboard_path = None
    manager.hf_checkpoint_path = evaluator.hf_checkpoint_path
    manager.run_config = SimpleNamespace(
        autometrics=SimpleNamespace(
            inference=ExternalApiInferenceConfig(
                base_url="http://fixture.invalid", model="fixture", api_key="fixture"
            )
        )
    )

    def forbidden(*args, **kwargs):
        pytest.fail("private external pipeline attempted publication")

    monkeypatch.setattr(JD, "internet_is_available", lambda: True)
    for name in (
        "ensure_autometrics_record",
        "add_evaluator_metrics",
        "publish_metrics_cache_per_measurement",
        "send_evaluator_metrics",
        "send_metrics",
        "upload_artifacts",
        "tar_metrics_dir",
        "symlink_metrics_for_transfer",
    ):
        monkeypatch.setattr(JD, name, forbidden)
    monkeypatch.setattr(tensorboard_saver, "SummaryWriter", forbidden)
    metrics = evaluator.get_metrics()
    manager.save_evaluator_metrics("safety_unified", evaluator)
    assert manager.tensorboard_saver.metrics_tensorboard is None
    assert evaluator.get_metrics() == metrics
    assert path.read_bytes() == before


def test_private_flag_survives_full_config_snapshot_and_reload(tmp_path):
    judge = "Qwen3-235B-A22B-Instruct-2507-FP8"
    config = NodeRunnerConfigModel.model_validate(
        {
            "run_name": "private-persistence",
            "tensorboard_path": None,
            "max_seq_len": 4096,
            "autometrics": {
                "globals": {"dist_init_path": str(tmp_path / "unused-dist-init")},
                "router": {},
                "inference": {
                    "engine_type": "external_api",
                    "base_url": "http://fixture.invalid",
                    "model": "fixture",
                    "api_key": "fixture-workload-credential",
                },
                "obm_proxy": {},
                "evaluators": {
                    "safety_unified": {
                        "judges": {judge: "all"},
                        "publish_metrics": False,
                    }
                },
                "preparation": {},
                "convertation": {"mappers": ["map_fsdp_to_hf"]},
                "model_info": {
                    "model_class": "sft",
                    "size": "fixture",
                    "username": "fixture",
                },
            },
        }
    )
    assert config.autometrics.evaluators.safety_unified.publish_metrics is False
    manager = ConfigManager()
    manager.config = config
    manager._set_paths(tmp_path / "private-artifacts")
    snapshot_path = manager.artifacts_path / "pam_config.yaml"
    saved = OmegaConf.to_container(OmegaConf.load(snapshot_path), resolve=True)
    assert (
        saved["autometrics"]["evaluators"]["safety_unified"]["publish_metrics"] is False
    )
    assert "api-key" not in saved["autometrics"]["inference"]
    saved["autometrics"]["inference"]["api_key"] = "fixture-workload-credential"
    reload_path = tmp_path / "reload-with-workload-credential.yaml"
    OmegaConf.save(config=saved, f=reload_path)
    restored = NodeRunnerConfigModel.load(reload_path)
    assert restored.is_external_api_mode
    settings = restored.autometrics.evaluators.safety_unified
    assert settings.publish_metrics is False
    settings.judge_endpoints = judge_endpoints({judge: "http://judge.invalid"})
    settings.apply_defaults(
        inference_instances=1,
        base_url="http://candidate.invalid",
    )
    overrides = settings.build_overrides(
        judge_model_name=judge,
        out_dir=tmp_path / "metrics",
        default_answers_dir=tmp_path / "answers",
        checkpoint_model_name="fixture",
        checkpoint_path=tmp_path / "candidate",
    )
    assert "publish_metrics" not in overrides


def test_selected_external_config_missing_blocks_execution(tmp_path, monkeypatch):
    judge = "unconfigured:Qwen3-235B-A22B-Instruct-2507-FP8"
    evaluator = _setup(
        tmp_path, _cfg(judges={judge: "all"}, selection_manifest="selection.json")
    )
    monkeypatch.setattr(InferenceManager, "_secondary_inference", {}, raising=False)
    monkeypatch.setattr(
        evaluator,
        "_run_phase",
        lambda *args: pytest.fail("unconfigured external source started execution"),
    )
    evaluator.run_on_model_name(judge)
    assert evaluator.get_metrics() == {}


@pytest.mark.parametrize("kind", ["source_proof", "manifest", "policy", "prompt"])
def test_selected_reader_rejects_escaped_symlink_with_unchanged_bytes(
    selected_evaluator, tmp_path, monkeypatch, kind
):
    evaluator, judges, _, _ = selected_evaluator
    for judge in judges:
        evaluator.run_on_model_name(judge)
    assert evaluator.get_metrics()
    dataset = Path(evaluator.autometric_runs_config.dataset_root)
    paths = {
        "source_proof": evaluator._judge_out_dir(judges[0])
        / "safety_source_complete.json",
        "manifest": dataset / "selection.json",
        "policy": dataset / "policy.txt",
        "prompt": dataset / "main.txt",
    }
    path = paths[kind]
    before = path.read_bytes()
    outside = tmp_path / ("outside-" + kind)
    outside.write_bytes(before)
    path.unlink()
    path.symlink_to(outside)
    assert path.read_bytes() == before
    result = evaluator._judge_out_dir("selected") / "safety_unified_metrics.json"
    result_before = result.read_bytes()
    monkeypatch.setattr(
        evaluator,
        "_run_phase",
        lambda *args: pytest.fail("readonly metrics started execution"),
    )
    assert evaluator.get_metrics() == {}
    assert result.read_bytes() == result_before


@pytest.fixture
def learned_selected_evaluator(selected_evaluator):
    evaluator, judges, _, _ = selected_evaluator
    for judge in judges:
        evaluator.run_on_model_name(judge)
    package = Path(evaluator.autometric_runs_config.dataset_root)
    manifest_path = package / "selection.json"
    manifest = json.loads(manifest_path.read_text())
    order = manifest["source_order"]
    identities = {name: "d" * 64 for name in order}

    def digest(value):
        return hashlib.sha256(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()

    artifact = {
        "format": "safety_classifier_v1",
        "feature_schema": "verdict_flags_v1",
        "decoder": "observed_unsafe_first_source_v1",
        "source_order": order,
        "source_identities": identities,
        "pool_identity_sha256": digest([[name, identities[name]] for name in order]),
        "weights": [0.0] * (len(order) * 6),
        "bias": 0.0,
        "threshold": 0.5,
        "training_provenance": {
            name: "e" * 64
            for name in (
                "train_ids_sha256",
                "split_sha256",
                "gold_sha256",
                "fit_code_sha256",
            )
        },
    }
    artifact_path = package / "classifier.json"
    artifact_path.write_text(json.dumps(artifact))
    manifest["mode"] = "learned"
    manifest["classifier"] = {
        "path": "classifier.json",
        "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "artifact_sha256": digest(artifact),
    }
    manifest_path.write_text(json.dumps(manifest))
    complete_path = (
        evaluator._judge_out_dir("selected") / "safety_unified_complete.json"
    )
    complete = json.loads(complete_path.read_text())
    complete.update(
        mode="learned",
        selection_manifest_sha256=hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        classifier_artifact_sha256=digest(artifact),
    )
    complete_path.write_text(json.dumps(complete))
    return evaluator, artifact_path, manifest_path, complete_path


def test_learned_selected_reader_accepts_verified_artifact_without_execution(
    learned_selected_evaluator, monkeypatch
):
    evaluator, _, _, _ = learned_selected_evaluator
    monkeypatch.setattr(
        evaluator,
        "_run_phase",
        lambda *args: pytest.fail("learned metric reader started execution"),
    )
    metrics = evaluator.get_metrics()
    assert metrics["safety_unified/selected/v2/score"] == 0.6
    assert evaluator.get_metrics() == metrics


@pytest.mark.parametrize(
    "change",
    ["missing", "changed_bytes", "outside", "missing_reference", "wrong_content_hash"],
)
def test_learned_selected_reader_rejects_unverified_artifact(
    learned_selected_evaluator, tmp_path, change
):
    evaluator, artifact_path, manifest_path, complete_path = learned_selected_evaluator
    assert evaluator.get_metrics()
    if change == "missing":
        artifact_path.unlink()
    elif change == "changed_bytes":
        artifact_path.write_text(artifact_path.read_text() + " ")
    elif change == "outside":
        outside = tmp_path / "unpackaged-classifier.json"
        outside.write_bytes(artifact_path.read_bytes())
        artifact_path.unlink()
        artifact_path.symlink_to(outside)
    elif change == "missing_reference":
        manifest = json.loads(manifest_path.read_text())
        del manifest["classifier"]
        manifest_path.write_text(json.dumps(manifest))
        complete = json.loads(complete_path.read_text())
        complete["selection_manifest_sha256"] = hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
        complete_path.write_text(json.dumps(complete))
    elif change == "wrong_content_hash":
        complete = json.loads(complete_path.read_text())
        complete["classifier_artifact_sha256"] = "0" * 64
        complete_path.write_text(json.dumps(complete))
    assert evaluator.get_metrics() == {}


@pytest.mark.parametrize(
    "change", ["domain_rules", "classifier", "missing_domain_hash"]
)
def test_hierarchy_selected_reader_requires_packaged_domain_and_classifier(
    learned_selected_evaluator, monkeypatch, change
):
    evaluator, artifact_path, manifest_path, complete_path = learned_selected_evaluator
    package = manifest_path.parent
    domain_path = package / "domain_rules.json"
    domain_path.write_text(
        json.dumps(
            {
                "format": "safety-domain-rules-v1",
                "categories": {
                    "fixture": {"rules": "rule", "domain_patch": "", "checklist": ""}
                },
            }
        )
    )
    evaluator.autometric_runs_config.domain_rules_file = "domain_rules.json"
    order = [
        f"{model}:{role}"
        for model in ("B235", "B397")
        for role in ("dom", "warn", "cot", "str", "mid", "ban")
    ]
    artifact = {
        "format": "safety-hierarchy-v1",
        "source_order": order,
        "source_identities": {name: "d" * 64 for name in order},
        "heads": {
            name: {"weights": [0.0] * 48, "bias": 0.0} for name in ("A", "B", "C")
        },
        "training_provenance": json.loads(artifact_path.read_text())[
            "training_provenance"
        ],
    }
    artifact_path.write_text(json.dumps(artifact))
    artifact_hash = hashlib.sha256(
        json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        format="safety-production-selection-v3",
        mode="hierarchy",
        source_order=order,
        sources={name: {"prompt": "main.txt"} for name in order},
        classifier={
            "path": artifact_path.name,
            "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
            "artifact_sha256": artifact_hash,
        },
    )
    manifest_path.write_text(json.dumps(manifest))
    complete = json.loads(complete_path.read_text())
    complete.update(
        mode="hierarchy",
        source_order=order,
        selection_manifest_sha256=hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        classifier_artifact_sha256=artifact_hash,
    )
    complete["source_asset_hashes"]["domain_rules.json"] = hashlib.sha256(
        domain_path.read_bytes()
    ).hexdigest()
    complete_path.write_text(json.dumps(complete))
    monkeypatch.setattr(
        evaluator,
        "_run_phase",
        lambda *args: pytest.fail("publisher started execution"),
    )
    metrics = evaluator.get_metrics()
    assert metrics["safety_unified/selected/v2/score"] == 0.6
    assert evaluator.get_metrics() == metrics
    if change == "missing_domain_hash":
        del complete["source_asset_hashes"]["domain_rules.json"]
        complete_path.write_text(json.dumps(complete))
    else:
        target = domain_path if change == "domain_rules" else artifact_path
        target.write_text(target.read_text() + " ")
    assert evaluator.get_metrics() == {}
