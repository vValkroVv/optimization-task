import json
from pathlib import Path
from types import SimpleNamespace

import node_runner.src.metrics.evaluators.lmms_eval as lmms_eval_module
import pytest
from node_runner.src.job_dispatcher import JD
import yaml
from node_runner.src.config_model import (
    ExternalApiInferenceConfig,
    InferenceConfig,
    LmmsEvalBenchmarksConfig,
    LmmsEvalEvaluatorConfig,
    LmmsEvalRunConfig,
    SglangInferenceConfig,
    SglangPdInferenceConfig,
)
from node_runner.src.config_model import ModelInfoConfig as ModelInfo
from node_runner.src.metrics.evaluators.lmms_eval import LmmsEvalEvaluator

pytestmark = pytest.mark.unit

# ============================================================================
# Test constants
# ============================================================================

# Paths
ARTIFACTS_DIRNAME = "artifacts"
METRICS_DIRNAME = "metrics"
LMMS_EVAL_METRICS_SUBDIR = "lmms_eval"
LMMS_EVAL_CONFIGS_SUBDIR = "lmms_eval_configs"
HF_CHECKPOINT_DIRNAME = "hf_checkpoint"
CHECKPOINT_EP_BA_DIRNAME = "ep0_ba10"
CHECKPOINT_TMP_DIRNAME = "checkpoint"
METRICS_TAR_FILENAME = "metrics.tar"
AUTOMETRICS_LOG_BASENAME = "auto.log"
TMP_LOG_PATH = "/tmp/log"

# Runs and benchmarks
RUN_ALL = "all"
RUN_CHARTQA = "chartqa"
RUN_DEFAULT = "default"
RUN_MULTIPLE_TASKS = "multiple_tasks"
RUN_MIXED_TASKS_AND_GROUP = "mixed_tasks_and_group"
RUN_HALLUCINATION_GROUP = "hallucination_group"
RUN_BAD_NAME = "bad/run name"
RUN_BAD_SAFE_NAME = "bad_run_name"
RUN_BAD_SLASH = "bad/run"
RUN_BAD_SPACE = "bad run"
RUN_TASK = "task"
TASK_CHARTQA = "chartqa_giga"
TASK_GIGA_COUNTBENCH = "giga_countbench"
TASK_AI2D = "ai2d"
CATEGORY_ALL = "all"
CATEGORY_HALLUCINATION = "hallucination"

# Files
MODEL_RESULTS_RUN_ID = "20260101_000000"
TASK_RESULTS_JSON = f"{TASK_CHARTQA}_results.json"

# Model config
MODEL_CLASS_VISION = "vision"
MODEL_SIZE_LITE = "lite"
MODEL_USERNAME_TESTER = "tester"
MODEL_NAME_LITE = "model-lite"
MODEL_STEP = 10
CHATML_VERSION_V4 = "v4"
CHATML_VERSION_V41 = "v4.1"
DATASET_VALIDATION_MODE_NEVER = "never"
OBM_PROXY_API_PORT = 31234
OBM_PROXY_PARALLEL = 23
DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE = 50
INFERENCE_INSTANCES_SINGLE = 1
INFERENCE_INSTANCES_DEFAULT = 2
INFERENCE_INSTANCES_MULTI = 4
TASK_LIMIT = 10
SMOKE_LIMIT = 5
LOCAL_OBM_PROXY_BASE_URL = f"http://127.0.0.1:{OBM_PROXY_API_PORT}"
EXTERNAL_API_BASE_URL = "http://127.0.0.1:30000"
EXTERNAL_MODEL = "anthropic/claude-sonnet-4"
EXTERNAL_PARALLEL = 7
EXTERNAL_TIMEOUT = 7200.0

# Commands
LMMS_EVAL_VENV_PYTHON = "/home/user/lmms_eval/bin/python"
PYTHON_MODULE_FLAG = "-m"
GIGAVISION_CLI_MODULE = "gigavision_eval.cli.main"
GIGAVISION_EVAL_COMMAND = "eval"
COMMAND_CMD = "cmd"
COMMAND_PYTHON = "python"
COMMAND_MODULE = "module"
RUN_NAME_SUBPROCESS = "run-name"
CONFIG_FLAG = "--config"
OUTPUT_FLAG = "--output"

# Environment
ENV_GIGAVISION_MODEL_NAME = "GIGAVISION_MODEL__NAME"
ENV_GIGAVISION_BENCHMARK_TASKS = "GIGAVISION_EVALUATION__BENCHMARKS__TASKS"
ENV_GIGAVISION_JOB_RUN_ID = "GIGAVISION_JOB_RUN_ID"
ENV_GIGAVISION_JOB_MODE = "GIGAVISION_JOB_MODE"
ENV_OTHER = "OTHER_ENV"
ENV_GENERIC = "ENV"
ENV_MODEL_VALUE = "env-model"
ENV_TASKS_AI2D_VALUE = f'["{TASK_AI2D}"]'
ENV_STABLE_RUN_VALUE = "stable-run"
ENV_KEEP_VALUE = "keep"
ENV_ONE_VALUE = "1"
JOB_MODE_AUTO = "job_auto"

# Metrics
METRIC_GENERIC = "metric"
METRIC_COUNTBENCH_ACCURACY = "GIGA_COUNTBENCH_accuracy"
METRIC_COUNTBENCH_NOT_PARSED = "GIGA_COUNTBENCH_not_parsed"
METRIC_POPE_ACCURACY = "POPE_accuracy"
METRIC_POPE_F1 = "POPE_f1"

# Values
VALUE_0_0 = 0.0
VALUE_0_5 = 0.5
VALUE_0_7272727273 = 0.7272727273
VALUE_0_8 = 0.8
VALUE_1_0 = 1.0

MIXED_LMMS_EVAL_EXPORTED_METRICS = {
    METRIC_COUNTBENCH_ACCURACY: VALUE_0_5,
    METRIC_COUNTBENCH_NOT_PARSED: VALUE_0_0,
    METRIC_POPE_ACCURACY: VALUE_0_8,
    METRIC_POPE_F1: VALUE_0_7272727273,
}

NESTED_LMMS_EVAL_EXPORTED_METRICS = {
    METRIC_COUNTBENCH_ACCURACY: {
        "value": VALUE_0_5,
        "submetrics": {METRIC_COUNTBENCH_NOT_PARSED: VALUE_0_0},
    },
    METRIC_POPE_ACCURACY: {
        "value": VALUE_0_8,
        "submetrics": {METRIC_POPE_F1: VALUE_0_7272727273},
    },
}


def _model_info() -> ModelInfo:
    return ModelInfo(
        model_class=MODEL_CLASS_VISION,
        size=MODEL_SIZE_LITE,
        username=MODEL_USERNAME_TESTER,
    )


def _hf_checkpoint_path(tmp_path: Path) -> Path:
    path = tmp_path / MODEL_NAME_LITE / CHECKPOINT_EP_BA_DIRNAME / HF_CHECKPOINT_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _lmms_eval_metrics_dir(artifacts_path: Path) -> Path:
    return artifacts_path / METRICS_DIRNAME / LMMS_EVAL_METRICS_SUBDIR


def _run_config_path(metrics_dir: Path, run_name: str) -> Path:
    return metrics_dir / LMMS_EVAL_CONFIGS_SUBDIR / f"{run_name}.yaml"


def _config() -> LmmsEvalEvaluatorConfig:
    cfg = LmmsEvalEvaluatorConfig(
        chatml_version=CHATML_VERSION_V41,
        gigavision={
            "environment": {
                "dataset_validation": {"mode": DATASET_VALIDATION_MODE_NEVER},
            },
            "results": {
                "export": {
                    "jd": {
                        "enabled": True,
                        "include_all_metrics": True,
                    },
                },
            },
        },
        runs={
            RUN_CHARTQA: LmmsEvalRunConfig(
                benchmarks=LmmsEvalBenchmarksConfig(
                    tasks=[TASK_CHARTQA],
                    limit=TASK_LIMIT,
                ),
                gigavision={
                    "model": {
                        "api": {
                            "obm_proxy": {
                                "parallel": OBM_PROXY_PARALLEL,
                            },
                        },
                    },
                },
            ),
            RUN_ALL: LmmsEvalRunConfig(
                benchmarks=LmmsEvalBenchmarksConfig(categories=[CATEGORY_ALL]),
            ),
        },
    )
    _apply_local_defaults(cfg, inference_instances=INFERENCE_INSTANCES_DEFAULT)
    return cfg


def _apply_local_defaults(
    config: LmmsEvalEvaluatorConfig,
    *,
    inference_instances: int = INFERENCE_INSTANCES_SINGLE,
) -> None:
    config.apply_defaults(
        inference_instances=inference_instances,
        base_url=LOCAL_OBM_PROXY_BASE_URL,
        direct_base_url="http://127.0.0.1:30000",
        inference_config=SglangInferenceConfig(),
    )


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_export_metrics(command: list[str], metrics) -> None:
    output_path = Path(command[command.index(OUTPUT_FLAG) + 1])
    output_path.write_text(json.dumps(metrics), encoding="utf-8")


class TestLmmsEvalEvaluatorSetup:
    def test_setup_generates_gigavision_configs_and_commands(self, tmp_path: Path):
        artifacts_path = tmp_path / ARTIFACTS_DIRNAME
        metrics_dir = _lmms_eval_metrics_dir(artifacts_path)
        existing_result = (
            metrics_dir
            / RUN_CHARTQA
            / MODEL_NAME_LITE
            / MODEL_RESULTS_RUN_ID
            / TASK_RESULTS_JSON
        )
        existing_result.parent.mkdir(parents=True, exist_ok=True)
        existing_result.write_text("{}", encoding="utf-8")

        evaluator = LmmsEvalEvaluator()
        evaluator.setup(
            autometric_runs_config=_config(),
            artifacts_path=artifacts_path,
            hf_checkpoint_path=_hf_checkpoint_path(tmp_path),
            model_info=_model_info(),
        )

        assert evaluator.metrics_dir == metrics_dir
        assert set(evaluator.cmds_to_run) == {RUN_ALL}

        all_cmd = evaluator.cmds_to_run[RUN_ALL]
        assert all_cmd[:4] == [
            LMMS_EVAL_VENV_PYTHON,
            PYTHON_MODULE_FLAG,
            GIGAVISION_CLI_MODULE,
            GIGAVISION_EVAL_COMMAND,
        ]
        assert CONFIG_FLAG in all_cmd
        all_config_path = Path(all_cmd[all_cmd.index(CONFIG_FLAG) + 1])
        assert all_config_path == _run_config_path(metrics_dir, RUN_ALL)
        assert all_config_path.exists()

        chartqa_config = _read_yaml(_run_config_path(metrics_dir, RUN_CHARTQA))
        assert chartqa_config["model_source"] == "api"
        assert "compute" not in chartqa_config
        assert "execution_mode" not in chartqa_config
        assert chartqa_config["username"] == MODEL_USERNAME_TESTER
        assert chartqa_config["model"]["name"] == MODEL_NAME_LITE
        assert chartqa_config["model"]["step"] == MODEL_STEP
        assert chartqa_config["model"]["chatml_version"] == CHATML_VERSION_V41
        assert chartqa_config["model"]["api"]["shooter"] == "obm_proxy"
        assert (
            chartqa_config["model"]["api"]["obm_proxy"]["api"]["base_url"]
            == LOCAL_OBM_PROXY_BASE_URL
        )
        assert chartqa_config["model"]["api"]["obm_proxy"]["api"]["timeout"] == 21600
        assert (
            chartqa_config["model"]["api"]["obm_proxy"]["parallel"]
            == OBM_PROXY_PARALLEL
        )
        assert chartqa_config["evaluation"]["benchmarks"] == {
            "mode": "sequential",
            "tasks": [TASK_CHARTQA],
            "limit": TASK_LIMIT,
        }
        assert chartqa_config["evaluation"]["execution"]["cache"]["cache_dir"] == str(
            metrics_dir / "cache"
        )
        assert (
            chartqa_config["environment"]["dataset_validation"]["mode"]
            == DATASET_VALIDATION_MODE_NEVER
        )
        assert chartqa_config["results"]["output_dir"] == str(metrics_dir / RUN_CHARTQA)
        assert chartqa_config["results"]["export"]["jd"]["enabled"] is False
        assert chartqa_config["results"]["export"]["jd"]["include_all_metrics"] is True
        assert "api" not in chartqa_config["results"]["export"]
        assert "xlsx" not in chartqa_config["results"]["export"]
        assert "model_eval" not in chartqa_config["results"]["export"]

        all_config = _read_yaml(all_config_path)
        assert all_config["evaluation"]["benchmarks"] == {
            "mode": "sequential",
            "categories": [CATEGORY_ALL],
        }
        assert all_config["evaluation"]["execution"]["cache"]["cache_dir"] == str(
            metrics_dir / "cache"
        )
        assert (
            all_config["model"]["api"]["obm_proxy"]["parallel"]
            == INFERENCE_INSTANCES_DEFAULT * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
        )

    def test_setup_external_api_uses_native_lmms_eval_backend(self, tmp_path: Path):
        cfg = LmmsEvalEvaluatorConfig(
            benchmarks=LmmsEvalBenchmarksConfig(tasks=[TASK_AI2D], limit=SMOKE_LIMIT),
            timeout=EXTERNAL_TIMEOUT,
        )
        cfg.apply_defaults(
            inference_instances=INFERENCE_INSTANCES_SINGLE,
            base_url="http://127.0.0.1:31234",
            direct_base_url=EXTERNAL_API_BASE_URL,
            inference_config=ExternalApiInferenceConfig(
                formatting="openai",
                base_url="https://openrouter.ai/api/v1",
                model=EXTERNAL_MODEL,
                api_key="dummy-key",
                max_concurrency=EXTERNAL_PARALLEL,
            ),
        )
        artifacts_path = tmp_path / ARTIFACTS_DIRNAME
        evaluator = LmmsEvalEvaluator()

        evaluator.setup(
            autometric_runs_config=cfg,
            artifacts_path=artifacts_path,
            hf_checkpoint_path=_hf_checkpoint_path(tmp_path),
            model_info=_model_info(),
        )

        config_path = Path(evaluator.cmds_to_run[RUN_DEFAULT][-1])
        data = _read_yaml(config_path)
        assert data["model_source"] == "lmms_eval"
        assert "chatml_version" not in data["model"]
        assert "api" not in data["model"]
        assert data["model"]["lmms_eval"] == {
            "model": "async_openai",
            "model_args": (
                f"model={EXTERNAL_MODEL},"
                f"base_url={EXTERNAL_API_BASE_URL}/v1,"
                "api_key=unused,"
                f"num_cpus={EXTERNAL_PARALLEL},"
                f"timeout={EXTERNAL_TIMEOUT},"
                "max_retries=1,"
                "retry_backoff_s=0"
            ),
        }
        assert data["evaluation"]["benchmarks"] == {
            "mode": "sequential",
            "tasks": [TASK_AI2D],
            "limit": SMOKE_LIMIT,
        }

    @pytest.mark.parametrize("chatml_version", ["v4", "v4.1", "v5"])
    @pytest.mark.parametrize("disaggregated", [False, True])
    def test_setup_sglang_uses_obm_proxy_backend(
        self, tmp_path: Path, chatml_version: str, disaggregated: bool
    ):
        cfg = LmmsEvalEvaluatorConfig(
            chatml_version=chatml_version,
            benchmarks=LmmsEvalBenchmarksConfig(tasks=[TASK_AI2D]),
        )
        cfg.apply_defaults(
            inference_instances=INFERENCE_INSTANCES_MULTI,
            base_url=LOCAL_OBM_PROXY_BASE_URL,
            direct_base_url="http://127.0.0.1:30000",
            inference_config=SglangPdInferenceConfig(
                prefill=SglangInferenceConfig(),
                decode=SglangInferenceConfig(),
            )
            if disaggregated
            else SglangInferenceConfig(),
        )
        evaluator = LmmsEvalEvaluator()

        evaluator.setup(
            autometric_runs_config=cfg,
            artifacts_path=tmp_path / ARTIFACTS_DIRNAME,
            hf_checkpoint_path=_hf_checkpoint_path(tmp_path),
            model_info=_model_info(),
        )

        config_path = Path(evaluator.cmds_to_run[RUN_DEFAULT][-1])
        data = _read_yaml(config_path)
        assert data["model_source"] == "api"
        assert data["model"]["chatml_version"] == chatml_version
        assert data["model"]["api"]["shooter"] == "obm_proxy"
        assert data["model"]["api"]["obm_proxy"]["parallel"] == (
            INFERENCE_INSTANCES_MULTI * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
        )

    def test_setup_requires_runtime_target(self, tmp_path: Path):
        cfg = LmmsEvalEvaluatorConfig(chatml_version=CHATML_VERSION_V41)
        evaluator = LmmsEvalEvaluator()

        with pytest.raises(ValueError, match="runtime target must be set"):
            evaluator.setup(
                autometric_runs_config=cfg,
                artifacts_path=tmp_path / ARTIFACTS_DIRNAME,
                hf_checkpoint_path=_hf_checkpoint_path(tmp_path),
                model_info=_model_info(),
            )

    def test_local_runtime_target_requires_chatml_version(self):
        cfg = LmmsEvalEvaluatorConfig()

        with pytest.raises(ValueError, match="chatml_version is required"):
            _apply_local_defaults(cfg)

    def test_runtime_target_rejects_unknown_inference_type(self):
        cfg = LmmsEvalEvaluatorConfig()

        with pytest.raises(TypeError, match="InferenceConfig"):
            cfg.apply_defaults(
                inference_instances=INFERENCE_INSTANCES_SINGLE,
                base_url=LOCAL_OBM_PROXY_BASE_URL,
                direct_base_url=EXTERNAL_API_BASE_URL,
                inference_config=InferenceConfig(),
            )

    def test_setup_requires_checkpoint_path(self, tmp_path: Path):
        cfg = LmmsEvalEvaluatorConfig(chatml_version=CHATML_VERSION_V41)
        _apply_local_defaults(cfg)
        evaluator = LmmsEvalEvaluator()
        evaluator.hf_checkpoint_path = None
        evaluator.model_info = _model_info()

        with pytest.raises(ValueError, match="hf_checkpoint_path is not set"):
            evaluator._gigavision_eval_runs_to_bash_list(
                cfg,
                tmp_path / METRICS_DIRNAME,
            )

    def test_setup_requires_model_info(self, tmp_path: Path):
        cfg = LmmsEvalEvaluatorConfig(chatml_version=CHATML_VERSION_V41)
        _apply_local_defaults(cfg)
        evaluator = LmmsEvalEvaluator()
        evaluator.hf_checkpoint_path = _hf_checkpoint_path(tmp_path)
        evaluator.model_info = None

        with pytest.raises(ValueError, match="model_info is not set"):
            evaluator._gigavision_eval_runs_to_bash_list(
                cfg,
                tmp_path / METRICS_DIRNAME,
            )

    def test_default_run_uses_top_level_benchmarks(self, tmp_path: Path):
        cfg = LmmsEvalEvaluatorConfig(
            chatml_version=CHATML_VERSION_V41,
            benchmarks=LmmsEvalBenchmarksConfig(tasks=[TASK_AI2D]),
        )
        _apply_local_defaults(cfg)

        evaluator = LmmsEvalEvaluator()
        evaluator.setup(
            autometric_runs_config=cfg,
            artifacts_path=tmp_path / ARTIFACTS_DIRNAME,
            hf_checkpoint_path=_hf_checkpoint_path(tmp_path),
            model_info=_model_info(),
        )

        assert set(evaluator.cmds_to_run) == {RUN_DEFAULT}
        config_path = Path(evaluator.cmds_to_run[RUN_DEFAULT][-1])
        data = _read_yaml(config_path)
        assert data["evaluation"]["benchmarks"] == {
            "mode": "sequential",
            "tasks": [TASK_AI2D],
        }
        assert (
            data["model"]["api"]["obm_proxy"]["parallel"]
            == INFERENCE_INSTANCES_SINGLE * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
        )

    def test_setup_supports_multiple_tasks_and_category_runs(self, tmp_path: Path):
        cfg = LmmsEvalEvaluatorConfig(
            chatml_version=CHATML_VERSION_V4,
            runs={
                RUN_MULTIPLE_TASKS: LmmsEvalRunConfig(
                    benchmarks=LmmsEvalBenchmarksConfig(
                        tasks=[TASK_GIGA_COUNTBENCH, TASK_AI2D],
                        limit=SMOKE_LIMIT,
                    ),
                ),
                RUN_MIXED_TASKS_AND_GROUP: LmmsEvalRunConfig(
                    benchmarks=LmmsEvalBenchmarksConfig(
                        tasks=[TASK_GIGA_COUNTBENCH, TASK_AI2D],
                        categories=[CATEGORY_HALLUCINATION],
                        limit=SMOKE_LIMIT,
                    ),
                ),
                RUN_HALLUCINATION_GROUP: LmmsEvalRunConfig(
                    benchmarks=LmmsEvalBenchmarksConfig(
                        categories=[CATEGORY_HALLUCINATION],
                        limit=SMOKE_LIMIT,
                    ),
                ),
            },
        )
        _apply_local_defaults(
            cfg,
            inference_instances=INFERENCE_INSTANCES_MULTI,
        )

        evaluator = LmmsEvalEvaluator()
        evaluator.setup(
            autometric_runs_config=cfg,
            artifacts_path=tmp_path / ARTIFACTS_DIRNAME,
            hf_checkpoint_path=_hf_checkpoint_path(tmp_path),
            model_info=_model_info(),
        )

        multiple_tasks_path = Path(evaluator.cmds_to_run[RUN_MULTIPLE_TASKS][-1])
        mixed_tasks_path = Path(evaluator.cmds_to_run[RUN_MIXED_TASKS_AND_GROUP][-1])
        hallucination_path = Path(evaluator.cmds_to_run[RUN_HALLUCINATION_GROUP][-1])

        assert _read_yaml(multiple_tasks_path)["evaluation"]["benchmarks"] == {
            "mode": "sequential",
            "tasks": [TASK_GIGA_COUNTBENCH, TASK_AI2D],
            "limit": SMOKE_LIMIT,
        }
        assert _read_yaml(mixed_tasks_path)["evaluation"]["benchmarks"] == {
            "mode": "sequential",
            "tasks": [TASK_GIGA_COUNTBENCH, TASK_AI2D],
            "categories": [CATEGORY_HALLUCINATION],
            "limit": SMOKE_LIMIT,
        }
        assert _read_yaml(hallucination_path)["evaluation"]["benchmarks"] == {
            "mode": "sequential",
            "categories": [CATEGORY_HALLUCINATION],
            "limit": SMOKE_LIMIT,
        }

    def test_run_name_is_sanitized_for_paths(self, tmp_path: Path):
        cfg = LmmsEvalEvaluatorConfig(
            chatml_version=CHATML_VERSION_V41,
            runs={RUN_BAD_NAME: LmmsEvalRunConfig()},
        )
        _apply_local_defaults(cfg)

        evaluator = LmmsEvalEvaluator()
        evaluator.setup(
            autometric_runs_config=cfg,
            artifacts_path=tmp_path / ARTIFACTS_DIRNAME,
            hf_checkpoint_path=_hf_checkpoint_path(tmp_path),
            model_info=_model_info(),
        )

        assert set(evaluator.cmds_to_run) == {RUN_BAD_SAFE_NAME}
        assert (
            Path(evaluator.cmds_to_run[RUN_BAD_SAFE_NAME][-1]).name
            == f"{RUN_BAD_SAFE_NAME}.yaml"
        )

    def test_duplicate_safe_run_names_are_rejected(self, tmp_path: Path):
        cfg = LmmsEvalEvaluatorConfig(
            chatml_version=CHATML_VERSION_V41,
            runs={
                RUN_BAD_SLASH: LmmsEvalRunConfig(),
                RUN_BAD_SPACE: LmmsEvalRunConfig(),
            },
        )
        _apply_local_defaults(cfg)

        evaluator = LmmsEvalEvaluator()
        with pytest.raises(ValueError, match="resolve to the same output name"):
            evaluator.setup(
                autometric_runs_config=cfg,
                artifacts_path=tmp_path / ARTIFACTS_DIRNAME,
                hf_checkpoint_path=_hf_checkpoint_path(tmp_path),
                model_info=_model_info(),
            )


class TestLmmsEvalEnvironment:
    def test_subprocess_env_strips_gigavision_config_overrides(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv(ENV_GIGAVISION_MODEL_NAME, ENV_MODEL_VALUE)
        monkeypatch.setenv(ENV_GIGAVISION_BENCHMARK_TASKS, ENV_TASKS_AI2D_VALUE)
        monkeypatch.setenv(ENV_GIGAVISION_JOB_RUN_ID, ENV_STABLE_RUN_VALUE)
        monkeypatch.setenv(ENV_GIGAVISION_JOB_MODE, JOB_MODE_AUTO)
        monkeypatch.setenv(ENV_OTHER, ENV_KEEP_VALUE)

        env = LmmsEvalEvaluator._build_subprocess_env()

        assert ENV_GIGAVISION_MODEL_NAME not in env
        assert ENV_GIGAVISION_BENCHMARK_TASKS not in env
        assert env[ENV_GIGAVISION_JOB_RUN_ID] == ENV_STABLE_RUN_VALUE
        assert env[ENV_GIGAVISION_JOB_MODE] == JOB_MODE_AUTO
        assert env[ENV_OTHER] == ENV_KEEP_VALUE


class TestLmmsEvalRun:
    def test_after_run_sends_and_links_metrics(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = LmmsEvalEvaluator()
        evaluator.metrics_dir = tmp_path / METRICS_DIRNAME
        evaluator.artifacts_path = tmp_path / ARTIFACTS_DIRNAME
        evaluator.hf_checkpoint_path = tmp_path / CHECKPOINT_TMP_DIRNAME
        metrics_tar = tmp_path / METRICS_TAR_FILENAME
        calls = []

        monkeypatch.setattr(
            evaluator,
            "_send_metrics_to_jd",
            lambda metrics: calls.append(("send", metrics)),
        )
        monkeypatch.setattr(
            JD,
            "tar_metrics_dir",
            lambda metrics_dir, artifacts_path, _name: (
                calls.append(("tar", metrics_dir, artifacts_path)) or metrics_tar
            ),
        )
        monkeypatch.setattr(
            JD,
            "symlink_metrics_for_transfer",
            lambda tar_path, hf_checkpoint_path, _name: calls.append(
                ("symlink", tar_path, hf_checkpoint_path)
            ),
        )

        evaluator.after_run({METRIC_GENERIC: VALUE_1_0})

        assert calls == [
            ("send", {METRIC_GENERIC: VALUE_1_0}),
            (
                "tar",
                evaluator.metrics_dir,
                evaluator.artifacts_path,
            ),
            (
                "symlink",
                metrics_tar,
                evaluator.hf_checkpoint_path,
            ),
        ]

    def test_run_uses_current_no_arg_contract(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = LmmsEvalEvaluator()
        evaluator.metrics_dir = tmp_path
        evaluator.cmds_to_run = {RUN_TASK: [COMMAND_CMD]}
        commands = []

        monkeypatch.setattr(
            evaluator,
            "_run_command",
            lambda command, run_name: commands.append((command, run_name)),
        )

        evaluator.run()

        assert commands == [([COMMAND_CMD], RUN_TASK)]

    def test_run_executes_commands_for_master(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = LmmsEvalEvaluator()
        evaluator.metrics_dir = tmp_path
        evaluator.cmds_to_run = {RUN_TASK: [COMMAND_CMD]}
        evaluator.run_env = {ENV_GENERIC: ENV_ONE_VALUE}
        commands = []

        def fake_run_command(command: list[str], run_name: str) -> None:
            commands.append((command, run_name))

        monkeypatch.setattr(evaluator, "_run_command", fake_run_command)
        evaluator.run()

        assert commands == [([COMMAND_CMD], RUN_TASK)]

    def test_run_command_uses_subprocess_manager(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        events = []

        class FakeSubprocessManager:
            def __init__(
                self,
                command: list[str],
                log_file: Path,
                stdout_file,
                name: str,
                env: dict[str, str],
            ) -> None:
                events.append(("init", command, log_file, stdout_file, name, env))

            def start(self) -> None:
                events.append(("start",))

            def wait_for_completion(self) -> None:
                events.append(("wait",))

            def raise_for_exit_code(self) -> None:
                events.append(("raise",))

        monkeypatch.setattr(
            lmms_eval_module, "SubprocessManager", FakeSubprocessManager
        )
        evaluator = LmmsEvalEvaluator()
        evaluator.autometrics_log_file = tmp_path / AUTOMETRICS_LOG_BASENAME
        evaluator.run_env = {ENV_GENERIC: ENV_ONE_VALUE}

        evaluator._run_command(
            [COMMAND_PYTHON, PYTHON_MODULE_FLAG, COMMAND_MODULE],
            RUN_NAME_SUBPROCESS,
        )

        assert events == [
            (
                "init",
                [COMMAND_PYTHON, PYTHON_MODULE_FLAG, COMMAND_MODULE],
                tmp_path / AUTOMETRICS_LOG_BASENAME,
                None,
                RUN_NAME_SUBPROCESS,
                {ENV_GENERIC: ENV_ONE_VALUE},
            ),
            ("start",),
            ("wait",),
            ("raise",),
        ]


class TestLmmsEvalMetrics:
    def test_get_metrics_exports_each_run_through_gigavision_cli(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = LmmsEvalEvaluator()
        evaluator.metrics_dir = tmp_path / "metrics" / "lmms_eval"
        evaluator.metrics_dir.mkdir(parents=True)
        nested_results_dir = (
            evaluator.metrics_dir
            / RUN_CHARTQA
            / MODEL_NAME_LITE
            / "20260522_071103_abc"
        )
        nested_results_dir.mkdir(parents=True)
        (nested_results_dir / "20260522_151119_results.json").write_text(
            "{}", encoding="utf-8"
        )
        evaluator.lmms_eval_venv_python = "/home/user/lmms_eval/bin/python"
        evaluator.run_exports = {
            "chartqa": {
                "config_path": evaluator.metrics_dir
                / "lmms_eval_configs"
                / "chartqa.yaml",
                "output_dir": evaluator.metrics_dir / "chartqa",
                "metrics_path": evaluator.metrics_dir / "chartqa_metrics.json",
            }
        }
        commands = []
        warnings = []
        monkeypatch.setattr(lmms_eval_module.Logger, "warning", warnings.append)

        def fake_run_command(command: list[str], run_name: str) -> None:
            commands.append((command, run_name))
            evaluator.run_exports["chartqa"]["metrics_path"].write_text(
                json.dumps(
                    {
                        "CHARTQA_GIGA_relaxed_overall": 0.75,
                        "CHARTQA_GIGA_non_cyclic": 1.0,
                    }
                ),
                encoding="utf-8",
            )

        monkeypatch.setattr(evaluator, "_run_command", fake_run_command)

        assert evaluator.get_metrics() == {
            "CHARTQA_GIGA_relaxed_overall": 0.75,
            "CHARTQA_GIGA_non_cyclic": 1.0,
        }
        assert commands == [
            (
                [
                    "/home/user/lmms_eval/bin/python",
                    "-m",
                    "gigavision_eval.cli.main",
                    "jd",
                    "metrics",
                    "--config",
                    str(evaluator.run_exports["chartqa"]["config_path"]),
                    "--results-dir",
                    str(nested_results_dir),
                    "--output",
                    str(evaluator.run_exports["chartqa"]["metrics_path"]),
                ],
                "Export lmms-eval metrics chartqa",
            )
        ]
        assert "--include-all-metrics" not in commands[0][0]
        assert warnings == []

    @pytest.mark.parametrize(
        "exported_metrics",
        [MIXED_LMMS_EVAL_EXPORTED_METRICS, NESTED_LMMS_EVAL_EXPORTED_METRICS],
        ids=["flat", "nested"],
    )
    def test_get_metrics_uses_exported_metrics_contract(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, exported_metrics: dict
    ):
        evaluator = LmmsEvalEvaluator()
        evaluator.metrics_dir = tmp_path / "metrics" / "lmms_eval"
        evaluator.metrics_dir.mkdir(parents=True)
        evaluator.lmms_eval_venv_python = "/home/user/lmms_eval/bin/python"
        output_dir = evaluator.metrics_dir / "mixed"
        evaluator.run_exports = {
            "mixed": {
                "config_path": evaluator.metrics_dir
                / "lmms_eval_configs"
                / "mixed.yaml",
                "output_dir": output_dir,
                "metrics_path": evaluator.metrics_dir / "mixed_metrics.json",
            }
        }

        def fake_run_command(command: list[str], run_name: str) -> None:
            assert command[:5] == [
                "/home/user/lmms_eval/bin/python",
                "-m",
                "gigavision_eval.cli.main",
                "jd",
                "metrics",
            ]
            assert command[command.index("--results-dir") + 1] == str(output_dir)
            assert run_name == "Export lmms-eval metrics mixed"
            _write_export_metrics(command, exported_metrics)

        monkeypatch.setattr(evaluator, "_run_command", fake_run_command)

        assert evaluator.get_metrics() == MIXED_LMMS_EVAL_EXPORTED_METRICS
        assert evaluator._jd_metrics == exported_metrics

    def test_get_metrics_exports_already_calculated_runs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        cfg = _config()
        artifacts_path = tmp_path / "artifacts"
        metrics_dir = artifacts_path / "metrics" / "lmms_eval"
        existing_result = metrics_dir / "chartqa" / "results.json"
        existing_result.parent.mkdir(parents=True, exist_ok=True)
        existing_result.write_text("{}", encoding="utf-8")
        evaluator = LmmsEvalEvaluator()
        evaluator.setup(
            autometric_runs_config=cfg,
            artifacts_path=artifacts_path,
            hf_checkpoint_path=_hf_checkpoint_path(tmp_path),
            model_info=_model_info(),
        )
        commands = []

        def fake_run_command(command: list[str], run_name: str) -> None:
            commands.append((command, run_name))
            _write_export_metrics(command, {"chartqa_metric": 0.5})

        monkeypatch.setattr(evaluator, "_run_command", fake_run_command)

        assert "chartqa" not in evaluator.cmds_to_run
        assert evaluator.get_metrics()["chartqa_metric"] == 0.5
        assert commands[0][1] == "Export lmms-eval metrics chartqa"

    def test_get_metrics_uses_prepared_export_not_raw_results(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = LmmsEvalEvaluator()
        evaluator.metrics_dir = tmp_path / "metrics" / "lmms_eval"
        raw_results = evaluator.metrics_dir / "chartqa" / "results.json"
        raw_results.parent.mkdir(parents=True)
        raw_results.write_text(
            json.dumps({"results": {"chartqa_giga": {"relaxed_overall": 99.0}}}),
            encoding="utf-8",
        )
        evaluator.lmms_eval_venv_python = "/home/user/lmms_eval/bin/python"
        evaluator.run_exports = {
            "chartqa": {
                "config_path": evaluator.metrics_dir
                / "lmms_eval_configs"
                / "chartqa.yaml",
                "output_dir": evaluator.metrics_dir / "chartqa",
                "metrics_path": evaluator.metrics_dir / "chartqa_metrics.json",
            }
        }

        def fake_run_command(command: list[str], run_name: str) -> None:
            _write_export_metrics(command, {"CHARTQA_GIGA_relaxed_overall": 0.75})

        monkeypatch.setattr(evaluator, "_run_command", fake_run_command)

        assert evaluator.get_metrics() == {"CHARTQA_GIGA_relaxed_overall": 0.75}

    def test_get_metrics_overwrites_duplicate_metric_keys(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = LmmsEvalEvaluator()
        evaluator.metrics_dir = tmp_path / "metrics" / "lmms_eval"
        evaluator.metrics_dir.mkdir(parents=True)
        evaluator.lmms_eval_venv_python = "/home/user/lmms_eval/bin/python"
        evaluator.run_exports = {
            "first": {
                "config_path": evaluator.metrics_dir
                / "lmms_eval_configs"
                / "first.yaml",
                "output_dir": evaluator.metrics_dir / "first",
                "metrics_path": evaluator.metrics_dir / "first_metrics.json",
            },
            "second": {
                "config_path": evaluator.metrics_dir
                / "lmms_eval_configs"
                / "second.yaml",
                "output_dir": evaluator.metrics_dir / "second",
                "metrics_path": evaluator.metrics_dir / "second_metrics.json",
            },
        }
        warnings = []
        monkeypatch.setattr(lmms_eval_module.Logger, "warning", warnings.append)

        def fake_run_command(command: list[str], run_name: str) -> None:
            output_path = Path(command[command.index("--output") + 1])
            _write_export_metrics(
                command,
                {"DUP_metric": 0.1 if output_path.name.startswith("first") else 0.2},
            )

        monkeypatch.setattr(evaluator, "_run_command", fake_run_command)

        assert evaluator.get_metrics() == {"DUP_metric": 0.2}
        assert warnings == [
            "Duplicate lmms-eval metric DUP_metric; overwriting previous entry with the result from second"
        ]

    def test_get_metrics_aggregates_disjoint_runs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = LmmsEvalEvaluator()
        evaluator.metrics_dir = tmp_path / "metrics" / "lmms_eval"
        evaluator.metrics_dir.mkdir(parents=True)
        evaluator.lmms_eval_venv_python = "/home/user/lmms_eval/bin/python"
        evaluator.run_exports = {
            "countbench": {
                "config_path": evaluator.metrics_dir
                / "lmms_eval_configs"
                / "countbench.yaml",
                "output_dir": evaluator.metrics_dir / "countbench",
                "metrics_path": evaluator.metrics_dir / "countbench_metrics.json",
            },
            "hallucination": {
                "config_path": evaluator.metrics_dir
                / "lmms_eval_configs"
                / "hallucination.yaml",
                "output_dir": evaluator.metrics_dir / "hallucination",
                "metrics_path": evaluator.metrics_dir / "hallucination_metrics.json",
            },
        }

        def fake_run_command(command: list[str], run_name: str) -> None:
            output_path = Path(command[command.index("--output") + 1])
            metric_prefix = output_path.name.removesuffix("_metrics.json")
            _write_export_metrics(command, {f"{metric_prefix}_metric": 0.5})

        monkeypatch.setattr(evaluator, "_run_command", fake_run_command)

        assert evaluator.get_metrics() == {
            "countbench_metric": 0.5,
            "hallucination_metric": 0.5,
        }

    def test_get_metrics_without_metrics_dir(self, monkeypatch: pytest.MonkeyPatch):
        warnings = []
        monkeypatch.setattr(lmms_eval_module.Logger, "warning", warnings.append)
        evaluator = LmmsEvalEvaluator()
        assert evaluator.get_metrics() == {}
        assert warnings == ["metrics_dir is not set for LmmsEvalEvaluator"]

    def test_get_metrics_continues_after_export_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = LmmsEvalEvaluator()
        evaluator.metrics_dir = tmp_path / "metrics" / "lmms_eval"
        evaluator.metrics_dir.mkdir(parents=True)
        evaluator.lmms_eval_venv_python = "/home/user/lmms_eval/bin/python"
        evaluator.run_exports = {
            "chartqa": {
                "config_path": evaluator.metrics_dir
                / "lmms_eval_configs"
                / "chartqa.yaml",
                "output_dir": evaluator.metrics_dir / "chartqa",
                "metrics_path": evaluator.metrics_dir / "chartqa_metrics.json",
            },
            "ai2d": {
                "config_path": evaluator.metrics_dir
                / "lmms_eval_configs"
                / "ai2d.yaml",
                "output_dir": evaluator.metrics_dir / "ai2d",
                "metrics_path": evaluator.metrics_dir / "ai2d_metrics.json",
            },
        }

        def fail_export(command: list[str], run_name: str) -> None:
            if run_name.endswith("chartqa"):
                raise lmms_eval_module.SubprocessManagerError("export failed")
            _write_export_metrics(command, {"AI2D_acc": 0.8})

        errors = []
        monkeypatch.setattr(lmms_eval_module.Logger, "error", errors.append)
        monkeypatch.setattr(evaluator, "_run_command", fail_export)

        assert evaluator.get_metrics() == {"AI2D_acc": 0.8}
        assert errors == [
            "Failed to export lmms-eval metrics for chartqa: export failed"
        ]

    def test_get_metrics_logs_missing_exported_metrics_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = LmmsEvalEvaluator()
        evaluator.metrics_dir = tmp_path / "metrics" / "lmms_eval"
        evaluator.metrics_dir.mkdir(parents=True)
        evaluator.lmms_eval_venv_python = "/home/user/lmms_eval/bin/python"
        evaluator.run_exports = {
            "chartqa": {
                "config_path": evaluator.metrics_dir
                / "lmms_eval_configs"
                / "chartqa.yaml",
                "output_dir": evaluator.metrics_dir / "chartqa",
                "metrics_path": evaluator.metrics_dir / "chartqa_metrics.json",
            }
        }
        monkeypatch.setattr(evaluator, "_run_command", lambda *_: None)

        errors = []
        monkeypatch.setattr(lmms_eval_module.Logger, "error", errors.append)
        assert evaluator.get_metrics() == {}
        assert len(errors) == 1
        assert str(evaluator.run_exports["chartqa"]["metrics_path"]) in errors[0]

    def test_get_metrics_logs_malformed_exported_metrics_json(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = LmmsEvalEvaluator()
        evaluator.metrics_dir = tmp_path / "metrics" / "lmms_eval"
        evaluator.metrics_dir.mkdir(parents=True)
        evaluator.lmms_eval_venv_python = "/home/user/lmms_eval/bin/python"
        evaluator.run_exports = {
            "chartqa": {
                "config_path": evaluator.metrics_dir
                / "lmms_eval_configs"
                / "chartqa.yaml",
                "output_dir": evaluator.metrics_dir / "chartqa",
                "metrics_path": evaluator.metrics_dir / "chartqa_metrics.json",
            }
        }

        def fake_run_command(command: list[str], run_name: str) -> None:
            Path(command[command.index("--output") + 1]).write_text(
                "{", encoding="utf-8"
            )

        monkeypatch.setattr(evaluator, "_run_command", fake_run_command)

        errors = []
        monkeypatch.setattr(lmms_eval_module.Logger, "error", errors.append)
        assert evaluator.get_metrics() == {}
        assert len(errors) == 1
        assert str(evaluator.run_exports["chartqa"]["metrics_path"]) in errors[0]

    @pytest.mark.parametrize(
        "raw_metrics",
        [
            [],
            {},
            {"AI2D_acc": 0.8, "skip": "not numeric"},
            {"AI2D_acc": True},
            {"AI2D_acc": None},
            {"AI2D_acc": {"value": 0.8}},
            {"AI2D_acc": {"value": 0.8, "submetrics": {"secondary": "bad"}}},
        ],
    )
    def test_get_metrics_rejects_invalid_or_empty_exported_metrics(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        raw_metrics,
    ):
        evaluator = LmmsEvalEvaluator()
        evaluator.metrics_dir = tmp_path / "metrics" / "lmms_eval"
        evaluator.metrics_dir.mkdir(parents=True)
        evaluator.lmms_eval_venv_python = "/home/user/lmms_eval/bin/python"
        evaluator.run_exports = {
            "chartqa": {
                "config_path": evaluator.metrics_dir
                / "lmms_eval_configs"
                / "chartqa.yaml",
                "output_dir": evaluator.metrics_dir / "chartqa",
                "metrics_path": evaluator.metrics_dir / "chartqa_metrics.json",
            }
        }

        def fake_run_command(command: list[str], run_name: str) -> None:
            _write_export_metrics(command, raw_metrics)

        monkeypatch.setattr(evaluator, "_run_command", fake_run_command)

        errors = []
        monkeypatch.setattr(lmms_eval_module.Logger, "error", errors.append)
        assert evaluator.get_metrics() == {}
        assert len(errors) == 1
        assert errors[0].startswith(
            "Failed to read exported lmms-eval metrics"
            if raw_metrics != {}
            else "No JD metrics exported for lmms-eval run"
        )


class TestLmmsEvalAfterRun:
    def _prepared_evaluator(self, tmp_path: Path) -> tuple[LmmsEvalEvaluator, Path]:
        evaluator = LmmsEvalEvaluator()
        evaluator.hf_checkpoint_path = _hf_checkpoint_path(tmp_path)
        evaluator.metrics_dir = tmp_path / "metrics"
        evaluator.metrics_dir.mkdir()
        evaluator.model_info = _model_info()
        evaluator.lmms_eval_version_file = str(tmp_path / "version.txt")
        Path(evaluator.lmms_eval_version_file).write_text("0.5.0", encoding="utf-8")
        evaluator.lmms_eval_venv_python = "/home/user/lmms_eval/bin/python"
        evaluator.run_env = {"ENV": "1"}
        evaluator._jd_metrics = {"AI2D_acc": 0.8}
        config_path = evaluator.metrics_dir / "lmms_eval_configs" / "chartqa.yaml"
        evaluator.run_exports = {
            "chartqa": {
                "config_path": config_path,
                "output_dir": evaluator.metrics_dir / "chartqa",
                "metrics_path": evaluator.metrics_dir / "chartqa_metrics.json",
            }
        }
        return evaluator, config_path

    @pytest.mark.parametrize(
        ("field_name", "warning"),
        [
            (
                "hf_checkpoint_path",
                "hf_checkpoint_path is not set, skipping JD metrics send",
            ),
            ("metrics_dir", "metrics_dir is not set, skipping JD metrics send"),
            ("model_info", "model_info is not set, skipping JD metrics send"),
        ],
    )
    def test_send_metrics_to_jd_skips_when_context_is_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        field_name: str,
        warning: str,
    ):
        evaluator, _ = self._prepared_evaluator(tmp_path)
        warnings = []
        setattr(evaluator, field_name, None)
        monkeypatch.setattr(lmms_eval_module.Logger, "warning", warnings.append)

        evaluator._send_metrics_to_jd({"AI2D_acc": 0.8})

        assert warnings == [warning]

    def test_send_metrics_to_jd_uses_pam_transport_and_syncs_mapping(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator, config_path = self._prepared_evaluator(tmp_path)
        evaluator._jd_metrics["GIGA_COUNTBENCH_not_parsed"] = 0.0
        monkeypatch.setattr(
            lmms_eval_module, "uuid4", lambda: SimpleNamespace(hex="test-publication")
        )
        monkeypatch.setenv("AUTOCOMPARE_ID", "compare-1")
        commands = []
        captured = []
        cache_s3_path = (
            "gigaeye-s3:gigaeye-pretrain-checkpoints/"
            "pretrain-auto-metrics-artifacts/model-lite/step_10/"
            "artifacts/metrics/lmms_eval/0.5.0/cache_scores.test-publication.jsonl.zst"
        )

        def fake_run_command(command: list[str], run_name: str) -> None:
            commands.append((command, run_name))
            if command[3:6] == ["jd", "cache", "upload"]:
                shooter_version = command[command.index("--shooter-version") + 1]
                Path(command[command.index(OUTPUT_FLAG) + 1]).write_text(
                    json.dumps(
                        {
                            "action": "publish",
                            "cache_s3_path": cache_s3_path.replace(
                                "/0.5.0/", f"/{shooter_version}/"
                            ),
                            "error": None,
                        }
                    ),
                    encoding="utf-8",
                )

        monkeypatch.setattr(evaluator, "_run_command", fake_run_command)

        def fake_send_metrics(body: dict, _artifacts: Path) -> bool:
            captured.append(body)
            return True

        monkeypatch.setattr(
            JD,
            "send_metrics",
            fake_send_metrics,
        )

        evaluator._send_metrics_to_jd(
            {"AI2D_acc": 0.8, "GIGA_COUNTBENCH_not_parsed": 0.0}
        )

        assert commands[:2] == [
            (
                [
                    "/home/user/lmms_eval/bin/python",
                    "-m",
                    "gigavision_eval.cli.main",
                    "jd",
                    "cache",
                    "upload",
                    "--config",
                    str(config_path),
                    "--results-dir",
                    str(evaluator.metrics_dir / "chartqa"),
                    "--model-name",
                    "model-lite",
                    "--step",
                    "10",
                    "--shooter-version",
                    "0.5.0",
                    "--publication-id",
                    "test-publication",
                    "--output",
                    str(evaluator.metrics_dir / "chartqa_cache_delivery.json"),
                ],
                "Upload lmms-eval JD cache chartqa",
            ),
            (
                [
                    "/home/user/lmms_eval/bin/python",
                    "-m",
                    "gigavision_eval.cli.main",
                    "jd",
                    "mapping",
                    "sync",
                    "--config",
                    str(config_path),
                ],
                "Sync lmms-eval JD metric mapping",
            ),
        ]
        latest_merged_command = commands[0][0].copy()
        latest_merged_command[latest_merged_command.index("--shooter-version") + 1] = (
            "latest-merged"
        )
        latest_merged_command[latest_merged_command.index(OUTPUT_FLAG) + 1] = str(
            evaluator.metrics_dir / "chartqa_latest_merged_cache_delivery.json"
        )
        assert commands[2][0][3:6] == ["jd", "cache", "commit"]
        assert commands[2][0][-2:] == ["--cache-s3-path", cache_s3_path]
        assert commands[3] == (
            latest_merged_command,
            "Upload lmms-eval JD cache chartqa",
        )
        assert commands[4][0][3:6] == ["jd", "cache", "commit"]
        assert commands[4][0][-2:] == [
            "--cache-s3-path",
            cache_s3_path.replace("/0.5.0/", "/latest-merged/"),
        ]
        body, latest_merged_body = captured
        assert body["name"] == "model-lite"
        assert body["metrics"] == {
            10: {"AI2D_acc": 0.8, "GIGA_COUNTBENCH_not_parsed": 0.0}
        }
        assert body["shooter_name"] == "lmms_eval"
        assert body["shooter_version"] == "0.5.0"
        assert body["model_class"] == "vision"
        assert body["size"] == "lite"
        assert body["username"] == "tester"
        assert body["cache_s3_path"] == cache_s3_path
        assert body["autocompare_id"] == "compare-1"
        assert latest_merged_body == {
            **body,
            "shooter_version": "latest-merged",
            "cache_s3_path": cache_s3_path.replace("/0.5.0/", "/latest-merged/"),
        }
        assert json.loads(
            (evaluator.metrics_dir / "metrics_step_10.json").read_text(encoding="utf-8")
        ) == {
            **body,
            "metrics": {"10": {"AI2D_acc": 0.8, "GIGA_COUNTBENCH_not_parsed": 0.0}},
        }

    def test_commit_jd_cache_skips_when_run_config_is_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator, _ = self._prepared_evaluator(tmp_path)
        evaluator.run_exports = {}
        errors = []
        monkeypatch.setattr(lmms_eval_module.Logger, "error", errors.append)

        assert not evaluator._commit_jd_cache(
            model_name="model-lite",
            step=10,
            shooter_version="0.5.0",
            cache_s3_path="gigaeye-s3:bucket/cache_scores.jsonl.zst",
        )
        assert errors == ["Cannot mark JD cache as published: no lmms-eval config"]

    def test_commit_jd_cache_returns_false_when_marker_command_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator, config_path = self._prepared_evaluator(tmp_path)
        commands = []
        errors = []

        def fail_command(command: list[str], run_name: str) -> None:
            commands.append((command, run_name))
            raise lmms_eval_module.SubprocessManagerError("marker failed")

        monkeypatch.setattr(evaluator, "_run_command", fail_command)
        monkeypatch.setattr(lmms_eval_module.Logger, "error", errors.append)

        assert not evaluator._commit_jd_cache(
            model_name="model-lite",
            step=10,
            shooter_version="0.5.0",
            cache_s3_path="gigaeye-s3:bucket/cache_scores.jsonl.zst",
        )
        assert commands[0][0][3:6] == ["jd", "cache", "commit"]
        assert commands[0][0][commands[0][0].index(CONFIG_FLAG) + 1] == str(config_path)
        assert errors == [
            "JD accepted metrics but cache publication marker failed; "
            "future partial runs need manual recovery: marker failed"
        ]

    def test_send_metrics_to_jd_skips_latest_merged_after_versioned_send_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator, _ = self._prepared_evaluator(tmp_path)
        cache_versions = []

        def fake_upload(*, model_name: str, step: int, shooter_version: str) -> str:
            cache_versions.append(shooter_version)
            return "gigaeye-s3:bucket/cache_scores.jsonl.zst"

        monkeypatch.setattr(evaluator, "_upload_jd_cache", fake_upload)
        monkeypatch.setattr(
            evaluator,
            "_commit_jd_cache",
            lambda **kwargs: pytest.fail("failed JD send must not commit a cache"),
        )
        monkeypatch.setattr(evaluator, "_sync_jd_dataset_mapping", lambda _: None)
        monkeypatch.setattr(JD, "send_metrics", lambda *_: False)

        evaluator._send_metrics_to_jd({"AI2D_acc": 0.8})

        assert cache_versions == ["0.5.0"]

    def test_send_metrics_to_jd_does_not_commit_latest_merged_after_send_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator, _ = self._prepared_evaluator(tmp_path)
        cache_versions = []
        sent_versions = []
        committed_versions = []

        def fake_run_command(command: list[str], _run_name: str) -> None:
            if command[3:6] == ["jd", "cache", "upload"]:
                shooter_version = command[command.index("--shooter-version") + 1]
                cache_versions.append(shooter_version)
                Path(command[command.index(OUTPUT_FLAG) + 1]).write_text(
                    json.dumps(
                        {
                            "action": "publish",
                            "cache_s3_path": f"gigaeye-s3:bucket/{shooter_version}/cache_scores.jsonl.zst",
                            "error": None,
                        }
                    ),
                    encoding="utf-8",
                )
            elif command[3:6] == ["jd", "cache", "commit"]:
                committed_versions.append(
                    command[command.index("--shooter-version") + 1]
                )

        def fake_send(body: dict, _checkpoint: Path) -> bool:
            version = body["shooter_version"]
            sent_versions.append(version)
            return version != "latest-merged"

        monkeypatch.setattr(evaluator, "_run_command", fake_run_command)
        monkeypatch.setattr(evaluator, "_sync_jd_dataset_mapping", lambda _: None)
        monkeypatch.setattr(JD, "send_metrics", fake_send)

        evaluator._send_metrics_to_jd({"AI2D_acc": 0.8})

        assert cache_versions == ["0.5.0", "latest-merged"]
        assert sent_versions == ["0.5.0", "latest-merged"]
        assert committed_versions == ["0.5.0"]

    def test_send_metrics_to_jd_skips_latest_merged_when_cache_commit_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator, _ = self._prepared_evaluator(tmp_path)
        cache_versions: list[str] = []
        sent_versions: list[str] = []

        def fake_upload(*, model_name: str, step: int, shooter_version: str) -> str:
            cache_versions.append(shooter_version)
            return "gigaeye-s3:bucket/cache_scores.run-id.jsonl.zst"

        def fake_send(body: dict, _checkpoint: Path) -> bool:
            sent_versions.append(body["shooter_version"])
            return True

        monkeypatch.setattr(evaluator, "_upload_jd_cache", fake_upload)
        monkeypatch.setattr(evaluator, "_commit_jd_cache", lambda **kwargs: False)
        monkeypatch.setattr(evaluator, "_sync_jd_dataset_mapping", lambda _: None)
        monkeypatch.setattr(JD, "send_metrics", fake_send)

        evaluator._send_metrics_to_jd({"AI2D_acc": 0.8})

        assert cache_versions == ["0.5.0"]
        assert sent_versions == ["0.5.0"]

    @pytest.mark.parametrize("shooter_version", ["0.5.0", "latest-merged"])
    @pytest.mark.parametrize(
        ("deliveries", "expected_versioned_path", "expected_latest_merged_path"),
        [
            (
                {
                    "first": (
                        "publish",
                        "gigaeye-s3:bucket/cache_scores.jsonl.zst",
                    ),
                    "second": (
                        "publish",
                        "gigaeye-s3:bucket/cache_scores.jsonl.zst",
                    ),
                },
                "gigaeye-s3:bucket/cache_scores.jsonl.zst",
                "gigaeye-s3:bucket/cache_scores.jsonl.zst",
            ),
            (
                {
                    "first": (
                        "publish",
                        "gigaeye-s3:bucket/cache_scores.jsonl.zst",
                    ),
                    "second": ("clear", None),
                },
                "",
                "",
            ),
            (
                {"first": ("preserve", None), "second": ("preserve", None)},
                None,
                "",
            ),
            (
                {
                    "first": (
                        "publish",
                        "gigaeye-s3:bucket/cache_scores.jsonl.zst",
                    ),
                    "second": ("preserve", None),
                },
                "gigaeye-s3:bucket/cache_scores.jsonl.zst",
                "",
            ),
            (
                {
                    "first": ("publish", "gigaeye-s3:bucket/cache-a.jsonl.zst"),
                    "second": ("publish", "gigaeye-s3:bucket/cache-b.jsonl.zst"),
                },
                "",
                "",
            ),
        ],
    )
    def test_upload_jd_cache_aggregates_run_deliveries(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        deliveries: dict[str, tuple[str, str | None]],
        expected_versioned_path: str | None,
        expected_latest_merged_path: str,
        shooter_version: str,
    ):
        evaluator, _ = self._prepared_evaluator(tmp_path)
        evaluator.run_exports = {
            run_name: {
                "config_path": evaluator.metrics_dir
                / "lmms_eval_configs"
                / f"{run_name}.yaml",
                "output_dir": evaluator.metrics_dir / run_name,
                "metrics_path": evaluator.metrics_dir / f"{run_name}_metrics.json",
            }
            for run_name in ("first", "second")
        }
        commands = []

        def fake_run_command(command: list[str], run_name: str) -> None:
            commands.append(command)
            delivery_run_name = run_name.rsplit(" ", 1)[-1]
            action, cache_s3_path = deliveries[delivery_run_name]
            Path(command[command.index(OUTPUT_FLAG) + 1]).write_text(
                json.dumps(
                    {
                        "action": action,
                        "cache_s3_path": cache_s3_path,
                        "error": None,
                    }
                ),
                encoding="utf-8",
            )

        monkeypatch.setattr(evaluator, "_run_command", fake_run_command)

        expected_cache_s3_path = (
            expected_latest_merged_path
            if shooter_version == "latest-merged"
            else expected_versioned_path
        )
        assert (
            evaluator._upload_jd_cache(
                model_name="model-lite",
                step=10,
                shooter_version=shooter_version,
            )
            == expected_cache_s3_path
        )
        assert len(commands) == 2
        assert all(
            command[command.index("--shooter-version") + 1] == shooter_version
            for command in commands
        )
        assert (
            len(
                {command[command.index("--publication-id") + 1] for command in commands}
            )
            == 1
        )

    @pytest.mark.parametrize(
        ("action", "cache_s3_path"),
        [("clear", ""), ("preserve", None), ("publish", "")],
    )
    def test_send_metrics_to_jd_skips_latest_merged_without_uploaded_cache(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        action: str,
        cache_s3_path: str | None,
    ):
        evaluator, _ = self._prepared_evaluator(tmp_path)
        captured = []
        versioned_cache_s3_path = "gigaeye-s3:bucket/0.5.0/cache_scores.jsonl.zst"

        def fake_run_command(command: list[str], _run_name: str) -> None:
            if command[3:6] != ["jd", "cache", "upload"]:
                return
            latest_merged = (
                command[command.index("--shooter-version") + 1] == "latest-merged"
            )
            Path(command[command.index(OUTPUT_FLAG) + 1]).write_text(
                json.dumps(
                    {
                        "action": action if latest_merged else "publish",
                        "cache_s3_path": (
                            cache_s3_path if latest_merged else versioned_cache_s3_path
                        ),
                        "error": None,
                    }
                ),
                encoding="utf-8",
            )

        monkeypatch.setattr(evaluator, "_run_command", fake_run_command)

        def fake_send_metrics(body: dict, _artifacts: Path) -> bool:
            captured.append(body)
            return True

        monkeypatch.setattr(JD, "send_metrics", fake_send_metrics)

        evaluator._send_metrics_to_jd({"AI2D_acc": 0.8})

        assert len(captured) == 1
        assert captured[0]["shooter_version"] == "0.5.0"
        assert captured[0]["cache_s3_path"] == versioned_cache_s3_path

    def test_upload_jd_cache_clears_path_when_one_run_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator, _ = self._prepared_evaluator(tmp_path)
        evaluator.run_exports = {
            run_name: {
                "config_path": evaluator.metrics_dir
                / "lmms_eval_configs"
                / f"{run_name}.yaml",
                "output_dir": evaluator.metrics_dir / run_name,
                "metrics_path": evaluator.metrics_dir / f"{run_name}_metrics.json",
            }
            for run_name in ("first", "second")
        }

        def fake_run_command(command: list[str], run_name: str) -> None:
            if run_name.endswith("first"):
                raise lmms_eval_module.SubprocessManagerError("cache upload is down")
            Path(command[command.index(OUTPUT_FLAG) + 1]).write_text(
                json.dumps(
                    {
                        "action": "publish",
                        "cache_s3_path": "gigaeye-s3:bucket/cache.jsonl.zst",
                        "error": None,
                    }
                ),
                encoding="utf-8",
            )

        monkeypatch.setattr(evaluator, "_run_command", fake_run_command)

        assert (
            evaluator._upload_jd_cache(
                model_name="model-lite",
                step=10,
                shooter_version="0.5.0",
            )
            == ""
        )

    @pytest.mark.parametrize(
        ("delivery", "expected_warning"),
        [
            (
                {
                    "action": "clear",
                    "cache_s3_path": None,
                    "error": "cache upload failed",
                },
                "Failed to upload lmms-eval JD cache for chartqa; "
                "clearing its JD path: cache upload failed",
            ),
            (
                {
                    "action": "unexpected",
                    "cache_s3_path": None,
                    "error": None,
                },
                "Invalid lmms-eval JD cache delivery for chartqa; clearing its JD path",
            ),
        ],
    )
    def test_upload_jd_cache_clears_error_deliveries(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        delivery: dict[str, str | None],
        expected_warning: str,
    ):
        evaluator, _ = self._prepared_evaluator(tmp_path)
        warnings = []

        def fake_run_command(command: list[str], _run_name: str) -> None:
            Path(command[command.index(OUTPUT_FLAG) + 1]).write_text(
                json.dumps(delivery), encoding="utf-8"
            )

        monkeypatch.setattr(evaluator, "_run_command", fake_run_command)
        monkeypatch.setattr(lmms_eval_module.Logger, "warning", warnings.append)

        assert (
            evaluator._upload_jd_cache(
                model_name="model-lite",
                step=10,
                shooter_version="0.5.0",
            )
            == ""
        )
        assert warnings == [expected_warning]

    def test_send_metrics_to_jd_clears_cache_path_when_cache_upload_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator, _ = self._prepared_evaluator(tmp_path)
        captured = {}

        def fail_cache_upload(command: list[str], _run_name: str) -> None:
            if command[3:6] == ["jd", "cache", "upload"]:
                raise lmms_eval_module.SubprocessManagerError("cache upload is down")

        monkeypatch.setattr(evaluator, "_run_command", fail_cache_upload)

        def fake_send_metrics(body: dict, _artifacts: Path) -> bool:
            captured.update(body=body)
            return True

        monkeypatch.setattr(
            JD,
            "send_metrics",
            fake_send_metrics,
        )

        evaluator._send_metrics_to_jd({"AI2D_acc": 0.8})

        assert captured["body"]["cache_s3_path"] == ""

    def test_send_metrics_to_jd_keeps_zero_exported_metrics(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator, _ = self._prepared_evaluator(tmp_path)
        evaluator._jd_metrics = NESTED_LMMS_EVAL_EXPORTED_METRICS
        captured = {}
        monkeypatch.setattr(evaluator, "_run_command", lambda *_: None)

        def fake_send_metrics(body: dict, _artifacts: Path) -> bool:
            captured.update(body=body)
            return True

        monkeypatch.setattr(
            JD,
            "send_metrics",
            fake_send_metrics,
        )

        evaluator._send_metrics_to_jd(MIXED_LMMS_EVAL_EXPORTED_METRICS)

        assert captured["body"]["metrics"] == {10: NESTED_LMMS_EVAL_EXPORTED_METRICS}
        assert json.loads(
            (evaluator.metrics_dir / "metrics_step_10.json").read_text(encoding="utf-8")
        )["metrics"] == {
            "10": NESTED_LMMS_EVAL_EXPORTED_METRICS,
        }

    def test_send_metrics_to_jd_forwards_autocompare_id(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator, _ = self._prepared_evaluator(tmp_path)
        captured = {}
        monkeypatch.setenv("AUTOCOMPARE_ID", "compare-1")
        monkeypatch.setattr(evaluator, "_run_command", lambda *_: None)

        def fake_send_metrics(body: dict, _artifacts: Path) -> bool:
            captured.update(body=body)
            return True

        monkeypatch.setattr(
            JD,
            "send_metrics",
            fake_send_metrics,
        )

        evaluator._send_metrics_to_jd({"AI2D_acc": 0.8})

        assert captured["body"]["autocompare_id"] == "compare-1"

    def test_send_metrics_to_jd_skips_empty_metrics(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator, _ = self._prepared_evaluator(tmp_path)
        monkeypatch.setattr(
            JD,
            "send_metrics",
            lambda body, _artifacts: pytest.fail("must not send empty metrics"),
        )

        evaluator._send_metrics_to_jd({})

    def test_send_metrics_to_jd_skips_fully_filtered_metrics(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator, _ = self._prepared_evaluator(tmp_path)
        evaluator._jd_metrics = {"AI2D_acc": float("nan")}
        warnings = []
        monkeypatch.setattr(lmms_eval_module.Logger, "warning", warnings.append)
        monkeypatch.setattr(
            JD,
            "send_metrics",
            lambda *_: pytest.fail("must not send unavailable primary metrics"),
        )

        evaluator._send_metrics_to_jd({"AI2D_acc": float("nan")})
        assert warnings == ["No valid lmms-eval primary metrics to send to JD"]

    def test_send_metrics_to_jd_continues_when_mapping_sync_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator, _ = self._prepared_evaluator(tmp_path)
        captured = {}
        warnings = []

        def fail_mapping_sync(command: list[str], run_name: str) -> None:
            if command[3:6] == ["jd", "cache", "upload"]:
                Path(command[command.index(OUTPUT_FLAG) + 1]).write_text(
                    json.dumps(
                        {
                            "action": "preserve",
                            "cache_s3_path": None,
                            "error": None,
                        }
                    ),
                    encoding="utf-8",
                )
                return
            raise lmms_eval_module.SubprocessManagerError("mapping is down")

        monkeypatch.setattr(evaluator, "_run_command", fail_mapping_sync)

        def fake_send_metrics(body: dict, _artifacts: Path) -> bool:
            captured.update(body=body)
            return True

        monkeypatch.setattr(
            JD,
            "send_metrics",
            fake_send_metrics,
        )
        monkeypatch.setattr(lmms_eval_module.Logger, "warning", warnings.append)

        evaluator._send_metrics_to_jd({"AI2D_acc": 0.8})

        assert captured["body"]["metrics"] == {10: {"AI2D_acc": 0.8}}
        assert warnings == [
            "Failed to sync lmms-eval JD metric mapping, metrics send will continue: "
            "mapping is down",
            "Skipping lmms-eval latest-merged publication: CI cache is unavailable or incomplete",
        ]

    def test_send_metrics_to_jd_sends_without_run_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator, _ = self._prepared_evaluator(tmp_path)
        captured = {}
        warnings = []
        evaluator.run_exports = {}

        def fake_send_metrics(body: dict, _artifacts: Path) -> bool:
            captured.update(body=body)
            return True

        monkeypatch.setattr(
            JD,
            "send_metrics",
            fake_send_metrics,
        )
        monkeypatch.setattr(lmms_eval_module.Logger, "warning", warnings.append)

        evaluator._send_metrics_to_jd({"AI2D_acc": 0.8})

        assert captured["body"]["metrics"] == {10: {"AI2D_acc": 0.8}}
        assert warnings == [
            "No lmms-eval run config found, skipping JD metric mapping sync",
            "Skipping lmms-eval latest-merged publication: CI cache is unavailable or incomplete",
        ]
