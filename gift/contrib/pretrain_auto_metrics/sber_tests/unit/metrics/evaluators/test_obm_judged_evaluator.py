import os
from pathlib import Path

import pytest
from node_runner.src.job_dispatcher import JD
import yaml

from node_runner.src.config_model import (
    DevicesObmJudgedEvaluatorConfig,
    ObmAllConfig,
    OpenaiObmJudgedEvaluatorConfig,
)
from node_runner.src.config_model import ModelInfoConfig as ModelInfo
from node_runner.src.metrics.evaluators.obm import (
    NewObmJudgedEvaluator,
    ObmJudgedEvaluator,
)
from sber_tests.test_support import judge_endpoints

pytestmark = pytest.mark.unit


def _make_cfg(
    judges: dict[str, list[str] | str] | None = None,
    **all_kwargs,
) -> DevicesObmJudgedEvaluatorConfig:
    all_kwargs.setdefault("tasks", ["default_task"])
    if judges is None:
        judges = {"judge-a": ["ds1", "ds2"], "judge-b": ["ds3"]}
    return DevicesObmJudgedEvaluatorConfig(
        obm_config="devices",
        all=ObmAllConfig(**all_kwargs),
        n_threads=8,
        judges=judges,
    )


class TestObmJudgedSecondaryModels:
    # The judge -> datasets grouping now lives directly in the config: judges
    # maps a judge model to its task subset, secondary_model_names lists judges.
    def test_judges_map_model_to_task_subset(self):
        cfg = _make_cfg(judges={"model-a": ["ds1", "ds3"], "model-b": ["ds2"]})
        assert cfg.secondary_model_names == ["model-a", "model-b"]
        assert cfg.judges["model-a"] == ["ds1", "ds3"]

    def test_judges_normalize_single_task_string_to_list(self):
        cfg = _make_cfg(judges={"model-a": "ds1"})
        assert cfg.judges == {"model-a": ["ds1"]}

    def test_empty_judges_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            _make_cfg(judges={})


class TestObmJudgedSetup:
    def test_setup_initializes_state_and_secondary_inference_runs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = ObmJudgedEvaluator()
        artifacts_path = tmp_path / "artifacts"
        artifacts_path.mkdir(parents=True, exist_ok=True)
        cfg = _make_cfg()
        monkeypatch.setattr(evaluator, "_unzip_obm", lambda: tmp_path / "obm_repo")

        evaluator.setup(
            autometric_runs_config=cfg,
            artifacts_path=artifacts_path,
            hf_checkpoint_path=tmp_path / "hf",
            model_info=ModelInfo(model_class="sft", size="lite", username="u"),
        )

        assert evaluator.obm_repo_path == tmp_path / "obm_repo"
        assert evaluator.obm_config is cfg
        assert evaluator.metrics_dir == artifacts_path / "metrics" / "obm_judged"
        assert evaluator.original_artifacts_path == artifacts_path
        assert set(cfg.secondary_model_names) == {"judge-a", "judge-b"}


def _setup_for_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cfg
) -> ObmJudgedEvaluator:
    evaluator = ObmJudgedEvaluator()
    monkeypatch.setattr(evaluator, "_unzip_obm", lambda: tmp_path / "obm_repo")
    artifacts_path = tmp_path / "artifacts"
    artifacts_path.mkdir(parents=True, exist_ok=True)
    cfg.judge_endpoints = judge_endpoints(
        {
            "judge-a": "http://10.0.0.1:8000",
            "judge-b": "http://10.0.0.2:8001",
        }
    )
    cfg.apply_defaults(
        inference_instances=1,
        base_url="http://localhost:29999",
    )
    evaluator.setup(
        autometric_runs_config=cfg,
        artifacts_path=artifacts_path,
        hf_checkpoint_path=tmp_path / "hf",
        model_info=ModelInfo(model_class="sft", size="lite", username="u"),
    )
    return evaluator


class TestObmJudgedRunOnModelName:
    def test_runs_one_stage_per_judge_next_to_the_evaluated_model(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        from node_runner.src.metrics.metrics_manager import EVALUATED
        from node_runner.src.metrics.metrics_manager import EVALUATORS

        evaluator = _setup_for_run(monkeypatch, tmp_path, _make_cfg())
        assert [stage.engine_keys for stage in evaluator.stages()] == [
            (EVALUATED, "judge-a"),
            (EVALUATED, "judge-b"),
        ]
        assert EVALUATORS["obm_judged"] is ObmJudgedEvaluator

    def test_run_on_model_name_runs_subprocess_and_writes_yaml(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = _setup_for_run(monkeypatch, tmp_path, _make_cfg())
        captured: dict[str, object] = {
            "start": 0,
            "wait": 0,
            "log_error": 0,
            "cmd": None,
        }

        class _SP:
            def __init__(self, cmd, log_file, cwd, run_name="", env=None):
                captured["cmd"] = cmd
                captured["env"] = env

            def start(self):
                captured["start"] = int(captured["start"]) + 1

            def wait_for_completion(self):
                captured["wait"] = int(captured["wait"]) + 1

            def log_error(self):
                captured["log_error"] = int(captured["log_error"]) + 1

        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.obm.SubprocessManager", _SP
        )

        evaluator.run_on_model_name("judge-a")

        assert captured["start"] == 1
        assert captured["wait"] == 1
        assert captured["log_error"] == 1
        cmd = captured["cmd"]
        assert "obm.text" in cmd
        tasks_idx = cmd.index("--tasks")
        assert cmd[tasks_idx + 1 :] == ["ds1", "ds2"]
        assert captured["env"] is not None
        cfg_path = cmd[cmd.index("-cfg") + 1]
        assert cfg_path == str(
            evaluator.artifacts_path / "obm_judged_config_judge-a.yaml"
        )

    def test_run_on_model_name_writes_judge_url_per_judge(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = _setup_for_run(monkeypatch, tmp_path, _make_cfg())
        cmds: list[list[str]] = []

        class _SP:
            def __init__(self, cmd, log_file, cwd, run_name="", env=None):
                cmds.append(cmd)

            def start(self):
                pass

            def wait_for_completion(self):
                pass

            def log_error(self):
                pass

        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.obm.SubprocessManager", _SP
        )

        evaluator.run_on_model_name("judge-a")
        evaluator.run_on_model_name("judge-b")

        judge_urls = []
        for cmd in cmds:
            with open(cmd[cmd.index("-cfg") + 1]) as f:
                data = yaml.safe_load(f)
            judge_urls.append(
                data["benchmark"]["runner"]["llm_judge_shooter"]["openai"]["api"][
                    "base_url"
                ]
            )
        assert judge_urls == ["http://10.0.0.1:8000/v1", "http://10.0.0.2:8001/v1"]


class TestObmJudgedBuildCmd:
    def _setup_evaluator(self, tmp_path, cfg) -> ObmJudgedEvaluator:
        cfg.judge_endpoints = judge_endpoints(
            {"judge-a": "http://x:1", "judge-b": "http://x:1"}
        )
        cfg.apply_defaults(
            inference_instances=1,
            base_url="http://localhost:29999",
        )
        evaluator = ObmJudgedEvaluator()
        evaluator.autometric_runs_config = cfg
        evaluator.artifacts_path = tmp_path
        evaluator.metrics_dir = tmp_path / "metrics_dir"
        return evaluator

    def test_cmd_contains_workers_restart_and_cfg(self, tmp_path: Path):
        evaluator = self._setup_evaluator(
            tmp_path, _make_cfg(n_workers=3, restart=True)
        )
        cmd = evaluator._obm_judged_run_cmd(tmp_path / "runner.yaml")

        w_idx = cmd.index("-w")
        assert cmd[w_idx + 1] == "3"

        assert "-r" in cmd

        cfg_idx = cmd.index("-cfg")
        assert cmd[cfg_idx + 1] == str(tmp_path / "runner.yaml")

    def test_cmd_with_explicit_context_overrides_generation(self, tmp_path: Path):
        evaluator = self._setup_evaluator(
            tmp_path, _make_cfg(context="/custom/ctx.yaml")
        )
        cmd = evaluator._obm_judged_run_cmd(tmp_path / "runner.yaml")
        ctx_idx = cmd.index("-ctx")
        assert cmd[ctx_idx + 1] == "/custom/ctx.yaml"

    def test_cmd_generates_context_when_disable_few_shots(self, tmp_path: Path):
        evaluator = self._setup_evaluator(tmp_path, _make_cfg(disable_few_shots=True))
        cmd = evaluator._obm_judged_run_cmd(tmp_path / "runner.yaml")
        ctx_idx = cmd.index("-ctx")
        ctx_path = cmd[ctx_idx + 1]
        assert Path(ctx_path).exists()
        with open(ctx_path) as f:
            data = yaml.safe_load(f)
        assert data["disable_few_shots"] is True

    def test_offline_run_env_sets_obm_encrypt_without_mutating_environ(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = self._setup_evaluator(tmp_path, _make_cfg())
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.internet_is_available",
            lambda: False,
        )
        monkeypatch.delenv("OBM_ENCRYPT", raising=False)
        env = evaluator._obm_run_env()
        assert env["OBM_ENCRYPT"] == "1"
        assert os.environ.get("OBM_ENCRYPT") is None

    def test_online_run_env_leaves_obm_encrypt_unset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = self._setup_evaluator(tmp_path, _make_cfg())
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.internet_is_available",
            lambda: True,
        )
        monkeypatch.delenv("OBM_ENCRYPT", raising=False)
        assert "OBM_ENCRYPT" not in evaluator._obm_run_env()

    def test_cmd_skips_workers_flag_when_n_workers_is_zero(self, tmp_path: Path):
        evaluator = self._setup_evaluator(tmp_path, _make_cfg(n_workers=0))
        cmd = evaluator._obm_judged_run_cmd(tmp_path / "runner.yaml")
        assert "-w" not in cmd


class TestObmJudgedWriteRunnerYaml:
    def test_yaml_contains_overridden_llm_judge_shooter(self, tmp_path: Path):
        evaluator = ObmJudgedEvaluator()
        cfg = _make_cfg()
        cfg.judge_endpoints = judge_endpoints(
            {
                "judge-a": "http://10.0.0.1:8000",
                "judge-b": "http://10.0.0.2:8001",
            }
        )
        cfg.apply_defaults(
            inference_instances=1,
            base_url="http://localhost:29999",
        )
        evaluator.autometric_runs_config = cfg
        artifacts_path = tmp_path / "artifacts"
        artifacts_path.mkdir(parents=True, exist_ok=True)
        evaluator.artifacts_path = artifacts_path

        path = evaluator._write_runner_yaml("judge-a")

        assert path == artifacts_path / "obm_judged_config_judge-a.yaml"
        assert path.exists()
        with open(path) as f:
            data = yaml.safe_load(f)
        # The judge shooter targets the secondary (judge) engine URL.
        judge_api = data["benchmark"]["runner"]["llm_judge_shooter"]["openai"]["api"]
        assert judge_api["base_url"] == "http://10.0.0.1:8000/v1"
        assert judge_api["timeout"] == 21600


class TestObmJudgedAfterRun:
    def test_after_run_invokes_jd_tar_and_symlink(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = ObmJudgedEvaluator()
        evaluator.metrics_dir = tmp_path / "metrics"
        evaluator.original_artifacts_path = tmp_path / "artifacts"
        evaluator.hf_checkpoint_path = tmp_path / "hf"

        called: dict[str, object] = {}
        monkeypatch.setattr(
            evaluator,
            "_send_metrics_to_jd",
            lambda metrics: called.update({"send": metrics}),
        )
        monkeypatch.setattr(
            JD,
            "tar_metrics_dir",
            lambda metrics_dir, artifacts, _name: tmp_path / "m.tar.zst",
        )
        monkeypatch.setattr(
            JD,
            "symlink_metrics_for_transfer",
            lambda tar_path, hf, _name: called.update({"symlink": tar_path}),
        )

        evaluator.after_run({"m": 1.0})

        assert called["send"] == {"m": 1.0}
        assert called["symlink"] == tmp_path / "m.tar.zst"


class TestObmJudgedOpenaiVariant:
    def test_openai_variant_writes_yaml(self, tmp_path: Path):
        cfg = OpenaiObmJudgedEvaluatorConfig(
            obm_config="openai",
            all=ObmAllConfig(tasks=["t"]),
            n_threads=8,
            judges={"j-a": ["t"]},
        )
        cfg.judge_endpoints = judge_endpoints({"j-a": "http://10.0.0.1:8000"})
        cfg.apply_defaults(
            inference_instances=1,
            base_url="http://localhost:29999",
        )
        evaluator = ObmJudgedEvaluator()
        evaluator.autometric_runs_config = cfg
        artifacts_path = tmp_path / "artifacts"
        artifacts_path.mkdir(parents=True, exist_ok=True)
        evaluator.artifacts_path = artifacts_path
        path = evaluator._write_runner_yaml("j-a")
        with open(path) as f:
            data = yaml.safe_load(f)
        assert "shooter" in data["benchmark"]["runner"]
        assert "llm_judge_shooter" in data["benchmark"]["runner"]


def test_obm_judged_and_new_obm_judged_write_separate_files(tmp_path: Path):
    paths = []
    for evaluator in (ObmJudgedEvaluator(), NewObmJudgedEvaluator()):
        cfg = _make_cfg(disable_few_shots=True)
        cfg.judge_endpoints = judge_endpoints({"judge-a": "http://10.0.0.1:8000"})
        evaluator.autometric_runs_config = cfg
        evaluator.artifacts_path = tmp_path
        paths.append(evaluator._write_runner_yaml("judge-a"))
        paths.append(Path(evaluator._generate_obm_context("all", cfg.all)))

    assert len(set(paths)) == 4
