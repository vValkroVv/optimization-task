import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from node_runner.src.config_model import (
    AppWorldEvaluatorConfig,
    EvaluatorsConfig,
    ModelInfoConfig,
)
from node_runner.src.metrics.evaluators import appworld as appworld_module
from node_runner.src.metrics.evaluators.appworld import AppWorldEvaluator
from node_runner.src.metrics.metrics_manager import EVALUATORS

pytestmark = pytest.mark.unit


def _build_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    **cfg_overrides,
) -> AppWorldEvaluator:
    version_file = tmp_path / "appworld_version.txt"
    version_file.write_text("appworld-version\n")
    monkeypatch.setattr(appworld_module, "APPWORLD_VERSION_FILE", str(version_file))
    checkpoint = cfg_overrides.pop("hf_checkpoint_path", tmp_path / "hf_ckpt")
    cfg = AppWorldEvaluatorConfig(
        appworld_root=cfg_overrides.pop("appworld_root", str(tmp_path / "appworld")),
        **cfg_overrides,
    )
    cfg.apply_defaults(inference_instances=8, base_url="http://localhost:29999")
    evaluator = AppWorldEvaluator()
    evaluator.setup(
        autometric_runs_config=cfg,
        artifacts_path=tmp_path / "artifacts",
        hf_checkpoint_path=checkpoint,
        model_info=ModelInfoConfig(model_class="sft", size="lite", username="u"),
    )
    return evaluator


def _write_eval_json(
    evaluator: AppWorldEvaluator,
    task: float = 75.0,
    scenario: float = 50.0,
) -> Path:
    eval_json = evaluator._eval_json_path()
    eval_json.parent.mkdir(parents=True, exist_ok=True)
    eval_json.write_text(
        json.dumps(
            {
                "aggregate": {
                    "task_goal_completion": task,
                    "scenario_goal_completion": scenario,
                }
            }
        )
    )
    return eval_json


class TestAppWorldConfig:
    def test_defaults_and_registration(self):
        cfg = AppWorldEvaluatorConfig()

        assert cfg.dataset == "test_normal"
        assert cfg.num_processes is None
        assert cfg.max_steps == 50
        assert cfg.request_timeout == pytest.approx(21600.0)
        assert cfg.max_retries == 20
        assert dict(EvaluatorsConfig(appworld={})) == {"appworld": cfg}
        assert EVALUATORS["appworld"] is AppWorldEvaluator

    @pytest.mark.parametrize("dataset", ["test_normal", "test_challenge"])
    def test_supported_datasets(self, dataset: str):
        assert AppWorldEvaluatorConfig(dataset=dataset).dataset == dataset

    @pytest.mark.parametrize(
        "config",
        [
            {"dataset": "train"},
            {"num_processes": 0},
            {"max_steps": 0},
            {"request_timeout": 0},
            {"max_retries": 0},
            {"num_proceses": 4},
        ],
    )
    def test_rejects_invalid_config(self, config: dict):
        with pytest.raises(ValidationError):
            AppWorldEvaluatorConfig.model_validate(config)

    def test_apply_defaults_sets_base_url_and_preserves_override(self):
        default = AppWorldEvaluatorConfig()
        explicit = AppWorldEvaluatorConfig(
            base_url="http://explicit/v1", num_processes=7
        )

        default.apply_defaults(inference_instances=3, base_url="http://host:29999")
        explicit.apply_defaults(inference_instances=3, base_url="http://ignored:29999")

        assert default.base_url == "http://host:29999/v1"
        assert default.num_processes == 384
        assert explicit.base_url == "http://explicit/v1"
        assert explicit.num_processes == 7


class TestAppWorldSetup:
    def test_setup_resolves_absolute_paths(self, tmp_path: Path, monkeypatch):
        evaluator = _build_evaluator(tmp_path, monkeypatch)

        assert evaluator.appworld_root == tmp_path / "appworld"
        assert evaluator.metrics_path == (
            tmp_path / "artifacts" / "metrics" / "appworld" / "appworld_metrics.json"
        )
        assert evaluator.fingerprint_path.name == "appworld_setup.json"
        assert evaluator.metrics_dir.is_dir()

    def test_relative_root_is_anchored_at_repo_root(self, tmp_path: Path, monkeypatch):
        evaluator = _build_evaluator(
            tmp_path, monkeypatch, appworld_root="relative/appworld"
        )

        assert evaluator.appworld_root == Path(__file__).resolve().parents[4] / (
            "relative/appworld"
        )

    def test_experiment_name_and_binary(self, tmp_path: Path, monkeypatch):
        evaluator = _build_evaluator(
            tmp_path,
            monkeypatch,
            dataset="test_challenge",
            venv_python="/opt/appworld_env/bin/python",
        )

        assert evaluator._experiment_name.endswith("/test_challenge")
        assert evaluator._appworld_bin() == "/opt/appworld_env/bin/appworld"


class TestAppWorldRuntimeConfig:
    def test_override_contains_effective_model_config(
        self, tmp_path: Path, monkeypatch
    ):
        evaluator = _build_evaluator(
            tmp_path,
            monkeypatch,
            model_name="candidate",
            max_steps=25,
            request_timeout=123.5,
            max_retries=7,
        )

        assert json.loads(evaluator.autometric_runs_config.to_override_json()) == {
            "config": {
                "agent": {
                    "model_config": {
                        "client_name": "openai",
                        "api_type": "chat_completions",
                        "name": "candidate",
                        "base_url": "http://localhost:29999/v1",
                        "api_key_env_name": "OPENAI_API_KEY",
                        "timeout": 123.5,
                        "max_retries": 7,
                        "use_cache": True,
                    },
                    "max_steps": 25,
                }
            }
        }

    def test_fingerprint_has_backend_and_shooter_identity(self, tmp_path, monkeypatch):
        evaluator = _build_evaluator(tmp_path, monkeypatch)

        fingerprint = evaluator._fingerprint()

        model = fingerprint["override"]["config"]["agent"]["model_config"]
        assert model["client_name"] == "openai"
        assert model["api_type"] == "chat_completions"
        assert model["base_url"] == "http://localhost:29999/v1"
        assert fingerprint["shooter_version"] == "appworld-version"

    def test_environment_points_to_installed_root(self, tmp_path: Path, monkeypatch):
        evaluator = _build_evaluator(tmp_path, monkeypatch)

        env = evaluator._build_env()

        assert env["OPENAI_API_KEY"] == "empty"
        assert env["APPWORLD_ROOT"] == str(evaluator.appworld_root)

    def test_run_cmd_uses_subprocess_manager(self, tmp_path, monkeypatch):
        evaluator = _build_evaluator(tmp_path, monkeypatch)
        calls: list[str] = []
        captured: dict[str, object] = {}

        class FakeManager:
            def __init__(self, args, log_file, cwd, run_name, env):
                captured.update(
                    args=args, log_file=log_file, cwd=cwd, run_name=run_name, env=env
                )

            def start(self):
                calls.append("start")

            def wait_for_completion(self):
                calls.append("wait")

            def log_error(self):
                calls.append("log_error")

        monkeypatch.setattr(appworld_module, "SubprocessManager", FakeManager)

        evaluator._run_cmd(["appworld", "run"], "appworld_run")

        assert calls == ["start", "wait", "log_error"]
        assert captured["cwd"] == str(evaluator.appworld_root)
        assert captured["log_file"] == evaluator.log_file


class TestAppWorldRun:
    def test_run_collects_metrics_and_reuses_compatible_result(
        self, tmp_path: Path, monkeypatch
    ):
        evaluator = _build_evaluator(tmp_path, monkeypatch, num_processes=64)
        recorded: list[tuple[list[str], str]] = []

        def fake_run(args: list[str], run_name: str) -> None:
            recorded.append((args, run_name))
            _write_eval_json(evaluator)

        monkeypatch.setattr(evaluator, "_run_cmd", fake_run)

        evaluator.run()
        evaluator.run()

        assert len(recorded) == 1
        args, run_name = recorded[0]
        assert run_name == "appworld_run"
        assert args[:3] == [
            evaluator._appworld_bin(),
            "run",
            evaluator._experiment_name,
        ]
        assert args[args.index("--num-processes") + 1] == "64"
        assert {"--clear-first", "--with-evaluation", "--without-setup"} <= set(args)
        assert evaluator.get_metrics() == {
            "appworld/test_normal/task_goal_completion": pytest.approx(0.75),
            "appworld/test_normal/scenario_goal_completion": pytest.approx(0.5),
        }

    def test_changed_semantic_config_reruns_instead_of_failing(
        self, tmp_path: Path, monkeypatch
    ):
        evaluator = _build_evaluator(tmp_path, monkeypatch)
        runs = 0

        def fake_run(*_):
            nonlocal runs
            runs += 1
            _write_eval_json(evaluator, task=50 + runs * 10)

        monkeypatch.setattr(evaluator, "_run_cmd", fake_run)
        evaluator.run()
        evaluator.autometric_runs_config.max_steps = 25
        evaluator.run()

        assert runs == 2
        assert evaluator.get_metrics()[
            "appworld/test_normal/task_goal_completion"
        ] == pytest.approx(0.7)

    def test_concurrency_change_keeps_compatible_result(self, tmp_path, monkeypatch):
        evaluator = _build_evaluator(tmp_path, monkeypatch, num_processes=8)
        runs = 0

        def fake_run(*_):
            nonlocal runs
            runs += 1
            _write_eval_json(evaluator)

        monkeypatch.setattr(evaluator, "_run_cmd", fake_run)
        evaluator.run()
        evaluator.autometric_runs_config.num_processes = 64
        evaluator.run()

        assert runs == 1

    @pytest.mark.parametrize("marker", [None, "{bad json", "{}"])
    def test_missing_or_invalid_fingerprint_reruns(self, marker, tmp_path, monkeypatch):
        evaluator = _build_evaluator(tmp_path, monkeypatch)
        evaluator.metrics_path.write_text("{}")
        if marker is not None:
            evaluator.fingerprint_path.write_text(marker)
        runs = 0

        def fake_run(*_):
            nonlocal runs
            runs += 1
            _write_eval_json(evaluator)

        monkeypatch.setattr(evaluator, "_run_cmd", fake_run)

        evaluator.run()

        assert runs == 1

    def test_failed_rerun_drops_stale_pam_metrics(self, tmp_path, monkeypatch):
        evaluator = _build_evaluator(tmp_path, monkeypatch)
        evaluator.metrics_path.write_text("{}")
        evaluator.fingerprint_path.write_text("{}")
        monkeypatch.setattr(
            evaluator,
            "_run_cmd",
            lambda *_: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        with pytest.raises(RuntimeError, match="boom"):
            evaluator.run()

        assert evaluator.get_metrics() == {}
        assert not evaluator.fingerprint_path.exists()


class TestAppWorldMetrics:
    def test_collect_metrics_writes_pam_result(self, tmp_path: Path, monkeypatch):
        evaluator = _build_evaluator(tmp_path, monkeypatch)
        _write_eval_json(evaluator)

        evaluator._collect_metrics()

        assert evaluator.get_metrics() == {
            "appworld/test_normal/task_goal_completion": 0.75,
            "appworld/test_normal/scenario_goal_completion": 0.5,
        }

    def test_collect_metrics_fails_when_evaluation_is_missing(
        self, tmp_path, monkeypatch
    ):
        evaluator = _build_evaluator(tmp_path, monkeypatch)

        with pytest.raises(RuntimeError, match="not found after run"):
            evaluator._collect_metrics()

    def test_collect_metrics_fails_on_malformed_evaluation(self, tmp_path, monkeypatch):
        evaluator = _build_evaluator(tmp_path, monkeypatch)
        eval_json = evaluator._eval_json_path()
        eval_json.parent.mkdir(parents=True, exist_ok=True)
        eval_json.write_text(json.dumps({"aggregate": {"task_goal_completion": 0.5}}))

        with pytest.raises(RuntimeError, match="malformed"):
            evaluator._collect_metrics()

    def test_get_metrics_is_empty_before_run(self, tmp_path: Path, monkeypatch):
        assert _build_evaluator(tmp_path, monkeypatch).get_metrics() == {}


class TestAppWorldReporting:
    def test_after_run_sends_and_archives(self, tmp_path: Path, monkeypatch):
        evaluator = _build_evaluator(tmp_path, monkeypatch)
        called: dict[str, object] = {}
        monkeypatch.setattr(
            evaluator,
            "_send_metrics_to_jd",
            lambda metrics: called.update(send=metrics),
        )
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.tar_metrics_dir",
            lambda metrics_dir, artifacts_path, evaluator_name: (
                tmp_path / "archive.zst"
            ),
        )
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.symlink_metrics_for_transfer",
            lambda tar_path, checkpoint, evaluator_name: called.update(
                tar_path=tar_path, checkpoint=checkpoint
            ),
        )
        metrics = {"appworld/test_normal/task_goal_completion": 0.5}

        evaluator.after_run(metrics)

        assert called == {
            "send": metrics,
            "tar_path": tmp_path / "archive.zst",
            "checkpoint": evaluator.hf_checkpoint_path,
        }

    def test_jd_keys_include_methodology(self, tmp_path: Path, monkeypatch):
        evaluator = _build_evaluator(tmp_path, monkeypatch, max_steps=25)

        converted = evaluator._convert_metrics_to_jd_keys(
            {"appworld/test_normal/task_goal_completion": 0.75}
        )

        assert converted == {
            "APPWORLD_TEST_NORMAL_TASK_GOAL_COMPLETION__"
            "SCAFFOLD_SIMPLIFIED_REACT_CODE_AGENT_MAX_STEPS_25": 0.75
        }

    @pytest.mark.usefixtures("scalar_ci_upload")
    def test_send_builds_jd_payload(self, tmp_path: Path, monkeypatch):
        evaluator = _build_evaluator(
            tmp_path,
            monkeypatch,
            hf_checkpoint_path=(
                tmp_path / "model" / "ep1_ba2" / "artifacts" / "hf_checkpoint"
            ),
        )
        sent: dict[str, object] = {}
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.save_metrics_json",
            lambda *_: None,
        )
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.send_metrics",
            lambda body, artifacts_path: sent.update(body=body),
        )

        evaluator._send_metrics_to_jd(
            {"appworld/test_normal/task_goal_completion": 0.5}
        )

        body = sent["body"]
        assert body["shooter_name"] == "appworld"
        assert body["shooter_version"] == "appworld-version"
        assert body["name"] == "model"
        assert body["metrics"] == {
            2: {
                "APPWORLD_TEST_NORMAL_TASK_GOAL_COMPLETION__"
                "SCAFFOLD_SIMPLIFIED_REACT_CODE_AGENT_MAX_STEPS_50": 0.5,
            }
        }

    @pytest.mark.parametrize("missing_path", ["hf_checkpoint_path", "metrics_dir"])
    @pytest.mark.usefixtures("scalar_ci_upload")
    def test_send_skips_when_required_path_is_missing(
        self, missing_path: str, tmp_path: Path, monkeypatch
    ):
        evaluator = _build_evaluator(tmp_path, monkeypatch)
        monkeypatch.setattr(evaluator, missing_path, None)
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.send_metrics",
            lambda body, artifacts_path: pytest.fail("JD request must not be sent"),
        )

        evaluator._send_metrics_to_jd(
            {"appworld/test_normal/task_goal_completion": 0.5}
        )


@pytest.mark.parametrize("case", ["complete", "unscored", "missing_task", "malformed"])
def test_ci_groups_tasks_by_scenario_with_fixed_task_weights(
    tmp_path, monkeypatch, case
):
    evaluator = _build_evaluator(tmp_path, monkeypatch)
    roster = evaluator.appworld_root / "data/datasets/test_normal.txt"
    roster.parent.mkdir(parents=True)
    roster.write_text("a_1:description\na_2\nb_1\n\n")
    individual = {
        "a_1": {"success": True},
        "a_2": {"success": False},
        "b_1": {"passes_fully": True},
    }
    if case == "unscored":
        individual["a_1"]["success"] = None
    elif case == "missing_task":
        individual.pop("a_2")
    path = evaluator.metrics_dir / "evaluation.json"
    path.write_text(
        "invalid" if case == "malformed" else json.dumps({"individual": individual})
    )
    metrics = {
        "APPWORLD_SCENARIO_GOAL_COMPLETION__TEST": 50,
        "APPWORLD_TASK_GOAL_COMPLETION__TEST": 200 / 3,
        "ERRORS": 0,
    }
    scores = evaluator._cache_scores(metrics)
    if case == "complete":
        assert {
            key: [row["score"] for row in rows] for key, rows in scores.records.items()
        } == {
            "APPWORLD_SCENARIO_GOAL_COMPLETION__TEST": [0, 100],
            "APPWORLD_TASK_GOAL_COMPLETION__TEST": [200 / 3, 200 / 3],
        }
    else:
        assert scores.records == {}
