import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import node_runner.src.metrics.evaluators.claweval as claweval
from node_runner.src.config_model import (
    CLAWEVAL_DEFAULT_JUDGE_PARALLEL_PER_INFERENCE_INSTANCE,
    CLAWEVAL_DEFAULT_LLM_RETRIES,
    CLAWEVAL_DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE,
    DEFAULT_REQUEST_TIMEOUT,
    ClawEvalEvaluatorConfig,
)
from node_runner.src.config_model import ModelInfoConfig as ModelInfo
from node_runner.src.metrics.evaluators.claweval import ClawEvalEvaluator
from sber_tests.test_support import judge_endpoints

pytestmark = pytest.mark.unit

UPSTREAM_PIN = "d3f02d4938ab0832377d90535013def2b1a2fdc0"


def _with_version_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pin: str = UPSTREAM_PIN
) -> Path:
    version_file = tmp_path / "upstream_commit.txt"
    version_file.write_text(f"{pin}\n", encoding="utf-8")
    monkeypatch.setattr(claweval, "Path", lambda _: version_file)
    return version_file


def _apply_defaults(
    cfg: ClawEvalEvaluatorConfig,
    inference_instances: int = 1,
    base_url: str = "http://localhost:29999",
    judge_instances: int = 1,
) -> None:
    cfg.judge_endpoints = judge_endpoints(
        {cfg.judge.model_name: "http://judge-router:31000"},
        {cfg.judge.model_name: judge_instances},
    )
    cfg.apply_defaults(
        inference_instances=inference_instances,
        base_url=base_url,
    )


def _build_evaluator(
    tmp_path: Path,
    cfg: ClawEvalEvaluatorConfig | None = None,
    judge_instances: int = 1,
) -> ClawEvalEvaluator:
    cfg = cfg or ClawEvalEvaluatorConfig(judge={"model_name": "judge-model"})
    _apply_defaults(cfg, judge_instances=judge_instances)
    evaluator = ClawEvalEvaluator()
    evaluator.setup(
        autometric_runs_config=cfg,
        artifacts_path=tmp_path / "artifacts",
        hf_checkpoint_path=tmp_path / "hf_ckpt",
        model_info=ModelInfo(model_class="sft", size="lite", username="u"),
    )
    return evaluator


class _FakeSubprocessManager:
    instances: list["_FakeSubprocessManager"] = []

    def __init__(self, cmd, log_file, cwd, run_name):
        self.cmd = cmd
        self.log_file = log_file
        self.cwd = cwd
        self.run_name = run_name
        self.started = False
        self.waited = False
        self.logged = False
        self.overrides = None
        self.overrides_mode = None
        self.exit_code = 0
        _FakeSubprocessManager.instances.append(self)

    def start(self) -> None:
        self.started = True
        if "--overrides-file" in self.cmd:
            path = Path(self.cmd[self.cmd.index("--overrides-file") + 1])
            self.overrides = json.loads(path.read_text(encoding="utf-8"))
            self.overrides_mode = path.stat().st_mode & 0o777

    def wait_for_completion(self) -> None:
        self.waited = True

    def log_error(self) -> None:
        self.logged = True

    def is_successful(self) -> bool:
        return self.exit_code == 0


class TestClawEvalRunnerRoot:
    def test_runner_root_defaults_to_the_image_path(self):
        assert ClawEvalEvaluatorConfig().runner_root == "/home/user/claweval/runner"

    def test_runner_root_is_not_serialized_into_the_overrides(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        assert "runner_root" not in evaluator._build_overrides()


class TestClawEvalEvaluatorSetup:
    def test_setup_creates_metrics_dir_and_resolves_paths(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        assert evaluator.metrics_dir.is_dir()
        assert evaluator.metrics_dir == tmp_path / "artifacts" / "metrics" / "claweval"
        assert evaluator._runner_root == Path("/home/user/claweval/runner")


class TestClawEvalConfigDefaults:
    def test_parallel_derived_from_inference_instances(self):
        cfg = ClawEvalEvaluatorConfig()
        _apply_defaults(cfg, 4, "http://localhost:29999")
        assert cfg.parallel == 4 * CLAWEVAL_DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE

    def test_configured_parallel_survives_apply_defaults(self):
        cfg = ClawEvalEvaluatorConfig(parallel=7)
        _apply_defaults(cfg, 4, "http://localhost:29999")
        assert cfg.parallel == 7

    def test_judge_parallel_is_deferred_to_the_scoring_engine(self):
        cfg = ClawEvalEvaluatorConfig()
        _apply_defaults(cfg, 4, "http://localhost:29999")
        assert cfg.judge_parallel is None

    def test_configured_judge_parallel_survives_apply_defaults(self):
        cfg = ClawEvalEvaluatorConfig(judge_parallel=9)
        _apply_defaults(cfg, 4, "http://localhost:29999")
        assert cfg.judge_parallel == 9

    def test_llm_timeout_default_comes_from_code(self):
        cfg = ClawEvalEvaluatorConfig()
        assert cfg.llm_timeout == DEFAULT_REQUEST_TIMEOUT

    def test_llm_retries_default_comes_from_code(self):
        cfg = ClawEvalEvaluatorConfig()
        assert cfg.llm_retries == CLAWEVAL_DEFAULT_LLM_RETRIES
        assert cfg.llm_retries == 20

    def test_timeout_alias_is_accepted(self):
        cfg = ClawEvalEvaluatorConfig(**{"timeout": 42})
        assert cfg.llm_timeout == 42

    def test_retries_alias_is_accepted(self):
        cfg = ClawEvalEvaluatorConfig(**{"retries": 4})
        assert cfg.llm_retries == 4

    def test_base_url_is_taken_from_apply_defaults(self):
        cfg = ClawEvalEvaluatorConfig()
        _apply_defaults(cfg, 1, "http://host:29999")
        assert cfg.require_base_url() == "http://host:29999/v1"

    def test_unknown_search_backend_is_rejected(self):
        with pytest.raises(ValidationError):
            ClawEvalEvaluatorConfig(search_backend="totally-bogus")


class TestClawEvalJudgeModelName:
    def test_judge_is_the_secondary_model(self):
        cfg = ClawEvalEvaluatorConfig(judge={"model_name": "judge-model"})
        assert cfg.secondary_model_names == ["judge-model"]

    def test_rejects_empty_judge_model_name(self):
        with pytest.raises(ValidationError):
            ClawEvalEvaluatorConfig(judge={"model_name": ""})


class TestClawEvalScoringDefaults:
    def test_uses_the_judge_engine_endpoint_and_capacity(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path, judge_instances=2)

        payload = evaluator._build_overrides()
        assert payload["judge"]["api_base"] == "http://judge-router:31000/v1"
        assert (
            payload["judge"]["concurrency"]
            == 2 * CLAWEVAL_DEFAULT_JUDGE_PARALLEL_PER_INFERENCE_INSTANCE
        )

    def test_preserves_explicit_judge_settings(self, tmp_path: Path):
        evaluator = _build_evaluator(
            tmp_path,
            ClawEvalEvaluatorConfig(
                judge={"model_name": "judge-model", "api_base": "http://external/v1"},
                judge_parallel=9,
            ),
            judge_instances=2,
        )

        payload = evaluator._build_overrides()
        assert payload["judge"]["api_base"] == "http://external/v1"
        assert payload["judge"]["concurrency"] == 9

    def test_preserves_explicit_judge_concurrency(self, tmp_path: Path):
        evaluator = _build_evaluator(
            tmp_path,
            ClawEvalEvaluatorConfig(
                judge={"model_name": "judge-model", "concurrency": 3}
            ),
            judge_instances=2,
        )

        payload = evaluator._build_overrides()
        assert payload["judge"]["api_base"] == "http://judge-router:31000/v1"
        assert payload["judge"]["concurrency"] == 3


class TestClawEvalRunnerPython:
    def test_defaults_to_bundled_interpreter(self):
        assert (
            ClawEvalEvaluatorConfig().python_path
            == "/home/user/claweval_env/bin/python"
        )

    def test_uses_config_override(self):
        cfg = ClawEvalEvaluatorConfig(python_path="/opt/py/bin/python")
        assert cfg.python_path == "/opt/py/bin/python"


class TestClawEvalBuildOverrides:
    def test_payload_shape_and_passthrough(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        payload = evaluator._build_overrides()
        assert payload["out_dir"] == str(evaluator.metrics_dir)
        assert payload["model"]["api_base"] == "http://localhost:29999/v1"
        assert payload["model"]["name"] == "hf_ckpt"
        assert payload["judge"]["api_base"] == "http://judge-router:31000/v1"
        assert payload["upstream_root"] == "/home/user/claweval/upstream"
        assert payload["splits"] == ["general", "multi_turn"]
        assert payload["skip_web_real"] is True
        assert payload["search_backend"] == "sber"
        assert payload["trials"] == 3
        assert payload["parallel"] == CLAWEVAL_DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
        assert (
            payload["judge"]["concurrency"]
            == CLAWEVAL_DEFAULT_JUDGE_PARALLEL_PER_INFERENCE_INSTANCE
        )
        assert payload["user_model"] == {}
        assert payload["env"] == {}
        assert "limit" not in payload

    def test_limit_is_passed_through_when_set(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        evaluator.autometric_runs_config.limit = 5
        assert evaluator._build_overrides()["limit"] == 5

    def test_configured_endpoints_do_not_override_the_served_model(
        self, tmp_path: Path
    ):
        evaluator = _build_evaluator(
            tmp_path,
            ClawEvalEvaluatorConfig(
                judge={"model_name": "judge-model", "api_base": "http://judge-gw/v1"},
                model={"api_base": "http://elsewhere/v1"},
            ),
        )
        payload = evaluator._build_overrides()
        assert payload["model"]["api_base"] == "http://localhost:29999/v1"
        assert payload["judge"]["api_base"] == "http://judge-gw/v1"

    def test_llm_timeout_reaches_model_and_judge(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        payload = evaluator._build_overrides()
        assert payload["model"]["timeout"] == DEFAULT_REQUEST_TIMEOUT
        assert payload["judge"]["timeout"] == DEFAULT_REQUEST_TIMEOUT

    def test_configured_llm_timeout_wins(self, tmp_path: Path):
        evaluator = _build_evaluator(
            tmp_path,
            ClawEvalEvaluatorConfig(
                judge={"model_name": "judge-model"}, llm_timeout=42
            ),
        )
        payload = evaluator._build_overrides()
        assert payload["model"]["timeout"] == 42
        assert payload["judge"]["timeout"] == 42

    def test_llm_retries_reach_model_and_judge(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        payload = evaluator._build_overrides()
        assert payload["model"]["max_retries"] == CLAWEVAL_DEFAULT_LLM_RETRIES
        assert payload["judge"]["max_retries"] == CLAWEVAL_DEFAULT_LLM_RETRIES

    def test_configured_llm_retries_win(self, tmp_path: Path):
        evaluator = _build_evaluator(
            tmp_path,
            ClawEvalEvaluatorConfig(judge={"model_name": "judge-model"}, llm_retries=3),
        )
        payload = evaluator._build_overrides()
        assert payload["model"]["max_retries"] == 3
        assert payload["judge"]["max_retries"] == 3

    def test_configured_user_model_gets_the_llm_timeout(self, tmp_path: Path):
        evaluator = _build_evaluator(
            tmp_path,
            ClawEvalEvaluatorConfig(
                judge={"model_name": "judge-model"},
                user_model={"api_base": "http://u/v1"},
            ),
        )
        payload = evaluator._build_overrides()
        assert payload["user_model"]["timeout"] == DEFAULT_REQUEST_TIMEOUT
        assert payload["user_model"]["max_retries"] == CLAWEVAL_DEFAULT_LLM_RETRIES

    def test_explicit_user_model_timeout_wins(self, tmp_path: Path):
        evaluator = _build_evaluator(
            tmp_path,
            ClawEvalEvaluatorConfig(
                judge={"model_name": "judge-model"},
                user_model={"api_base": "http://u/v1", "timeout": 5},
            ),
        )
        assert evaluator._build_overrides()["user_model"]["timeout"] == 5

    def test_explicit_user_model_retries_win(self, tmp_path: Path):
        evaluator = _build_evaluator(
            tmp_path,
            ClawEvalEvaluatorConfig(
                judge={"model_name": "judge-model"},
                user_model={"api_base": "http://u/v1", "max_retries": 0},
            ),
        )
        assert evaluator._build_overrides()["user_model"]["max_retries"] == 0

    def test_explicit_per_block_retries_beat_llm_retries(self, tmp_path: Path):
        evaluator = _build_evaluator(
            tmp_path,
            ClawEvalEvaluatorConfig(
                model={"max_retries": 1},
                judge={"model_name": "judge-model", "max_retries": 2},
            ),
        )
        payload = evaluator._build_overrides()
        assert payload["model"]["max_retries"] == 1
        assert payload["judge"]["max_retries"] == 2

    def test_explicit_per_block_timeouts_beat_llm_timeout(self, tmp_path: Path):
        evaluator = _build_evaluator(
            tmp_path,
            ClawEvalEvaluatorConfig(
                model={"timeout": 5},
                judge={"model_name": "judge-model", "timeout": 6},
            ),
        )
        payload = evaluator._build_overrides()
        assert payload["model"]["timeout"] == 5
        assert payload["judge"]["timeout"] == 6


class TestClawEvalRun:
    def setup_method(self):
        _FakeSubprocessManager.instances = []

    @staticmethod
    def _with_runner(evaluator: ClawEvalEvaluator, runner_root: Path) -> Path:
        runner_root.mkdir(parents=True, exist_ok=True)
        (runner_root / "run_claweval.py").write_text("")
        evaluator._runner_root = runner_root
        return runner_root

    def test_run_raises_when_runner_missing_from_the_image(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        evaluator._runner_root = tmp_path / "does_not_exist"
        with pytest.raises(RuntimeError, match="runner not found in the image"):
            evaluator.run_phase("generation")

    def test_run_success_drives_subprocess(self, tmp_path: Path, monkeypatch):
        evaluator = _build_evaluator(tmp_path)
        runner_root = self._with_runner(evaluator, tmp_path / "runner")
        monkeypatch.setattr(claweval, "SubprocessManager", _FakeSubprocessManager)
        evaluator.run_phase("generation")
        assert len(_FakeSubprocessManager.instances) == 1
        sm = _FakeSubprocessManager.instances[0]
        assert sm.started and sm.waited and sm.logged
        assert sm.cmd[0] == "/home/user/claweval_env/bin/python"
        assert sm.cmd[1] == str(runner_root / "run_claweval.py")
        assert "--phase" in sm.cmd
        assert "generation" in sm.cmd
        assert "--overrides-file" in sm.cmd
        assert sm.overrides_mode == 0o600
        assert sm.cwd == str(runner_root)

    def test_failed_generation_raises(self, tmp_path: Path, monkeypatch):
        evaluator = _build_evaluator(tmp_path)
        self._with_runner(evaluator, tmp_path / "runner")

        class _FailingSubprocessManager(_FakeSubprocessManager):
            def wait_for_completion(self) -> None:
                self.exit_code = 1

        monkeypatch.setattr(claweval, "SubprocessManager", _FailingSubprocessManager)
        with pytest.raises(RuntimeError, match="generation failed with exit code 1"):
            evaluator.run_phase("generation")

    def test_run_does_not_log_the_credentials(self, tmp_path: Path, monkeypatch):
        evaluator = _build_evaluator(tmp_path)
        self._with_runner(evaluator, tmp_path / "runner")
        monkeypatch.setattr(claweval, "SubprocessManager", _FakeSubprocessManager)
        evaluator.autometric_runs_config.model.api_key = "candidate-secret"
        evaluator.autometric_runs_config.judge.api_key = "judge-secret"
        evaluator.autometric_runs_config.env = {"SERP_API_KEY": "env-secret"}
        logged: list[str] = []
        monkeypatch.setattr(
            claweval.Logger, "info", staticmethod(lambda msg: logged.append(str(msg)))
        )
        evaluator.run_phase("generation")
        joined = "\n".join(logged)
        assert "candidate-secret" not in joined
        assert "judge-secret" not in joined
        assert "env-secret" not in joined
        cmd = _FakeSubprocessManager.instances[0].cmd
        assert "candidate-secret" not in " ".join(cmd)
        assert "judge-secret" not in " ".join(cmd)
        assert "env-secret" not in " ".join(cmd)


class TestClawEvalGetMetrics:
    def test_get_metrics_keeps_only_numeric_values(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        metrics_path = evaluator.metrics_dir / "claweval_metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "general/avg_score": 0.75,
                    "multi_turn/pass_at_1": 3,
                    "note": "text",
                    "breakdown": {"a": 1},
                }
            )
        )
        assert evaluator.get_metrics() == {
            "general/avg_score": pytest.approx(0.75),
            "multi_turn/pass_at_1": pytest.approx(3.0),
        }

    def test_get_metrics_raises_when_evaluation_left_no_metrics(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        with pytest.raises(RuntimeError, match="produced no metrics"):
            evaluator.get_metrics()

    def test_get_metrics_raises_when_metrics_dir_unset(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        evaluator.metrics_dir = None
        with pytest.raises(RuntimeError, match="metrics_dir is not set"):
            evaluator.get_metrics()

    def test_get_metrics_raises_on_corrupt_json(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        metrics_path = evaluator.metrics_dir / "claweval_metrics.json"
        metrics_path.write_text("{ this is not valid json")
        with pytest.raises(RuntimeError, match="Failed to read"):
            evaluator.get_metrics()

    def test_get_metrics_raises_when_nothing_is_numeric(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        metrics_path = evaluator.metrics_dir / "claweval_metrics.json"
        metrics_path.write_text(json.dumps({"note": "text"}))
        with pytest.raises(RuntimeError, match="no numeric metrics"):
            evaluator.get_metrics()


class TestJdModelSlug:
    def test_short_name_is_kept_whole(self):
        assert ClawEvalEvaluator._model_slug("minimax-m2.7") == "minimax_m2_7"

    def test_gateway_prefixes_are_trimmed(self):
        assert (
            ClawEvalEvaluator._model_slug("chatwm_openrouter:minimax/minimax-m2")
            == "minimax_m2"
        )


class TestClawEvalCandidateModelName:
    def test_defaults_to_the_checkpoint_directory(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        assert evaluator._build_overrides()["model"]["name"] == "hf_ckpt"

    def test_configured_model_name_wins(self, tmp_path: Path):
        evaluator = _build_evaluator(
            tmp_path,
            ClawEvalEvaluatorConfig(
                judge={"model_name": "judge-model"},
                model={"name": "gigachat-ift"},
            ),
        )
        assert evaluator._build_overrides()["model"]["name"] == "gigachat-ift"


class TestClawEvalJdMetricKeys:
    def test_convert_metrics_to_jd_keys_filters_and_normalizes(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        jd_metrics = evaluator._convert_metrics_to_jd_keys(
            {
                "claweval/general/avg_score": 0.75,
                "multi_turn/pass_at_1": 0.5,
                "general/total_count": 42.0,
            }
        )
        assert jd_metrics == {
            "CLAWEVAL_GENERAL_AVG_SCORE_SBER_JUDGE_MODEL": 0.75,
            "CLAWEVAL_MULTI_TURN_PASS_AT_1_SBER_JUDGE_MODEL": 0.5,
        }

    def test_metric_name_reflects_search_backend(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        evaluator.autometric_runs_config.search_backend = "serp"
        assert (
            evaluator._jd_metric_name("claweval/avg_score")
            == "CLAWEVAL_AVG_SCORE_SERP_JUDGE_MODEL"
        )

    def test_metric_name_omits_the_measured_model(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        evaluator.autometric_runs_config.model.name = "GigaChat-Ultra-3.2"
        assert (
            evaluator._jd_metric_name("claweval/avg_score")
            == "CLAWEVAL_AVG_SCORE_SBER_JUDGE_MODEL"
        )

    def test_two_candidates_on_one_setup_share_the_metric_name(self, tmp_path: Path):
        names = set()
        for candidate in ("gigachat-ultra", "qwen3-235b-a22b"):
            evaluator = _build_evaluator(tmp_path)
            evaluator.autometric_runs_config.model.name = candidate
            names.add(evaluator._jd_metric_name("claweval/avg_score"))
        assert names == {"CLAWEVAL_AVG_SCORE_SBER_JUDGE_MODEL"}

    def test_metric_name_keeps_only_the_judge_model_id(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        evaluator.autometric_runs_config.judge.model_name = (
            "chatwm_openrouter:minimax/minimax-m2"
        )
        assert (
            evaluator._jd_metric_name("claweval/avg_score")
            == "CLAWEVAL_AVG_SCORE_SBER_MINIMAX_M2"
        )

    def test_config_changes_do_not_change_metric_name(self, tmp_path: Path):
        baseline = _build_evaluator(tmp_path / "baseline")
        changed_config = ClawEvalEvaluatorConfig(
            judge={"model_name": "judge-model"},
            splits=["multimodal"],
            skip_web_real=False,
            trials=4,
            limit=2,
            model={"temperature": 0.5, "extra_body": {"top_p": 0.9}},
            user_model={
                "api_base": "http://user-model/v1",
                "model_name": "user-model",
            },
        )
        changed = _build_evaluator(tmp_path / "changed", changed_config)

        baseline_name = baseline._jd_metric_name("claweval/avg_score")
        changed_name = changed._jd_metric_name("claweval/avg_score")

        assert changed_name == baseline_name

    def test_metric_names_fit_the_jd_budget(self, tmp_path: Path):
        keys = [
            f"claweval/{scope}{leaf}"
            for scope in ("", "general/", "multi_turn/", "multimodal/")
            for leaf in (
                "avg_score",
                "pass_hat_10",
                "pass_at_10",
                "n_skipped_web_real",
                "n_skipped_no_user_sim",
                "n_skipped_no_sandbox",
                "rollout_error_rate",
                "judge_error_rate",
            )
        ]
        for search_backend in ("sber", "serp"):
            evaluator = _build_evaluator(tmp_path)
            evaluator.autometric_runs_config.judge.model_name = (
                "Qwen3-235B-A22B-Instruct-2507-FP8"
            )
            evaluator.autometric_runs_config.search_backend = search_backend
            jd_metrics = evaluator._convert_metrics_to_jd_keys(dict.fromkeys(keys, 1.0))
            assert len(jd_metrics) == len(keys)
            assert max(len(name) for name in jd_metrics) <= 70


class TestClawEvalShooterVersion:
    def test_version_is_the_upstream_pin_from_the_image(
        self, tmp_path: Path, monkeypatch
    ):
        _with_version_file(tmp_path, monkeypatch)
        assert ClawEvalEvaluator._shooter_version() == f"vendored-{UPSTREAM_PIN[:12]}"

    def test_version_raises_when_the_pin_file_is_missing(
        self, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setattr(claweval, "Path", lambda _: tmp_path / "absent.txt")
        with pytest.raises(RuntimeError, match="cannot read the upstream pin"):
            ClawEvalEvaluator._shooter_version()

    def test_version_raises_when_the_pin_file_is_empty(
        self, tmp_path: Path, monkeypatch
    ):
        _with_version_file(tmp_path, monkeypatch, pin="")
        with pytest.raises(RuntimeError, match="pin file is empty"):
            ClawEvalEvaluator._shooter_version()


class TestClawEvalSendMetricsToJd:
    @pytest.mark.usefixtures("scalar_ci_upload")
    def test_skips_when_metrics_dir_unset(self, tmp_path: Path):
        evaluator = _build_evaluator(tmp_path)
        evaluator.metrics_dir = None
        evaluator._send_metrics_to_jd({"general/avg_score": 0.5})

    @pytest.mark.usefixtures("scalar_ci_upload")
    def test_full_pipeline_builds_body_and_posts(self, tmp_path: Path, monkeypatch):
        evaluator = _build_evaluator(tmp_path)
        _with_version_file(tmp_path, monkeypatch)
        posted: dict = {}
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.send_metrics",
            lambda body, artifacts_path: posted.update(body),
        )
        evaluator._send_metrics_to_jd({"general/avg_score": 0.9, "note/other": 1.0})
        assert posted["shooter_name"] == "claweval"
        assert posted["shooter_version"] == f"vendored-{UPSTREAM_PIN[:12]}"
        assert posted["model_class"] == "sft"
        saved = list(evaluator.metrics_dir.glob("metrics_step_*.json"))
        assert saved
        body_on_disk = json.loads(saved[0].read_text())
        assert (
            body_on_disk["metrics"]["0"]["CLAWEVAL_GENERAL_AVG_SCORE_SBER_JUDGE_MODEL"]
            == 0.9
        )


class TestClawEvalAfterRun:
    def test_after_run_invokes_full_pipeline(self, tmp_path: Path, monkeypatch):
        evaluator = _build_evaluator(tmp_path)
        seen: dict = {}
        tar_path = tmp_path / "m.tar"

        def fake_send(metrics):
            seen["send"] = metrics

        def fake_tar(metrics_dir, artifacts_path, evaluator_name):
            seen["tar"] = (metrics_dir, artifacts_path)
            return tar_path

        def fake_symlink(archive_path, checkpoint_path, evaluator_name):
            seen["link"] = (archive_path, checkpoint_path)

        monkeypatch.setattr(evaluator, "_send_metrics_to_jd", fake_send)
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.tar_metrics_dir",
            fake_tar,
        )
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.symlink_metrics_for_transfer",
            fake_symlink,
        )
        evaluator.after_run({"general/avg_score": 0.5})
        assert seen["send"] == {"general/avg_score": 0.5}
        assert seen["tar"][0] == evaluator.metrics_dir
        assert seen["link"][0] == tar_path


@pytest.mark.parametrize(
    "case", ["complete", "duplicate", "ungraded", "missing_trial", "empty", "malformed"]
)
def test_ci_averages_trials_and_preserves_split_populations(tmp_path, case):
    evaluator = _build_evaluator(tmp_path)
    manifest = {
        "trials": 2,
        "selected": ["a", "b"],
        "content_sha256": "content",
        "split_of": {"a": "general", "b": "general"},
    }
    rows = [
        {
            "task_id": uid,
            "trial_index": trial,
            "status": "graded",
            "task_score": float(uid == "a"),
            "passed": uid == "a",
        }
        for uid in ("a", "b")
        for trial in (0, 1)
    ]
    if case == "duplicate":
        rows.append(rows[0])
    elif case == "ungraded":
        rows[0]["status"] = "error"
    elif case == "missing_trial":
        rows.pop()
    elif case == "empty":
        manifest["selected"], rows = [], []
    elif case == "malformed":
        rows = [{}]
    (evaluator.metrics_dir / "generation_manifest.json").write_text(
        json.dumps(manifest)
    )
    (evaluator.metrics_dir / "claweval_results.json").write_text(json.dumps(rows))
    metrics = {
        evaluator._jd_metric_name(f"{prefix}/{metric}"): 0.5
        for prefix in ("claweval", "claweval/general")
        for metric in ("avg_score", "pass_hat_2", "pass_at_2")
    }
    scores = evaluator._cache_scores(metrics)
    assert set(scores.records) == (set(metrics) if case == "complete" else set())
    for rows in scores.records.values():
        assert [row["score"] for row in rows] == [1, 0]
