import json
from pathlib import Path

import pytest

from node_runner.src.config_model import ModelInfoConfig, SweBenchEvaluatorConfig
from node_runner.src.logger import Logger
from node_runner.src.metrics.evaluators import swe_bench as swe_bench_module
from node_runner.src.metrics.evaluators.swe_bench import SweBenchEvaluator

pytestmark = pytest.mark.unit


@pytest.fixture
def model_info() -> ModelInfoConfig:
    return ModelInfoConfig(model_class="sft", size="lite", username="u")


@pytest.fixture
def evaluator_factory(tmp_path: Path, model_info: ModelInfoConfig):
    def _build(
        config_payload: SweBenchEvaluatorConfig | dict | None = None,
    ) -> SweBenchEvaluator:
        evaluator = SweBenchEvaluator()
        if isinstance(config_payload, SweBenchEvaluatorConfig):
            config = config_payload
        else:
            config = SweBenchEvaluatorConfig.model_validate(
                config_payload
                or {
                    "n-concurrent": 3,
                    "agent": "mini-swe-agent",
                }
            )
        if config.model_api_base is None:
            config.apply_defaults(
                inference_instances=1,
                base_url="http://127.0.0.1:30000",
            )
        evaluator.setup(
            config,
            artifacts_path=tmp_path / "artifacts",
            hf_checkpoint_path=tmp_path / "model" / "global_step_7",
            model_info=model_info,
        )
        return evaluator

    return _build


class TestConfigPayload:
    def test_scales_default_n_concurrent(self) -> None:
        cfg = SweBenchEvaluatorConfig.model_validate({})

        cfg.apply_defaults(
            inference_instances=4,
            base_url="http://127.0.0.1:31234",
        )

        assert cfg.n_concurrent == 32

    def test_applies_proxy_port_and_defaults(self) -> None:
        cfg = SweBenchEvaluatorConfig.model_validate(
            {"n-concurrent": 1, "agent": "mini-swe-agent"}
        )
        cfg.apply_defaults(
            inference_instances=2,
            base_url="http://127.0.0.1:31234",
        )

        payload = cfg.to_dict()

        assert payload["model-api-base"] == "http://127.0.0.1:31234/v1"
        assert payload["n-concurrent"] == 1
        assert payload["pass-at-k"] == 1
        assert payload["agent"] == "mini-swe-agent"
        assert "agent-timeout" not in payload
        assert payload["dataset"] == (
            "/home/user/swe-bench-verified/datasets/swebench-verified"
        )
        assert payload["env-prefix"] == "swe-bench"
        assert payload["env-version"] == "1.0.2"
        assert payload["request-retries"] == 1
        assert payload["create-session-jitter-max-sec"] == 180.0
        assert "api-port" not in payload
        assert "venv-python" not in payload
        assert "venv_python" not in payload
        assert "force-recalc" not in payload

    def test_preserves_external_model_api_base(self) -> None:
        cfg = SweBenchEvaluatorConfig.model_validate(
            {"n-concurrent": 1, "model-api-base": "https://api.external.test/v1"}
        )
        cfg.apply_defaults(
            inference_instances=2,
            base_url="http://127.0.0.1:31234",
        )

        payload = cfg.to_dict()

        assert payload["model-api-base"] == "https://api.external.test/v1"

    def test_serializes_pass_at_k_alias(self) -> None:
        cfg = SweBenchEvaluatorConfig.model_validate({"pass_at_k": 3})

        assert cfg.to_dict()["pass-at-k"] == 3

    def test_preserves_explicit_env_api_base(self) -> None:
        cfg = SweBenchEvaluatorConfig.model_validate(
            {"env-api-base": "https://rli.example.test"}
        )

        assert cfg.to_dict()["env-api-base"] == "https://rli.example.test"


class TestBuildCommand:
    def test_uses_cli_module_and_metrics_dir(self, evaluator_factory) -> None:
        cfg = SweBenchEvaluatorConfig.model_validate(
            {"n-concurrent": 1, "venv_python": "python"}
        )
        evaluator = evaluator_factory(cfg)

        cmd = evaluator._build_cmd()

        assert cmd[:4] == ["python", "-m", "swe_bench_harbor_rli.cli", "run"]
        assert "--metrics-dir" in cmd
        assert str(evaluator.metrics_dir) in cmd

    def test_rejects_missing_model_api_base(self, evaluator_factory) -> None:
        evaluator = evaluator_factory()
        evaluator.autometric_runs_config.model_api_base = None

        with pytest.raises(
            ValueError,
            match="model_api_base must be set before run",
        ):
            evaluator._build_cmd()


class TestOfferedLoad:
    def test_uses_configured_concurrency(self, evaluator_factory) -> None:
        evaluator = evaluator_factory({"n-concurrent": 7})

        assert evaluator.offered_load == 7


class TestBuildEnv:
    def test_points_pythonpath_at_repo_gigarli_and_contrib(
        self,
        evaluator_factory,
        monkeypatch,
    ) -> None:
        monkeypatch.delenv("PYTHONPATH", raising=False)
        evaluator = evaluator_factory()
        monkeypatch.setattr(
            evaluator,
            "get_rli_url",
            lambda: "https://gym.static.space",
        )

        env = evaluator._build_env()
        pythonpath = env["PYTHONPATH"].split(":")

        assert pythonpath[0].endswith("/contrib/gigarli/src")
        assert pythonpath[1].endswith("/contrib")
        assert len(pythonpath) == 2
        assert env["RLI_API_BASE"] == "https://gym.static.space"

    def test_appends_to_existing_pythonpath(
        self,
        evaluator_factory,
        monkeypatch,
    ) -> None:
        monkeypatch.setenv("PYTHONPATH", "/existing/path")
        evaluator = evaluator_factory()
        monkeypatch.setattr(
            evaluator,
            "get_rli_url",
            lambda: "https://rli.sberdevices.ru",
        )

        env = evaluator._build_env()
        pythonpath = env["PYTHONPATH"].split(":")

        assert pythonpath[0].endswith("/contrib/gigarli/src")
        assert pythonpath[1].endswith("/contrib")
        assert pythonpath[-1] == "/existing/path"
        assert len(pythonpath) == 3
        assert env["RLI_API_BASE"] == "https://rli.sberdevices.ru"


class TestRun:
    def test_skips_when_already_calculated(
        self, evaluator_factory, monkeypatch
    ) -> None:
        evaluator = evaluator_factory()
        evaluator.metrics_path.write_text(
            json.dumps({"pass_rate": 0.5}), encoding="utf-8"
        )

        def _fail(*args, **kwargs):
            raise AssertionError("SubprocessManager must not run when calculated")

        monkeypatch.setattr(swe_bench_module, "SubprocessManager", _fail)

        evaluator.run()

    def test_runs_subprocess_manager(self, evaluator_factory, monkeypatch) -> None:
        evaluator = evaluator_factory()
        calls = {}

        class FakeSubprocessManager:
            def __init__(self, cmd, log, cwd, run_name, env):
                calls["init"] = {
                    "cmd": cmd,
                    "log": log,
                    "cwd": cwd,
                    "run_name": run_name,
                    "env": env,
                }

            def start(self):
                calls["started"] = True

            def wait_for_completion(self):
                calls["waited"] = True

            def log_error(self):
                calls["logged"] = True

        monkeypatch.setattr(
            swe_bench_module, "SubprocessManager", FakeSubprocessManager
        )

        evaluator.run()

        assert calls["init"]["cmd"][:4] == [
            evaluator.autometric_runs_config.venv_python,
            "-m",
            "swe_bench_harbor_rli.cli",
            "run",
        ]
        assert calls["init"]["cwd"] == str(evaluator.node_runner_root)
        assert calls["init"]["log"] == evaluator.log_file
        assert calls["init"]["run_name"] == "swe_bench"
        assert "PYTHONPATH" in calls["init"]["env"]
        assert calls["started"] is True
        assert calls["waited"] is True
        assert calls["logged"] is True

    def test_force_recalc_reruns_even_when_file_exists(
        self, evaluator_factory, monkeypatch
    ) -> None:
        cfg = SweBenchEvaluatorConfig.model_validate(
            {"n-concurrent": 1, "force-recalc": True}
        )
        evaluator = evaluator_factory(cfg)
        evaluator.metrics_path.write_text(
            json.dumps({"pass_rate": 0.5}), encoding="utf-8"
        )
        calls = {}

        class FakeSubprocessManager:
            def __init__(self, *args, **kwargs):
                calls["init"] = True

            def start(self):
                calls["started"] = True

            def wait_for_completion(self):
                calls["waited"] = True

            def log_error(self):
                calls["logged"] = True

        monkeypatch.setattr(
            swe_bench_module, "SubprocessManager", FakeSubprocessManager
        )

        evaluator.run()

        assert calls == {
            "init": True,
            "started": True,
            "waited": True,
            "logged": True,
        }


class TestPostReverify:
    @staticmethod
    def _cfg() -> SweBenchEvaluatorConfig:
        return SweBenchEvaluatorConfig.model_validate(
            {"n-concurrent": 1, "post-reverify": True, "venv_python": "python"}
        )

    def test_build_cmd_sets_skip_primary_run_flag(self, evaluator_factory) -> None:
        evaluator = evaluator_factory(self._cfg())

        cmd = evaluator._build_cmd(skip_primary_run=True)
        config = json.loads(cmd[-1])

        assert config["skip-primary-run"] is True

    def test_is_already_calculated_false_for_post_reverify_only(
        self, evaluator_factory
    ) -> None:
        evaluator = evaluator_factory(self._cfg())
        evaluator.metrics_path.write_text(
            json.dumps({"pass_rate": 0.5}), encoding="utf-8"
        )

        assert evaluator._should_run_post_reverify_only() is True
        assert evaluator._is_already_calculated() is False

    def test_run_triggers_post_reverify_only(
        self, evaluator_factory, monkeypatch
    ) -> None:
        evaluator = evaluator_factory(self._cfg())
        evaluator.metrics_path.write_text(
            json.dumps({"pass_rate": 0.5}), encoding="utf-8"
        )
        calls = {}
        info_messages = []
        monkeypatch.setattr(
            Logger,
            "info",
            staticmethod(lambda message: info_messages.append(message)),
        )

        class FakeSubprocessManager:
            def __init__(self, cmd, *args, **kwargs):
                calls["cmd"] = cmd

            def start(self):
                calls["started"] = True

            def wait_for_completion(self):
                calls["waited"] = True

            def log_error(self):
                calls["logged"] = True

        monkeypatch.setattr(
            swe_bench_module, "SubprocessManager", FakeSubprocessManager
        )

        evaluator.run()

        assert json.loads(calls["cmd"][-1])["skip-primary-run"] is True
        assert any("Starting SWE-bench post-reverify" in msg for msg in info_messages)


class TestGetMetrics:
    def test_returns_named_metrics(self, evaluator_factory) -> None:
        evaluator = evaluator_factory()
        evaluator.metrics_path.write_text(
            json.dumps(
                {
                    "metrics": {
                        "SWE_BENCH_VERIFIED_MINI_SWE_AGENT_MAX_TOKENS_16384_TIMEOUT_14400": 0.375,
                        "metadata": "ignored",
                    }
                }
            ),
            encoding="utf-8",
        )

        assert evaluator.get_metrics() == {
            "SWE_BENCH_VERIFIED_MINI_SWE_AGENT_MAX_TOKENS_16384_TIMEOUT_14400": 0.375
        }

    def test_returns_resolved_rate(self, evaluator_factory) -> None:
        evaluator = evaluator_factory()
        evaluator.metrics_path.write_text(
            json.dumps({"pass_rate": 0.375}),
            encoding="utf-8",
        )

        assert evaluator.get_metrics() == {"swe_bench_verified/resolved": 0.375}

    def test_returns_pass_at_k(self, evaluator_factory) -> None:
        evaluator = evaluator_factory()
        evaluator.metrics_path.write_text(
            json.dumps(
                {
                    "pass_rate": 0.375,
                    "pass_at_k": {"k": 3, "score": 0.625, "complete": True},
                }
            ),
            encoding="utf-8",
        )

        assert evaluator.get_metrics() == {
            "swe_bench_verified/resolved": 0.375,
            "swe_bench_verified/pass@3": 0.625,
        }

    def test_does_not_duplicate_pass_at_1(self, evaluator_factory) -> None:
        evaluator = evaluator_factory()
        evaluator.metrics_path.write_text(
            json.dumps(
                {
                    "pass_rate": 0.375,
                    "pass_at_k": {"k": 1, "score": 0.375, "complete": True},
                }
            ),
            encoding="utf-8",
        )

        assert evaluator.get_metrics() == {"swe_bench_verified/resolved": 0.375}

    def test_ignores_missing_or_invalid_file(self, evaluator_factory) -> None:
        evaluator = evaluator_factory()

        assert evaluator.get_metrics() == {}

        evaluator.metrics_path.write_text(
            json.dumps({"pass_rate": "bad"}),
            encoding="utf-8",
        )
        assert evaluator.get_metrics() == {}


class TestSendMetricsToJd:
    @pytest.mark.usefixtures("scalar_ci_upload")
    def test_builds_and_sends_payload(self, evaluator_factory, monkeypatch) -> None:
        evaluator = evaluator_factory()
        saved = {}
        info_messages = []

        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.send_metrics",
            lambda body, artifacts_path: saved.setdefault("request_body", body),
        )
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.save_metrics_json",
            lambda body, step, metrics_dir: saved.update(
                {"saved_body": body, "saved_step": step, "saved_dir": metrics_dir}
            ),
        )
        monkeypatch.setattr(
            Logger,
            "info",
            staticmethod(lambda message: info_messages.append(message)),
        )

        evaluator._send_metrics_to_jd({"swe_bench_verified/resolved": 0.227})

        assert "request_body" in saved
        assert saved["request_body"]["shooter_name"] == "swe_bench"
        assert saved["request_body"]["metrics"] == {
            7: {"SWE_BENCH_VERIFIED_RESOLVED": 0.227}
        }
        assert saved["saved_step"] == 7

    @pytest.mark.usefixtures("scalar_ci_upload")
    def test_skips_when_hf_checkpoint_path_missing(
        self, evaluator_factory, monkeypatch
    ) -> None:
        evaluator = evaluator_factory()
        evaluator.hf_checkpoint_path = None
        warnings = []
        monkeypatch.setattr(
            Logger,
            "warning",
            staticmethod(lambda message: warnings.append(message)),
        )

        evaluator._send_metrics_to_jd({"swe_bench_verified/resolved": 0.227})

        assert warnings == ["hf_checkpoint_path is not set, skipping JD metrics send"]

    @pytest.mark.usefixtures("scalar_ci_upload")
    def test_skips_when_metrics_dir_missing(
        self, evaluator_factory, monkeypatch
    ) -> None:
        evaluator = evaluator_factory()
        evaluator.metrics_dir = None
        warnings = []
        monkeypatch.setattr(
            Logger,
            "warning",
            staticmethod(lambda message: warnings.append(message)),
        )

        evaluator._send_metrics_to_jd({"swe_bench_verified/resolved": 0.227})

        assert warnings == ["metrics_dir is not set, skipping JD metrics send"]


class TestAfterRun:
    def test_sends_and_transfers_metrics(
        self, evaluator_factory, monkeypatch, tmp_path
    ) -> None:
        evaluator = evaluator_factory()
        evaluator.metrics_path.write_text(
            json.dumps({"completed": 500}),
            encoding="utf-8",
        )
        calls = {}

        monkeypatch.setattr(
            evaluator,
            "_send_metrics_to_jd",
            lambda metrics: calls.setdefault("metrics", metrics),
        )

        def fake_tar_metrics_dir(metrics_dir, artifacts_path, evaluator_name):
            calls["tar"] = (metrics_dir, artifacts_path)
            return tmp_path / "swe_bench_artifacts.tar.zst"

        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.tar_metrics_dir",
            fake_tar_metrics_dir,
        )
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.symlink_metrics_for_transfer",
            lambda tar_path, hf_checkpoint_path, evaluator_name: calls.setdefault(
                "symlink",
                (
                    tar_path,
                    hf_checkpoint_path,
                ),
            ),
        )

        evaluator.after_run({"swe_bench_verified/resolved": 0.5})

        assert calls["metrics"] == {"swe_bench_verified/resolved": 0.5}
        assert calls["tar"] == (evaluator.metrics_dir, evaluator.artifacts_path)
        assert calls["symlink"] == (
            tmp_path / "swe_bench_artifacts.tar.zst",
            evaluator.hf_checkpoint_path,
        )

    @pytest.mark.parametrize(
        "metrics_payload",
        [
            {"completed": 499},
            {"completed": 500, "rli_infrastructure_errors": ["ReadError"]},
            {
                "completed": 1500,
                "pass_at_k": {"k": 3, "score": 0.5, "complete": False},
            },
        ],
        ids=["incomplete-run", "rli-error", "incomplete-pass-at-k"],
    )
    def test_skips_jd_metrics(
        self, evaluator_factory, monkeypatch, tmp_path, metrics_payload
    ) -> None:
        evaluator = evaluator_factory()
        evaluator.metrics_path.write_text(
            json.dumps(metrics_payload),
            encoding="utf-8",
        )
        calls = []

        monkeypatch.setattr(
            evaluator,
            "_send_metrics_to_jd",
            lambda metrics: calls.append(metrics),
        )
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.tar_metrics_dir",
            lambda metrics_dir, artifacts_path, evaluator_name: (
                tmp_path / "metrics.tar.zst"
            ),
        )
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.JD.symlink_metrics_for_transfer",
            lambda tar_path, hf_checkpoint_path, evaluator_name: None,
        )

        evaluator.after_run({"swe_bench_verified/resolved": 0.5})

        assert calls == []


@pytest.mark.usefixtures("scalar_ci_upload")
def test_ci_exports_task_means_and_observed_pass_at_k(
    evaluator_factory, harbor_job, monkeypatch
):
    from node_runner.src.job_dispatcher import JD

    evaluator = evaluator_factory({"pass_at_k": 2})
    path = harbor_job({"a": [1, 0], "b": [0, 0]})
    evaluator.metrics_path.write_text(
        json.dumps({"job_result_path": str(path), "total": 4})
    )
    monkeypatch.setattr(JD, "can_send_metrics", lambda **kwargs: True)
    monkeypatch.setattr(JD, "send_evaluator_metrics", lambda **kwargs: None)
    evaluator._send_metrics_to_jd({"quality/mean": 0.25, "quality/pass_at_2": 0.5})
    rows = [
        json.loads(line)
        for line in (evaluator.metrics_dir / "ci_samples.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [(row["benchmark"], row["score"]) for row in rows] == [
        ("QUALITY_MEAN", 0.5),
        ("QUALITY_MEAN", 0),
        ("QUALITY_PASS_AT_2", 1),
        ("QUALITY_PASS_AT_2", 0),
    ]
