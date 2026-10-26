import json
from pathlib import Path

import pytest
from node_runner.src.job_dispatcher import JD

from node_runner.src.logger import Logger
from node_runner.src.config_model import ModelInfoConfig, TerminalBenchEvaluatorConfig
from node_runner.src.metrics.evaluators import terminal_bench as terminal_bench_module
from node_runner.src.metrics.evaluators.terminal_bench import TerminalBenchEvaluator

pytestmark = pytest.mark.unit


@pytest.fixture
def model_info() -> ModelInfoConfig:
    return ModelInfoConfig(model_class="sft", size="lite", username="u")


@pytest.fixture
def evaluator_factory(tmp_path: Path, model_info: ModelInfoConfig):
    def _build(
        config_payload: TerminalBenchEvaluatorConfig | dict | None = None,
    ) -> TerminalBenchEvaluator:
        evaluator = TerminalBenchEvaluator()
        if isinstance(config_payload, TerminalBenchEvaluatorConfig):
            config = config_payload
        else:
            config = TerminalBenchEvaluatorConfig.model_validate(
                config_payload
                or {
                    "n-concurrent": 3,
                    "task-name": "adaptive-rejection-sampler",
                    "env-api-base": "https://rli.example.test",
                }
            )
        config.apply_defaults(
            inference_instances=1,
            base_url="http://127.0.0.1:29999",
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
    def test_applies_proxy_port_and_overrides(self) -> None:
        cfg = TerminalBenchEvaluatorConfig.model_validate(
            {"n-concurrent": 1, "agent": "terminus-2"}
        )
        cfg.apply_defaults(
            inference_instances=1,
            base_url="http://127.0.0.1:31234",
        )

        payload = cfg.to_dict()

        assert payload["model-api-base"] == "http://127.0.0.1:31234/v1"
        assert payload["n-concurrent"] == 1
        assert payload["pass-at-k"] == 1
        assert payload["agent"] == "terminus-2"
        assert "agent-timeout" not in payload
        assert payload["max-tokens"] == 32768
        assert payload["dataset"] == "/home/user/terminal-bench-2"
        assert payload["env-prefix"] == "terminal-bench"
        assert payload["request-retries"] == 1
        assert payload["transfer-retries"] == 10
        assert payload["enable-probe-warmup"] is True
        assert payload["create-session-jitter-max-sec"] == 180.0
        assert payload["probe-retry-backoff-sec"] == 2.0
        assert "api-port" not in payload
        assert "venv-python" not in payload
        assert "venv_python" not in payload

    def test_preserves_explicit_startup_hardening_overrides(self) -> None:
        cfg = TerminalBenchEvaluatorConfig.model_validate(
            {
                "n-concurrent": 1,
                "request-retries": 5,
                "enable-probe-warmup": False,
                "create-session-jitter-max-sec": 1.5,
                "probe-retry-backoff-sec": 4.0,
            }
        )
        cfg.apply_defaults(
            inference_instances=1,
            base_url="http://127.0.0.1:31234",
        )

        payload = cfg.to_dict()

        assert payload["request-retries"] == 5
        assert payload["enable-probe-warmup"] is False
        assert payload["create-session-jitter-max-sec"] == 1.5
        assert payload["probe-retry-backoff-sec"] == 4.0

    def test_preserves_external_model_api_base(self) -> None:
        cfg = TerminalBenchEvaluatorConfig.model_validate(
            {"n-concurrent": 1, "model-api-base": "https://api.external.test/v1"}
        )
        cfg.apply_defaults(
            inference_instances=1,
            base_url="http://127.0.0.1:31234",
        )

        payload = cfg.to_dict()

        assert payload["model-api-base"] == "https://api.external.test/v1"

    def test_serializes_pass_at_k_alias(self) -> None:
        cfg = TerminalBenchEvaluatorConfig.model_validate({"pass_at_k": 3})

        assert cfg.to_dict()["pass-at-k"] == 3

    def test_preserves_explicit_env_api_base(self) -> None:
        cfg = TerminalBenchEvaluatorConfig.model_validate(
            {"env-api-base": "https://rli.example.test"}
        )

        assert cfg.to_dict()["env-api-base"] == "https://rli.example.test"


class TestBuildCommand:
    def test_requires_model_api_base(self, evaluator_factory) -> None:
        cfg = TerminalBenchEvaluatorConfig.model_validate({"n-concurrent": 1})
        evaluator = evaluator_factory(cfg)
        evaluator.autometric_runs_config.model_api_base = None

        with pytest.raises(ValueError, match="model_api_base must be set"):
            evaluator._build_cmd()

    def test_uses_cli_module_and_metrics_dir(self, evaluator_factory) -> None:
        cfg = TerminalBenchEvaluatorConfig.model_validate(
            {"n-concurrent": 1, "venv_python": "python"}
        )
        evaluator = evaluator_factory(cfg)

        cmd = evaluator._build_cmd()

        assert cmd[:4] == ["python", "-m", "terminal_bench.cli", "run"]
        assert "--metrics-dir" in cmd
        assert str(evaluator.metrics_dir) in cmd


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
    def test_run_uses_current_no_arg_contract(
        self, evaluator_factory, monkeypatch
    ) -> None:
        evaluator = evaluator_factory()
        calls = []

        class FakeSubprocessManager:
            def __init__(self, *args, **kwargs):
                calls.append("init")

            def start(self):
                calls.append("start")

            def wait_for_completion(self):
                calls.append("wait")

            def log_error(self):
                calls.append("log_error")

        monkeypatch.setattr(
            terminal_bench_module, "SubprocessManager", FakeSubprocessManager
        )

        evaluator.run()

        assert calls == ["init", "start", "wait", "log_error"]

    def test_skips_when_already_calculated(
        self, evaluator_factory, monkeypatch, tmp_path
    ) -> None:
        evaluator = evaluator_factory()
        evaluator.metrics_path.write_text(
            json.dumps({"pass_rate": 0.5}), encoding="utf-8"
        )

        def _fail(*args, **kwargs):
            raise AssertionError("SubprocessManager must not run when calculated")

        monkeypatch.setattr(terminal_bench_module, "SubprocessManager", _fail)

        evaluator.run()

    def test_runs_subprocess_manager(
        self, evaluator_factory, monkeypatch, tmp_path
    ) -> None:
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
                calls["raised"] = True

        monkeypatch.setattr(
            terminal_bench_module, "SubprocessManager", FakeSubprocessManager
        )

        evaluator.run()

        assert calls["init"]["cmd"][:4] == [
            evaluator.autometric_runs_config.venv_python,
            "-m",
            "terminal_bench.cli",
            "run",
        ]
        assert calls["init"]["cwd"] == str(evaluator.node_runner_root)
        assert calls["init"]["run_name"] == "terminal_bench"
        assert "PYTHONPATH" in calls["init"]["env"]
        assert calls["started"] is True
        assert calls["waited"] is True
        assert calls["raised"] is True


class TestGetMetrics:
    def test_returns_named_metrics(self, evaluator_factory) -> None:
        evaluator = evaluator_factory()
        evaluator.metrics_path.write_text(
            json.dumps(
                {
                    "metrics": {
                        "TERMINAL_BENCH_TERMINUS_2_MAX_TOKENS_16384_TIMEOUT_14400": 0.375,
                        "metadata": "ignored",
                    }
                }
            ),
            encoding="utf-8",
        )

        assert evaluator.get_metrics() == {
            "TERMINAL_BENCH_TERMINUS_2_MAX_TOKENS_16384_TIMEOUT_14400": 0.375
        }

    def test_returns_pass_rate(self, evaluator_factory) -> None:
        evaluator = evaluator_factory()
        evaluator.metrics_path.write_text(
            json.dumps({"pass_rate": 0.375}),
            encoding="utf-8",
        )

        assert evaluator.get_metrics() == {"terminal_bench/resolved_percent": 0.375}

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
            "terminal_bench/resolved_percent": 0.375,
            "terminal_bench/pass@3": 0.625,
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

        assert evaluator.get_metrics() == {"terminal_bench/resolved_percent": 0.375}

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

        monkeypatch.setattr(
            JD,
            "send_metrics",
            lambda body, _artifacts: saved.setdefault("request_body", body),
        )
        monkeypatch.setattr(
            JD,
            "save_metrics_json",
            lambda body, step, metrics_dir: saved.update(
                {"saved_body": body, "saved_step": step, "saved_dir": metrics_dir}
            ),
        )

        evaluator._send_metrics_to_jd({"terminal_bench/resolved_percent": 22.7})

        assert "request_body" in saved
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

        evaluator._send_metrics_to_jd({"terminal_bench/resolved_percent": 22.7})

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

        evaluator._send_metrics_to_jd({"terminal_bench/resolved_percent": 22.7})

        assert warnings == ["metrics_dir is not set, skipping JD metrics send"]


class TestAfterRun:
    def test_sends_and_transfers_metrics(
        self, evaluator_factory, monkeypatch, tmp_path
    ) -> None:
        evaluator = evaluator_factory()
        evaluator.metrics_path.write_text("{}", encoding="utf-8")
        calls = {}

        monkeypatch.setattr(
            evaluator,
            "_send_metrics_to_jd",
            lambda metrics: calls.setdefault("metrics", metrics),
        )

        def fake_tar_metrics_dir(metrics_dir, artifacts_path, _name):
            calls["tar"] = (metrics_dir, artifacts_path)
            return tmp_path / "terminal_bench_artifacts.tar.zst"

        monkeypatch.setattr(JD, "tar_metrics_dir", fake_tar_metrics_dir)
        monkeypatch.setattr(
            JD,
            "symlink_metrics_for_transfer",
            lambda tar_path, hf_checkpoint_path, _name: calls.setdefault(
                "symlink", (tar_path, hf_checkpoint_path)
            ),
        )

        evaluator.after_run({"terminal_bench/resolved_percent": 50.0})

        assert calls["metrics"] == {"terminal_bench/resolved_percent": 50.0}
        assert calls["tar"] == (
            evaluator.metrics_dir,
            evaluator.artifacts_path,
        )
        assert calls["symlink"] == (
            tmp_path / "terminal_bench_artifacts.tar.zst",
            evaluator.hf_checkpoint_path,
        )

    def test_skips_jd_for_incomplete_pass_at_k(
        self, evaluator_factory, monkeypatch, tmp_path
    ) -> None:
        evaluator = evaluator_factory()
        evaluator.metrics_path.write_text(
            json.dumps({"pass_at_k": {"k": 3, "score": 0.5, "complete": False}}),
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

        evaluator.after_run({"terminal_bench/resolved_percent": 0.5})

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
