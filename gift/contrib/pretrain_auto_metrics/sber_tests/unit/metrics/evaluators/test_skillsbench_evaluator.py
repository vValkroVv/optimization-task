import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from node_runner.src.config_model import (
    DEFAULT_REQUEST_TIMEOUT,
    ModelInfoConfig,
    SkillsBenchEvaluatorConfig,
    SweBenchEvaluatorConfig,
)
from node_runner.src.job_dispatcher import JD
from node_runner.src.metrics.evaluators import skillsbench as skillsbench_module
from node_runner.src.metrics.evaluators.skillsbench import SkillsBenchEvaluator


pytestmark = pytest.mark.unit


@pytest.fixture
def evaluator_factory(tmp_path: Path):
    def build(*, apply_defaults: bool = True, **overrides) -> SkillsBenchEvaluator:
        config = SkillsBenchEvaluatorConfig(**overrides)
        if apply_defaults:
            config.apply_defaults(
                inference_instances=2,
                base_url="http://localhost:29999",
            )
        evaluator = SkillsBenchEvaluator()
        evaluator.setup(
            autometric_runs_config=config,
            artifacts_path=tmp_path / "artifacts",
            hf_checkpoint_path=tmp_path / "model" / "global_step_7",
            model_info=ModelInfoConfig(
                model_class="sft",
                size="10b",
                username="max",
            ),
        )
        return evaluator

    return build


def write_metrics(
    evaluator: SkillsBenchEvaluator,
    *,
    complete: bool = True,
    publish_metrics: bool | None = None,
) -> None:
    evaluator.metrics_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "skillsbench/no_skill/avg": 0.25,
                    "skillsbench/no_skill/pass@3": 0.5,
                    "skillsbench/skill_lift/avg": 0.25,
                    "ignored": "text",
                },
                "conditions": {
                    "no-skill": {
                        "rli_errors": 0,
                        "api_errors": 0,
                    }
                },
                "complete": complete,
                "publish_metrics": (
                    complete if publish_metrics is None else publish_metrics
                ),
            }
        ),
        encoding="utf-8",
    )


def write_fingerprint(evaluator: SkillsBenchEvaluator) -> None:
    evaluator.fingerprint_path.write_text(
        json.dumps(evaluator._fingerprint()),
        encoding="utf-8",
    )


class TestConfig:
    def test_defaults(self):
        config = SkillsBenchEvaluatorConfig()

        assert config.agent == "openhands-cli"
        assert config.env_version == "1.0.1"
        assert config.conditions == ["no-skill", "with-skill"]
        assert config.pass_at_k == 1
        assert config.text_only is True
        assert config.n_tasks is None
        assert config.task_name is None
        assert config.exclude_task_name is None
        assert config.max_tokens == 16384
        assert config.max_input_tokens == 98304
        assert config.llm_timeout == DEFAULT_REQUEST_TIMEOUT
        assert config.n_concurrent is None
        assert config.venv_python == "/home/user/terminal_bench_env/bin/python"

    def test_fields_match_swe_bench(self):
        skillsbench_fields = set(SkillsBenchEvaluatorConfig.model_fields)
        swe_bench_fields = set(SweBenchEvaluatorConfig.model_fields)

        assert skillsbench_fields - {"conditions", "text_only"} <= swe_bench_fields
        assert len(skillsbench_fields) <= len(swe_bench_fields)

    def test_aliases_and_serialization(self):
        config = SkillsBenchEvaluatorConfig(
            **{
                "model-api-base": "http://model/v1",
                "max-tokens": 10,
                "max-input-tokens": 20,
                "agent-timeout": 30,
                "n-concurrent": 4,
                "pass-at-k": 3,
                "n-tasks": 10,
                "task-name": ["task-a"],
                "exclude-task-name": "task-b",
                "env-version": "2.0.0",
                "timeout-rli": 40,
                "request-retries": 0,
                "transfer-retries": 2,
                "session-ready-timeout": 50,
                "create-session-jitter-max-sec": 0,
            }
        )

        payload = config.to_dict()

        assert payload["model-api-base"] == "http://model/v1"
        assert payload["max-tokens"] == 10
        assert payload["agent-timeout"] == 30
        assert payload["n-concurrent"] == 4
        assert payload["pass-at-k"] == 3
        assert payload["n-tasks"] == 10
        assert payload["task-name"] == ["task-a"]
        assert payload["exclude-task-name"] == "task-b"
        assert "venv-python" not in payload

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("agent", "unknown"),
            ("conditions", ["unknown"]),
            ("conditions", []),
            ("pass_at_k", 0),
            ("max_tokens", 0),
            ("n_concurrent", 0),
            ("request_retries", -1),
        ],
    )
    def test_rejects_invalid_values(self, field, value):
        with pytest.raises(ValidationError):
            SkillsBenchEvaluatorConfig(**{field: value})

    def test_apply_defaults(self):
        config = SkillsBenchEvaluatorConfig()

        config.apply_defaults(inference_instances=3, base_url="http://model")

        assert config.model_api_base == "http://model/v1"
        assert config.n_concurrent == 24

    def test_apply_defaults_preserves_explicit_values(self):
        config = SkillsBenchEvaluatorConfig(
            model_api_base="http://custom/v1",
            n_concurrent=5,
        )

        config.apply_defaults(inference_instances=3, base_url="http://model")

        assert config.model_api_base == "http://custom/v1"
        assert config.n_concurrent == 5

    def test_accepts_single_condition_and_empty_selector(self):
        config = SkillsBenchEvaluatorConfig(
            conditions=["no-skill"],
            task_name=None,
        )

        assert config.conditions == ["no-skill"]
        assert config.task_name is None


class TestEvaluator:
    def test_setup_and_build_command(self, evaluator_factory):
        evaluator = evaluator_factory(venv_python="/opt/python")

        command = evaluator._build_cmd()
        config = json.loads(command[-1])

        assert evaluator.metrics_path == evaluator.metrics_dir / "metrics.json"
        assert command[:4] == ["/opt/python", "-m", "skillsbench.cli", "run"]
        assert config["model-api-base"] == "http://localhost:29999/v1"
        assert config["n-concurrent"] == 16
        assert config["pass-at-k"] == 1

    def test_build_command_does_not_validate_runtime_defaults(self, evaluator_factory):
        evaluator = evaluator_factory(apply_defaults=False)

        config = json.loads(evaluator._build_cmd()[-1])

        assert "model-api-base" not in config

    def test_build_env_adds_rli_and_python_paths(self, evaluator_factory, monkeypatch):
        evaluator = evaluator_factory()
        monkeypatch.setattr(evaluator, "get_rli_url", lambda: "https://rli")
        monkeypatch.setenv("PYTHONPATH", "/existing")

        env = evaluator._build_env()

        assert env["RLI_API_BASE"] == "https://rli"
        assert env["PYTHONPATH"].split(":")[0] == "/home/user/terminal_bench_env"
        assert "/home/user/skillsbench" in env["PYTHONPATH"]
        assert "contrib/skillsbench" not in env["PYTHONPATH"]
        assert env["PYTHONPATH"].endswith(":/existing")

    def test_build_env_without_existing_pythonpath(
        self, evaluator_factory, monkeypatch
    ):
        evaluator = evaluator_factory()
        monkeypatch.setattr(evaluator, "get_rli_url", lambda: "https://rli")
        monkeypatch.delenv("PYTHONPATH", raising=False)

        assert "/home/user/skillsbench" in evaluator._build_env()["PYTHONPATH"]

    def test_already_calculated_and_force_recalc(self, evaluator_factory):
        evaluator = evaluator_factory()
        assert evaluator._is_already_calculated() is False
        write_metrics(evaluator)
        assert evaluator._is_already_calculated() is False
        evaluator.fingerprint_path.write_text("{", encoding="utf-8")
        assert evaluator._is_already_calculated() is False
        write_fingerprint(evaluator)
        assert evaluator._is_already_calculated() is True
        evaluator.autometric_runs_config.request_retries += 1
        assert evaluator._is_already_calculated() is True
        evaluator.autometric_runs_config.request_retries -= 1
        evaluator.autometric_runs_config.max_tokens += 1
        assert evaluator._is_already_calculated() is False
        evaluator.autometric_runs_config.max_tokens -= 1
        write_metrics(evaluator, complete=False)
        assert evaluator._is_already_calculated() is False
        write_metrics(evaluator)
        evaluator.autometric_runs_config.force_recalc = True
        assert evaluator._is_already_calculated() is False

    def test_offered_load(self, evaluator_factory):
        assert evaluator_factory(n_concurrent=7).offered_load == 7
        assert evaluator_factory(apply_defaults=False).offered_load == 1

    def test_fingerprint_uses_agent_timeout_instead_of_concurrency(
        self, evaluator_factory
    ):
        evaluator = evaluator_factory(agent_timeout=100, n_concurrent=4)

        methodology = evaluator._fingerprint()["config"]

        assert methodology["agent-timeout"] == 100
        assert "n-concurrent" not in methodology

    def test_run_skips_existing_metrics(self, evaluator_factory, monkeypatch):
        evaluator = evaluator_factory()
        write_metrics(evaluator)
        write_fingerprint(evaluator)
        monkeypatch.setattr(evaluator, "_build_cmd", lambda: pytest.fail("called"))

        evaluator.run()

    def test_run_invokes_subprocess(self, evaluator_factory, monkeypatch):
        evaluator = evaluator_factory()
        calls = []

        class FakeSubprocessManager:
            def __init__(self, cmd, log_file, cwd, run_name, env):
                calls.append((cmd, log_file, cwd, run_name, env))

            def start(self):
                calls.append("start")

            def wait_for_completion(self):
                calls.append("wait")

            def log_error(self):
                calls.append("log_error")

        monkeypatch.setattr(
            skillsbench_module, "SubprocessManager", FakeSubprocessManager
        )
        monkeypatch.setattr(evaluator, "get_rli_url", lambda: "https://rli")

        evaluator.run()

        assert calls[0][3] == "skillsbench"
        assert calls[0][4]["RLI_API_BASE"] == "https://rli"
        assert calls[1:] == ["start", "wait", "log_error"]
        assert not evaluator.fingerprint_path.exists()

    def test_run_writes_fingerprint_for_publishable_result(
        self, evaluator_factory, monkeypatch
    ):
        evaluator = evaluator_factory()

        class FakeSubprocessManager:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                pass

            def wait_for_completion(self):
                write_metrics(evaluator)

            def log_error(self):
                pass

        monkeypatch.setattr(
            skillsbench_module, "SubprocessManager", FakeSubprocessManager
        )

        evaluator.run()

        assert json.loads(evaluator.fingerprint_path.read_text()) == (
            evaluator._fingerprint()
        )

    def test_get_metrics_missing_and_invalid(self, evaluator_factory):
        evaluator = evaluator_factory()
        assert evaluator.get_metrics() == {}
        evaluator.metrics_path.write_text(json.dumps({"metrics": []}))
        assert evaluator.get_metrics() == {}
        evaluator.metrics_path.write_text(json.dumps([]))
        assert evaluator.get_metrics() == {}
        evaluator.metrics_path.write_text("{")
        assert evaluator.get_metrics() == {}

    def test_get_metrics_rejects_non_numeric_metrics(self, evaluator_factory):
        evaluator = evaluator_factory()
        write_metrics(evaluator)
        payload = json.loads(evaluator.metrics_path.read_text())
        payload["metrics"] = []
        evaluator.metrics_path.write_text(json.dumps(payload))

        assert evaluator.get_metrics() == {}

    def test_get_metrics_keeps_incomplete_result(self, evaluator_factory):
        evaluator = evaluator_factory()
        write_metrics(evaluator, complete=False)

        assert evaluator.get_metrics() == {
            "skillsbench/no_skill/avg": 0.25,
            "skillsbench/no_skill/pass@3": 0.5,
            "skillsbench/skill_lift/avg": 0.25,
        }

    def test_get_metrics_returns_only_numbers(self, evaluator_factory):
        evaluator = evaluator_factory()
        write_metrics(evaluator)

        assert evaluator.get_metrics() == {
            "skillsbench/no_skill/avg": 0.25,
            "skillsbench/no_skill/pass@3": 0.5,
            "skillsbench/skill_lift/avg": 0.25,
        }

    @pytest.mark.parametrize(
        ("filtered", "publish_metrics", "expected_send"),
        [
            (False, True, True),
            (True, True, False),
            (False, False, False),
        ],
    )
    def test_after_run_respects_publication_gate(
        self,
        evaluator_factory,
        monkeypatch,
        filtered,
        publish_metrics,
        expected_send,
    ):
        evaluator = evaluator_factory(n_tasks=1 if filtered else None)
        write_metrics(
            evaluator,
            complete=publish_metrics,
            publish_metrics=publish_metrics,
        )
        calls = []
        monkeypatch.setattr(
            evaluator, "_send_metrics_to_jd", lambda metrics: calls.append(metrics)
        )
        monkeypatch.setattr(JD, "tar_metrics_dir", lambda *_: Path("archive"))
        monkeypatch.setattr(
            JD, "symlink_metrics_for_transfer", lambda *args: calls.append(args)
        )

        evaluator.after_run({"metric": 1.0})

        assert ({"metric": 1.0} in calls) is expected_send
        assert any(isinstance(call, tuple) for call in calls)

    def test_after_run_skips_missing_metrics(self, evaluator_factory, monkeypatch):
        evaluator = evaluator_factory()
        monkeypatch.setattr(
            evaluator,
            "_send_metrics_to_jd",
            lambda metrics: pytest.fail("called"),
        )
        monkeypatch.setattr(
            JD,
            "tar_metrics_dir",
            lambda *_: pytest.fail("called"),
        )

        evaluator.after_run({"metric": 1.0})

    @pytest.mark.usefixtures("scalar_ci_upload")
    def test_send_metrics_to_jd(self, evaluator_factory, monkeypatch):
        evaluator = evaluator_factory()
        calls = {}
        monkeypatch.setattr(
            JD,
            "can_send_metrics",
            lambda **kwargs: True,
        )
        monkeypatch.setattr(
            JD,
            "get_shooter_version",
            lambda version_file: f"version:{version_file}",
        )
        monkeypatch.setattr(
            JD, "send_evaluator_metrics", lambda **kwargs: calls.update(kwargs)
        )

        evaluator._send_metrics_to_jd(
            {
                "SKILLSBENCH_NO_SKILL_SCORE_OPENHANDS_CLI_TEXT_ATTEMPTS_1_"
                "MAX_TOKENS_16384": 18.933,
                "SKILLSBENCH_NO_SKILL_RESOLUTION_RATE_OPENHANDS_CLI_TEXT_"
                "ATTEMPTS_1_MAX_TOKENS_16384": 23.374074074074073,
                "SKILLSBENCH_NO_SKILL_PASS@1_OPENHANDS_CLI_TEXT_ATTEMPTS_1_"
                "MAX_TOKENS_16384": 0.20987654320987653,
                "skillsbench/no_skill/domain/code/score": 5.0,
            }
        )

        assert calls["shooter_name"] == "skillsbench"
        assert calls["shooter_version"] == (
            "version:/home/user/skillsbench/skillsbench_version.txt"
        )
        assert calls["jd_metrics"] == {
            "SKILLSBENCH_NO_SKILL_SCORE_OPENHANDS_CLI_TEXT_ATTEMPTS_1_MAX_TOKENS_16384": 0.18933,
            "SKILLSBENCH_NO_SKILL_RESOLUTION_RATE_OPENHANDS_CLI_TEXT_ATTEMPTS_1_MAX_TOKENS_16384": 0.23374074074074073,
            "SKILLSBENCH_NO_SKILL_PASS@1_OPENHANDS_CLI_TEXT_ATTEMPTS_1_MAX_TOKENS_16384": 0.20987654320987653,
        }
        assert calls["hf_checkpoint_path"] == evaluator.hf_checkpoint_path
        assert calls["metrics_dir"] == evaluator.metrics_dir
        assert calls["model_info"] == evaluator.model_info

    @pytest.mark.usefixtures("scalar_ci_upload")
    def test_skips_jd_when_metrics_cannot_be_sent(self, evaluator_factory, monkeypatch):
        evaluator = evaluator_factory()
        calls = []
        monkeypatch.setattr(JD, "can_send_metrics", lambda **kwargs: False)
        monkeypatch.setattr(
            JD,
            "send_evaluator_metrics",
            lambda **kwargs: calls.append(kwargs),
        )

        evaluator._send_metrics_to_jd({"skillsbench/no_skill/avg": 0.25})

        assert calls == []


@pytest.mark.parametrize(
    "case",
    ["complete", "wrong_roster", "duplicate_basename", "unpublishable", "missing_job"],
)
def test_ci_preserves_condition_and_domain_aggregation(
    evaluator_factory, harbor_job, case
):
    evaluator = evaluator_factory(pass_at_k=2)
    rewards = {"a": [1, 0], "b": [0, 0]}
    if case == "duplicate_basename":
        rewards = {"one/a": [1, 0], "two/a": [0, 0]}
    path = harbor_job(rewards)
    write_metrics(evaluator, complete=case != "unpublishable")
    payload = json.loads(evaluator.metrics_path.read_text())
    payload["conditions"]["no-skill"].update(
        {
            "ci_run_dir": str(path.parent),
            "planned": 4,
            "ci_task_domains": {"a": "coding", "b": "coding"},
        }
    )
    if case == "wrong_roster":
        payload["conditions"]["no-skill"]["ci_task_domains"].pop("b")
    elif case == "missing_job":
        path.unlink()
    evaluator.metrics_path.write_text(json.dumps(payload))
    metrics = {
        "SKILLSBENCH_NO_SKILL_SCORE_AT_2": 0.005,
        "SKILLSBENCH_NO_SKILL_RESOLUTION_RATE_AT_2": 0.25,
        "SKILLSBENCH_NO_SKILL_PASS@2_AT_2": 0.5,
        "SKILLSBENCH_NO_SKILL_DOMAIN_CODING_RESOLUTION_RATE_AT_2": 0.25,
        "SKILLSBENCH_NO_SKILL_ERRORS_AT_2": 0,
        "OTHER": 0,
    }
    scores = evaluator._cache_scores(metrics)
    assert {
        key: [row["score"] for row in rows] for key, rows in scores.records.items()
    } == (
        {
            "SKILLSBENCH_NO_SKILL_SCORE_AT_2": [0.01, 0],
            "SKILLSBENCH_NO_SKILL_RESOLUTION_RATE_AT_2": [0.5, 0],
            "SKILLSBENCH_NO_SKILL_PASS@2_AT_2": [1, 0],
            "SKILLSBENCH_NO_SKILL_DOMAIN_CODING_RESOLUTION_RATE_AT_2": [0.5, 0],
        }
        if case == "complete"
        else {}
    )
