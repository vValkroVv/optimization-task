import json
from pathlib import Path

import pytest

from node_runner.src.config_model import ModelInfoConfig, Nl2RepoEvaluatorConfig
from node_runner.src.metrics.evaluators import nl2repo as nl2repo_module
from node_runner.src.metrics.evaluators.nl2repo import Nl2RepoEvaluator

pytestmark = pytest.mark.unit


def _publishable_payload() -> dict:
    return {
        "metrics": {"NL2REPO_METRIC": 0.5},
        "run": {
            "planned": 2,
            "completed": 2,
            "infrastructure_errors": 0,
            "api_errors": 0,
            "exception_types": [],
        },
        "complete": True,
    }


@pytest.fixture
def evaluator_factory(tmp_path: Path):
    def _build(config_payload: dict | None = None) -> Nl2RepoEvaluator:
        config = Nl2RepoEvaluatorConfig.model_validate(
            config_payload or {"n-concurrent": 3}
        )
        config.apply_defaults(
            inference_instances=1,
            base_url="http://127.0.0.1:29999",
        )
        evaluator = Nl2RepoEvaluator()
        evaluator.setup(
            config,
            artifacts_path=tmp_path / "artifacts",
            hf_checkpoint_path=tmp_path / "model" / "global_step_7",
            model_info=ModelInfoConfig(
                model_class="sft",
                size="lite",
                username="user",
            ),
        )
        return evaluator

    return _build


class TestBuildCommand:
    def test_serializes_new_public_fields(self, evaluator_factory) -> None:
        evaluator = evaluator_factory(
            {
                "n-concurrent": 2,
                "pass-at-k": 3,
                "task-name": ["aiofiles"],
                "publish-incomplete-metrics": True,
                "agent-env-version": "1.0.3",
                "verifier-env-version": "1.0.1",
                "venv_python": "python",
            }
        )

        command = evaluator._build_cmd()
        payload = json.loads(command[-1])

        assert command[:4] == ["python", "-m", "nl2repo.cli", "run"]
        assert payload["pass-at-k"] == 3
        assert payload["task-name"] == ["aiofiles"]
        assert payload["publish-incomplete-metrics"] is True
        assert payload["agent-env-version"] == "1.0.3"
        assert payload["verifier-env-version"] == "1.0.1"
        assert "trials" not in payload
        assert "tasks-root" not in payload

    def test_does_not_validate_runtime_defaults(self, evaluator_factory) -> None:
        evaluator = evaluator_factory()
        evaluator.autometric_runs_config.model_api_base = None

        payload = json.loads(evaluator._build_cmd()[-1])

        assert "model-api-base" not in payload


class TestEnvironment:
    def test_builds_rli_and_python_environment(
        self, evaluator_factory, monkeypatch
    ) -> None:
        monkeypatch.setenv("PYTHONPATH", "/existing")
        evaluator = evaluator_factory()
        monkeypatch.setattr(
            evaluator,
            "get_rli_url",
            lambda: "https://gym.static.space",
        )

        env = evaluator._build_env()
        pythonpath = env["PYTHONPATH"].split(":")

        assert env["RLI_API_BASE"] == "https://gym.static.space"
        assert pythonpath[0] == "/home/user/terminal_bench_env"
        assert pythonpath[1] == "/home/user/nl2repo"
        assert pythonpath[2].endswith("/contrib")
        assert pythonpath[-1] == "/existing"

    def test_builds_pythonpath_without_existing_value(
        self, evaluator_factory, monkeypatch
    ) -> None:
        monkeypatch.delenv("PYTHONPATH", raising=False)
        evaluator = evaluator_factory()
        monkeypatch.setattr(
            evaluator,
            "get_rli_url",
            lambda: "https://rli.sberdevices.ru",
        )

        env = evaluator._build_env()

        assert len(env["PYTHONPATH"].split(":")) == 3

    def test_offered_load_uses_concurrency(self, evaluator_factory) -> None:
        assert evaluator_factory({"n-concurrent": 7}).offered_load == 7

    def test_fingerprint_uses_agent_timeout_instead_of_concurrency(
        self, evaluator_factory
    ) -> None:
        evaluator = evaluator_factory({"agent-timeout": 100, "n-concurrent": 4})

        methodology = evaluator._fingerprint()["config"]

        assert methodology["agent-timeout"] == 100
        assert "n-concurrent" not in methodology


class TestRun:
    def test_skips_existing_result(self, evaluator_factory, monkeypatch) -> None:
        evaluator = evaluator_factory()
        evaluator.metrics_path.write_text(
            json.dumps(_publishable_payload()), encoding="utf-8"
        )
        evaluator.fingerprint_path.write_text(
            json.dumps(evaluator._fingerprint()), encoding="utf-8"
        )

        def _fail(*args, **kwargs):
            raise AssertionError("subprocess must not run")

        monkeypatch.setattr(nl2repo_module, "SubprocessManager", _fail)

        evaluator.run()

    @pytest.mark.parametrize(
        ("metrics", "fingerprint"),
        [
            ("{}", None),
            ("{bad json", "{}"),
            (json.dumps(_publishable_payload()), None),
            (json.dumps(_publishable_payload()), "{}"),
            (json.dumps({"metrics": {"NL2REPO_METRIC": 0.5}, "run": []}), "{}"),
            (json.dumps(_publishable_payload()), "{bad json"),
        ],
    )
    def test_rejects_incomplete_or_stale_cache(
        self,
        evaluator_factory,
        metrics,
        fingerprint,
    ) -> None:
        evaluator = evaluator_factory()
        evaluator.metrics_path.write_text(metrics, encoding="utf-8")
        if fingerprint is not None:
            evaluator.fingerprint_path.write_text(fingerprint, encoding="utf-8")

        assert not evaluator._is_already_calculated()

    def test_rejects_cache_from_other_shooter_version(
        self, evaluator_factory, monkeypatch
    ) -> None:
        evaluator = evaluator_factory()
        versions = iter(["old-version", "new-version"])
        monkeypatch.setattr(
            nl2repo_module.JD,
            "get_shooter_version",
            lambda _: next(versions),
        )
        evaluator.metrics_path.write_text(
            json.dumps(_publishable_payload()), encoding="utf-8"
        )
        evaluator.fingerprint_path.write_text(
            json.dumps(evaluator._fingerprint()), encoding="utf-8"
        )

        assert not evaluator._is_already_calculated()

    def test_force_recalc_runs_subprocess(self, evaluator_factory, monkeypatch) -> None:
        evaluator = evaluator_factory({"n-concurrent": 1, "force-recalc": True})
        evaluator.metrics_path.write_text("{}", encoding="utf-8")
        calls: list[str] = []

        class FakeSubprocessManager:
            def __init__(self, cmd, log, cwd, run_name, env):
                calls.append("init")
                assert cmd == evaluator._build_cmd()
                assert log == evaluator.log_file
                assert cwd == str(evaluator.node_runner_root)
                assert run_name == "nl2repo"
                assert "RLI_API_BASE" in env

            def start(self):
                calls.append("start")

            def wait_for_completion(self):
                calls.append("wait")
                evaluator.metrics_path.write_text(
                    json.dumps(_publishable_payload()), encoding="utf-8"
                )

            def log_error(self):
                calls.append("log_error")

        monkeypatch.setattr(
            evaluator,
            "get_rli_url",
            lambda: "https://rli.sberdevices.ru",
        )
        monkeypatch.setattr(
            nl2repo_module,
            "SubprocessManager",
            FakeSubprocessManager,
        )

        evaluator.run()

        assert calls == ["init", "start", "wait", "log_error"]
        assert json.loads(evaluator.fingerprint_path.read_text(encoding="utf-8")) == (
            evaluator._fingerprint()
        )

    def test_does_not_cache_unpublishable_result(
        self, evaluator_factory, monkeypatch
    ) -> None:
        evaluator = evaluator_factory()
        payload = _publishable_payload()
        payload["complete"] = False

        class FakeSubprocessManager:
            def __init__(self, cmd, log, cwd, run_name, env):
                pass

            def start(self):
                pass

            def wait_for_completion(self):
                evaluator.metrics_path.write_text(json.dumps(payload), encoding="utf-8")

            def log_error(self):
                pass

        monkeypatch.setattr(
            evaluator,
            "get_rli_url",
            lambda: "https://rli.sberdevices.ru",
        )
        monkeypatch.setattr(
            nl2repo_module,
            "SubprocessManager",
            FakeSubprocessManager,
        )

        evaluator.run()

        assert not evaluator.fingerprint_path.exists()


class TestMetrics:
    def test_returns_only_numeric_headline_metrics(self, evaluator_factory) -> None:
        evaluator = evaluator_factory()
        evaluator.metrics_path.write_text(
            json.dumps(
                {
                    "metrics": {
                        "NL2REPO_OPENHANDS_0_56_MAX_TOKENS_16384_"
                        "TIMEOUT_21600_MEAN_SCORE": 0.4,
                        "ignored": "value",
                    },
                    "run": {
                        "planned": 2,
                        "completed": 2,
                        "infrastructure_errors": 0,
                        "api_errors": 0,
                        "exception_types": [],
                    },
                    "complete": True,
                }
            ),
            encoding="utf-8",
        )

        assert evaluator.get_metrics() == {
            "NL2REPO_OPENHANDS_0_56_MAX_TOKENS_16384_TIMEOUT_21600_MEAN_SCORE": 0.4
        }

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            (
                {
                    "metrics": {"NL2REPO_METRIC": 0.5},
                    "run": {
                        "planned": 2,
                        "completed": 1,
                        "infrastructure_errors": 1,
                        "exception_types": ["RLIError"],
                    },
                    "complete": False,
                },
                0.5,
            ),
            (
                {
                    "metrics": {"NL2REPO_METRIC": 0.0},
                    "run": {
                        "planned": 0,
                        "completed": 0,
                        "infrastructure_errors": 0,
                        "exception_types": [],
                    },
                    "complete": True,
                },
                0.0,
            ),
            (
                {
                    "metrics": {"NL2REPO_METRIC": 0.5},
                    "run": {
                        "planned": 2,
                        "completed": 2,
                        "infrastructure_errors": 0,
                        "api_errors": 1,
                        "exception_types": ["EndpointUnavailableError"],
                    },
                    "complete": False,
                },
                0.5,
            ),
        ],
    )
    def test_returns_numeric_metrics_for_incomplete_run(
        self, evaluator_factory, payload, expected
    ) -> None:
        evaluator = evaluator_factory()
        raw_metrics = json.dumps(payload)
        evaluator.metrics_path.write_text(raw_metrics, encoding="utf-8")

        assert evaluator.get_metrics() == {"NL2REPO_METRIC": expected}
        assert evaluator.metrics_path.read_text(encoding="utf-8") == raw_metrics

    def test_invalid_metrics_returns_empty(self, evaluator_factory) -> None:
        evaluator = evaluator_factory()
        evaluator.metrics_path.write_text(
            json.dumps({"metrics": []}),
            encoding="utf-8",
        )

        assert evaluator.get_metrics() == {}

    def test_missing_or_malformed_metrics_returns_empty(
        self, evaluator_factory
    ) -> None:
        evaluator = evaluator_factory()
        assert evaluator.get_metrics() == {}
        evaluator.metrics_path.write_text("{bad json", encoding="utf-8")
        assert evaluator.get_metrics() == {}
        evaluator.metrics_path.write_text("[]", encoding="utf-8")
        assert evaluator.get_metrics() == {}

    def test_unconfigured_evaluator_returns_empty(self) -> None:
        evaluator = Nl2RepoEvaluator()
        evaluator.metrics_path = Path("/missing/metrics.json")

        assert evaluator.get_metrics() == {}

    @pytest.mark.parametrize(
        ("config_payload", "publish_metrics", "expected_send"),
        [
            ({}, True, True),
            ({"n-tasks": 1}, True, False),
            ({}, False, False),
        ],
    )
    def test_after_run_respects_cli_publication_and_filtered_runs(
        self,
        evaluator_factory,
        monkeypatch,
        config_payload,
        publish_metrics,
        expected_send,
    ) -> None:
        evaluator = evaluator_factory(config_payload)
        payload = _publishable_payload()
        payload["publish_metrics"] = publish_metrics
        evaluator.metrics_path.write_text(json.dumps(payload), encoding="utf-8")
        calls: list[dict[str, float]] = []
        monkeypatch.setattr(
            evaluator,
            "_send_metrics_to_jd",
            lambda metrics: calls.append(metrics),
        )
        monkeypatch.setattr(nl2repo_module.JD, "tar_metrics_dir", lambda *args: None)
        monkeypatch.setattr(
            nl2repo_module.JD,
            "symlink_metrics_for_transfer",
            lambda *args: None,
        )

        evaluator.after_run({"NL2REPO_METRIC": 0.5})

        assert bool(calls) is expected_send

    def test_after_run_without_result_skips_jd(
        self, evaluator_factory, monkeypatch
    ) -> None:
        evaluator = evaluator_factory()

        def _fail(*args, **kwargs):
            raise AssertionError("JD must not be called without a result")

        monkeypatch.setattr(nl2repo_module.JD, "tar_metrics_dir", _fail)
        monkeypatch.setattr(nl2repo_module.JD, "symlink_metrics_for_transfer", _fail)
        monkeypatch.setattr(evaluator, "_send_metrics_to_jd", _fail)

        evaluator.after_run({"NL2REPO_METRIC": 0.5})


class TestJdPayload:
    @pytest.mark.usefixtures("scalar_ci_upload")
    def test_builds_methodology_qualified_payload(
        self, evaluator_factory, monkeypatch
    ) -> None:
        evaluator = evaluator_factory()
        calls: list[dict] = []
        monkeypatch.setattr(
            nl2repo_module.JD,
            "send_evaluator_metrics",
            lambda **kwargs: calls.append(kwargs),
        )
        monkeypatch.setattr(
            nl2repo_module.JD,
            "get_shooter_version",
            lambda _: "nl2repo-version",
        )

        evaluator._send_metrics_to_jd({"NL2REPO_OPENHANDS_0_56_PASS_AT_3": 0.4})

        assert calls == [
            {
                "hf_checkpoint_path": evaluator.hf_checkpoint_path,
                "metrics_dir": evaluator.metrics_dir,
                "jd_metrics": {"NL2REPO_OPENHANDS_0_56_PASS_AT_3": 0.4},
                "shooter_name": "nl2repo",
                "shooter_version": "nl2repo-version",
                "model_info": evaluator.model_info,
                "extra_body": {
                    "cache_s3_path": "s3://test/cache_scores.jsonl.zst",
                    "cache_dir_s3_path": "s3://test/",
                },
                "submetric_groups": None,
            }
        ]

    @pytest.mark.usefixtures("scalar_ci_upload")
    def test_skips_without_paths(self, evaluator_factory) -> None:
        evaluator = evaluator_factory()
        evaluator.hf_checkpoint_path = None
        evaluator._send_metrics_to_jd({"metric": 1.0})
        evaluator.hf_checkpoint_path = Path("/model/global_step_1")
        evaluator.metrics_dir = None
        evaluator._send_metrics_to_jd({"metric": 1.0})

    @pytest.mark.usefixtures("scalar_ci_upload")
    def test_propagates_request_failure(self, evaluator_factory, monkeypatch) -> None:
        evaluator = evaluator_factory()

        def _raise(*args, **kwargs):
            raise RuntimeError("JD unavailable")

        monkeypatch.setattr(nl2repo_module.JD, "send_evaluator_metrics", _raise)

        with pytest.raises(RuntimeError, match="JD unavailable"):
            evaluator._send_metrics_to_jd({"metric": 1.0})


@pytest.mark.parametrize(
    "case",
    [
        "complete",
        "wrong_roster",
        "duplicate_basename",
        "wrong_test_count",
        "zero_tests",
        "unpublishable",
        "missing_job",
    ],
)
def test_ci_uses_fixed_test_counts_and_task_means(evaluator_factory, harbor_job, case):
    evaluator = evaluator_factory({"pass_at_k": 1})
    rewards = {"a": [0.5], "b": [0]}
    if case == "duplicate_basename":
        rewards = {"one/a": [0.5], "two/a": [0]}
    path = harbor_job(rewards)
    counts = {"a": 2, "b": 6} if case != "zero_tests" else {"a": 0, "b": 0}
    for i, uid in enumerate(("a", "b")):
        verifier = path.parent / str(i) / "verifier"
        verifier.mkdir()
        (verifier / "pytest_summary.json").write_text(
            json.dumps({"test_case_count": counts[uid], "passed": int(uid == "a")})
        )
    payload = _publishable_payload()
    payload["run"].update({"ci_run_dir": str(path.parent), "ci_test_counts": counts})
    if case == "wrong_roster":
        payload["run"]["ci_test_counts"].pop("b")
    elif case == "wrong_test_count":
        payload["run"]["ci_test_counts"]["b"] = 7
    elif case == "unpublishable":
        payload["complete"] = False
    elif case == "missing_job":
        path.unlink()
    evaluator.metrics_path.write_text(json.dumps(payload))
    metrics = {
        "NL2REPO_MEAN_SCORE": 0.25,
        "NL2REPO_PASS_RATE": 0.125,
        "NL2REPO_ERRORS": 0,
    }
    scores = evaluator._cache_scores(metrics)
    expected = {}
    if case in {"complete", "wrong_test_count", "zero_tests"}:
        expected["NL2REPO_MEAN_SCORE"] = [0.5, 0]
    if case == "complete":
        expected["NL2REPO_PASS_RATE"] = [(1, 2), (0, 6)]
    assert {
        key: [
            row["score"]
            if row["scoring_kind"] == "mean"
            else (row["numerator"], row["denominator"])
            for row in rows
        ]
        for key, rows in scores.records.items()
    } == expected
