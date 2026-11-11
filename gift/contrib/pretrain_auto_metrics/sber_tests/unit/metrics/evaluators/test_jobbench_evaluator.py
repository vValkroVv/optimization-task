import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from node_runner.src.config_model import (
    JobBenchEvaluatorConfig,
    JudgeEndpoint,
    ModelInfoConfig,
)
from node_runner.src.job_dispatcher import JD
from node_runner.src.metrics.evaluators import harbor_rli_judged as shared
from node_runner.src.metrics.evaluators.jobbench import JobBenchEvaluator

pytestmark = pytest.mark.unit


@pytest.fixture
def evaluator(tmp_path, monkeypatch):
    root = tmp_path / "shooter"
    versions = root / "harbor_rli/harbor_rli_version.txt"
    versions.parent.mkdir(parents=True)
    versions.write_text(
        json.dumps(
            {
                "rli": "harbor+sdk",
                "benchmarks": {"jobbench": "upstream"},
            }
        )
    )
    config = JobBenchEvaluatorConfig(
        judge_model_name="org/Judge-Model", shooter_root=str(root)
    )
    config.judge_endpoints = {config.judge_model_name: JudgeEndpoint("http://judge", 3)}
    config.apply_defaults(inference_instances=2, base_url="http://model")
    evaluator = JobBenchEvaluator()
    evaluator.setup(
        config,
        artifacts_path=tmp_path / "artifacts",
        hf_checkpoint_path=tmp_path / "global_step_7",
        model_info=ModelInfoConfig(model_class="sft", size="10b", username="user"),
    )
    monkeypatch.setattr(evaluator, "get_rli_url", lambda: "https://rli.test")
    return evaluator


@pytest.fixture
def subprocess_factory(monkeypatch):
    factory = Mock()
    factory.return_value.is_successful.return_value = True

    def capture(command, *args, **kwargs):
        factory.payload = json.loads(Path(command[-1]).read_text())
        return factory.return_value

    factory.side_effect = capture
    monkeypatch.setattr(shared, "SubprocessManager", factory)
    return factory


def write_metrics(evaluator, value=0.5):
    payload = {
        "complete": True,
        "metrics": {f"jobbench/{evaluator.autometric_runs_config.split}/score": value},
        "rewards": {"error": None, "partial": 0.5, "success": 1.0},
        "identity": {"aggregation": "harbor_mean", "split": "main"},
    }
    evaluator.metrics_path.write_text(json.dumps(payload))
    return payload


@pytest.mark.parametrize("phase", ["generation", "scoring"])
def test_run_uses_shared_harbor_environment(evaluator, subprocess_factory, phase):
    evaluator.run_phase(phase)

    args, kwargs = subprocess_factory.call_args
    command, log_file = args
    assert command[:-1] == [
        "/home/user/harbor_rli_env/bin/python",
        "-m",
        "harbor_rli.cli",
        "run",
        "--benchmark",
        "jobbench",
        "--metrics-dir",
        str(evaluator.metrics_dir),
        "--phase",
        phase,
        "--config",
    ]
    assert (
        subprocess_factory.payload
        == evaluator.autometric_runs_config.to_shooter_config()
    )
    assert not Path(command[-1]).exists()
    assert not log_file.is_relative_to(evaluator.metrics_dir)
    assert kwargs["cwd"] == evaluator.autometric_runs_config.shooter_root
    assert kwargs["run_name"] == f"jobbench_{phase}"
    assert kwargs["env"]["RLI_API_BASE"] == "https://rli.test"
    assert kwargs["env"]["PYTHONPATH"] == evaluator.autometric_runs_config.shooter_root
    subprocess_factory.return_value.start.assert_called_once_with()
    subprocess_factory.return_value.wait_for_completion.assert_called_once_with()


@pytest.mark.parametrize("failure", ["start", "wait", "exit"])
def test_generation_failure_skips_scoring_and_stale_metrics(
    evaluator, subprocess_factory, failure
):
    write_metrics(evaluator)
    process = subprocess_factory.return_value
    if failure == "start":
        process.start.side_effect = OSError("cannot start")
    elif failure == "wait":
        process.wait_for_completion.side_effect = OSError("cannot wait")
    else:
        process.is_successful.return_value = False

    with pytest.raises((OSError, RuntimeError)):
        evaluator.run_phase("generation")
    with pytest.raises(RuntimeError, match="previous phase failed"):
        evaluator.run_phase("scoring")

    assert subprocess_factory.call_count == 1
    assert evaluator.get_metrics() == {}


def test_scoring_failure_suppresses_stale_metrics(evaluator, subprocess_factory):
    write_metrics(evaluator)
    evaluator.run_phase("generation")
    subprocess_factory.return_value.is_successful.return_value = False

    with pytest.raises(RuntimeError, match="scoring failed"):
        evaluator.run_phase("scoring")

    assert evaluator.get_metrics() == {}


@pytest.mark.parametrize("endpoint", [None, "https://external/v1"])
def test_scoring_defaults_preserve_explicit_judge_endpoint(evaluator, endpoint):
    evaluator.autometric_runs_config.judge_api_base = endpoint

    evaluator.autometric_runs_config.apply_defaults(
        inference_instances=2, base_url="http://model"
    )

    assert evaluator.autometric_runs_config.judge_api_base == (
        endpoint if endpoint else "http://judge/v1"
    )
    assert evaluator.judge_model_name == "org/Judge-Model"


def test_generation_does_not_publish_previous_score(evaluator):
    write_metrics(evaluator)

    assert evaluator.get_metrics() == {}


def test_scoring_rejects_result_from_another_split(evaluator):
    write_metrics(evaluator)
    evaluator.autometric_runs_config.split = "easy"
    evaluator._phase = "scoring"

    assert evaluator.get_metrics() == {}


@pytest.mark.parametrize("split", ["main", "easy"])
@pytest.mark.parametrize("value", [0, 0.5, 1])
def test_scoring_publishes_harbor_mean_despite_trial_errors(evaluator, split, value):
    evaluator.autometric_runs_config.split = split
    write_metrics(evaluator, value)
    evaluator._phase = "scoring"

    assert evaluator.get_metrics() == {
        (
            f"jobbench/{split}/mini_swe_agent/max_tokens_131072/"
            "timeout_7200/judge_judge_model/score"
        ): float(value)
    }


@pytest.mark.parametrize(
    "value", [True, False, -0.1, 1.1, float("nan"), float("inf"), None, "0.5"]
)
def test_scoring_rejects_invalid_score(evaluator, value):
    write_metrics(evaluator, value)
    evaluator._phase = "scoring"

    assert evaluator.get_metrics() == {}


@pytest.mark.parametrize("content", [None, "{", "{}", '{"metrics": []}'])
def test_scoring_handles_missing_or_malformed_results(evaluator, content):
    if content is not None:
        evaluator.metrics_path.write_text(content)
    evaluator._phase = "scoring"

    assert evaluator.get_metrics() == {}


@pytest.mark.parametrize("publish", [True, False])
def test_after_run_preserves_errors_and_uses_zero_contribution_for_ci(
    evaluator, monkeypatch, publish
):
    payload = write_metrics(evaluator)
    evaluator._phase = "scoring"
    metrics = evaluator.get_metrics()
    can_send = Mock(return_value=publish)
    send = Mock()
    archive_path = evaluator.artifacts_path / "jobbench.tar.gz"
    archive = Mock(return_value=archive_path)
    transfer = Mock()
    monkeypatch.setattr(JD, "can_send_metrics", can_send)
    monkeypatch.setattr(JD, "send_evaluator_metrics_with_scores", send)
    monkeypatch.setattr(JD, "get_shooter_version", lambda _: "adapter-revision")
    monkeypatch.setattr(JD, "tar_metrics_dir", archive)
    monkeypatch.setattr(JD, "symlink_metrics_for_transfer", transfer)

    evaluator.after_run(metrics)

    can_send.assert_called_once_with(
        metrics=metrics,
        hf_checkpoint_path=evaluator.hf_checkpoint_path,
        metrics_dir=evaluator.metrics_dir,
        model_info=evaluator.model_info,
    )
    if publish:
        sent = send.call_args.kwargs
        scores = sent["scores"]
        key = (
            "JOBBENCH_MAIN_MINI_SWE_AGENT_MAX_TOKENS_131072_"
            "TIMEOUT_7200_JUDGE_JUDGE_MODEL_SCORE"
        )
        assert scores.metrics == {key: 0.5}
        assert [row["score"] for row in scores.records[key]] == [0.0, 0.5, 1.0]
        assert sent["shooter_name"] == "jobbench"
        assert sent["shooter_version"] == "harbor+sdk+upstream"
        assert sent["hf_checkpoint_path"] == evaluator.hf_checkpoint_path
        assert sent["model_info"] == evaluator.model_info
    else:
        send.assert_not_called()
    assert json.loads(evaluator.metrics_path.read_text()) == payload
    archive.assert_called_once_with(
        evaluator.metrics_dir, evaluator.artifacts_path, "JobBenchEvaluator"
    )
    transfer.assert_called_once_with(
        archive_path, evaluator.hf_checkpoint_path, "JobBenchEvaluator"
    )
