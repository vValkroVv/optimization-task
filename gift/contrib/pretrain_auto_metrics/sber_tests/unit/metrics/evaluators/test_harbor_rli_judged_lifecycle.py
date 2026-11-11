import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from node_runner.src.config_model import (
    EvaluatorsConfig,
    HarborRliJudgedEvaluatorConfig,
    ModelInfoConfig,
)
from node_runner.src.job_dispatcher import JD
from node_runner.src.metrics.evaluators import harbor_rli_judged as shared

pytestmark = pytest.mark.unit


class ExampleEvaluator(shared.HarborRliJudgedEvaluator):
    BENCHMARK = "example"

    def _build_shooter_config(self) -> dict:
        return {}

    def get_metrics(self) -> dict:
        payload = self.read_complete_result()
        return payload["metrics"] if payload else {}


@pytest.fixture
def setup_evaluator(tmp_path, monkeypatch):
    version_path = tmp_path / "shooter/harbor_rli/harbor_rli_version.txt"
    version_path.parent.mkdir(parents=True)
    manager = Mock()
    manager.return_value.is_successful.return_value = True
    monkeypatch.setattr(shared, "SubprocessManager", manager)
    monkeypatch.setattr(shared.Logger, "error", Mock())

    def create(content):
        if content is not None:
            version_path.write_text(content)
        evaluator = ExampleEvaluator()
        evaluator.setup(
            HarborRliJudgedEvaluatorConfig(
                judge_model_name="judge", shooter_root=str(tmp_path / "shooter")
            ),
            tmp_path / "artifacts",
            tmp_path / "checkpoint",
            ModelInfoConfig(model_class="sft", size="10b", username="fixture"),
        )
        monkeypatch.setattr(evaluator, "get_rli_url", lambda: "https://rli.invalid")
        return evaluator, version_path, manager

    return create


@pytest.mark.parametrize(
    "content",
    [
        None,
        "{",
        "[]",
        "{}",
        '{"rli": "rli", "benchmarks": []}',
        '{"rli": "rli", "benchmarks": {"other": "revision"}}',
        '{"rli": " ", "benchmarks": {"example": "revision"}}',
        '{"rli": "rli", "benchmarks": {"example": 1}}',
    ],
)
def test_invalid_version_fails_phase_before_starting_shooter(setup_evaluator, content):
    evaluator, _, manager = setup_evaluator(content)

    with pytest.raises(RuntimeError, match="shooter version unavailable"):
        evaluator.run_phase("generation")

    manager.assert_not_called()
    assert evaluator.get_metrics() == {}
    with pytest.raises(RuntimeError, match="previous phase failed"):
        evaluator.run_phase("scoring")


def test_validated_version_is_used_for_publication(setup_evaluator, monkeypatch):
    evaluator, version_path, _ = setup_evaluator(
        json.dumps({"rli": "rli-ref", "benchmarks": {"example": "bench-ref"}})
    )
    version_path.unlink()
    evaluator.run_phase("generation")
    evaluator.run_phase("scoring")
    send = Mock()
    monkeypatch.setattr(JD, "can_send_metrics", Mock(return_value=True))
    monkeypatch.setattr(JD, "send_evaluator_metrics_with_scores", send)
    monkeypatch.setattr(
        JD, "tar_metrics_dir", Mock(return_value=Path("archive.tar.gz"))
    )
    monkeypatch.setattr(JD, "symlink_metrics_for_transfer", Mock())

    evaluator.after_run({"example/score": 0.5})

    assert send.call_args.kwargs["shooter_version"] == "rli-ref+bench-ref"


def test_bad_version_does_not_prevent_another_evaluator_from_running(setup_evaluator):
    invalid, _, manager = setup_evaluator("{}")
    valid, _, _ = setup_evaluator(
        json.dumps({"rli": "rli-ref", "benchmarks": {"example": "bench-ref"}})
    )

    with pytest.raises(RuntimeError, match="shooter version unavailable"):
        invalid.run_phase("generation")
    valid.run_phase("generation")
    valid.run_phase("scoring")

    assert manager.call_count == 2


@pytest.mark.parametrize("phase", ["generation", "scoring"])
def test_config_error_blocks_stale_metrics_and_next_phase(setup_evaluator, phase):
    evaluator, _, manager = setup_evaluator(
        json.dumps({"rli": "rli-ref", "benchmarks": {"example": "bench-ref"}})
    )
    if phase == "scoring":
        evaluator.run_phase("generation")
    manager.reset_mock()
    evaluator.metrics_path.write_text(
        json.dumps({"complete": True, "metrics": {"example/score": 0.5}})
    )
    evaluator._build_shooter_config = Mock(side_effect=ValueError("bad config"))

    with pytest.raises(ValueError, match="bad config"):
        evaluator.run_phase(phase)

    manager.assert_not_called()
    assert evaluator.get_metrics() == {}
    with pytest.raises(RuntimeError, match="previous phase failed"):
        evaluator.run_phase("scoring")


def test_old_grouped_config_requires_explicit_migration():
    with pytest.raises(ValidationError, match="separate evaluators"):
        EvaluatorsConfig(harbor_rli_judged={"judge_model_name": "judge"})


def test_phase_logs_are_anonymized_and_preserved(setup_evaluator):
    evaluator, _, manager = setup_evaluator(
        json.dumps({"rli": "rli-ref", "benchmarks": {"example": "bench-ref"}})
    )

    def capture(command, log, **kwargs):
        log.write_text(f"sber {command[-3]}\n")
        return manager.return_value

    manager.side_effect = capture
    evaluator.run_phase("generation")
    evaluator.run_phase("scoring")

    assert evaluator.log_file.read_text() == "papaya generation\npapaya scoring\n"


@pytest.mark.parametrize(
    "content",
    [
        None,
        "{",
        "[]",
        "{}",
        '{"complete": true}',
        '{"complete": true, "metrics": []}',
        '{"complete": true, "metrics": {}}',
        '{"complete": true, "metrics": {"other/score": 0.5}}',
        '{"complete": true, "metrics": {"example/score": true}}',
        '{"complete": true, "metrics": {"example/score": "0.5"}}',
        '{"complete": true, "metrics": {"example/score": NaN}}',
    ],
)
def test_scoring_rejects_unconfirmed_results(setup_evaluator, content):
    evaluator, _, _ = setup_evaluator(
        json.dumps({"rli": "rli-ref", "benchmarks": {"example": "bench-ref"}})
    )
    evaluator.run_phase("generation")
    evaluator.run_phase("scoring")
    if content is not None:
        evaluator.metrics_path.write_text(content)

    assert evaluator.get_metrics() == {}


def test_successful_scoring_reads_confirmed_metrics(setup_evaluator):
    evaluator, _, _ = setup_evaluator(
        json.dumps({"rli": "rli-ref", "benchmarks": {"example": "bench-ref"}})
    )
    evaluator.run_phase("generation")
    evaluator.run_phase("scoring")
    evaluator.metrics_path.write_text(
        json.dumps({"complete": True, "metrics": {"example/score": 0.5}})
    )

    assert evaluator.get_metrics() == {"example/score": 0.5}


@pytest.mark.parametrize("phase", ["generation", "scoring"])
def test_failed_process_blocks_metrics_and_publication(
    setup_evaluator, monkeypatch, phase
):
    evaluator, _, manager = setup_evaluator(
        json.dumps({"rli": "rli-ref", "benchmarks": {"example": "bench-ref"}})
    )
    if phase == "scoring":
        evaluator.run_phase("generation")
    manager.return_value.is_successful.return_value = False
    with pytest.raises(RuntimeError, match="failed"):
        evaluator.run_phase(phase)
    send = Mock()
    monkeypatch.setattr(JD, "send_evaluator_metrics_with_scores", send)
    monkeypatch.setattr(JD, "tar_metrics_dir", Mock())
    monkeypatch.setattr(JD, "symlink_metrics_for_transfer", Mock())

    evaluator.after_run({})

    assert evaluator.get_metrics() == {}
    send.assert_not_called()
