import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from node_runner.src.config_model import ModelInfoConfig, PierRliEvaluatorConfig
from node_runner.src.job_dispatcher import JD
from node_runner.src.metrics.evaluators.pier_rli import PierRliEvaluator

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "network_config",
    [{}, {"allow-internet-access": False}, {"allow_internet_access": False}],
)
def test_deepswe_defaults_to_isolated_network(network_config):
    config = PierRliEvaluatorConfig.model_validate(
        {"benchmark": "deepswe", **network_config}
    )

    assert config.to_dict()["allow-internet-access"] is False


@pytest.mark.parametrize("alias", ["allow-internet-access", "allow_internet_access"])
def test_deepswe_preserves_explicit_internet_access(alias):
    config = PierRliEvaluatorConfig.model_validate(
        {"benchmark": "deepswe", alias: True}
    )

    assert config.to_dict()["allow-internet-access"] is True


def test_pier_runtime_and_reward_selection(tmp_path: Path, monkeypatch):
    evaluator = PierRliEvaluator()
    evaluator.setup(
        PierRliEvaluatorConfig(benchmark="deepswe"),
        artifacts_path=tmp_path / "artifacts",
        hf_checkpoint_path=tmp_path / "checkpoint",
        model_info=ModelInfoConfig(model_class="sft", size="lite", username="u"),
    )

    assert evaluator.ENV_ROOT == "/home/user/pier_rli_env"
    assert evaluator._build_cmd()[:4] == [
        "/home/user/pier_rli_env/bin/python",
        "-m",
        "pier_rli.cli",
        "run",
    ]

    result = tmp_path / "result.json"
    result.write_text('{"n_total_trials": 2}')
    for task, reward in (("failed", 0), ("passed", 1)):
        trial = tmp_path / task
        trial.mkdir()
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": task,
                    "verifier_result": {"rewards": {"reward": reward, "p2p": 1}},
                }
            )
        )
    evaluator.metrics_path.write_text(
        json.dumps({"job_result_path": str(result), "total": 2})
    )
    send = Mock()
    monkeypatch.setattr(JD, "send_evaluator_metrics_with_scores", send)

    evaluator._send_metrics_to_jd({"DEEPSWE": 0.5})

    send.assert_called_once()
    scores = send.call_args.kwargs["scores"]
    assert [row["score"] for row in scores.records["DEEPSWE"]] == [0.0, 1.0]
