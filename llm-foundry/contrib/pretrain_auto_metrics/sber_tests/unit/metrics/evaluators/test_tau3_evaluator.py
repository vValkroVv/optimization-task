import json
from pathlib import Path

import pytest

from node_runner.src.config_model import ModelInfoConfig, Tau3EvaluatorConfig
from node_runner.src.job_dispatcher import JD
from node_runner.src.metrics.evaluators.tau3 import Tau3Evaluator
from sber_tests.test_support import judge_endpoints

pytestmark = pytest.mark.unit


def _build_evaluator(tmp_path: Path, **config_overrides) -> Tau3Evaluator:
    config = Tau3EvaluatorConfig(
        domains=config_overrides.pop("domains", ["airline"]),
        **config_overrides,
    )
    config.judge_endpoints = judge_endpoints(
        {config.user_model_name: "http://127.0.0.1:8000"}
    )
    config.apply_defaults(
        inference_instances=1,
        base_url="http://localhost:29999",
    )
    evaluator = Tau3Evaluator()
    evaluator.setup(
        autometric_runs_config=config,
        artifacts_path=tmp_path / "artifacts",
        hf_checkpoint_path=tmp_path / "hf_ckpt",
        model_info=ModelInfoConfig(model_class="sft", size="lite", username="u"),
    )
    return evaluator


def test_config_defaults_and_knowledge_cli():
    config = Tau3EvaluatorConfig()
    config.judge_endpoints = judge_endpoints({config.user_model_name: "http://user"})
    config.apply_defaults(
        inference_instances=1,
        base_url="http://agent",
    )

    assert config.domains == ["airline", "retail", "telecom", "banking_knowledge"]
    assert config.secondary_model_names == ["Qwen3-235B-A22B-Instruct-2507-FP8"]
    assert "--retrieval-config" not in config.build_cli_run(
        domain="airline", save_to="airline"
    )
    assert config.build_cli_run(domain="banking_knowledge", save_to="knowledge")[
        -2:
    ] == ["--retrieval-config", "bm25"]


def _write_results(evaluator: Tau3Evaluator, failed_domain: str | None) -> None:
    for domain, save_to in evaluator._save_to_by_domain.items():
        results_path = Path(save_to) / "results.json"
        results_path.parent.mkdir(parents=True)
        results_path.write_text(
            json.dumps(
                {
                    "tasks": [{"id": "task_001"}],
                    "simulations": [
                        {
                            "task_id": "task_001",
                            "trial": 0,
                            "reward_info": {"reward": 1},
                        },
                        {
                            "task_id": "task_001",
                            "trial": 1,
                            "reward_info": (
                                None if domain == failed_domain else {"reward": 0}
                            ),
                        },
                    ],
                }
            )
        )


def test_run_collects_mean_at_k(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    evaluator = _build_evaluator(tmp_path, num_trials=2)

    class _FakePool:
        def __init__(self, **kwargs):
            pass

        def run_all(self):
            _write_results(evaluator, failed_domain=None)

    monkeypatch.setattr(
        "node_runner.src.metrics.evaluators.tau2.SubprocessPoolRunner", _FakePool
    )

    evaluator.run_on_model_name("user-sim")

    assert evaluator.get_metrics() == {"tau3/airline/mean@2": 0.5}


def test_run_keeps_complete_domains_and_raises_on_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    evaluator = _build_evaluator(
        tmp_path, num_trials=2, domains=["banking_knowledge", "airline"]
    )

    class _FakePool:
        def __init__(self, **kwargs):
            pass

        def run_all(self):
            _write_results(evaluator, failed_domain="banking_knowledge")

    monkeypatch.setattr(
        "node_runner.src.metrics.evaluators.tau2.SubprocessPoolRunner", _FakePool
    )

    with pytest.raises(RuntimeError, match="no metrics for: banking_knowledge"):
        evaluator.run_on_model_name("user-sim")

    assert evaluator.get_metrics() == {"tau3/airline/mean@2": 0.5}


def test_run_raises_when_results_are_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    evaluator = _build_evaluator(tmp_path)
    errors: list[str] = []
    monkeypatch.setattr(
        "node_runner.src.metrics.evaluators.tau3.Logger.error", errors.append
    )

    class _FakePool:
        def __init__(self, **kwargs):
            pass

        def run_all(self):
            pass

    monkeypatch.setattr(
        "node_runner.src.metrics.evaluators.tau2.SubprocessPoolRunner", _FakePool
    )

    with pytest.raises(RuntimeError, match="no metrics for: airline"):
        evaluator.run_on_model_name("user-sim")

    assert evaluator.get_metrics() == {}
    assert len(errors) == 1
    assert "domain airline failed" in errors[0]
    assert (
        str(Path(evaluator._save_to_by_domain["airline"]) / "results.json") in errors[0]
    )


@pytest.mark.parametrize(
    ("simulations", "message"),
    [
        ([], "are empty"),
        (
            [{"task_id": "task_001", "trial": 0}],
            "1/1 simulations have infrastructure errors or no reward",
        ),
        (
            [{"task_id": "task_001", "trial": 0, "reward_info": {"reward": None}}],
            "1/1 simulations have infrastructure errors or no reward",
        ),
        (
            [{"task_id": "task_002", "trial": 0, "reward_info": {"reward": 1}}],
            "expected 1 simulations, found 1",
        ),
        (
            [{"task_id": "task_001", "trial": 0, "reward_info": {"reward": 1}}] * 2,
            "expected 1 simulations, found 2",
        ),
    ],
)
def test_parse_rejects_incomplete_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    simulations: list[dict],
    message: str,
):
    evaluator = _build_evaluator(tmp_path, num_trials=1)
    errors: list[str] = []
    monkeypatch.setattr(
        "node_runner.src.metrics.evaluators.tau3.Logger.error", errors.append
    )
    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps({"tasks": [{"id": "task_001"}], "simulations": simulations})
    )

    assert evaluator._parse_mean_reward_from_results(results_path, "airline") == {}
    assert len(errors) == 1
    assert message in errors[0]
    assert str(results_path) in errors[0]


@pytest.mark.usefixtures("scalar_ci_upload")
def test_send_metrics_to_jd_skips_without_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    evaluator = _build_evaluator(tmp_path)
    evaluator.hf_checkpoint_path = None
    monkeypatch.setattr(
        "node_runner.src.job_dispatcher.JD.send_metrics",
        lambda body, artifacts_path: pytest.fail("must not send without a checkpoint"),
    )

    evaluator._send_metrics_to_jd({"tau3/airline/mean@1": 0.5})


@pytest.mark.usefixtures("scalar_ci_upload")
def test_send_metrics_to_jd_includes_baseline_k_and_retrieval(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    evaluator = _build_evaluator(tmp_path, num_trials=3)
    evaluator.hf_checkpoint_path = (
        tmp_path / "model" / "ep1_ba2" / "artifacts" / "hf_checkpoint"
    )
    saved: dict[str, object] = {}
    sent: dict[str, object] = {}
    monkeypatch.setattr(
        "node_runner.src.job_dispatcher.JD.get_shooter_version",
        lambda _: "640ea513",
    )
    monkeypatch.setattr(
        "node_runner.src.job_dispatcher.JD.save_metrics_json",
        lambda body, step, metrics_dir: saved.update(
            body=body, step=step, metrics_dir=metrics_dir
        ),
    )
    monkeypatch.setattr(
        "node_runner.src.job_dispatcher.JD.send_metrics",
        lambda body, artifacts_path: sent.update(body=body),
    )

    evaluator._send_metrics_to_jd(
        {
            "tau3/airline/mean@3": 0.25,
            "tau3/banking_knowledge/mean@3": 0.5,
        }
    )

    body = sent["body"]
    assert body["metrics"] == {
        2: {
            "TAU3_MEAN@3_VS_QWEN3_235B_A22B_INSTRUCT_2507_FP8": {
                "value": 0.375,
                "submetrics": {
                    "TAU3_AIRLINE_MEAN@3_VS_QWEN3_235B_A22B_INSTRUCT_2507_FP8": 0.25,
                    "TAU3_BANKING_KNOWLEDGE_MEAN@3_VS_QWEN3_235B_A22B_INSTRUCT_2507_FP8_RETRIEVAL_BM25": 0.5,
                },
            },
        }
    }
    assert body["shooter_name"] == "tau3"
    assert body["shooter_version"] == "640ea513"
    assert saved["step"] == 2
    assert saved["metrics_dir"] == evaluator.metrics_dir


class TestTau3Headline:
    @pytest.mark.usefixtures("scalar_ci_upload")
    def test_domains_are_averaged_under_the_user_model_keyed_mean(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = _build_evaluator(
            tmp_path, domains=["airline", "banking_knowledge"], num_trials=2
        )
        evaluator.hf_checkpoint_path = (
            tmp_path / "model" / "ep1_ba2" / "artifacts" / "hf_checkpoint"
        )
        monkeypatch.setattr(JD, "get_shooter_version", lambda path: "2.1.0")
        sent: dict[str, object] = {}
        monkeypatch.setattr(
            JD, "save_metrics_json", lambda body, step, metrics_dir: None
        )
        monkeypatch.setattr(
            JD, "send_metrics", lambda body, _artifacts: sent.update(body=body)
        )

        evaluator._send_metrics_to_jd(
            {"tau3/airline/mean@2": 0.5, "tau3/banking_knowledge/mean@2": 0.7}
        )

        suffix = "MEAN@2_VS_QWEN3_235B_A22B_INSTRUCT_2507_FP8"
        assert sent["body"]["metrics"] == {
            2: {
                f"TAU3_{suffix}": {
                    "value": pytest.approx(0.6),
                    "submetrics": {
                        f"TAU3_AIRLINE_{suffix}": 0.5,
                        f"TAU3_BANKING_KNOWLEDGE_{suffix}_RETRIEVAL_BM25": 0.7,
                    },
                },
            }
        }

    @pytest.mark.usefixtures("scalar_ci_upload")
    def test_mean_is_skipped_when_a_domain_is_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = _build_evaluator(
            tmp_path, domains=["airline", "banking_knowledge"], num_trials=2
        )
        evaluator.hf_checkpoint_path = (
            tmp_path / "model" / "ep1_ba2" / "artifacts" / "hf_checkpoint"
        )
        monkeypatch.setattr(JD, "get_shooter_version", lambda path: "2.1.0")
        sent: dict[str, object] = {}
        monkeypatch.setattr(
            JD, "save_metrics_json", lambda body, step, metrics_dir: None
        )
        monkeypatch.setattr(
            JD, "send_metrics", lambda body, _artifacts: sent.update(body=body)
        )

        evaluator._send_metrics_to_jd({"tau3/airline/mean@2": 0.5})

        assert sent["body"]["metrics"] == {
            2: {"TAU3_AIRLINE_MEAN@2_VS_QWEN3_235B_A22B_INSTRUCT_2507_FP8": 0.5}
        }
