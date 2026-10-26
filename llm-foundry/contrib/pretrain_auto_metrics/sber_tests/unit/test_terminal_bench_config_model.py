from pathlib import Path

import pytest

from node_runner.src.config_model import NodeRunnerConfigModel

pytestmark = pytest.mark.unit


def _base_config() -> dict:
    return {
        "run_name": "run",
        "tensorboard_path": "/tmp/tb",
        "max_seq_len": 32768,
        "autometrics": {
            "globals": {},
            "router": {},
            "inference": {"tool-call-parser": "qwen3_coder"},
            "obm_proxy": {"api-port": 31234},
            "evaluators": {
                "terminal_bench": {
                    "task-name": "adaptive-rejection-sampler",
                    "n-concurrent": 2,
                    "max-input-tokens": 98304,
                    "request-retries": 5,
                    "request-retry-backoff": 0.5,
                    "transfer-retries": 4,
                    "transfer-retry-backoff": 0.25,
                    "use-pip-proxy": False,
                }
            },
            "preparation": {},
            "convertation": {},
            "model_info": {
                "model_class": "sft",
                "size": "lite",
                "username": "u",
            },
        },
    }


def test_node_runner_config_accepts_terminal_bench_and_applies_api_port(tmp_path: Path):
    model = NodeRunnerConfigModel.model_validate(_base_config())
    terminal_bench = model.autometrics.evaluators.terminal_bench
    assert terminal_bench is not None

    terminal_bench.apply_defaults(
        inference_instances=1,
        base_url=f"http://127.0.0.1:{model.autometrics.obm_proxy.api_port}",
    )

    assert terminal_bench.model_api_base == "http://127.0.0.1:31234/v1"
    assert terminal_bench.task_name == "adaptive-rejection-sampler"
    assert terminal_bench.to_dict()["agent"] == "terminus-2"
    assert terminal_bench.to_dict()["n-concurrent"] == 2  # explicit value preserved
    assert terminal_bench.to_dict()["max-input-tokens"] == 98304
    assert "agent-import-path" not in terminal_bench.to_dict()
    assert terminal_bench.to_dict()["request-retries"] == 5
    assert terminal_bench.to_dict()["request-retry-backoff"] == 0.5
    assert terminal_bench.to_dict()["transfer-retries"] == 4
    assert terminal_bench.to_dict()["transfer-retry-backoff"] == 0.25


def test_terminal_bench_apply_defaults_scales_n_concurrent(tmp_path: Path):
    config = _base_config()
    del config["autometrics"]["evaluators"]["terminal_bench"]["n-concurrent"]
    model = NodeRunnerConfigModel.model_validate(config)
    terminal_bench = model.autometrics.evaluators.terminal_bench

    terminal_bench.apply_defaults(
        inference_instances=4, base_url="http://127.0.0.1:31234"
    )

    # 4 * TERMINAL_BENCH_DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
    assert terminal_bench.n_concurrent == 32
    assert terminal_bench.to_dict()["use-pip-proxy"] is False
    assert [name for name, _ in model.autometrics.evaluators] == ["terminal_bench"]
