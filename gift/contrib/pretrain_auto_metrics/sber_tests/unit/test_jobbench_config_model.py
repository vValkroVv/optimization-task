import pytest
from pydantic import ValidationError

from node_runner.src.config_model import JobBenchEvaluatorConfig, JudgeEndpoint

pytestmark = pytest.mark.unit


def test_defaults_separate_agent_and_judge_limits():
    config = JobBenchEvaluatorConfig()
    config.judge_endpoints = {config.judge_model_name: JudgeEndpoint("http://judge", 3)}
    config.apply_defaults(inference_instances=3, base_url="http://model")

    assert config.model_api_base == "http://model/v1"
    assert config.n_concurrent == 24
    assert config.max_tokens == 131072
    assert config.max_input_tokens is None
    assert config.judge_max_tokens == 16384
    assert config.judge_retries == 3
    assert config.judge_api_base == "http://judge/v1"
    assert config.judge_n_concurrent == 24
    assert config.agent == "mini-swe-agent"
    assert config.secondary_model_names == ["Qwen3.5-397B-A17B-FP8"]
    assert "venv_python" not in config.to_dict()
    assert config.to_dict()["judge-api-base"] == "http://judge/v1"


def test_defaults_preserve_explicit_endpoints_concurrency_and_context():
    config = JobBenchEvaluatorConfig(
        judge_model_name="judge",
        model_api_base="https://agent/v1",
        judge_api_base="https://judge/v1",
        n_concurrent=5,
        max_input_tokens=98304,
    )
    config.judge_endpoints = {config.judge_model_name: JudgeEndpoint("http://judge", 3)}
    config.apply_defaults(inference_instances=4, base_url="http://model")

    assert config.model_api_base == "https://agent/v1"
    assert config.judge_api_base == "https://judge/v1"
    assert config.n_concurrent == 5
    assert config.to_dict()["max-input-tokens"] == 98304


def test_serialization_forwards_shared_rli_settings_using_shooter_names():
    settings = {
        "request-retries": 4,
        "timeout-rli": 456,
        "transfer-retries": 2,
        "transfer-retry-backoff": 0.25,
        "session-ready-timeout": 123,
        "session-ready-poll-interval": 2,
        "enable-probe-warmup": False,
        "create-session-jitter-max-sec": 3,
        "probe-retry-backoff-sec": 4,
        "command-timeout": 567,
    }
    config = JobBenchEvaluatorConfig.model_validate(
        {"judge-model-name": "judge", "judge-n-concurrent": 5, **settings}
    )
    payload = config.to_shooter_config()

    for key, value in settings.items():
        assert payload[key.replace("-", "_")] == value
    assert config.to_dict()["judge-n-concurrent"] == 5
    assert payload["judge_n_concurrent"] == 5


@pytest.mark.parametrize(
    "values",
    [
        {"judge_model_name": ""},
        {"judge_model_name": "judge", "judge_retries": 0},
        {"judge_model_name": "judge", "judge_max_tokens": 0},
        {"judge_model_name": "judge", "max_input_tokens": 0},
        {"judge_model_name": "judge", "split": "unknown"},
        {"judge_model_name": "judge", "agent": "unknown"},
    ],
)
def test_rejects_invalid_settings(values):
    with pytest.raises(ValidationError):
        JobBenchEvaluatorConfig(**values)
