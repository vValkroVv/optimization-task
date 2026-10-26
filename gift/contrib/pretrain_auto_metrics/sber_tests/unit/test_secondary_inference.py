import pytest

from node_runner.src.config_model import (
    ExternalApiInferenceConfig,
    NodeRunnerConfigModel,
)

pytestmark = pytest.mark.unit


def _node_runner_config(**autometrics_extra) -> dict:
    return {
        "run_name": "run_name",
        "max_seq_len": 4096,
        "autometrics": {
            "globals": {},
            "router": {},
            "inference": {"tp": 1},
            "obm_proxy": {},
            "evaluators": {
                "obm_judged": {
                    "obm_config": "openai",
                    "all": {"tasks": ["pollux"]},
                    "judges": {
                        "openrouter:qwen/qwen3-235b": ["pollux"],
                        "pollux-judge-32b": ["side_by_side"],
                    },
                },
                "dredd": {"judge": {"model_name": "openrouter:openai/gpt-5"}},
            },
            "preparation": {},
            "convertation": {"mappers": ["map_fsdp_to_hf"]},
            "model_info": {
                "model_class": "sft",
                "size": "lite",
                "username": "user",
            },
            **autometrics_extra,
        },
    }


def test_build_merges_defaults_user_params_and_model():
    cfg = ExternalApiInferenceConfig.for_secondary_platform(
        "openrouter",
        "qwen/qwen3-235b",
        {"openrouter": {"api_key": "sk-or-test", "max_concurrency": 42}},
    )
    # /v1 suffix is normalized away by ExternalApiInferenceConfig
    assert cfg.base_url == "https://openrouter.ai/api"
    assert cfg.model == "qwen/qwen3-235b"
    assert cfg.api_key == "sk-or-test"
    assert cfg.max_concurrency == 42
    assert cfg.formatting == "openai"


def test_ift_platform_defaults():
    cfg = ExternalApiInferenceConfig.for_secondary_platform(
        "ift", "GigaChat-2-Max", {"ift": {"user_pass": "u:p"}}
    )
    assert cfg.formatting == "ift"
    assert cfg.base_url == "https://gigachat.ift.sberdevices.ru"
    assert cfg.model == "GigaChat-2-Max"


def test_build_fails_without_api_key():
    with pytest.raises(ValueError, match="api_key is required"):
        ExternalApiInferenceConfig.for_secondary_platform(
            "openrouter", "qwen/qwen3-235b", {}
        )


def test_build_fails_for_unknown_platform():
    with pytest.raises(ValueError, match="unknown external API platform 'myapi'"):
        ExternalApiInferenceConfig.for_secondary_platform("myapi", "some/model", {})


def test_build_fails_for_unknown_platform_without_base_url():
    with pytest.raises(ValueError, match="base_url"):
        ExternalApiInferenceConfig.for_secondary_platform(
            "myapi", "some/model", {"myapi": {"api_key": "sk"}}
        )


def test_build_allows_custom_platform_with_full_user_params():
    cfg = ExternalApiInferenceConfig.for_secondary_platform(
        "myapi",
        "some/model",
        {"myapi": {"api_key": "sk", "base_url": "https://myapi.example/v1"}},
    )
    assert cfg.base_url == "https://myapi.example"
    assert cfg.model == "some/model"


def test_build_allows_custom_platform_with_ift_formatting_and_user_pass():
    cfg = ExternalApiInferenceConfig.for_secondary_platform(
        "myapi",
        "some-model",
        {
            "myapi": {
                "base_url": "https://myapi.example",
                "formatting": "ift",
                "user_pass": "user:pass",
                "request_customization": {"temperature": 0.0},
            },
        },
    )
    assert cfg.formatting == "ift"
    assert cfg.user_pass == "user:pass"
    assert cfg.api_key is None
    assert cfg.request_customization == {"temperature": 0.0}


def test_secondary_inference_secrets_dropped_on_dump():
    config = NodeRunnerConfigModel.model_validate(
        _node_runner_config(
            secondary_inference={
                "openrouter": {
                    "api_key": "sk-or-secret",
                    "max_concurrency": 42,
                },
                "ift": {"user_pass": "u:p"},
            }
        )
    )
    dumped = config.model_dump(
        by_alias=True, exclude_none=True, mode="json", round_trip=True
    )
    dumped_secondary = dumped["autometrics"]["secondary_inference"]
    assert dumped_secondary["openrouter"] == {"max_concurrency": 42}
    assert dumped_secondary["ift"] == {}
    # in-memory config keeps the secrets
    assert (
        config.autometrics.secondary_inference["openrouter"]["api_key"]
        == "sk-or-secret"
    )
