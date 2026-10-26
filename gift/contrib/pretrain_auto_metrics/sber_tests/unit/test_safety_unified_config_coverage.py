from pathlib import Path

import pytest
import yaml

from node_runner.src.config_model import (
    MODEL_NAME_PARAMS_MAPPER,
    SafetyUnifiedEvaluatorConfig,
    SglangInferenceConfig,
)
from sber_tests.test_support import judge_endpoints

pytestmark = pytest.mark.unit


def _config_paths():
    configs = Path(__file__).resolve().parents[2] / "configs"
    paths = sorted(configs.glob("*.yaml"))
    paths += sorted((configs / "alignment_prod" / "full").glob("*.yaml"))
    paths += sorted((configs / "alignment_prod" / "general").glob("*.yaml"))
    return paths


@pytest.mark.parametrize("path", _config_paths(), ids=str)
def test_base_and_safety_alignment_configs_use_registered_production_judge(path):
    autometrics = yaml.safe_load(path.read_text())["autometrics"]
    evaluators = autometrics["evaluators"]
    assert "safety_unified" in evaluators
    config = SafetyUnifiedEvaluatorConfig.model_validate(evaluators["safety_unified"])
    assert config.judge == {}
    assert config.judge_parallel is None
    assert config.judge_prompt_file == "judge_prompt_production.txt"
    assert config.judge_schema == "production"
    assert config.selection_manifest == "big_pipeline.json"
    assert config.domain_rules_file == "domain_rules.json"
    assert set(config.secondary_model_names) == {
        "Qwen3-235B-A22B-Instruct-2507-FP8",
        "Qwen3.5-397B-A17B-FP8",
    }
    for name in config.secondary_model_names:
        assert name in MODEL_NAME_PARAMS_MAPPER
        assert name not in autometrics.get("secondary_inference", {})
        server = SglangInferenceConfig.for_secondary_preset(
            name, autometrics.get("secondary_inference", {})
        )
        expected = SglangInferenceConfig.model_validate(MODEL_NAME_PARAMS_MAPPER[name])
        assert server.to_dict() == expected.to_dict()
        assert server.to_dict()["context-length"] >= 131072
    config.judge_endpoints = judge_endpoints(
        {name: "http://judge.test:31000" for name in config.secondary_model_names}
    )
    config.apply_defaults(
        inference_instances=1,
        base_url="http://candidate.test:30000",
    )
    overrides = config.build_overrides(
        judge_model_name=config.secondary_model_names[0],
        out_dir=Path("metrics/first"),
        default_answers_dir=Path("metrics/answers"),
        checkpoint_model_name="candidate",
        checkpoint_path=Path("checkpoints/candidate"),
    )
    assert overrides["judge"]["parallel"] == 8
    assert overrides["judge"]["max_tokens"] == 1536
    assert overrides["judge"]["temperature"] == 0.0
    assert overrides["judge"]["seed"] == 12345
    assert "runtime_revision" not in overrides["judge"]
