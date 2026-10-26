"""Backend selection follows model capability metadata at both validation stages."""

from typing import Any, cast, ClassVar

import pytest

from torchtitan.config.cli import load_document
from torchtitan.config.decode import decode
from torchtitan.config.pre_submit import run_pre_submit
from torchtitan.config.schema.distributed.sharding_backend import (
    model_sharding_backend_issues,
)
from torchtitan.config.schema.protocols.model import BaseModelConfig
from torchtitan.config.configurable import config
from torchtitan.config.sections import ModelSection


@config
class NewModelConfig(BaseModelConfig):
    supported_sharding_backends: ClassVar[tuple[str, ...]] = ("fsdp2", "gfsdp2")


def test_new_model_declares_capabilities_without_changing_core_family_rules():
    assert model_sharding_backend_issues("gfsdp2", NewModelConfig()) == []
    assert model_sharding_backend_issues("gfsdp2", BaseModelConfig())


@pytest.mark.parametrize(
    "experiment", ["gift_moe_debug.yaml", "gift_3_5_debug.yaml", "gift_4_debug.yaml"]
)
def test_gift_variants_inherit_gfsdp_capability(experiment):
    document, _ = load_document("configs/experiments/" + experiment)
    model = decode(ModelSection, document.data["model"]).config
    assert model_sharding_backend_issues("gfsdp2", model) == []
    assert model_sharding_backend_issues("fsdp2", model) == []


@pytest.mark.parametrize(
    "experiment",
    [
        "llama3_debug.yaml",
        "qwen3_debug.yaml",
        "deepseek_v3_debug.yaml",
        "flux_debug.yaml",
    ],
)
def test_unimplemented_model_backends_fail_before_runtime(experiment):
    result = run_pre_submit(
        "configs/experiments/" + experiment,
        [("parallelism.fsdp_backend", "gfsdp2")],
    )
    issues = [i for i in result.report.errors if i.path == "parallelism.fsdp_backend"]
    assert len(issues) == 1
    assert "not supported by model type" in issues[0].message


def test_backend_validation_requires_config_instead_of_family_string():
    with pytest.raises(ValueError, match="requires a model config"):
        model_sharding_backend_issues("gfsdp2", cast(Any, "gift_moe"))


def test_rl_policy_reports_model_capability_at_its_nested_config_path():
    result = run_pre_submit(
        "configs/experiments/rl_grpo_qwen3_0_6b.yaml",
        [("policy_trainer.parallelism.fsdp_backend", "gfsdp2")],
    )
    matching = [
        i for i in result.report.errors if "not supported by model type" in i.message
    ]
    assert len(matching) == 1
    assert matching[0].path == "policy_trainer.parallelism.fsdp_backend"


def test_yaml_cannot_override_runtime_capability_metadata():
    result = run_pre_submit(
        "configs/experiments/llama3_debug.yaml",
        [("model.config.supported_sharding_backends", "[fsdp2,gfsdp2]")],
    )
    assert not result.ok
    assert any(
        "supported_sharding_backends" in f"{i.path}: {i.message}"
        for i in result.report.errors
    )
