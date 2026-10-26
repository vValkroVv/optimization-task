"""Placed MoEs obey the batching contract of their runtime optimizer hook."""

import pytest

from torchtitan.config import load_raw_document
from torchtitan.config.decode import decode
from torchtitan.config.pre_submit import run_pre_submit
from torchtitan.config.schema.components.lora import FrozenConfig
from torchtitan.config.sections import ModelSection

RECIPE = "configs/experiments/gift_3_5_debug.yaml"
PREFIX = "model.config.blocks"


def _check(*overrides):
    return run_pre_submit(RECIPE, [("checkpoint.enable", "false"), *overrides])


@pytest.mark.parametrize(
    "override,message",
    [
        ((f"{PREFIX}.gdn_moe.moe.load_balance_coeff", "0.0"), "every placed MoE"),
        ((f"{PREFIX}.mla_moe.moe.load_balance_coeff", "0.0"), "every placed MoE"),
        ((f"{PREFIX}.gdn_moe.moe.experts.num_experts", "4"), "equal expert counts"),
    ],
)
def test_public_pre_submit_rejects_unsupported_batch(override, message):
    result = _check(override)
    assert not result.ok
    assert any(message in issue.message for issue in result.report.issues)


def test_positive_coefficients_need_not_be_equal():
    result = _check((f"{PREFIX}.gdn_moe.moe.load_balance_coeff", "0.005"))
    assert result.ok, result.report.issues


def test_all_disabled_can_have_different_expert_counts():
    result = _check(
        (f"{PREFIX}.gdn_moe.moe.load_balance_coeff", "0.0"),
        (f"{PREFIX}.mla_moe.moe.load_balance_coeff", "0.0"),
        (f"{PREFIX}.gdn_moe.moe.experts.num_experts", "4"),
    )
    assert result.ok, result.report.issues


def test_mtp_moes_participate_in_the_same_contract():
    result = _check(
        ("model.config.mtp.layers", "[{block: gdn_moe}]"),
        ("model.config.layers", "[{block: mla_moe}]"),
        (f"{PREFIX}.gdn_moe.moe.experts.num_experts", "4"),
    )
    assert not result.ok
    assert any("mtp.layers" in issue.message for issue in result.report.issues)


def test_frozen_completed_blocks_keep_the_balance_contract():
    raw = load_raw_document(RECIPE)
    tree = decode(ModelSection, raw.data["model"]).config.materialize()
    first_moe = next(layer for layer in tree.layers if layer.moe is not None)
    index = tree.layers.index(first_moe)
    tree.layers[index] = FrozenConfig(inner=first_moe)
    tree.validate_materialized()
    first_moe.moe.load_balance_coeff = 0.0
    with pytest.raises(ValueError, match="every placed MoE"):
        tree.validate_materialized()


@pytest.mark.parametrize(
    "recipe",
    [
        "deepseek_v3_debug",
        "gift_3_5_debug",
        "gpt_oss_debug",
        "llama4_debug",
        "qwen3_moe_debug",
    ],
)
def test_every_moe_tree_meets_the_batched_contract(recipe):
    raw = load_raw_document(f"configs/experiments/{recipe}.yaml")
    written = decode(ModelSection, raw.data["model"]).config
    written.materialize().validate_materialized()
