"""HF checkpoint requests fail before training for unrepresentable GIFT models."""

from copy import deepcopy
from dataclasses import replace

import pytest

from tests.unit_tests.torch_free import REPO_ROOT
from torchtitan.config import run_pre_submit
from torchtitan.config.decode import decode
from torchtitan.config.loader import load_raw_document
from torchtitan.config.sections import ModelSection
from torchtitan.config.schema.models.gift_moe._base.hf_config import build_hf_config
from torchtitan.config.schema.models.common.attention.linear.const import (
    GatedNormActivation,
)
from torchtitan.config.schema.components.lora import FrozenConfig


def _hybrid():
    raw = load_raw_document(str(REPO_ROOT / "configs/experiments/gift_3_5_debug.yaml"))
    return decode(ModelSection, raw.data["model"]).config.materialize()


@pytest.mark.parametrize("flag", ["save_in_hf", "last_save_in_hf"])
def test_requested_hf_save_rejects_mixed_norms(flag):
    result = run_pre_submit(
        str(REPO_ROOT / "configs/experiments/gift_3_5_debug.yaml"),
        [
            (
                "model.config.blocks.mla_moe.post_attn_norm",
                "!replace {type: rmsnorm, eps: 1e-6}",
            ),
            ("checkpoint.enable", "true"),
            ("checkpoint.last_save_model_only", "true"),
            (f"checkpoint.{flag}", "true"),
        ],
    )
    assert not result.report.ok
    assert any("norm class differs" in issue.message for issue in result.report.errors)


def test_training_without_hf_save_allows_mixed_norms():
    result = run_pre_submit(
        str(REPO_ROOT / "configs/experiments/gift_3_5_debug.yaml"),
        [
            (
                "model.config.blocks.mla_moe.post_attn_norm",
                "!replace {type: rmsnorm, eps: 1e-6}",
            )
        ],
    )
    assert result.report.ok, result.report.errors


@pytest.mark.parametrize("recipe", ["gift_moe_debug", "gift_3_5_debug", "gift_4_debug"])
def test_supported_gift_hf_requests_pass(recipe):
    result = run_pre_submit(
        str(REPO_ROOT / f"configs/experiments/{recipe}.yaml"),
        [
            ("checkpoint.enable", "true"),
            ("checkpoint.last_save_model_only", "true"),
            ("checkpoint.last_save_in_hf", "true"),
        ],
    )
    assert result.report.ok, result.report.errors


def test_disabled_checkpoint_ignores_inherited_hf_save_flag():
    result = run_pre_submit(
        str(REPO_ROOT / "configs/experiments/gift_3_5_debug.yaml"),
        [
            (
                "model.config.blocks.mla_moe.post_attn_norm",
                "!replace {type: rmsnorm, eps: 1e-6}",
            ),
            ("checkpoint.enable", "false"),
            ("checkpoint.last_save_in_hf", "true"),
        ],
    )
    assert result.report.ok, result.report.errors


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("gate_rank", 8, "hardcodes rank 16"),
        ("gating_weight", 3.0, "gating_weight differs"),
    ],
)
def test_norm_contract_rejects_unrepresentable_gated_settings(field, value, message):
    model = _hybrid()
    model.layers[0].post_attn_norm = replace(
        model.layers[0].post_attn_norm, **{field: value}
    )
    with pytest.raises(ValueError, match=message):
        build_hf_config(model)
    assert any(message in issue.message for issue in model.validate_hf_export())


def test_hf_contract_rejects_one_sided_sandwich_norm():
    model = _hybrid()
    model.layers[0].post_attn_norm = None
    with pytest.raises(ValueError, match="only one of post_attn_norm"):
        build_hf_config(model)
    assert model.validate_hf_export()


def test_gdn_output_eps_has_a_separate_uniform_hf_field():
    model = _hybrid()
    gdn_layers = [layer for layer in model.layers if hasattr(layer.attention, "o_norm")]
    assert len(gdn_layers) > 1
    for layer in gdn_layers:
        layer.attention.o_norm = replace(layer.attention.o_norm, eps=1e-4)
    fields = build_hf_config(model)
    assert fields["linear_attn_o_norm_eps"] == 1e-4
    assert fields["rms_norm_eps"] == model.norm.eps
    assert model.validate_hf_export() == []
    gdn_layers[-1].attention.o_norm = replace(gdn_layers[-1].attention.o_norm, eps=1e-3)
    with pytest.raises(ValueError, match="GDN o_norm.eps differs"):
        build_hf_config(model)


def test_gdn_silu_output_multiplier_is_not_representable_in_hf():
    model = deepcopy(_hybrid())
    gdn = next(
        layer.attention for layer in model.layers if hasattr(layer.attention, "o_norm")
    )
    gdn.o_norm = replace(
        gdn.o_norm, activation=GatedNormActivation.SILU, zero_centered=False
    )
    gdn.o_norm_output_multiplier = 2.0
    with pytest.raises(ValueError, match="output_multiplier != 1"):
        build_hf_config(model)
    assert model.validate_hf_export()


@pytest.mark.parametrize("target", ["norm", "attention", "post_attn_norm", "layer"])
def test_frozen_modules_preserve_the_hf_contract(target):
    model = _hybrid()
    expected = build_hf_config(model)
    if target == "norm":
        model.norm = FrozenConfig(inner=model.norm)
    elif target == "layer":
        model.layers[0] = FrozenConfig(inner=model.layers[0])
    else:
        child = getattr(model.layers[0], target)
        setattr(model.layers[0], target, FrozenConfig(inner=child))
    assert build_hf_config(model) == expected
    assert model.validate_hf_export() == []
