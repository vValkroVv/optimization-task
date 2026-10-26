"""``norm_unit_scale`` starts every norm at scale one: a zero weight when the norm is
``zero_centered``, one when plain."""

import pytest
import torch

from tests.unit_tests.config_helpers import load_model_spec, PRESET_ROOT
from torchtitan.config import ConfigSchemaError, deep_merge, load_raw_document
from torchtitan.config.decode import decode
from torchtitan.config.schema.models.common.init_policy import _norm_unit_scale
from torchtitan.config.schema.models.common.nn_modules import LinearConfig
from torchtitan.config.schema.models.common.norms.fused_gated_rmsnorm import (
    FusedRMSNormGatedConfig,
)
from torchtitan.config.schema.models.common.norms.gated_norm import GatedRMSNormConfig
from torchtitan.config.schema.models.common.norms.rmsnorm import RMSNormConfig
from torchtitan.config.schema.models.common.param_init import NormUnitScale
from torchtitan.config.schema.protocols.module import iter_module_configs, Ones, Zeros
from torchtitan.config.sections import ModelSection

GIFT_3_5 = "gift_moe/gift_3_5/{}.yaml"
GIFT_4 = "gift_moe/gift_4/{}.yaml"
SIZES = ("debugmodel", "3b", "700b")


def _compact(preset: str, overrides: dict | None = None):
    """The preset's decoded ``model.config``, *overrides* layered on it."""
    model = deep_merge(
        load_raw_document(str(PRESET_ROOT / preset)).data["model"],
        {"config": overrides or {}},
    )
    return decode(ModelSection, model, path="model").config


def _parametrized_norms(tree):
    return [
        (fqn, m)
        for fqn, m in iter_module_configs(tree)
        if isinstance(m, (GatedRMSNormConfig, FusedRMSNormGatedConfig))
    ]


@pytest.mark.parametrize(
    "module, expected",
    [
        (GatedRMSNormConfig(zero_centered=True), Zeros()),
        (GatedRMSNormConfig(zero_centered=False), Ones()),
        (FusedRMSNormGatedConfig(zero_centered=True), Zeros()),
        (FusedRMSNormGatedConfig(zero_centered=False), Ones()),
        (RMSNormConfig(), Ones()),
    ],
)
def test_norm_unit_scale_follows_the_norm(module, expected):
    assert _norm_unit_scale(NormUnitScale(), module, "x.norm") == expected
    # Any other spec passes through.
    assert _norm_unit_scale(Zeros(), module, "x.norm") == Zeros()


def test_norm_unit_scale_is_for_norms_only():
    linear = LinearConfig(in_features=2, out_features=2)
    with pytest.raises(ValueError, match="x.wo: norm_unit_scale"):
        _norm_unit_scale(NormUnitScale(), linear, "x.wo")


@pytest.mark.parametrize(
    "preset, overrides",
    [
        *((GIFT_3_5.format(size), {}) for size in SIZES),
        *((GIFT_4.format(size), {}) for size in SIZES),
        # A norm's parametrization flipped in YAML alone.
        (GIFT_3_5.format("debugmodel"), {"norm": {"zero_centered": False}}),
    ],
)
def test_every_norm_starts_at_unit_scale(preset, overrides):
    tree = _compact(preset, overrides).materialize()
    norms = _parametrized_norms(tree)
    assert norms
    for fqn, norm in norms:
        assert norm.param_init is not None
        if preset.startswith("gift_moe/gift_4/") and fqn.endswith("post_attn_norm"):
            # gift_4 starts this branch switched off (plain, zero weight).
            assert not norm.zero_centered and norm.param_init["weight"] == Zeros()
            continue
        unit = Zeros() if norm.zero_centered else Ones()
        assert norm.param_init["weight"] == unit, fqn


@pytest.mark.parametrize("size", SIZES)
def test_gift_4_norms_are_plain(size):
    tree = _compact(GIFT_4.format(size)).materialize()
    assert all(not n.zero_centered for _, n in _parametrized_norms(tree))


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"norm": {"zero_centered": True}}, "norm class differs"),
        (
            {
                "blocks": {
                    "gdn_dense": {"attention": {"o_norm": {"zero_centered": True}}}
                }
            },
            "GDN gating type differs",
        ),
    ],
)
def test_gift_4_zero_centered_norm_is_kept_and_rejected_by_hf_export(
    overrides, message
):
    """gift_4 is a gift_3_5 preset: an explicit centered norm is the user's
    choice, and only the HF export contract (one norm kind) refuses it."""
    tree = _compact(GIFT_4.format("debugmodel"), overrides).materialize()
    issues = tree.validate_hf_export()
    assert any(message in issue.message for issue in issues), issues


def _mla_attentions(tree):
    from torchtitan.config.schema.models.common.attention.softmax.mla import MLAConfig

    return [
        (fqn, m) for fqn, m in iter_module_configs(tree) if isinstance(m, MLAConfig)
    ]


def test_q_norm_is_the_kv_norms_twin():
    """The low-rank presets write the q norm with the kv norm's settings; it
    takes the q latent's width, not the kv one's."""
    attentions = _mla_attentions(_compact(GIFT_4.format("700b")).materialize())
    assert attentions
    for fqn, attention in attentions:
        assert attention.q_lora_rank > 0, fqn
        assert isinstance(attention.q_norm, GatedRMSNormConfig), fqn
        assert isinstance(attention.kv_norm, GatedRMSNormConfig), fqn
        assert not attention.q_norm.zero_centered, fqn
        assert attention.q_norm.eps == attention.kv_norm.eps == 1e-6, fqn
        assert attention.q_norm.normalized_shape == attention.q_lora_rank, fqn

    hybrid = _mla_attentions(_compact(GIFT_3_5.format("700b")).materialize())
    assert hybrid
    for fqn, attention in hybrid:
        assert isinstance(attention.q_norm, GatedRMSNormConfig), fqn
        assert attention.q_norm.zero_centered, fqn
    # Without a q low-rank path there is no q norm.
    debug = _mla_attentions(_compact(GIFT_4.format("debugmodel")).materialize())
    assert debug and all(a.q_norm is None for _, a in debug)


def test_an_explicit_q_norm_is_kept():
    q_norm = {"type": "gated_rmsnorm", "zero_centered": False, "eps": 1e-6}
    attention = (
        _compact(
            GIFT_3_5.format("700b"),
            {"blocks": {"mla_moe": {"attention": {"q_norm": q_norm}}}},
        )
        .blocks["mla_moe"]
        .attention
    )
    assert not attention.q_norm.zero_centered and attention.kv_norm.zero_centered


@pytest.mark.parametrize("removed", ["gift_4_mla", "gift_4_gdn"])
def test_gift_4_attention_types_are_gone(removed):
    block = "mla_moe" if removed.endswith("mla") else "gdn_dense"
    with pytest.raises(ConfigSchemaError, match=removed):
        _compact(
            GIFT_4.format("debugmodel"),
            {"blocks": {block: {"attention": {"type": removed}}}},
        )


def test_layernorm_bias_has_an_init():
    """A norm with a bias (``layernorm``) in a ``norm`` role gets an init for
    it from the policy and builds and initializes."""
    spec = load_model_spec(
        "llama3/debugmodel.yaml", {"config": {"norm": {"type": "layernorm"}}}
    )
    norms = [m for fqn, m in iter_module_configs(spec.model) if fqn.endswith("norm")]
    assert norms
    for norm in norms:
        assert norm.param_init is not None
        assert norm.param_init["bias"] == Zeros()

    model = spec.model.build()
    model.init_weights()
    assert torch.equal(model.norm.bias, torch.zeros_like(model.norm.bias))
