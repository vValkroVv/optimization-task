"""GIFT HF constraints fail during torch-free validation, before export.

The norm eps uniformity is part of the HF export contract (``build_hf_config``),
so it is checked whenever the run asks for HF checkpoints.
"""

import pytest

from torchtitan.config import run_pre_submit

_HF_EXPORT = [
    ("checkpoint.enable", "true"),
    ("checkpoint.last_save_in_hf", "true"),
    ("checkpoint.last_save_model_only", "true"),
]
_EPS_MESSAGE = "hf_config: norm eps differs between layers"


@pytest.mark.parametrize("family", ["gift_moe", "gift_3_5", "gift_4"])
@pytest.mark.parametrize("field", ["norm", "blocks.mla_moe.attention.kv_norm"])
def test_gift_norm_eps_mismatch_is_rejected_before_export(family, field):
    experiment = {
        "gift_moe": "gift_moe_debug",
        "gift_3_5": "gift_3_5_debug",
        "gift_4": "gift_4_debug",
    }[family]
    result = run_pre_submit(
        f"configs/experiments/{experiment}.yaml",
        [*_HF_EXPORT, (f"model.config.{field}.eps", "2e-5")],
    )
    issues = [issue for issue in result.report.errors if _EPS_MESSAGE in issue.message]
    assert issues
    assert all(issue.path == "model.config" for issue in issues)


@pytest.mark.parametrize("family", ["gift_moe", "gift_3_5"])
def test_gift_norm_eps_mismatch_is_allowed_without_hf_export(family):
    experiment = {"gift_moe": "gift_moe_debug", "gift_3_5": "gift_3_5_debug"}[family]
    result = run_pre_submit(
        f"configs/experiments/{experiment}.yaml", [("model.config.norm.eps", "2e-5")]
    )
    assert not any(_EPS_MESSAGE in issue.message for issue in result.report.errors)


@pytest.mark.parametrize(
    "experiment", ["gift_moe_debug", "gift_3_5_debug", "gift_4_debug"]
)
def test_gift_defaults_have_consistent_export_eps(experiment):
    result = run_pre_submit(f"configs/experiments/{experiment}.yaml", _HF_EXPORT)
    assert result.ok, result.report.issues


@pytest.mark.parametrize(
    "experiment", ["gift_moe_debug", "gift_3_5_debug", "gift_4_debug"]
)
def test_every_materialized_norm_role_is_checked_including_mtp(experiment):
    from torchtitan.config.cli import load_document
    from torchtitan.config.decode import decode
    from torchtitan.config.schema.models.common.norms.base import BaseNormConfig
    from torchtitan.config.schema.protocols.module import iter_module_configs
    from torchtitan.config.sections import ModelSection

    mtp_block = "mla_moe" if experiment == "gift_moe_debug" else "mtp"
    document, _ = load_document(
        f"configs/experiments/{experiment}.yaml",
        [("model.config.mtp.layers", f"[{{block: {mtp_block}}}]")],
    )
    model = decode(
        ModelSection, document.data["model"], path="model"
    ).config.materialize()
    checked = set()
    for path, norm in iter_module_configs(model):
        # The GDN output norm exports its own eps (linear_attn_o_norm_eps).
        if not isinstance(norm, BaseNormConfig) or path.endswith("o_norm"):
            continue
        previous = norm.eps
        norm.eps = 2e-5
        try:
            issues = model.validate_hf_export()
            assert any(_EPS_MESSAGE in issue.message for issue in issues), path
        finally:
            norm.eps = previous
        checked.add(path)
    assert "mtp.layers.0.token_norm" in checked
    assert "mtp.layers.0.hidden_norm" in checked
    assert "mtp.layers.0.final_norm" in checked
    assert "mtp.layers.0.block.attention_norm" in checked
