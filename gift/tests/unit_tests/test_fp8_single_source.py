"""FP8 YAML settings live in one section; module flags are derived runtime state."""

from pathlib import Path

import pytest

from torchtitan.config import load_document, run_pre_submit
from torchtitan.config.build import prepare_document
from torchtitan.config.save import get_typed_document
from torchtitan.config.schema.models.common.ffn import FeedForwardConfig
from torchtitan.config.schema.models.common.moe.experts.gmm import GroupedExpertsConfig

EXPERIMENT = "configs/experiments/gift_moe_debug_deepep_fp8.yaml"
MODULE_FLAGS = [
    "model.config.blocks.mla_dense.feed_forward.use_shared_quantizer",
    "model.config.blocks.mla_dense.feed_forward.dense_fused_swiglu_quant",
    "model.config.blocks.mla_moe.moe.experts.use_float8_grouped_gemm",
    "model.config.blocks.mla_moe.moe.experts.deep_gemm_num_sms",
    "model.config.blocks.mla_moe.moe.experts.sparse_fused_swiglu_quant",
]
FP8_SETTINGS = [
    "fp8.dense_use_shared_quantizer",
    "fp8.dense_fused_swiglu_quant",
    "fp8.sparse_use_grouped_gemm",
    "fp8.sparse_deep_gemm_num_sms",
    "fp8.sparse_fused_swiglu_quant",
]


@pytest.mark.parametrize("path", MODULE_FLAGS)
@pytest.mark.parametrize("value", ["true", "null"])
def test_module_fp8_settings_are_rejected_even_as_resets(path, value):
    result = run_pre_submit(EXPERIMENT, [(path, value)])
    assert not result.ok
    issue = next(issue for issue in result.report.errors if issue.path == path)
    assert (
        f"Configure {FP8_SETTINGS[MODULE_FLAGS.index(path)]} instead" in issue.message
    )


def test_resolved_bundle_keeps_fp8_policy_without_duplicated_module_flags(
    tmp_path, monkeypatch
):
    from torchtitan.config import save
    from torchtitan.config.loader import parse_yaml_value
    from torchtitan.config.validation import RuntimeInfo

    document, _ = load_document(
        EXPERIMENT,
        [
            ("fp8.sparse_deep_gemm_num_sms", "112"),
            ("fp8.sparse_fused_swiglu_quant", "false"),
        ],
    )
    prepared = prepare_document(document)
    # Derived state cannot leak even when somebody sets it programmatically.
    prepared.model_section.config.blocks[
        "mla_dense"
    ].feed_forward.use_shared_quantizer = True
    prepared.model_section.config.blocks[
        "mla_moe"
    ].moe.experts.use_float8_grouped_gemm = True
    prepared.model_section.config.blocks["mla_moe"].moe.experts.deep_gemm_num_sms = 99
    typed = get_typed_document(prepared.trainer_config, prepared.model_section)
    assert typed["fp8"]["sparse_deep_gemm_num_sms"] == 112
    assert typed["fp8"]["sparse_fused_swiglu_quant"] is False
    assert (
        "use_shared_quantizer"
        not in typed["model"]["config"]["blocks"]["mla_dense"]["feed_forward"]
    )
    experts = typed["model"]["config"]["blocks"]["mla_moe"]["moe"]["experts"]
    assert (
        not {
            "use_float8_grouped_gemm",
            "deep_gemm_num_sms",
            "sparse_fused_swiglu_quant",
        }
        & experts.keys()
    )
    monkeypatch.setattr(save, "collect_metadata", lambda **kwargs: {})
    path = save.save_run_bundle(
        prepared.trainer_config,
        document.data,
        dest=str(tmp_path),
        command=["train"],
        runtime=RuntimeInfo(),
    )
    resolved = parse_yaml_value(Path(path).read_text(), source="resolved")
    assert resolved["fp8"] == typed["fp8"]
    assert run_pre_submit(path).ok


def test_derived_module_flags_survive_config_copy():
    ffn = FeedForwardConfig(dim=512, hidden_dim=1024)
    ffn.use_shared_quantizer = True
    ffn.dense_fused_swiglu_quant = False
    copied = ffn._copy()
    assert copied.use_shared_quantizer is True
    assert copied.dense_fused_swiglu_quant is False
    experts = GroupedExpertsConfig(dim=512, hidden_dim=1024, num_experts=4)
    experts.use_float8_grouped_gemm = True
    experts.deep_gemm_num_sms = 112
    experts.sparse_fused_swiglu_quant = False
    copied = experts._copy()
    assert copied.use_float8_grouped_gemm is True
    assert copied.deep_gemm_num_sms == 112
    assert copied.sparse_fused_swiglu_quant is False


def test_earlier_resolved_bundle_module_defaults_have_an_actionable_migration_error(
    tmp_path,
):
    from torchtitan.config.save import write_yaml

    document, _ = load_document(EXPERIMENT)
    prepared = prepare_document(document)
    earlier = get_typed_document(prepared.trainer_config, prepared.model_section)
    earlier["model"]["config"]["blocks"]["mla_dense"]["feed_forward"].update(
        use_shared_quantizer=False, dense_fused_swiglu_quant=True
    )
    earlier["model"]["config"]["blocks"]["mla_moe"]["moe"]["experts"].update(
        use_float8_grouped_gemm=False,
        deep_gemm_num_sms=116,
        sparse_fused_swiglu_quant=True,
    )
    path = tmp_path / "resolved_config.yaml"
    write_yaml(earlier, str(path))
    result = run_pre_submit(str(path))
    assert not result.ok
    issue = next(issue for issue in result.report.errors if issue.path in MODULE_FLAGS)
    assert (
        f"Configure {FP8_SETTINGS[MODULE_FLAGS.index(issue.path)]} instead"
        in issue.message
    )
