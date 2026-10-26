"""Model owners reject missing MTP plans and unsupported attention kernels."""

import pytest

from torchtitan.config import load_raw_document, run_pre_submit


def _mtp_run(recipe, *overrides):
    path = f"configs/experiments/{recipe}.yaml"
    document = load_raw_document(path)
    block = next(iter(document.data["model"]["config"]["blocks"]))
    return run_pre_submit(
        path,
        [
            ("checkpoint.enable", "false"),
            ("model.config.mtp.layers", f"[{{block: {block}}}]"),
            ("loss", "!replace {type: mtp, loss_weights: [1.0]}"),
            ("parallelism.enable_sequence_parallel", "false"),
            *overrides,
        ],
    )


@pytest.mark.parametrize(
    "recipe",
    [
        "llama3_debug",
        "qwen3_debug",
        "deepseek_v3_debug",
        "llama4_debug",
        "gpt_oss_debug",
        "qwen3_vl_debug",
    ],
)
@pytest.mark.parametrize(
    "degree",
    [
        "tensor_parallel_degree",
        "pipeline_parallel_degree",
        "expert_parallel_degree",
    ],
)
def test_non_gift_mtp_rejects_missing_model_parallel_plan(recipe, degree):
    result = _mtp_run(recipe, (f"parallelism.{degree}", "2"))
    assert not result.ok
    assert any(
        issue.path == f"parallelism.{degree}" and "MTP model-parallel" in issue.message
        for issue in result.report.errors
    ), result.report.issues


@pytest.mark.parametrize(
    "recipe",
    [
        "llama3_debug",
        "qwen3_debug",
        "deepseek_v3_debug",
        "llama4_debug",
        "gpt_oss_debug",
        "qwen3_vl_debug",
    ],
)
def test_non_gift_mtp_remains_available_without_model_parallelism(recipe):
    result = _mtp_run(recipe)
    assert result.ok, result.report.issues


@pytest.mark.parametrize(
    "degree",
    [
        "tensor_parallel_degree",
        "pipeline_parallel_degree",
        "expert_parallel_degree",
    ],
)
def test_gift_mtp_runtime_has_model_parallel_plans(degree):
    result = _mtp_run("gift_moe_debug", (f"parallelism.{degree}", "2"))
    assert result.ok, result.report.issues


def test_gpt_oss_rejects_sdpa_even_when_every_block_uses_it():
    path = "configs/experiments/gpt_oss_debug.yaml"
    document = load_raw_document(path)
    overrides = [
        (f"model.config.blocks.{name}.attention.inner_attention.type", "sdpa")
        for name in document.data["model"]["config"]["blocks"]
    ]
    result = run_pre_submit(path, overrides)
    assert not result.ok
    assert any(
        "requires a flex kernel" in issue.message for issue in result.report.errors
    )


def test_gpt_oss_accepts_frozen_flex_kernel():
    from torchtitan.config.schema.components.lora import FrozenConfig
    from torchtitan.config.schema.models.gpt_oss.model import AttentionConfig
    from torchtitan.config.schema.models.common.attention.softmax.kernels.flex import (
        FlexAttentionConfig,
    )

    config = AttentionConfig(
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        sliding_window_size=16,
        inner_attention=FrozenConfig(inner=FlexAttentionConfig()),
    )
    assert isinstance(config.inner_attention, FrozenConfig)
