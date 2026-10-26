"""Model dimension contracts reject impossible arithmetic before runtime."""

import pytest

from torchtitan.config import run_pre_submit


@pytest.mark.parametrize(
    "recipe,path,value",
    [
        ("flux_debug", "model.config.num_heads", "0"),
        ("flux_debug", "model.config.num_heads", "-1"),
        ("flux_debug", "model.config.hidden_size", "0"),
        ("flux_debug", "model.config.hidden_size", "250"),
        ("flux_debug", "model.config.pe_config.axes_dim", "[1, 1, 30]"),
        ("flux_debug", "model.config.in_channels", "0"),
        ("flux_debug", "model.config.out_channels", "0"),
        ("flux_debug", "model.config.vec_in_dim", "0"),
        ("flux_debug", "model.config.context_in_dim", "0"),
        ("qwen3_vl_debug", "model.config.vision_encoder.out_hidden_size", "17"),
        ("qwen3_vl_debug", "model.config.vision_encoder.n_heads", "0"),
        ("qwen3_vl_debug", "model.config.vision_encoder.dim", "250"),
        ("qwen3_vl_debug", "model.config.vision_encoder.dim", "248"),
        ("qwen3_vl_debug", "model.config.vision_encoder.spatial_merge_size", "0"),
        (
            "qwen3_vl_debug",
            "model.config.vision_encoder.num_position_embeddings",
            "1000",
        ),
        ("qwen3_vl_debug", "model.config.vision_encoder.patch_embed.patch_size", "0"),
        ("qwen3_vl_debug", "model.config.mrope_section", "[8, 8]"),
        ("qwen3_vl_debug", "model.config.mrope_section", "[8, 12, 8]"),
        ("qwen3_vl_debug", "model.config.mrope_section", "[8, 8, 11]"),
    ],
)
def test_invalid_model_dimensions_report_written_field(recipe, path, value):
    result = run_pre_submit(f"configs/experiments/{recipe}.yaml", [(path, value)])
    assert not result.ok
    assert any(issue.path == path for issue in result.report.errors), (
        result.report.errors
    )


@pytest.mark.parametrize("field", ["in_channels", "out_channels"])
def test_flux_training_channels_must_agree_with_vae_packing(field):
    result = run_pre_submit(
        "configs/experiments/flux_debug.yaml", [(f"model.config.{field}", "32")]
    )
    assert not result.ok
    assert any(
        issue.path.endswith(field)
        and "packed autoencoder.z_channels (64)" in issue.message
        for issue in result.report.errors
    )


@pytest.mark.parametrize(
    "recipe", ["flux_debug", "qwen3_vl_debug", "qwen3_vl_moe_debug"]
)
def test_shipped_multimodal_dimensions_remain_valid(recipe):
    result = run_pre_submit(f"configs/experiments/{recipe}.yaml")
    assert result.ok, result.report.errors
