"""HF export requirements apply only when HF checkpoints will be saved."""

import pytest

from torchtitan.config import run_pre_submit


@pytest.mark.parametrize(
    "experiment, field, value, error",
    [
        (
            "gift_3_5_debug",
            "blocks.mla_moe.attention.kv_norm.zero_centered",
            "false",
            "norm class differs",
        ),
        ("llama3_debug", "converters", "[{type: lora}]", "unmerged LoRA"),
    ],
)
@pytest.mark.parametrize("hf_flag", ["save_in_hf", "last_save_in_hf"])
@pytest.mark.parametrize("load_only", [False, True])
def test_load_only_skips_export_guards(
    experiment, field, value, error, hf_flag, load_only
):
    result = run_pre_submit(
        f"configs/experiments/{experiment}.yaml",
        [
            (f"model.config.{field}", value),
            ("checkpoint.enable", "true"),
            ("checkpoint.last_save_model_only", "true"),
            (f"checkpoint.{hf_flag}", "true"),
            ("checkpoint.load_only", str(load_only).lower()),
        ],
    )
    assert result.ok is load_only, result.report.to_dict()
    if not load_only:
        assert any(error in issue.message for issue in result.report.errors)


@pytest.mark.parametrize(
    "hf_flag", ["save_in_hf", "last_save_in_hf", "initial_load_in_hf"]
)
def test_load_only_on_native_backend_skips_hf_save_but_validates_hf_load(hf_flag):
    result = run_pre_submit(
        "configs/experiments/gift_moe_debug.yaml",
        [
            ("parallelism.fsdp_backend", "gfsdp2"),
            ("checkpoint.enable", "true"),
            ("checkpoint.load_only", "true"),
            (f"checkpoint.{hf_flag}", "true"),
        ],
    )
    assert result.ok is (hf_flag != "initial_load_in_hf"), result.report.to_dict()
    if not result.ok:
        assert any(
            issue.path == f"checkpoint.{hf_flag}" for issue in result.report.errors
        )
