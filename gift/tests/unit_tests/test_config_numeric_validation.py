"""Public YAML validation rejects numerics that poison runtime math."""

from pathlib import Path

import pytest

from torchtitan.config import run_pre_submit
from torchtitan.config.errors import ConfigFieldError
from torchtitan.config.schema.models.common.moe.experts.gmm import (
    GroupedExpertsConfig,
)
from torchtitan.config.schema.models.common.moe.moe import MoEConfig
from torchtitan.config.schema.models.common.moe.router import (
    TokenChoiceTopKRouterConfig,
)


EXPERIMENT = (
    Path(__file__).resolve().parents[2] / "configs/experiments/llama3_debug.yaml"
)


@pytest.mark.parametrize("value", [".nan", ".inf", "-.inf", "0", "1"])
def test_rope_theta_rejected_before_submission(value):
    path = "model.config.rope.theta"
    result = run_pre_submit(str(EXPERIMENT), [(path, value)])
    assert not result.ok
    assert any(issue.path == path for issue in result.report.issues)


@pytest.mark.parametrize(
    "field", ["scaling_factor", "low_freq_factor", "high_freq_factor"]
)
@pytest.mark.parametrize("value", [".nan", ".inf", "-.inf", "0", "-1"])
def test_llama_scaling_rejected_before_submission(field, value):
    path = f"model.config.rope.scaling.{field}"
    result = run_pre_submit(str(EXPERIMENT), [(path, value)])
    assert not result.ok
    assert any(issue.path == path for issue in result.report.issues)


@pytest.mark.parametrize("field", ["rope_factor", "beta_slow", "beta_fast"])
@pytest.mark.parametrize("value", [".nan", ".inf", "-.inf", "0"])
def test_yarn_scaling_rejected_before_submission(field, value):
    scaling = "{type: yarn, rope_factor: 8, original_seq_len: 8192}"
    path = f"model.config.rope.scaling.{field}"
    result = run_pre_submit(
        str(EXPERIMENT),
        [("model.config.rope.scaling", scaling), (path, value)],
    )
    assert not result.ok
    assert any(issue.path == path for issue in result.report.issues)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rank", "0"),
        ("rank", "-1"),
        ("alpha", ".nan"),
        ("alpha", ".inf"),
        ("alpha", "-.inf"),
    ],
)
def test_lora_scalars_rejected_before_submission(field, value):
    path = f"model.config.converters[0].{field}"
    result = run_pre_submit(
        str(EXPERIMENT),
        [("model.config.converters", f"[{{type: lora, {field}: {value}}}]")],
    )
    assert not result.ok
    assert any(issue.path == path for issue in result.report.issues)


@pytest.mark.parametrize("alpha", ["0", "-2", "16"])
def test_finite_lora_scales_remain_supported(alpha):
    result = run_pre_submit(
        str(EXPERIMENT),
        [("model.config.converters", f"[{{type: lora, alpha: {alpha}}}]")],
    )
    assert result.ok, result.report.issues


def test_llama_scaling_equal_bounds_remain_supported():
    result = run_pre_submit(
        str(EXPERIMENT),
        [("model.config.rope.scaling.high_freq_factor", "1")],
    )
    assert result.ok, result.report.issues


@pytest.mark.parametrize("value", [".nan", ".inf"])
def test_nonfinite_theta_cannot_round_trip_into_run_bundle(value):
    from torchtitan.config import load_document
    from torchtitan.config.build import build_trainer_config
    from torchtitan.config.errors import ConfigSchemaError
    from torchtitan.config.save import check_round_trip

    document, _ = load_document(str(EXPERIMENT))
    trainer = build_trainer_config(document, with_model_spec=False)
    document.data["model"]["config"]["rope"]["theta"] = float(
        "nan" if value == ".nan" else "inf"
    )
    with pytest.raises(ConfigSchemaError, match=r"model\.config\.rope\.theta"):
        check_round_trip(trainer, document.data, document.data, "resolved_config.yaml")


@pytest.mark.parametrize(
    "field", ["load_balance_coeff", "router_aux_loss_coef", "router_z_loss_coef"]
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1e-3])
def test_moe_coefficients_rejected_on_partial_configs(field, value):
    with pytest.raises(ConfigFieldError, match="finite and >= 0") as raised:
        MoEConfig(
            experts=GroupedExpertsConfig(hidden_dim=8, num_experts=4),
            router=TokenChoiceTopKRouterConfig(),
            **{field: value},
        )
    assert raised.value.field == field
