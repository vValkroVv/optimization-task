"""Regression checks for config review input and inspection issues."""

import pytest

from torchtitan.config.overrides import build_override_layers
from torchtitan.config.pre_submit import main
from torchtitan.config.schema.components.optimizer import (
    OptimizersContainerConfig,
    ParamGroupConfig,
)


@pytest.mark.parametrize("field", ["lr", "eps", "weight_decay", "beta1", "beta2"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_optimizer_rejects_invalid_numbers(field, value):
    with pytest.raises(ValueError, match=field):
        OptimizersContainerConfig(**{field: value})


@pytest.mark.parametrize(
    "field", ["lr_multiplier", "weight_decay_multiplier", "beta1", "beta2"]
)
def test_group_rejects_nonfinite_numbers(field):
    with pytest.raises(ValueError, match=field):
        ParamGroupConfig(pattern="weight", **{field: float("nan")})


def test_group_rejects_invalid_regex():
    with pytest.raises(ValueError, match="pattern"):
        ParamGroupConfig(pattern="[")


@pytest.mark.parametrize("value", ["runs/a #1", "runs/a\t#1"])
def test_cli_plain_string_keeps_hash(value):
    layers, _ = build_override_layers([("dump_folder", value)])
    assert layers == [{"dump_folder": value}]


def test_print_resolved_includes_component_defaults(capsys):
    assert main(["configs/experiments/gift_3_5_debug.yaml", "--print-resolved"]) == 0
    output = capsys.readouterr().out
    assert "conv_size: 4" in output
    assert "weight_decay: 0.1" in output


def test_print_merged_preserves_compact_document(capsys):
    assert main(["configs/experiments/gift_3_5_debug.yaml", "--print-merged"]) == 0
    output = capsys.readouterr().out
    assert "conv_size: 4" in output
    assert "allow_neg_eigval:" not in output


def test_partial_norm_init_keeps_role_entries_and_policy_overrides():
    from dataclasses import make_dataclass, replace
    from torchtitan.config.schema.models.common.init_policy import apply_init
    from torchtitan.config.schema.protocols.module import Normal, Zeros
    from torchtitan.config.schema.models.common.norms.gated_norm import (
        GatedRMSNormConfig,
    )
    from tests.unit_tests.config_helpers import init_policy

    GIFT_HYBRID_INIT = init_policy("gift_hybrid")

    # gift_4's plain post_attn_norm, starting at zero scale.
    norm = GatedRMSNormConfig(eps=1e-6, param_init={"weight": Zeros()})
    assert GIFT_HYBRID_INIT.gated_norm is not None
    policy = replace(
        GIFT_HYBRID_INIT,
        gated_norm={
            **GIFT_HYBRID_INIT.gated_norm,
            "gate_up_weight": Normal(std=0.123),
        },
    )
    apply_init(
        make_dataclass("NormTree", [("norm", type(norm))])(norm),
        policy,
        layer_id=0,
        dim=896,
    )
    assert norm.param_init is not None
    assert isinstance(norm.param_init["weight"], Zeros)
    gate_init = norm.param_init["gate_up_weight"]
    assert isinstance(gate_init, Normal)
    assert gate_init.std == 0.123
    assert "gate_down_weight" in norm.param_init


def test_type_switch_warns_with_inherited_key_source(tmp_path):
    from torchtitan.config.loader import load_raw_document

    (tmp_path / "base.yaml").write_text(
        "dataloader: {type: hf_text, dataset: c4, num_workers: 4}\n"
    )
    path = tmp_path / "run.yaml"
    path.write_text("imports: [base.yaml]\ndataloader: {type: hf_chat}\n")
    from torchtitan.config.validation import merge_type_change_issues

    loaded = load_raw_document(str(path))
    issues = merge_type_change_issues(loaded)
    assert len(issues) == 1
    assert "dataset" in issues[0].message and "base.yaml" in issues[0].message
    assert "num_workers" in issues[0].message
    assert loaded.data["dataloader"]["dataset"] == "c4"


def test_section_reset_discards_type_inheritance_without_warning(tmp_path):
    import warnings
    from torchtitan.config.loader import load_raw_document

    path = tmp_path / "run.yaml"
    path.write_text("dataloader: {type: hf_text, dataset: c4}\n")
    with warnings.catch_warnings(record=True) as seen:
        loaded = load_raw_document(
            str(path),
            overrides=[{"dataloader": None}, {"dataloader": {"type": "hf_chat"}}],
        )
    assert not seen
    assert loaded.data["dataloader"] == {"type": "hf_chat"}


@pytest.mark.parametrize(
    "experiment,path,value",
    [
        ("llama3_debug.yaml", "model.config.dim", "0"),
        ("deepseek_v3_debug.yaml", "model.config.dim", "0"),
        (
            "llama3_debug.yaml",
            "model.config.blocks.dense.feed_forward.multiple_of",
            "0",
        ),
        ("gpt_oss_debug.yaml", "model.config.rope.scaling.beta_slow", "0"),
    ],
)
def test_invalid_numeric_model_config_reports_error(experiment, path, value):
    from torchtitan.config.pre_submit import run_pre_submit

    result = run_pre_submit("configs/experiments/" + experiment, [(path, value)])
    assert not result.ok
    assert "positive" in str(result.report.errors)


@pytest.mark.parametrize(
    "experiment,overrides",
    [
        ("llama3_debug.yaml", [("trainer.type", "ft")]),
        ("llama3_debug.yaml", [("trainer.type", "graph_trainer")]),
        ("graph_llama3_debug.yaml", [("trainer.type", "default")]),
        ("ft_llama3_debug.yaml", [("trainer.type", "default")]),
        ("flux_debug.yaml", [("trainer.type", "default")]),
    ],
)
def test_model_trainer_runtime_contract_rejected_before_runtime(experiment, overrides):
    from torchtitan.config.pre_submit import run_pre_submit

    result = run_pre_submit("configs/experiments/" + experiment, overrides)
    assert not result.ok
    assert "trainer.type" in str(result.report.errors)


@pytest.mark.parametrize(
    "experiment",
    [
        "llama3_debug.yaml",
        "ft_llama3_debug.yaml",
        "graph_llama3_debug.yaml",
        "graph_qwen3_moe_debug.yaml",
        "flux_debug.yaml",
        "rl_grpo_qwen3_0_6b.yaml",
    ],
)
def test_valid_model_trainer_contract_remains_torch_free(experiment):
    from torchtitan.config.pre_submit import run_pre_submit

    assert run_pre_submit("configs/experiments/" + experiment).ok
