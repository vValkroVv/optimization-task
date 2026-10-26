"""Model restrictions are identical in pre-submit and runtime preparation."""

import pytest

from torchtitan.config import load_document
from torchtitan.config.build import build_trainer_config, materialize_model_config
from torchtitan.config.decode import decode
from torchtitan.config.pre_submit import run_pre_submit
from torchtitan.config.sections import ModelSection


CASES = [
    (
        "gift_3_5_debug.yaml",
        [("parallelism.tensor_parallel_degree", "2")],
        "Tensor Parallel does not support ['gdn'].",
    ),
    (
        "gift_3_5_debug.yaml",
        [("parallelism.context_parallel_degree", "2")],
        "Context Parallel does not support ['gdn'].",
    ),
    (
        "gift_moe_debug.yaml",
        [("parallelism.tensor_parallel_degree", "2")],
        "MTP is incompatible with sequence",
    ),
    (
        "gift_moe_debug.yaml",
        [("parallelism.context_parallel_degree", "2")],
        "MTP is incompatible with context",
    ),
    (
        "llama3_debug.yaml",
        [
            ("model.config.blocks.dense.attention.inner_attention.type", "varlen"),
            ("parallelism.context_parallel_degree", "2"),
        ],
        "Context Parallel does not support ['varlen'].",
    ),
    (
        "llama4_debug.yaml",
        [("parallelism.context_parallel_degree", "2")],
        "Context Parallel does not support ['llama4_chunked'].",
    ),
    (
        "deepseek_v3_debug_flex_attn.yaml",
        [("parallelism.context_parallel_degree", "2")],
        "Context Parallel does not support ['mla with flex'].",
    ),
    (
        "qwen3_debug.yaml",
        [
            ("model.config.enable_weight_tying", "true"),
            ("parallelism.pipeline_parallel_degree", "2"),
        ],
        "Weight tying",
    ),
    (
        "deepseek_v3_debug_hybridep.yaml",
        [("parallelism.expert_parallel_degree", "1")],
        "expert_parallel_degree > 1",
    ),
    (
        "gift_moe_3b_hybridep_dense_fp8_compile_non_blocking.yaml",
        [("fp8.sparse_use_grouped_gemm", "true")],
        "non_blocking_capacity_factor",
    ),
]


@pytest.mark.parametrize("experiment,overrides,message", CASES)
def test_pre_submit_rejects_runtime_restriction(experiment, overrides, message):
    result = run_pre_submit("configs/experiments/" + experiment, overrides)
    assert not result.ok
    assert message in str(result.report.errors)


@pytest.mark.parametrize("experiment,overrides,message", CASES)
def test_in_container_validation_rejects_same_rule_without_runtime_import(
    experiment, overrides, message
):
    from torchtitan.config.validation import ValidationContext, validate_typed_config

    document, _ = load_document("configs/experiments/" + experiment, overrides)
    config = build_trainer_config(document, with_model_spec=False)
    compact = decode(ModelSection, document.data["model"]).config
    model = materialize_model_config(compact)
    report = validate_typed_config(
        config, model, ValidationContext(stage="in_container")
    )
    assert message in str(report.errors)


@pytest.mark.parametrize(
    "experiment",
    [
        "gift_3_5_debug.yaml",
        "gift_4_debug.yaml",
        "llama3_debug.yaml",
        "llama4_debug.yaml",
        "deepseek_v3_debug.yaml",
        "qwen3_debug.yaml",
        "gift_moe_3b_hybridep_dense_fp8_compile_non_blocking.yaml",
    ],
)
def test_valid_presets_pass_pure_validation(experiment):
    assert run_pre_submit("configs/experiments/" + experiment).ok


def test_gdn_heads_are_checked_separately():
    result = run_pre_submit(
        "configs/experiments/gift_3_5_debug.yaml",
        [("parallelism.tensor_parallel_degree", "16")],
    )
    assert any("num_k_heads" in issue.path for issue in result.report.errors)


def test_gift_mtp_allows_tp_without_sequence_parallel():
    result = run_pre_submit(
        "configs/experiments/gift_moe_debug.yaml",
        [
            ("parallelism.tensor_parallel_degree", "2"),
            ("parallelism.enable_sequence_parallel", "false"),
        ],
    )
    assert result.ok


@pytest.mark.parametrize(
    "experiment", ["deepseek_v3_debug.yaml", "gift_moe_debug.yaml"]
)
def test_context_parallel_checks_kernels_after_the_first_layer(experiment):
    from torchtitan.config.schema.models.common.attention.softmax.kernels.flash_fa2 import (
        FlashAttention2Config,
    )

    document, _ = load_document("configs/experiments/" + experiment)
    trainer = build_trainer_config(document, with_model_spec=False)
    tree = decode(ModelSection, document.data["model"]).config.materialize()
    tree.mtp = None
    tree.layers[1].attention.inner_attention = FlashAttention2Config()
    trainer.parallelism.context_parallel_degree = 2
    from torchtitan.config.validation import ValidationContext, validate_typed_config

    result = validate_typed_config(trainer, tree, ValidationContext(stage="pre_submit"))
    assert any(
        "mla with flash_fa2" in issue.message
        and issue.path == "parallelism.context_parallel_degree"
        for issue in result.errors
    ), result.errors


@pytest.mark.parametrize(
    "path,value",
    [
        ("model.config.param_init", "{weight: {type: zeros}}"),
        ("model.config.init_role", "linear"),
        ("model.config.rope.param_init", "{weight: {type: zeros}}"),
        ("model.config.rope.hidden_z_loss", "{coef: 1e-5}"),
        ("model.config.blocks.dense.attention.sharding_config", "{}"),
    ],
)
def test_ignored_module_fields_are_rejected(path, value):
    result = run_pre_submit("configs/experiments/llama3_debug.yaml", [(path, value)])
    assert not result.ok
    assert path in str(result.report.errors)


def test_moe_without_parameters_cannot_claim_init_role():
    result = run_pre_submit(
        "configs/experiments/gift_moe_debug.yaml",
        [("model.config.blocks.mla_moe.moe.init_role", "linear")],
    )
    assert not result.ok
    assert "model.config.blocks.mla_moe.moe.init_role" in str(result.report.errors)


def test_parameter_owner_init_override_is_preserved():
    result = run_pre_submit(
        "configs/experiments/llama3_debug.yaml",
        [
            (
                "model.config.blocks.dense.feed_forward.w1.param_init",
                "{weight: {type: zeros}}",
            )
        ],
    )
    assert result.ok


def test_parameter_owning_conv2d_accepts_module_initializers():
    from dataclasses import make_dataclass
    from torchtitan.config.schema.models.common.cross_section import (
        model_module_field_issues,
    )
    from torchtitan.config.schema.models.common.nn_modules import Conv2dConfig
    from torchtitan.config.schema.protocols.module import Zeros
    from torchtitan.config.schema.protocols.model import BaseModelConfig

    root_type = make_dataclass(
        "ConvModelConfig",
        [("conv", Conv2dConfig)],
        bases=(BaseModelConfig,),
        kw_only=True,
    )
    model = root_type(
        conv=Conv2dConfig(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            param_init={"weight": Zeros(), "bias": Zeros()},
        )
    )
    assert model_module_field_issues(model) == []
