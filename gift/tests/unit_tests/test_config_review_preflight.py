"""Pre-submit rejects unsupported runtime/export combinations without torch."""

import pytest
from torchtitan.config import run_pre_submit
from torchtitan.config.pre_submit import main


@pytest.mark.parametrize(
    "path,value",
    [
        ("checkpoint.interval", "0"),
        ("checkpoint.keep_latest_k", "1"),
        ("lr_scheduler.warmup_steps", "-3"),
        ("lr_scheduler.total_steps", "0"),
    ],
)
def test_invalid_schedule_or_checkpoint_range_is_rejected(path, value):
    result = run_pre_submit("configs/experiments/llama3_debug.yaml", [(path, value)])
    assert not result.ok


def test_planned_world_size_is_checked_without_reading_local_world_size(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "999")
    overrides = [("parallelism.tensor_parallel_degree", "2")]
    recipe = "configs/experiments/llama3_debug.yaml"
    assert run_pre_submit(recipe, overrides).ok
    assert not run_pre_submit(recipe, overrides, world_size=1).ok
    assert run_pre_submit(recipe, overrides, world_size=2).ok


def test_world_size_cli_flag(capsys):
    assert (
        main(
            [
                "configs/experiments/llama3_debug.yaml",
                "--world-size",
                "1",
                "--parallelism.tensor_parallel_degree",
                "2",
            ]
        )
        == 1
    )
    assert "parallelism" in capsys.readouterr().err
    assert main(["configs/experiments/llama3_debug.yaml", "--world-size=0"]) == 2


def test_qwen_vision_context_parallel_fails_before_runtime():
    result = run_pre_submit(
        "configs/experiments/qwen3_vl_debug.yaml",
        [("parallelism.context_parallel_degree", "2")],
    )
    assert not result.ok
    assert any(
        "Qwen3VLVisionEncoderConfig" in issue.message for issue in result.report.issues
    )


def test_specialized_head_fp8_fails_before_runtime():
    result = run_pre_submit(
        "configs/experiments/gift_3_5_debug.yaml",
        [
            ("model.config.lm_head", "{type: unembedding, logit_softcap: 1.0}"),
            ("fp8.dense_include_linear_layers", "lm_head"),
        ],
    )
    assert not result.ok
    assert any("specialized linear" in issue.message for issue in result.report.issues)


_HF_SAVE = [
    ("checkpoint.enable", "true"),
    ("checkpoint.last_save_in_hf", "true"),
    ("checkpoint.last_save_model_only", "true"),
]
_DENSE2 = (
    "{attention: {type: gqa, n_heads: 16%s}, feed_forward: {type: llama, "
    "multiple_of: 256%s}, attention_norm: {type: rmsnorm}, ffn_norm: {type: rmsnorm}}"
)
_SPLIT_LAYERS = (
    "model.config.layers",
    "[{block: dense, repeat: 3}, {block: dense2, repeat: 3}]",
)
_MOE2 = (
    "model.config.blocks.moe2",
    "{attention: {type: gqa, n_heads: 16, n_kv_heads: 8, head_dim: 128}, "
    "moe: {experts: {hidden_dim: 768, num_experts: 64, token_dispatcher: {type: local}}, "
    "router: {top_k: 8, score_func: sigmoid, route_norm: true}}, "
    "attention_norm: {type: rmsnorm}, ffn_norm: {type: rmsnorm}}",
)
_PERIODIC_HF = [("checkpoint.enable", "true"), ("checkpoint.save_in_hf", "true")]
_RESIDUAL_SCALE = (
    "model.config.blocks.dense.attn_residual_scale",
    "{warmup_steps: 100, start_value: 0}",
)


@pytest.mark.parametrize(
    "recipe,overrides,message",
    [
        (
            "qwen3_debug",
            [
                (
                    "model.config.blocks.dense.attention_norm",
                    "!replace {type: layernorm, elementwise_affine: false}",
                )
            ],
            "affine RMSNorm",
        ),
        (
            "qwen3_debug",
            [
                (
                    "model.config.blocks.dense.ffn_norm",
                    "!replace {type: rmsnorm, elementwise_affine: false}",
                )
            ],
            "affine RMSNorm",
        ),
        (
            "llama3_debug",
            [
                ("model.config.blocks.dense2", _DENSE2 % ("", ", hidden_dim: 1024")),
                _SPLIT_LAYERS,
            ],
            "dense FFN layout",
        ),
        (
            "llama3_debug",
            [
                ("model.config.blocks.dense2", _DENSE2 % (", n_kv_heads: 4", "")),
                _SPLIT_LAYERS,
            ],
            "n_kv_heads=4",
        ),
        (
            "qwen3_moe_debug",
            [
                _MOE2,
                (
                    "model.config.layers",
                    "[{block: moe, repeat: 4}, {block: moe2, repeat: 4}]",
                ),
            ],
            "score_func=sigmoid vs softmax",
        ),
    ],
)
def test_hf_export_layout_fails_pre_submit_only_when_hf_save_is_on(
    recipe, overrides, message
):
    path = f"configs/experiments/{recipe}.yaml"
    assert run_pre_submit(path, overrides).ok
    result = run_pre_submit(path, overrides + _HF_SAVE)
    assert not result.ok
    assert any(message in issue.message for issue in result.report.errors)


@pytest.mark.parametrize(
    "hf_save,steps,ok",
    [
        (_HF_SAVE, 101, True),  # the last save is past the warmup
        (_HF_SAVE, 100, False),  # its weights come from schedule step 99
        # Periodic HF saves: the first one sets the bound.
        ([*_PERIODIC_HF, ("checkpoint.interval", "1000")], 2000, True),
        ([*_PERIODIC_HF, ("checkpoint.interval", "100")], 2000, False),
        (
            [
                *_PERIODIC_HF,
                ("checkpoint.interval", "1000"),
                ("checkpoint.enable_first_step_checkpoint", "true"),
            ],
            2000,
            False,
        ),
    ],
)
def test_residual_scale_hf_save_must_be_past_the_warmup(hf_save, steps, ok):
    result = run_pre_submit(
        "configs/experiments/llama3_debug.yaml",
        [_RESIDUAL_SCALE, ("training.steps", str(steps)), *hf_save],
    )
    assert result.ok == ok, result.report.errors


@pytest.mark.parametrize("decay_type", ["linear", "sqrt", "cosine"])
@pytest.mark.parametrize("decay_ratio", [0.0, 1.0])
def test_scheduler_holds_its_final_factor_after_the_horizon(
    monkeypatch, decay_type, decay_ratio
):
    from torchtitan.config.schema.components.lr_scheduler import (
        LRSchedulersContainerConfig,
    )

    monkeypatch.setattr(
        LRSchedulersContainerConfig, "_owner", lambda optimizers, fn: fn
    )
    config = LRSchedulersContainerConfig(
        warmup_steps=0,
        total_steps=5,
        decay_ratio=decay_ratio,
        decay_type=decay_type,
        min_lr_factor=0.1,
    )
    schedule = config.build(optimizers=None, training_steps=20)
    expected = 0.1 if decay_ratio else 1.0
    assert all(schedule(step) == expected for step in range(5, 21))
    assert all(0.1 <= schedule(step) <= 1.0 for step in range(21))


def test_pre_submit_reports_round_trip_and_validation_errors(monkeypatch):
    from torchtitan.config import save

    typed = save.get_typed_document

    def lossy(config, model_section):
        # A serializer that loses a field: the resolved YAML cannot reproduce it.
        document = typed(config, model_section)
        document["training"]["steps"] += 1
        return document

    monkeypatch.setattr(save, "get_typed_document", lossy)
    result = run_pre_submit(
        "configs/experiments/llama3_debug.yaml",
        [("parallelism.tensor_parallel_degree", "2")],
        world_size=1,
    )
    messages = [issue.message for issue in result.report.issues]
    assert any("does not decode back" in m for m in messages), messages
    assert any("parallelism" in i.path for i in result.report.issues), messages


def test_llama3_mla_fails_at_the_declared_runtime_contract():
    result = run_pre_submit(
        "configs/experiments/llama3_debug.yaml",
        [
            (
                "model.config.blocks.dense.attention",
                "!replace {type: mla, n_heads: 4, qk_rope_head_dim: 16}",
            )
        ],
    )
    assert not result.ok
    assert any("runtime supports" in issue.message for issue in result.report.issues)


@pytest.mark.parametrize(
    "recipe,path",
    [
        ("llama4_debug", "model.config.blocks.dense.fixed_attn_block_size"),
        (
            "gpt_oss_debug",
            "model.config.blocks.sliding_gpt_oss_sink_moe.attention.sliding_window_size",
        ),
    ],
)
@pytest.mark.parametrize("value", ["0", "-1"])
def test_mask_chunk_and_window_sizes_are_positive(recipe, path, value):
    result = run_pre_submit(f"configs/experiments/{recipe}.yaml", [(path, value)])
    assert not result.ok
    assert any("positive integer" in issue.message for issue in result.report.issues)


@pytest.mark.parametrize(
    "field", ["num_k_heads", "num_v_heads", "head_k_dim", "head_v_dim", "conv_size"]
)
@pytest.mark.parametrize("value", ["0", "-1"])
def test_linear_attention_geometry_fails_as_a_config_error(field, value):
    result = run_pre_submit(
        "configs/experiments/gift_3_5_debug.yaml",
        [(f"model.config.blocks.gdn_dense.attention.{field}", value)],
    )
    assert not result.ok
    assert any("positive integer" in issue.message for issue in result.report.issues)
