"""Reject invalid final saves and callback dependencies before training."""

import pytest
from typing import Any

from torchtitan.config import run_pre_submit

RECIPE = "configs/experiments/llama3_debug.yaml"


def _check(*overrides):
    return run_pre_submit(RECIPE, list(overrides))


def _unused_runtime_arguments() -> dict[str, Any]:
    # Invalid options must fail before inspecting or allocating these dependencies.
    return dict(
        dataloader=None,
        model_parts=[],
        sharding_backend=None,
        optimizers=None,
        lr_schedulers=None,
        states={},
        sd_adapter=None,
    )


@pytest.mark.parametrize(
    "overrides,path,message",
    [
        (
            [
                ("checkpoint.enable", "true"),
                ("checkpoint.last_save_ema_model_also", "true"),
                ("callbacks", "!replace {}"),
            ],
            "checkpoint.last_save_ema_model_also",
            "EMA callback",
        ),
        (
            [
                ("checkpoint.enable", "true"),
                ("checkpoint.last_save_ema_model_also", "true"),
                ("callbacks", "!replace {one: {type: ema}, two: {type: ema}}"),
            ],
            "checkpoint.last_save_ema_model_also",
            "found 2",
        ),
        (
            [
                ("checkpoint.enable", "true"),
                ("checkpoint.last_save_in_hf", "true"),
                ("checkpoint.last_save_model_only", "false"),
            ],
            "checkpoint.last_save_in_hf",
            "last_save_model_only",
        ),
        (
            [("callbacks", "!replace {rescale: {type: dynamic_block_lr_rescaler}}")],
            "callbacks.rescale",
            "before it",
        ),
        (
            [
                (
                    "callbacks",
                    "!replace {rescale: {type: dynamic_block_lr_rescaler}, stats: {type: monitoring}}",
                )
            ],
            "callbacks.rescale",
            "before it",
        ),
        (
            [
                (
                    "callbacks",
                    "!replace {monitoring: {type: ema}, rescale: {type: dynamic_block_lr_rescaler}}",
                )
            ],
            "callbacks.rescale",
            "found 0",
        ),
        (
            [
                (
                    "callbacks",
                    "!replace {one: {type: monitoring}, two: {type: monitoring}, rescale: {type: dynamic_block_lr_rescaler}}",
                )
            ],
            "callbacks.rescale",
            "found 2",
        ),
    ],
)
def test_pre_submit_rejects_late_failures(overrides, path, message):
    result = _check(*overrides)
    assert not result.ok
    assert any(
        issue.path == path and message in issue.message
        for issue in result.report.errors
    ), result.report.issues


@pytest.mark.parametrize(
    "overrides",
    [
        [
            ("checkpoint.enable", "false"),
            ("checkpoint.last_save_ema_model_also", "true"),
            ("checkpoint.last_save_in_hf", "true"),
            ("checkpoint.last_save_model_only", "false"),
            ("callbacks", "!replace {}"),
        ],
        [
            ("checkpoint.enable", "true"),
            ("checkpoint.last_save_ema_model_also", "true"),
            ("callbacks", "!replace {average: {type: ema}}"),
        ],
        [
            ("checkpoint.enable", "true"),
            ("checkpoint.save_in_hf", "true"),
            ("checkpoint.last_save_in_hf", "false"),
            ("checkpoint.last_save_model_only", "false"),
        ],
        [
            (
                "callbacks",
                "!replace {renamed_stats: {type: monitoring}, rescale: {type: dynamic_block_lr_rescaler}}",
            )
        ],
        [
            (
                "callbacks",
                "!replace {one: {type: monitoring}, rescale: {type: dynamic_block_lr_rescaler}, two: {type: monitoring}}",
            )
        ],
    ],
)
def test_pre_submit_accepts_valid_and_inactive_options(overrides):
    result = _check(*overrides)
    assert result.ok, result.report.issues


def test_load_only_does_not_require_final_save_options():
    result = _check(
        ("checkpoint.enable", "true"),
        ("checkpoint.load_only", "true"),
        ("checkpoint.last_save_ema_model_also", "true"),
        ("checkpoint.last_save_in_hf", "true"),
        ("checkpoint.last_save_model_only", "false"),
        ("callbacks", "!replace {}"),
    )
    assert result.ok, result.report.issues


def test_runtime_rejects_mutated_hf_options_before_backend_use():
    from torchtitan.components.checkpoint import CheckpointManager
    from torchtitan.config.schema.components.checkpoint import CheckpointManagerConfig

    config = CheckpointManagerConfig(enable=True)
    config.last_save_in_hf = True
    config.last_save_model_only = False
    with pytest.raises(ValueError, match="last_save_model_only"):
        CheckpointManager(config, **_unused_runtime_arguments())


def test_runtime_rejects_missing_ema_provider_before_backend_use():
    from torchtitan.components.checkpoint import CheckpointManager
    from torchtitan.config.schema.components.checkpoint import CheckpointManagerConfig

    config = CheckpointManagerConfig(enable=True, last_save_ema_model_also=True)
    with pytest.raises(ValueError, match="found 0"):
        CheckpointManager(config, **_unused_runtime_arguments())


def test_runtime_ema_provider_selection_matches_final_save():
    from torchtitan.components.checkpoint import _ema_weights_source

    class Weights:
        holds_average = True
        use_ema = True

        def model_state_dict(self):
            return {}

    weights = Weights()
    assert _ema_weights_source({"renamed_ema": weights}) is weights
    with pytest.raises(ValueError, match="found 2"):
        _ema_weights_source({"one": weights, "two": Weights()})
