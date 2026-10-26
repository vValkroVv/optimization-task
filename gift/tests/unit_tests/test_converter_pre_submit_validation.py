"""Converter selectors are checked locally without importing converter runtimes."""

import builtins

import pytest

from torchtitan.config import load_document, run_pre_submit
from torchtitan.config.build import materialize_model_config
from torchtitan.config.decode import decode
from torchtitan.config.schema.components.freeze import FreezeConverterConfig
from torchtitan.config.schema.components.lora import FrozenConfig
from torchtitan.config.sections import ModelSection


def test_missing_freeze_selector_is_a_pre_submit_config_error():
    result = run_pre_submit(
        "configs/experiments/llama3_debug.yaml",
        [("model.config.converters", "[{type: freeze, selectors: [layers.999]}]")],
    )
    issue = next(
        item for item in result.report.errors if "freeze selector" in item.message
    )
    assert issue.path == "model.config.converters[0]"
    assert "layers.999" in issue.message


def test_pure_converter_validation_does_not_freeze_or_import_runtime(monkeypatch):
    document, _ = load_document("configs/experiments/llama3_debug.yaml")
    compact = decode(ModelSection, document.data["model"]).config
    compact.converters = [FreezeConverterConfig(selectors=["layers.0"])]
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if (
            name == "torch"
            or name.startswith("torch.")
            or name == "torchtitan.components.freeze"
        ):
            raise AssertionError(f"pure converter validation imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    tree = materialize_model_config(compact)
    assert not isinstance(tree.layers[0], FrozenConfig)
    compact.converters[0].validate_pipeline(tree, [])


def test_repeated_lora_reports_the_second_converter_before_runtime():
    result = run_pre_submit(
        "configs/experiments/llama3_debug.yaml",
        [
            (
                "model.config.converters",
                "[{type: lora}, {type: freeze, selectors: [layers.0]}, {type: lora}]",
            )
        ],
    )
    assert not result.ok
    assert len(result.report.errors) == 1
    issue = result.report.errors[0]
    assert issue.path == "model.config.converters[2]"
    assert "repeated LoRA" in issue.message


def test_overlapping_lora_dense_fp8_is_a_precise_pre_submit_error():
    result = run_pre_submit(
        "configs/experiments/llama3_debug.yaml",
        [
            ("model.config.converters", "[{type: lora, target_modules: [wo]}]"),
            ("fp8.dense_include_linear_layers", ".*wo$"),
        ],
    )
    assert not result.ok
    assert len(result.report.errors) == 1
    issue = result.report.errors[0]
    assert issue.path == "model.config.converters[0]"
    assert "layers.0.attention.wo" in issue.message
    assert "LoRA and dense FP8" in issue.message


def test_disjoint_lora_dense_fp8_is_supported():
    result = run_pre_submit(
        "configs/experiments/llama3_debug.yaml",
        [
            ("model.config.converters", "[{type: lora, target_modules: [wo]}]"),
            ("fp8.dense_include_linear_layers", ".*w1$"),
        ],
    )
    assert result.ok, result.report.to_dict()


def test_empty_lora_targets_do_not_conflict_with_dense_fp8():
    result = run_pre_submit(
        "configs/experiments/llama3_debug.yaml",
        [
            ("model.config.converters", "[{type: lora, target_modules: []}]"),
            ("fp8.dense_include_linear_layers", ".*"),
        ],
    )
    assert result.ok, result.report.to_dict()


def test_sparse_only_fp8_does_not_conflict_with_lora():
    result = run_pre_submit(
        "configs/experiments/gift_moe_debug_deepep_fp8.yaml",
        [
            ("model.config.converters", "[{type: lora}]"),
            ("fp8.dense_include_linear_layers", '""'),
            ("fp8.dense_use_shared_quantizer", "false"),
        ],
    )
    assert result.ok, result.report.to_dict()


@pytest.mark.parametrize("flag", ["save_in_hf", "last_save_in_hf"])
def test_lora_rejects_active_hf_checkpoint_export_before_training(flag):
    result = run_pre_submit(
        "configs/experiments/llama3_debug.yaml",
        [
            ("model.config.converters", "[{type: lora}]"),
            ("checkpoint.enable", "true"),
            ("checkpoint.last_save_model_only", "true"),
            (f"checkpoint.{flag}", "true"),
        ],
    )
    assert not result.ok
    issue = result.report.errors[0]
    assert issue.path == "model.config.converters[0]"
    assert "HF checkpoint export cannot save unmerged LoRA adapters" in issue.message


def test_disabled_checkpoint_does_not_reject_lora_for_stale_hf_flags():
    result = run_pre_submit(
        "configs/experiments/llama3_debug.yaml",
        [
            ("model.config.converters", "[{type: lora}]"),
            ("checkpoint.enable", "false"),
            ("checkpoint.save_in_hf", "true"),
            ("checkpoint.last_save_in_hf", "true"),
        ],
    )
    assert result.ok, result.report.to_dict()


def test_empty_lora_targets_allow_hf_checkpoint_export():
    result = run_pre_submit(
        "configs/experiments/llama3_debug.yaml",
        [
            ("model.config.converters", "[{type: lora, target_modules: []}]"),
            ("checkpoint.enable", "true"),
            ("checkpoint.save_in_hf", "true"),
        ],
    )
    assert result.ok, result.report.to_dict()
