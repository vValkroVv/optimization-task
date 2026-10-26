"""Runtime loading consumes one prepared tree and applies each converter once."""

import pytest

from torchtitan.config import build, cli, load_config
from torchtitan.config.errors import ConfigSchemaError
from torchtitan.config.schema.models.llama3.model import Llama3ModelConfig
from torchtitan.config.sections import ModelSection
from torchtitan.config.validation import ValidationReport


@pytest.mark.parametrize("pattern,valid", [(".*w1$", True), (".*wo$", False)])
def test_runtime_loader_materializes_once_before_lora_fp8_check(
    monkeypatch, pattern, valid
):
    from torchtitan.components.lora import LoRAConverter

    calls = {"decode": 0, "materialize": 0, "convert": 0, "tree": None}
    decode, materialize = build.decode, Llama3ModelConfig.materialize
    convert = LoRAConverter.convert

    def counted_decode(cls, *args, **kwargs):
        if cls is ModelSection:
            calls["decode"] += 1
        return decode(cls, *args, **kwargs)

    def counted_materialize(config):
        calls["materialize"] += 1
        calls["tree"] = materialize(config)
        return calls["tree"]

    def counted_convert(converter, tree):
        calls["convert"] += 1
        assert tree is calls["tree"]
        return convert(converter, tree)

    monkeypatch.setattr(build, "decode", counted_decode)
    monkeypatch.setattr(Llama3ModelConfig, "materialize", counted_materialize)
    monkeypatch.setattr(LoRAConverter, "convert", counted_convert)
    # Filesystem/assets preflight is separate from model preparation.
    monkeypatch.setattr(
        cli, "preflight", lambda *args: ValidationReport(stage="in_container")
    )
    overrides = [
        {
            "model": {
                "config": {"converters": [{"type": "lora", "target_modules": ["wo"]}]}
            },
            "fp8": {"dense_include_linear_layers": pattern},
        }
    ]
    if valid:
        loaded = load_config(
            "configs/experiments/llama3_debug.yaml", overrides=overrides
        )
        assert loaded.trainer_config.model_spec.model is calls["tree"]
        assert calls["convert"] == 1
    else:
        with pytest.raises(
            ConfigSchemaError, match=r"converters\[0\].*LoRA and dense FP8"
        ):
            load_config("configs/experiments/llama3_debug.yaml", overrides=overrides)
        assert calls["convert"] == 0
    assert calls["decode"] == calls["materialize"] == 1
