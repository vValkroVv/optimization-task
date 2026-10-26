"""Checkpoint export must not silently drop adapters from a training config."""

from pathlib import Path
from unittest.mock import patch

import pytest

from torchtitan.components.checkpoint_conversion import load_export_model_spec
from torchtitan.config.errors import ConfigSchemaError


@pytest.mark.parametrize("target_modules", ["null", "[]", "[wq]"])
def test_lora_export_rejected_before_runtime_build(tmp_path, target_modules):
    preset = Path("configs/presets/models/llama3/debugmodel.yaml").resolve()
    config_file = tmp_path / "export.yaml"
    config_file.write_text(
        f"imports: [{preset}]\n"
        "model:\n  config:\n    converters:\n"
        f"      - type: lora\n        target_modules: {target_modules}\n"
    )
    with patch("torchtitan.components.checkpoint_conversion.build_model_spec") as build:
        with pytest.raises(ValueError, match="discard the trained adapters"):
            load_export_model_spec(str(config_file))
        build.assert_not_called()


def test_plain_export_builds_without_converters():
    sentinel = object()
    with patch(
        "torchtitan.components.checkpoint_conversion.build_model_spec",
        return_value=sentinel,
    ) as build:
        result = load_export_model_spec("configs/presets/models/llama3/debugmodel.yaml")
    assert result is sentinel
    assert build.call_args.kwargs == {"apply_converters": False}


def test_missing_model_rejected_before_runtime_build(tmp_path):
    config_file = tmp_path / "export.yaml"
    config_file.write_text("training: {steps: 1}\n")
    with patch("torchtitan.components.checkpoint_conversion.build_model_spec") as build:
        with pytest.raises(ConfigSchemaError, match="no `model:` section"):
            load_export_model_spec(str(config_file))
        build.assert_not_called()
