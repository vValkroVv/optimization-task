"""Configuration policy for exporting training checkpoints to Hugging Face."""

from typing import TYPE_CHECKING

from torchtitan.config.build import build_model_spec
from torchtitan.config.decode import checking_missing_cleared_paths, decode
from torchtitan.config.errors import ConfigSchemaError
from torchtitan.config.loader import load_raw_document
from torchtitan.config.schema.components.lora import LoRAConverterConfig
from torchtitan.config.sections import ModelSection


if TYPE_CHECKING:
    from torch import dtype, Tensor

    from torchtitan.protocols.model_spec import ModelSpec


def load_export_model_spec(config_path: str) -> "ModelSpec":
    """Reject adapter checkpoints until export can merge their learned weights."""
    document = load_raw_document(config_path)
    if "model" not in document.data:
        raise ConfigSchemaError(f"{config_path}: no `model:` section")
    with checking_missing_cleared_paths(document.missing_cleared_paths):
        section = decode(ModelSection, document.data["model"], path="model")
        if any(
            isinstance(converter, LoRAConverterConfig)
            for converter in getattr(section.config, "converters", ())
        ):
            raise ValueError(
                "LoRA checkpoint export requires merging adapter weights; "
                "exporting the base model would discard the trained adapters"
            )
        return build_model_spec(section, apply_converters=False)


def cast_export_state_dict(
    state_dict: dict[str, "Tensor"], target_dtype: "dtype"
) -> dict[str, "Tensor"]:
    """Cast floating-point weights while preserving integer and other buffers."""
    return {
        key: value.to(target_dtype) if value.is_floating_point() else value
        for key, value in state_dict.items()
    }
