from __future__ import annotations

import torch

from typing import TYPE_CHECKING, Dict, Any, Optional, Union

from .base_modality import BaseModalityConfig, BaseModality
from .data_flow import build_data_flow_from_dict
from .data_flow_presets import (
    data_flow_presets,
    modality_output_presets,
)

if TYPE_CHECKING:
    from ..modelling_gigaomni import GigaOmniDataState


class VisionModalityConfig(BaseModalityConfig):
    model_type = "giga_omni_vision_modality"
    allowed_data_flow_presets = (
        "vision_intern_vit",
        "vision_qwen_vit",
        "vision_qwen3_vit",
    )

    def __init__(
        self,
        data_flow_preset: Optional[str] = None,
        data_flow: Optional[Dict[str, Any]] = None,
        modality_output: Optional[Dict[str, Union[str, int]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.validate_data_flow_preset_compatibility(data_flow_preset)
        self.data_flow_preset = data_flow_preset
        self.data_flow = build_data_flow_from_dict(
            self.merge_from_preset(
                data_flow_preset,
                data_flow,
                data_flow_presets,
                mode="data_flow",
            ),
        )

        self.modality_output = self.merge_from_preset(
            data_flow_preset,
            modality_output,
            modality_output_presets,
            mode="modality_output",
        )


class VisionModality(BaseModality):
    model_type = "giga_omni_vision_modality"
    config_class = VisionModalityConfig

    def __init__(self, config: VisionModalityConfig):
        super().__init__(config)

    def process_inputs_for_encoders(self, data_state: GigaOmniDataState) -> None:
        self.config.data_flow.validate_runtime_inputs(
            data_state,
            self.config.data_flow.encoders,
        )

    def process_inputs_for_adapters(self, data_state: GigaOmniDataState) -> None:
        self.config.data_flow.validate_runtime_inputs(
            data_state,
            self.config.data_flow.adapters,
        )

    def process_inputs_for_language_model(self, data_state: GigaOmniDataState) -> None:
        pass

    def process_inputs_for_heads(self, data_state: GigaOmniDataState) -> None:
        pass

    def update_modality_inputs_with_dummy_values(
        self, modality_inputs: Dict[str, Any]
    ) -> None:
        if "image" in modality_inputs:
            return

        if self.config.data_flow_preset == "vision_intern_vit":
            modality_inputs.update(
                {
                    "image": {
                        # TODO (avtereshchenko): get image crop size from config and correct device identification
                        "pixel_values": torch.zeros(
                            1, 3, 448, 448, device="cuda", dtype=torch.bfloat16
                        ),
                        "n_crops_per_sample": torch.tensor(
                            [1], device="cuda", dtype=torch.long
                        ),
                        "dummy_value": True,
                    }
                }
            )

        if self.config.data_flow_preset in ["vision_qwen_vit", "vision_qwen3_vit"]:
            patch_size = (
                16 if self.config.data_flow_preset == "vision_qwen3_vit" else 14
            )
            flattened_patch_size = 3 * 2 * patch_size**2
            modality_inputs.update(
                {
                    "image": {
                        # TODO (avtereshchenko): get image crop size from config and correct device identification
                        "pixel_values": torch.zeros(
                            4, flattened_patch_size, device="cuda", dtype=torch.bfloat16
                        ),
                        "grid_thw": torch.tensor(
                            [[1, 2, 2]], device="cuda", dtype=torch.long
                        ),
                        "dummy_value": True,
                    }
                }
            )
