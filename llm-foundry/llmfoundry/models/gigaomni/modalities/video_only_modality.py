from __future__ import annotations

import torch
import torch.nn.functional as F

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from .base_modality import BaseModalityConfig, BaseModality
from .data_flow import build_data_flow_from_dict
from .data_flow_presets import (
    data_flow_presets,
    modality_output_presets,
)

if TYPE_CHECKING:
    from ..modelling_gigaomni import GigaOmniDataState


def _pad_to_multiple(data_state: GigaOmniDataState) -> None:
    """Pad the text sequence to ``data_state.pad_to_multiple_of`` positions.

    The collator aligns the length to the divisor the language model shards
    by, and token reduction then drops an arbitrary number of video tokens.
    Appended positions get ``pad_token_id``, label ``-100``, mask ``0`` and a
    zero embedding, so they never reach the loss. A state built by an older
    bundled ``modelling_gigaomni.py`` has neither field and is not padded,
    like ``generate()``.
    """
    text_inputs = data_state.modality_inputs["text"]
    inputs_embeds = data_state.encoder_outputs["text"]["inputs_embeds"]
    multiple = getattr(data_state, "pad_to_multiple_of", 1)
    pad_length = -inputs_embeds.shape[1] % multiple

    if pad_length == 0:
        return

    text_inputs["input_ids"] = F.pad(
        text_inputs["input_ids"],
        (0, pad_length),
        value=getattr(data_state, "pad_token_id", 0),
    )

    for key, value in (("labels", -100), ("attention_mask", 0)):
        if text_inputs.get(key) is not None:
            text_inputs[key] = F.pad(text_inputs[key], (0, pad_length), value=value)

    data_state.encoder_outputs["text"]["inputs_embeds"] = F.pad(
        inputs_embeds, (0, 0, 0, pad_length)
    )


class VideoOnlyModalityConfig(BaseModalityConfig):
    allowed_data_flow_presets = (
        "video_only_intern_vit",
        "video_only_qwen_vit",
        "video_only_qwen3_vit",
        "video_only_token_reduction_intern_vit",
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


class VideoOnlyModality(BaseModality):
    config_class = VideoOnlyModalityConfig

    def __init__(self, config: VideoOnlyModalityConfig):
        super().__init__(config)

    def process_inputs_for_encoders(self, data_state: GigaOmniDataState) -> None:
        self.config.data_flow.validate_runtime_inputs(
            data_state, self.config.data_flow.encoders
        )

    def process_inputs_for_adapters(self, data_state: GigaOmniDataState) -> None:
        encoder_outputs = self.config.data_flow.encoders[0].outputs[0].name
        data_state.custom_inputs["adapter_inputs"] = {
            "input_ids": data_state.modality_inputs["text"]["input_ids"],
            "labels": data_state.modality_inputs["text"].get("labels"),
            "attention_mask": data_state.modality_inputs["text"].get("attention_mask"),
            "pixel_embeds": data_state.encoder_outputs[encoder_outputs]["pixel_embeds"],
            "num_crops_per_frame": data_state.modality_inputs["video"].get(
                "num_crops_per_frame"
            ),
        }

        self.config.data_flow.validate_runtime_inputs(
            data_state, self.config.data_flow.adapters
        )

    def process_inputs_for_language_model(self, data_state: GigaOmniDataState) -> None:
        adapter_bundle = self.config.modality_output["name"]
        for key in ["input_ids", "attention_mask", "labels"]:
            if key in data_state.adapter_outputs[adapter_bundle]:
                data_state.modality_inputs["text"][key] = data_state.adapter_outputs[
                    adapter_bundle
                ][key]

        text_mask = data_state.adapter_outputs[adapter_bundle].get("text_mask")

        if text_mask is not None:
            # Packed positions would not be re-numbered after the deletion.
            assert data_state.modality_inputs["text"].get("position_ids") is None, (
                "Token reduction does not support `position_ids` in the text"
                " inputs (TextFlashAttnCollator), use a padding text collator!"
            )

            # The reducer asserts batch_size = 1, unsqueeze(0) restores the batch axis.
            inputs_embeds = data_state.encoder_outputs["text"]["inputs_embeds"]
            inputs_embeds = inputs_embeds[text_mask].unsqueeze(0)
            data_state.encoder_outputs["text"]["inputs_embeds"] = inputs_embeds
            _pad_to_multiple(data_state)

    def process_inputs_for_heads(self, data_state: GigaOmniDataState) -> None:
        pass

    def update_modality_inputs_with_dummy_values(
        self, modality_inputs: Dict[str, Dict[str, Any]]
    ) -> None:
        if "video" in modality_inputs:
            return

        if self.config.data_flow_preset in [
            "video_only_intern_vit",
            "video_only_token_reduction_intern_vit",
        ]:
            modality_inputs.update(
                {
                    "video": {
                        # TODO (sbr): get crop size from config and correct device identification
                        "pixel_values": torch.zeros(
                            1, 3, 448, 448, device="cuda", dtype=torch.bfloat16
                        ),
                        "n_crops_per_sample": torch.tensor(
                            [1], device="cuda", dtype=torch.long
                        ),
                        "num_crops_per_frame": torch.tensor(
                            [1], device="cuda", dtype=torch.long
                        ),
                        "dummy_value": True,
                    }
                }
            )

        if self.config.data_flow_preset in [
            "video_only_qwen_vit",
            "video_only_qwen3_vit",
        ]:
            patch_size = (
                16 if self.config.data_flow_preset == "video_only_qwen3_vit" else 14
            )
            flattened_patch_size = 3 * 2 * patch_size**2
            modality_inputs.update(
                {
                    "video": {
                        # TODO (sbr): get crop size from config and correct device identification
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

    def get_custom_input_contract(self) -> Dict[str, list[str]]:
        return {
            "adapter_inputs": [
                "input_ids",
                "labels",
                "attention_mask",
                "pixel_embeds",
                "num_crops_per_frame",
            ]
        }
