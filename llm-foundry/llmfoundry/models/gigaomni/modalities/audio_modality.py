from __future__ import annotations

import torch

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from .base_modality import BaseModality, BaseModalityConfig
from .data_flow import build_data_flow_from_dict
from .data_flow_presets import data_flow_presets, modality_output_presets

if TYPE_CHECKING:
    from ..modelling_gigaomni import GigaOmniDataState


class AudioModalityConfig(BaseModalityConfig):
    model_type = "giga_omni_audio_modality"
    allowed_data_flow_presets = ("audio_conformer", "video_audio_conformer")

    def __init__(
        self,
        data_flow_preset: Optional[str] = None,
        data_flow: Optional[Dict[str, Any]] = None,
        modality_output: Optional[Dict[str, Union[str, int]]] = None,
        **kwargs: Any,
    ) -> None:
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


class AudioModality(BaseModality):
    config_class = AudioModalityConfig

    def process_inputs_for_encoders(self, data_state: GigaOmniDataState) -> None:
        self.config.data_flow.validate_runtime_inputs(
            data_state, self.config.data_flow.encoders
        )

    def process_inputs_for_adapters(self, data_state: GigaOmniDataState) -> None:
        self.config.data_flow.validate_runtime_inputs(
            data_state, self.config.data_flow.adapters
        )

    def process_inputs_for_language_model(self, data_state: GigaOmniDataState) -> None:
        pass

    def process_inputs_for_heads(self, data_state: GigaOmniDataState) -> None:
        pass

    def update_modality_inputs_with_dummy_values(
        self, modality_inputs: Dict[str, Dict[str, Any]]
    ) -> None:
        # `audio` for `audio_conformer`, `video_audio` for `video_audio_conformer`.
        bundle_name = self.config.data_flow.encoders[0].inputs[0].name

        if bundle_name in modality_inputs:
            return

        # All ranks must execute the acoustic modules, including ranks whose local
        # batch contains no audio. The default feature size matches the audio
        # processor; custom feature sizes should provide an explicit audio input.
        modality_inputs[bundle_name] = {
            "spectrograms": torch.zeros(20, 64, device="cuda", dtype=torch.float32),
            "spectrogram_lengths": torch.tensor([20], device="cuda", dtype=torch.long),
            "dummy_value": True,
        }
