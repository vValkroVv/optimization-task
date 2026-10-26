from typing import Any, Dict, Union


vision_intern_vit_data_flow: Dict[str, Any] = {
    "encoders": [
        {
            "name": "InternViTForVisualLanguageModeling",
            "inputs": [
                {
                    "type": "modality_inputs",
                    "name": "image",
                    "keys": ["pixel_values"],
                }
            ],
            "outputs": [
                {
                    "name": "intern_vit.image",
                    "keys": ["pixel_embeds"],
                }
            ],
        }
    ],
    "adapters": [
        {
            "name": "VisionPixelShuffleAdapter",
            "inputs": [
                {
                    "type": "encoder_outputs",
                    "name": "intern_vit.image",
                    "keys": ["pixel_embeds"],
                }
            ],
            "outputs": [
                {
                    "name": "pixel_shuffle_mlp.image",
                    "keys": ["pixel_embeds"],
                }
            ],
        }
    ],
}


vision_qwen_vit_data_flow: Dict[str, Any] = {
    "encoders": [
        {
            "name": "Qwen2_5ViTForVisualLanguageModeling",
            "inputs": [
                {
                    "type": "modality_inputs",
                    "name": "image",
                    "keys": ["pixel_values", "grid_thw"],
                }
            ],
            "outputs": [
                {
                    "name": "qwen_vit.image",
                    "keys": ["pixel_embeds", "window_index"],
                }
            ],
        }
    ],
    "adapters": [
        {
            "name": "VisionQwen2_5Adapter",
            "inputs": [
                {
                    "type": "encoder_outputs",
                    "name": "qwen_vit.image",
                    "keys": ["pixel_embeds", "window_index"],
                }
            ],
            "outputs": [
                {
                    "name": "qwen_adapter.image",
                    "keys": ["pixel_embeds"],
                }
            ],
        }
    ],
}


video_only_intern_vit_data_flow: Dict[str, Any] = {
    "encoders": [
        {
            "name": "InternViTForVisualLanguageModeling",
            "inputs": [
                {
                    "type": "modality_inputs",
                    "name": "video",
                    "keys": ["pixel_values"],
                }
            ],
            "outputs": [
                {
                    "name": "intern_vit.video",
                    "keys": ["pixel_embeds"],
                }
            ],
        }
    ],
    "adapters": [
        {
            "name": "VisionPixelShuffleAdapter",
            "inputs": [
                {
                    "type": "custom_inputs",
                    "name": "adapter_inputs",
                    "keys": ["pixel_embeds"],
                }
            ],
            "outputs": [
                {
                    "name": "pixel_shuffle_mlp.video",
                    "keys": ["pixel_embeds"],
                }
            ],
        }
    ],
}


video_only_token_reduction_intern_vit_data_flow: Dict[str, Any] = {
    "encoders": [
        {
            "name": "InternViTForVisualLanguageModeling",
            "inputs": [
                {
                    "type": "modality_inputs",
                    "name": "video",
                    "keys": ["pixel_values"],
                }
            ],
            "outputs": [
                {
                    "name": "intern_vit.video",
                    "keys": ["pixel_embeds"],
                }
            ],
        }
    ],
    "adapters": [
        {
            "name": "VisionPixelShuffleAdapter",
            "inputs": [
                {
                    "type": "custom_inputs",
                    "name": "adapter_inputs",
                    "keys": [
                        "input_ids",
                        "attention_mask",
                        "labels",
                        "pixel_embeds",
                        "num_crops_per_frame",
                    ],
                }
            ],
            "outputs": [
                {
                    "name": "pixel_shuffle_mlp.video",
                    "keys": [
                        "input_ids",
                        "attention_mask",
                        "labels",
                        "pixel_embeds",
                        "text_mask",
                    ],
                }
            ],
        }
    ],
}


video_only_qwen_vit_data_flow: Dict[str, Any] = {
    "encoders": [
        {
            "name": "Qwen2_5ViTForVisualLanguageModeling",
            "inputs": [
                {
                    "type": "modality_inputs",
                    "name": "video",
                    "keys": ["pixel_values", "grid_thw"],
                }
            ],
            "outputs": [
                {
                    "name": "qwen_vit.video",
                    "keys": ["pixel_embeds", "window_index"],
                }
            ],
        }
    ],
    "adapters": [
        {
            "name": "VisionQwen2_5Adapter",
            "inputs": [
                {
                    "type": "encoder_outputs",
                    "name": "qwen_vit.video",
                    "keys": ["pixel_embeds", "window_index"],
                }
            ],
            "outputs": [
                {
                    "name": "qwen_adapter.video",
                    "keys": ["pixel_embeds"],
                }
            ],
        }
    ],
}


vision_intern_vit_modality_output: Dict[str, Union[str, int]] = {
    "type": "adapter_outputs",
    "name": "pixel_shuffle_mlp.image",
    "key": "pixel_embeds",
    "placeholder_token_id": 128004,
}


vision_qwen_vit_modality_output: Dict[str, Union[str, int]] = {
    "type": "adapter_outputs",
    "name": "qwen_adapter.image",
    "key": "pixel_embeds",
    "placeholder_token_id": 128004,
}


video_only_intern_vit_modality_output: Dict[str, Union[str, int]] = {
    "type": "adapter_outputs",
    "name": "pixel_shuffle_mlp.video",
    "key": "pixel_embeds",
    "placeholder_token_id": 128005,
}


video_only_qwen_vit_modality_output: Dict[str, Union[str, int]] = {
    "type": "adapter_outputs",
    "name": "qwen_adapter.video",
    "key": "pixel_embeds",
    "placeholder_token_id": 128005,
}


vision_qwen3_vit_data_flow: Dict[str, Any] = {
    "encoders": [
        {
            "name": "Qwen3ViTForVisualLanguageModeling",
            "inputs": [
                {
                    "type": "modality_inputs",
                    "name": "image",
                    "keys": ["pixel_values", "grid_thw"],
                }
            ],
            "outputs": [
                {
                    "name": "qwen3_vit.image",
                    "keys": ["pixel_embeds"],
                }
            ],
        }
    ],
    "adapters": [
        {
            "name": "VisionQwen3Adapter",
            "inputs": [
                {
                    "type": "encoder_outputs",
                    "name": "qwen3_vit.image",
                    "keys": ["pixel_embeds"],
                }
            ],
            "outputs": [
                {
                    "name": "qwen3_adapter.image",
                    "keys": ["pixel_embeds"],
                }
            ],
        }
    ],
}

video_only_qwen3_vit_data_flow: Dict[str, Any] = {
    "encoders": [
        {
            "name": "Qwen3ViTForVisualLanguageModeling",
            "inputs": [
                {
                    "type": "modality_inputs",
                    "name": "video",
                    "keys": ["pixel_values", "grid_thw"],
                }
            ],
            "outputs": [
                {
                    "name": "qwen3_vit.video",
                    "keys": ["pixel_embeds"],
                }
            ],
        }
    ],
    "adapters": [
        {
            "name": "VisionQwen3Adapter",
            "inputs": [
                {
                    "type": "encoder_outputs",
                    "name": "qwen3_vit.video",
                    "keys": ["pixel_embeds"],
                }
            ],
            "outputs": [
                {
                    "name": "qwen3_adapter.video",
                    "keys": ["pixel_embeds"],
                }
            ],
        }
    ],
}

vision_qwen3_vit_modality_output: Dict[str, Union[str, int]] = {
    "type": "adapter_outputs",
    "name": "qwen3_adapter.image",
    "key": "pixel_embeds",
    "placeholder_token_id": 128004,
}

video_only_qwen3_vit_modality_output: Dict[str, Union[str, int]] = {
    "type": "adapter_outputs",
    "name": "qwen3_adapter.video",
    "key": "pixel_embeds",
    "placeholder_token_id": 128005,
}

audio_conformer_data_flow: Dict[str, Any] = {
    "encoders": [
        {
            "name": "ConformerForAudioLanguageModeling",
            "inputs": [
                {
                    "type": "modality_inputs",
                    "name": "audio",
                    "keys": ["spectrograms", "spectrogram_lengths"],
                }
            ],
            "outputs": [
                {
                    "name": "conformer.audio",
                    "keys": ["spectrograms", "spectrogram_lengths"],
                }
            ],
        }
    ],
    "adapters": [
        {
            "name": "AudioLinearAdapterWithStackingSubsampler",
            "inputs": [
                {
                    "type": "encoder_outputs",
                    "name": "conformer.audio",
                    "keys": ["spectrograms", "spectrogram_lengths"],
                }
            ],
            "outputs": [
                {
                    "name": "audio_linear_adapter.audio",
                    "keys": ["audio_embeds"],
                }
            ],
        }
    ],
}

audio_conformer_modality_output: Dict[str, Union[str, int]] = {
    "type": "adapter_outputs",
    "name": "audio_linear_adapter.audio",
    "key": "audio_embeds",
    "placeholder_token_id": 128006,
}

# Soundtracks of `video_audio` files arrive in the `video_audio` bundle
# (VideoAudioCollator) and have their own placeholder id. The preset names the
# same conformer/adapter components as `audio_conformer`, so a model with both
# modalities runs each component once per microbatch on both bundles.
video_audio_conformer_data_flow: Dict[str, Any] = {
    "encoders": [
        {
            "name": "ConformerForAudioLanguageModeling",
            "inputs": [
                {
                    "type": "modality_inputs",
                    "name": "video_audio",
                    "keys": ["spectrograms", "spectrogram_lengths"],
                }
            ],
            "outputs": [
                {
                    "name": "conformer.video_audio",
                    "keys": ["spectrograms", "spectrogram_lengths"],
                }
            ],
        }
    ],
    "adapters": [
        {
            "name": "AudioLinearAdapterWithStackingSubsampler",
            "inputs": [
                {
                    "type": "encoder_outputs",
                    "name": "conformer.video_audio",
                    "keys": ["spectrograms", "spectrogram_lengths"],
                }
            ],
            "outputs": [
                {
                    "name": "audio_linear_adapter.video_audio",
                    "keys": ["audio_embeds"],
                }
            ],
        }
    ],
}

video_audio_conformer_modality_output: Dict[str, Union[str, int]] = {
    "type": "adapter_outputs",
    "name": "audio_linear_adapter.video_audio",
    "key": "audio_embeds",
    "placeholder_token_id": 128007,
}

data_flow_presets: Dict[str, Dict[str, Any]] = {
    "vision_intern_vit": vision_intern_vit_data_flow,
    "vision_qwen_vit": vision_qwen_vit_data_flow,
    "vision_qwen3_vit": vision_qwen3_vit_data_flow,
    "video_only_intern_vit": video_only_intern_vit_data_flow,
    "video_only_token_reduction_intern_vit": video_only_token_reduction_intern_vit_data_flow,
    "video_only_qwen_vit": video_only_qwen_vit_data_flow,
    "video_only_qwen3_vit": video_only_qwen3_vit_data_flow,
    "audio_conformer": audio_conformer_data_flow,
    "video_audio_conformer": video_audio_conformer_data_flow,
}

modality_output_presets: Dict[str, Dict[str, Union[str, int]]] = {
    "vision_intern_vit": vision_intern_vit_modality_output,
    "vision_qwen_vit": vision_qwen_vit_modality_output,
    "vision_qwen3_vit": vision_qwen3_vit_modality_output,
    "video_only_intern_vit": video_only_intern_vit_modality_output,
    "video_only_token_reduction_intern_vit": video_only_intern_vit_modality_output,
    "video_only_qwen_vit": video_only_qwen_vit_modality_output,
    "video_only_qwen3_vit": video_only_qwen3_vit_modality_output,
    "audio_conformer": audio_conformer_modality_output,
    "video_audio_conformer": video_audio_conformer_modality_output,
}
