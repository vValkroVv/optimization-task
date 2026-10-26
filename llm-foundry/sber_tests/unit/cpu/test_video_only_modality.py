"""``VideoOnlyModality``: the token-reduction ``text_mask`` shortens the text
``inputs_embeds`` like the adapter shortened ``input_ids``, and the adapter
inputs carry ``num_crops_per_frame``.
"""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from sber_tests.test_utils.gigaomni_cpu import import_modules

(video_only_modality,) = import_modules(
    "llmfoundry.models.gigaomni.modalities.video_only_modality"
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]

PRESET = "video_only_token_reduction_intern_vit"
BUNDLE = "pixel_shuffle_mlp.video"


def build_modality():
    config = video_only_modality.VideoOnlyModalityConfig(data_flow_preset=PRESET)
    return video_only_modality.VideoOnlyModality(config)


def build_state(batch_size, seq_len, text_mask, adapter_keys=("input_ids",)):
    torch.manual_seed(0)
    inputs_embeds = torch.randn(batch_size, seq_len, 3)
    adapter_outputs = {key: torch.zeros(batch_size, seq_len) for key in adapter_keys}

    if text_mask is not None or "text_mask" in adapter_keys:
        adapter_outputs["text_mask"] = text_mask

    return SimpleNamespace(
        modality_inputs=defaultdict(dict, {"text": {}}),
        encoder_outputs=defaultdict(dict, {"text": {"inputs_embeds": inputs_embeds}}),
        adapter_outputs=defaultdict(dict, {BUNDLE: adapter_outputs}),
        pad_to_multiple_of=1,
        pad_token_id=0,
    )


def test_text_mask_filters_inputs_embeds():
    modality = build_modality()
    text_mask = torch.tensor([[True, False, True, True, False]])
    state = build_state(1, 5, text_mask)
    original = state.encoder_outputs["text"]["inputs_embeds"].clone()

    modality.process_inputs_for_language_model(state)

    filtered = state.encoder_outputs["text"]["inputs_embeds"]
    assert filtered.shape == (1, 3, 3)
    assert torch.equal(filtered[0], original[0][[0, 2, 3]])
    assert "input_ids" in state.modality_inputs["text"]


def test_text_mask_with_packed_position_ids_raises():
    modality = build_modality()
    state = build_state(1, 5, torch.ones(1, 5, dtype=torch.bool))
    state.modality_inputs["text"]["position_ids"] = torch.arange(5).unsqueeze(0)

    with pytest.raises(AssertionError, match="position_ids"):
        modality.process_inputs_for_language_model(state)


@pytest.mark.parametrize("adapter_keys", [("input_ids",), ("input_ids", "text_mask")])
def test_without_text_mask_inputs_embeds_are_untouched(adapter_keys):
    modality = build_modality()
    state = build_state(2, 5, None, adapter_keys=adapter_keys)
    original = state.encoder_outputs["text"]["inputs_embeds"]

    modality.process_inputs_for_language_model(state)

    assert state.encoder_outputs["text"]["inputs_embeds"] is original


ENCODER_BUNDLE = "intern_vit.video"


def build_adapter_state(video_inputs):
    torch.manual_seed(0)
    return SimpleNamespace(
        modality_inputs=defaultdict(
            dict,
            {
                "text": {
                    "input_ids": torch.zeros(1, 4, dtype=torch.long),
                    "labels": torch.zeros(1, 4, dtype=torch.long),
                },
                "video": video_inputs,
            },
        ),
        custom_inputs=defaultdict(dict),
        encoder_outputs=defaultdict(
            dict, {ENCODER_BUNDLE: {"pixel_embeds": torch.randn(2, 4, 3)}}
        ),
        adapter_outputs=defaultdict(dict),
    )


def test_adapter_inputs_carry_num_crops_per_frame():
    modality = build_modality()
    num_crops_per_frame = torch.tensor([3])
    state = build_adapter_state(
        {
            "pixel_values": torch.zeros(2, 3, 8, 8),
            "num_crops_per_frame": num_crops_per_frame,
        }
    )

    modality.process_inputs_for_adapters(state)

    adapter_inputs = state.custom_inputs["adapter_inputs"]
    assert adapter_inputs["num_crops_per_frame"] is num_crops_per_frame
    assert adapter_inputs["attention_mask"] is None
    assert set(adapter_inputs) == set(
        modality.get_custom_input_contract()["adapter_inputs"]
    )


def test_token_reduction_preset_graph_is_consistent_with_the_contract():
    modality = build_modality()

    modality.config.data_flow.validate_graph(
        available_components={
            "encoders": ["InternViTForVisualLanguageModeling"],
            "adapters": ["VisionPixelShuffleAdapter"],
            "heads": [],
        },
        modality_output=modality.config.modality_output,
        modality_name=modality.name,
        custom_input_contract=modality.get_custom_input_contract(),
    )
