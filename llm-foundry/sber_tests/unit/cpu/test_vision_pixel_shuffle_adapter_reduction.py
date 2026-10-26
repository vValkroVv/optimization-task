"""``VisionPixelShuffleAdapter`` hands ``num_crops_per_frame`` to the reducer.

The adapter is the only production site that reads the per-sample crop count
out of its input bundle; a 4x4-patch crop shuffles down to ``P = 4`` tokens so
the kept counts are small and exact.
"""

import pytest
import torch

from sber_tests.test_utils.gigaomni_cpu import import_modules

(vision_pixel_shuffle_adapter,) = import_modules(
    "llmfoundry.models.gigaomni.modalities.adapters.vision_pixel_shuffle_adapter"
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]

VE_HIDDEN, LLM_HIDDEN, PATCHES, P = 8, 16, 16, 4
VIDEO_TOKEN = 100
AVG = 3


def build_adapter():
    torch.manual_seed(0)
    return vision_pixel_shuffle_adapter.VisionPixelShuffleAdapter(
        ve_hidden_size=VE_HIDDEN,
        llm_hidden_size=LLM_HIDDEN,
        downsample_ratio=0.5,
        use_video_tokens_reduction=True,
        video_tokens_reduction_params={
            "num_frames": 8,
            "avg_tokens_per_frame": AVG,
            "frame_aspect_ratio": "anyres",
            "video_image_token_id": VIDEO_TOKEN,
        },
    )


def bundle(num_frames, num_crops):
    torch.manual_seed(1)
    input_ids = torch.tensor([[1, 2] + [VIDEO_TOKEN] * (num_frames * num_crops * P)])
    return {
        "pixel_embeds": torch.randn(num_frames * num_crops, PATCHES, VE_HIDDEN),
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "attention_mask": torch.ones_like(input_ids),
        "num_crops_per_frame": torch.tensor([num_crops]),
    }


def test_adapter_passes_the_crop_count_to_the_reducer():
    adapter = build_adapter()
    inputs = [bundle(4, 2)]

    outputs = adapter.unwrap_tuple_output(inputs, adapter(inputs))

    (output,) = outputs
    assert set(output) == {
        "pixel_embeds",
        "input_ids",
        "labels",
        "attention_mask",
        "text_mask",
    }
    assert output["pixel_embeds"].shape == (AVG * 4, LLM_HIDDEN)
    assert (output["input_ids"] == VIDEO_TOKEN).sum() == AVG * 4
    assert output["text_mask"].sum() == output["input_ids"].shape[1]


def test_bundles_without_input_ids_bypass_the_reduction():
    adapter = build_adapter()
    image = {"pixel_embeds": torch.randn(2, PATCHES, VE_HIDDEN)}

    (output,) = adapter.unwrap_tuple_output([image], adapter([image]))

    assert set(output) == {"pixel_embeds"}
    assert output["pixel_embeds"].shape == (2, P, LLM_HIDDEN)
