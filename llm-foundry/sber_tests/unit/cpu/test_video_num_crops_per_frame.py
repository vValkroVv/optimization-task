"""``num_crops_per_frame``: set by ``GigaOmniVideoProcessor`` per sample, carried
and split by ``VideoCollator``.

The processor gets pre-decoded videos (``file["content"]``), so only the sample
assembly runs; it still needs the video decode deps to import.
"""

import pytest
import torch

from sber_tests.test_utils.gigaomni_cpu import import_modules

for dependency in ("av", "decord", "cv2"):
    pytest.importorskip(dependency)

_, gigaomni_video_processor, video_collator = import_modules(
    "llmfoundry.data.processors.base_processor",
    "llmfoundry.data.processors.gigaomni_video_processor",
    "llmfoundry.data.collators.video_collator",
)
VideoCollator = video_collator.VideoCollator

pytestmark = [pytest.mark.unit, pytest.mark.fast]

TOKENS_PER_CROP = 4
VIDEO_TOKEN = 100


def build_processor():
    return gigaomni_video_processor.GigaOmniVideoProcessor(
        vision_processor_params={
            "num_image_tokens_per_crop": TOKENS_PER_CROP,
            "image_aspect_ratio": "anyres",
        },
        video_image_token_id=VIDEO_TOKEN,
        video_audio_tokens_order="sequential",
    )


def content(num_crops_per_frame):
    return {
        "pixel_values": torch.zeros(sum(num_crops_per_frame), 3, 8, 8),
        "frames_lengths": [c * TOKENS_PER_CROP for c in num_crops_per_frame],
        "timestamps": [float(i) for i in range(len(num_crops_per_frame))],
    }


@pytest.mark.parametrize(
    "videos,expected",
    [([[3, 3, 3]], 3), ([[5, 5], [5, 5, 5]], 5), ([[3, 3], [5, 5]], 0), ([[3, 2]], 0)],
)
def test_processor_count_is_shared_by_all_frames_or_zero(videos, expected):
    processor = build_processor()

    assert processor._get_num_crops_per_frame([content(v) for v in videos]) == expected


def test_processor_emits_the_count_with_matching_placeholders():
    files = [
        {"type": "video", "content": content([3, 3])},
        {"type": "video", "content": content([3, 3, 3])},
    ]
    input_ids = torch.tensor([1, VIDEO_TOKEN, 2, VIDEO_TOKEN, 3])

    sample = build_processor()(
        files=files, input_ids=input_ids, labels=input_ids.clone()
    )

    assert sample["video"]["num_crops_per_frame"] == 3
    assert sample["video"]["pixel_values"].shape[0] == 15
    assert (sample["text"]["input_ids"] == VIDEO_TOKEN).sum() == 15 * TOKENS_PER_CROP


def sample(num_crops, num_crops_per_frame):
    return {
        "pixel_values": torch.zeros(num_crops, 2),
        "num_crops_per_frame": num_crops_per_frame,
        "processor_fallback_count": 0,
    }


def test_collator_carries_and_splits_the_count_per_sample():
    examples = [sample(4, 2), None, sample(3, 1), sample(10, 5)]
    batch = {
        "input_ids": torch.zeros(len(examples), 3, dtype=torch.long),
        "modality_inputs": {"video": VideoCollator()(examples)},
    }

    microbatches = VideoCollator.split_batch(batch, microbatch_size=2)

    video = batch["modality_inputs"]["video"]
    assert video["num_crops_per_frame"].tolist() == [2, 0, 1, 5]
    assert video["num_crops_per_frame"].dtype == torch.long
    assert [mb["video"]["num_crops_per_frame"].tolist() for mb in microbatches] == [
        [2, 0],
        [1, 5],
    ]
    assert [mb["video"]["pixel_values"].shape[0] for mb in microbatches] == [4, 13]
