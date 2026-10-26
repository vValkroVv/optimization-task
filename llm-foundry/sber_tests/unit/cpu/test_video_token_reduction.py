"""Video token reduction: short-clip clamp and multicrop fold.

Tiny ``P=4, D=8`` videos with seeded tensors; the oracle is omni-v1's
``_get_patchwise_compression_mask``, where the crops of a frame are compared
with the crops of the previous frame.
"""

import warnings

import pytest
import torch

from sber_tests.test_utils.gigaomni_cpu import import_modules

(video_token_reduction,) = import_modules(
    "llmfoundry.models.gigaomni.modalities.adapters.video_token_reduction"
)
VideoTokenReduction = video_token_reduction.VideoTokenReduction

pytestmark = [pytest.mark.unit, pytest.mark.fast]

P, D = 4, 8
VIDEO_TOKEN = 100
TEXT_PREFIX = [1, 2, 3]
TEXT_SUFFIX = [4]


def omni_v1_mask(videos, num_to_keep):
    """omni-v1 `_get_patchwise_compression_mask` (pixel_shuffle_projector.py), verbatim.

    It keeps the top `num_to_keep` of all T*PC diffs (a zero row stands for
    frame 0) and then forces frame 0, i.e. keeps budget + first frame.
    """
    B, T, PC, D = videos.shape
    diffs = (videos[:, 1:, :, :] - videos[:, :-1, :, :]).abs().mean(dim=-1)
    first = torch.zeros((B, 1, PC), device=videos.device)
    diffs = torch.cat([first, diffs], dim=1)
    flat_batched = diffs.view(B, -1)
    _, topk_indices = torch.topk(flat_batched, k=num_to_keep, dim=-1)
    mask = torch.zeros_like(flat_batched, dtype=torch.bool)
    mask.scatter_(dim=-1, index=topk_indices, value=True)
    mask = mask.view(B, T, PC)
    mask[:, 0, :] = True
    return mask


def build(avg_tokens_per_frame, frame_aspect_ratio="anyres"):
    return VideoTokenReduction(
        num_frames=8,
        avg_tokens_per_frame=avg_tokens_per_frame,
        frame_aspect_ratio=frame_aspect_ratio,
        video_image_token_id=VIDEO_TOKEN,
    )


def video(num_frames, num_crops=1):
    torch.manual_seed(0)
    pixel_embeds = torch.randn(num_frames * num_crops, P, D)
    input_ids = torch.tensor(
        [TEXT_PREFIX + [VIDEO_TOKEN] * (num_frames * num_crops * P) + TEXT_SUFFIX]
    )
    return pixel_embeds, input_ids


def run(module, num_frames, num_crops=1):
    pixel_embeds, input_ids = video(num_frames, num_crops)
    outputs = module(
        pixel_embeds=pixel_embeds,
        input_ids=input_ids,
        num_crops_per_frame=torch.tensor([num_crops]),
        labels=input_ids.clone(),
        attention_mask=torch.ones_like(input_ids),
    )
    return pixel_embeds, outputs


@pytest.mark.parametrize("num_frames", [2, 3, P])
def test_budget_up_to_one_frame_keeps_the_first_frame(num_frames):
    # avg 1 x T frames against P tokens per frame; below one frame it warns.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pixel_embeds, outputs = run(build(1, "pad"), num_frames)

    out_embeds, out_ids, out_labels, out_mask, text_mask = outputs
    assert torch.equal(out_embeds, pixel_embeds[0])
    assert out_ids.tolist() == [TEXT_PREFIX + [VIDEO_TOKEN] * P + TEXT_SUFFIX]
    assert out_labels.tolist() == [TEXT_PREFIX + [-100] * P + TEXT_SUFFIX]
    assert out_mask.shape == out_ids.shape
    assert text_mask.sum() == out_ids.shape[1]
    short = [w for w in caught if "first frame only" in str(w.message)]
    assert len(short) == int(num_frames < P)


def test_dummy_video_is_returned_unchanged():
    pixel_embeds, input_ids = video(1)

    outputs = build(1)(
        pixel_embeds=pixel_embeds,
        input_ids=input_ids,
        num_crops_per_frame=torch.tensor([0]),
    )

    assert outputs[0] is pixel_embeds
    assert outputs[1] is input_ids
    assert outputs[-1] is None


def test_pad_budget_of_a_whole_frame_is_rejected():
    with pytest.raises(AssertionError, match="nothing to reduce"):
        run(build(P, "pad"), 8)


@pytest.mark.parametrize(
    "num_frames,num_crops,avg",
    [(5, 1, 1), (8, 1, 2), (16, 1, 1), (6, 2, 5), (4, 3, 7), (3, 5, 15)],
)
def test_selection_matches_omni_v1(num_frames, num_crops, avg):
    PC = P * num_crops

    pixel_embeds, outputs = run(build(avg), num_frames, num_crops)

    out_embeds, out_ids, out_labels, _, text_mask = outputs
    folded = pixel_embeds.view(1, num_frames, PC, D)
    assert torch.equal(out_embeds, folded[omni_v1_mask(folded, avg * num_frames - PC)])
    assert (out_ids == VIDEO_TOKEN).sum() == avg * num_frames
    assert (out_labels == -100).sum() == avg * num_frames
    assert text_mask.sum() == out_ids.shape[1]


def test_motion_in_one_crop_is_found_in_that_crop():
    pixel_embeds = video(1, num_crops=2)[0]
    second_frame = pixel_embeds.clone()
    second_frame[1, [0, 3]] += 100.0
    pixel_embeds = torch.cat([pixel_embeds, second_frame])

    out_embeds = build(5)(
        pixel_embeds=pixel_embeds,
        input_ids=torch.tensor([[VIDEO_TOKEN] * (2 * 2 * P)]),
        num_crops_per_frame=torch.tensor([2]),
    )[0]

    # budget 5 x 2 frames = the first frame (2 crops of P) + the two moved patches
    assert torch.equal(out_embeds[: 2 * P], pixel_embeds[:2].reshape(-1, D))
    assert torch.equal(out_embeds[2 * P :], second_frame[1, [0, 3]])


def test_kept_positions_follow_the_selected_patches():
    # A text token between the frames and attention_mask = arange(L) tell the
    # positions apart; the omni-v1 oracle says which patches survive.
    num_frames, num_crops, avg = 3, 2, 5
    PC = P * num_crops
    pixel_embeds = video(num_frames, num_crops)[0]
    frame = [VIDEO_TOKEN] * PC
    input_ids = torch.tensor(
        [TEXT_PREFIX + frame + [7] + frame + [8] + frame + TEXT_SUFFIX]
    )
    folded = pixel_embeds.view(1, num_frames, PC, D)
    oracle_mask = omni_v1_mask(folded, avg * num_frames - PC)
    video_positions = (input_ids[0] == VIDEO_TOKEN).nonzero().flatten()
    text_positions = (input_ids[0] != VIDEO_TOKEN).nonzero().flatten()
    expected = torch.cat([video_positions[oracle_mask.flatten()], text_positions])
    expected = expected.sort().values.tolist()

    out_embeds, out_ids, _, out_mask, text_mask = build(avg)(
        pixel_embeds=pixel_embeds,
        input_ids=input_ids,
        num_crops_per_frame=torch.tensor([num_crops]),
        attention_mask=torch.arange(input_ids.shape[1]).unsqueeze(0),
    )

    assert out_mask[0].tolist() == expected
    assert text_mask[0].nonzero().flatten().tolist() == expected
    assert out_ids[0].tolist() == input_ids[0][expected].tolist()
    assert torch.equal(out_embeds, folded[oracle_mask])


@pytest.mark.parametrize("num_frames,num_crops,avg", [(4, 2, 8), (4, 2, 9), (1, 3, 2)])
def test_budget_covering_the_frames_keeps_the_video_whole(num_frames, num_crops, avg):
    # avg >= P * C under anyres, or a single frame with nothing to compare to.
    pixel_embeds, outputs = run(build(avg), num_frames, num_crops)

    out_embeds, out_ids, _, _, text_mask = outputs
    assert torch.equal(out_embeds, pixel_embeds.reshape(-1, D))
    assert (out_ids == VIDEO_TOKEN).sum() == pixel_embeds.shape[0] * P
    assert text_mask.all()


def test_mixed_crop_counts_in_the_sample_are_rejected():
    pixel_embeds, input_ids = video(4, num_crops=2)

    with pytest.raises(AssertionError, match="same number of crops per frame"):
        build(2)(
            pixel_embeds=pixel_embeds,
            input_ids=input_ids,
            num_crops_per_frame=torch.tensor([0]),
        )
