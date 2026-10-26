from __future__ import annotations

import numpy as np
import pytest

from metric_sampler.capture.layout import Layout
from metric_sampler.contract.catalog import metric_by_name
from metric_sampler.node.rank_slots import recorded_slot_indices, recorded_words, slot_key

pytestmark = pytest.mark.unit

_LOSS = metric_by_name("loss/total")
_COUNTS = metric_by_name("router/tokens_per_expert")
_BIAS = metric_by_name("router/expert_bias_before_update")
_BITS_PER_MASK_WORD = 32


@pytest.fixture
def three_slots() -> Layout:
    return Layout([(_LOSS, -1, -1, 1), (_COUNTS, 0, 1, 4), (_BIAS, 2, -1, 3)])


def _mask(three_slots: Layout, recorded: list[int]) -> np.ndarray:
    words = np.zeros(-(-len(three_slots.slots) // _BITS_PER_MASK_WORD), dtype=np.uint32)
    for index in recorded:
        words[index // _BITS_PER_MASK_WORD] |= np.uint32(1) << np.uint32(index % _BITS_PER_MASK_WORD)
    return words


def test_the_key_of_a_slot_is_its_metric_id_layer_and_microbatch(three_slots: Layout) -> None:
    assert [slot_key(slot) for slot in three_slots.slots] == [
        (_LOSS.metric_id, -1, -1),
        (_COUNTS.metric_id, 0, 1),
        (_BIAS.metric_id, 2, -1),
    ]


@pytest.mark.parametrize(
    ("recorded", "expected_words"),
    [([], 0), ([0], 1), ([1], 5), ([0, 2], 8), ([0, 1, 2], 8)],
)
def test_recorded_slots_and_the_words_up_to_the_last_of_them_follow_the_mask(
    three_slots: Layout, recorded: list[int], expected_words: int
) -> None:
    indices = recorded_slot_indices(three_slots, _mask(three_slots, recorded))

    assert indices.tolist() == recorded
    assert recorded_words(three_slots, indices) == expected_words


def test_mask_bits_beyond_the_last_slot_are_not_recorded_slots(three_slots: Layout) -> None:
    mask = _mask(three_slots, [1])
    mask[0] |= np.uint32(1) << np.uint32(len(three_slots.slots))

    assert recorded_slot_indices(three_slots, mask).tolist() == [1]
