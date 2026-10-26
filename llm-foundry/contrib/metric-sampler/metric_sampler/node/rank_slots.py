from __future__ import annotations

import numpy as np

from metric_sampler.capture.layout import Layout, Slot

# (metric_id, layer, microbatch) of a recorded slot.
SlotKey = tuple[int, int, int]
# A recorded slot's words as the ring holds them, and the same words viewed as the metric's value type.
SlotWords = tuple[np.ndarray, np.ndarray]
# The slots one rank recorded in one step.
RankSlots = dict[SlotKey, SlotWords]


def slot_key(slot: Slot) -> SlotKey:
    return (slot.metric.metric_id, slot.layer, slot.microbatch)


def recorded_slot_indices(layout: Layout, mask: np.ndarray) -> np.ndarray:
    """The indices in `layout.slots` of the slots whose bit is set in the recorded-slots mask, found with one
    vectorized unpack instead of a bit test per slot."""
    return np.flatnonzero(np.unpackbits(mask.view(np.uint8), bitorder="little")[: len(layout.slots)])


def recorded_words(layout: Layout, recorded_indices: np.ndarray) -> int:
    """The words of the payload up to the end of the last recorded slot. Detailed slots occupy the tail, so a step
    that did not record them is far shorter than the layout."""
    if not len(recorded_indices):
        return 0
    last_slot = layout.slots[recorded_indices[-1]]
    return last_slot.offset + last_slot.width
