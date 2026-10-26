from __future__ import annotations

import os
import uuid
from multiprocessing import shared_memory

import numpy as np
import pytest

from metric_sampler.capture.layout import Layout
from metric_sampler.capture.ring import (
    STATE_FREE,
    STATE_READY,
    STATE_WRITING,
    RingNotReady,
    ShmRing,
    SlotHeader,
    mask_word_count,
)
from metric_sampler.capture.shm import ALIGNMENT, WORD_BYTES, round_up
from metric_sampler.contract.catalog import metric_by_name
from metric_sampler.contract.metrics import SCHEMA_VERSION

pytestmark = pytest.mark.unit

_LOSS = metric_by_name("loss/total")
_COUNTS = metric_by_name("router/tokens_per_expert")


@pytest.fixture
def layout() -> Layout:
    return Layout([(_LOSS, -1, -1, 1), (_COUNTS, 0, -1, 4)])


@pytest.mark.parametrize(("slot_count", "expected"), [(0, 0), (1, 1), (32, 1), (33, 2), (64, 2), (65, 3)])
def test_mask_word_count(slot_count: int, expected: int) -> None:
    assert mask_word_count(slot_count) == expected


def test_create_writes_control_block_fields(shm_rings, layout: Layout) -> None:
    ring = shm_rings(layout, slot_count=3)

    assert ring.slot_count == 3
    assert ring.layout.words == layout.words
    assert ring.mask_words == mask_word_count(len(layout.slots))
    assert ring.payload_words == layout.words + ring.mask_words + 1
    assert ring.owner_pid == os.getpid()
    assert ring.verbose_until_step == -1
    assert ring.stop_requested is False


def test_all_slots_start_free(shm_rings, layout: Layout) -> None:
    ring = shm_rings(layout, slot_count=2)

    for slot_index in range(ring.slot_count):
        header = ring.header(slot_index)
        assert header.state == STATE_FREE


def test_attach_reads_the_same_layout(shm_rings, layout: Layout) -> None:
    ring = shm_rings(layout, slot_count=2)

    attached = ShmRing.attach(ring._shm.name)
    try:
        assert attached.slot_count == ring.slot_count
        assert attached.payload_words == ring.payload_words
        assert attached.mask_words == ring.mask_words
        assert attached.owner_pid == ring.owner_pid
        assert [(slot.metric.name, slot.layer, slot.microbatch, slot.width, slot.offset)
                for slot in attached.layout.slots] == [
            (slot.metric.name, slot.layer, slot.microbatch, slot.width, slot.offset) for slot in ring.layout.slots
        ]
    finally:
        attached.close()


def test_attach_raises_value_error_on_schema_mismatch(layout: Layout) -> None:
    name = f"ms_test_{uuid.uuid4().hex[:20]}"
    ring = ShmRing.create(name, layout, slot_count=2)

    raw = shared_memory.SharedMemory(name=name, create=False)
    control = np.frombuffer(raw.buf, dtype=np.int64, count=1, offset=0)
    control[0] = SCHEMA_VERSION + 1
    del control
    raw.close()

    bad_handle = shared_memory.SharedMemory(name=name, create=False)
    with pytest.raises(ValueError):
        ShmRing(bad_handle)
    bad_handle.close()

    ring.close()


def test_attach_raises_ring_not_ready_before_the_creator_publishes_the_schema_version(layout: Layout) -> None:
    name = f"ms_test_{uuid.uuid4().hex[:20]}"
    ring = ShmRing.create(name, layout, slot_count=2)
    raw = shared_memory.SharedMemory(name=name, create=False)
    control = np.frombuffer(raw.buf, dtype=np.int64, count=1, offset=0)
    control[0] = 0
    del control
    raw.close()

    with pytest.raises(RingNotReady):
        ShmRing.attach(name)

    ring.close()


def test_payload_view_is_writable_and_sized_to_payload_words(shm_rings, layout: Layout) -> None:
    ring = shm_rings(layout, slot_count=1)

    payload = ring.payload(0)
    payload[:] = np.arange(ring.payload_words, dtype=np.int32)

    assert list(ring.payload(0)) == list(range(ring.payload_words))


def test_payload_region_spans_every_slot_including_padding(shm_rings, layout: Layout) -> None:
    ring = shm_rings(layout, slot_count=3)

    region = ring.payload_region()

    expected_stride = round_up(ring.payload_words * WORD_BYTES, ALIGNMENT)
    assert region.dtype == np.uint8
    assert len(region) == ring.slot_count * expected_stride


def test_claim_free_slot_marks_writing_and_is_not_returned_again(shm_rings, layout: Layout) -> None:
    ring = shm_rings(layout, slot_count=3)

    first_round = [ring.claim_free_slot() for _ in range(3)]
    assert first_round == [0, 1, 2]
    for index in first_round:
        assert ring.header(index).state == STATE_WRITING

    assert ring.claim_free_slot() is None

    for index in first_round:
        ring.mark_ready(index, step=index, seq=index, dropped_before=0)

    ring.release(1)
    assert ring.claim_free_slot() == 1


def test_mark_ready_sets_header_and_ready_slots_sorts_by_seq(shm_rings, layout: Layout) -> None:
    ring = shm_rings(layout, slot_count=3)

    ring.mark_ready(0, step=10, seq=5, dropped_before=2)
    ring.mark_ready(1, step=11, seq=3, dropped_before=0)
    ring.mark_ready(2, step=12, seq=4, dropped_before=1)

    assert ring.ready_slots() == [1, 2, 0]
    assert ring.header(0) == SlotHeader(state=STATE_READY, step=10, seq=5, dropped_before=2)
    assert ring.header(1) == SlotHeader(state=STATE_READY, step=11, seq=3, dropped_before=0)


def test_release_frees_a_ready_slot(shm_rings, layout: Layout) -> None:
    ring = shm_rings(layout, slot_count=1)
    ring.mark_ready(0, step=1, seq=1, dropped_before=0)

    ring.release(0)

    assert ring.header(0).state == STATE_FREE
    assert ring.ready_slots() == []


def test_verbose_until_step_round_trips(shm_rings, layout: Layout) -> None:
    ring = shm_rings(layout, slot_count=1)

    ring.verbose_until_step = 42

    assert ring.verbose_until_step == 42


def test_request_stop_round_trips(shm_rings, layout: Layout) -> None:
    ring = shm_rings(layout, slot_count=1)

    assert ring.stop_requested is False
    ring.request_stop()
    assert ring.stop_requested is True


def test_close_is_idempotent(layout: Layout) -> None:
    name = f"ms_test_{uuid.uuid4().hex[:20]}"
    ring = ShmRing.create(name, layout, slot_count=1)

    ring.close()
    ring.close()


def test_unlink_removes_the_segment(layout: Layout) -> None:
    name = f"ms_test_{uuid.uuid4().hex[:20]}"
    ring = ShmRing.create(name, layout, slot_count=1)

    ring.close()
    ring.unlink()
