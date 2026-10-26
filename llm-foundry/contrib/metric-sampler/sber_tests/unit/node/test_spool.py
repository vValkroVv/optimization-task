from __future__ import annotations

import json
import struct
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest

from metric_sampler.capture.layout import Layout
from metric_sampler.capture.ring import mask_word_count
from metric_sampler.contract.catalog import metric_by_name
from metric_sampler.node.spool import Spool, SpooledSlot

pytestmark = pytest.mark.unit

_LOSS = metric_by_name("loss/total")
_COUNTS = metric_by_name("router/tokens_per_expert")
_DETAILED_HISTOGRAM = metric_by_name("expert_loss/hist")
_METADATA = {"node": "n0", "attempt": "a1", "window_steps": 10, "capture_interval": 4}


@pytest.fixture
def layout() -> Layout:
    return Layout([(_LOSS, -1, -1, 1), (_COUNTS, 0, -1, 4), (_DETAILED_HISTOGRAM, 0, 0, 6)])


@pytest.fixture
def node_directory(tmp_path: Path) -> Path:
    return tmp_path / "attempt=a1" / "node=n0"


def _payload(layout: Layout, recorded_slots: list[int], values: list[int]) -> np.ndarray:
    """A ring payload of `layout`: the words of `values` from the start, the mask of `recorded_slots`, the step word."""
    mask = np.zeros(mask_word_count(len(layout.slots)), dtype=np.uint32)
    for index in recorded_slots:
        mask[index // 32] |= np.uint32(1) << np.uint32(index % 32)
    payload = np.zeros(layout.words + len(mask) + 1, dtype=np.int32)
    payload[: len(values)] = values
    payload[layout.words : layout.words + len(mask)] = mask.view(np.int32)
    return payload


def _slot_names(node_directory: Path) -> list[str]:
    return sorted(path.name for path in (node_directory / "_spool").glob("*.zst"))


def test_a_new_spool_keeps_its_configuration_and_a_spool_with_the_same_one_opens_it(node_directory, layout) -> None:
    Spool(node_directory, {0: layout}, _METADATA)
    stored = (node_directory / "_spool" / "metadata.json").read_bytes()

    Spool(node_directory, {0: layout}, _METADATA)

    assert Spool.read_metadata(node_directory) == _METADATA
    assert (node_directory / "_spool" / "metadata.json").read_bytes() == stored


def test_a_spool_refuses_a_configuration_that_differs_from_the_one_it_was_started_with(node_directory, layout) -> None:
    Spool(node_directory, {0: layout}, _METADATA)

    with pytest.raises(ValueError, match="configuration changed"):
        Spool(node_directory, {0: layout}, _METADATA | {"window_steps": 5})


@pytest.mark.parametrize(
    ("reopened_interval", "outcome"),
    [(1, nullcontext()), (4, pytest.raises(ValueError, match="configuration changed"))],
    ids=["captures every step", "captures every 4th step"],
)
def test_a_spool_started_before_capture_intervals_captured_every_step(
    node_directory, layout, reopened_interval, outcome
) -> None:
    without_interval = {key: value for key, value in _METADATA.items() if key != "capture_interval"}
    Spool(node_directory, {0: layout}, without_interval)
    assert Spool.read_metadata(node_directory) == without_interval | {"capture_interval": 1}

    with outcome:
        Spool(node_directory, {0: layout}, without_interval | {"capture_interval": reopened_interval})


def test_committed_steps_are_kept_across_spools_and_grow_by_every_commit(node_directory, layout) -> None:
    spool = Spool(node_directory, {0: layout}, _METADATA)
    assert spool.committed_steps == set()

    spool.commit([3, 1])
    spool.commit([2, 3])

    assert spool.committed_steps == {1, 2, 3}
    assert json.loads((node_directory / "_spool" / "committed.json").read_text()) == [1, 2, 3]
    assert Spool(node_directory, {0: layout}, _METADATA).committed_steps == {1, 2, 3}


def test_a_commit_that_cannot_be_written_commits_nothing(node_directory, layout) -> None:
    spool = Spool(node_directory, {0: layout}, _METADATA)
    spool.commit([1])
    committed = node_directory / "_spool" / "committed.json"
    committed.unlink()
    committed.mkdir()

    with pytest.raises(OSError):
        spool.commit([2])

    assert spool.committed_steps == {1}


def test_the_publish_intent_is_written_read_and_cleared(node_directory, layout) -> None:
    spool = Spool(node_directory, {0: layout}, _METADATA)
    assert not spool.has_intent()

    spool.write_intent(b'{"steps": [1]}')
    assert spool.has_intent()
    assert spool.read_intent() == b'{"steps": [1]}'
    assert Spool(node_directory, {0: layout}, _METADATA).has_intent()

    spool.clear_intent()
    assert not spool.has_intent()


@pytest.mark.parametrize("verbose", [False, True])
def test_a_persisted_slot_comes_back_with_its_recorded_words_only(node_directory, layout, verbose) -> None:
    spool = Spool(node_directory, {0: layout, 1: layout}, _METADATA)
    payload = _payload(layout, recorded_slots=[0, 1], values=[7, 1, 2, 3, 4])

    spool.persist_slot(rank=1, step=75, dropped_before=3, verbose=verbose, layout=layout, raw=payload)

    assert _slot_names(node_directory) == ["step=00000075-rank=00001.zst"]
    (slot,) = Spool(node_directory, {0: layout, 1: layout}, _METADATA).load_slots()
    assert slot == SpooledSlot(75, 1, 3, verbose, slot.words, slot.mask)
    assert slot.words.tolist() == [7, 1, 2, 3, 4]
    assert slot.mask.dtype == np.uint32
    assert slot.mask.tolist() == [0b11]


def test_loading_slots_removes_the_files_of_committed_steps_and_returns_the_others_in_order(
    node_directory, layout,
) -> None:
    spool = Spool(node_directory, {0: layout, 1: layout}, _METADATA)
    for step, rank in ((9, 1), (8, 0), (9, 0), (7, 1)):
        spool.persist_slot(rank, step, 0, False, layout, _payload(layout, [0], [step * 10 + rank]))
    spool.commit([8])

    restarted = Spool(node_directory, {0: layout, 1: layout}, _METADATA)
    slots = list(restarted.load_slots())

    assert [(slot.step, slot.rank, slot.words.tolist()) for slot in slots] == [(7, 1, [71]), (9, 0, [90]), (9, 1, [91])]
    assert _slot_names(node_directory) == [
        "step=00000007-rank=00001.zst", "step=00000009-rank=00000.zst", "step=00000009-rank=00001.zst",
    ]


def test_a_spooled_slot_longer_than_its_layout_is_corrupt(node_directory, layout) -> None:
    spool = Spool(node_directory, {0: layout}, _METADATA)
    spool.persist_slot(0, 1, 0, False, layout, _payload(layout, [0], [1]))
    path = node_directory / "_spool" / "step=00000001-rank=00000.zst"
    data = path.read_bytes()
    path.write_bytes(struct.pack("<Q", layout.words + 1) + data[8:])

    with pytest.raises(ValueError, match="invalid payload length"):
        list(spool.load_slots())


def test_discarding_steps_removes_their_slots_of_every_rank(node_directory, layout) -> None:
    spool = Spool(node_directory, {0: layout, 1: layout}, _METADATA)
    for step in (1, 2, 3):
        spool.persist_slot(0, step, 0, False, layout, _payload(layout, [0], [step]))
    spool.persist_slot(1, 2, 0, False, layout, _payload(layout, [0], [2]))

    spool.discard_slots([2, 3, 4])

    assert _slot_names(node_directory) == ["step=00000001-rank=00000.zst"]
