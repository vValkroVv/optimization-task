from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from metric_sampler.capture.dump_buffer import DumpBuffer
from metric_sampler.capture.layout import Layout
from metric_sampler.capture.ring import ShmRing, mask_word_count
from metric_sampler.contract.catalog import metric_by_name
from metric_sampler.contract.metrics import ValueType
from metric_sampler.node.aggregator import RankRing

_LOSS = metric_by_name("loss/total")
_COUNTS = metric_by_name("router/tokens_per_expert")
_SCORES = metric_by_name("router/scores_mean")
_BIAS = metric_by_name("router/expert_bias_before_update")


@pytest.fixture
def layout() -> Layout:
    return Layout(
        [
            (_LOSS, -1, -1, 1),
            (_COUNTS, 0, -1, 4),
            (_SCORES, 0, -1, 4),
            (_BIAS, 0, -1, 4),
        ]
    )


def rank_rings(*rings: ShmRing) -> list[RankRing]:
    """The rings of ranks 0, 1, ... in the order given."""
    return [RankRing(rank, ring) for rank, ring in enumerate(rings)]


def write_step(
    ring: ShmRing, layout: Layout, step: int, seq: int, values: dict, dropped_before: int = 0,
    trailer: int | None = None,
) -> int:
    """Writes a slot as a rank would: every key in `values` is marked recorded in the mask,
    every other slot in the layout is left unrecorded (as if that step never wrote to it)."""
    slot_index = ring.claim_free_slot()
    assert slot_index is not None
    payload = ring.payload(slot_index)
    payload[:] = 0
    mask = np.zeros(mask_word_count(len(layout.slots)), dtype=np.uint32)
    for (metric_name, layer, microbatch), recorded_values in values.items():
        slot = layout.slot(metric_name, layer, microbatch)
        words = payload[slot.offset : slot.offset + slot.width]
        if slot.metric.value_type is ValueType.FLOAT32:
            words.view(np.float32)[:] = recorded_values
        else:
            words[:] = recorded_values
        word, bit = divmod(layout.slot_index(metric_name, layer, microbatch), 32)
        mask[word] |= np.uint32(1) << np.uint32(bit)
    if len(mask):
        payload[layout.words : layout.words + len(mask)] = mask.view(np.int32)
    payload[-1] = step if trailer is None else trailer
    ring.mark_ready(slot_index, step=step, seq=seq, dropped_before=dropped_before)
    return slot_index


def full_step(
    loss: float = 1.0,
    counts: Sequence[int] = (1, 2, 3, 4),
    scores: Sequence[float] = (0.1, 0.1, 0.1, 0.1),
    bias: Sequence[float] = (0.5, 0.5, 0.5, 0.5),
) -> dict:
    """The values of every slot of the `layout` fixture."""
    return {
        ("loss/total", -1, -1): [loss],
        ("router/tokens_per_expert", 0, -1): counts,
        ("router/scores_mean", 0, -1): scores,
        ("router/expert_bias_before_update", 0, -1): bias,
    }


def named(paths: list[Path], name: str) -> Path:
    return next(path for path in paths if path.name == name)


def read_rows(paths: list[Path], name: str) -> list[dict]:
    """The rows of the Parquet file called `name` among the written `paths`."""
    return pq.read_table(named(paths, name), partitioning=None).to_pylist()


def _encode_entry(values: np.ndarray, code: int) -> np.ndarray:
    """The inverse of `decode_dump_entry`, for building a dump buffer's raw bytes in tests."""
    if code == 0:
        bits = values.astype(np.float32).view(np.uint32)
        return (bits >> 16).astype(np.uint16).view(np.uint8)
    if code == 1:
        return values.astype(np.float16).view(np.uint8)
    return values.astype(np.float32).view(np.uint8)


def fill_dump_buffer(
    buffer: DumpBuffer,
    step: int,
    token_ids: list[int],
    loss: list[float],
    stream_ids: list[int],
    entries_values: list[tuple[int, np.ndarray]],
) -> None:
    """entries_values pairs a dtype code (or -1 for "not captured") with a [tokens, hidden] array,
    aligned with `buffer.entries`; mirrors what DumpCapture would have written for one dump."""
    assert buffer.claim_for_writing() is True
    buffer.token_ids()[:] = token_ids
    buffer.token_loss()[:] = loss
    buffer.stream_ids()[:] = stream_ids
    codes = buffer.entry_dtype_codes()
    for index, (code, values) in enumerate(entries_values):
        codes[index] = code
        if code == -1:
            continue
        raw = _encode_entry(values, code).reshape(-1)
        buffer.entry_bytes(index)[: raw.size] = raw
    buffer.mark_ready(step)
