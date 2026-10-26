from __future__ import annotations

import uuid
from multiprocessing import shared_memory

import numpy as np
import pytest
import torch

from metric_sampler.capture.dump_buffer import (
    DTYPE_ITEMSIZES,
    DTYPE_NAMES,
    DTYPE_NOT_CAPTURED,
    DtypeCode,
    DumpBuffer,
    DumpEntry,
    decode_dump_entry,
)
from metric_sampler.capture.ring import STATE_FREE, STATE_READY, STATE_WRITING, RingNotReady
from metric_sampler.contract.metrics import SCHEMA_VERSION

pytestmark = pytest.mark.unit

_TOKENS = 4
_HIDDEN = 8
_ENTRIES = (DumpEntry("residual", 0), DumpEntry("residual", 1), DumpEntry("final_norm", -1))


def test_create_writes_control_fields(dump_buffers) -> None:
    buffer = dump_buffers(_ENTRIES, _TOKENS, _HIDDEN)

    assert buffer.tokens == _TOKENS
    assert buffer.hidden == _HIDDEN
    assert buffer.entries == _ENTRIES
    assert buffer.state == STATE_FREE
    assert buffer.is_ready is False
    assert list(buffer.entry_dtype_codes()) == [DTYPE_NOT_CAPTURED] * len(_ENTRIES)


def test_attach_reads_the_same_shape(dump_buffers) -> None:
    buffer = dump_buffers(_ENTRIES, _TOKENS, _HIDDEN)

    attached = DumpBuffer.attach(buffer._shm.name)
    try:
        assert attached.tokens == buffer.tokens
        assert attached.hidden == buffer.hidden
        assert attached.entries == buffer.entries
    finally:
        attached.close()


def test_attach_raises_ring_not_ready_before_schema_is_published() -> None:
    name = f"ms_test_dump_{uuid.uuid4().hex[:20]}"
    buffer = DumpBuffer.create(name, _ENTRIES, _TOKENS, _HIDDEN)
    raw = shared_memory.SharedMemory(name=name, create=False)
    control = np.frombuffer(raw.buf, dtype=np.int64, count=1, offset=0)
    control[0] = 0
    del control
    raw.close()

    with pytest.raises(RingNotReady):
        DumpBuffer.attach(name)

    buffer.close()
    buffer.unlink()


def test_init_raises_value_error_on_schema_mismatch() -> None:
    name = f"ms_test_dump_{uuid.uuid4().hex[:20]}"
    buffer = DumpBuffer.create(name, _ENTRIES, _TOKENS, _HIDDEN)
    raw = shared_memory.SharedMemory(name=name, create=False)
    control = np.frombuffer(raw.buf, dtype=np.int64, count=1, offset=0)
    control[0] = SCHEMA_VERSION + 1
    del control
    raw.close()

    bad_handle = shared_memory.SharedMemory(name=name, create=False)
    with pytest.raises(ValueError):
        DumpBuffer(bad_handle)
    bad_handle.close()

    buffer.close()
    buffer.unlink()


def test_claim_for_writing_succeeds_from_free_and_resets_dtype_codes(dump_buffers) -> None:
    buffer = dump_buffers(_ENTRIES, _TOKENS, _HIDDEN)
    buffer.entry_dtype_codes()[:] = 2

    claimed = buffer.claim_for_writing()

    assert claimed is True
    assert buffer.state == STATE_WRITING
    assert list(buffer.entry_dtype_codes()) == [DTYPE_NOT_CAPTURED] * len(_ENTRIES)


def test_claim_for_writing_fails_when_not_free(dump_buffers) -> None:
    buffer = dump_buffers(_ENTRIES, _TOKENS, _HIDDEN)
    buffer.claim_for_writing()

    assert buffer.claim_for_writing() is False


def test_mark_ready_and_release_round_trip(dump_buffers) -> None:
    buffer = dump_buffers(_ENTRIES, _TOKENS, _HIDDEN)
    buffer.claim_for_writing()

    buffer.mark_ready(step=7)

    assert buffer.state == STATE_READY
    assert buffer.step == 7
    assert buffer.is_ready is True

    buffer.release()

    assert buffer.state == STATE_FREE
    assert buffer.is_ready is False



def test_payload_views_are_writable_and_correctly_sized(dump_buffers) -> None:
    buffer = dump_buffers(_ENTRIES, _TOKENS, _HIDDEN)

    buffer.token_ids()[:] = np.arange(_TOKENS, dtype=np.int32)
    buffer.token_loss()[:] = np.arange(_TOKENS, dtype=np.float32)
    buffer.stream_ids()[:] = np.ones(_TOKENS, dtype=np.int32)

    assert list(buffer.token_ids()) == list(range(_TOKENS))
    assert list(buffer.token_loss()) == pytest.approx(list(range(_TOKENS)))
    assert list(buffer.stream_ids()) == [1] * _TOKENS

    for index in range(len(_ENTRIES)):
        region = buffer.entry_bytes(index)
        assert region.dtype == np.uint8
        assert region.size == _TOKENS * _HIDDEN * 4
        region[:] = index + 1
        assert (buffer.entry_bytes(index) == index + 1).all()


def test_payload_region_spans_every_field_a_dump_write_touches(dump_buffers) -> None:
    buffer = dump_buffers(_ENTRIES, _TOKENS, _HIDDEN)

    region = buffer.payload_region()

    expected = 3 * _TOKENS * 4 + len(_ENTRIES) * 4 + len(_ENTRIES) * _TOKENS * _HIDDEN * 4
    assert region.dtype == np.uint8
    assert len(region) == expected


def test_required_bytes_grows_with_entries_tokens_and_hidden() -> None:
    base = DumpBuffer.required_bytes(_ENTRIES, _TOKENS, _HIDDEN)

    assert DumpBuffer.required_bytes(_ENTRIES, _TOKENS * 2, _HIDDEN) > base
    assert DumpBuffer.required_bytes(_ENTRIES, _TOKENS, _HIDDEN * 2) > base
    assert DumpBuffer.required_bytes((*_ENTRIES, DumpEntry("attn_out", 0)), _TOKENS, _HIDDEN) > base


def test_close_is_idempotent(dump_buffers) -> None:
    buffer = dump_buffers(_ENTRIES, _TOKENS, _HIDDEN)

    buffer.close()
    buffer.close()


def test_unlink_removes_the_segment() -> None:
    name = f"ms_test_dump_{uuid.uuid4().hex[:20]}"
    buffer = DumpBuffer.create(name, _ENTRIES, _TOKENS, _HIDDEN)

    buffer.close()
    buffer.unlink()

    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name, create=False)


def test_dtype_codes_index_the_dtype_names_and_itemsizes() -> None:
    assert [(code.value, DTYPE_NAMES[code], DTYPE_ITEMSIZES[code]) for code in DtypeCode] == [
        (0, "bfloat16", 2), (1, "float16", 2), (2, "float32", 4)
    ]


def test_decode_dump_entry_bfloat16_roundtrips_exact_values() -> None:
    values = torch.tensor([[1.5, -2.0, 0.0, 3.25], [4.0, -0.5, 8.0, -1.0]], dtype=torch.bfloat16)
    raw = np.frombuffer(values.view(torch.int16).numpy().tobytes(), dtype=np.uint8)

    decoded = decode_dump_entry(raw, 0, tokens=2, hidden=4)

    np.testing.assert_allclose(decoded, values.float().numpy())


def test_decode_dump_entry_float16() -> None:
    values = torch.tensor([[1.5, -2.5], [3.5, 4.5]], dtype=torch.float16)
    raw = np.frombuffer(values.numpy().tobytes(), dtype=np.uint8)

    decoded = decode_dump_entry(raw, 1, tokens=2, hidden=2)

    np.testing.assert_allclose(decoded, values.float().numpy())


def test_decode_dump_entry_float32() -> None:
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    raw = np.frombuffer(values.tobytes(), dtype=np.uint8)

    decoded = decode_dump_entry(raw, 2, tokens=2, hidden=2)

    np.testing.assert_allclose(decoded, values)


def test_decode_dump_entry_raises_for_an_unknown_code() -> None:
    raw = np.zeros(16, dtype=np.uint8)

    with pytest.raises(ValueError):
        decode_dump_entry(raw, 99, tokens=2, hidden=2)
