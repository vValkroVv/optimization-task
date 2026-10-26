from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property
from multiprocessing import shared_memory

import numpy as np

from metric_sampler.capture.layout import Layout
from metric_sampler.capture.shm import ALIGNMENT, WORD_BYTES, open_untracked, round_up
from metric_sampler.contract.metrics import SCHEMA_VERSION

STATE_FREE = 0
STATE_READY = 1
STATE_WRITING = 2

_MASK_WORD_BITS = 32
NO_STEP = -1

_CONTROL_DTYPE = np.dtype(
    [
        ("schema_version", "i8"),
        ("slot_count", "i8"),
        ("payload_words", "i8"),
        ("layout_json_bytes", "i8"),
        ("verbose_until_step", "i8"),
        ("stop", "i8"),
        ("owner_pid", "i8"),
        ("mask_words", "i8"),
    ]
)

_HEADER_DTYPE = np.dtype(
    {
        "names": ["state", "step", "seq", "dropped_before"],
        "formats": ["u4", "i8", "i8", "i8"],
        "offsets": [0, 8, 16, 24],
        "itemsize": 32,
    }
)


def _read_schema_version(segment: shared_memory.SharedMemory) -> int:
    # No lingering view on the buffer, so the segment can still be closed if the version is bad.
    return int(np.frombuffer(segment.buf, dtype=_CONTROL_DTYPE, count=1, offset=0)["schema_version"][0])


def mask_word_count(slot_count: int) -> int:
    """Number of int32 words needed to hold one recorded-bit per layout slot."""
    return -(-slot_count // _MASK_WORD_BITS)


def _header_offset(layout_json_bytes: int) -> int:
    return round_up(_CONTROL_DTYPE.itemsize + layout_json_bytes, ALIGNMENT)


def _payload_offset(layout_json_bytes: int, slot_count: int) -> int:
    return round_up(_header_offset(layout_json_bytes) + slot_count * _HEADER_DTYPE.itemsize, ALIGNMENT)


class RingNotReady(Exception):
    """The segment exists but its creator has not published the control block yet."""


@dataclass(frozen=True)
class SlotHeader:
    state: int
    step: int
    seq: int
    dropped_before: int


class ShmRing:
    """A fixed-layout ring of payload slots in a `SharedMemory` segment.

    One ring per training rank: the rank is the sole writer of payloads and slot headers,
    the node aggregator is the sole reader and releaser.
    """

    def __init__(self, segment: shared_memory.SharedMemory) -> None:
        self._shm = segment
        self._cursor = 0
        self._closed = False
        schema_version = _read_schema_version(segment)
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"ring schema version {schema_version} does not match the running schema version {SCHEMA_VERSION}"
            )
        self._control = np.frombuffer(segment.buf, dtype=_CONTROL_DTYPE, count=1, offset=0)

        self._slot_count = int(self._control["slot_count"][0])
        self._payload_words = int(self._control["payload_words"][0])
        self._mask_words = int(self._control["mask_words"][0])
        layout_json_bytes = int(self._control["layout_json_bytes"][0])

        header_offset = _header_offset(layout_json_bytes)
        self._headers = np.frombuffer(
            segment.buf, dtype=_HEADER_DTYPE, count=self._slot_count, offset=header_offset
        )
        self._payload_slot_bytes = round_up(self._payload_words * WORD_BYTES, ALIGNMENT)
        self._payload_start = _payload_offset(layout_json_bytes, self._slot_count)

    @classmethod
    def create(cls, name: str, layout: Layout, slot_count: int) -> ShmRing:
        layout_json = layout.to_json().encode("utf-8")
        # A payload is the metric words, one recorded-bit per slot (mask_words int32 words) and a trailing
        # step-number word, written just before the device-to-host copy so that a read in the middle of the copy
        # is detectable.
        mask_words = mask_word_count(len(layout.slots))
        payload_words = layout.words + mask_words + 1
        total_size = _payload_offset(len(layout_json), slot_count) + slot_count * round_up(
            payload_words * WORD_BYTES, ALIGNMENT
        )

        segment = shared_memory.SharedMemory(name=name, create=True, size=total_size)
        control_bytes = _CONTROL_DTYPE.itemsize
        control = np.frombuffer(segment.buf, dtype=_CONTROL_DTYPE, count=1, offset=0)
        control["slot_count"][0] = slot_count
        control["payload_words"][0] = payload_words
        control["layout_json_bytes"][0] = len(layout_json)
        control["verbose_until_step"][0] = NO_STEP
        control["stop"][0] = 0
        control["owner_pid"][0] = os.getpid()
        control["mask_words"][0] = mask_words
        segment.buf[control_bytes : control_bytes + len(layout_json)] = layout_json

        headers = np.frombuffer(
            segment.buf, dtype=_HEADER_DTYPE, count=slot_count, offset=_header_offset(len(layout_json))
        )
        headers["state"][:] = STATE_FREE
        headers["step"][:] = NO_STEP
        headers["seq"][:] = NO_STEP
        headers["dropped_before"][:] = 0
        # Written last: the segment is zero-filled from creation, so a non-zero version tells an
        # attaching reader the control block, layout and headers are all complete.
        control["schema_version"][0] = SCHEMA_VERSION
        del control, headers

        return cls(segment)

    @classmethod
    def attach(cls, name: str) -> ShmRing:
        segment = open_untracked(name)
        if _read_schema_version(segment) == 0:
            segment.close()
            raise RingNotReady(name)
        return cls(segment)

    @property
    def slot_count(self) -> int:
        return self._slot_count

    @property
    def used_slots(self) -> int:
        return int(np.count_nonzero(self._headers["state"] != STATE_FREE))

    @property
    def payload_words(self) -> int:
        return self._payload_words

    @property
    def mask_words(self) -> int:
        return self._mask_words

    @property
    def owner_pid(self) -> int:
        return int(self._control["owner_pid"][0])

    @cached_property
    def layout(self) -> Layout:
        control_bytes = _CONTROL_DTYPE.itemsize
        layout_json_bytes = int(self._control["layout_json_bytes"][0])
        text = bytes(self._shm.buf[control_bytes : control_bytes + layout_json_bytes]).decode("utf-8")
        return Layout.from_json(text)

    def payload(self, slot_index: int) -> np.ndarray:
        start = self._payload_start + slot_index * self._payload_slot_bytes
        return np.frombuffer(self._shm.buf, dtype=np.int32, count=self._payload_words, offset=start)

    def payload_region(self) -> np.ndarray:
        """A byte view spanning every slot's payload area, including its alignment padding."""
        total_bytes = self._slot_count * self._payload_slot_bytes
        return np.frombuffer(self._shm.buf, dtype=np.uint8, count=total_bytes, offset=self._payload_start)

    def claim_free_slot(self) -> int | None:
        """Finds a FREE slot and marks it WRITING so no other claim can pick it up mid-copy."""
        for _ in range(self._slot_count):
            index = self._cursor
            self._cursor = (self._cursor + 1) % self._slot_count
            if int(self._headers["state"][index]) == STATE_FREE:
                self._headers["state"][index] = STATE_WRITING
                return index
        return None

    def mark_ready(self, slot_index: int, step: int, seq: int, dropped_before: int) -> None:
        self._headers["step"][slot_index] = step
        self._headers["seq"][slot_index] = seq
        self._headers["dropped_before"][slot_index] = dropped_before
        self._headers["state"][slot_index] = STATE_READY

    def ready_slots(self) -> list[int]:
        ready = [i for i in range(self._slot_count) if int(self._headers["state"][i]) == STATE_READY]
        return sorted(ready, key=lambda i: int(self._headers["seq"][i]))

    def release(self, slot_index: int) -> None:
        self._headers["state"][slot_index] = STATE_FREE

    def header(self, slot_index: int) -> SlotHeader:
        return SlotHeader(
            state=int(self._headers["state"][slot_index]),
            step=int(self._headers["step"][slot_index]),
            seq=int(self._headers["seq"][slot_index]),
            dropped_before=int(self._headers["dropped_before"][slot_index]),
        )

    @property
    def verbose_until_step(self) -> int:
        return int(self._control["verbose_until_step"][0])

    @verbose_until_step.setter
    def verbose_until_step(self, step: int) -> None:
        self._control["verbose_until_step"][0] = step

    def request_stop(self) -> None:
        self._control["stop"][0] = 1

    @property
    def stop_requested(self) -> bool:
        return bool(self._control["stop"][0])

    def close(self) -> None:
        if self._closed:
            return
        del self._control
        del self._headers
        self._shm.close()
        self._closed = True

    def unlink(self) -> None:
        self._shm.unlink()
