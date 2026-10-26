from __future__ import annotations

import json
from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property
from multiprocessing import shared_memory

import numpy as np

from metric_sampler.capture.ring import NO_STEP, STATE_FREE, STATE_READY, STATE_WRITING, RingNotReady
from metric_sampler.capture.shm import ALIGNMENT, WORD_BYTES, open_untracked, round_up
from metric_sampler.contract.metrics import SCHEMA_VERSION


class DtypeCode(IntEnum):
    """The code stored per entry in `DumpBuffer.entry_dtype_codes()`: an entry keeps the dtype the model used."""

    BFLOAT16 = 0
    FLOAT16 = 1
    FLOAT32 = 2


# Indexed by the dtype code. The node converts every entry to float32 before it writes Parquet.
DTYPE_NAMES = ("bfloat16", "float16", "float32")
DTYPE_ITEMSIZES = (2, 2, 4)
# One entry slot is reserved at this width whatever the entry's dtype, so that a run captured in float32 cannot
# overflow a buffer sized for bfloat16.
MAX_ITEMSIZE = 4
_BFLOAT16_BITS = 16
_PER_TOKEN_ARRAYS = 3

# The code of an entry the rank did not write in this dump (a layer the forward pass skipped, or a bug): the node
# leaves it out of the Parquet file instead of reading it back as an all-zero bfloat16 row. Codes start at this
# value and are reset to it on every claim.
DTYPE_NOT_CAPTURED = -1

# Per-rank cap on the shared-memory size of one buffer (see `DumpBuffer.required_bytes`), enforced by the caller
# before it creates the buffer.
MAX_BUFFER_BYTES = 1024**3


@dataclass(frozen=True)
class DumpEntry:
    """One sampled (activation point, layer) pair; `layer` is -1 for a point that is not scoped to a layer,
    such as `final_norm` or `embed`."""

    point: str
    layer: int


def _entries_json(entries: tuple[DumpEntry, ...]) -> bytes:
    return json.dumps([{"point": entry.point, "layer": entry.layer} for entry in entries]).encode("utf-8")


_CONTROL_DTYPE = np.dtype(
    [
        ("schema_version", "i8"),
        ("entry_count", "i8"),
        ("tokens", "i8"),
        ("hidden", "i8"),
        ("entries_json_bytes", "i8"),
    ]
)

_HEADER_DTYPE = np.dtype(
    {
        "names": ["state", "step"],
        "formats": ["u4", "i8"],
        "offsets": [0, 8],
        "itemsize": 16,
    }
)


def _read_schema_version(segment: shared_memory.SharedMemory) -> int:
    return int(np.frombuffer(segment.buf, dtype=_CONTROL_DTYPE, count=1, offset=0)["schema_version"][0])


def _header_offset(entries_json_bytes: int) -> int:
    return round_up(_CONTROL_DTYPE.itemsize + entries_json_bytes, ALIGNMENT)


def _payload_start(entries_json_bytes: int) -> int:
    return round_up(_header_offset(entries_json_bytes) + _HEADER_DTYPE.itemsize, ALIGNMENT)


class DumpBuffer:
    """A single-slot shared-memory buffer carrying one rank's activation dump to the node process.

    Follows `ShmRing`'s create, attach, close and unlink protocol and its slot states, for one dump at a time:
    the (point, layer) entries, the token count and the hidden size are fixed for the buffer's lifetime, only
    the payload changes between dumps.
    """

    def __init__(self, segment: shared_memory.SharedMemory) -> None:
        self._shm = segment
        self._closed = False
        schema_version = _read_schema_version(segment)
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"dump buffer schema version {schema_version} does not match the running schema "
                f"version {SCHEMA_VERSION}"
            )
        self._control = np.frombuffer(segment.buf, dtype=_CONTROL_DTYPE, count=1, offset=0)
        self._entry_count = int(self._control["entry_count"][0])
        self._tokens = int(self._control["tokens"][0])
        self._hidden = int(self._control["hidden"][0])
        entries_json_bytes = int(self._control["entries_json_bytes"][0])

        header_offset = _header_offset(entries_json_bytes)
        self._header = np.frombuffer(segment.buf, dtype=_HEADER_DTYPE, count=1, offset=header_offset)
        payload_start = _payload_start(entries_json_bytes)

        offset = payload_start
        self._token_ids = np.frombuffer(segment.buf, dtype=np.int32, count=self._tokens, offset=offset)
        offset += self._tokens * WORD_BYTES
        self._token_loss = np.frombuffer(segment.buf, dtype=np.float32, count=self._tokens, offset=offset)
        offset += self._tokens * WORD_BYTES
        self._stream_ids = np.frombuffer(segment.buf, dtype=np.int32, count=self._tokens, offset=offset)
        offset += self._tokens * WORD_BYTES
        self._entry_dtype_codes = np.frombuffer(segment.buf, dtype=np.int32, count=self._entry_count, offset=offset)
        offset += self._entry_count * WORD_BYTES
        self._entry_stride = self._tokens * self._hidden * MAX_ITEMSIZE
        self._entries_start = offset
        self._payload_start = payload_start

    @staticmethod
    def required_bytes(entries: tuple[DumpEntry, ...], tokens: int, hidden: int) -> int:
        """Total shared-memory size a buffer for this shape needs; used both by `create` and by
        the caller's check against `MAX_BUFFER_BYTES`."""
        entries_json_bytes = len(_entries_json(entries))
        payload_bytes = (
            _PER_TOKEN_ARRAYS * tokens * WORD_BYTES
            + len(entries) * WORD_BYTES
            + len(entries) * tokens * hidden * MAX_ITEMSIZE
        )
        return _payload_start(entries_json_bytes) + round_up(payload_bytes, ALIGNMENT)

    @classmethod
    def create(cls, name: str, entries: tuple[DumpEntry, ...], tokens: int, hidden: int) -> DumpBuffer:
        entries_json = _entries_json(entries)
        total_size = cls.required_bytes(entries, tokens, hidden)

        segment = shared_memory.SharedMemory(name=name, create=True, size=total_size)
        control_bytes = _CONTROL_DTYPE.itemsize
        control = np.frombuffer(segment.buf, dtype=_CONTROL_DTYPE, count=1, offset=0)
        control["entry_count"][0] = len(entries)
        control["tokens"][0] = tokens
        control["hidden"][0] = hidden
        control["entries_json_bytes"][0] = len(entries_json)
        segment.buf[control_bytes : control_bytes + len(entries_json)] = entries_json

        header = np.frombuffer(segment.buf, dtype=_HEADER_DTYPE, count=1, offset=_header_offset(len(entries_json)))
        header["state"][0] = STATE_FREE
        header["step"][0] = NO_STEP

        # Written last, as in `ShmRing.create`: an attaching reader takes a zero version for "not published yet".
        control["schema_version"][0] = SCHEMA_VERSION
        del control, header

        buffer = cls(segment)
        buffer._entry_dtype_codes[:] = DTYPE_NOT_CAPTURED
        return buffer

    @classmethod
    def attach(cls, name: str) -> DumpBuffer:
        segment = open_untracked(name)
        if _read_schema_version(segment) == 0:
            segment.close()
            raise RingNotReady(name)
        return cls(segment)

    @property
    def tokens(self) -> int:
        return self._tokens

    @property
    def hidden(self) -> int:
        return self._hidden

    @cached_property
    def entries(self) -> tuple[DumpEntry, ...]:
        control_bytes = _CONTROL_DTYPE.itemsize
        entries_json_bytes = int(self._control["entries_json_bytes"][0])
        text = bytes(self._shm.buf[control_bytes : control_bytes + entries_json_bytes]).decode("utf-8")
        return tuple(DumpEntry(point=entry["point"], layer=entry["layer"]) for entry in json.loads(text))

    @property
    def state(self) -> int:
        return int(self._header["state"][0])

    @property
    def is_ready(self) -> bool:
        return self.state == STATE_READY

    @property
    def step(self) -> int:
        return int(self._header["step"][0])

    def claim_for_writing(self) -> bool:
        """Moves the buffer from FREE to WRITING and resets the dtype codes of the dump; returns whether it
        was FREE."""
        if int(self._header["state"][0]) != STATE_FREE:
            return False
        self._entry_dtype_codes[:] = DTYPE_NOT_CAPTURED
        self._header["state"][0] = STATE_WRITING
        return True

    def mark_ready(self, step: int) -> None:
        self._header["step"][0] = step
        self._header["state"][0] = STATE_READY

    def release(self) -> None:
        self._header["state"][0] = STATE_FREE

    def token_ids(self) -> np.ndarray:
        return self._token_ids

    def token_loss(self) -> np.ndarray:
        return self._token_loss

    def stream_ids(self) -> np.ndarray:
        return self._stream_ids

    def entry_dtype_codes(self) -> np.ndarray:
        return self._entry_dtype_codes

    def entry_bytes(self, entry_index: int) -> np.ndarray:
        """The raw byte region reserved for one entry. Only its first `tokens * hidden * itemsize` bytes are
        meaningful, for the dtype that `entry_dtype_codes()[entry_index]` names."""
        start = self._entries_start + entry_index * self._entry_stride
        return np.frombuffer(self._shm.buf, dtype=np.uint8, count=self._entry_stride, offset=start)

    def payload_region(self) -> np.ndarray:
        """A byte view spanning every field a dump write touches, for `cudaHostRegister`."""
        total = self._entries_start + self._entry_count * self._entry_stride - self._payload_start
        return np.frombuffer(self._shm.buf, dtype=np.uint8, count=total, offset=self._payload_start)

    def close(self) -> None:
        if self._closed:
            return
        del self._control
        del self._header
        del self._token_ids
        del self._token_loss
        del self._stream_ids
        del self._entry_dtype_codes
        self._shm.close()
        self._closed = True

    def unlink(self) -> None:
        self._shm.unlink()


def decode_dump_entry(raw: np.ndarray, dtype_code: int, tokens: int, hidden: int) -> np.ndarray:
    """Raw little-endian bytes for one entry, converted to a `[tokens, hidden]` float32 array."""
    if dtype_code == DtypeCode.BFLOAT16:
        # numpy has no bfloat16: its 16 bits are the high half of the float32 with the same value.
        upper = raw.view(np.uint16).astype(np.uint32) << _BFLOAT16_BITS
        values = upper.view(np.float32)
    elif dtype_code == DtypeCode.FLOAT16:
        values = raw.view(np.float16).astype(np.float32)
    elif dtype_code == DtypeCode.FLOAT32:
        values = raw.view(np.float32)
    else:
        raise ValueError(f"unknown activation dump dtype code {dtype_code}")
    return values.reshape(tokens, hidden)
