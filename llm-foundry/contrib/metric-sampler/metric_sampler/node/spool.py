from __future__ import annotations

import json
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pyarrow as pa

from metric_sampler.capture.layout import Layout
from metric_sampler.capture.ring import mask_word_count
from metric_sampler.capture.shm import WORD_BYTES
from metric_sampler.contract.raw_layout import PUBLISH_INTENT_NAME, SPOOL_DIR_NAME, write_atomically
from metric_sampler.node.rank_slots import recorded_slot_indices, recorded_words

_COMPRESSION = "zstd"
_COMPRESSION_LEVEL = 1

_METADATA_NAME = "metadata.json"
_COMMITTED_NAME = "committed.json"
_SLOT_FILE_GLOB = "step=*-rank=*.zst"
# One spooled slot starts with: the words of its recorded prefix, the steps its rank dropped before it, and whether
# its step is verbose.
_SLOT_HEADER = struct.Struct("<QQ?")
# A spool written before capture intervals existed has none: it captured every step.
_LEGACY_CAPTURE_INTERVAL = 1


class SpooledSlot(NamedTuple):
    step: int
    rank: int
    dropped_before: int
    verbose: bool
    words: np.ndarray
    mask: np.ndarray


class Spool:
    """The durable journal of a node writer, kept in `<node directory>/_spool`: its configuration, the slots of the
    steps that no window has published yet, the intent of the window being published and the steps already
    published."""

    def __init__(self, node_directory: Path, rank_layouts: dict[int, Layout], metadata: dict[str, Any]) -> None:
        self._directory = node_directory / SPOOL_DIR_NAME
        self._rank_layouts = rank_layouts
        self._codec = pa.Codec(_COMPRESSION, compression_level=_COMPRESSION_LEVEL)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._check_metadata(node_directory, metadata)
        self._committed_steps: set[int] = set()
        committed_path = self._directory / _COMMITTED_NAME
        if committed_path.exists():
            self._committed_steps.update(json.loads(committed_path.read_text()))

    @staticmethod
    def read_metadata(node_directory: Path) -> dict[str, Any]:
        stored: dict[str, Any] = json.loads((node_directory / SPOOL_DIR_NAME / _METADATA_NAME).read_text())
        return {"capture_interval": _LEGACY_CAPTURE_INTERVAL} | stored

    def _check_metadata(self, node_directory: Path, metadata: dict[str, Any]) -> None:
        metadata_path = self._directory / _METADATA_NAME
        if not metadata_path.exists():
            write_atomically(metadata_path, json.dumps(metadata).encode())
        elif self.read_metadata(node_directory) != metadata:
            raise ValueError("node spool configuration changed within an attempt")

    @property
    def committed_steps(self) -> set[int]:
        return self._committed_steps

    def commit(self, steps: list[int]) -> None:
        committed = self._committed_steps | set(steps)
        write_atomically(self._directory / _COMMITTED_NAME, json.dumps(sorted(committed)).encode())
        self._committed_steps = committed

    def has_intent(self) -> bool:
        return (self._directory / PUBLISH_INTENT_NAME).exists()

    def read_intent(self) -> bytes:
        return (self._directory / PUBLISH_INTENT_NAME).read_bytes()

    def write_intent(self, content: bytes) -> None:
        write_atomically(self._directory / PUBLISH_INTENT_NAME, content)

    def clear_intent(self) -> None:
        (self._directory / PUBLISH_INTENT_NAME).unlink()

    def persist_slot(
        self, rank: int, step: int, dropped_before: int, verbose: bool, layout: Layout, raw: np.ndarray
    ) -> None:
        mask_words = mask_word_count(len(layout.slots))
        mask = raw[layout.words : layout.words + mask_words]
        words = recorded_words(layout, recorded_slot_indices(layout, mask))
        # The unused detailed tail can be much larger than an ordinary step's entire payload.
        payload = np.concatenate((raw[:words], mask))
        compressed = self._codec.compress(payload).to_pybytes()
        header = _SLOT_HEADER.pack(words, dropped_before, verbose)
        write_atomically(self._slot_path(step, rank), header + compressed)

    def load_slots(self) -> Iterator[SpooledSlot]:
        """The spooled slots of the steps no window has published, removing the files of published ones."""
        for path in sorted(self._directory.glob(_SLOT_FILE_GLOB)):
            step_text, rank_text = path.stem.split("-rank=")
            step, rank = int(step_text.removeprefix("step=")), int(rank_text)
            if step in self._committed_steps:
                path.unlink()
                continue
            layout = self._rank_layouts[rank]
            data = path.read_bytes()
            words, dropped_before, verbose = _SLOT_HEADER.unpack_from(data)
            if words > layout.words:
                raise ValueError(f"invalid payload length in {path}")
            mask_words = mask_word_count(len(layout.slots))
            decoded_bytes = self._codec.decompress(data[_SLOT_HEADER.size :], (words + mask_words) * WORD_BYTES)
            raw = np.frombuffer(decoded_bytes, dtype=np.int32)
            yield SpooledSlot(step, rank, dropped_before, verbose, raw[:words], raw[words:].view(np.uint32))

    def discard_slots(self, steps: list[int]) -> None:
        for step in steps:
            for rank in self._rank_layouts:
                self._slot_path(step, rank).unlink(missing_ok=True)

    def _slot_path(self, step: int, rank: int) -> Path:
        return self._directory / f"step={step:08d}-rank={rank:05d}.zst"
