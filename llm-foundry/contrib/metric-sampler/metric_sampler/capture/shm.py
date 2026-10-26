from __future__ import annotations

import sys
from multiprocessing import resource_tracker, shared_memory

ALIGNMENT = 64
WORD_BYTES = 4


def round_up(value: int, alignment: int) -> int:
    return -(-value // alignment) * alignment


def open_untracked(name: str) -> shared_memory.SharedMemory:
    # Resource-tracker registration would unlink the segment when this process exits, although a rank still owns it.
    # From Python 3.13 SharedMemory can skip it (`track`); the version stays a literal for mypy.
    if sys.version_info >= (3, 13):
        return shared_memory.SharedMemory(name=name, create=False, track=False)
    segment = shared_memory.SharedMemory(name=name, create=False)
    resource_tracker.unregister(segment._name, "shared_memory")  # type: ignore[attr-defined]
    return segment
