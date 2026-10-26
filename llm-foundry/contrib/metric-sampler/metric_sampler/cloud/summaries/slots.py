from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import BinaryIO

from metric_sampler.cloud.summaries.clickhouse import ClickHouse
from metric_sampler.cloud.summaries.clock import Clock

STATE_DIRECTORY = Path("/tmp")
SLOTS = 8
SLOT_WAIT_SECONDS = 5
_LOCKED_BYTE_COUNT = 1


def _lock_exclusively(lock_file: BinaryIO) -> None:
    if sys.platform == "win32":
        import msvcrt

        if lock_file.seek(0, os.SEEK_END) == 0:
            lock_file.write(b"\0")
            lock_file.flush()
        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, _LOCKED_BYTE_COUNT)
    else:  # pragma: no cover -- Linux advisory lock; Windows exercises the same ownership contract.
        import fcntl

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


class InsertSlots:
    """At most `slots` chunk inserts of all the loops, and of recompute, run at once.

    Loops that are all behind, as after the collector caught up a backlog, would otherwise run together and exhaust
    the memory. A slot is an exclusive lock on a file, held for the duration of the insert.
    """

    def __init__(
        self, client: ClickHouse, clock: Clock, directory: Path = STATE_DIRECTORY, slots: int = SLOTS
    ) -> None:
        self._client = client
        self._clock = clock
        self._directory = directory
        self._slots = slots

    def insert(self, statement: str) -> None:
        """Runs the statement in the first free slot, waiting while every slot is busy."""
        while not self._insert_in_free_slot(statement):
            self._clock.sleep(SLOT_WAIT_SECONDS)

    def _insert_in_free_slot(self, statement: str) -> bool:
        for slot in range(1, self._slots + 1):
            with (self._directory / f"summaries-slot-{slot}.lock").open("a+b") as lock_file:
                try:
                    _lock_exclusively(lock_file)
                except OSError:
                    continue
                self._client.execute(statement)
                return True
        return False
