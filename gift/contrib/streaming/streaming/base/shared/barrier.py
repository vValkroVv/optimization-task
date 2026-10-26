# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Barrier that lives in shared memory.

Implemented with shared array and a filelock.
"""

import logging
import os
from time import monotonic, sleep
from typing import Callable

import numpy as np
from filelock import FileLock

from streaming.base.constant import TICK, TIMEOUT, WARN_EVERY, WARN_FIRST
from streaming.base.shared.array import SharedArray

logger = logging.getLogger(__name__)


def _timeout_from_env() -> float:
    value = os.environ.get('STREAMING_BARRIER_TIMEOUT')
    if value is None:
        return TIMEOUT
    try:
        return float(value)
    except ValueError:
        raise ValueError(f'STREAMING_BARRIER_TIMEOUT must be a number of seconds, got {value!r}.') from None

_TIMEOUT = _timeout_from_env()


class SharedBarrier:
    """A barrier that works inter-process using a filelock and shared memory.

    We set the number of processes (and thereby initialize num_exit) on the first time this object
    is called. This is because the object is created in a per-rank process, and called by worker
    processes.

    Args:
        filelock_path (str): Path to lock file on local filesystem.
        shm_name (str): Shared memory object name in /dev/shm.
    """

    def __init__(self, filelock_path: str, shm_name: str) -> None:
        # Create lock.
        self.filelock_path = filelock_path
        self.lock = FileLock(self.filelock_path)

        # Create three int32 fields in shared memory: num_enter, num_exit, flag.
        self._arr = SharedArray(3, np.int32, shm_name)
        self.num_enter = 0
        self.num_exit = -1
        self.flag = True

    @property
    def num_enter(self) -> int:
        """Get property num_enter.

        Returns:
            int: Number of processes that have entered the barrier.
        """
        return self._arr[0]

    @num_enter.setter
    def num_enter(self, num_enter: int) -> None:
        """Set property num_enter.

        Args:
            num_enter (int): Number of processes that have entered the barrier.
        """
        self._arr[0] = num_enter

    @property
    def num_exit(self) -> int:
        """Get property num_exit.

        Returns:
            int: Number of processes that have exited the barrier.
        """
        return self._arr[1]

    @num_exit.setter
    def num_exit(self, num_exit: int) -> None:
        """Set property num_exit.

        Args:
            num_exit (int): Number of processes that have exited the barrier.
        """
        self._arr[1] = num_exit

    @property
    def flag(self) -> bool:
        """Get property flag.

        Returns:
            bool: The flag value.
        """
        return bool(self._arr[2])

    @flag.setter
    def flag(self, flag: bool) -> None:
        """Set property flag.

        Args:
            flag (bool): The flag value.
        """
        self._arr[2] = bool(flag)

    def _diagnostics(self, num_procs: int) -> str:
        """Barrier state for logs and errors; lock-free reads are fine for diagnostics"""
        return (f'{self.num_enter} of {num_procs} processes entered, num_exit={self.num_exit}, flag={self.flag} ({self.filelock_path})')

    def _wait(self, condition: Callable[[], bool], num_procs: int) -> None:
        """Wait for the condition, warning periodically and raising after the deadline"""
        start = monotonic()
        deadline = (start + _TIMEOUT) if _TIMEOUT > 0 else None
        next_warn = start + WARN_FIRST
        while not condition():
            now = monotonic()
            if deadline is not None and now >= deadline:
                raise RuntimeError(
                    f'SharedBarrier timed out after {_TIMEOUT:.0f}s: ' +
                    f'{self._diagnostics(num_procs)}. A participant likely died or hung; ' +
                    f'adjust with STREAMING_BARRIER_TIMEOUT.')
            if now >= next_warn:
                logger.warning(f'SharedBarrier still waiting after {now - start:.0f}s: {self._diagnostics(num_procs)}')
                next_warn = now + WARN_EVERY
            sleep(TICK)

    def __call__(self, num_procs: int) -> None:
        """A set number of processes enter, wait, and exit the barrier.

        Args:
            num_procs (int): How many processes are sharing this barrier.

        Raises:
            RuntimeError: If the wait exceeds the barrier deadline (see ``STREAMING_BARRIER_TIMEOUT``)
        """
        # Initialize num_exit to the number of processes.
        with self.lock:
            if self.num_exit == -1:
                self.num_exit = num_procs

        # If we are the first to arrive, wait for everyone to exit, then set flag to "don't go".
        self.lock.acquire()
        if not self.num_enter:
            self.lock.release()
            self._wait(lambda: self.num_exit == num_procs, num_procs)
            self.lock.acquire()
            self.flag = False

        # Note that we entered.
        self.num_enter += 1

        # If we are the last to arrive, reset `enter` and `exit`, and set flag to "go".
        if self.num_enter == num_procs:
            self.num_enter = 0
            self.num_exit = 0
            self.flag = True
        self.lock.release()

        # Everybody waits until the flag is set to "go".
        self._wait(lambda: self.flag, num_procs)

        # Note that we exited.
        with self.lock:
            self.num_exit += 1
            if self.num_exit == num_procs:
                self.num_exit = -1
