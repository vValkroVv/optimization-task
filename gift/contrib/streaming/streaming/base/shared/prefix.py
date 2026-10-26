# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Register or look up the prefix to use for all shared resources.

The prefix is used by all workers using this StreamingDataset of this training job. This is used to
prevent shared resources like shared memory from colliding.
"""

import logging
import os
from collections import Counter
from tempfile import gettempdir
from time import sleep
from typing import Iterator, Union

import numpy as np

from streaming.base.constant import (
    _NAMESPACE_SEPARATOR,
    BARRIER_FILELOCK,
    CACHE_FILELOCK,
    LOCALS,
    SHM_TO_CLEAN,
    TICK
)
from streaming.base.context import get_shared_memory_namespace
from streaming.base.distributed import all_gather_object, barrier
from streaming.base.shared import SharedMemory
from streaming.base.world import World

logger = logging.getLogger(__name__)

# Occupied slots the leader may skip over in one registration before giving up.
_MAX_OCCUPIED_SLOT_SCAN = 1000


def _each_prefix_int() -> Iterator[int]:
    """Get each possible prefix int to check in order.

    Returns:
        Iterator[int]: Each prefix int.
    """
    prefix_int = 0
    while True:
        yield prefix_int
        prefix_int += 1


def _get_path(prefix_int: int, name: str) -> str:
    """Get the name of the shared memory.

    Args:
        prefix (int): The prefix int.
        name (str): The name of the shared memory.

    Returns:
        str: Unique shared memory name.
    """
    return f'{prefix_int:06}_{name}'


def _pack_locals(dirnames: list[str], prefix_int: int) -> bytes:
    """Pack local dirnames and prefix int.

    Args:
        dirnames (List[str]): Unpacked local dirnames.
        prefix_int (int): Prefix int.

    Returns:
        bytes: Packed local dirnames and prefix int.
    """
    text = '\0'.join(dirnames) + f'\0{prefix_int}'
    data = text.encode('utf-8')
    size = 4 + len(data)
    return b''.join([np.int32(size).tobytes(), data])


def _unpack_locals(data: bytes) -> tuple[list[str], int]:
    """Unpack local dirnames and prefix int.

    Args:
        data (bytes): Packed local dirnames and prefix int.

    Returns:
        List[str]: Unpacked local dirnames and prefix int.
    """
    size = np.frombuffer(data[:4], np.int32)[0]
    text = data[4:size].decode('utf-8')
    text = text.split('\0')
    return text[:-1], int(text[-1] or 0)


def _check_self(streams_local: list[str]) -> None:
    """Check our local working directories for overlap.

    Args:
        streams_local (List[str]): Local dirs.

    Raises:
        ValueError: If there is overlap.
    """
    occurrences = Counter(streams_local)
    duplicate_local_dirs = [dirname for dirname, count in occurrences.items() if count > 1]
    if duplicate_local_dirs:
        raise ValueError(
            f'Reused local directory: {duplicate_local_dirs}. Provide a different one.')


def _namespace_locals(streams_local: list[str]) -> list[str]:
    """Apply the active shared-memory namespace to local directory identities."""
    namespace = get_shared_memory_namespace()
    if namespace is None:
        return streams_local
    return [f'{dirname}{_NAMESPACE_SEPARATOR}{namespace}' for dirname in streams_local]


def _strip_namespace(dirname: str) -> str:
    """Get the physical local dir behind a registered (possibly namespaced) identity"""
    return dirname.split(_NAMESPACE_SEPARATOR, 1)[0]


def _check_and_find(streams_local: list[str], streams_remote: list[Union[str, None]],
                    shm_name: str) -> int:
    """Find the next available prefix while checking existing local dirs for overlap.

    Local leader walks the existing shm prefixes starting from zero, verifying that there is no
    local working directory overlap when remote directories exist. When attaching to an existing
    shm fails, we have reached the end of the existing shms. We will register the next one.

    Args:
        streams_local (List[str]): Our local working directories.
        streams_remote (List[Union[str, None]]): Our remote working directories.
        shm_name (str): The shared memory file name, e.g., LOCALS, BARRIER etc.

    Returns:
        int: Next available prefix int.
    """
    prefix_int = 0

    for prefix_int in _each_prefix_int():

        name = _get_path(prefix_int, shm_name)

        # Check if any shared memory filelocks exist for the current prefix
        try:
            filelock_exists = any(
                os.path.exists(os.path.join(gettempdir(), _get_path(prefix_int, filelock_name)))
                for filelock_name in [BARRIER_FILELOCK, CACHE_FILELOCK])
            if filelock_exists:
                continue
        except PermissionError:
            continue

        # Attempt to access shared memory by name. Use prefix_int if files do not exist
        try:
            shm = SharedMemory(name, False, auto_cleanup=False)
        except PermissionError:
            continue
        except FileNotFoundError:
            break
        except ValueError as err:
            # Mid-creation segment (shm_open done, ftruncate pending): occupied, keep scanning.
            logger.debug('Attach to %s raised "%s"; treating the slot as occupied.', name, err)
            continue

        try:
            if shm_name != LOCALS:
                continue
            their_locals, _ = _unpack_locals(bytes(shm.buf))
        finally:
            shm.cleanup()

        their_dirs = [_strip_namespace(dirname) for dirname in their_locals]

        # Do not check for a conflicting local directories across existing shared memory if
        # remote directories are None. Get the next prefix.
        if any(streams_remote):
            # Get the indices of the local directories which matches with the current
            # shared memory.
            matching_index = np.where(np.isin(streams_local, their_dirs))[0]
            if matching_index.size > 0:
                for idx in matching_index:
                    # If there is a conflicting local directory for a non-None remote directory,
                    # raise an exception.
                    if streams_remote[idx] is not None:
                        raise ValueError(
                            f'Reused local directory: {streams_local} vs ' +
                            f'{their_dirs}. Provide a different one. If using ' +
                            f'a unique local directory, try deleting the local directory and ' +
                            f'call `streaming.base.util.clean_stale_shared_memory()` only once ' +
                            f'in your script to clean up the stale shared memory before ' +
                            f'instantiation of `StreamingDataset`.')
    return prefix_int


def _check_and_find_retrying(streams_local: list[str], streams_remote: list[Union[str, None]],
                             shm_name: str, retry: int) -> int:
    """Find the next available prefix while checking existing dirs for overlap.

    If an overlap is found, sleeps for a tick and then tries again, up to "retry" times. We allow
    this grace period because modifying python shared memory in a destructor intermediated through
    a numpy array appears to be racy.

    Args:
        streams_local (List[str]): Our local working directories.
        streams_remote (List[Union[str, None]]): Our remote working directories.
        shm_name (str): The shared memory file name, e.g., LOCALS, BARRIER etc.
        retry (int): Number of retries upon failure before raising an exception.

    Returns:
        int: Next available prefix int.
    """
    if retry < 0:
        raise ValueError(f'Specify at least zero retries (provided {retry}).')
    errs = []
    for _ in range(1 + retry):
        try:
            return _check_and_find(streams_local, streams_remote, shm_name)
        except ValueError as err:
            errs.append(err)
            sleep(TICK)
    raise errs[-1]


def _scan_all_names(streams_local: list[str], streams_remote: list[Union[str, None]],
                    retry: int) -> int:
    """Find the next free prefix across every shm name, checking local dirs for overlap.

    Args:
        streams_local (List[str]): Our local working directories.
        streams_remote (List[Union[str, None]]): Our remote working directories.
        retry (int): Number of retries upon failure before raising an exception.

    Returns:
        int: Next available prefix int.
    """
    return max(
        _check_and_find_retrying(streams_local, streams_remote, shm_name=shm_name, retry=retry)
        for shm_name in SHM_TO_CLEAN)


def _slot_is_free(prefix_int: int) -> bool:
    """Check that a prefix slot has no shm region of any name and no file locks.

    Segments are probed as /dev/shm files (our training and test hosts are Linux-only)
    so occupied slots cost no mappings or fds. Without /dev/shm every slot reads as
    free and registration degrades to the upstream create-and-retry behavior.

    Args:
        prefix_int (int): Prefix int.

    Returns:
        bool: Whether the slot is free.
    """
    try:
        for filelock_name in (BARRIER_FILELOCK, CACHE_FILELOCK):
            if os.path.exists(os.path.join(gettempdir(), _get_path(prefix_int, filelock_name))):
                return False
    except PermissionError:
        return False
    return not any(
        os.path.exists(os.path.join('/dev/shm', _get_path(prefix_int, shm_name)))
        for shm_name in SHM_TO_CLEAN)


def get_shm_prefix(streams_local: list[str],
                   streams_remote: list[Union[str, None]],
                   world: World,
                   retry: int = 100) -> tuple[int, SharedMemory]:
    """Register or lookup our shared memory prefix.

    Args:
        streams_local (List[str]): Local working dir of each stream, relative to /.
            We need to verify that there is no overlap with any other currently
            running StreamingDataset.
        streams_remote (List[Union[str, None]]): Remote working dir of each stream.
        world (World): Information about nodes, ranks, and workers.
        retry (int): Number of retries upon failure before raising an exception.
            Defaults to ``100``.

    Returns:
        Tuple[int, SharedMemory]: Shared memory integer prefix and object. The name
            is required to be very short due to limitations of Python on Mac OSX.
    """
    registered_locals = _namespace_locals(streams_local)

    # Check my locals for overlap.
    _check_self(streams_local)

    prefix_int = _scan_all_names(streams_local, streams_remote, retry)

    barrier()

    # First, the local leader registers the first available shm prefix, recording its locals.
    # Leaders of concurrent worlds (e.g. per-dp-group dataloaders of TP/PP replica groups)
    # race for slots: on collision, rescan with full validation and try again.
    if world.is_local_leader:
        occupied_scanned = 0
        while True:
            while not _slot_is_free(prefix_int):
                prefix_int += 1
                occupied_scanned += 1
                if occupied_scanned >= _MAX_OCCUPIED_SLOT_SCAN:
                    raise RuntimeError(
                        f'Scanned {_MAX_OCCUPIED_SLOT_SCAN} shared memory prefixes, all ' +
                        f'occupied. Run streaming.base.util.clean_stale_shared_memory() or ' +
                        f'check for stale segments in /dev/shm.')
            name = _get_path(prefix_int, LOCALS)
            data = _pack_locals(registered_locals, prefix_int)
            try:
                shm = SharedMemory(name, True, len(data))
            except FileExistsError:
                prefix_int = _scan_all_names(streams_local, streams_remote, retry)
                continue
            shm.buf[:len(data)] = data
            break

    # Publish every node-local leader's selected prefix. Group ranks are node-contiguous in World,
    # so each rank can select its own leader's value instead of guessing how far the leader moved.
    leader_prefixes = all_gather_object(prefix_int if world.is_local_leader else None)
    local_leader_rank = world.rank - world.rank_of_node
    leader_prefix_int = leader_prefixes[local_leader_rank]
    if leader_prefix_int is None:
        raise RuntimeError(f'Internal error: node-local leader rank {local_leader_rank} did not ' +
                           f'register a shared memory prefix.')
    prefix_int = leader_prefix_int

    if not world.is_local_leader:
        name = _get_path(prefix_int, LOCALS)
        try:
            shm = SharedMemory(name, False)
        except (FileNotFoundError, PermissionError, ValueError) as err:
            raise RuntimeError(f'Internal error: shared memory prefix={prefix_int} was not ' +
                               f'registered by local leader rank {local_leader_rank}.') from err

        their_locals, their_prefix_int = _unpack_locals(bytes(shm.buf))
        if registered_locals != their_locals or prefix_int != their_prefix_int:
            raise RuntimeError(f'Internal error: shared memory registered by local leader ' +
                               f'does not match. Expected locals={registered_locals}, ' +
                               f'prefix={prefix_int}; got locals={their_locals}, ' +
                               f'prefix={their_prefix_int}.')
    return prefix_int, shm  # pyright: ignore
