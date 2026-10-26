"""
Tests for shared-memory prefix coordination: leader publication, slot races,
process groups, and namespaces
"""

from contextlib import ExitStack
from itertools import chain, repeat
from pathlib import Path
from typing import Optional
from unittest.mock import patch
from uuid import uuid4

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from streaming.base import distributed as streaming_dist
from streaming.base.constant import LOCALS, SHM_TO_CLEAN
from streaming.base.context import use_streaming_world
from streaming.base.shared.memory import SharedMemory
from streaming.base.shared.prefix import (_NAMESPACE_SEPARATOR, _get_path, _pack_locals,
                                          _strip_namespace, _unpack_locals, get_shm_prefix)
from streaming.base.util import clean_stale_shared_memory
from streaming.base.world import World


@pytest.fixture(autouse=True)
def _clean_shm():
    clean_stale_shared_memory()
    yield
    clean_stale_shared_memory()


def _register_locals(prefix_int: int, streams_local: list[str]) -> SharedMemory:
    data = _pack_locals(streams_local, prefix_int)
    shm = SharedMemory(_get_path(prefix_int, LOCALS), True, len(data), auto_cleanup=False)
    shm.buf[:len(data)] = data
    return shm


@pytest.mark.parametrize(
    ('num_nodes', 'ranks_per_node', 'worker', 'leader_rank', 'leader_offset'),
    [
        (1, 2, 1, 0, 64),
        (2, 2, 3, 2, 128),
    ],
)
def test_follower_attaches_to_exact_node_leader_prefix(num_nodes: int, ranks_per_node: int,
                                                       worker: int, leader_rank: int,
                                                       leader_offset: int):
    """Followers must not guess how far their node-local leader moved during a race."""
    scanned_prefix = uuid4().int % 1_000_000_000
    leader_prefix = scanned_prefix + leader_offset
    streams_local = [f'local-node-{worker // ranks_per_node}']
    leader_shm = _register_locals(leader_prefix, streams_local)
    follower_shm = None
    world = World(num_nodes, ranks_per_node, workers_per_rank=1, worker=worker)
    gathered_prefixes = [None] * world.num_ranks
    gathered_prefixes[leader_rank] = leader_prefix

    try:
        with patch('streaming.base.shared.prefix._check_and_find_retrying',
                   return_value=scanned_prefix), \
             patch('streaming.base.shared.prefix.barrier'), \
             patch('streaming.base.shared.prefix.all_gather_object',
                   return_value=gathered_prefixes) as gather:
            prefix_int, follower_shm = get_shm_prefix(streams_local, [None], world, retry=0)

        assert prefix_int == leader_prefix
        gather.assert_called_once_with(None)
    finally:
        if follower_shm is not None:
            follower_shm.cleanup()
        leader_shm.cleanup()


def test_leader_rescans_after_losing_slot_race():
    """A leader that loses the segment-create race must rescan and take a free slot."""
    scanned_prefix = uuid4().int % 1_000_000_000
    rescanned_prefix = scanned_prefix + 64
    winner_shm = _register_locals(scanned_prefix, ['winner-local'])
    world = World(1, 1, workers_per_rank=1, worker=0)
    # one full scan before the race, one full rescan after losing it
    scan_results = chain(repeat(scanned_prefix, len(SHM_TO_CLEAN)),
                         repeat(rescanned_prefix, len(SHM_TO_CLEAN)))
    loser_shm = None

    try:
        with patch('streaming.base.shared.prefix._check_and_find_retrying',
                   side_effect=lambda *args, **kwargs: next(scan_results)), \
             patch('streaming.base.shared.prefix._slot_is_free', return_value=True):
            prefix_int, loser_shm = get_shm_prefix(['loser-local'], [None], world, retry=0)

        assert prefix_int == rescanned_prefix
        their_locals, _ = _unpack_locals(bytes(loser_shm.buf))
        assert their_locals == ['loser-local']
    finally:
        if loser_shm is not None:
            loser_shm.cleanup()
        winner_shm.cleanup()


def _run_prefix_process_groups(global_rank: int, world_size: int, store_path: str,
                               scanned_prefix: int) -> None:
    dist.init_process_group('gloo',
                            init_method=f'file://{store_path}',
                            rank=global_rank,
                            world_size=world_size)
    try:
        groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
        group_id = global_rank % 2
        group = groups[group_id]
        group_rank = dist.get_rank(group)
        leader_prefix = scanned_prefix + 64 * (group_id + 1)
        namespace = f'prefix-test-{scanned_prefix}-{group_id}'

        with use_streaming_world(rank=group_rank,
                                 world_size=2,
                                 local_rank=group_rank,
                                 local_world_size=2,
                                 init_dist=False,
                                 shared_memory_namespace=namespace,
                                 process_group=group), \
             patch('streaming.base.shared.prefix._check_and_find_retrying',
                   return_value=scanned_prefix), \
             patch('streaming.base.shared.prefix._slot_is_free',
                   side_effect=lambda prefix_int: prefix_int >= leader_prefix):
            prefix_int, shm = get_shm_prefix([f'local-group-{group_id}'], [None], World.detect(),
                                             retry=0)

        assert prefix_int == leader_prefix
        if group_rank != 0:
            shm.cleanup()
        dist.barrier(group=group)
        if group_rank == 0:
            shm.cleanup()
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_prefix_collective_uses_configured_process_group(tmp_path: Path):
    """Concurrent PP-like groups must exchange prefixes only with their own ranks."""
    store_path = str(tmp_path / 'store')
    scanned_prefix = uuid4().int % 1_000_000_000
    mp.spawn(_run_prefix_process_groups,
             args=(4, store_path, scanned_prefix),
             nprocs=4,
             join=True)


def test_single_rank_streaming_world_skips_global_collective():
    """PP with one data rank must not enter the larger default process group."""
    with use_streaming_world(rank=0, world_size=1, init_dist=False), \
         patch.object(streaming_dist.dist, 'is_initialized', return_value=True), \
         patch.object(streaming_dist.dist, 'all_gather_object') as gather:
        assert streaming_dist.all_gather_object(7) == [7]

    gather.assert_not_called()


def _register(local: str, remote: Optional[str], namespace: Optional[str], stack: ExitStack):
    with use_streaming_world(rank=0, world_size=1, init_dist=False,
                             shared_memory_namespace=namespace):
        world = World.detect()
        prefix_int, shm = get_shm_prefix([local], [remote], world, retry=0)
    stack.callback(shm.cleanup)
    return prefix_int, shm


def test_same_dir_remote_backed_rejected_across_namespaces(tmp_path: Path):
    """The namespace must not bypass local-dir collision protection."""
    local = str(tmp_path / 'local')
    with ExitStack() as stack:
        _register(local, 'fake-remote', 'ns_a', stack)
        with pytest.raises(ValueError, match='Reused local directory'):
            _register(local, 'fake-remote', 'ns_b', stack)


def test_same_dir_remote_backed_rejected_without_namespace(tmp_path: Path):
    local = str(tmp_path / 'local')
    with ExitStack() as stack:
        _register(local, 'fake-remote', None, stack)
        with pytest.raises(ValueError, match='Reused local directory'):
            _register(local, 'fake-remote', None, stack)


def test_namespaced_registration_rejects_plain_same_dir(tmp_path: Path):
    local = str(tmp_path / 'local')
    with ExitStack() as stack:
        _register(local, 'fake-remote', 'ns_a', stack)
        with pytest.raises(ValueError, match='Reused local directory'):
            _register(local, 'fake-remote', None, stack)


def test_same_dir_local_only_coexists_across_namespaces(tmp_path: Path):
    """Co-existing local-only dataset instances are exactly what namespaces enable."""
    local = str(tmp_path / 'local')
    with ExitStack() as stack:
        prefix_a, _ = _register(local, None, 'ns_a', stack)
        prefix_b, _ = _register(local, None, 'ns_b', stack)
        assert prefix_a != prefix_b


def test_different_dirs_remote_backed_coexist(tmp_path: Path):
    with ExitStack() as stack:
        prefix_a, _ = _register(str(tmp_path / 'a'), 'fake-remote-a', 'ns_a', stack)
        prefix_b, _ = _register(str(tmp_path / 'b'), 'fake-remote-b', 'ns_b', stack)
        assert prefix_a != prefix_b


def test_mixed_remote_overlap_checks_only_remote_backed_stream(tmp_path: Path):
    shared = str(tmp_path / 'shared')
    with ExitStack() as stack:
        _register(shared, None, 'ns_a', stack)
        # our remote-backed stream overlaps a registered local-only dir -> reject
        with pytest.raises(ValueError, match='Reused local directory'):
            _register(shared, 'fake-remote', 'ns_b', stack)


def test_registered_identity_keeps_namespace(tmp_path: Path):
    local = str(tmp_path / 'local')
    with ExitStack() as stack:
        _, shm = _register(local, None, 'ns_a', stack)
        their_locals, _ = _unpack_locals(bytes(shm.buf))
        assert their_locals == [f'{local}{_NAMESPACE_SEPARATOR}ns_a']
        assert _strip_namespace(their_locals[0]) == local
