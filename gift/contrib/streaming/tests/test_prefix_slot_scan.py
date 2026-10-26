"""Tests for the shm slot-scan path: probe resource hygiene and the scan cap."""

import logging
import os
import sys
from unittest.mock import patch
from uuid import uuid4

import pytest

from streaming.base.constant import BARRIER, LOCALS
from streaming.base.shared.memory import SharedMemory
from streaming.base.shared.prefix import (_check_and_find, _get_path, _pack_locals, _slot_is_free,
                                          get_shm_prefix)
from streaming.base.world import World

pytestmark = pytest.mark.skipif(sys.platform != 'linux',
                                reason='_slot_is_free probes /dev/shm paths')


def _shm_fd_count() -> int:
    count = 0
    for fd in os.listdir('/proc/self/fd'):
        try:
            if os.readlink(f'/proc/self/fd/{fd}').startswith('/dev/shm/'):
                count += 1
        except OSError:
            continue
    return count


@pytest.fixture()
def occupied_slot():
    slot = uuid4().int % 1_000_000_000
    shm = SharedMemory(_get_path(slot, LOCALS), True, 8, auto_cleanup=False)
    yield slot
    shm.cleanup()


def test_probe_keeps_slot_semantics(occupied_slot: int):
    assert _slot_is_free(occupied_slot) is False
    assert _slot_is_free(occupied_slot + 1) is True


def test_probe_leaves_no_open_fds(occupied_slot: int):
    fds_before = _shm_fd_count()
    for _ in range(10):
        assert _slot_is_free(occupied_slot) is False
    assert _shm_fd_count() == fds_before


@pytest.mark.parametrize('shm_name', [LOCALS, BARRIER])
def test_scan_releases_attached_segments(shm_name: str):
    slot = uuid4().int % 1_000_000_000
    data = _pack_locals(['registered-local'], slot) if shm_name == LOCALS else b'0'
    segment = SharedMemory(_get_path(slot, shm_name), True, len(data), auto_cleanup=False)
    segment.buf[:len(data)] = data

    probe_handles = []
    original_init = SharedMemory.__init__

    def tracking_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        probe_handles.append(self)

    try:
        fds_before = _shm_fd_count()
        with patch.object(SharedMemory, '__init__', tracking_init), \
             patch('streaming.base.shared.prefix._each_prefix_int',
                   return_value=iter((slot, slot + 1))), \
             patch('streaming.base.shared.memory.atexit.register') as register:
            try:
                assert _check_and_find(['new-local'], [None], shm_name) == slot + 1
                assert _shm_fd_count() == fds_before
                register.assert_not_called()
            finally:
                for probe_handle in probe_handles:
                    probe_handle.cleanup()
    finally:
        segment.cleanup()


def test_probe_sees_mid_creation_segment():
    slot = uuid4().int % 1_000_000_000
    path = os.path.join('/dev/shm', _get_path(slot, LOCALS))
    open(path, 'w').close()
    try:
        assert _slot_is_free(slot) is False
    finally:
        os.unlink(path)


def test_mid_creation_segment_logged_and_skipped(caplog: pytest.LogCaptureFixture):
    with patch('streaming.base.shared.prefix.SharedMemory',
               side_effect=[ValueError('cannot mmap an empty file'),
                            FileNotFoundError()]), \
         patch('os.path.exists', return_value=False), \
         caplog.at_level(logging.DEBUG, logger='streaming.base.shared.prefix'):
        assert _check_and_find(['local-log-test'], [None], LOCALS) == 1
    assert any('cannot mmap an empty file' in record.message for record in caplog.records)


def test_leader_scan_capped_with_clear_error():
    world = World(num_nodes=1, ranks_per_node=1, workers_per_rank=1, worker=0)
    with patch('streaming.base.shared.prefix._check_and_find_retrying', return_value=0), \
         patch('streaming.base.shared.prefix.barrier'), \
         patch('streaming.base.shared.prefix._slot_is_free', return_value=False), \
         pytest.raises(RuntimeError, match='all occupied'):
        get_shm_prefix(['local-cap-test'], [None], world, retry=0)
