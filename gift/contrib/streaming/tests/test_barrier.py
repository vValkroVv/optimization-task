"""Tests for the shared-memory barrier: protocol, death timeout, and diagnostics."""

import logging
import multiprocessing as mp
import os
import re
import subprocess
import sys
from multiprocessing.managers import ListProxy
from random import random
from time import sleep
from typing import Any
from uuid import uuid4

import pytest

from streaming.base.shared import barrier as barrier_module
from streaming.base.shared.barrier import SharedBarrier


def _make_barrier(tmp_path: Any) -> SharedBarrier:
    return SharedBarrier(str(tmp_path / 'lock'), f'test_barrier_{uuid4().hex[:12]}')


def _enter_twice(num_procs: int, barrier: Any, shared_list: ListProxy) -> None:
    sleep(random())
    shared_list.append(f'Hit barrier, waiting: {mp.current_process().name}')
    barrier(num_procs)
    shared_list.append(f'passed barrier: {mp.current_process().name}')
    barrier(num_procs)
    shared_list.append(f'passed barrier again: {mp.current_process().name}')


@pytest.mark.parametrize('num_procs', [2, 3])
def test_barrier_synchronizes_and_resets(num_procs: int, tmp_path: Any):
    """Two consecutive calls fully phase-separate; the second proves state reset."""
    mp.set_start_method('fork', force=True)
    shared_list = mp.Manager().list()
    barrier = _make_barrier(tmp_path)
    processes = [
        mp.Process(target=_enter_twice, args=(num_procs, barrier, shared_list))
        for _ in range(num_procs)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(30)

    expected = [''] * 3 * num_procs
    for i in range(num_procs):
        expected[i] = 'Hit barrier, waiting: Process-\\d+'
        expected[i + num_procs] = 'passed barrier: Process-\\d+'
        expected[i + 2 * num_procs] = 'passed barrier again: Process-\\d+'
    assert re.fullmatch(' '.join(expected), ' '.join(shared_list))


def _enter_or_die(num_procs: int, barrier: Any, shared_list: ListProxy, idx: int) -> None:
    if idx == num_procs - 1:
        return  # simulate a participant dying before entering the barrier
    try:
        barrier(num_procs)
        shared_list.append('passed')
    except RuntimeError:
        shared_list.append('timeout')


def test_barrier_times_out_on_missing_participant(tmp_path: Any,
                                                  monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(barrier_module, '_TIMEOUT', 2.0)
    mp.set_start_method('fork', force=True)
    shared_list = mp.Manager().list()
    barrier = _make_barrier(tmp_path)
    num_procs = 3
    processes = [
        mp.Process(target=_enter_or_die, args=(num_procs, barrier, shared_list, i))
        for i in range(num_procs)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(30)

    assert list(shared_list) == ['timeout', 'timeout']


def test_timeout_error_carries_diagnostics(tmp_path: Any, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(barrier_module, '_TIMEOUT', 1.0)
    barrier = _make_barrier(tmp_path)
    with pytest.raises(RuntimeError, match='1 of 2 processes entered'):
        barrier(2)  # the second participant never arrives


def _enter_after(barrier: SharedBarrier, delay: float, num_procs: int) -> None:
    sleep(delay)
    barrier(num_procs)


def test_warning_carries_progress_diagnostics(tmp_path: Any, caplog: pytest.LogCaptureFixture,
                                              monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(barrier_module, 'WARN_FIRST', 0.2)
    monkeypatch.setattr(barrier_module, 'WARN_EVERY', 0.2)
    mp.set_start_method('fork', force=True)
    barrier = _make_barrier(tmp_path)
    late = mp.Process(target=_enter_after, args=(barrier, 1.0, 2))
    late.start()
    try:
        with caplog.at_level(logging.WARNING, logger=barrier_module.__name__):
            barrier(2)  # waits ~1s for the late participant, warning along the way
    finally:
        late.join(30)

    assert late.exitcode == 0
    assert caplog.records, 'expected progress warnings during the wait'
    first_warning = caplog.records[0].message
    assert 'SharedBarrier still waiting after' in first_warning
    assert '1 of 2 processes entered' in first_warning
    assert str(tmp_path / 'lock') in first_warning


def test_single_process_passes_without_warnings(tmp_path: Any,
                                                caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING, logger=barrier_module.__name__):
        _make_barrier(tmp_path)(1)
    assert not caplog.records


def test_zero_timeout_disables_deadline(tmp_path: Any, caplog: pytest.LogCaptureFixture,
                                        monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(barrier_module, '_TIMEOUT', 0.0)
    monkeypatch.setattr(barrier_module, 'WARN_FIRST', 0.2)
    mp.set_start_method('fork', force=True)
    barrier = _make_barrier(tmp_path)
    late = mp.Process(target=_enter_after, args=(barrier, 0.7, 2))
    late.start()
    try:
        with caplog.at_level(logging.WARNING, logger=barrier_module.__name__):
            barrier(2)  # would raise before the late arrival if 0 armed a deadline
    finally:
        late.join(30)

    assert late.exitcode == 0
    assert any('still waiting' in record.message for record in caplog.records)


def _import_barrier(timeout_value: str) -> 'subprocess.CompletedProcess[str]':
    return subprocess.run(
        [sys.executable, '-c', 'from streaming.base.shared import barrier; print(barrier._TIMEOUT)'],
        env={**os.environ, 'STREAMING_BARRIER_TIMEOUT': timeout_value},
        capture_output=True,
        text=True)


def test_timeout_env_applies_at_import():
    result = _import_barrier('7.5')
    assert result.returncode == 0
    assert result.stdout.strip() == '7.5'


def test_invalid_timeout_env_fails_at_import_with_clear_error():
    result = _import_barrier('30s')
    assert result.returncode != 0
    assert "STREAMING_BARRIER_TIMEOUT must be a number of seconds, got '30s'" in result.stderr
