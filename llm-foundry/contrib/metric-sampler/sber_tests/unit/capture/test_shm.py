from __future__ import annotations

import sys
import uuid
from multiprocessing import shared_memory

import pytest

import metric_sampler.capture.shm as shm_module
from metric_sampler.capture.shm import ALIGNMENT, open_untracked, round_up

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("value", "expected"), [(0, 0), (1, ALIGNMENT), (ALIGNMENT, ALIGNMENT), (ALIGNMENT + 1, 2 * ALIGNMENT)]
)
def test_round_up_to_the_alignment(value: int, expected: int) -> None:
    assert round_up(value, ALIGNMENT) == expected


@pytest.fixture
def segment():
    created = shared_memory.SharedMemory(name=f"ms_test_shm_{uuid.uuid4().hex[:20]}", create=True, size=64)
    yield created
    created.close()
    created.unlink()


@pytest.mark.skipif(sys.version_info < (3, 13), reason="SharedMemory takes track from Python 3.13")
def test_open_untracked_passes_track_false_on_python_3_13_plus(segment, monkeypatch) -> None:
    unregister_calls = []
    monkeypatch.setattr(shm_module.resource_tracker, "unregister", lambda *args: unregister_calls.append(args))
    monkeypatch.setattr(shm_module.sys, "version_info", (3, 13, 0))

    opened = open_untracked(segment.name)

    assert unregister_calls == []
    assert bytes(opened.buf[:64]) == bytes(segment.buf[:64])
    opened.close()


def test_open_untracked_unregisters_from_the_resource_tracker_before_python_3_13(segment, monkeypatch) -> None:
    unregister_calls = []
    monkeypatch.setattr(shm_module.resource_tracker, "unregister", lambda *args: unregister_calls.append(args))
    monkeypatch.setattr(shm_module.sys, "version_info", (3, 12, 0))

    opened = open_untracked(segment.name)

    assert unregister_calls == [(opened._name, "shared_memory")]
    opened.close()
