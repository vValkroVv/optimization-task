from __future__ import annotations

import os
import uuid

import pytest

from metric_sampler.capture.dump_buffer import DumpBuffer, DumpEntry
from metric_sampler.capture.layout import Layout
from metric_sampler.capture.ring import ShmRing

_SEGMENT_SUFFIX_LENGTH = 20


def _unique_segment_name(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:_SEGMENT_SUFFIX_LENGTH]}"


@pytest.fixture(autouse=True)
def process_liveness(monkeypatch):
    # os.kill(pid, 0) probes Linux processes but can interrupt the test runner on Windows.
    def probe(pid, signal):
        assert signal == 0
        if pid != os.getpid():
            raise ProcessLookupError

    monkeypatch.setattr("metric_sampler.node.step_collector.os.kill", probe)


@pytest.fixture
def shm_rings():
    """Creates `ShmRing`s on uniquely named segments and unlinks them all after the test."""

    created = []

    def factory(layout: Layout, slot_count: int) -> ShmRing:
        name = _unique_segment_name("ms_test")
        ring = ShmRing.create(name, layout, slot_count)
        created.append(ring)
        return ring

    yield factory

    for ring in created:
        ring.close()
        ring.unlink()


@pytest.fixture
def dump_buffers():
    """Creates `DumpBuffer`s on uniquely named segments and unlinks them all after the test."""

    created = []

    def factory(entries: tuple[DumpEntry, ...], tokens: int, hidden: int) -> DumpBuffer:
        name = _unique_segment_name("ms_test_dump")
        buffer = DumpBuffer.create(name, entries, tokens, hidden)
        created.append(buffer)
        return buffer

    yield factory

    for buffer in created:
        buffer.close()
        buffer.unlink()
