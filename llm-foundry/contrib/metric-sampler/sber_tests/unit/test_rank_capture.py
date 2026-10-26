from __future__ import annotations

import sys
import uuid
from multiprocessing import shared_memory
from typing import ClassVar

import pytest
import torch

import metric_sampler.node.process as process_module
from metric_sampler.capture.dump import ActivationDumpConfigError, DumpConfig
from metric_sampler.capture.dump_buffer import MAX_BUFFER_BYTES, DumpEntry
from metric_sampler.capture.layout import Layout
from metric_sampler.contract.catalog import metric_by_name
from metric_sampler.node.process import NodeProcess
from metric_sampler.rank_capture import (
    NODE_STOP_TIMEOUT_S,
    RING_SLOTS,
    RankCapture,
    RankPlacement,
)

pytestmark = pytest.mark.unit


class _Popen:
    """The node process as `subprocess.Popen` gives it: running until `exit_code` is set."""

    instances: ClassVar[list[_Popen]] = []

    def __init__(self, argv, stdout=None, stderr=None) -> None:
        self.argv = argv
        self.returncode: int | None = None
        self.exit_code = 0
        self.wait_timeouts: list[float | None] = []
        _Popen.instances.append(self)

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        self.returncode = self.exit_code
        return self.returncode


@pytest.fixture
def popen(monkeypatch):
    _Popen.instances = []
    monkeypatch.setattr(process_module.subprocess, "Popen", _Popen)
    return _Popen.instances


@pytest.fixture
def layout() -> Layout:
    return Layout([(metric_by_name("loss/total"), -1, 0, 1)])


def _placement(local_rank: int = 0, node_rank: int = 1, **overrides) -> RankPlacement:
    settings = dict(attempt=f"t{uuid.uuid4().hex[:8]}-0", node="host7.example", global_rank=node_rank * 2 + local_rank,
                    local_rank=local_rank, local_world_size=2, node_rank=node_rank, world_size=6)
    return RankPlacement(**{**settings, **overrides})


@pytest.fixture
def captures(tmp_path, layout):
    created = []

    def factory(placement: RankPlacement) -> RankCapture:
        capture = RankCapture(tmp_path, placement, layout, torch.device("cpu"))
        created.append(capture)
        return capture

    yield factory
    for capture in created:
        capture.close()


def test_placement_names_the_rings_and_dump_buffers_by_attempt_short_node_and_local_rank() -> None:
    placement = _placement(attempt="123-0")

    assert placement.ring_name(1) == "ms_123-0_host7_1"
    assert placement.dump_buffer_name(0) == "msd_123-0_host7_0"


def test_a_rank_capture_records_into_its_own_ring(captures) -> None:
    placement = _placement(local_rank=1)
    capture = captures(placement)

    assert capture.ring.slot_count == RING_SLOTS
    capture.recorder.begin_step(False)
    capture.recorder.record("loss/total", -1, 0, torch.tensor(2.5))
    assert capture.recorder.accumulator("loss/total", -1, 0).tolist() == [2.5]
    attached = shared_memory.SharedMemory(name=placement.ring_name(1))
    attached.close()


def test_a_rank_holds_its_claim_on_the_attempt_until_closed(tmp_path, captures) -> None:
    placement = _placement()
    capture = captures(placement)

    # Another holder of the rank's claim is refused (BlockingIOError on POSIX, PermissionError on Windows).
    with pytest.raises(OSError):
        NodeProcess.acquire_rank_lock(tmp_path, placement.node, placement.attempt, placement.global_rank)
    capture.close()
    NodeProcess.acquire_rank_lock(tmp_path, placement.node, placement.attempt, placement.global_rank).close()


def test_a_ring_that_cannot_be_created_releases_the_claim(tmp_path, layout, captures) -> None:
    placement = _placement()
    captures(placement)
    second = _placement(attempt=placement.attempt, global_rank=placement.global_rank + 1)

    with pytest.raises(FileExistsError):
        RankCapture(tmp_path, second, layout, torch.device("cpu"))

    NodeProcess.acquire_rank_lock(tmp_path, second.node, second.attempt, second.global_rank).close()


def test_close_unlinks_the_ring_and_is_idempotent(captures) -> None:
    placement = _placement()
    capture = captures(placement)

    capture.close()
    capture.close()

    assert capture.recorder is None
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=placement.ring_name(0))


def _dump_config(tokens: int = 4) -> DumpConfig:
    return DumpConfig(interval=2, entries=(DumpEntry("residual", 0), DumpEntry("embed", -1)), tokens=tokens,
                      max_bytes=10**6)


def test_dumps_get_a_buffer_named_for_the_rank_and_a_capture_on_its_device(captures) -> None:
    placement = _placement(node_rank=0)
    capture = captures(placement)

    capture.start_dumps(_dump_config(), hidden_size=8)

    assert capture.dump_buffer.entries == _dump_config().entries
    assert (capture.dump_buffer.tokens, capture.dump_buffer.hidden) == (4, 8)
    capture.dump_capture.begin_step(2)
    assert capture.dump_buffer.state != 0
    attached = shared_memory.SharedMemory(name=placement.dump_buffer_name(0))
    attached.close()


def test_dumps_over_the_buffer_limit_are_refused_before_any_buffer_exists(captures) -> None:
    capture = captures(_placement())
    hidden = MAX_BUFFER_BYTES // 4

    with pytest.raises(ActivationDumpConfigError, match="over the 1073741824-byte hard limit"):
        capture.start_dumps(_dump_config(tokens=4), hidden_size=hidden)

    assert capture.dump_buffer is None and capture.dump_capture is None


def test_stopped_dumps_release_their_buffer_and_stopping_again_does_nothing(captures) -> None:
    placement = _placement()
    capture = captures(placement)
    capture.start_dumps(_dump_config(), hidden_size=8)

    capture.stop_dumps()
    capture.stop_dumps()

    assert capture.dump_buffer is None and capture.dump_capture is None
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=placement.dump_buffer_name(0))


def test_close_releases_started_dumps(captures) -> None:
    placement = _placement()
    capture = captures(placement)
    capture.start_dumps(_dump_config(), hidden_size=8)

    capture.close()

    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=placement.dump_buffer_name(0))


def test_only_the_first_local_rank_of_a_node_starts_the_node_process(captures, popen) -> None:
    capture = captures(_placement(local_rank=1))

    capture.start_node(window_steps=10, capture_interval=1, dumps_enabled=False)

    assert capture.node_process is None and popen == []
    assert capture.node_running()


@pytest.mark.parametrize(("node_rank", "dumps"), [(0, True), (2, False)])
def test_the_node_process_watches_every_ring_of_the_node(tmp_path, captures, popen, node_rank, dumps) -> None:
    placement = _placement(node_rank=node_rank)
    capture = captures(placement)

    capture.start_node(window_steps=5, capture_interval=4, dumps_enabled=dumps)

    argv = popen[0].argv
    first = node_rank * 2
    assert argv[:3] == [sys.executable, "-m", "metric_sampler.node"]
    assert argv[argv.index("--ring-names") + 1] == (
        f"{first}={placement.ring_name(0)},{first + 1}={placement.ring_name(1)}")
    assert argv[argv.index("--output-dir") + 1] == str(tmp_path)
    assert argv[argv.index("--window-steps") + 1] == "5"
    assert argv[argv.index("--capture-interval") + 1] == "4"
    assert argv[argv.index("--nodes-in-job") + 1] == "3"
    assert argv[argv.index("--rank-timeout-s") + 1] == "120.0"
    assert argv[argv.index("--poll-interval-s") + 1] == "0.5"
    assert argv[argv.index("--attach-deadline-s") + 1] == "300.0"
    assert ("--primary-node" in argv) == (node_rank == 0)
    expected_dumps = f"{first}={placement.dump_buffer_name(0)},{first + 1}={placement.dump_buffer_name(1)}"
    assert (argv[argv.index("--dump-buffer-names") + 1] if "--dump-buffer-names" in argv else None) == (
        expected_dumps if dumps else None)


def test_node_running_turns_false_once_the_node_process_exits(captures, popen) -> None:
    capture = captures(_placement())
    capture.start_node(window_steps=10, capture_interval=1, dumps_enabled=False)

    assert capture.node_running()
    popen[0].returncode = 3
    assert not capture.node_running()


def test_close_waits_for_the_node_process_then_releases_the_claim(tmp_path, captures, popen) -> None:
    placement = _placement()
    capture = captures(placement)
    capture.start_node(window_steps=10, capture_interval=1, dumps_enabled=False)

    capture.close()

    assert popen[0].wait_timeouts == [NODE_STOP_TIMEOUT_S]
    assert capture.node_process is None
    NodeProcess.acquire_rank_lock(tmp_path, placement.node, placement.attempt, placement.global_rank).close()


def test_a_failed_node_process_still_releases_the_claim_and_reports_the_failure(tmp_path, captures, popen) -> None:
    placement = _placement()
    capture = captures(placement)
    capture.start_node(window_steps=10, capture_interval=1, dumps_enabled=False)
    popen[0].exit_code = 7

    with pytest.raises(BufferError, match="exited with code 7"):
        capture.close()
    capture.close()

    assert capture.node_process is None
    NodeProcess.acquire_rank_lock(tmp_path, placement.node, placement.attempt, placement.global_rank).close()


def test_the_node_process_log_is_kept_beside_the_rank_locks(tmp_path, captures, popen) -> None:
    placement = _placement()
    capture = captures(placement)

    capture.start_node(window_steps=10, capture_interval=1, dumps_enabled=False)

    assert (tmp_path / f"attempt={placement.attempt}" / f"node={placement.node}" / "aggregator.log").is_file()
    assert popen[0].argv[popen[0].argv.index("--node") + 1] == placement.node
