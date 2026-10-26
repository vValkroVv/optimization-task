from __future__ import annotations

import importlib
import signal
import subprocess
import sys
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import ClassVar

import pytest

from metric_sampler.capture.dump_buffer import DumpBuffer, DumpEntry
from metric_sampler.capture.layout import Layout
from metric_sampler.capture.ring import RingNotReady, ShmRing
from metric_sampler.contract.catalog import metric_by_name
from metric_sampler.node import process
from metric_sampler.node.aggregator import NodeAggregator

pytestmark = pytest.mark.unit

_LOSS = metric_by_name("loss/total")
_DUMP_ENTRIES = (DumpEntry("residual", 0),)


def test_node_lock_excludes_recovery_until_writer_releases_it(tmp_path) -> None:
    with process._acquire_node_lock(tmp_path), pytest.raises(OSError):
        process._acquire_node_lock(tmp_path)
    with process._acquire_node_lock(tmp_path) as lock_file:
        assert not lock_file.closed
    assert lock_file.closed


def test_recovery_command_preserves_live_writer_and_recovers_after_stop(shm_rings, tmp_path, capsys) -> None:
    layout = Layout([(_LOSS, -1, -1, 1)])
    ring = shm_rings(layout, slot_count=1)
    NodeAggregator([process.RankRing(0, ring)], tmp_path, "n0", "a1", 10, 5.0, 1, True)
    node_dir = tmp_path / "attempt=a1" / "node=n0"
    with process._acquire_node_lock(node_dir), pytest.raises(OSError):
        process.recover_main([str(node_dir)])
    assert capsys.readouterr().out == ""
    process.recover_main([str(node_dir)])
    assert capsys.readouterr().out == "Recovered 0 metric windows\n"


def test_recovery_refuses_live_training_ranks_even_without_node_writer(shm_rings, tmp_path, capsys) -> None:
    layout = Layout([(_LOSS, -1, -1, 1)])
    ring = shm_rings(layout, slot_count=1)
    NodeAggregator([process.RankRing(0, ring)], tmp_path, "n0", "a1", 10, 5.0, 1, True)
    node_dir = tmp_path / "attempt=a1" / "node=n0"
    with process.NodeProcess.acquire_rank_lock(tmp_path, "n0", "a1", 0), pytest.raises(OSError):
        process.recover_main([str(node_dir)])
    assert capsys.readouterr().out == ""
    process.recover_main([str(node_dir)])
    assert capsys.readouterr().out == "Recovered 0 metric windows\n"


@pytest.fixture
def layout() -> Layout:
    return Layout([(_LOSS, -1, -1, 1)])


def test_dunder_main_module_imports_without_running_main() -> None:
    importlib.import_module("metric_sampler.node.__main__")


def test_parse_rank_names_reads_rank_equals_name_pairs() -> None:
    assert process._parse_rank_names("0=ring-a,1=ring-b") == {0: "ring-a", 1: "ring-b"}


def test_format_rank_names_lists_the_pairs_in_rank_order() -> None:
    formatted = process._format_rank_names({1: "ring-b", 0: "ring-a"})

    assert formatted == "0=ring-a,1=ring-b"
    assert process._parse_rank_names(formatted) == {0: "ring-a", 1: "ring-b"}


def test_attach_rings_succeeds_immediately_when_rings_already_exist(shm_rings, layout: Layout) -> None:
    ring0 = shm_rings(layout, slot_count=1)
    sleep_calls = []

    result, absent = process._attach_rings(
        {0: ring0._shm.name}, deadline_s=5.0, clock=lambda: 0.0, sleep=sleep_calls.append
    )

    assert [rank_ring.rank for rank_ring in result] == [0]
    assert result[0].ring.slot_count == 1
    assert absent == []
    assert sleep_calls == []
    result[0].ring.close()


def test_attach_rings_with_no_ring_names_returns_an_empty_list() -> None:
    result, absent = process._attach_rings({}, deadline_s=1.0, clock=lambda: 0.0, sleep=lambda seconds: None)

    assert result == []
    assert absent == []


def test_attach_rings_retries_until_the_ring_appears(layout: Layout) -> None:
    name = f"ms_test_{uuid.uuid4().hex[:20]}"
    created = {}
    sleep_calls = []

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        created["ring"] = ShmRing.create(name, layout, slot_count=1)

    result, absent = process._attach_rings({0: name}, deadline_s=5.0, clock=lambda: 0.0, sleep=fake_sleep)

    assert [rank_ring.rank for rank_ring in result] == [0]
    assert absent == []
    assert sleep_calls == [process._ATTACH_RETRY_INTERVAL_S]
    result[0].ring.close()
    created["ring"].close()
    created["ring"].unlink()


def test_attach_rings_retries_while_the_ring_is_not_ready(layout: Layout, monkeypatch) -> None:
    name = f"ms_test_{uuid.uuid4().hex[:20]}"
    ring = ShmRing.create(name, layout, slot_count=1)
    real_attach = ShmRing.attach
    attempts = []

    def attach_not_ready_once(ring_name: str) -> ShmRing:
        attempts.append(ring_name)
        if len(attempts) == 1:
            raise RingNotReady(ring_name)
        return real_attach(ring_name)

    monkeypatch.setattr(ShmRing, "attach", staticmethod(attach_not_ready_once))

    result, absent = process._attach_rings({0: name}, deadline_s=5.0, clock=lambda: 0.0, sleep=lambda seconds: None)

    assert [rank_ring.rank for rank_ring in result] == [0]
    assert absent == []
    assert attempts == [name, name]
    result[0].ring.close()
    ring.close()
    ring.unlink()


def test_attach_rings_raises_timeout_when_nothing_ever_attaches() -> None:
    clock_box = [0.0]

    def fake_sleep(seconds: float) -> None:
        clock_box[0] += 10.0

    with pytest.raises(process.RingAttachTimeout) as exc_info:
        process._attach_rings(
            {0: "missing-a", 1: "missing-b"}, deadline_s=1.0, clock=lambda: clock_box[0], sleep=fake_sleep
        )

    message = str(exc_info.value)
    assert "missing-a" in message
    assert "missing-b" in message


def test_attach_rings_continues_with_whatever_attached_by_the_deadline(shm_rings, layout: Layout) -> None:
    ring0 = shm_rings(layout, slot_count=1)
    clock_box = [0.0]

    def fake_sleep(seconds: float) -> None:
        clock_box[0] += 10.0

    result, absent = process._attach_rings(
        {0: ring0._shm.name, 1: "missing-b"}, deadline_s=1.0, clock=lambda: clock_box[0], sleep=fake_sleep
    )

    assert [rank_ring.rank for rank_ring in result] == [0]
    assert absent == [1]
    result[0].ring.close()


def test_attach_dump_buffers_succeeds_immediately_when_buffers_already_exist(dump_buffers) -> None:
    buffer0 = dump_buffers(_DUMP_ENTRIES, 4, 8)
    sleep_calls = []

    result, absent = process._attach_dump_buffers(
        {0: buffer0._shm.name}, deadline_s=5.0, clock=lambda: 0.0, sleep=sleep_calls.append
    )

    assert [rank_dump.rank for rank_dump in result] == [0]
    assert result[0].buffer.tokens == 4
    assert absent == []
    assert sleep_calls == []
    result[0].buffer.close()


def test_attach_dump_buffers_with_no_names_returns_an_empty_list() -> None:
    result, absent = process._attach_dump_buffers({}, deadline_s=1.0, clock=lambda: 0.0, sleep=lambda seconds: None)

    assert result == []
    assert absent == []


def test_attach_dump_buffers_retries_until_the_buffer_appears() -> None:
    name = f"ms_test_dump_{uuid.uuid4().hex[:20]}"
    created = {}
    sleep_calls = []

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        created["buffer"] = DumpBuffer.create(name, _DUMP_ENTRIES, 4, 8)

    result, absent = process._attach_dump_buffers({0: name}, deadline_s=5.0, clock=lambda: 0.0, sleep=fake_sleep)

    assert [rank_dump.rank for rank_dump in result] == [0]
    assert absent == []
    assert sleep_calls == [process._ATTACH_RETRY_INTERVAL_S]
    result[0].buffer.close()
    created["buffer"].close()
    created["buffer"].unlink()


def test_attach_dump_buffers_retries_while_the_buffer_is_not_ready(monkeypatch) -> None:
    name = f"ms_test_dump_{uuid.uuid4().hex[:20]}"
    buffer = DumpBuffer.create(name, _DUMP_ENTRIES, 4, 8)
    real_attach = DumpBuffer.attach
    attempts = []

    def attach_not_ready_once(buffer_name: str) -> DumpBuffer:
        attempts.append(buffer_name)
        if len(attempts) == 1:
            raise RingNotReady(buffer_name)
        return real_attach(buffer_name)

    monkeypatch.setattr(DumpBuffer, "attach", staticmethod(attach_not_ready_once))

    result, absent = process._attach_dump_buffers(
        {0: name}, deadline_s=5.0, clock=lambda: 0.0, sleep=lambda seconds: None
    )

    assert [rank_dump.rank for rank_dump in result] == [0]
    assert absent == []
    assert attempts == [name, name]
    result[0].buffer.close()
    buffer.close()
    buffer.unlink()


def test_attach_dump_buffers_raises_timeout_when_nothing_ever_attaches() -> None:
    clock_box = [0.0]

    def fake_sleep(seconds: float) -> None:
        clock_box[0] += 10.0

    with pytest.raises(process.RingAttachTimeout) as exc_info:
        process._attach_dump_buffers(
            {0: "missing-a", 1: "missing-b"}, deadline_s=1.0, clock=lambda: clock_box[0], sleep=fake_sleep
        )

    message = str(exc_info.value)
    assert "missing-a" in message
    assert "missing-b" in message


def test_attach_dump_buffers_continues_with_whatever_attached_by_the_deadline(dump_buffers) -> None:
    buffer0 = dump_buffers(_DUMP_ENTRIES, 4, 8)
    clock_box = [0.0]

    def fake_sleep(seconds: float) -> None:
        clock_box[0] += 10.0

    result, absent = process._attach_dump_buffers(
        {0: buffer0._shm.name, 1: "missing-b"}, deadline_s=1.0, clock=lambda: clock_box[0], sleep=fake_sleep
    )

    assert [rank_dump.rank for rank_dump in result] == [0]
    assert absent == [1]
    result[0].buffer.close()


def test_set_parent_death_signal_calls_prctl_only_on_linux(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(process, "_prctl_set_pdeathsig", lambda: calls.append(True))
    monkeypatch.setattr(process.sys, "platform", "linux")

    process._set_parent_death_signal()

    assert calls == [True]


def test_set_parent_death_signal_is_a_noop_off_linux(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(process, "_prctl_set_pdeathsig", lambda: calls.append(True))
    monkeypatch.setattr(process.sys, "platform", "win32")

    process._set_parent_death_signal()

    assert calls == []


def test_install_sigterm_handler_sets_the_flag_when_invoked() -> None:
    original_handler = signal.getsignal(signal.SIGTERM)
    try:
        should_stop = process._install_sigterm_handler()
        assert should_stop() is False

        handler = signal.getsignal(signal.SIGTERM)
        handler(signal.SIGTERM, None)

        assert should_stop() is True
    finally:
        signal.signal(signal.SIGTERM, original_handler)


def _main_argv(
    output_dir: Path,
    ring_names: str,
    dump_buffer_names: str | None = None,
    primary_node: bool = False,
    attach_deadline_s: float = 1.0,
) -> list[str]:
    argv = [
        "--ring-names", ring_names,
        "--output-dir", str(output_dir),
        "--node", "n0",
        "--attempt", "a1",
        "--window-steps", "4",
        "--rank-timeout-s", "2.5",
        "--nodes-in-job", "16",
        "--poll-interval-s", "0.1",
        "--attach-deadline-s", str(attach_deadline_s),
    ]
    if dump_buffer_names is not None:
        argv += ["--dump-buffer-names", dump_buffer_names]
    if primary_node:
        argv.append("--primary-node")
    return argv


def test_main_builds_the_aggregator_with_the_parsed_settings(shm_rings, layout: Layout, tmp_path, monkeypatch) -> None:
    ring0 = shm_rings(layout, slot_count=1)
    calls = {}

    def fake_run(self, poll_interval_s, rank_pids, should_stop=None):
        calls["poll_interval_s"] = poll_interval_s
        calls["rank_pids"] = rank_pids
        calls["should_stop_is_callable"] = callable(should_stop)
        calls["window_steps"] = self._window_steps
        calls["rank_timeout_s"] = self._rank_timeout_s
        calls["nodes_in_job"] = self._nodes_in_job
        calls["node"] = self._node
        calls["attempt"] = self._attempt

    monkeypatch.setattr(NodeAggregator, "run", fake_run)

    process.main(_main_argv(tmp_path, f"0={ring0._shm.name}", primary_node=True))

    assert calls["poll_interval_s"] == 0.1
    assert calls["rank_pids"] == [ring0.owner_pid]
    assert calls["should_stop_is_callable"] is True
    assert calls["window_steps"] == 4
    assert calls["rank_timeout_s"] == 2.5
    assert calls["nodes_in_job"] == 16
    assert calls["node"] == "n0"
    assert calls["attempt"] == "a1"
    assert (tmp_path / "attempt=a1" / "metrics.json").exists()  # --primary-node was passed


def test_main_attaches_dump_buffers_and_closes_them_when_the_aggregator_returns(
    shm_rings, layout: Layout, dump_buffers, tmp_path, monkeypatch
) -> None:
    ring0 = shm_rings(layout, slot_count=1)
    dump0 = dump_buffers(_DUMP_ENTRIES, 4, 8)
    calls = {}

    class RecordingAggregator:
        def __init__(self, rank_rings, output_dir, **kwargs) -> None:
            calls["dump_ranks"] = sorted(rank_dump.rank for rank_dump in kwargs["dump_rings"])

        def run(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(process, "NodeAggregator", RecordingAggregator)

    process.main(
        _main_argv(tmp_path, f"0={ring0._shm.name}", dump_buffer_names=f"0={dump0._shm.name}", primary_node=True)
    )

    assert calls["dump_ranks"] == [0]
    # main()'s finally block already closed dump0's view of the segment; closing it again here
    # (the fixture's own teardown) must stay a no-op rather than raise.


def test_main_exits_with_a_clear_error_when_a_dump_buffer_never_appears(shm_rings, layout: Layout, tmp_path) -> None:
    ring0 = shm_rings(layout, slot_count=1)
    argv = _main_argv(
        tmp_path, f"0={ring0._shm.name}", dump_buffer_names="0=does-not-exist-1234", attach_deadline_s=0.05
    )

    with pytest.raises(SystemExit) as exc_info:
        process.main(argv)

    assert "does-not-exist-1234" in str(exc_info.value)


def test_main_logs_absent_dump_ranks_to_stderr_and_still_runs(
    shm_rings, layout: Layout, dump_buffers, tmp_path, monkeypatch, capsys
) -> None:
    ring0 = shm_rings(layout, slot_count=1)
    dump0 = dump_buffers(_DUMP_ENTRIES, 4, 8)
    monkeypatch.setattr(NodeAggregator, "run", lambda self, *args, **kwargs: None)

    process.main(
        _main_argv(
            tmp_path,
            f"0={ring0._shm.name}",
            dump_buffer_names=f"0={dump0._shm.name},1=missing-dump",
            attach_deadline_s=0.05,
        )
    )

    assert "1" in capsys.readouterr().err


def test_main_without_the_primary_node_flag_skips_the_attempt_registries(
    shm_rings, layout: Layout, tmp_path, monkeypatch
) -> None:
    ring0 = shm_rings(layout, slot_count=1)
    monkeypatch.setattr(NodeAggregator, "run", lambda self, *args, **kwargs: None)

    process.main(_main_argv(tmp_path, f"0={ring0._shm.name}"))

    assert not (tmp_path / "attempt=a1" / "metrics.json").exists()


def test_main_logs_absent_ranks_to_stderr_and_still_runs_the_aggregator(
    shm_rings, layout: Layout, tmp_path, monkeypatch, capsys
) -> None:
    ring0 = shm_rings(layout, slot_count=1)
    calls = {}
    monkeypatch.setattr(NodeAggregator, "run", lambda self, *args, **kwargs: calls.setdefault("ran", True))

    process.main(_main_argv(tmp_path, f"0={ring0._shm.name},1=missing-rank", attach_deadline_s=0.05))

    assert calls["ran"] is True
    assert "1" in capsys.readouterr().err


def test_main_exits_with_a_clear_error_when_a_ring_never_appears(tmp_path) -> None:
    argv = _main_argv(tmp_path, "0=does-not-exist-1234", attach_deadline_s=0.05)

    with pytest.raises(SystemExit) as exc_info:
        process.main(argv)

    assert "0=does-not-exist-1234" in str(exc_info.value)


class _FakePopen:
    instances: ClassVar[list[_FakePopen]] = []

    def __init__(self, argv, stdout=None, stderr=None) -> None:
        self.argv = argv
        self.stdout = stdout
        self.stderr = stderr
        self.terminated = False
        self.killed = False
        self.returncode = 0
        self.wait_calls: list[float | None] = []
        _FakePopen.instances.append(self)

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> None:
        self.wait_calls.append(timeout)

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


class _FakePopenNeedsTerminate(_FakePopen):
    """wait(timeout_s) times out, but the process is gone by the terminate grace period."""

    def wait(self, timeout: float | None = None) -> None:
        self.wait_calls.append(timeout)
        if len(self.wait_calls) == 1:
            raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)


class _FakePopenNeedsKill(_FakePopen):
    """Both wait(timeout_s) and the post-terminate wait time out; only kill() reaps it."""

    def wait(self, timeout: float | None = None) -> None:
        self.wait_calls.append(timeout)
        if len(self.wait_calls) <= 2:
            raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)


@pytest.fixture(autouse=True)
def _clear_fake_popen_instances():
    _FakePopen.instances = []
    yield
    _FakePopen.instances = []


def _start_node_process(output_dir: Path, dump_buffer_names: dict[int, str] | None = None) -> process.NodeProcess:
    return process.NodeProcess.start(
        ring_names={0: "ring-a"},
        output_dir=output_dir,
        node="n0",
        attempt="att1",
        window_steps=1,
        rank_timeout_s=1.0,
        nodes_in_job=16,
        poll_interval_s=0.1,
        attach_deadline_s=1.0,
        primary_node=False,
        dump_buffer_names=dump_buffer_names,
    )


def test_node_process_start_launches_the_expected_argv_and_log_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(process.subprocess, "Popen", _FakePopen)

    node_process = process.NodeProcess.start(
        ring_names={0: "ring-a", 1: "ring-b"},
        output_dir=tmp_path,
        node="n0",
        attempt="att1",
        window_steps=4,
        rank_timeout_s=2.0,
        nodes_in_job=16,
        poll_interval_s=0.5,
        attach_deadline_s=30.0,
        primary_node=True,
        capture_interval=5,
    )

    popen = _FakePopen.instances[0]
    assert popen.argv[:3] == [sys.executable, "-m", "metric_sampler.node"]
    assert popen.argv[popen.argv.index("--ring-names") + 1] == "0=ring-a,1=ring-b"
    assert popen.argv[popen.argv.index("--output-dir") + 1] == str(tmp_path)
    assert popen.argv[popen.argv.index("--node") + 1] == "n0"
    assert popen.argv[popen.argv.index("--attempt") + 1] == "att1"
    assert popen.argv[popen.argv.index("--window-steps") + 1] == "4"
    assert popen.argv[popen.argv.index("--capture-interval") + 1] == "5"
    assert popen.argv[popen.argv.index("--rank-timeout-s") + 1] == "2.0"
    assert popen.argv[popen.argv.index("--nodes-in-job") + 1] == "16"
    assert popen.argv[popen.argv.index("--poll-interval-s") + 1] == "0.5"
    assert popen.argv[popen.argv.index("--attach-deadline-s") + 1] == "30.0"
    assert "--primary-node" in popen.argv
    assert popen.stderr is subprocess.STDOUT

    log_path = tmp_path / "attempt=att1" / "node=n0" / "aggregator.log"
    assert log_path.exists()

    node_process.stop(timeout_s=1.0)
    assert popen.wait_calls == [1.0]
    assert popen.terminated is False
    assert popen.killed is False


def test_node_process_start_includes_dump_buffer_names_when_given(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(process.subprocess, "Popen", _FakePopen)

    _start_node_process(tmp_path, dump_buffer_names={0: "dump-a", 1: "dump-b"})

    popen = _FakePopen.instances[0]
    assert popen.argv[popen.argv.index("--dump-buffer-names") + 1] == "0=dump-a,1=dump-b"


def test_node_process_stop_terminates_when_still_running_after_the_wait(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(process.subprocess, "Popen", _FakePopenNeedsTerminate)

    node_process = _start_node_process(tmp_path)

    node_process.stop(timeout_s=1.0)

    popen = _FakePopen.instances[0]
    assert popen.wait_calls == [1.0, process._TERMINATE_GRACE_S]
    assert popen.terminated is True
    assert popen.killed is False
    assert "--primary-node" not in popen.argv


def test_node_process_stop_kills_when_still_running_after_terminate(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(process.subprocess, "Popen", _FakePopenNeedsKill)

    node_process = _start_node_process(tmp_path)

    with pytest.raises(BufferError, match="exited with code -9"):
        node_process.stop(timeout_s=1.0)

    popen = _FakePopen.instances[0]
    assert popen.wait_calls == [1.0, process._TERMINATE_GRACE_S, None]
    assert popen.terminated is True
    assert popen.killed is True


@pytest.mark.parametrize("exit_code", [0, 7])
def test_node_writer_exit_is_reported_and_log_is_closed(tmp_path, exit_code) -> None:
    log_file = (tmp_path / "aggregator.log").open("ab")
    child = subprocess.Popen(
        [sys.executable, "-c", f"input(); raise SystemExit({exit_code})"],
        stdin=subprocess.PIPE, stdout=log_file, stderr=subprocess.STDOUT,
    )
    writer = process.NodeProcess(child, log_file)
    assert writer.is_running()
    child.communicate(b"stop\n", timeout=10)
    assert not writer.is_running()
    expected = pytest.raises(BufferError, match="exited with code 7") if exit_code else nullcontext()
    with expected:
        writer.stop(timeout_s=1)
    assert log_file.closed
    assert child.returncode == exit_code
