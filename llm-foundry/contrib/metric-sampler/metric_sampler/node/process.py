from __future__ import annotations

import argparse
import ctypes
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from types import FrameType
from typing import BinaryIO, Literal, TypeVar

from metric_sampler.capture.dump_buffer import DumpBuffer
from metric_sampler.capture.ring import RingNotReady, ShmRing
from metric_sampler.contract.raw_layout import MANIFEST_NAME, node_dir
from metric_sampler.node.aggregator import DEFAULT_CAPTURE_INTERVAL, NodeAggregator, RankDumpBuffer, RankRing

_ATTACH_RETRY_INTERVAL_S = 0.1
_PR_SET_PDEATHSIG = 1
_TERMINATE_GRACE_S = 10
_LOCKED_BYTE_COUNT = 1
_LOG_NAME = "aggregator.log"
_NODE_LOCK_NAME = ".writer.lock"
_RANK_LOCK_PREFIX = ".rank-"
_RANK_LOCK_SUFFIX = ".lock"

_Attached = TypeVar("_Attached")


class RingAttachTimeout(Exception):
    """Raised when no rank ring at all appears before the attach deadline."""


def _parse_rank_names(spec: str) -> dict[int, str]:
    rank_names = {}
    for pair in spec.split(","):
        rank_text, name = pair.split("=", 1)
        rank_names[int(rank_text)] = name
    return rank_names


def _format_rank_names(rank_names: dict[int, str]) -> str:
    return ",".join(f"{rank}={name}" for rank, name in sorted(rank_names.items()))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Node-level metric aggregator process.")
    parser.add_argument("--ring-names", required=True, help="comma-separated rank=shm_name pairs")
    parser.add_argument(
        "--dump-buffer-names",
        default=None,
        help="comma-separated rank=shm_name pairs for activation-dump buffers; omitted when "
        "activation dumps are disabled for the run",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--window-steps", type=int, required=True)
    parser.add_argument(
        "--capture-interval",
        type=int,
        default=DEFAULT_CAPTURE_INTERVAL,
        help="the training captures every Nth step",
    )
    parser.add_argument("--rank-timeout-s", type=float, required=True)
    parser.add_argument("--nodes-in-job", type=int, required=True)
    parser.add_argument("--poll-interval-s", type=float, required=True)
    parser.add_argument("--attach-deadline-s", type=float, required=True)
    parser.add_argument(
        "--primary-node",
        action="store_true",
        help="this is the job's primary node: it owns metrics.json/streams.json for the attempt "
        "and the rows of job_identical metrics (exactly one node per job)",
    )
    return parser.parse_args(argv)


def _attach_until_deadline(
    names: dict[int, str],
    attach: Callable[[str], _Attached],
    description: str,
    deadline_s: float,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> tuple[dict[int, _Attached], list[int]]:
    """Attaches every named shared-memory object that appears before the deadline. Ranks that never show up (for
    example with collection disabled) are returned as absent instead of failing the node, unless none attach."""
    deadline = clock() + deadline_s
    attached: dict[int, _Attached] = {}
    remaining = dict(names)
    while remaining:
        for rank, name in list(remaining.items()):
            try:
                attached[rank] = attach(name)
            except (FileNotFoundError, RingNotReady):
                continue
            del remaining[rank]
        if not remaining:
            break
        if clock() >= deadline:
            if not attached:
                missing = ", ".join(f"{rank}={name}" for rank, name in sorted(remaining.items()))
                raise RingAttachTimeout(f"{description} never appeared before the attach deadline: {missing}")
            break
        sleep(_ATTACH_RETRY_INTERVAL_S)
    return attached, sorted(remaining)


def _attach_rings(
    ring_names: dict[int, str],
    deadline_s: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[list[RankRing], list[int]]:
    attached, absent_ranks = _attach_until_deadline(ring_names, ShmRing.attach, "rings", deadline_s, clock, sleep)
    return [RankRing(rank, attached[rank]) for rank in sorted(attached)], absent_ranks


def _attach_dump_buffers(
    buffer_names: dict[int, str],
    deadline_s: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[list[RankDumpBuffer], list[int]]:
    attached, absent_ranks = _attach_until_deadline(
        buffer_names, DumpBuffer.attach, "dump buffers", deadline_s, clock, sleep
    )
    return [RankDumpBuffer(rank, attached[rank]) for rank in sorted(attached)], absent_ranks


def _prctl_set_pdeathsig() -> None:  # pragma: no cover
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)


def _set_parent_death_signal() -> None:
    if sys.platform == "linux":
        _prctl_set_pdeathsig()


def _install_sigterm_handler() -> Callable[[], bool]:
    stop_flag = {"stop": False}

    def handle_sigterm(signum: int, frame: FrameType | None) -> None:
        stop_flag["stop"] = True

    signal.signal(signal.SIGTERM, handle_sigterm)
    return lambda: stop_flag["stop"]


def _lock_file_exclusively(lock_file: BinaryIO) -> None:
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


def _open_locked(path: Path, mode: Literal["a+b", "r+b"]) -> BinaryIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(path, mode)  # noqa: SIM115
    try:
        _lock_file_exclusively(lock_file)
    except OSError:
        lock_file.close()
        raise
    return lock_file


def _acquire_node_lock(node_directory: Path) -> BinaryIO:
    return _open_locked(node_directory / _NODE_LOCK_NAME, "a+b")


def _acquire_rank_lock(node_directory: Path, rank: int, existing: bool) -> BinaryIO:
    lock_path = node_directory / f"{_RANK_LOCK_PREFIX}{rank:05d}{_RANK_LOCK_SUFFIX}"
    return _open_locked(lock_path, "r+b" if existing else "a+b")


def recover_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Recover durable metric steps after the training ranks and node writer have stopped."
    )
    parser.add_argument("node_dir", type=Path)
    args = parser.parse_args(argv)
    with ExitStack() as locks:
        for path in sorted(args.node_dir.glob(f"{_RANK_LOCK_PREFIX}*{_RANK_LOCK_SUFFIX}")):
            rank = int(path.stem.removeprefix(_RANK_LOCK_PREFIX))
            locks.enter_context(_acquire_rank_lock(args.node_dir, rank, existing=True))
        locks.enter_context(_acquire_node_lock(args.node_dir))
        written = NodeAggregator.recover(args.node_dir)
    print(f"Recovered {sum(path.name == MANIFEST_NAME for path in written)} metric windows")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    ring_names = _parse_rank_names(args.ring_names)

    try:
        rank_rings, absent_ranks = _attach_rings(ring_names, args.attach_deadline_s)
    except RingAttachTimeout as error:
        raise SystemExit(str(error)) from error

    if absent_ranks:
        print(
            f"ranks never attached before the deadline, continuing without them: {absent_ranks}",
            file=sys.stderr,
        )

    dump_rings: list[RankDumpBuffer] = []
    if args.dump_buffer_names:
        dump_buffer_names = _parse_rank_names(args.dump_buffer_names)
        try:
            dump_rings, absent_dump_ranks = _attach_dump_buffers(dump_buffer_names, args.attach_deadline_s)
        except RingAttachTimeout as error:
            # rank_rings already attached above must not leak just because dump buffers failed.
            for rank_ring in rank_rings:
                rank_ring.ring.close()
            raise SystemExit(str(error)) from error
        if absent_dump_ranks:
            print(
                f"dump buffers never attached before the deadline, continuing without them: {absent_dump_ranks}",
                file=sys.stderr,
            )

    _set_parent_death_signal()
    should_stop = _install_sigterm_handler()

    try:
        with _acquire_node_lock(node_dir(args.output_dir, args.attempt, args.node)):
            aggregator = NodeAggregator(
                rank_rings,
                args.output_dir,
                node=args.node,
                attempt=args.attempt,
                window_steps=args.window_steps,
                rank_timeout_s=args.rank_timeout_s,
                nodes_in_job=args.nodes_in_job,
                primary_node=args.primary_node,
                absent_ranks=absent_ranks,
                dump_rings=tuple(dump_rings),
                capture_interval=args.capture_interval,
            )
            rank_pids = [rank_ring.ring.owner_pid for rank_ring in rank_rings]
            aggregator.run(args.poll_interval_s, rank_pids, should_stop=should_stop)
    finally:
        for rank_ring in rank_rings:
            rank_ring.ring.close()
        for rank_dump in dump_rings:
            rank_dump.buffer.close()


class NodeProcess:
    """Launches and manages the `python -m metric_sampler.node` aggregator subprocess."""

    def __init__(self, popen: subprocess.Popen[bytes], log_file: BinaryIO) -> None:
        self._popen = popen
        self._log_file = log_file

    @staticmethod
    def acquire_rank_lock(output_dir: Path, node: str, attempt: str, rank: int) -> BinaryIO:
        return _acquire_rank_lock(node_dir(output_dir, attempt, node), rank, existing=False)

    @classmethod
    def start(
        cls,
        ring_names: dict[int, str],
        output_dir: Path,
        node: str,
        attempt: str,
        window_steps: int,
        rank_timeout_s: float,
        nodes_in_job: int,
        poll_interval_s: float,
        attach_deadline_s: float,
        primary_node: bool,
        dump_buffer_names: dict[int, str] | None = None,
        capture_interval: int = DEFAULT_CAPTURE_INTERVAL,
    ) -> NodeProcess:
        log_dir = node_dir(output_dir, attempt, node)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(log_dir / _LOG_NAME, "ab")  # noqa: SIM115

        argv = [
            sys.executable,
            "-m",
            "metric_sampler.node",
            "--ring-names",
            _format_rank_names(ring_names),
            "--output-dir",
            str(output_dir),
            "--node",
            node,
            "--attempt",
            attempt,
            "--window-steps",
            str(window_steps),
            "--capture-interval",
            str(capture_interval),
            "--rank-timeout-s",
            str(rank_timeout_s),
            "--nodes-in-job",
            str(nodes_in_job),
            "--poll-interval-s",
            str(poll_interval_s),
            "--attach-deadline-s",
            str(attach_deadline_s),
        ]
        if primary_node:
            argv.append("--primary-node")
        if dump_buffer_names:
            argv += ["--dump-buffer-names", _format_rank_names(dump_buffer_names)]
        popen = subprocess.Popen(argv, stdout=log_file, stderr=subprocess.STDOUT)
        return cls(popen, log_file)

    def is_running(self) -> bool:
        return self._popen.poll() is None

    def stop(self, timeout_s: float) -> None:
        """The node process exits on its own once every ring it watches requests stop, so this
        waits for that first and only escalates to terminate/kill if it is still running."""
        try:
            self._popen.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self._popen.terminate()
            try:
                self._popen.wait(timeout=_TERMINATE_GRACE_S)
            except subprocess.TimeoutExpired:
                self._popen.kill()
                self._popen.wait()
        self._log_file.close()
        if self._popen.returncode != 0:
            raise BufferError(
                f"metric sampler node writer exited with code {self._popen.returncode}; "
                "inspect aggregator.log before recovering its journal"
            )
