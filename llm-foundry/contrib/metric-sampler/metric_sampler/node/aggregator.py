from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from metric_sampler.capture.dump_buffer import DumpBuffer
from metric_sampler.capture.layout import Layout
from metric_sampler.capture.ring import ShmRing
from metric_sampler.contract.raw_layout import AttemptDir, RawRoot, attempt_dir_name, node_dir, node_dir_name
from metric_sampler.node.attempt_registries import AttemptRegistries
from metric_sampler.node.dump_writer import DumpWriter
from metric_sampler.node.spool import Spool
from metric_sampler.node.step_collector import StepCollector
from metric_sampler.node.step_merger import StepMerger
from metric_sampler.node.window_counters import WindowCounters
from metric_sampler.node.window_writer import WindowWriter

# The training captures every step unless told otherwise.
DEFAULT_CAPTURE_INTERVAL = 1


@dataclass(frozen=True)
class RankRing:
    rank: int
    ring: ShmRing


@dataclass(frozen=True)
class RankDumpBuffer:
    rank: int
    buffer: DumpBuffer


class NodeAggregator:
    """Merges the rings of every local rank into per-window Parquet files plus a manifest."""

    def __init__(
        self,
        rank_rings: list[RankRing],
        output_dir: Path,
        node: str,
        attempt: str,
        window_steps: int,
        rank_timeout_s: float,
        nodes_in_job: int,
        primary_node: bool,
        absent_ranks: Sequence[int] = (),
        clock: Callable[[], float] = time.monotonic,
        dump_rings: tuple[RankDumpBuffer, ...] = (),
        rank_layouts: dict[int, Layout] | None = None,
        capture_interval: int = DEFAULT_CAPTURE_INTERVAL,
    ) -> None:
        self._rank_layouts = (
            rank_layouts
            if rank_layouts is not None
            else {rank_ring.rank: rank_ring.ring.layout for rank_ring in rank_rings}
        )
        self._node = node
        self._attempt = attempt
        self._window_steps = window_steps
        # The training captures every capture_interval-th step: consecutive windows are that many steps apart.
        self._capture_interval = capture_interval
        self._rank_timeout_s = rank_timeout_s
        self._nodes_in_job = nodes_in_job
        self._primary_node = primary_node
        self._absent_ranks = sorted(absent_ranks)
        stream_names = self._common_stream_names()
        # The output directory is `<root>/<experiment>/<run>`.
        raw_root = RawRoot(output_dir.parent.parent)
        attempt_dir = AttemptDir(
            path=output_dir / attempt_dir_name(attempt),
            exp=output_dir.parent.name,
            run=output_dir.name,
            attempt=attempt,
        )
        node_directory = node_dir(output_dir, attempt, node)
        counters = WindowCounters(self._rank_layouts)

        attempt_dir.path.mkdir(parents=True, exist_ok=True)
        spool = Spool(node_directory, self._rank_layouts, self._spool_metadata())
        self._collector = StepCollector(
            {rank_ring.rank: rank_ring.ring for rank_ring in rank_rings},
            self._rank_layouts,
            spool,
            counters,
            clock,
            rank_timeout_s,
        )
        self._merger = StepMerger(self._rank_layouts, primary_node, counters)
        self._writer = WindowWriter(
            node,
            node_directory,
            attempt_dir,
            raw_root,
            nodes_in_job,
            list(self._rank_layouts),
            self._absent_ranks,
            capture_interval,
            spool,
            counters,
        )
        self._dump_writer = DumpWriter(attempt_dir, {rank_dump.rank: rank_dump.buffer for rank_dump in dump_rings})
        self._recovered_paths: list[Path] = []
        if spool.has_intent():
            _, self._recovered_paths = self._writer.finish_publication()
        self._collector.restore_spooled_slots()
        if primary_node:
            registries = AttemptRegistries(attempt_dir)
            registries.write_metrics()
            registries.write_streams(stream_names)

    @classmethod
    def recover(cls, node_directory: Path) -> list[Path]:
        """Publish durable steps left behind by a stopped node process."""
        config = Spool.read_metadata(node_directory)
        expected_node_dir_name = node_dir_name(config["node"])
        expected_attempt_dir_name = attempt_dir_name(config["attempt"])
        if node_directory.name != expected_node_dir_name or node_directory.parent.name != expected_attempt_dir_name:
            raise ValueError("spool metadata does not match its node/attempt directory")
        aggregator = cls(
            [],
            node_directory.parent.parent,
            config["node"],
            config["attempt"],
            config["window_steps"],
            config["rank_timeout_s"],
            config["nodes_in_job"],
            primary_node=config["primary_node"],
            absent_ranks=config["absent_ranks"],
            rank_layouts={int(rank): Layout.from_json(layout) for rank, layout in config["layouts"].items()},
            capture_interval=config["capture_interval"],
        )
        return aggregator.flush()

    def poll_once(self) -> list[Path]:
        self._collector.collect_ready_slots()
        self._settle_pending_steps()
        written = self._publish_full_windows()
        self._dump_writer.write_ready_dumps()
        return written

    def flush(self) -> list[Path]:
        self._collector.collect_ready_slots()
        self._settle_pending_steps(force=True)
        written = self._publish_full_windows()
        if self._writer.buffered_step_count:
            written.extend(self._writer.write_window(self._writer.buffered_step_count))
        self._dump_writer.write_ready_dumps()
        return written

    def run(self, poll_interval_s: float, rank_pids: list[int], should_stop: Callable[[], bool] | None = None) -> None:
        should_stop = should_stop or (lambda: False)
        while True:
            self.poll_once()
            if self._collector.all_ranks_stopped() or not self._collector.any_rank_alive(rank_pids) or should_stop():
                break
            time.sleep(poll_interval_s)
        self.flush()

    def _common_stream_names(self) -> tuple[str, ...]:
        stream_names = next(iter(self._rank_layouts.values())).stream_names
        for rank, layout in self._rank_layouts.items():
            if layout.stream_names != stream_names:
                raise ValueError(
                    f"rank {rank}'s layout has stream_names {layout.stream_names!r}, expected {stream_names!r}"
                )
        return stream_names

    def _spool_metadata(self) -> dict[str, Any]:
        return {
            "node": self._node,
            "attempt": self._attempt,
            "window_steps": self._window_steps,
            "capture_interval": self._capture_interval,
            "rank_timeout_s": self._rank_timeout_s,
            "nodes_in_job": self._nodes_in_job,
            "primary_node": self._primary_node,
            "absent_ranks": self._absent_ranks,
            "layouts": {str(rank): layout.to_json() for rank, layout in self._rank_layouts.items()},
        }

    def _settle_pending_steps(self, force: bool = False) -> None:
        for step, pending in self._collector.settleable_steps(force):
            rows = self._merger.merge(step, pending.ranks, pending.verbose)
            self._writer.add_step(step, rows, self._collector.missing_ranks(pending))
            self._collector.settle(step)

    def _publish_full_windows(self) -> list[Path]:
        """The files of windows recovered at start-up plus those of every full window the settled steps make."""
        written, self._recovered_paths = self._recovered_paths, []
        while self._writer.buffered_step_count >= self._window_steps:
            written.extend(self._writer.write_window(self._window_steps))
        return written
