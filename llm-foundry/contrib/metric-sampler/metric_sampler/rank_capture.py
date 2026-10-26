from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import torch

from metric_sampler.capture.dump import ActivationDumpConfigError, DumpCapture, DumpConfig
from metric_sampler.capture.dump_buffer import MAX_BUFFER_BYTES, DumpBuffer
from metric_sampler.capture.layout import Layout
from metric_sampler.capture.recorder import RankRecorder
from metric_sampler.capture.ring import ShmRing
from metric_sampler.node.process import NodeProcess

RING_SLOTS = 16
RANK_TIMEOUT_S = 120.0
POLL_INTERVAL_S = 0.5
ATTACH_DEADLINE_S = 300.0
NODE_STOP_TIMEOUT_S = 120.0


@dataclass(frozen=True)
class RankPlacement:
    """Where a rank sits in the job and the attempt it captures; global ranks are contiguous per node."""

    attempt: str
    node: str
    global_rank: int
    local_rank: int
    local_world_size: int
    node_rank: int
    world_size: int

    @property
    def _short_node(self) -> str:
        return self.node.split(".")[0]

    def ring_name(self, local_rank: int) -> str:
        return f"ms_{self.attempt}_{self._short_node}_{local_rank}"

    def dump_buffer_name(self, local_rank: int) -> str:
        return f"msd_{self.attempt}_{self._short_node}_{local_rank}"


class RankCapture:
    """One rank's end of capture, for any training framework: its claim on the attempt's output, its ring and
    recorder, its activation dump buffer and, on a node's first local rank, the node process that merges the node's
    rings into windows."""

    def __init__(self, output_dir: Path, placement: RankPlacement, layout: Layout, device: torch.device) -> None:
        self._output_dir = output_dir
        self._placement = placement
        self._device = device
        self._lock: BinaryIO | None = NodeProcess.acquire_rank_lock(
            output_dir, placement.node, placement.attempt, placement.global_rank
        )
        try:
            self.ring = ShmRing.create(placement.ring_name(placement.local_rank), layout, RING_SLOTS)
        except BaseException:
            self._lock.close()
            raise
        self.recorder: RankRecorder | None = RankRecorder(layout, self.ring, device)
        self.dump_buffer: DumpBuffer | None = None
        self.dump_capture: DumpCapture | None = None
        self.node_process: NodeProcess | None = None

    def start_dumps(self, config: DumpConfig, hidden_size: int) -> None:
        required_bytes = DumpBuffer.required_bytes(config.entries, config.tokens, hidden_size)
        if required_bytes > MAX_BUFFER_BYTES:
            raise ActivationDumpConfigError(
                f"activation_dump config needs {required_bytes} bytes per rank, over the "
                f"{MAX_BUFFER_BYTES}-byte hard limit; reduce tokens, points or layers"
            )
        placement = self._placement
        self.dump_buffer = DumpBuffer.create(
            placement.dump_buffer_name(placement.local_rank), config.entries, config.tokens, hidden_size
        )
        self.dump_capture = DumpCapture(config, self.dump_buffer, self._device, placement.global_rank)

    def stop_dumps(self) -> None:
        if self.dump_capture is not None:
            self.dump_capture.close()
        if self.dump_buffer is not None:
            self.dump_buffer.unlink()
        self.dump_capture = None
        self.dump_buffer = None

    def start_node(self, window_steps: int, capture_interval: int, dumps_enabled: bool) -> None:
        """Starts the node process on the node's first local rank; the node's other ranks have none."""
        placement = self._placement
        if placement.local_rank != 0:
            return
        first_rank = placement.node_rank * placement.local_world_size
        local_ranks = range(placement.local_world_size)
        self.node_process = NodeProcess.start(
            ring_names={first_rank + local_rank: placement.ring_name(local_rank) for local_rank in local_ranks},
            output_dir=self._output_dir,
            node=placement.node,
            attempt=placement.attempt,
            window_steps=window_steps,
            rank_timeout_s=RANK_TIMEOUT_S,
            nodes_in_job=placement.world_size // placement.local_world_size,
            # One writer for the attempt-level files: concurrent renames onto one path from every node fail on the
            # shared filesystem.
            primary_node=placement.node_rank == 0,
            poll_interval_s=POLL_INTERVAL_S,
            attach_deadline_s=ATTACH_DEADLINE_S,
            dump_buffer_names=(
                {first_rank + local_rank: placement.dump_buffer_name(local_rank) for local_rank in local_ranks}
                if dumps_enabled else {}
            ),
            capture_interval=capture_interval,
        )

    def node_running(self) -> bool:
        """False only when this rank started the node process and it has exited."""
        return self.node_process is None or self.node_process.is_running()

    def close(self) -> None:
        """Closes the dumps, the recorder and the ring, then stops the node process; the claim on the attempt is
        released even when the node process failed. Closing again does nothing."""
        self.stop_dumps()
        if self.recorder is not None:
            self.recorder.close()
            self.ring.unlink()
            self.recorder = None
        node_process, self.node_process = self.node_process, None
        try:
            if node_process is not None:
                node_process.stop(NODE_STOP_TIMEOUT_S)
        finally:
            if self._lock is not None:
                self._lock.close()
                self._lock = None
