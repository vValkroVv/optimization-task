# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch
import torch.distributed as dist


_BLOCK = 128
_TILE_NUMEL = _BLOCK * _BLOCK


@dataclass(frozen=True)
class ExpertWeightSlice:
    projection: str
    shape: tuple[int, int, int]
    offset: int
    numel: int

    @property
    def scale_shape(self) -> tuple[int, int, int]:
        experts, rows, cols = self.shape
        return experts, rows // _BLOCK, cols // _BLOCK

    def views(
        self, data: torch.Tensor, scales: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            data.narrow(0, self.offset, self.numel)
            .view(torch.float8_e4m3fn)
            .view(self.shape),
            scales.narrow(
                0, self.offset // _TILE_NUMEL, self.numel // _TILE_NUMEL
            ).view(self.scale_shape),
        )


@dataclass(frozen=True)
class ExpertWeightLayout:
    global_slices: tuple[ExpertWeightSlice, ...]
    local_slices: tuple[ExpertWeightSlice, ...]
    global_numel: int
    local_numel: int

    @classmethod
    def for_flat_shards(
        cls,
        projections: tuple[tuple[str, tuple[int, int, int]], ...],
        shard_numel: int,
        shard_count: int,
        shard_rank: int,
    ) -> ExpertWeightLayout | None:
        """Select an unpadded rank-major layout with complete 128x128 tiles."""
        if tuple(name for name, _ in projections) != ("w13", "w2"):
            return None
        if shard_count <= 0 or not 0 <= shard_rank < shard_count or shard_numel <= 0:
            return None
        if any(
            experts <= 0 or rows <= 0 or cols <= 0 or rows % _BLOCK or cols % _BLOCK
            for _, (experts, rows, cols) in projections
        ):
            return None
        total = sum(prod(shape) for _, shape in projections)
        if total != shard_numel * shard_count:
            return None

        # Validate every rank's split, so eligibility does not depend on rank.
        # A split inside a projection must land on a whole block row. That
        # makes the rank-major concatenation valid for both data and scales.
        offset = 0
        local_start = shard_rank * shard_numel
        local_end = local_start + shard_numel
        global_slices = []
        local_slices = []
        for name, shape in projections:
            numel = prod(shape)
            end = offset + numel
            _, _, cols = shape
            first_boundary = (offset // shard_numel + 1) * shard_numel
            if any(
                (boundary - offset) % (_BLOCK * cols)
                for boundary in range(first_boundary, end, shard_numel)
            ):
                return None
            global_slices.append(ExpertWeightSlice(name, shape, offset, numel))
            start = max(local_start, offset)
            stop = min(local_end, end)
            if start < stop:
                local_shape = (1, (stop - start) // cols, cols)
                local_slices.append(
                    ExpertWeightSlice(
                        name, local_shape, start - local_start, stop - start
                    )
                )
            offset = end
        return cls(tuple(global_slices), tuple(local_slices), total, shard_numel)

    def receive_sizes_by_dtype(self) -> dict[torch.dtype, int]:
        if self.global_numel == self.local_numel:
            return {}
        return {
            torch.uint8: self.global_numel,
            torch.float32: self.global_numel // _TILE_NUMEL * torch.float32.itemsize,
        }


@dataclass
class ExpertAllGatherResult:
    all_gather_event: torch.Event
    work: tuple[dist.Work, ...]

    def wait(self) -> None:
        self.all_gather_event.wait()
        for handle in self.work:
            handle.wait()


class ExpertFP8WeightCache:
    """Persistent local planes plus optional views of a shared receive slot."""

    def __init__(
        self,
        layout: ExpertWeightLayout,
        master: torch.Tensor,
        buffers: dict[torch.dtype | None, torch.Tensor],
    ) -> None:
        self.layout = layout
        self.local_data = torch.empty(
            layout.local_numel, dtype=torch.uint8, device=master.device
        )
        self.local_scales = torch.empty(
            layout.local_numel // _TILE_NUMEL,
            dtype=torch.float32,
            device=master.device,
        )
        if layout.local_numel == layout.global_numel:
            self.output_data = self.local_data
            self.output_scales = self.local_scales
        else:
            sizes = layout.receive_sizes_by_dtype()
            self.output_data = buffers[torch.uint8][: sizes[torch.uint8]]
            self.output_scales = buffers[torch.float32][: sizes[torch.float32]].view(
                torch.float32
            )
        self._local_views = [
            (item, *item.views(self.local_data, self.local_scales))
            for item in layout.local_slices
        ]
        self.views = {
            item.projection: item.views(self.output_data, self.output_scales)
            for item in layout.global_slices
        }
        self.generation = 0
        self.prepared_generation = -1
        self.ready_event: torch.Event | None = None
        self.reader_event: torch.Event | None = None
        self.writer_event: torch.Event | None = None

    def resolve(
        self, projection: str, expected_generation: int | None
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if self.prepared_generation != self.generation:
            raise RuntimeError(
                "Expert FP8 weights are not prepared for this generation"
            )
        if expected_generation is not None and expected_generation != self.generation:
            raise RuntimeError(
                f"Expert FP8 weight generation changed from {expected_generation} "
                f"to {self.generation} during backward"
            )
        return (*self.views[projection], self.generation)

    def invalidate(self, writer_stream: torch.Stream) -> None:
        self.generation += 1
        for event in (self.ready_event, self.reader_event, self.writer_event):
            if event is not None:
                writer_stream.wait_event(event)

    def gather(
        self,
        master: torch.Tensor,
        group: dist.ProcessGroup,
        stream: torch.Stream,
        async_op: bool,
    ) -> ExpertAllGatherResult:
        from torchtitan.ops.float8.triton_kernels.block_quantization import (
            quantize_fp32_to_blockwise_fp8_,
        )

        with torch.cuda.stream(stream):
            if self.prepared_generation != self.generation:
                for event in (self.ready_event, self.reader_event, self.writer_event):
                    if event is not None:
                        stream.wait_event(event)
                with torch.profiler.record_function("GFSDP::fp8_expert_refresh"):
                    for item, data, scales in self._local_views:
                        source = master.narrow(0, item.offset, item.numel).view(
                            item.shape
                        )
                        quantize_fp32_to_blockwise_fp8_(source, data, scales)
                self.ready_event = stream.record_event()
                self.prepared_generation = self.generation

            assert self.ready_event is not None
            if self.output_data is self.local_data:
                return ExpertAllGatherResult(self.ready_event, ())
            stream.wait_event(self.ready_event)
            work = tuple(
                handle
                for result, source in (
                    (self.output_data, self.local_data),
                    (self.output_scales, self.local_scales),
                )
                if (
                    handle := dist.all_gather_into_tensor(
                        result, source, group=group, async_op=async_op
                    )
                )
                is not None
            )
            return ExpertAllGatherResult(stream.record_event(), work)
