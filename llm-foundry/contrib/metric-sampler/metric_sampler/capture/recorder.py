from __future__ import annotations

import queue
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from metric_sampler.capture.host_registration import register_host_memory, unregister_host_memory
from metric_sampler.capture.layout import Layout, Slot
from metric_sampler.capture.ring import ShmRing
from metric_sampler.capture.shm import WORD_BYTES
from metric_sampler.contract.metrics import ValueType

_MILLISECONDS_PER_SECOND = 1000.0


class GpuEvent(Protocol):
    def elapsed_time(self, other: GpuEvent) -> float: ...


class CaptureCapacityError(BufferError):
    """Capture cannot accept an observation or completed step without losing statistics."""


@dataclass(frozen=True)
class ResolvedSlot:
    """A slot's offset, width and flags resolved once for a caller with a fixed (metric, layer, microbatch),
    which avoids the `Layout` lookups `accumulator` repeats on every call."""

    offset: int
    width: int
    slot_index: int
    detailed: bool
    value_type: ValueType


@dataclass(frozen=True, eq=False)
class ResolvedRows:
    """Same-width slots of one metric for many layers, resolved once so that `record_rows` writes all of them
    with a single indexed copy instead of one copy per slot."""

    word_positions: torch.Tensor
    offsets: np.ndarray
    slot_indices: np.ndarray
    width: int
    detailed: bool
    value_type: ValueType


@dataclass(frozen=True)
class _GpuDuration:
    # Elapsed time is only known once both events have completed, so it is resolved in step_end.
    offset: int
    start: GpuEvent
    end: GpuEvent

    def to_overlay_word(self) -> tuple[np.ndarray, np.ndarray]:
        seconds = self.start.elapsed_time(self.end) / _MILLISECONDS_PER_SECOND
        words = np.zeros(1, dtype=np.int32)
        words.view(np.float32)[:] = seconds
        return np.array([self.offset]), words


class RankRecorder:
    """Copies a rank's metrics into one flat buffer and hands a step off to the ring.

    A full ring fails explicitly; a completed step is never silently discarded."""

    def __init__(self, layout: Layout, ring: ShmRing, device: torch.device) -> None:
        self._layout = layout
        self._ring = ring
        self._device = device
        # Two buffers alternate across steps so that recording the next step never overwrites data that the
        # device-to-host copy of the previous step may still be reading.
        self._buffers = [torch.zeros(ring.payload_words, dtype=torch.int32, device=device) for _ in range(2)]
        # int32 and float32 words have the same size, so each buffer's float32 view is taken once.
        self._float_buffers = [buffer.view(torch.float32) for buffer in self._buffers]
        self._active = 0
        # Per buffer, the slot views made so far (see view and _slot_view).
        self._resolved_views: list[dict[int, torch.Tensor]] = [{}, {}]
        self._slot_views: list[dict[tuple[str, int, int], torch.Tensor]] = [{}, {}]
        self._last_copy_event: list[torch.cuda.Event | None] = [None, None]
        self._host_slots = [torch.from_numpy(ring.payload(i)) for i in range(ring.slot_count)]
        self._host_staging = np.zeros(ring.payload_words, dtype=np.int32)
        # Word positions of the step's host-staged values, overlaid on its copy in one scatter.
        self._staged_positions: list[np.ndarray] = []
        self._mask_words = ring.mask_words
        self._recorded = np.zeros(len(layout.slots), dtype=bool)
        self._gpu_durations: list[_GpuDuration] = []
        self._detailed = False
        self.dropped_steps = 0
        self._dropped_since_last = 0
        self._seq = 0
        if device.type == "cuda":  # pragma: no cover
            self._init_cuda_path()

    @property
    def buffer(self) -> torch.Tensor:
        return self._buffers[self._active]

    @property
    def ring_used_slots(self) -> int:
        return self._ring.used_slots

    @property
    def ring_capacity_slots(self) -> int:
        return self._ring.slot_count

    def float_buffer(self) -> torch.Tensor:
        """The active buffer's float32 view, stable for the duration of one step like `buffer`."""
        return self._float_buffers[self._active]

    def begin_step(self, detailed: bool) -> None:
        """Zeroes the active buffer so metrics this step never records keep an older value."""
        self.buffer.zero_()
        self._staged_positions = []
        self._gpu_durations = []
        self._recorded[:] = False
        self._detailed = detailed

    def _check_detailed(self, slot: Slot) -> None:
        if slot.metric.detailed and not self._detailed:
            raise ValueError(f"{slot.metric.name} is a detailed-mode metric but the step is not detailed")

    def _scalar_slot(self, metric_name: str, layer: int, microbatch: int) -> Slot:
        slot = self._layout.slot(metric_name, layer, microbatch)
        self._check_detailed(slot)
        if slot.width != 1:
            raise ValueError(f"{metric_name} is not a scalar (width {slot.width})")
        return slot

    def resolve(self, metric_name: str, layer: int, microbatch: int) -> ResolvedSlot:
        """The slot's layout as a value to resolve once and reuse."""
        slot = self._layout.slot(metric_name, layer, microbatch)
        return ResolvedSlot(
            offset=slot.offset,
            width=slot.width,
            slot_index=self._layout.slot_index(metric_name, layer, microbatch),
            detailed=slot.metric.detailed,
            value_type=slot.metric.value_type,
        )

    def resolve_rows(self, metric_name: str, layers: Sequence[int], microbatch: int) -> ResolvedRows:
        if not layers:
            raise ValueError(f"{metric_name} has no rows to resolve")
        slots = [self._layout.slot(metric_name, layer, microbatch) for layer in layers]
        width = slots[0].width
        if any(slot.width != width for slot in slots):
            raise ValueError(f"{metric_name} rows must share one width")
        offsets = np.array([slot.offset for slot in slots], dtype=np.int64)
        return ResolvedRows(
            word_positions=(torch.from_numpy(offsets)[:, None] + torch.arange(width)).reshape(-1).to(self._device),
            offsets=offsets,
            slot_indices=np.array([self._layout.slot_index(metric_name, layer, microbatch) for layer in layers]),
            width=width,
            detailed=slots[0].metric.detailed,
            value_type=slots[0].metric.value_type,
        )

    def record_rows(self, rows: ResolvedRows, values: torch.Tensor) -> None:
        """Writes row i of `values` into the i-th resolved slot."""
        if values.shape != (len(rows.slot_indices), rows.width):
            raise ValueError(
                f"expected {len(rows.slot_indices)} rows of {rows.width} values, got {tuple(values.shape)}"
            )
        self.mark_rows_recorded(rows)
        buffer = self.float_buffer() if rows.value_type is ValueType.FLOAT32 else self.buffer
        buffer.index_put_((rows.word_positions,), values.detach().reshape(-1).to(buffer.dtype))

    def mark_rows_recorded(self, rows: ResolvedRows) -> None:
        """Marks resolved rows written through their offsets (as `mark_recorded` does slots)."""
        if rows.detailed and not self._detailed:
            raise ValueError("detailed-mode rows recorded outside a detailed step")
        self._recorded[rows.slot_indices] = True

    def record_host_words(self, metric_name: str, layer: int, microbatch: int, words: Sequence[int]) -> None:
        """Stages an integer slot's 32-bit words from host values (wrapped to int32), overlaid on
        the step's copy like `record_host`: no host-to-device copy, so no stream synchronization."""
        slot = self._layout.slot(metric_name, layer, microbatch)
        self._check_detailed(slot)
        if slot.metric.value_type is not ValueType.INT32:
            raise ValueError(f"{metric_name} does not hold integer words")
        if len(words) != slot.width:
            raise ValueError(f"{metric_name} expected {slot.width} words, got {len(words)}")
        self._host_staging[slot.offset : slot.offset + slot.width] = np.asarray(words, dtype=np.int64).astype(np.int32)
        self._staged_positions.append(np.arange(slot.offset, slot.offset + slot.width))
        self._recorded[self._layout.slot_index(metric_name, layer, microbatch)] = True

    def record_host_rows(self, rows: ResolvedRows, words: np.ndarray) -> None:
        """`record_host_words` for every resolved slot at once; row i of `words` is slot i."""
        if rows.detailed and not self._detailed:
            raise ValueError("detailed-mode rows recorded outside a detailed step")
        if rows.value_type is not ValueType.INT32:
            raise ValueError("host rows must hold integer words")
        if words.shape != (len(rows.slot_indices), rows.width):
            raise ValueError(f"expected {len(rows.slot_indices)} rows of {rows.width} words, got {words.shape}")
        targets = (rows.offsets[:, None] + np.arange(rows.width)).reshape(-1)
        self._host_staging[targets] = words.reshape(-1).astype(np.int64).astype(np.int32)
        self._staged_positions.append(targets)
        self._recorded[rows.slot_indices] = True

    def recorded_flags(self) -> torch.Tensor:
        """The recorded flag of every slot, a host bool tensor sharing their memory, for the extension's
        observers to mark the slots they write."""
        return torch.from_numpy(self._recorded)

    def mark_recorded(self, slot_indices: Sequence[int]) -> None:
        """Marks slots written via a resolved offset as recorded. All-or-nothing: if any index is
        a detailed-mode metric outside a detailed step, none of them are marked. Detailed slots come
        after every other one, so only an index from the first detailed slot on can fail."""
        last = max(slot_indices, default=-1)
        if not self._detailed and last >= self._layout.non_detailed_slots:
            self._check_detailed(self._layout.slots[last])
        recorded = self._recorded
        for index in slot_indices:
            recorded[index] = True

    def view(self, slot: ResolvedSlot) -> torch.Tensor:
        """The slot's words typed as its metric's values, a view of the active buffer. Slicing the
        buffer is a tensor operation on every call, so each buffer's view of a slot is made once."""
        views = self._resolved_views[self._active]
        view = views.get(slot.slot_index)
        if view is None:
            buffer = self.float_buffer() if slot.value_type is ValueType.FLOAT32 else self.buffer
            view = views[slot.slot_index] = buffer[slot.offset : slot.offset + slot.width]
        return view

    def _slot_view(self, metric_name: str, layer: int, microbatch: int) -> torch.Tensor:
        views = self._slot_views[self._active]
        key = (metric_name, layer, microbatch)
        view = views.get(key)
        if view is None:
            view = views[key] = self.view(self.resolve(metric_name, layer, microbatch))
        return view

    def accumulator(self, metric_name: str, layer: int, microbatch: int) -> torch.Tensor:
        """A typed view of the slot of the active buffer for in-place accumulation (e.g. `index_add_`)."""
        self.mark_recorded([self._layout.slot_index(metric_name, layer, microbatch)])
        return self._slot_view(metric_name, layer, microbatch)

    def record(self, metric_name: str, layer: int, microbatch: int, tensor: torch.Tensor) -> None:
        slot = self._layout.slot(metric_name, layer, microbatch)
        self._check_detailed(slot)
        if tensor.numel() != slot.width:
            raise ValueError(f"{metric_name} expected {slot.width} values, got {tensor.numel()}")
        # Copying a tensor that requires grad would join the buffer to its autograd graph. `copy_`
        # broadcasts a scalar and takes a vector as it is; only a higher rank needs flattening.
        if tensor.requires_grad:
            tensor = tensor.detach()
        if tensor.ndim > 1:
            tensor = tensor.flatten()
        self._slot_view(metric_name, layer, microbatch).copy_(tensor)
        self._recorded[self._layout.slot_index(metric_name, layer, microbatch)] = True

    def record_host(self, metric_name: str, layer: int, microbatch: int, value: float) -> None:
        # Bypasses the device buffer: a pageable-host-to-device copy can serialize with the copy
        # stream, so the scalar is staged here and overlaid onto the slot after that copy lands.
        slot = self._scalar_slot(metric_name, layer, microbatch)
        words = self._host_staging[slot.offset : slot.offset + slot.width]
        if slot.metric.value_type is ValueType.FLOAT32:
            words.view(np.float32)[:] = value
        else:
            words[:] = int(value)
        self._staged_positions.append(np.arange(slot.offset, slot.offset + slot.width))
        self._recorded[self._layout.slot_index(metric_name, layer, microbatch)] = True

    def record_gpu_duration(
        self, metric_name: str, layer: int, microbatch: int, start: GpuEvent, end: GpuEvent
    ) -> None:
        """Resolved lazily in `step_end`, once the copy that reads it has completed."""
        slot = self._scalar_slot(metric_name, layer, microbatch)
        self._gpu_durations.append(_GpuDuration(slot.offset, start, end))
        self._recorded[self._layout.slot_index(metric_name, layer, microbatch)] = True

    def _pack_recorded_mask(self) -> np.ndarray:
        packed = np.zeros(self._mask_words * WORD_BYTES, dtype=np.uint8)
        bits = np.packbits(self._recorded, bitorder="little")
        packed[: bits.size] = bits
        return packed.view(np.int32)

    def step_end(self, step: int) -> bool:
        slot_index = self._ring.claim_free_slot()
        if slot_index is None:
            raise CaptureCapacityError(
                f"metric sampler ring is full at step {step}; statistics remain in the device buffer. "
                "Check the node aggregator and its output filesystem before resuming."
            )

        dropped_before = self._dropped_since_last
        self._dropped_since_last = 0
        seq = self._seq
        self._seq += 1

        active_buffer = self.buffer
        # The last word of a payload holds the step number. A fill: assigning the int would copy it from pageable
        # host memory, which waits for the queued step.
        active_buffer[-1:].fill_(step)
        staged = np.concatenate(self._staged_positions) if self._staged_positions else np.empty(0, dtype=np.int64)
        mask = self._pack_recorded_mask()
        overlay = [
            (staged, self._host_staging[staged]),
            (np.arange(self._layout.words, self._layout.words + mask.size), mask),
        ]
        copy_words = self._layout.words if self._detailed else self._layout.non_detailed_words

        if self._device.type == "cuda":  # pragma: no cover
            self._step_end_cuda(
                active_buffer, slot_index, step, seq, dropped_before, copy_words, overlay, self._gpu_durations
            )
        else:
            overlay = overlay + [duration.to_overlay_word() for duration in self._gpu_durations]
            self._host_slots[slot_index][:copy_words].copy_(active_buffer[:copy_words])
            self._host_slots[slot_index][-1:].copy_(active_buffer[-1:])
            self._apply_overlay(slot_index, overlay)
            self._ring.mark_ready(slot_index, step, seq, dropped_before)
        self._active = 1 - self._active
        return True

    def _apply_overlay(self, slot_index: int, overlay: list[tuple[np.ndarray, np.ndarray]]) -> None:
        # Each entry is word positions and their values: one vectorized scatter apiece.
        payload = self._host_slots[slot_index].numpy()
        for positions, words in overlay:
            payload[positions] = words

    def verbose(self, step: int) -> bool:
        return self._ring.verbose_until_step >= step

    def close(self) -> None:
        if self._device.type == "cuda":  # pragma: no cover
            self._join_marker_thread()
            # A later mapping at the same address could not be registered while this one is.
            unregister_host_memory(self._registered_payload)
        self._ring.request_stop()
        del self._host_slots
        self._ring.close()

    def _init_cuda_path(self) -> None:  # pragma: no cover
        self._registered_payload = register_host_memory(self._ring.payload_region())
        self._side_stream = torch.cuda.Stream()
        # The marker thread waits for each step's copy to land, overlays the host-staged values on it and only then
        # publishes the slot to the node.
        self._marker_queue: queue.SimpleQueue[
            tuple[torch.cuda.Event, int, int, int, int, list[tuple[np.ndarray, np.ndarray]], list[_GpuDuration]] | None
        ] = queue.SimpleQueue()
        self._marker_thread = threading.Thread(target=self._marker_loop, daemon=True)
        self._marker_thread.start()

    def _marker_loop(self) -> None:  # pragma: no cover
        while True:
            item = self._marker_queue.get()
            if item is None:
                return
            event, slot_index, step, seq, dropped_before, overlay, gpu_durations = item
            event.synchronize()
            overlay = overlay + [duration.to_overlay_word() for duration in gpu_durations]
            self._apply_overlay(slot_index, overlay)
            self._ring.mark_ready(slot_index, step, seq, dropped_before)

    def _step_end_cuda(
        self,
        active_buffer: torch.Tensor,
        slot_index: int,
        step: int,
        seq: int,
        dropped_before: int,
        copy_words: int,
        overlay: list[tuple[np.ndarray, np.ndarray]],
        gpu_durations: list[_GpuDuration],
    ) -> None:  # pragma: no cover
        side = self._side_stream
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            self._host_slots[slot_index][:copy_words].copy_(active_buffer[:copy_words], non_blocking=True)
            self._host_slots[slot_index][-1:].copy_(active_buffer[-1:], non_blocking=True)
            event = torch.cuda.Event()
            event.record(side)
        active_buffer.record_stream(side)
        self._marker_queue.put((event, slot_index, step, seq, dropped_before, overlay, gpu_durations))

        # The next step writes the other buffer, whose previous copy must have finished reading it.
        next_active = 1 - self._active
        prior_event = self._last_copy_event[next_active]
        if prior_event is not None:
            torch.cuda.current_stream().wait_event(prior_event)
        self._last_copy_event[self._active] = event

    def _join_marker_thread(self) -> None:  # pragma: no cover
        self._marker_queue.put(None)
        self._marker_thread.join()
