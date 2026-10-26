from __future__ import annotations

import queue
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from metric_sampler.capture.dump_buffer import DTYPE_ITEMSIZES, MAX_ITEMSIZE, DtypeCode, DumpBuffer, DumpEntry
from metric_sampler.capture.host_registration import register_host_memory, unregister_host_memory
from metric_sampler.capture.ring import STATE_FREE
from metric_sampler.contract.metrics import NO_LAYER, NO_MICROBATCH, STANDALONE_ACTIVATION_POINTS

_TORCH_DTYPE_CODES = {
    torch.bfloat16: DtypeCode.BFLOAT16,
    torch.float16: DtypeCode.FLOAT16,
    torch.float32: DtypeCode.FLOAT32,
}
_OUTSIDE_BACKWARD = -1

# Points whose rows are tokens, hidden wide; not swiglu_in/down_in, whose rows are tokens permuted by the expert
# dispatcher.
DUMPABLE_POINTS = (
    "residual",
    "attn_out",
    "mlp_out",
    "final_norm",
    "shared_expert_out",
    "attn_residual_contribution",
    "mlp_residual_contribution",
    "embed",
)
_DEFAULT_TOKENS = 256
_DEFAULT_MAX_GB = 10.0
_BYTES_PER_GB = 1e9


class ActivationDumpConfigError(ValueError):
    """A bad activation dump config fails the run at its start instead of disabling capture."""


@dataclass(frozen=True)
class DumpSettings:
    """A run's activation dump settings: every `interval` steps (0: no dumps), the first `tokens` rows of
    microbatch 0 at each of `points` in each of `layers`, at most `max_gb` per rank and attempt."""

    interval: int
    points: tuple[str, ...]
    layers: tuple[int, ...]
    tokens: int
    max_gb: float

    @classmethod
    def parse(cls, config: Mapping[str, Any] | None) -> DumpSettings:
        """The settings of a run's `activation_dump` config; no config, or an interval of 0, disables dumps."""
        config = config or {}
        interval = int(config.get("interval") or 0)
        if interval <= 0:
            return cls(interval=0, points=(), layers=(), tokens=0, max_gb=0.0)
        points = tuple(config.get("points") or ())
        layers = tuple(config.get("layers") or ())
        tokens = int(config.get("tokens", _DEFAULT_TOKENS))
        max_gb = float(config.get("max_gb", _DEFAULT_MAX_GB))
        if not points or not layers:
            raise ActivationDumpConfigError("activation_dump.interval > 0 requires points and layers")
        unsupported = [point for point in points if point not in DUMPABLE_POINTS]
        if unsupported:
            raise ActivationDumpConfigError(
                f"activation_dump.points has unsupported points {unsupported}; only hidden-width "
                f"points are dumpable: {DUMPABLE_POINTS}"
            )
        if tokens <= 0:
            raise ActivationDumpConfigError(f"activation_dump.tokens must be positive, got {tokens}")
        if max_gb <= 0:
            raise ActivationDumpConfigError(f"activation_dump.max_gb must be positive, got {max_gb}")
        return cls(interval=interval, points=points, layers=layers, tokens=tokens, max_gb=max_gb)

    def entries(self, activation_layers: Mapping[str, Iterable[int]]) -> tuple[DumpEntry, ...]:
        """The (point, layer) entries of these settings that the model has: a layer-less point once, a decoder
        point in each configured layer that observes it."""
        entries = []
        for point in self.points:
            if point in STANDALONE_ACTIVATION_POINTS:
                entries.append(DumpEntry(point, NO_LAYER))
                continue
            valid_layers = set(activation_layers.get(point, ()))
            for layer in self.layers:
                if layer in valid_layers:
                    entries.append(DumpEntry(point, layer))
        if not entries:
            raise ActivationDumpConfigError(
                f"activation_dump.points={self.points!r} and layers={self.layers!r} produced no "
                "valid (point, layer) combination for this model"
            )
        return tuple(entries)

    def config(self, entries: tuple[DumpEntry, ...]) -> DumpConfig:
        return DumpConfig(
            interval=self.interval, entries=entries, tokens=self.tokens, max_bytes=int(self.max_gb * _BYTES_PER_GB)
        )


@dataclass(frozen=True)
class DumpConfig:
    """The static shape of one rank's activation dumps, fixed for the buffer's whole lifetime."""

    interval: int
    entries: tuple[DumpEntry, ...]
    tokens: int
    max_bytes: int


class DumpCapture:
    """Captures one rank's sampled activation dumps into its `DumpBuffer`.

    One dump is in flight at a time. `begin_step` decides whether the step captures a dump, by
    `config.interval` or because the node raised the buffer's request flag, and claims the buffer. `observe`,
    `observe_input_ids` and `observe_loss` fill it for microbatch 0 only, each with an asynchronous copy on a side
    CUDA stream. `step_end` marks the buffer READY once every queued copy has landed.
    """

    def __init__(self, config: DumpConfig, buffer: DumpBuffer, device: torch.device, global_rank: int) -> None:
        self._config = config
        self._buffer = buffer
        self._device = device
        self._global_rank = global_rank
        self._entry_index = {(entry.point, entry.layer): index for index, entry in enumerate(config.entries)}
        self._bytes_per_dump = len(config.entries) * config.tokens * buffer.hidden * MAX_ITEMSIZE
        self._bytes_written = 0
        self.dumps_skipped = 0
        self.budget_exhausted = False
        self._active = False
        self._step = -1
        self._microbatch = NO_MICROBATCH
        self._stream_ids: torch.Tensor | None = None
        self._pending_sources: list[torch.Tensor] = []

        self._token_ids_destination = torch.from_numpy(buffer.token_ids())
        self._token_loss_destination = torch.from_numpy(buffer.token_loss())
        self._stream_ids_destination = torch.from_numpy(buffer.stream_ids())
        self._entry_dtype_codes = buffer.entry_dtype_codes()
        self._entry_byte_destinations = [
            torch.from_numpy(buffer.entry_bytes(entry_index)) for entry_index in range(len(config.entries))
        ]

        if device.type == "cuda":  # pragma: no cover
            self._init_cuda_path()

    def begin_step(self, step: int) -> None:
        self._step = step
        self._active = False
        if self._config.interval <= 0:
            return
        if step % self._config.interval != 0 or self._global_rank != 0:
            return
        if self._buffer.state != STATE_FREE:
            self.dumps_skipped += 1
            return
        if self._bytes_written + self._bytes_per_dump > self._config.max_bytes:
            self.budget_exhausted = True
            self.dumps_skipped += 1
            return
        self._buffer.claim_for_writing()
        self._bytes_written += self._bytes_per_dump
        self._pending_sources = []
        self._active = True

    def begin_microbatch(self, microbatch: int, stream_ids: torch.Tensor | None) -> None:
        self._microbatch = microbatch
        self._stream_ids = stream_ids

    def _captures_microbatch(self) -> bool:
        return self._active and self._microbatch == 0

    def observe_input_ids(self, input_ids: torch.Tensor) -> None:
        if not self._captures_microbatch():
            return
        tokens = self._config.tokens
        flat = input_ids.detach().reshape(-1)
        if flat.numel() < tokens:
            raise ValueError(f"input_ids has {flat.numel()} tokens, fewer than the configured {tokens}")
        self._copy_async(self._token_ids_destination, flat[:tokens].to(torch.int32))

    def observe(self, point: str, layer: int, tensor: torch.Tensor) -> None:
        if not self._captures_microbatch():
            return
        entry_index = self._entry_index.get((point, layer))
        if entry_index is None:
            return
        # Skip a forward with gradients disabled (FSDP warm-up, eval) and the recompute of activation
        # checkpointing in backward, which would capture the same point twice.
        if not torch.is_grad_enabled():
            return
        if torch._C._current_graph_task_id() != _OUTSIDE_BACKWARD:
            return
        tokens = self._config.tokens
        hidden = tensor.shape[-1]
        activations = tensor.detach().reshape(-1, hidden)
        row_count = activations.shape[0]
        if row_count < tokens:
            raise ValueError(f"{point} layer {layer} has {row_count} rows, fewer than the configured {tokens}")
        dtype_code = _TORCH_DTYPE_CODES.get(activations.dtype)
        if dtype_code is None:
            raise ValueError(f"activation dump dtype {activations.dtype} is not supported")
        self._entry_dtype_codes[entry_index] = dtype_code
        itemsize = DTYPE_ITEMSIZES[dtype_code]
        destination_bytes = self._entry_byte_destinations[entry_index][: tokens * hidden * itemsize]
        destination = destination_bytes.view(activations.dtype).view(tokens, hidden)
        self._copy_async(destination, activations[:tokens])

    def observe_loss(self, loss: torch.Tensor, labels_valid: torch.Tensor) -> None:
        if not self._captures_microbatch():
            return
        tokens = self._config.tokens
        sequence_length = loss.shape[-1]
        main_loss = loss[0].detach().reshape(-1)
        main_valid = labels_valid[0].reshape(-1)
        if main_loss.numel() < tokens:
            raise ValueError(f"loss has {main_loss.numel()} tokens, fewer than the configured {tokens}")
        main_loss = main_loss[:tokens]
        main_valid = main_valid[:tokens]
        valid = main_valid & torch.isfinite(main_loss)
        loss_or_nan = torch.where(valid, main_loss, torch.full_like(main_loss, float("nan"))).to(torch.float32)
        self._copy_async(self._token_loss_destination, loss_or_nan)

        token_index = torch.arange(tokens, device=loss.device)
        sample_index = token_index // sequence_length
        if self._stream_ids is None:
            token_stream_ids = torch.zeros(tokens, dtype=torch.int32, device=loss.device)
        else:
            token_stream_ids = self._stream_ids.to(torch.int32)[sample_index]
        self._copy_async(self._stream_ids_destination, token_stream_ids)

    def step_end(self) -> None:
        if not self._active:
            return
        if self._device.type == "cuda":  # pragma: no cover
            self._step_end_cuda()
        else:
            self._buffer.mark_ready(self._step)
        self._active = False

    def close(self) -> None:
        if self._device.type == "cuda":  # pragma: no cover
            # Copies are issued during the forward pass, so a dump cut short by disable() may still
            # be writing into the mapping; it must land before the memory is unregistered and unmapped.
            self._side_stream.synchronize()
            self._join_marker_thread()
            # A later mapping at the same address could not be registered while this one is.
            unregister_host_memory(self._registered_payload)
        # These torch/numpy views hold buffer-protocol exports on the shm mapping, which must be
        # released before DumpBuffer.close() can unmap it.
        del self._token_ids_destination, self._token_loss_destination, self._stream_ids_destination
        del self._entry_dtype_codes, self._entry_byte_destinations
        self._buffer.close()

    def _copy_async(self, destination: torch.Tensor, source: torch.Tensor) -> None:
        if self._device.type == "cuda":  # pragma: no cover
            self._copy_async_cuda(destination, source)
        else:
            destination.copy_(source)

    def _init_cuda_path(self) -> None:  # pragma: no cover
        self._registered_payload = register_host_memory(self._buffer.payload_region())
        self._side_stream = torch.cuda.Stream()
        self._marker_queue: queue.SimpleQueue[tuple[torch.cuda.Event, int, list[torch.Tensor]] | None] = (
            queue.SimpleQueue()
        )
        self._marker_thread = threading.Thread(target=self._marker_loop, daemon=True)
        self._marker_thread.start()

    def _copy_async_cuda(self, destination: torch.Tensor, source: torch.Tensor) -> None:  # pragma: no cover
        side = self._side_stream
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            destination.copy_(source, non_blocking=True)
        source.record_stream(side)
        self._pending_sources.append(source)

    def _step_end_cuda(self) -> None:  # pragma: no cover
        event = torch.cuda.Event()
        event.record(self._side_stream)
        self._marker_queue.put((event, self._step, self._pending_sources))
        self._pending_sources = []

    def _marker_loop(self) -> None:  # pragma: no cover
        while True:
            item = self._marker_queue.get()
            if item is None:
                return
            # The source tensors stay referenced until their copies have landed.
            event, step, _sources = item
            event.synchronize()
            self._buffer.mark_ready(step)

    def _join_marker_thread(self) -> None:  # pragma: no cover
        self._marker_queue.put(None)
        self._marker_thread.join()
