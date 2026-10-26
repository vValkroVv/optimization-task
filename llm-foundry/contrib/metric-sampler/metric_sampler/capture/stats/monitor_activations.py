from __future__ import annotations

import functools
import itertools
import math
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NoReturn, Protocol

import numpy as np
import torch

from metric_sampler.capture.kernels import activation_ext
from metric_sampler.capture.recorder import CaptureCapacityError, RankRecorder, ResolvedRows
from metric_sampler.capture.stats.helpers.norms import (
    accumulate_row_norm_stats,
    accumulate_row_norms,
    accumulate_row_norms_batch,
)
from metric_sampler.contract.metrics import INT32_MAX

# Tensors the forward's monitored modules take in before their row norms are reduced together, which
# bounds the activations this keeps alive past their modules; the extension takes up to 32 per launch.
ROW_NORM_BATCH = 16
# What MonitorObservations is given for a tensor the monitor does not take, and for one the catalog lacks.
_NOT_TAKEN = -1
_NOT_IN_CATALOG = -2


class MonitorObservations(Protocol):
    """The extension's observers of one microbatch's monitored modules."""

    def observe_module(self, inputs: Sequence[object], output: object, slots: list[int]) -> None: ...

    def flush(self) -> int:
        """The catalog index of a tensor modified in place before its row norms were taken, or -1."""


@dataclass
class _MonitorActivations:
    """One microbatch's monitored activations: a float64 [catalog, 2] device accumulator of
    [absolute max, row norm sum] the extension updates in place, and the exact [observations, rows]
    counts on the host, as Python integers until the flush, made by the first Python count."""

    accumulator: torch.Tensor
    counts: list[list[int]] | None = None
    # On CUDA with the extension: its C++ observers of the microbatch's monitored modules, which observe their
    # forward tensors and gradients in place of the Python ones, and the [observations, rows] they counted, which
    # the flush adds to `counts`.
    observations: MonitorObservations | None = None
    extension_counts: torch.Tensor | None = None


class _GradientObserver:
    """Observes the gradient of one tensor under the name of each monitored module the tensor passed through:
    one hook, which reduces the gradient once, whatever the number of modules."""

    def __init__(self, stats: MonitorActivationStats, microbatch: int, name: str) -> None:
        self._stats = stats
        self._microbatch = microbatch
        self.names = [name]

    def __call__(self, gradient: torch.Tensor) -> None:
        self._stats.observe_gradient(self.names, self._microbatch, gradient)


class MonitorActivationStats:
    """The `catalog` names' activations of the training forwards' monitored module calls and of their gradients,
    as `monitor/activation_local` and `monitor/activation_local_observations` rows: the absolute max and the
    row-weighted mean last-axis L2 norm, with the exact observation and row counts. The inputs are observed always,
    the outputs with `log_output`, their gradients with `log_gradients` (the inputs' with `log_grad_input` too). A
    tensor of `skipped_type` (quantized) is not observed."""

    def __init__(
        self, recorder: RankRecorder, catalog: list[str], log_output: bool, log_gradients: bool, log_grad_input: bool,
        skipped_type: type[torch.Tensor],
    ) -> None:
        self._recorder = recorder
        self._catalog = catalog
        self._indices = {name: index for index, name in enumerate(catalog)}
        self._log_output = log_output
        self._log_gradients = log_gradients
        self._log_grad_input = log_grad_input
        self._skipped_type = skipped_type
        self._pending: dict[int, _MonitorActivations] = {}
        # id(tensor) -> (the tensor, its version when observed, the catalog indices it is taken into) of the
        # tensors the forward in progress has observed and not reduced yet; the indices are those of the
        # slots they are queued for.
        self._pending_row_norms: dict[int, tuple[torch.Tensor, int, list[int]]] = {}
        self._pending_row_norm_slots: set[int] = set()
        # id(tensor) -> (a weak reference to the tensor, the observer of its gradient) for the forward in progress.
        self._gradient_observers: dict[int, tuple[weakref.ref[torch.Tensor], _GradientObserver]] = {}
        # (module name, inputs, outputs) -> the catalog slots of a call of the module, as MonitorObservations
        # takes them.
        self._module_slots: dict[tuple[str, int, int], list[int]] = {}
        self._rows: dict[tuple[str, int], ResolvedRows] = {}

    def observe_module(
        self, name: str, microbatch: int, inputs: Sequence[object], output: torch.Tensor | Sequence[object]
    ) -> MonitorObservations | None:
        """Observes one call of the monitored module `name`. Returns the microbatch's MonitorObservations when this
        call made them: they take the module calls of the rest of the forward without Python."""
        outputs = (output,) if isinstance(output, torch.Tensor) else output
        pending = self._pending.get(microbatch)
        created = None
        if pending is None:
            reference = next((value for value in (*inputs, *outputs)
                              if isinstance(value, torch.Tensor) and value.is_floating_point()), None)
            extension = activation_ext.prebuilt() if reference is not None and reference.is_cuda else None
            if reference is not None and extension:  # pragma: no cover
                pending = self._pending_for(microbatch, reference)
                pending.extension_counts = torch.zeros(pending.accumulator.shape, dtype=torch.int64)
                pending.observations = created = extension.MonitorObservations(
                    pending.accumulator, pending.extension_counts, ROW_NORM_BATCH,
                    functools.partial(self._reduce_tensor, pending.accumulator),
                    functools.partial(self._observe_watched_gradient, microbatch), self._skipped_type,
                )
        if pending is not None and pending.observations is not None:  # pragma: no cover
            # One extension call takes the module's tensors: no Python call per tensor.
            pending.observations.observe_module(inputs, output, self.module_slots(name, inputs, outputs))
            return created
        self._observe_tensors_in_python(name, microbatch, inputs, outputs)
        return None

    def _observe_tensors_in_python(
        self, name: str, microbatch: int, inputs: Sequence[object], outputs: Sequence[object]
    ) -> None:
        for index, value in enumerate(inputs):
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                continue
            self._observe_forward_tensor(f"{name}:input.{index}", microbatch, value)
            if self._log_gradients and self._log_grad_input and value.requires_grad:
                self._watch_gradient(f"{name}:grad_input.{index}", microbatch, value)
        for index, value in enumerate(outputs):
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                continue
            if self._log_output:
                self._observe_forward_tensor(f"{name}:output.{index}", microbatch, value)
            if self._log_gradients and value.requires_grad:
                self._watch_gradient(f"{name}:grad_output.{index}", microbatch, value)

    def module_slots(
        self, name: str, inputs: Sequence[object], outputs: Sequence[object]
    ) -> list[int]:  # pragma: no cover
        """The catalog slots of a call of the module as MonitorObservations takes them: per input, then per output,
        its forward observation's and its gradient's, with `_NOT_TAKEN` for one the monitor does not take and
        `_NOT_IN_CATALOG` for one the catalog lacks. Built from the module's first call with these numbers of
        inputs and outputs, which looks up the names the Python observers look up."""
        key = (name, len(inputs), len(outputs))
        slots = self._module_slots.get(key)
        if slots is None:
            slots = []
            for kind, values, observed, watched in (
                ("input", inputs, True, self._log_gradients and self._log_grad_input),
                ("output", outputs, self._log_output, self._log_gradients),
            ):
                for index, value in enumerate(values):
                    if isinstance(value, torch.Tensor) and value.is_floating_point():
                        forward = not isinstance(value, self._skipped_type) and value.numel() > 0
                        gradient = value.requires_grad
                    else:
                        forward = gradient = False
                    slots.append(self._catalog_slot(f"{name}:{kind}.{index}", observed, forward))
                    slots.append(self._catalog_slot(f"{name}:grad_{kind}.{index}", watched, gradient))
            self._module_slots[key] = slots
        return slots

    def end_forward(self, microbatch: int) -> None:
        """Reduces what the forward left queued; the gradients of its tensors stay watched by their hooks."""
        self._flush_row_norms(microbatch)
        self._gradient_observers.clear()

    def flush(self, microbatch: int) -> None:
        """Records the microbatch's rows once its backward is done: NaN statistics for a name without
        observations."""
        entries = len(self._catalog)
        pending = self._pending.pop(microbatch, None)
        accumulator = None if pending is None else pending.accumulator
        if pending is None or pending.counts is None:
            counts = np.zeros((entries, 2), dtype=np.int64)
        else:
            # Flattened in C: numpy converts a list of pairs element by element, several times slower.
            counts = np.fromiter(
                itertools.chain.from_iterable(pending.counts), dtype=np.int64, count=2 * entries
            ).reshape(entries, 2)
        if pending is not None and pending.extension_counts is not None:  # pragma: no cover
            counts += pending.extension_counts.numpy()
            if (counts > INT32_MAX).any():
                raise CaptureCapacityError("activation observation counts exceed Int32")
        self._recorder.record_host_rows(self._rows_for("monitor/activation_local_observations", microbatch), counts)
        value_rows = self._rows_for("monitor/activation_local", microbatch)
        extension = activation_ext.prebuilt() if accumulator is not None and accumulator.is_cuda else None
        if accumulator is None:
            values = torch.full((entries, 2), float("nan"), device=self._recorder.buffer.device)
        elif extension:  # pragma: no cover
            # One extension call copies the rows and divides.
            self._recorder.mark_rows_recorded(value_rows)
            extension.monitor_values_into(
                accumulator, torch.from_numpy(counts), self._recorder.float_buffer(), value_rows.word_positions)
            return
        else:
            # Unobserved entries have no rows: NaN for both statistics.
            row_counts = torch.from_numpy(np.where(counts[:, 0] > 0, np.maximum(counts[:, 1], 1), np.nan))
            if accumulator.is_cuda:  # pragma: no cover
                row_counts = row_counts.pin_memory().to(accumulator.device, non_blocking=True)
            absolute_max = torch.where(row_counts.isnan(), row_counts, accumulator[:, 0])
            values = torch.stack((absolute_max, accumulator[:, 1] / row_counts), dim=1).float()
        self._recorder.record_rows(value_rows, values)

    def observe_gradient(self, names: list[str], microbatch: int, gradient: torch.Tensor) -> None:
        # The ordinary ActivationMonitor skips FP8-quantized inputs too.
        if isinstance(gradient, self._skipped_type) or gradient.numel() == 0:
            return
        index = self._count_observation(names[0], microbatch, gradient)
        accumulator = self._pending[microbatch].accumulator
        # The reductions read the gradient without joining an autograd graph: no no_grad scope, which
        # would cost more than the extension call.
        stats = accumulate_row_norms(gradient, accumulator, index, keep_stats=len(names) > 1)
        if stats is not None:
            for name in names[1:]:
                accumulate_row_norm_stats(stats, accumulator, self._count_observation(name, microbatch, gradient))

    def _observe_forward_tensor(self, name: str, microbatch: int, value: torch.Tensor) -> None:
        # The row norms of the tensors a forward observes are taken in batches, and one tensor feeding
        # several monitored modules (the q, k and v projections) is reduced once.
        if isinstance(value, self._skipped_type) or value.numel() == 0:
            return
        index = self._count_observation(name, microbatch, value)
        queued = self._pending_row_norms.get(id(value))
        # A slot is taken into once per batch, so that the observations of it keep their order.
        if index in self._pending_row_norm_slots or (queued is None and len(self._pending_row_norms) == ROW_NORM_BATCH):
            self._flush_row_norms(microbatch)
            queued = None
        if queued is None:
            self._pending_row_norms[id(value)] = (value, value._version, [index])
        else:
            queued[2].append(index)
        self._pending_row_norm_slots.add(index)

    def _catalog_slot(self, name: str, taken: bool, looked_up: bool) -> int:  # pragma: no cover
        if not taken:
            return _NOT_TAKEN
        return self._indices[name] if looked_up else self._indices.get(name, _NOT_IN_CATALOG)

    def _flush_row_norms(self, microbatch: int) -> None:
        """Reduces the queued tensors and takes them into their slots of the microbatch's accumulator."""
        pending = self._pending.get(microbatch)
        if pending is not None and pending.observations is not None:  # pragma: no cover
            modified = pending.observations.flush()
            if modified >= 0:
                self._raise_modified_in_place(modified)
        queued = list(self._pending_row_norms.values())
        self._pending_row_norms.clear()
        self._pending_row_norm_slots.clear()
        for value, version, indices in queued:
            if value._version != version:
                self._raise_modified_in_place(indices[0])
        if queued:
            accumulate_row_norms_batch(
                [value for value, _, _ in queued], self._pending[microbatch].accumulator,
                [indices for _, _, indices in queued],
            )

    def _raise_modified_in_place(self, entry: int) -> NoReturn:
        raise RuntimeError(f"{self._catalog[entry]} was modified in place before its row norms were taken")

    def _watch_gradient(self, name: str, microbatch: int, value: torch.Tensor) -> None:
        # The gradient of a tensor several monitored modules saw is observed under each name by one hook.
        watched = self._gradient_observers.get(id(value))
        if watched is None or watched[0]() is not value:
            observer = _GradientObserver(self, microbatch, name)
            self._gradient_observers[id(value)] = (weakref.ref(value), observer)
            value.register_hook(observer)
        else:
            watched[1].names.append(name)

    def _reduce_tensor(
        self, accumulator: torch.Tensor, value: torch.Tensor, slots: list[int]
    ) -> None:  # pragma: no cover
        # A forward tensor MonitorObservations does not reduce, taken as the Python observer takes it.
        accumulate_row_norms_batch([value], accumulator, [slots])

    def _observe_watched_gradient(
        self, microbatch: int, slots: list[int], gradient: torch.Tensor
    ) -> None:  # pragma: no cover
        # A gradient MonitorObservations does not reduce, taken as the Python observer takes it.
        self.observe_gradient([self._catalog[slot] for slot in slots], microbatch, gradient)

    def _count_observation(self, name: str, microbatch: int, value: torch.Tensor) -> int:
        """Counts one observation of `name` and its last-axis rows (a scalar is one row) in the
        microbatch's population, creating the microbatch's accumulator; returns the catalog index."""
        index = self._indices[name]
        pending = self._pending_for(microbatch, value)
        if pending.counts is None:
            pending.counts = [[0, 0] for _ in self._catalog]
        counts = pending.counts[index]
        counts[0] += 1
        counts[1] += value.numel() // value.shape[-1] if value.ndim else 1
        if counts[0] > INT32_MAX or counts[1] > INT32_MAX:
            raise CaptureCapacityError("activation observation counts exceed Int32")
        return index

    def _pending_for(self, microbatch: int, reference: torch.Tensor) -> _MonitorActivations:
        """The microbatch's monitored activations, created on the device of its first observed tensor."""
        pending = self._pending.get(microbatch)
        if pending is None:
            accumulator = torch.zeros((len(self._catalog), 2), dtype=torch.float64, device=reference.device)
            accumulator[:, 0] = -math.inf
            pending = self._pending[microbatch] = _MonitorActivations(accumulator)
        return pending

    def _rows_for(self, metric_name: str, microbatch: int) -> ResolvedRows:
        rows = self._rows.get((metric_name, microbatch))
        if rows is None:
            rows = self._recorder.resolve_rows(metric_name, range(len(self._catalog)), microbatch)
            self._rows[metric_name, microbatch] = rows
        return rows
