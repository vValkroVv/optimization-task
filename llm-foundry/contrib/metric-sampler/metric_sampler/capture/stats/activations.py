from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from types import ModuleType
from typing import Protocol

import torch

from metric_sampler.capture.kernels import activation_ext
from metric_sampler.capture.recorder import CaptureCapacityError, RankRecorder, ResolvedSlot
from metric_sampler.capture.stats.helpers.histograms import bin_indices
from metric_sampler.capture.stats.helpers.norms import SUPPORTED_DTYPES
from metric_sampler.capture.stats.quantiles import ExactQuantileStats
from metric_sampler.contract.metrics import ACTIVATION_COUNTS_WIDTH, ACTIVATION_LOG2_BINS, INT32_MAX

# The width of each slot of an activation point, by the suffix of its metric name, in the order of the layout.
_POINT_SLOT_WIDTHS = {"moments": 4, "counts": ACTIVATION_COUNTS_WIDTH, "amax": 1, "power_sums": 2}
# The floor(log2) given to a zero: far below the first bin of any grid.
_ZERO_LOG2 = -1000
# torch's graph task id outside a backward pass.
_NO_GRAPH_TASK = -1


def point_slots(point: str) -> tuple[tuple[str, int], ...]:
    """The slots one activation point records per layer and microbatch, as (metric name, width)."""
    return tuple((f"act/{point}/{suffix}", width) for suffix, width in _POINT_SLOT_WIDTHS.items())


def _floor_log2(magnitudes: torch.Tensor) -> torch.Tensor:
    # floor(log2 m), exact via frexp (matches the CUDA kernel).
    _, exponent = torch.frexp(magnitudes)
    finite_log2 = torch.where(magnitudes > 0, exponent - 1, torch.full_like(exponent, _ZERO_LOG2)).to(torch.float32)
    return torch.where(torch.isinf(magnitudes), magnitudes, finite_log2)


def _last_axis_rows(tensor: torch.Tensor) -> torch.Tensor:
    # Detach keeps this out of the autograd graph so nothing is saved for backward; reshape's copy
    # for a non-contiguous input (e.g. MoE dispatch) is deliberate.
    return tensor.detach().reshape(-1, tensor.shape[-1])


@dataclass
class _ObservationPoint:
    moments: ResolvedSlot
    counts: ResolvedSlot
    absolute_max: ResolvedSlot
    power_sums: ResolvedSlot
    slot_indices: list[int]
    # Present when quantiles are recorded: the extension writes them in the statistics call.
    quantiles: ResolvedSlot | None = None
    quantile_counts: ResolvedSlot | None = None
    # The quantile metric, every slot the extension's statistics call writes and the call's offset and bin
    # arguments, made once per point.
    quantile_metric: str = ""
    quantile_slot_indices: list[int] = field(default_factory=list)
    quantile_arguments: tuple[int, ...] = ()
    # The step `elements` counts in, and the elements the point has observed in it.
    step: int = -1
    elements: int = 0


class _NativeObserver(Protocol):
    def begin_step(self, buffer: torch.Tensor) -> None: ...

    def begin_microbatch(self, microbatch: int) -> None: ...

    def flush(self) -> None: ...


class ActivationStats:
    """Per-microbatch activation statistics (moments, log2 histograms, absolute max), computed in one
    memory pass into the recorder's preallocated slots, excluding non-finite elements. With
    `quantiles`, each observation's exact quantiles are recorded too, the CUDA extension counting
    their first selection pass in the same memory pass."""

    def __init__(self, recorder: RankRecorder, quantiles: ExactQuantileStats | None = None) -> None:
        self._recorder = recorder
        self._quantiles = quantiles
        self._microbatch = 0
        self._step = 0
        self._points: dict[tuple[str, int, int], _ObservationPoint] = {}
        # The extension's actlog observer, once native_observer made it; its step, microbatch and
        # quantile populations follow this object's.
        self._native: _NativeObserver | None = None

    @functools.cached_property
    def _extension(self) -> ModuleType | None:
        return activation_ext.prebuilt()

    def begin_step(self, detailed: bool) -> None:
        self._microbatch = 0
        self._step += 1
        if self._native is not None:  # pragma: no cover
            self._native.begin_step(self._recorder.float_buffer())

    def begin_microbatch(self, microbatch: int) -> None:
        self._microbatch = microbatch
        if self._native is not None:  # pragma: no cover
            self._native.begin_microbatch(microbatch)

    def flush(self) -> None:
        """Ends the microbatch's quantile populations the extension's actlog observer took, as
        ExactQuantileStats.flush ends the others."""
        if self._native is not None:  # pragma: no cover
            self._native.flush()

    def native_observer(
        self, resolve_name: Callable[[str, str], tuple[str, int | None] | None]
    ) -> object | None:  # pragma: no cover
        """The extension's observer of actlog calls (tensor, name, suffix, layer_depends, layer_index), which
        observes a call as `observe` with training_forward does, without a Python call: `resolve_name` gives a
        call's point and fixed layer (None: the call's layer index), or None for a call not recorded, once per
        name and suffix. None without the extension or exact quantiles."""
        extension = self._extension
        if not extension or self._quantiles is None:
            return None
        self._native = extension.ActlogObserver(
            resolve_name, self._native_point, functools.partial(self.observe, training_forward=True),
            self._recorder.recorded_flags(), CaptureCapacityError, torch._C._current_graph_task_id,
        )
        return self._native

    def _native_point(
        self, point: str, layer: int
    ) -> tuple[str, tuple[int, ...], list[int]] | None:  # pragma: no cover
        """What the extension's actlog observer needs of a point in the current microbatch, or None for one with a
        detailed-mode slot, which `observe` takes."""
        resolved = self._resolve_point(point, layer)
        assert resolved.quantiles is not None and resolved.quantile_counts is not None
        slots = (resolved.moments, resolved.counts, resolved.absolute_max, resolved.power_sums, resolved.quantiles,
                 resolved.quantile_counts)
        if any(slot.detailed for slot in slots):
            return None
        return resolved.quantile_metric, resolved.quantile_arguments, resolved.quantile_slot_indices

    # Hot path: avoid host sync here (`.item()`, boolean-mask indexing, `nonzero`, `bincount`).
    def observe(self, point: str, layer: int, tensor: torch.Tensor, *, training_forward: bool = False) -> None:
        # A custom autograd forward disables grad even during training. Only an explicitly
        # installed training observer may capture it; backward recomputation remains excluded.
        if not training_forward and not torch.is_grad_enabled():
            return
        if torch._C._current_graph_task_id() != _NO_GRAPH_TASK:
            return
        key = (point, layer, self._microbatch)
        resolved = self._points.get(key)
        if resolved is None:
            resolved = self._resolve_point(point, layer)
            self._points[key] = resolved
        if tensor.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"activation dtype {tensor.dtype} is not supported")
        if resolved.step != self._step:
            resolved.step = self._step
            resolved.elements = 0
        numel = tensor.numel()
        elements = resolved.elements + numel
        if elements > INT32_MAX:
            raise CaptureCapacityError(
                f"activation {point} layer {layer} microbatch {self._microbatch} exceeds int32 counts"
            )
        resolved.elements = elements
        if numel == 0:
            self._recorder.mark_recorded(resolved.slot_indices)
            self._observe_quantiles(point, layer, tensor, None)
            return

        if tensor.is_cuda:  # pragma: no cover
            self._observe_cuda(point, layer, resolved, tensor)
        else:
            self._observe_cpu(resolved, _last_axis_rows(tensor))
            self._observe_quantiles(point, layer, tensor, None)

    def _observe_quantiles(
        self, point: str, layer: int, tensor: torch.Tensor, first_pass: torch.Tensor | None
    ) -> None:
        if self._quantiles is not None:
            self._quantiles.observe(
                f"act/{point}/quantiles", f"act/{point}/quantile_counts", layer, self._microbatch, tensor,
                first_pass=first_pass,
            )

    def _resolve_point(self, point: str, layer: int) -> _ObservationPoint:
        moments = self._recorder.resolve(f"act/{point}/moments", layer, self._microbatch)
        counts = self._recorder.resolve(f"act/{point}/counts", layer, self._microbatch)
        absolute_max = self._recorder.resolve(f"act/{point}/amax", layer, self._microbatch)
        power_sums = self._recorder.resolve(f"act/{point}/power_sums", layer, self._microbatch)
        resolved = _ObservationPoint(
            moments,
            counts,
            absolute_max,
            power_sums,
            slot_indices=[moments.slot_index, counts.slot_index, absolute_max.slot_index, power_sums.slot_index],
        )
        if self._quantiles is not None:
            resolved.quantile_metric = f"act/{point}/quantiles"
            resolved.quantiles = self._recorder.resolve(resolved.quantile_metric, layer, self._microbatch)
            resolved.quantile_counts = self._recorder.resolve(f"act/{point}/quantile_counts", layer, self._microbatch)
            resolved.quantile_slot_indices = [
                *resolved.slot_indices, resolved.quantiles.slot_index, resolved.quantile_counts.slot_index]
            resolved.quantile_arguments = (
                moments.offset, counts.offset, absolute_max.offset, power_sums.offset,
                int(ACTIVATION_LOG2_BINS.start), ACTIVATION_LOG2_BINS.count, resolved.quantiles.offset,
                resolved.quantile_counts.offset,
            )
        return resolved

    def _observe_cpu(self, resolved: _ObservationPoint, rows: torch.Tensor) -> None:
        values = rows.to(torch.float32)
        finite = torch.isfinite(values)
        zeroed = torch.where(finite, values, torch.zeros_like(values))
        magnitudes = values.abs()

        # int64: adding 1.0 per element in float32 stops counting correctly past 2^24.
        finite_count = finite.sum(dtype=torch.int64)
        nonfinite_count = finite.numel() - finite_count

        float_buffer = self._recorder.float_buffer()

        moments = float_buffer[resolved.moments.offset : resolved.moments.offset + resolved.moments.width]
        moments[0] += zeroed.sum()
        moments[1] += (zeroed * zeroed).sum()
        power_sums = float_buffer[resolved.power_sums.offset : resolved.power_sums.offset + resolved.power_sums.width]
        power_sums[0] += (zeroed * zeroed * zeroed).sum()
        power_sums[1] += (zeroed * zeroed * zeroed * zeroed).sum()
        counts = self._recorder.buffer[resolved.counts.offset : resolved.counts.offset + resolved.counts.width]
        counts[0] += nonfinite_count.to(torch.int32)
        counts[1] += finite_count.to(torch.int32)

        # A non-finite element's bin is never counted (its histogram weight is 0), but it
        # must still be finite itself so the bin-index arithmetic below never sees a NaN.
        safe_magnitudes = torch.where(finite, magnitudes, torch.ones_like(magnitudes))
        exponent_bin = bin_indices(_floor_log2(safe_magnitudes), ACTIVATION_LOG2_BINS)
        exponent_counts = torch.zeros(ACTIVATION_LOG2_BINS.count, dtype=torch.int64, device=values.device)
        exponent_counts.index_add_(0, exponent_bin.flatten(), finite.flatten().to(torch.int64))
        exponent_histogram = counts[2 : 2 + ACTIVATION_LOG2_BINS.count]
        exponent_histogram += exponent_counts.to(exponent_histogram.dtype)

        row_finite_count = finite.sum(dim=-1)
        row_has_finite = row_finite_count > 0
        row_sum_of_squares = (zeroed * zeroed).sum(dim=-1)
        row_mean_square = row_sum_of_squares / row_finite_count.clamp_min(1).to(torch.float32)
        root_mean_square_bin = bin_indices(_floor_log2(row_mean_square.sqrt()), ACTIVATION_LOG2_BINS)
        root_mean_square_counts = torch.zeros(ACTIVATION_LOG2_BINS.count, dtype=torch.int64, device=values.device)
        root_mean_square_counts.index_add_(0, root_mean_square_bin, row_has_finite.to(torch.int64))
        root_mean_square_histogram = counts[2 + ACTIVATION_LOG2_BINS.count :]
        root_mean_square_histogram += root_mean_square_counts.to(root_mean_square_histogram.dtype)

        call_absolute_max = torch.where(finite, magnitudes, torch.zeros_like(magnitudes)).max()
        absolute_max = float_buffer[
            resolved.absolute_max.offset : resolved.absolute_max.offset + resolved.absolute_max.width
        ]
        absolute_max[0] = torch.maximum(absolute_max[0], call_absolute_max)

        self._recorder.mark_recorded(resolved.slot_indices)

    def _observe_cuda(
        self, point: str, layer: int, resolved: _ObservationPoint, tensor: torch.Tensor,
    ) -> None:  # pragma: no cover
        """With the extension, one call computes the statistics and selects the exact quantiles
        from the first pass they counted."""
        extension = self._extension
        if extension and self._quantiles is not None:
            # Rejects a repeated population before anything is written.
            self._quantiles.claim(resolved.quantile_metric, layer, self._microbatch)
            self._recorder.mark_recorded(resolved.quantile_slot_indices)
            extension.observe_with_quantiles(tensor, self._recorder.float_buffer(), *resolved.quantile_arguments)
            return
        self._recorder.mark_recorded(resolved.slot_indices)
        float_buffer = self._recorder.float_buffer()
        if extension:
            extension.observe(
                tensor, float_buffer, resolved.moments.offset, resolved.counts.offset, resolved.absolute_max.offset,
                resolved.power_sums.offset, int(ACTIVATION_LOG2_BINS.start), ACTIVATION_LOG2_BINS.count,
                quantile_first_pass=False,
            )
        else:
            from metric_sampler.capture.kernels.activation_kernel import launch_activation_stats_kernel

            launch_activation_stats_kernel(
                _last_axis_rows(tensor), float_buffer, resolved.moments.offset, resolved.counts.offset,
                resolved.absolute_max.offset, resolved.power_sums.offset, ACTIVATION_LOG2_BINS,
            )
        self._observe_quantiles(point, layer, tensor, None)
