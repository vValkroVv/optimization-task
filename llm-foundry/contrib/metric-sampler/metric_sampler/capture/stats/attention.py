from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from types import ModuleType
from typing import Protocol

import torch

from metric_sampler.capture.kernels import activation_ext
from metric_sampler.capture.recorder import RankRecorder, ResolvedSlot
from metric_sampler.capture.stats.helpers.quantiles import quantile_slots
from metric_sampler.contract.metrics import NO_MICROBATCH, ROUTER_ENTROPY_BINS


class _Reduction(Enum):
    """How a field of an attention vector combines two observations of one layer and microbatch."""

    SUM = "sum"
    MIN = "min"
    MAX = "max"


_SUM, _MIN, _MAX = _Reduction.SUM, _Reduction.MIN, _Reduction.MAX
_SUM_MIN_MAX = (_SUM, _MIN, _MAX)
# Five norm sums (the query and the key without and with their rotary part, the value), then the peaks of the
# query, the key and the value.
_QKV_FIELDS = (_SUM,) * 5 + (_MAX,) * 3
_LINEAR_GATE_FIELDS = (_SUM, _MIN, _MAX, _SUM, _MIN, _MAX, _SUM)
_LINEAR_NORM_FIELDS = (_SUM, _SUM, _MAX)
_LINEAR_STATE_FIELDS = (_SUM, _MAX)
# attn/softmax/row_stats after its row count: ent_sum, ent_sumsq, ent_min, ent_max, entn_sum, entn_min, entn_max,
# pmax_sum, pmax_min, pmax_max, pmin_sum, lpmin_sum, lpmin_min, lpmin_max, mlogit_sum, mlogit_max, coll_sum,
# sink_sum, sink_max, dist_sum, local_sum.
_ROW_STAT_FIELDS = (
    _SUM, _SUM, _MIN, _MAX, _SUM, _MIN, _MAX, _SUM, _MIN, _MAX, _SUM,
    _SUM, _MIN, _MAX, _SUM, _MAX, _SUM, _SUM, _MAX, _SUM, _SUM,
)

_LSE, _QKV, _SOFTMAX_GATE = "attn/softmax/lse_stats", "attn/softmax/qkv_stats", "attn/softmax/gate_stats"
_ROW_STATS, _ENTROPY_HIST = "attn/softmax/row_stats", "attn/softmax/entropy_hist"
_HEAD_ENTROPY_HIST = "attn/softmax/head_entropy_hist"
_QUANTILES, _QUANTILE_COUNTS = "attn/softmax/entropy_quantiles", "attn/softmax/entropy_quantile_counts"
_LINEAR_GATE, _LINEAR_NORM, _DECAY = "attn/linear/gate_stats", "attn/linear/norm_stats", "attn/linear/decay_params"
_LINEAR_STATE = "attn/linear/state_stats"
# The decay parameters of a value head: A and softplus(dt_bias).
_DECAY_FIELDS = 2
# A linear attention point's decay parameters slot follows its gate and norm slots.
_LINEAR_DECAY_SLOT = 2
# The points of the calls the extension's AttentionCalls takes, by the kind it names them with.
_NATIVE_POINTS = {
    "softmax": (_LSE, _QKV), "gate": (_SOFTMAX_GATE,), "linear": (_LINEAR_GATE, _LINEAR_NORM),
    "linear_state": (_LINEAR_STATE,),
}


class NativeAttentionCalls(Protocol):
    """The extension's AttentionCalls: the layers' attention observers, made from C++."""

    def begin_step(self, words: torch.Tensor) -> None: ...

    def softmax(self, layer: int, nope_dim: int, rows: Callable[..., None]) -> Callable[..., None]: ...

    def gate(self, layer: int) -> Callable[..., None]: ...

    def linear(self, layer: int) -> Callable[..., None]: ...


def _attention_kernel() -> ModuleType:  # pragma: no cover
    from metric_sampler.capture.kernels import attention_kernel

    return attention_kernel


def _vector_width(fields: tuple[_Reduction, ...], heads: int) -> int:
    """The width of an attention vector: a row count, then each field per head."""
    return 1 + len(fields) * heads


def softmax_slots(heads: int, gated: bool) -> tuple[tuple[str, int], ...]:
    """The slots of one softmax attention layer per microbatch, as (metric name, width): a row count, then each
    field per head."""
    return (
        (_LSE, _vector_width(_SUM_MIN_MAX, heads)),
        (_QKV, _vector_width(_QKV_FIELDS, heads)),
        *(((_SOFTMAX_GATE, _vector_width(_SUM_MIN_MAX, heads)),) if gated else ()),
        (_ROW_STATS, _vector_width(_ROW_STAT_FIELDS, heads)),
        (_ENTROPY_HIST, ROUTER_ENTROPY_BINS.count),
        (_HEAD_ENTROPY_HIST, heads * ROUTER_ENTROPY_BINS.count),
        *quantile_slots(_QUANTILES, _QUANTILE_COUNTS),
    )


def linear_slots(value_heads: int) -> tuple[tuple[str, int], ...]:
    """The slots of one linear attention layer per microbatch, as (metric name, width)."""
    return (
        (_LINEAR_GATE, _vector_width(_LINEAR_GATE_FIELDS, value_heads)),
        (_LINEAR_NORM, _vector_width(_LINEAR_NORM_FIELDS, value_heads)),
        (_LINEAR_STATE, _vector_width(_LINEAR_STATE_FIELDS, value_heads)),
    )


def linear_step_slots(value_heads: int) -> tuple[tuple[str, int], ...]:
    """The once-per-step slots of one linear attention layer: its decay parameters, A and softplus(dt_bias) per
    value head."""
    return ((_DECAY, _DECAY_FIELDS * value_heads),)


def _row_norms(values: torch.Tensor) -> torch.Tensor:
    # Half precision is reduced in float32 straight from the source, without a float32 copy.
    return torch.linalg.vector_norm(values, dim=-1, dtype=torch.float32)


def _row_peaks(values: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(values, ord=math.inf, dim=-1, dtype=torch.float32)


def _sum_rows(stacked: torch.Tensor) -> torch.Tensor:
    """Float64 [fields, heads] sums of `stacked` [fields, ..., heads] over every axis between."""
    return stacked.sum(dim=tuple(range(1, stacked.ndim - 1)), dtype=torch.float64)


def _max_rows(stacked: torch.Tensor) -> torch.Tensor:
    return stacked.amax(dim=tuple(range(1, stacked.ndim - 1)))


def _merge(accumulated: torch.Tensor, incoming: torch.Tensor, fields: tuple[_Reduction, ...]) -> None:
    """Folds float64 [1 + fields * heads] `incoming` into `accumulated`: the row counts add, then every field
    of every head by its reduction."""
    heads = (accumulated.numel() - 1) // len(fields)
    accumulated[0] += incoming[0]
    for index, reduction in enumerate(fields):
        start = 1 + index * heads
        current, other = accumulated[start : start + heads], incoming[start : start + heads]
        if reduction is _SUM:
            current += other
        elif reduction is _MIN:
            torch.minimum(current, other, out=current)
        else:
            torch.maximum(current, other, out=current)


@dataclass
class _Point:
    """The slots one observation writes, and what this step has already put in them."""

    slots: tuple[ResolvedSlot, ...]
    slot_indices: list[int]
    step: int = -1
    rows: int = 0


class AttentionStats:
    """Rank/layer/microbatch statistics of softmax (MLA) and linear (GDN) attention, exact over every
    token, accumulated in place in the recorder's slots: one kernel launch per observation on CUDA, the
    torch reference on the CPU. A non-finite input makes the statistics of its head non-finite, like the
    monitor's activation statistics. The row statistics of softmax attention are computed by the caller's
    kernel, into the slots."""

    def __init__(self, recorder: RankRecorder) -> None:
        self._recorder = recorder
        self._step = 0
        self._detailed = False
        self._points: dict[tuple[str, int, int], _Point] = {}
        # The extension's observers, once native_calls made them; their points and steps follow this object's.
        self._native: NativeAttentionCalls | None = None

    def begin_step(self, detailed: bool) -> None:
        self._step += 1
        self._detailed = detailed
        if self._native is not None:  # pragma: no cover
            self._native.begin_step(self._recorder.float_buffer())

    def native_calls(self, sampler: object) -> NativeAttentionCalls | None:  # pragma: no cover
        """The extension's attention observers, which take a call as observe_softmax, observe_softmax_gate,
        observe_linear and observe_linear_state do, without a Python call: only in a training forward with grad,
        as `sampler`'s _capture_forward_active and _grad_forward tell, in `sampler`'s microbatch. None without the
        extension or over a host recorder: the extension writes the recorder's CUDA words."""
        extension = activation_ext.prebuilt()
        if not extension or not self._recorder.buffer.is_cuda:
            return None
        from metric_sampler.capture.kernels import triton_launch

        self._native = extension.AttentionCalls(
            sampler, self._native_point, triton_launch, _attention_kernel(), self._recorder.recorded_flags(),
            torch._C._current_graph_task_id,
        )
        return self._native

    def _native_point(
        self, kind: str, layer: int, microbatch: int
    ) -> tuple[list[int], list[int], list[int]]:  # pragma: no cover
        """The (offsets, widths, slot indices) of the point of a call AttentionCalls takes."""
        point = self._point(_NATIVE_POINTS[kind], layer, microbatch, decay=kind == "linear" and microbatch == 0)
        return [slot.offset for slot in point.slots], [slot.width for slot in point.slots], point.slot_indices

    def observe_softmax(
        self, layer: int, microbatch: int, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        lse: torch.Tensor, nope_dim: int,
    ) -> None:
        """`query` and `key` are [..., heads, nope_dim + rope_dim], `value` [..., heads, v_dim] and `lse`
        [heads, ...], one value per query row and head."""
        heads = lse.shape[0]
        rows = query.numel() // (heads * query.shape[-1])
        if lse.numel() != heads * rows:
            raise ValueError("the log-partition must hold one value per query row and head")
        point = self._point((_LSE, _QKV), layer, microbatch)
        rows_before = self._claim(point, rows)
        if query.is_cuda:  # pragma: no cover
            _attention_kernel().launch_softmax_stats_kernel(
                query, key, value, lse, self._recorder.float_buffer(), point.slots[0].offset,
                point.slots[1].offset, nope_dim, rows_before,
            )
            return
        with torch.no_grad():
            lse_axes = tuple(range(1, lse.ndim))
            self._write(point.slots[0], _SUM_MIN_MAX, rows_before, rows, (
                lse.sum(dim=lse_axes, dtype=torch.float64), lse.amin(dim=lse_axes), lse.amax(dim=lse_axes),
            ))
            norms = torch.stack((
                _row_norms(query[..., :nope_dim]), _row_norms(query[..., nope_dim:]),
                _row_norms(key[..., :nope_dim]), _row_norms(key[..., nope_dim:]), _row_norms(value),
            ))
            peaks = torch.stack((_row_peaks(query), _row_peaks(key), _row_peaks(value)))
            self._write(point.slots[1], _QKV_FIELDS, rows_before, rows,
                        (_sum_rows(norms).flatten(), _max_rows(peaks).flatten()))

    def observe_softmax_gate(self, layer: int, microbatch: int, gate: torch.Tensor) -> None:
        """`gate` holds the gate logits, [..., heads, v_dim]."""
        point = self._point((_SOFTMAX_GATE,), layer, microbatch)
        rows = gate.numel() // (gate.shape[-2] * gate.shape[-1])
        rows_before = self._claim(point, rows)
        if gate.is_cuda:  # pragma: no cover
            _attention_kernel().launch_gate_stats_kernel(
                gate, self._recorder.float_buffer(), point.slots[0].offset, rows_before
            )
            return
        with torch.no_grad():
            means = gate.mean(dim=-1, dtype=torch.float32).reshape(-1, gate.shape[-2])
            self._write(point.slots[0], _SUM_MIN_MAX, rows_before, rows, (
                means.sum(dim=0, dtype=torch.float64), means.amin(dim=0), means.amax(dim=0),
            ))

    def observe_linear(
        self, layer: int, microbatch: int, beta: torch.Tensor, log_decay: torch.Tensor, value: torch.Tensor,
        output: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor,
    ) -> None:
        """`beta` and `log_decay` are [..., value heads], `value` and `output` [..., value heads, v_dim]; `a_log`
        and `dt_bias` are the layer's decay parameters, recorded with the first microbatch of a step."""
        heads = beta.shape[-1]
        rows = beta.numel() // heads
        point = self._point((_LINEAR_GATE, _LINEAR_NORM), layer, microbatch, decay=microbatch == 0)
        rows_before = self._claim(point, rows)
        has_decay = len(point.slots) > _LINEAR_DECAY_SLOT
        if beta.is_cuda:  # pragma: no cover
            _attention_kernel().launch_linear_stats_kernel(
                beta, log_decay, value, output, a_log, dt_bias, self._recorder.float_buffer(), point.slots[0].offset,
                point.slots[1].offset, point.slots[_LINEAR_DECAY_SLOT].offset if has_decay else -1, rows_before,
            )
            return
        with torch.no_grad():
            log_decay = log_decay.float()
            # [beta, alpha, log-decay]: only beta and alpha have a minimum and a maximum.
            gates = torch.stack((beta.float(), log_decay.exp(), log_decay)).reshape(3, -1, heads)
            sums, low, high = gates.sum(dim=1, dtype=torch.float64), gates[:2].amin(dim=1), gates[:2].amax(dim=1)
            self._write(point.slots[0], _LINEAR_GATE_FIELDS, rows_before, rows,
                        (sums[0], low[0], high[0], sums[1], low[1], high[1], sums[2]))
            norms = torch.stack((_row_norms(value), _row_norms(output)))
            self._write(point.slots[1], _LINEAR_NORM_FIELDS, rows_before, rows,
                        (_sum_rows(norms).flatten(), _row_peaks(output).reshape(-1, heads).amax(dim=0)))
            if has_decay:
                decay_offset = point.slots[_LINEAR_DECAY_SLOT].offset
                decay = self._recorder.float_buffer()[decay_offset : decay_offset + _DECAY_FIELDS * heads]
                decay[:heads] = a_log.float().exp()
                decay[heads:] = torch.nn.functional.softplus(dt_bias.float())

    def observe_softmax_rows(
        self, layer: int, microbatch: int,
        compute: Callable[[tuple[torch.Tensor | None, ...] | None], tuple[torch.Tensor, ...]],
    ) -> None:
        """The words of the five row-statistics metrics of one attention call come from `compute`, each as its
        metric lays it out: row_stats, entropy_hist, head_entropy_hist (recorded on detailed steps only),
        entropy_quantiles and entropy_quantile_counts. The first call of a layer and microbatch in a step is
        given the metrics' own words to write them into, `entropy_hist` and `head_entropy_hist` zero, and
        `head_entropy_hist` None on other steps. A repeated call is given None, returns the words as new
        tensors, and merges into the earlier ones. Every call counts as one row of its point."""
        point = self._point((_ROW_STATS, _ENTROPY_HIST, _QUANTILES, _QUANTILE_COUNTS), layer, microbatch)
        earlier = self._claim(point, 1)
        # Claimed with the point: its rows, one per call, are those of the point.
        head_point = self._point((_HEAD_ENTROPY_HIST,), layer, microbatch) if self._detailed else None
        if head_point is not None:
            self._claim(head_point, 1)
        stats_slot, hist_slot, quantiles_slot, counts_slot = point.slots
        view = self._recorder.view
        if not earlier:
            compute((
                view(stats_slot), view(hist_slot), None if head_point is None else view(head_point.slots[0]),
                view(quantiles_slot), view(counts_slot),
            ))
            return
        row_stats, entropy_hist, head_entropy_hist, _, entropy_quantile_counts = compute(None)
        self._store(stats_slot, _ROW_STAT_FIELDS, earlier, row_stats)
        view(hist_slot).add_(entropy_hist)
        view(counts_slot).add_(entropy_quantile_counts)
        # Quantiles of two populations cannot be combined; as for an empty population they are NaN, and the
        # counts, which add, still cover every observation.
        view(quantiles_slot).fill_(math.nan)
        if head_point is not None:
            view(head_point.slots[0]).add_(head_entropy_hist)

    def observe_linear_state(self, layer: int, microbatch: int, state: torch.Tensor) -> None:
        """`state` is the final recurrent state of every document, [documents, value heads, key dim, value dim]."""
        point = self._point((_LINEAR_STATE,), layer, microbatch)
        documents = state.shape[0]
        rows_before = self._claim(point, documents)
        if state.is_cuda:  # pragma: no cover
            extension = activation_ext.prebuilt()
            if extension:
                # The same operations from C++, written into the slot without the concatenation.
                extension.linear_state_stats(state, self._recorder.view(point.slots[0]), rows_before > 0)
                return
        with torch.no_grad():
            norms = torch.linalg.vector_norm(state, dim=(-2, -1), dtype=torch.float32)
            self._write(point.slots[0], _LINEAR_STATE_FIELDS, rows_before, documents,
                        (norms.sum(dim=0, dtype=torch.float64), norms.amax(dim=0)))

    def _point(self, names: tuple[str, ...], layer: int, microbatch: int, decay: bool = False) -> _Point:
        key = (names[0], layer, microbatch)
        point = self._points.get(key)
        if point is None:
            slots = [self._recorder.resolve(name, layer, microbatch) for name in names]
            if decay:
                slots.append(self._recorder.resolve(_DECAY, layer, NO_MICROBATCH))
            point = self._points[key] = _Point(tuple(slots), [slot.slot_index for slot in slots])
        return point

    def _claim(self, point: _Point, rows: int) -> int:
        """Marks the point's slots recorded and returns the rows this step has put in them already."""
        if point.step != self._step:
            point.step, point.rows = self._step, 0
        rows_before = point.rows
        point.rows += rows
        self._recorder.mark_recorded(point.slot_indices)
        return rows_before

    def _write(
        self, slot: ResolvedSlot, fields: tuple[_Reduction, ...], rows_before: int, rows: int,
        columns: tuple[torch.Tensor, ...],
    ) -> None:
        values = torch.cat((torch.full((1,), rows, dtype=torch.float64, device=columns[0].device), *columns))
        self._store(slot, fields, rows_before, values)

    def _store(
        self, slot: ResolvedSlot, fields: tuple[_Reduction, ...], rows_before: int, values: torch.Tensor
    ) -> None:
        """Writes the slot's `values`, merged into what this step has put in it already."""
        view = self._recorder.view(slot)
        if rows_before:
            earlier = view.double()
            _merge(earlier, values, fields)
            values = earlier
        view.copy_(values)
