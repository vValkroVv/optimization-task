import math
import operator
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field, replace
from functools import reduce
from typing import Any, NamedTuple, TypeAlias, TypeVar

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.config.schema.callbacks.sources.modelling_source import (
    ModellingSourceConfig,
)
from torchtitan.config.schema.callbacks.sources.modelling_source import MoECaptureConfig
from torchtitan.config.schema.callbacks.sources.modelling_source import (
    CaptureOutput,
)
from torchtitan.protocols.sharding_backend import ShardStats

from torchtitan.callbacks.base import ObserverState
from torchtitan.callbacks.metrics.sample import MetricRecord, Sample
from torchtitan.config.schema.protocols.types import StrEnum
from torchtitan.config.schema.callbacks.metrics.sample import (
    DTensorMode,
    MetricStepKind,
    ReduceOp,
    AxisSpec,
    MetricConfig,
    MetricStepConfig,
    ReduceConfig,
    ReduceStageConfig,
    StatOpConfig,
    StatOpName,
    ThresholdCmp,
)
from torchtitan.callbacks.metrics.stats import (
    _local_tensor,
    _ACC_DTYPE,
    _EPS,
    _assert_order_stat_size,
    _dtensor_reduce_groups,
    _warn_full_tensor_fallback,
    apply_stat_op,
    histogram,
)
from torchtitan.callbacks.metrics.timers import callback_timer
from torchtitan.callbacks.sources.base import (
    MonitorSource,
    canonical_fqn,
)
from torchtitan.callbacks.sources.moe_units import (
    MoEMonitorUnit,
    iter_moe_monitor_units,
)
from torchtitan.distributed.pipeline_parallel import in_metadata_probe
from torchtitan.models.common.ffn import FeedForward
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.ops.float8.fp8_tensor import Float8BlockwiseQTensor
from torchtitan.tools.logging import logger

__all__ = [
    "CaptureOutput",
    "materialize",
    "ModellingSource",
    "MoECaptureConfig",
    "RouterOutput",
]


class RouterOutput(NamedTuple):
    """The callback-side declaration of ``TokenChoiceTopKRouter.forward``'s return.

    The router itself returns a bare tuple -- a module in the net returns
    tensors, full stop. This names the positions for the one consumer that
    hooks the router, so the read below is by field, and a widened return
    fails the arity check instead of being silently reinterpreted. Kept in
    sync with the router by that check and by the unit tests on the hook.
    """

    topk_scores_BLK: torch.Tensor  # noqa: N815
    topk_expert_ids_BLK: torch.Tensor  # noqa: N815
    # Full routing scores ``(B, L, E)`` (post sigmoid/softmax).
    scores_BLE: torch.Tensor  # noqa: N815
    # Raw pre-activation gate logits ``(B, L, E)`` (float32), for the z-loss.
    logits_BLE: torch.Tensor  # noqa: N815


class _CaptureKind(StrEnum):
    """Which tag family a hook payload belongs to. Private: only _capture cares."""

    activation = "activation"
    gradient = "gradient"


# Batch statistics are pooled over the loss mesh (data + context ranks).
_POOL_OVER_LOSS = ReduceStageConfig(group="loss", op=ReduceOp.mean)


@dataclass(slots=True)
class _RouterAccum:
    """One MoE layer's router statistics over the forwards of one batch"""

    # Mesh axes the local token histogram is split over (tp, ep). dp and cp are
    # not DTensor axes for activations; those are pooled over the loss mesh
    # later.
    token_axes: tuple[str, ...]
    tokens: list[torch.Tensor] = field(default_factory=list)
    scores: list[torch.Tensor] = field(default_factory=list)
    # Router ids that fell outside [0, E) — histogram's overflow bin, summed
    # over the batch's forwards. None until the first forward.
    out_of_range: torch.Tensor | None = None


_MetricName = str | tuple[str, ...]
_Name = TypeVar("_Name")


def materialize(
    pending: list[tuple[_Name, torch.Tensor]],
    *,
    full_precision: frozenset[int] = frozenset(),
) -> None:
    """Normalize ``pending`` in place: full logical tensor, detached, fp32.

    Exactly what ``plain_float_tensor`` did per entry, including the dtype --
    a metric must not be computed in whatever precision its input happened to
    be accumulated in, or the same tag lands in fp32 or fp64 depending on
    whether the parallelism made it a DTensor. What is new is only that the
    sharded entries of one placement travel in a SINGLE all-gather instead of
    one each: the per-expert statistics all come out as (E,) vectors with the
    same placement, so they stack into one tensor whose gather serves all of
    them. ``None`` passes through, as it did.
    """
    groups: dict[tuple, list[int]] = {}
    for index, (name, tensor) in enumerate(pending):
        dtype = _ACC_DTYPE if index in full_precision else torch.float32
        if not isinstance(tensor, DTensor):
            if tensor is not None:
                pending[index] = (name, _local_tensor(tensor, dtype=dtype))
            continue
        key = (
            tuple(tensor.shape),
            tensor.dtype,
            str(tensor.device),
            tuple(tensor.placements),
            tensor.device_mesh,
            dtype,
        )
        groups.setdefault(key, []).append(index)
    for key, indices in groups.items():
        stacked = torch.stack([pending[i][1] for i in indices])
        full = _local_tensor(stacked, full_tensor=True, dtype=key[-1])
        for position, index in enumerate(indices):
            pending[index] = (pending[index][0], full[position])


def _segment_expert_stat(
    support: ShardStats,
    vector: torch.Tensor,
    op: StatOpConfig,
    metric_name: str,
    *,
    mean: torch.Tensor | None = None,
    num_segments: int | None = None,
    num_groups: int = 1,
) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor] | None]:
    """Reduce equal flat segments, defaulting to one segment per expert.

    Packed weights use one segment per (expert, logical weight). ``num_groups``
    identifies interleaved logical weights for their separate actrel thresholds.
    """
    num_segments = vector.shape[0] if num_segments is None else num_segments
    if num_segments == 0:
        raise ValueError(f"Segment metric '{metric_name}': no experts")
    if num_groups <= 0 or num_segments % num_groups or vector.numel() % num_segments:
        raise ValueError("Segments must evenly divide the parameter and groups")
    segment_numel = vector.numel() // num_segments
    if segment_numel == 0:
        raise ValueError(f"Segment metric '{metric_name}': expert parameters are empty")
    name = op.op
    if name == StatOpName.NORM:
        p = 2.0 if op.p is None else float(op.p)
        if p == 0:
            return (
                support.sum_flat_segments(
                    vector, num_segments, transform=lambda rows, _: rows != 0
                ),
                None,
            )
        if math.isnan(p):
            raise NotImplementedError(
                f"Segment metric '{metric_name}': norm with p={p} "
                "requires a non-NaN order."
            )
        if p == -math.inf:
            return support.min_flat_segments(vector.abs(), num_segments), None
        if p == math.inf:
            return support.abs_max_flat_segments(vector, num_segments), None
        if p < 0:
            return (
                support.sum_flat_segments(
                    vector,
                    num_segments,
                    transform=lambda rows, _: rows.abs().to(_ACC_DTYPE).pow(p),
                    sum_dtype=_ACC_DTYPE,
                ),
                lambda sums: sums.pow(1.0 / p),
            )
        if p == 2.0:
            # fp64 squares keep their range for |x| down to ~1e-154.
            return (
                support.sum_flat_segments(
                    vector,
                    num_segments,
                    transform=lambda rows, _: rows.to(_ACC_DTYPE).square(),
                    sum_dtype=_ACC_DTYPE,
                ),
                lambda sums: sums.sqrt(),
            )
        scale = _local_tensor(
            support.abs_max_flat_segments(vector, num_segments),
            full_tensor=True,
            dtype=_ACC_DTYPE,
        )
        divisor = torch.where(scale > 0, scale, torch.ones_like(scale))
        return (
            support.sum_flat_segments(
                vector,
                num_segments,
                transform=lambda rows, segments: (
                    rows.abs().to(_ACC_DTYPE) / divisor[segments].unsqueeze(1)
                ).pow(p),
                sum_dtype=_ACC_DTYPE,
            ),
            lambda sums: scale * sums.pow(1.0 / p),
        )
    if name == StatOpName.SUM:
        return support.sum_flat_segments(vector, num_segments), None
    if name == StatOpName.MEAN:
        return (
            support.sum_flat_segments(vector, num_segments),
            lambda sums: sums / segment_numel,
        )
    if name == StatOpName.RMS:
        return (
            support.sum_flat_segments(vector * vector, num_segments),
            lambda sums: (sums / segment_numel).sqrt(),
        )
    if name == StatOpName.ABS_MAX:
        return support.abs_max_flat_segments(vector, num_segments), None
    if name == StatOpName.MAX:
        return support.max_flat_segments(vector, num_segments), None
    if name == StatOpName.MIN:
        return support.min_flat_segments(vector, num_segments), None
    if name == StatOpName.COUNT:
        return torch.full(
            (num_segments,), segment_numel, device=vector.device, dtype=_ACC_DTYPE
        ), None
    if name in (StatOpName.MAX_MIN, StatOpName.MAX_MIN_RATIO, StatOpName.MAX_VIO):
        high = _local_tensor(
            support.max_flat_segments(vector, num_segments),
            full_tensor=True,
            dtype=_ACC_DTYPE,
        )
        if name == StatOpName.MAX_VIO:
            average = (
                _local_tensor(
                    support.sum_flat_segments(vector, num_segments),
                    full_tensor=True,
                    dtype=_ACC_DTYPE,
                )
                / segment_numel
            )
            return (high - average) / average.clamp_min(1.0), None
        low = _local_tensor(
            support.min_flat_segments(vector, num_segments),
            full_tensor=True,
            dtype=_ACC_DTYPE,
        )
        if name == StatOpName.MAX_MIN:
            return high - low, None
        return torch.where(
            low > 0, high / low, torch.full_like(high, float("inf"))
        ), None
    if name in (StatOpName.ACTABS, StatOpName.ACTREL):
        if op.threshold is None:
            raise ValueError(f"{name} requires a threshold")
        threshold: float | torch.Tensor = op.threshold
        if name == StatOpName.ACTREL:
            abs_sums = _local_tensor(
                support.sum_flat_segments(vector.abs(), num_segments),
                full_tensor=True,
                dtype=_ACC_DTYPE,
            )
            threshold = (
                threshold
                * abs_sums.reshape(-1, num_groups).sum(dim=0)
                / (vector.numel() // num_groups)
            )

        def active_mask(rows: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
            bound = (
                threshold[segments % num_groups].unsqueeze(1)
                if isinstance(threshold, torch.Tensor)
                else threshold
            )
            return (
                rows.abs().gt(bound)
                if op.cmp == ThresholdCmp.gt
                else rows.abs().lt(bound)
            )

        partial = support.sum_flat_segments(
            vector,
            num_segments,
            transform=active_mask,
        )
        return partial, lambda sums: sums / segment_numel
    if name == StatOpName.ENTROPY:
        totals = _local_tensor(
            support.sum_flat_segments(vector, num_segments, sum_dtype=torch.float32),
            full_tensor=True,
            dtype=_ACC_DTYPE,
        ).clamp_min(_EPS)

        def entropy_terms(rows: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
            probs = rows / totals[segments].to(rows.dtype).unsqueeze(1)
            return probs.mul_(probs.clamp_min(_EPS).log_()).neg_()

        return support.sum_flat_segments(
            vector, num_segments, transform=entropy_terms, sum_dtype=torch.float32
        ), None
    if name == StatOpName.RAW_ENTROPY_NORM:
        partial = support.sum_flat_segments(
            vector,
            num_segments,
            transform=lambda rows, _: rows.clamp_min(_EPS).log_().mul_(rows).neg_(),
            sum_dtype=torch.float32,
        )
        if segment_numel > 1:
            return partial, lambda sums: sums / math.log(segment_numel)
        return partial, None
    if name in (
        StatOpName.NONZERO_COUNT,
        StatOpName.NAN_COUNT,
        StatOpName.INF_COUNT,
        StatOpName.ZERO_RATIO,
        StatOpName.FINITE_RATIO,
        StatOpName.POSITIVE_RATIO,
    ):
        masks = {
            "nonzero_count": lambda rows: rows != 0,
            "nan_count": torch.isnan,
            "inf_count": torch.isinf,
            "zero_ratio": lambda rows: rows == 0,
            "finite_ratio": torch.isfinite,
            "positive_ratio": lambda rows: rows > 0,
        }
        partial = support.sum_flat_segments(
            vector, num_segments, transform=lambda rows, _: masks[name](rows)
        )
        if name.endswith("_ratio"):
            return partial, lambda sums: sums / segment_numel
        return partial, None
    if name in (StatOpName.STD, StatOpName.VAR, StatOpName.CV, StatOpName.KURTOSIS):
        if mean is None:
            mean = (
                _local_tensor(
                    support.sum_flat_segments(vector, num_segments),
                    full_tensor=True,
                    dtype=_ACC_DTYPE,
                )
                / segment_numel
            )

        second = support.moment_flat_segments(vector, num_segments, mean, 2)
        if name == StatOpName.KURTOSIS:
            moments = (
                _local_tensor(
                    torch.stack(
                        (
                            second,
                            support.moment_flat_segments(vector, num_segments, mean, 4),
                        )
                    ),
                    full_tensor=True,
                    dtype=_ACC_DTYPE,
                )
                / segment_numel
            )
            return moments[1] / moments[0].square().clamp_min(_EPS**2), None

        def finish(sums: torch.Tensor) -> torch.Tensor:
            variance = (sums / segment_numel).clamp_min(0.0)
            if name == StatOpName.VAR:
                return variance
            std = variance.sqrt()
            return (
                std / mean.float().abs().clamp_min(_EPS)
                if name == StatOpName.CV
                else std
            )

        return second, finish
    raise NotImplementedError(
        f"Segment metric '{metric_name}': stat '{name}' does not "
        "decompose into the backend's per-segment sums and maxima, which is "
        "all a flat shard computes."
    )


def _split_packed_metrics(
    metrics: Iterable[tuple[_MetricName, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Split a materialized (E, 2) result into the two logical weight vectors."""
    for names, value in metrics:
        if isinstance(names, str):
            yield names, value
            continue
        yield from zip(names, value.reshape(-1, 2).unbind(1))


class ModellingSource(MonitorSource):
    """Statistics of the model's own tensors, read at module boundaries.

    Three families, one acquisition mechanism — an ``nn.Module`` hook — and no
    model-side cooperation: MoE layers are discovered by type
    (``iter_moe_monitor_units``):

    - activations and activation gradients, emitted as
      ``activations_{metric.name}/{fqn}_{role}.{i}`` and
      ``activations_gradients_*`` (the llm-f tag layout);
    - MoE router load balance, from a hook on each router (see
      :class:`MoECaptureConfig`);
    - per-expert weight statistics, walked off the experts module.

    Example:
        listed as ``sources.modelling`` of the preset's ``monitoring`` callback;
        its fields are then addressable as
        ``--callbacks.monitoring.sources.modelling.capture_backward``.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = ModellingSourceConfig

    config: Config

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        # Static per-config values, hoisted out of the per-tensor hook path.
        self._axes = AxisSpec(names=tuple(config.axes))
        # Hook targets resolved once in fit_start: (fqn, module) per module.
        self._targets: list[tuple[str, nn.Module]] = []
        self._units: list[MoEMonitorUnit] = []
        # unit name -> its router statistics for the batch in progress
        self._router_accum: dict[str, _RouterAccum] = {}
        # SwiGLU modules whose clamp counts are captured, and per fqn this
        # batch's float64 [gate clamped, up clamped, elements scanned].
        self._clip_targets: list[tuple[str, nn.Module]] = []
        self._clip_accum: dict[str, torch.Tensor] = {}
        self._partial_units_warned: set[str] = set()

    def fit_start(self, state: ObserverState) -> None:
        units = iter_moe_monitor_units(state.model_parts)
        router_ids = {id(unit.router) for unit in units}

        candidates: list[tuple[str, nn.Module]] = []
        seen_fqns: set[str] = set()
        for model_part in state.model_parts:
            for module_name, module in model_part.named_modules():
                fqn = canonical_fqn(module_name)
                if not fqn or fqn in seen_fqns or id(module) in router_ids:
                    continue
                seen_fqns.add(fqn)
                candidates.append((fqn, module))
        kept = self._filter_fqns([fqn for fqn, _ in candidates], "modules")
        self._targets = [(fqn, module) for fqn, module in candidates if fqn in kept]
        self._clip_targets = []
        if self.config.capture_swiglu_clips:
            self._clip_targets = self._resolve_clip_targets()
        self._units = []
        if self.config.moe.requested:
            kept_units = self._filter_fqns([unit.name for unit in units], "MoE units")
            self._units = [unit for unit in units if unit.name in kept_units]

    def _attach_hooks(self) -> None:
        """Register the module hooks."""
        for fqn, module in self._targets:
            if self.config.capture_forward or self.config.capture_inputs:
                self._handles.append(
                    module.register_forward_hook(self._forward_hook(fqn))
                )
            if self.config.capture_backward or self.config.capture_grad_inputs:
                self._handles.append(
                    module.register_full_backward_hook(self._backward_hook(fqn))
                )
        if self.config.moe.needs_router:
            for unit in self._units:
                self._handles.append(
                    unit.router.register_forward_hook(self._router_hook(unit))
                )
        for fqn, module in self._clip_targets:
            self._handles.append(
                module.register_forward_pre_hook(self._clip_pre_hook())
            )
            self._handles.append(module.register_forward_hook(self._clip_hook(fqn)))

    def batch_start(self, state: ObserverState) -> None:
        super().batch_start(state)
        self._router_accum.clear()
        self._clip_accum.clear()
        self._detach_hooks()
        if self.active:
            self._attach_hooks()
            for _fqn, module in self._clip_targets:
                module.swiglu_clip_counts = torch.zeros(
                    3, dtype=torch.int64, device=state.device
                )

    def batch_end(self, state: ObserverState) -> None:
        super().batch_end(state)
        if self.active and self.config.moe.requested:
            self._record_moe(state)
        if self.active and self._clip_targets:
            self._record_swiglu_clips(state)
        self._release_clip_slots()
        self._detach_hooks()

    def close(self) -> None:
        self._release_clip_slots()
        super().close()

    def _resolve_clip_targets(self) -> list[tuple[str, nn.Module]]:
        """The kept modules whose SwiGLU clamps, hence has clamp counts."""
        targets = [
            (fqn, module)
            for fqn, module in self._targets
            if isinstance(module, (FeedForward, GroupedExperts))
            and module.swiglu_limit > 0.0
        ]
        if not targets:
            logger.warning(
                "ModellingSource: capture_swiglu_clips is set but no kept "
                "FeedForward / GroupedExperts has swiglu_limit > 0."
            )
        return targets

    def _release_clip_slots(self) -> None:
        """Stop the modules counting: an unmonitored step pays nothing."""
        for _fqn, module in self._clip_targets:
            module.swiglu_clip_counts = None

    def _clip_pre_hook(self):
        def hook(module: nn.Module, inputs: tuple[Any, ...]) -> None:
            if in_metadata_probe():
                return
            del inputs
            # A recompute's counts land in the slot too; the next forward
            # clears them here before they are ever read.
            if not self.active or self._is_recompute():
                return
            counts = module.swiglu_clip_counts
            if counts is not None:
                counts.zero_()

        return hook

    def _clip_hook(self, fqn: str):
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            if in_metadata_probe():
                return
            del inputs, output
            if not self.active or self._is_recompute() or torch.compiler.is_compiling():
                return
            counts = module.swiglu_clip_counts
            if counts is None:
                return
            with torch.no_grad():
                sample = counts.to(torch.float64)
                accum = self._clip_accum.get(fqn)
                self._clip_accum[fqn] = sample if accum is None else accum + sample

        return hook

    def _record_swiglu_clips(self, state: ObserverState) -> None:
        """Publish the batch's clamp counts and shares, one collective for all."""
        assert self.reducer is not None, "MonitorSource records before fit_start"
        names = [fqn for fqn, _ in self._clip_targets]
        zero = torch.zeros(3, dtype=torch.float64, device=state.device)
        # Every count is this rank's own scan, tp shards included, so one sum
        # over both meshes is exact under any parallelism.
        pooled = self.reducer.reduce_over_groups(
            torch.stack([self._clip_accum.get(name, zero) for name in names]),
            ("loss", "tp"),
            ReduceOp.sum,
        )
        scanned = pooled[:, 2].clamp_min(1.0)
        namespace = self.config.swiglu_clip_namespace
        for index, fqn in enumerate(names):
            # Named after the projections, as the other per-projection stats.
            for column, half in enumerate(("w1", "w3")):
                count = pooled[index, column]
                for leaf, value in (
                    (f"{half}_count", count),
                    (f"{half}_share", count / scanned[index]),
                ):
                    self._records.append(
                        MetricRecord(
                            name=f"{namespace}/{fqn}/{leaf}",
                            value=value.to(torch.float32),
                            step=self._step,
                            reduce=ReduceConfig(),
                        )
                    )

    def _forward_hook(self, fqn: str):
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            if in_metadata_probe():
                return
            del module
            if not self.active or self._is_recompute():
                return
            if self.config.capture_inputs:
                self._capture(fqn, inputs, role="input", kind=_CaptureKind.activation)
            if self.config.capture_forward:
                self._capture(fqn, output, role="output", kind=_CaptureKind.activation)

        return hook

    def _backward_hook(self, fqn: str):
        def hook(
            module: nn.Module,
            grad_input: tuple[Any, ...],
            grad_output: tuple[Any, ...],
        ) -> None:
            if in_metadata_probe():
                return
            del module
            if not self.active or torch.compiler.is_compiling():
                return
            if self.config.capture_grad_inputs:
                self._capture(
                    fqn, grad_input, role="grad_input", kind=_CaptureKind.gradient
                )
            if self.config.capture_backward:
                self._capture(
                    fqn, grad_output, role="grad_output", kind=_CaptureKind.gradient
                )

        return hook

    @callback_timer(timer_level=2)
    def _capture(
        self, fqn: str, tensors: Any, *, role: str, kind: _CaptureKind
    ) -> None:
        """Record one hook payload's tensors.

        Captures accumulate (a module may run several times per batch) and are
        combined into one record per tag when the batch ends.
        """
        tag_prefix = (
            "activations"
            if kind == _CaptureKind.activation
            else "activations_gradients"
        )
        for suffix, tensor in self._iter_capture_tensors(tensors, role):
            self._record(
                Sample(
                    name=f"{tag_prefix}_{{metric}}/{fqn}{suffix}",
                    value=tensor,
                    axes=self._axes,
                    metrics=self.config.metrics,
                ),
                accumulate=True,
            )
            if self.config.output == CaptureOutput.first_tensor:
                return

    @staticmethod
    def _iter_capture_tensors(
        output: Any, role: str
    ) -> Iterable[tuple[str, torch.Tensor]]:
        if isinstance(output, torch.Tensor):
            if not output.is_complex():
                yield f"_{role}.0", output
        elif isinstance(output, tuple):
            for idx, value in enumerate(output):
                if isinstance(value, torch.Tensor) and not value.is_complex():
                    yield f"_{role}.{idx}", value

    def _router_hook(self, unit: MoEMonitorUnit):
        num_experts = int(unit.router.num_experts)

        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            if in_metadata_probe():
                return
            del module, inputs
            if not self.active or self._is_recompute():
                return
            self._accumulate_router(unit, num_experts, output)

        return hook

    @callback_timer(timer_level=2)
    def _accumulate_router(
        self, unit: MoEMonitorUnit, num_experts: int, output: Any
    ) -> None:
        """Add one router forward to this batch's accumulator."""
        # The exact TokenChoiceTopKRouter.forward contract, not an arity
        # floor: a widened return must fail here, not be silently truncated.
        # RouterOutput names the positions so the reads below are by field.
        if not isinstance(output, tuple) or len(output) != len(RouterOutput._fields):
            raise ValueError(
                f"ModellingSource: router '{unit.name}' is expected to return "
                f"{RouterOutput._fields}, got {type(output).__name__} of arity "
                f"{len(output) if isinstance(output, tuple) else 'n/a'} — the "
                "router contract changed; update RouterOutput with it."
            )
        out = RouterOutput(*output)
        topk_expert_ids = out.topk_expert_ids_BLK
        router_scores = out.scores_BLE
        with torch.no_grad():
            token_axes = (
                tuple(
                    axis
                    for axis in (_dtensor_reduce_groups(topk_expert_ids) or ())
                    if axis not in ("dp_replicate", "dp_shard", "cp")
                )
                if isinstance(topk_expert_ids, DTensor)
                else ()
            )
            ids = (
                topk_expert_ids.to_local()
                if isinstance(topk_expert_ids, DTensor)
                else topk_expert_ids
            )
            hist = histogram(ids, num_experts)

            scores = router_scores.detach().mean(dim=(0, 1), dtype=_ACC_DTYPE)
            if unit.module.expert_bias_E is not None:
                scores = scores + unit.module.expert_bias_E

            accum = self._router_accum.setdefault(
                unit.name, _RouterAccum(token_axes=token_axes)
            )
            # The (E,) slice keeps the per-expert contract; the overflow bin
            # is a routing-bug detector and travels as its own scalar.
            accum.tokens.append(hist[:num_experts])
            accum.scores.append(scores)
            oob = hist[num_experts]
            accum.out_of_range = (
                oob if accum.out_of_range is None else accum.out_of_range + oob
            )

    def _record_moe(self, state: ObserverState) -> None:
        """Publish the batch's router statistics, and the expert weights."""

        assert self.reducer is not None, "MonitorSource records before fit_start"
        config = self.config.moe
        have_data = [
            unit.name in self._router_accum
            and bool(self._router_accum[unit.name].tokens)
            for unit in self._units
        ]
        if self._units and config.needs_router:
            have_data = self._units_with_data_everywhere(state, have_data)
        del state
        pending: list[tuple[_MetricName, torch.Tensor]] = []
        spreads: list[tuple[_MetricName, int]] = []
        post_ops: list[tuple[int, Callable[[torch.Tensor], torch.Tensor]]] = []
        for unit, unit_has_data in zip(self._units, have_data):
            if not unit_has_data:
                # No router data (none asked for, or none on some rank — the
                # gate above is rank-uniform): only what is read off the module.
                if config.bias_metrics and unit.module.expert_bias_E is not None:
                    vectors = [("bias", unit.module.expert_bias_E)]
                    materialize(vectors)
                    self._record(
                        Sample(
                            name=f"{config.namespace}/{unit.name}/expert_bias",
                            value=vectors[0][1],
                            axes=AxisSpec(names=("expert",)),
                            metrics=config.bias_metrics,
                        )
                    )
                if config.expert_weights:
                    self._record_expert_weights(unit, pending, spreads, post_ops)
                continue
            accum = self._router_accum[unit.name]
            prefix = f"{config.namespace}/{unit.name}"
            tokens = reduce(operator.add, accum.tokens)
            scores = reduce(operator.add, accum.scores) / len(accum.scores)
            # Materialised together with every other unit's, below: three
            # (E,) vectors of the same placement per unit, so one gather
            # serves all the layers instead of three each.
            vectors = [
                (
                    "tokens",
                    self.reducer.reduce_over_groups(
                        tokens, accum.token_axes, ReduceOp.sum
                    ),
                ),
                ("scores", scores),
                ("bias", unit.module.expert_bias_E),
            ]
            materialize(vectors)
            raw_tokens, raw_scores, bias = (value for _, value in vectors)
            if accum.out_of_range is not None:
                pending.append(
                    (
                        f"{prefix}/tokens_out_of_range",
                        self.reducer.reduce_over_groups(
                            accum.out_of_range, accum.token_axes, "sum"
                        ),
                    )
                )

            for leaf, vector, vector_metrics in (
                ("tokens_per_expert", raw_tokens, config.token_metrics),
                ("scores_per_expert", raw_scores, config.score_metrics),
                ("expert_bias", bias, config.bias_metrics),
            ):
                if vector is None or vector.numel() == 0 or not vector_metrics:
                    continue
                self._record(
                    Sample(
                        name=f"{prefix}/{leaf}",
                        value=vector,
                        axes=AxisSpec(names=("expert",)),
                        metrics=vector_metrics,
                    )
                )

            pooled = self.reducer.reduce_stage(
                torch.cat((raw_tokens, raw_scores)), _POOL_OVER_LOSS
            )
            pooled_tokens, pooled_scores = pooled.split(
                [raw_tokens.numel(), raw_scores.numel()]
            )
            if pooled_tokens.numel() > 0:
                fraction = pooled_tokens / pooled_tokens.sum().clamp_min(1.0)
                if config.per_expert_tokens:
                    pending.append(
                        (f"{prefix}/tokens_per_expert/expert_{{e}}", fraction)
                    )
                if pooled_scores.numel() == pooled_tokens.numel():
                    pending.append(
                        (
                            f"{prefix}/load_balance",
                            pooled_tokens.numel() * (fraction * pooled_scores).sum(),
                        )
                    )
                    if pooled_tokens.numel() > 1:
                        pending.append(
                            (
                                f"{prefix}/h_utilization_norm",
                                apply_stat_op(
                                    pooled_scores,
                                    AxisSpec(),
                                    StatOpConfig(op="raw_entropy_norm"),
                                ),
                            )
                        )

            if config.expert_weights:
                self._record_expert_weights(unit, pending, spreads, post_ops)

        materialize(pending, full_precision=frozenset(index for index, _ in post_ops))
        for index, finish in post_ops:
            finished = finish(pending[index][1]).to(torch.float32)
            pending[index] = (pending[index][0], finished)
        for prefix, norms in _split_packed_metrics(
            (prefixes, pending[index][1]) for prefixes, index in spreads
        ):
            for metric in config.weight_spread:
                # No implied dims here: the input is already the (E,) norm
                # vector, and AxisSpec() is empty, so a configured axis name
                # fails to resolve on its own rather than being silently
                # replaced.
                value = norms
                for step in self._stat_steps(metric, "weight_spread"):
                    value = apply_stat_op(value, AxisSpec(), step.stat)
                pending.append((f"{prefix}_{metric.name}", value))

        for name, tensor in _split_packed_metrics(pending):
            flat = tensor.detach().reshape(-1)
            if tensor.ndim == 0:
                self._records.append(
                    MetricRecord(
                        name=name,
                        value=flat[0],
                        step=self._step,
                        reduce=ReduceConfig(),
                    )
                )
                continue
            for expert in range(flat.numel()):
                self._records.append(
                    MetricRecord(
                        name=name.format(e=expert),
                        value=flat[expert],
                        step=self._step,
                        reduce=ReduceConfig(),
                    )
                )

    def _units_with_data_everywhere(
        self, state: ObserverState, have_data: list[bool]
    ) -> list[bool]:
        """Which units have router data on EVERY rank of the reduce groups."""
        assert self.reducer is not None
        mask = torch.tensor([1.0 if x else 0.0 for x in have_data], device=state.device)
        everywhere = [
            bool(value)
            for value in self.reducer.reduce_over_groups(
                mask, ("loss", "tp", "ep"), ReduceOp.min
            )
            .cpu()
            .tolist()
        ]
        for unit, mine, everyone in zip(self._units, have_data, everywhere):
            if mine and not everyone and unit.name not in self._partial_units_warned:
                self._partial_units_warned.add(unit.name)
                logger.warning(
                    "ModellingSource: MoE unit '%s' has router data on this "
                    "rank but not on every rank of the reduce groups; its "
                    "MoE statistics are skipped on such steps (in lockstep, "
                    "so nothing hangs). An empty batch on part of the ranks "
                    "does this transiently; rank-dependent include_patterns "
                    "do it permanently.",
                    unit.name,
                )
        return everywhere

    @staticmethod
    def _stat_steps(
        metric: MetricConfig, field_name: str
    ) -> tuple[MetricStepConfig, ...]:
        """The metric's steps, rejecting anything that is not a stat.

        Both expert-weight families run over a rank-local tensor — an expert
        parameter, or the norm vector derived from it — so a reduce step would
        have nothing to reduce and would quietly be a no-op.
        """
        steps = tuple(metric.effective_steps())
        if not steps:
            raise ValueError(
                f"MoE {field_name} metric '{metric.name}' has no steps; give it "
                "a pipeline of stat ops."
            )
        if any(step.kind != MetricStepKind.stat or step.stat is None for step in steps):
            raise ValueError(
                f"MoE {field_name} metric '{metric.name}' has a non-stat step; "
                "these run over a rank-local tensor and take stat steps only."
            )
        return steps

    def _iter_expert_weights(
        self, unit: MoEMonitorUnit
    ) -> Iterator[tuple[str, torch.Tensor, int, tuple[str, ...]]]:
        """Yield logical weights; packed ragged parameters stay whole."""
        assert self.reducer is not None
        param_names = {
            id(param): name for name, param in unit.experts.named_parameters()
        }
        slices = unit.experts.named_weight_slices()
        for param, num_experts in unit.experts.grouped_param_axes():
            param_name = param_names.get(id(param))
            if param_name is None:
                continue
            if isinstance(param, Float8BlockwiseQTensor):
                param = _local_tensor(param)
            parts = slices.get(param_name)
            if parts is None:
                yield param_name, param, num_experts, ()
            elif isinstance(param, self.reducer.ragged_tensor_types):
                if param.ndim != 4 or param.shape[1] != 2:
                    raise ValueError(
                        "Packed GFSDP statistics require shape (E, 2, F, D)"
                    )
                if [index for _, index in parts] != [
                    (slice(None), slice(0, 1)),
                    (slice(None), slice(1, 2)),
                ]:
                    raise ValueError(
                        "Packed GFSDP statistics require two contiguous weight halves"
                    )
                yield param_name, param, num_experts, tuple(name for name, _ in parts)
            else:
                for name, index in parts:
                    yield name, param[index], num_experts, ()

    @torch.no_grad()
    def _record_expert_weights(
        self,
        unit: MoEMonitorUnit,
        pending: list[tuple[_MetricName, torch.Tensor]],
        spreads: list[tuple[_MetricName, int]],
        post_ops: list[tuple[int, Callable[[torch.Tensor], torch.Tensor]]],
    ) -> None:
        """Per-expert statistics of every expert parameter.

        Decomposable statistics reduce small per-segment vectors across EP/FSDP
        shards. Order statistics share one full-weight gather per parameter.
        The ``weight_spread`` statistics summarise the norm vector as a whole,
        so they are not computed here: this method only points ``spreads`` at
        the vector, and the caller runs them once it is materialized.
        """
        assert self.reducer is not None
        support = self.reducer.shard_stats
        config = self.config.moe
        for param_name, param, num_experts, packed_names in self._iter_expert_weights(
            unit
        ):
            packed = bool(packed_names)
            # Dim 0 is the expert index, so a statistic of an expert parameter
            # reduces every OTHER dimension and comes out as one value per
            # expert.
            axes = AxisSpec(names=("expert",) + ("",) * (max(param.ndim, 1) - 1))
            expert_dims = tuple(range(1, param.ndim))
            base = f"{unit.name}.{param_name}"
            num_segments = num_experts * (2 if packed else 1)
            wide = param.detach().float()
            norms: int | None = None
            base_mean: torch.Tensor | None = None
            full_stats: list[
                tuple[str, _MetricName, tuple[MetricStepConfig, ...], StatOpConfig]
            ] = []
            for metric in config.weight_stats:
                *leading, last = self._stat_steps(metric, "weight_stats")
                if any(step.stat.dim is not None for step in (*leading, last)):
                    raise ValueError(
                        f"MoE weight_stats metric '{metric.name}' sets a dim, "
                        "but the dims are implied here and the setting would be "
                        "discarded: dim 0 is the expert index and the final step "
                        "reduces everything else onto it."
                    )

                names: _MetricName = f"{metric.name}/{base}__expert_idx_{{e}}"
                if packed:
                    names = tuple(
                        f"{metric.name}/{unit.name}.{name}__expert_idx_{{e}}"
                        for name in packed_names
                    )
                if last.stat.op in (
                    StatOpName.QUANTILE,
                    StatOpName.MIN_MEDIAN_RATIO,
                    StatOpName.GINI,
                ):
                    _assert_order_stat_size(last.stat.op, param.numel(), metric.name)
                    if packed and metric.dtensor_mode == DTensorMode.partial_stats:
                        _warn_full_tensor_fallback(metric, last.stat, param)
                    full_stats.append((metric.name, names, tuple(leading), last.stat))
                    continue

                vector = wide
                for step in leading:
                    vector = apply_stat_op(vector, axes, step.stat)
                if vector.shape != param.shape:
                    raise ValueError(
                        f"MoE weight_stats metric '{metric.name}': every step but "
                        "the last must be elementwise, so that the last one is "
                        "what reduces onto the expert axis. This pipeline already "
                        f"reduced {tuple(param.shape)} to {tuple(vector.shape)} "
                        "before reaching its last step."
                    )
                if support is not None and isinstance(
                    vector, self.reducer.ragged_tensor_types
                ):
                    if expert_dims:
                        if (
                            vector is wide
                            and last.stat.op
                            in (
                                StatOpName.STD,
                                StatOpName.VAR,
                                StatOpName.CV,
                                StatOpName.KURTOSIS,
                            )
                            and base_mean is None
                            and wide.shape[0]
                            and wide.numel()
                        ):
                            base_mean = _local_tensor(
                                support.sum_flat_segments(wide, num_segments),
                                full_tensor=True,
                                dtype=_ACC_DTYPE,
                            ) / (wide.numel() // num_segments)
                        vector, finish = _segment_expert_stat(
                            support,
                            vector,
                            last.stat,
                            metric.name,
                            mean=base_mean if vector is wide else None,
                            num_segments=num_segments,
                            num_groups=2 if packed else 1,
                        )
                    else:
                        vector, finish = (
                            support.sum_flat_segments(vector, vector.shape[0]),
                            None,
                        )
                    if finish is not None:
                        post_ops.append((len(pending), finish))
                elif expert_dims:
                    vector = apply_stat_op(
                        vector, axes, replace(last.stat, dim=expert_dims)
                    )

                pending.append((names, vector))
                if metric.name == config.norm_metric_name:
                    norms = len(pending) - 1

            if full_stats:
                full = _local_tensor(
                    wide, full_tensor=True, dtype=torch.float32, shard_stats=support
                )
                for name, names, leading, op in full_stats:
                    vector = full
                    for step in leading:
                        vector = apply_stat_op(vector, axes, step.stat)
                    if vector.shape != full.shape:
                        raise ValueError(
                            f"MoE weight_stats metric '{name}': every step but "
                            "the last must be elementwise."
                        )
                    if not vector.shape[0] or not vector.numel():
                        raise ValueError(
                            f"MoE weight_stats metric '{name}': expert "
                            "parameters are empty"
                        )
                    matrix = vector.reshape(num_segments, -1)
                    result = apply_stat_op(matrix, AxisSpec(), replace(op, dim=1))
                    pending.append((names, result))
                    if name == config.norm_metric_name:
                        norms = len(pending) - 1
                del matrix, vector, full

            if norms is None or num_experts <= 1:
                continue
            prefixes: _MetricName = (
                f"{config.namespace}/{unit.name}/expert_weights/{param_name}"
            )
            if packed:
                prefixes = tuple(
                    f"{config.namespace}/{unit.name}/expert_weights/{name}"
                    for name in packed_names
                )
            spreads.append((prefixes, norms))
