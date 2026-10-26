import operator
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from functools import reduce

import torch
from torch.distributed.tensor import DTensor
from torch.utils._python_dispatch import _disable_current_modes

from typing import TypeAlias

from torchtitan.config.schema.callbacks.sources.base import (
    FqnFilterConfig,
    MonitorSourceConfig,
)
from torchtitan.config.schema.callbacks.sources.base import FilteredSourceConfig
from torchtitan.callbacks.base import Observer, ObserverState
from torchtitan.config import Configurable
from torchtitan.callbacks.metrics.reducers import ParallelReducer
from torchtitan.callbacks.metrics.sample import MetricRecord, Sample
from torchtitan.config.schema.callbacks.metrics.sample import (
    DTensorMode,
    ReduceOp,
    AxisSpec,
    MetricConfig,
    ReduceConfig,
    ReduceStageConfig,
)
from torchtitan.callbacks.metrics.stats import _local_tensor, MetricPipeline
from torchtitan.ops.float8.fp8_tensor import Float8BlockwiseQTensor
from torchtitan.tools.logging import logger

__all__ = [
    "canonical_fqn",
    "FilteredSourceConfig",
    "MonitorSource",
]

_WRAPPER_FQN_PARTS = {
    "_checkpoint_wrapped_module",
    "_fsdp_wrapped_module",
    "_orig_mod",
}


def canonical_fqn(name: str) -> str:
    """Return a stable user-facing FQN by stripping framework wrapper segments."""
    return ".".join(
        part for part in name.split(".") if part and part not in _WRAPPER_FQN_PARTS
    )


def iter_logical_weights(
    model_parts: Iterable[torch.nn.Module],
) -> Iterator[tuple[str, torch.nn.Parameter, tuple]]:
    """``(fqn, parameter, index)`` per logical weight: a packed parameter once
    per seam it declares, every other parameter once with ``...``."""
    for model_part in model_parts:
        declare = getattr(model_part, "named_weight_slices", None)
        slices = declare() if declare is not None else {}
        for name, param in model_part.named_parameters():
            for view_name, index in slices.get(name, [(name, ...)]):
                fqn = canonical_fqn(view_name)
                if fqn:
                    yield fqn, param, index


@dataclass(slots=True)
class _CaptureAccum:
    """One tag's captures within a batch, combined when the batch ends."""

    # First reduce-stage op of the metric; None means "cannot combine".
    op: str | None
    reduce: ReduceConfig
    pending: tuple[ReduceStageConfig, ...]
    # Only read to name the metric in the "captured twice, no reduce op" error.
    metric_name: str
    values: list[torch.Tensor] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)


class MonitorSource(Configurable, Observer):
    """A monitoring source: an Observer that produces MetricRecords.

    Being an Observer is the invariant a source cannot opt out of: it reads
    the step and never changes it. Being ``Configurable`` is what puts it in
    the same config system as everything else: one ``Config`` per source,
    ``build()`` instead of a hand-written constructor call.

    ``active`` is assigned by MonitoringCallback immediately before batch_start,
    which is what lets batch_start keep Callback's one-argument signature.
    """

    Config: TypeAlias = MonitorSourceConfig

    def __init__(self, config: Config) -> None:
        self.config = config
        self.active = False
        self._active_before_validation = False
        self.reducer: ParallelReducer | None = None
        self.debug_full_tensor = False
        self._step = 0
        self._records: list[MetricRecord] = []
        self._capture_accum: dict[str, _CaptureAccum] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        if isinstance(config, FqnFilterConfig):
            self._include = [self._segment_regex(p) for p in config.include_patterns]
            self._exclude = [self._segment_regex(p) for p in config.exclude_patterns]
            self._raw_patterns = [*config.include_patterns, *config.exclude_patterns]
        else:
            self._include = []
            self._exclude = []
            self._raw_patterns = []

    def batch_start(self, state: ObserverState) -> None:
        self._step = state.step
        self._records.clear()
        self._capture_accum.clear()

    def validation_start(self, state: ObserverState) -> None:
        del state
        self._active_before_validation = self.active
        self.active = False

    def validation_end(self, state: ObserverState) -> None:
        del state
        self.active = self._active_before_validation

    def batch_end(self, state: ObserverState) -> None:
        del state
        self._flush_captures()

    def drain(self) -> list[MetricRecord]:
        records = self._records
        self._records = []
        return records

    def close(self) -> None:
        self._detach_hooks()

    def _detach_hooks(self) -> None:
        """Remove every hook this source registered."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _record(self, sample: Sample, *, accumulate: bool = False) -> None:
        """Compute every metric of the sample and emit one record per metric..

        With ``accumulate=True`` the computed values are stashed instead of
        recorded and combined into one record per tag when the batch ends —
        for hook-driven sources whose modules run several times per batch.
        """
        # Stats are computed at capture time: recorded tensors (grads, params,
        # activations) are mutated in place or freed later in the step, and
        # keeping them alive until batch_end would retain every captured
        # tensor of every microbatch in memory.
        if not self.active or torch.compiler.is_compiling() or not sample.metrics:
            return
        assert self.reducer is not None, "MonitorSource records before fit_start"
        with _disable_current_modes(), torch.no_grad():
            value = sample.value
            if isinstance(value, torch.Tensor):
                if isinstance(value, Float8BlockwiseQTensor):
                    value = _local_tensor(value)
                value = value.detach()
                if not value.is_floating_point():
                    value = value.float()
            weight = float(value.numel()) if isinstance(value, torch.Tensor) else 1.0
            stats = self.reducer.shard_stats
            ragged = isinstance(value, self.reducer.ragged_tensor_types)
            work: list[tuple[str, MetricConfig, MetricPipeline]] = []
            for metric in sample.metrics:
                variants = [("", metric)]
                if (
                    self.debug_full_tensor
                    and (isinstance(value, DTensor) or ragged)
                    and metric.dtensor_mode == DTensorMode.partial_stats
                ):
                    variants.append(
                        (
                            "debug_full/",
                            replace(metric, dtensor_mode=DTensorMode.full_tensor),
                        )
                    )
                for tag_prefix, variant in variants:
                    pipeline = MetricPipeline(variant, self.reducer)
                    name = tag_prefix + self._resolve_tag(sample.name, metric.name)
                    work.append((name, variant, pipeline))

            results = {}
            deferred = []
            for index, (_, _, pipeline) in enumerate(work):
                result = pipeline.compute_stats(
                    value, sample.axes, defer_full_tensor=ragged
                )
                if result is None:
                    deferred.append(index)
                else:
                    results[index] = result
            if deferred:
                assert isinstance(value, torch.Tensor)
                reduce_first: dict[tuple[ReduceStageConfig, ...], list[int]] = {}
                other_deferred = []
                for index in deferred:
                    prefix = work[index][2].collect_leading_reductions()
                    if prefix:
                        reduce_first.setdefault(prefix, []).append(index)
                    else:
                        other_deferred.append(index)
                full_value = _local_tensor(
                    value, full_tensor=True, dtype=torch.float32, shard_stats=stats
                )
                try:
                    for index in other_deferred:
                        pipeline = work[index][2]
                        result = pipeline.compute_stats(
                            value, sample.axes, full_value=full_value
                        )
                        assert result is not None
                        results[index] = result
                    # Share each leading reduction within this sample, then
                    # release its full-sized result before the next prefix.
                    copies = self.reducer.ranks_of(value.device_mesh)
                    for prefix, indices in reduce_first.items():
                        reduced, reduced_copies = self.reducer.reduce_counting_copies(
                            full_value, prefix, copies
                        )
                        for index in indices:
                            pipeline = work[index][2]
                            results[index] = (
                                pipeline.compute_after_reductions(
                                    reduced, sample.axes, copies=reduced_copies
                                ),
                                (),
                                (),
                            )
                        del reduced
                finally:
                    del full_value

            for index, (name, variant, pipeline) in enumerate(work):
                computed, pending, post = results[index]
                if accumulate:
                    if post:
                        computed = pipeline.compute(value, sample.axes)
                        pending = ()
                    self._accumulate(name, variant, computed, weight, pending)
                else:
                    self._records.append(
                        MetricRecord(
                            name=name,
                            value=computed,
                            step=self._step,
                            reduce=variant.reduce,
                            pending_stages=pending,
                            post_stats=post,
                            post_axes=sample.axes if post else AxisSpec(),
                        )
                    )

    @staticmethod
    def _is_recompute() -> bool:
        """Whether this call is an activation-checkpoint recompute, not a forward."""
        if torch.compiler.is_compiling():
            return False
        task_id = getattr(torch._C, "_current_graph_task_id", None)
        return task_id is not None and task_id() != -1

    @staticmethod
    def _resolve_tag(template: str, metric_name: str) -> str:
        if "{metric}" in template:
            return template.replace("{metric}", metric_name)
        return f"{template}/{metric_name}"

    def _accumulate(
        self,
        name: str,
        metric: MetricConfig,
        value: torch.Tensor,
        weight: float,
        pending: tuple[ReduceStageConfig, ...],
    ) -> None:
        """Stash one capture of a tag; combining waits for the batch to end.

        ``metric`` is carried on the accumulator rather than re-read at flush
        time because a tag is captured by exactly one metric.
        """
        stages = metric.reduce.effective_stages()
        accum = self._capture_accum.setdefault(
            name,
            _CaptureAccum(
                op=stages[0].op if stages else None,
                reduce=metric.reduce,
                pending=pending,
                metric_name=metric.name,
            ),
        )
        accum.values.append(value)
        accum.weights.append(weight)

    def _flush_captures(self) -> None:
        """Combine each tag's captures into one record.

        Outside the checkpointed region, so the combiners are free to branch.
        The weighted mean folds pairwise in capture order, which is how it read
        when it ran on arrival — a batch ``sum(v*w)/sum(w)`` would round
        differently and move every mean tag.
        """
        for name, accum in self._capture_accum.items():
            values, weights = accum.values, accum.weights
            if len(values) > 1 and accum.op is None:
                raise ValueError(
                    f"MonitorSource: '{name}' was captured {len(values)} times "
                    f"in one batch, but metric '{accum.metric_name}' declares "
                    "no reduce stages, so there is no operator to combine the "
                    "captures with. Give the metric a reduce op, or stop "
                    "hooking a module that runs several times per microbatch."
                )
            if accum.op == ReduceOp.max:
                value = reduce(torch.maximum, values)
            elif accum.op == ReduceOp.min:
                value = reduce(torch.minimum, values)
            elif accum.op == ReduceOp.sum:
                value = reduce(operator.add, values)
            else:
                value, carried = values[0], weights[0]
                for extra, extra_weight in zip(values[1:], weights[1:]):
                    if extra_weight == 0:
                        continue
                    if carried == 0:
                        value, carried = extra, extra_weight
                        continue
                    total = carried + extra_weight
                    value = value * (carried / total) + extra * (extra_weight / total)
                    carried = total
            self._records.append(
                MetricRecord(
                    name=name,
                    value=value,
                    step=self._step,
                    reduce=accum.reduce,
                    pending_stages=accum.pending,
                )
            )
        self._capture_accum.clear()

    def _matches(self, name: str) -> bool:
        if self._include and not any(pattern.search(name) for pattern in self._include):
            return False
        return not any(pattern.search(name) for pattern in self._exclude)

    @staticmethod
    def _segment_regex(pattern: str) -> re.Pattern:
        """The pattern, anchored to whole dot-segments of the FQN.

        A bare ``re.search`` made ``layers.1`` drop layers 1, 10 AND 11 — a
        substring match plus the unescaped dot. Both ends are pinned to a
        segment boundary instead, so a pattern matches a whole run of
        segments: ``layers.1`` matches ``layers.1.attention.wq`` and
        ``mtp.layers.1.f`` but not ``layers.10``; ``wq`` does not match
        ``wq_norm``. Everything inside is still a regex, and ``^`` keeps
        working — ``^layers\\.1`` pins to the top level, which is how the MTP
        head's ``mtp.layers.1`` is kept while ``layers.1`` is dropped.
        """
        return re.compile(rf"(?:^|(?<=\.))(?:{pattern})(?=\.|$)")

    def _filter_fqns(self, names: Sequence[str], what: str) -> set[str]:
        """The names the patterns keep, saying out loud what the filter did.

        Silence is how a typo'd pattern hides: one that matched nothing and
        one that dropped a third of the model would otherwise look identical.
        A configured filter therefore logs its kept/total count, and every
        pattern that matched no name at all warns.
        """
        if not self._include and not self._exclude:
            return set(names)
        kept = {name for name in names if self._matches(name)}
        for raw, compiled in zip(self._raw_patterns, [*self._include, *self._exclude]):
            if not any(compiled.search(name) for name in names):
                logger.warning(
                    "%s: pattern %r matched none of the %d %s — likely a typo, "
                    "or a substring of a segment (patterns match whole "
                    "dot-segments; use e.g. '.*proj' for a partial one).",
                    type(self).__name__,
                    raw,
                    len(names),
                    what,
                )
        logger.info(
            "%s: include/exclude patterns keep %d of %d %s",
            type(self).__name__,
            len(kept),
            len(names),
            what,
        )
        return kept
