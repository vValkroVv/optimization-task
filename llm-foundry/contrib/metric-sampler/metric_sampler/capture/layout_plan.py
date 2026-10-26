from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from metric_sampler.capture.stats.activations import point_slots
from metric_sampler.capture.stats.attention import linear_slots, linear_step_slots, softmax_slots
from metric_sampler.capture.stats.helpers.quantiles import quantile_slots
from metric_sampler.contract.catalog import metric_by_name
from metric_sampler.contract.metrics import (
    EXPERT_LOSS_BINS,
    LAYER_ACTIVATION_POINTS,
    LOSS_POSITION_BUCKETS,
    NO_LAYER,
    NO_MICROBATCH,
    ROUTER_ENTROPY_BINS,
    SAMPLE_LOSS_BINS,
    STANDALONE_ACTIVATION_POINTS,
    TOKEN_LOSS_BINS,
    WORST_SAMPLE_COUNT,
    WORST_SAMPLE_FIELDS,
    MetricSpec,
)

_SCALAR_WIDTH = 1
_WORST_SAMPLE_STATS_FIELDS = 2

# The dispatcher's assignment counts in the order its observer gives them.
DISPATCHER_METRICS = (
    "dispatcher/assignments_before_capacity",
    "dispatcher/valid_assignments_before_capacity",
    "dispatcher/assignments_after_capacity",
    "dispatcher/valid_assignments_after_capacity",
)
STEP_SCALARS = (
    "time/step_s",
    "time/loop_s",
    "lr",
    "sampler/dropped_steps",
    "sampler/microbatches",
    "sampler/step_end_s",
    "sampler/ring_used_slots",
    "sampler/ring_capacity_slots",
)
MEMORY_METRICS = ("memory/allocated_gb", "memory/peak_allocated_gb", "memory/reserved_gb", "memory/alloc_retries")

# (metric name, layer, microbatch, width)
SlotRequest = tuple[str, int, int, int]


@dataclass(frozen=True)
class SoftmaxAttentionLayer:
    layer: int
    heads: int
    gated: bool


@dataclass(frozen=True)
class LinearAttentionLayer:
    layer: int
    value_heads: int


@dataclass(frozen=True)
class LayoutPlan:
    """The slots one rank's layout holds, from the shape of the model and the metric families captured.

    `extra` are the slots of observers the training framework's integration adds on its own (its monitors'),
    placed after the sampler's."""

    microbatches: int
    routed_experts: int
    streams: int
    loss_metrics: tuple[str, ...]
    counting_gate_layers: tuple[int, ...] = ()
    dispatcher_layers: tuple[int, ...] = ()
    routed_layers: tuple[int, ...] = ()
    mtp_layers: tuple[int, ...] = ()
    pre_reduce_keys: frozenset[tuple[str, int]] = frozenset()
    activation_layers: dict[str, tuple[int, ...]] = field(default_factory=dict)
    softmax_attention: tuple[SoftmaxAttentionLayer, ...] = ()
    linear_attention: tuple[LinearAttentionLayer, ...] = ()
    token_loss: bool = True
    experts: bool = True
    activations: bool = True
    exact_quantiles: bool = False
    dumps: bool = False
    extra: tuple[SlotRequest, ...] = ()

    def requests(self) -> list[tuple[MetricSpec, int, int, int]]:
        # Layout requires every non-detailed slot before any detailed one.
        base: list[tuple[MetricSpec, int, int, int]] = []
        detailed: list[tuple[MetricSpec, int, int, int]] = []
        for name, layer, microbatch, width in self._slots():
            metric = metric_by_name(name)
            (detailed if metric.detailed else base).append((metric, layer, microbatch, width))
        return base + detailed

    def _slots(self) -> Iterator[SlotRequest]:
        yield from self._router_counts()
        yield from self._microbatch_scalars()
        yield from self._step_scalars()
        yield from self._distributions()
        if self.exact_quantiles:
            yield from self._quantiles()
        yield from self._attention()
        yield from self.extra

    @property
    def _expert_layers(self) -> tuple[int, ...]:
        return self.routed_layers if self.experts else ()

    def _router_counts(self) -> Iterator[SlotRequest]:
        for layer in self.counting_gate_layers:
            yield "router/local_tokens_before_bias_update", layer, NO_MICROBATCH, self.routed_experts
            yield "router/expert_bias_before_update", layer, NO_MICROBATCH, self.routed_experts
        for layer in self.dispatcher_layers:
            for microbatch in range(self.microbatches):
                for name in DISPATCHER_METRICS:
                    yield name, layer, microbatch, self.routed_experts

    def _microbatch_scalars(self) -> Iterator[SlotRequest]:
        for microbatch in range(self.microbatches):
            for name in self.loss_metrics:
                yield name, NO_LAYER, microbatch, _SCALAR_WIDTH
            yield "time/forward_gpu_s", NO_LAYER, microbatch, _SCALAR_WIDTH
            yield "time/backward_gpu_s", NO_LAYER, microbatch, _SCALAR_WIDTH
            if self.token_loss:
                yield "loss/token_max", NO_LAYER, microbatch, _SCALAR_WIDTH
                yield "loss/nonfinite_tokens", NO_LAYER, microbatch, _SCALAR_WIDTH
                yield "loss/worst_sample_info", NO_LAYER, microbatch, WORST_SAMPLE_COUNT * WORST_SAMPLE_FIELDS
                yield "loss/worst_sample_stats", NO_LAYER, microbatch, WORST_SAMPLE_COUNT * _WORST_SAMPLE_STATS_FIELDS

    def _step_scalars(self) -> Iterator[SlotRequest]:
        for name in STEP_SCALARS:
            yield name, NO_LAYER, NO_MICROBATCH, _SCALAR_WIDTH
        yield "sampler/missing_slots", NO_LAYER, NO_MICROBATCH, _SCALAR_WIDTH
        if self.dumps:
            yield "sampler/dumps_skipped", NO_LAYER, NO_MICROBATCH, _SCALAR_WIDTH
        for name in MEMORY_METRICS:
            yield name, NO_LAYER, NO_MICROBATCH, _SCALAR_WIDTH
        yield "time/step_gpu_s", NO_LAYER, NO_MICROBATCH, _SCALAR_WIDTH
        yield "time/optimizer_gpu_s", NO_LAYER, NO_MICROBATCH, _SCALAR_WIDTH
        yield "grad/clip_total_norm", NO_LAYER, NO_MICROBATCH, _SCALAR_WIDTH
        for name, layer in sorted(self.pre_reduce_keys):
            yield name, layer, NO_MICROBATCH, _SCALAR_WIDTH

    def _distributions(self) -> Iterator[SlotRequest]:
        for microbatch in range(self.microbatches):
            if self.token_loss:
                yield from self._token_loss_distributions(microbatch)
            for layer in self._expert_layers:
                yield from self._expert_distributions(layer, microbatch)
            if self.activations:
                yield from self._activation_distributions(microbatch)

    def _token_loss_distributions(self, microbatch: int) -> Iterator[SlotRequest]:
        streams = self.streams
        yield "loss/token_hist_by_stream", NO_LAYER, microbatch, streams * TOKEN_LOSS_BINS.count
        for layer in self.mtp_layers:
            yield "loss/mtp_token_hist", layer, microbatch, TOKEN_LOSS_BINS.count
        yield "loss/stream_sum", NO_LAYER, microbatch, streams
        yield "loss/stream_sumsq", NO_LAYER, microbatch, streams
        yield "loss/stream_count", NO_LAYER, microbatch, streams
        yield "loss/stream_samples", NO_LAYER, microbatch, streams
        yield "loss/position_sum", NO_LAYER, microbatch, LOSS_POSITION_BUCKETS
        yield "loss/position_count", NO_LAYER, microbatch, LOSS_POSITION_BUCKETS
        yield "loss/sample_hist_by_stream", NO_LAYER, microbatch, streams * SAMPLE_LOSS_BINS.count

    def _expert_distributions(self, layer: int, microbatch: int) -> Iterator[SlotRequest]:
        streams, experts = self.streams, self.routed_experts
        yield "router/tokens_per_expert", layer, microbatch, experts
        yield "router/stream_tokens_per_expert", layer, microbatch, experts * streams
        yield "expert_loss/sum", layer, microbatch, experts
        yield "expert_loss/sumsq", layer, microbatch, experts
        yield "expert_loss/count", layer, microbatch, experts
        yield "expert_loss/hist", layer, microbatch, experts * EXPERT_LOSS_BINS.count
        yield "expert_loss/stream_count", layer, microbatch, experts * streams
        yield "expert_loss/stream_sum", layer, microbatch, experts * streams
        yield "router/entropy_hist", layer, microbatch, ROUTER_ENTROPY_BINS.count

    def _activation_distributions(self, microbatch: int) -> Iterator[SlotRequest]:
        for point in LAYER_ACTIVATION_POINTS:
            for layer in self.activation_layers.get(point, ()):
                for name, width in point_slots(point):
                    yield name, layer, microbatch, width
        for point in STANDALONE_ACTIVATION_POINTS:
            for name, width in point_slots(point):
                yield name, NO_LAYER, microbatch, width

    def _quantiles(self) -> Iterator[SlotRequest]:
        for microbatch in range(self.microbatches):
            if self.token_loss:
                yield from self._token_loss_quantiles(microbatch)
            for layer in self._expert_layers:
                for name, width in quantile_slots(
                    "router/entropy_quantiles", "router/entropy_quantile_counts", float64=True
                ):
                    yield name, layer, microbatch, width
            if self.activations:
                yield from self._activation_quantiles(microbatch)

    def _token_loss_quantiles(self, microbatch: int) -> Iterator[SlotRequest]:
        for name, width in quantile_slots("loss/token_quantiles", "loss/token_quantile_counts"):
            yield name, NO_LAYER, microbatch, width
        for layer in self.mtp_layers:
            for name, width in quantile_slots("loss/mtp_token_quantiles", "loss/mtp_token_quantile_counts"):
                yield name, layer, microbatch, width

    def _activation_quantiles(self, microbatch: int) -> Iterator[SlotRequest]:
        for point in LAYER_ACTIVATION_POINTS:
            for layer in self.activation_layers.get(point, ()):
                for name, width in quantile_slots(f"act/{point}/quantiles", f"act/{point}/quantile_counts"):
                    yield name, layer, microbatch, width
        for point in STANDALONE_ACTIVATION_POINTS:
            for name, width in quantile_slots(f"act/{point}/quantiles", f"act/{point}/quantile_counts"):
                yield name, NO_LAYER, microbatch, width

    def _attention(self) -> Iterator[SlotRequest]:
        for microbatch in range(self.microbatches):
            for attention in self.softmax_attention:
                for name, width in softmax_slots(attention.heads, attention.gated):
                    yield name, attention.layer, microbatch, width
            for linear in self.linear_attention:
                for name, width in linear_slots(linear.value_heads):
                    yield name, linear.layer, microbatch, width
        for linear in self.linear_attention:
            for name, width in linear_step_slots(linear.value_heads):
                yield name, linear.layer, NO_MICROBATCH, width
