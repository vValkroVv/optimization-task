from __future__ import annotations

import sys
import time
import uuid
from dataclasses import dataclass, replace

import torch

from metric_sampler.capture.layout import Layout
from metric_sampler.capture.recorder import RankRecorder
from metric_sampler.capture.ring import ShmRing
from metric_sampler.capture.stats.token_loss import TokenLossStats
from metric_sampler.contract.catalog import metric_by_name
from metric_sampler.contract.metrics import (
    EXPERT_LOSS_BINS,
    LOSS_POSITION_BUCKETS,
    ROUTER_ENTROPY_BINS,
    SAMPLE_LOSS_BINS,
    TOKEN_LOSS_BINS,
    WORST_SAMPLE_COUNT,
    WORST_SAMPLE_FIELDS,
    ValueType,
)

_COUNT_METRICS = (
    "router/tokens_per_expert",
    "router/stream_tokens_per_expert",
    "loss/token_hist_by_stream",
    "loss/mtp_token_hist",
    "loss/stream_count",
    "loss/stream_samples",
    "loss/position_count",
    "loss/nonfinite_tokens",
    "loss/sample_hist_by_stream",
    "expert_loss/count",
    "expert_loss/hist",
    "expert_loss/stream_count",
    "router/entropy_hist",
)
_SUM_METRICS = (
    "loss/stream_sum",
    "loss/stream_sumsq",
    "loss/position_sum",
    "expert_loss/sum",
    "expert_loss/sumsq",
    "expert_loss/stream_sum",
    "loss/worst_sample_stats",
)
_EXACT_METRICS = ("loss/token_max", "loss/worst_sample_info")
_SUM_REL_TOL = 1e-5
_SUM_ABS_TOL = 1e-4
_MILLISECONDS_PER_SECOND = 1000
_STREAMS = 44
# A label is invalid when its uniform draw is at most the threshold.
_INVALID_LABEL_THRESHOLD = 0.05
_BATCH_SIZE = 12
_SEQUENCE_LENGTH = 4096
_MICROBATCHES = 2
_TIMING_REPEATS = 30
_TIMING_WARMUP = 5


@dataclass(frozen=True)
class _Shape:
    name: str
    layers: int
    experts: int
    top_k: int
    mtp_heads: int
    batch_size: int = _BATCH_SIZE
    sequence_length: int = _SEQUENCE_LENGTH

    @property
    def routed_layers(self) -> dict[int, int]:
        main = dict.fromkeys(range(self.layers), 0)
        mtp = {self.layers + head: head + 1 for head in range(self.mtp_heads)}
        return {**main, **mtp}

    @property
    def mtp_layers(self) -> list[int]:
        return [self.layers + head for head in range(self.mtp_heads)]

    @property
    def segments(self) -> int:
        return 1 + self.mtp_heads


_SHAPES = [
    _Shape(name="L8", layers=8, experts=256, top_k=8, mtp_heads=mtp_heads)
    for mtp_heads in (1, 2)
] + [
    _Shape(name="L16", layers=16, experts=256, top_k=8, mtp_heads=mtp_heads)
    for mtp_heads in (1, 2)
] + [
    _Shape(name="61 layers", layers=61, experts=256, top_k=8, mtp_heads=mtp_heads)
    for mtp_heads in (1, 2)
] + [
    _Shape(
        name="L8 partial tiles", layers=8, experts=256, top_k=8, mtp_heads=1, batch_size=3,
        sequence_length=_SEQUENCE_LENGTH - 37,
    )
]


def _layout_requests(shape: _Shape) -> list[tuple]:
    base: list[tuple] = []
    detailed: list[tuple] = []

    def add(name: str, layer: int, microbatch: int, width: int) -> None:
        metric = metric_by_name(name)
        (detailed if metric.detailed else base).append((metric, layer, microbatch, width))

    for microbatch in range(_MICROBATCHES):
        add("loss/nonfinite_tokens", -1, microbatch, 1)
        add("loss/token_max", -1, microbatch, 1)
        add("loss/token_hist_by_stream", -1, microbatch, _STREAMS * TOKEN_LOSS_BINS.count)
        for mtp_layer in shape.mtp_layers:
            add("loss/mtp_token_hist", mtp_layer, microbatch, TOKEN_LOSS_BINS.count)
        add("loss/stream_sum", -1, microbatch, _STREAMS)
        add("loss/stream_sumsq", -1, microbatch, _STREAMS)
        add("loss/stream_count", -1, microbatch, _STREAMS)
        add("loss/stream_samples", -1, microbatch, _STREAMS)
        add("loss/position_sum", -1, microbatch, LOSS_POSITION_BUCKETS)
        add("loss/position_count", -1, microbatch, LOSS_POSITION_BUCKETS)
        add("loss/sample_hist_by_stream", -1, microbatch, _STREAMS * SAMPLE_LOSS_BINS.count)
        add("loss/worst_sample_info", -1, microbatch, WORST_SAMPLE_COUNT * WORST_SAMPLE_FIELDS)
        add("loss/worst_sample_stats", -1, microbatch, WORST_SAMPLE_COUNT * 2)
        for layer in shape.routed_layers:
            add("router/tokens_per_expert", layer, microbatch, shape.experts)
            add("router/stream_tokens_per_expert", layer, microbatch, shape.experts * _STREAMS)
            add("expert_loss/sum", layer, microbatch, shape.experts)
            add("expert_loss/sumsq", layer, microbatch, shape.experts)
            add("expert_loss/count", layer, microbatch, shape.experts)
            add("expert_loss/hist", layer, microbatch, shape.experts * EXPERT_LOSS_BINS.count)
            add("expert_loss/stream_count", layer, microbatch, shape.experts * _STREAMS)
            add("expert_loss/stream_sum", layer, microbatch, shape.experts * _STREAMS)
            add("router/entropy_hist", layer, microbatch, ROUTER_ENTROPY_BINS.count)
    return base + detailed


@dataclass
class _Rig:
    """A `TokenLossStats` bound to its own ring/recorder, sharing one `Layout` with its peer so
    both land the same metric at the same buffer offset."""

    recorder: RankRecorder
    ring: ShmRing
    stats: TokenLossStats

    def close(self) -> None:
        self.recorder.close()
        self.ring.unlink()


def _make_rig(layout: Layout, shape: _Shape, device: torch.device) -> _Rig:
    ring = ShmRing.create(f"ms_check_{uuid.uuid4().hex[:20]}", layout, slot_count=1)
    recorder = RankRecorder(layout, ring, device)
    stats = TokenLossStats(recorder, shape.routed_layers, shape.mtp_layers, shape.experts, shape.top_k, _STREAMS)
    return _Rig(recorder, ring, stats)


@dataclass
class _Inputs:
    topk_indices: dict[int, torch.Tensor]
    scores: dict[int, torch.Tensor]
    loss: torch.Tensor
    labels_valid: torch.Tensor
    stream_ids: torch.Tensor
    sample_ids: torch.Tensor


def _make_inputs(shape: _Shape, device: torch.device, generator: torch.Generator) -> _Inputs:
    topk_indices, scores = {}, {}
    for layer in shape.routed_layers:
        logits = torch.randn(
            shape.batch_size * shape.sequence_length, shape.experts, generator=generator, device=device
        ) * 2.0
        scores[layer] = torch.softmax(logits, dim=-1)
        topk_indices[layer] = torch.topk(scores[layer], shape.top_k, dim=-1).indices.to(torch.int64)
    segment_shape = (shape.segments, shape.batch_size, shape.sequence_length)
    loss = (torch.rand(segment_shape, generator=generator, device=device) * 4.0)
    labels_valid = torch.rand(segment_shape, generator=generator, device=device) > _INVALID_LABEL_THRESHOLD
    # A pinch of NaN/Inf and fully-invalid samples, like a real batch's padding and rare blowups.
    loss[0, 0, :3] = float("nan")
    if shape.batch_size > 1:
        labels_valid[0, 1, :] = False
    stream_ids = torch.randint(0, _STREAMS, (shape.batch_size,), generator=generator, device=device)
    sample_ids = torch.stack(
        (
            torch.zeros(shape.batch_size, dtype=torch.int64, device=device),
            torch.arange(shape.batch_size, device=device),
        ),
        dim=-1,
    )
    return _Inputs(topk_indices, scores, loss, labels_valid, stream_ids, sample_ids)


def _reference_observe_routing(
    stats: TokenLossStats, layer: int, topk_indices: torch.Tensor, scores: torch.Tensor
) -> None:
    # Bypasses TokenLossStats.observe_routing's CUDA dispatch to force the torch-op reference path on a CUDA tensor.
    position = stats._layer_positions[layer]
    tokens = topk_indices.shape[0]
    if stats._topk_buffer is None or tokens > stats._topk_buffer.shape[1]:
        stats._topk_buffer = torch.zeros(
            (len(stats._routed_layers), topk_indices.shape[0], stats._top_k), dtype=torch.int32,
            device=topk_indices.device,
        )
    stats._topk_tokens = tokens
    stats._observed_layers.add(layer)
    stats._topk_buffer[position, :tokens].copy_(topk_indices)
    counts = torch.bincount(topk_indices.flatten().long(), minlength=stats._experts).to(torch.int32)
    stats._recorder.record("router/tokens_per_expert", layer, stats._microbatch, counts)
    tokens_per_sample = stats._tokens_per_sample(tokens, stats._stream_ids)
    stream_index = stats._stream_ids.to(topk_indices.device).long().repeat_interleave(tokens_per_sample)
    combined = topk_indices.flatten().long() * stats._streams + stream_index[:, None].expand(-1, stats._top_k).flatten()
    stream_counts = torch.bincount(combined, minlength=stats._experts * stats._streams).to(torch.int32)
    stats._recorder.record("router/stream_tokens_per_expert", layer, stats._microbatch, stream_counts)
    stats._record_entropy_histogram(layer, scores.detach())


def _reference_observe_loss(stats: TokenLossStats, loss: torch.Tensor, labels_valid: torch.Tensor) -> None:
    finite = torch.isfinite(loss)
    valid = labels_valid & finite
    nonfinite = (labels_valid & ~finite).sum()
    stats._recorder.record("loss/nonfinite_tokens", -1, stats._microbatch, nonfinite.to(torch.int32))
    masked_loss = torch.where(valid, loss, torch.zeros_like(loss))
    batch_size, sequence_length = loss.shape[1], loss.shape[2]
    stream_index = stats._stream_index(batch_size, sequence_length, loss.device)
    stats._record_token_max(masked_loss[0])
    stats._record_main_stream_stats(masked_loss[0], valid[0], stream_index)
    stats._record_position_stats(masked_loss[0], valid[0], sequence_length)
    if stats._detailed:
        stats._record_sample_histogram_by_stream(masked_loss[0], valid[0], stream_index, labels_valid[0] & ~finite[0])
    stats._record_mtp_histograms(masked_loss, valid)
    stats._record_expert_loss(masked_loss, valid, stream_index)


def _run(rig: _Rig, microbatches: tuple[_Inputs, ...], detailed: bool, observe_routing, observe_loss) -> None:
    rig.recorder.begin_step(detailed)
    rig.stats.begin_step(detailed)
    for microbatch, inputs in enumerate(microbatches):
        rig.stats.begin_microbatch(microbatch, inputs.stream_ids, inputs.sample_ids)
        for layer, topk_indices in inputs.topk_indices.items():
            observe_routing(rig.stats, layer, topk_indices, inputs.scores[layer])
        observe_loss(rig.stats, inputs.loss, inputs.labels_valid)


def _read(recorder: RankRecorder, name: str, layer: int, microbatch: int) -> torch.Tensor:
    slot = recorder._layout.slot(name, layer, microbatch)
    view = recorder.buffer[slot.offset : slot.offset + slot.width]
    return view.view(torch.float32) if metric_by_name(name).value_type is ValueType.FLOAT32 else view


def _compare(shape: _Shape, detailed: bool, reference: RankRecorder, got: RankRecorder) -> list[str]:
    problems = []
    names = list(_COUNT_METRICS) + list(_SUM_METRICS) + list(_EXACT_METRICS)
    layers_by_name = {
        "loss/mtp_token_hist": shape.mtp_layers,
        "expert_loss/sum": list(shape.routed_layers),
        "expert_loss/sumsq": list(shape.routed_layers),
        "expert_loss/count": list(shape.routed_layers),
        "expert_loss/hist": list(shape.routed_layers),
        "expert_loss/stream_count": list(shape.routed_layers),
        "expert_loss/stream_sum": list(shape.routed_layers),
        "router/entropy_hist": list(shape.routed_layers),
        "router/tokens_per_expert": list(shape.routed_layers),
        "router/stream_tokens_per_expert": list(shape.routed_layers),
    }
    for name in names:
        if metric_by_name(name).detailed and not detailed:
            continue
        for layer, microbatch in [(layer, microbatch) for layer in layers_by_name.get(name, [-1])
                                  for microbatch in range(_MICROBATCHES)]:
            label = f"{shape.name} mtp={shape.mtp_heads} detailed={detailed} {name} layer={layer} mb={microbatch}"
            expected = _read(reference, name, layer, microbatch)
            actual = _read(got, name, layer, microbatch)
            if name in _SUM_METRICS:
                if not torch.allclose(expected, actual, rtol=_SUM_REL_TOL, atol=_SUM_ABS_TOL):
                    problems.append(f"{label}: max abs diff {(expected - actual).abs().max().item():.3e}")
            elif not torch.equal(expected, actual):
                bad = (expected != actual).nonzero().flatten()[:5].tolist()
                problems.append(
                    f"{label}: mismatch at {bad} (ref {expected[bad[:5]].tolist()} vs got {actual[bad[:5]].tolist()})"
                )
    return problems


def _check_correctness(device: torch.device) -> list[str]:
    # Imported here: it needs triton, which the unit tests of the oracle run without.
    from metric_sampler.capture.kernels import triton_launch

    problems = []
    for shape in _SHAPES:
        layout = Layout(_layout_requests(shape), stream_names=tuple(f"s{i}" for i in range(_STREAMS)))
        for detailed in (False, True):
            generator = torch.Generator(device=device).manual_seed(0)
            inputs = (
                _make_inputs(shape, device, generator),
                _make_inputs(replace(shape, batch_size=shape.batch_size - 1), device, generator),
            )
            reference_rig = _make_rig(layout, shape, device)
            got_rig = _make_rig(layout, shape, device)
            jit_rig = _make_rig(layout, shape, device)
            _run(reference_rig, inputs, detailed, _reference_observe_routing, _reference_observe_loss)
            _run(got_rig, inputs, detailed, TokenLossStats.observe_routing, TokenLossStats.observe_loss)
            with triton_launch.jit_launches():
                _run(jit_rig, inputs, detailed, TokenLossStats.observe_routing, TokenLossStats.observe_loss)
            torch.cuda.synchronize()
            problems.extend(_compare(shape, detailed, reference_rig.recorder, got_rig.recorder))
            problems.extend(
                f"JIT launches: {problem}" for problem in _compare(shape, detailed, jit_rig.recorder, got_rig.recorder)
            )
            reference_rig.close()
            got_rig.close()
            jit_rig.close()
    return problems


def _time_cuda(function, repeats: int, warmup: int = _TIMING_WARMUP) -> tuple[float, float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    cpu_start = time.perf_counter()
    start.record()
    for _ in range(repeats):
        function()
    end.record()
    cpu_issue_s = (time.perf_counter() - cpu_start) / repeats
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats, cpu_issue_s


def _time_microbatch(device: torch.device) -> None:
    print(f"\n{'shape':30s} {'mtp':>3s} {'path':>10s} {'gpu ms/microbatch':>18s} {'cpu issue ms':>13s}")
    for shape in _SHAPES:
        layout = Layout(_layout_requests(shape), stream_names=tuple(f"s{i}" for i in range(_STREAMS)))
        generator = torch.Generator(device=device).manual_seed(1)
        inputs = _make_inputs(shape, device, generator)

        for label, observe_routing, observe_loss in (
            ("torch ref", _reference_observe_routing, _reference_observe_loss),
            ("triton", TokenLossStats.observe_routing, TokenLossStats.observe_loss),
        ):
            rig = _make_rig(layout, shape, device)
            gpu_ms, cpu_ms = _time_cuda(
                lambda: _run(rig, (inputs,), False, observe_routing, observe_loss), _TIMING_REPEATS
            )
            rig.close()
            print(
                f"{shape.name:30s} {shape.mtp_heads:>3d} {label:>10s} {gpu_ms:>18.4f} "
                f"{cpu_ms * _MILLISECONDS_PER_SECOND:>13.4f}"
            )


def main() -> None:
    if not torch.cuda.is_available():
        print("this check needs a CUDA device", file=sys.stderr)
        sys.exit(1)
    device = torch.device("cuda")

    problems = _check_correctness(device)
    if problems:
        print(f"{len(problems)} mismatch(es):")
        for problem in problems:
            print(f"  {problem}")
        sys.exit(1)
    else:
        print("all shapes match within tolerance")

    _time_microbatch(device)


if __name__ == "__main__":
    main()
