"""The extension's routing observers write every word the callback's Python ones write: the routing kernel's top-k
buffer, tokens-per-expert counts and entropy histograms, and the exact entropy quantiles, whether the extension
selects them, the scores are too many for its selection or a strided view goes to the Python selection."""

import functools
import uuid

import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.gpu]

pytest.importorskip("composer", reason="Requires the training runtime")

from llmfoundry.callbacks.metric_sampler import MetricSampler  # noqa: E402
from metric_sampler.capture.kernels import activation_ext  # noqa: E402
from metric_sampler.capture.layout import Layout  # noqa: E402
from metric_sampler.capture.recorder import RankRecorder  # noqa: E402
from metric_sampler.capture.ring import ShmRing  # noqa: E402
from metric_sampler.capture.stats.quantiles import ExactQuantileStats  # noqa: E402
from metric_sampler.capture.stats.token_loss import TokenLossStats  # noqa: E402

requires_extension = pytest.mark.skipif(not torch.cuda.is_available() or activation_ext.prebuilt() is None,
                                        reason="Requires CUDA and the prebuilt extension")

EXPERTS, TOP_K = 16, 4
STREAMS = 3
ROUTED = {0: 0, 1: 0, 2: 0}
ACTIVATION_POINTS = (
    "residual", "attn_out", "mlp_out", "shared_expert_out", "attn_residual_contribution",
    "mlp_residual_contribution", "swiglu_in", "down_in",
)
ROUTING_METRICS = {
    "router/tokens_per_expert", "router/stream_tokens_per_expert", "router/entropy_hist",
    "router/entropy_quantiles", "router/entropy_quantile_counts",
}
# More scores than the extension selects quantiles of in one block.
LARGE = (1 << 17) + 1


def _routing_capture(directory: str, native: bool):
    """A sampler over a CUDA recorder of the router slots of three routed layers and two microbatches, with exact
    quantiles; its routing observers the extension's or, without `native`, the Python ones."""
    sampler = MetricSampler(directory, exact_quantiles=True)
    sampler.microbatches = 2
    requests = sampler._layout_requests(
        experts=EXPERTS, routed_layers=ROUTED, mtp_layers=[], streams=1, pre_reduce_keys=set(),
        activation_layers={point: [] for point in ACTIVATION_POINTS}, dumps_enabled=False,
    )
    layout = Layout([request for request in requests if request[0].name in ROUTING_METRICS])
    ring = ShmRing.create(f"ms_test_{uuid.uuid4().hex}", layout, slot_count=1)
    sampler.recorder = RankRecorder(layout, ring, torch.device("cuda"))
    sampler.quantile_stats = ExactQuantileStats(sampler.recorder)
    sampler.token_loss_stats = TokenLossStats(sampler.recorder, ROUTED, [], EXPERTS, TOP_K, 1, token_loss=False)
    native_routing = sampler._native_routing(EXPERTS) if native else None
    if native:
        assert native_routing is not None, "native routing was requested but the extension observer was unavailable"
    observers = [functools.partial(sampler._observe_routing, layer) if native_routing is None
                 else native_routing.router(layer) for layer in ROUTED]
    return sampler, observers, ring


def _scores(layer: int, tokens: int, generator: torch.Generator) -> torch.Tensor:
    if layer == 1:
        # A strided view: its quantiles go to the Python selection.
        return torch.rand(tokens, 2 * EXPERTS, device="cuda", generator=generator).softmax(-1)[:, ::2]
    return torch.rand(tokens, EXPERTS, device="cuda", generator=generator).softmax(-1)


def _stream_routing_capture(directory: str, native: bool):
    """A small three-stream capture with main and MTP routed layers."""
    # Exact capture makes observer failures stop training, which enables the native routing path. Quantile
    # recording itself is covered separately; this fixture installs only routing counts and entropy slots.
    sampler = MetricSampler(directory, exact_quantiles=True)
    sampler.microbatches = 2
    # Layers 2 and 3 are MTP heads; routing counts still belong to their own layer IDs.
    routed = {0: 0, 1: 0, 2: 1, 3: 2}
    mtp_layers = [2, 3]
    requests = sampler._layout_requests(
        experts=EXPERTS, routed_layers=routed, mtp_layers=mtp_layers, streams=STREAMS, pre_reduce_keys=set(),
        activation_layers={point: [] for point in ACTIVATION_POINTS}, dumps_enabled=False,
    )
    routing_slots = {"router/tokens_per_expert", "router/stream_tokens_per_expert", "router/entropy_hist"}
    layout = Layout([request for request in requests if request[0].name in routing_slots])
    ring = ShmRing.create(f"ms_test_{uuid.uuid4().hex}", layout, slot_count=1)
    sampler.recorder = RankRecorder(layout, ring, torch.device("cuda"))
    sampler.token_loss_stats = TokenLossStats(sampler.recorder, routed, mtp_layers, EXPERTS, TOP_K, STREAMS,
                                              token_loss=False)
    native_routing = sampler._native_routing(EXPERTS) if native else None
    if native:
        assert native_routing is not None, "native routing was requested but the extension observer was unavailable"
    observers = [functools.partial(sampler._observe_routing, layer) if native_routing is None
                 else native_routing.router(layer) for layer in routed]
    return sampler, observers, ring, tuple(routed)


@requires_extension
def test_native_stream_routing_matches_python_and_cpu_counts_across_steps_and_mtp_layers(tmp_path):
    """Every selected assignment is counted by stream, including MTP layers, on each microbatch and step."""
    runs = []
    routed_layers = (0, 1, 2, 3)
    for native in (True, False):
        sampler, observers, ring, routed_layers = _stream_routing_capture(str(tmp_path / f"streams-{native}"), native)
        captured = []
        try:
            for step in range(2):
                sampler.recorder.begin_step(False)
                sampler.token_loss_stats.begin_step(False)
                for microbatch in range(2):
                    # Interleaved sample streams, with two routed tokens per sample.
                    stream_ids = torch.tensor([2, 0, 1], dtype=torch.int32, device="cuda")
                    sampler.microbatch = microbatch
                    sampler.token_loss_stats.begin_microbatch(microbatch, stream_ids)
                    generator = torch.Generator(device="cuda").manual_seed(1000 * step + 100 * microbatch)
                    expected_by_layer = []
                    for layer, observer in zip(routed_layers, observers):
                        scores = _scores(layer % 2, 6, generator)
                        topk = scores.topk(TOP_K, dim=-1).indices
                        observer(topk, scores)
                        expected = torch.zeros(EXPERTS, STREAMS, dtype=torch.int64)
                        token_streams = stream_ids.cpu().repeat_interleave(2)
                        for token, stream in enumerate(token_streams.tolist()):
                            for expert in topk[token].cpu().tolist():
                                expected[expert, stream] += 1
                        expected_by_layer.append(expected.flatten())
                    torch.cuda.synchronize()
                    actual_by_layer = [sampler.recorder.view(sampler.recorder.resolve(
                        "router/stream_tokens_per_expert", layer, microbatch)).cpu().to(torch.int64).clone()
                        for layer in routed_layers]
                    old_counts_by_layer = [sampler.recorder.view(sampler.recorder.resolve(
                        "router/tokens_per_expert", layer, microbatch)).cpu().to(torch.int64).clone()
                        for layer in routed_layers]
                    flags = sampler.recorder.recorded_flags().cpu().clone()
                    expected_flags = torch.tensor([
                        slot.microbatch <= microbatch for slot in sampler.recorder._layout.slots
                    ])
                    captured.append((actual_by_layer, old_counts_by_layer, expected_by_layer, flags, expected_flags))
        finally:
            sampler.recorder.close()
            ring.unlink()
        runs.append(captured)

    assert len(runs[0]) == len(runs[1]) == 4
    for (native_actual, native_old, native_expected, native_flags, native_expected_flags), \
            (python_actual, python_old, python_expected, python_flags, python_expected_flags) in zip(*runs):
        assert torch.equal(native_flags, native_expected_flags)
        assert torch.equal(python_flags, python_expected_flags)
        assert torch.equal(native_flags, python_flags)
        for native_counts, native_old_counts, native_cpu, python_counts, python_old_counts, python_cpu in zip(
                native_actual, native_old, native_expected, python_actual, python_old, python_expected):
            assert torch.equal(native_counts, native_cpu)
            assert torch.equal(python_counts, python_cpu)
            assert torch.equal(native_cpu, python_cpu)
            assert torch.equal(native_counts.view(EXPERTS, STREAMS).sum(dim=1), native_old_counts)
            assert torch.equal(python_counts.view(EXPERTS, STREAMS).sum(dim=1), python_old_counts)


@requires_extension
def test_the_extensions_routing_observers_write_every_word_the_python_ones_write(tmp_path):
    runs = []
    for native in (True, False):
        sampler, observers, ring = _routing_capture(str(tmp_path / f"native-{native}"), native)
        steps = []
        try:
            for step in range(2):
                sampler.recorder.begin_step(False)
                sampler.token_loss_stats.begin_step(False)
                for microbatch in range(2):
                    sampler.microbatch = microbatch
                    sampler.token_loss_stats.begin_microbatch(microbatch, None)
                    generator = torch.Generator(device="cuda").manual_seed(10 * step + microbatch)
                    tokens = LARGE if (step, microbatch) == (1, 1) else 300
                    for layer, observer in enumerate(observers):
                        scores = _scores(layer, tokens, generator)
                        observer(scores.topk(TOP_K, dim=-1).indices, scores)
                    sampler.quantile_stats.flush()
                torch.cuda.synchronize()
                steps.append((sampler.recorder.buffer.cpu().clone(), sampler.recorder.recorded_flags().clone(),
                              sampler.token_loss_stats._topk_buffer.cpu().clone()))
            sampler.token_loss_stats.begin_microbatch(0, None)
            generator = torch.Generator(device="cuda").manual_seed(99)
            scores = _scores(0, 300, generator)
            observers[0](scores.topk(TOP_K, dim=-1).indices, scores)
            with pytest.raises(ValueError) as mismatch:
                observers[2](scores[:299].topk(TOP_K, dim=-1).indices, scores[:299])
            steps.append(str(mismatch.value))
        finally:
            sampler.recorder.close()
            ring.unlink()
        runs.append(steps)

    native_steps, python_steps = runs
    assert native_steps[-1] == python_steps[-1] == "observe_routing for layer 2 got 299 tokens, expected 300"
    for (native_words, native_flags, native_topk), (python_words, python_flags, python_topk) in zip(
            native_steps[:-1], python_steps[:-1]):
        assert torch.equal(native_words, python_words)
        assert torch.equal(native_flags, python_flags) and bool(native_flags.all())
        assert torch.equal(native_topk, python_topk)
