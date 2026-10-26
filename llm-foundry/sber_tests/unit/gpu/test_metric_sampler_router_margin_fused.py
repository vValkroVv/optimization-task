"""The extension's sigmoid router margin statistics, written into a buffer and replayed in the callback's graph, are
those of the graph of the torch composition, bit for bit: random, tied, signed-zero and non-finite logits, float32,
bfloat16 and row-strided, populations the selection takes in one block and in several, up to 1024 experts; beyond
it the extension declines and writes nothing."""

import functools

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu]

pytest.importorskip("composer", reason="Requires the training runtime")

import torch  # noqa: E402

from metric_sampler.capture.kernels import activation_ext  # noqa: E402
from metric_sampler.capture.kernels.graphed import GraphedComputation  # noqa: E402
from metric_sampler.capture.stats.router_margin import router_margin_graph_stats, router_margin_stats  # noqa: E402

requires_extension = pytest.mark.skipif(not torch.cuda.is_available() or activation_ext.prebuilt() is None,
                                        reason="Requires CUDA and the prebuilt extension")

_OFFSET = 3


def _logits(case: str, rows: int, experts: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(rows * experts)
    logits = torch.randn(rows, experts + 5, device="cuda", generator=generator) * 3
    if case == "strided":
        return logits[:, 2:experts + 2]
    logits = logits[:, :experts].contiguous()
    if case == "ties":
        logits = logits.round()
    elif case == "signed zeros":
        logits = logits.round() * 0.0
    elif case == "nan":
        logits[0, :3] = float("nan")
        logits[rows // 2, 1] = -float("nan")
    elif case == "inf":
        logits[0, :] = float("inf")
        logits[rows // 2, :2] = float("inf")
        logits[rows - 1, 1] = -float("inf")
    elif case == "bfloat16":
        logits = logits.bfloat16()
    return logits


def _graph_stats(compute, logits: torch.Tensor, top_k: int, near_m: int) -> torch.Tensor:
    return GraphedComputation(functools.partial(compute, "sigmoid", top_k, near_m), logits)(logits).clone()


@requires_extension
@pytest.mark.parametrize("case", ["random", "ties", "signed zeros", "nan", "inf", "bfloat16", "strided"])
@pytest.mark.parametrize(("rows", "experts", "top_k", "near_m"), [
    (64, 16, 2, 3), (64, 16, 4, 12), (1, 32, 4, 4), (4096, 64, 6, 8), (3000, 256, 8, 8), (16, 1024, 8, 8),
])
def test_the_extension_writes_the_graphs_statistics(case, rows, experts, top_k, near_m):
    logits = _logits(case, rows, experts)
    buf = torch.zeros(_OFFSET + 9, device="cuda")

    assert activation_ext.prebuilt().router_margin_into(logits, top_k, near_m, buf, _OFFSET)

    expected = _graph_stats(router_margin_stats, logits, top_k, near_m)
    assert torch.equal(buf[_OFFSET:_OFFSET + 8].view(torch.int32), expected.view(torch.int32))
    assert buf[:_OFFSET].eq(0).all() and buf[_OFFSET + 8] == 0
    graphed = _graph_stats(router_margin_graph_stats, logits, top_k, near_m)
    assert torch.equal(graphed.view(torch.int32), expected.view(torch.int32))


@requires_extension
def test_more_than_1024_experts_are_declined_without_a_write():
    buf = torch.zeros(8, device="cuda")

    assert not activation_ext.prebuilt().router_margin_into(_logits("random", 4, 1100), 2, 2, buf, 0)

    assert buf.eq(0).all()


def _margin_capture(directory: str, native: bool):
    """A sampler over a CUDA recorder of the router margin slots of three routed layers and two microbatches, the
    gates' forward hooks the extension's or, without `native`, the Python ones: a float32 sigmoid router of 16
    experts, a bfloat16 softmax router of 32 and a sigmoid router of 8."""
    import types
    import uuid

    from llmfoundry.callbacks.metric_sampler import MetricSampler
    from metric_sampler.capture.layout import Layout
    from metric_sampler.capture.recorder import RankRecorder
    from metric_sampler.capture.ring import ShmRing
    from metric_sampler.contract.catalog import metric_by_name

    routers = [("sigmoid", 16, 2, torch.float32), ("softmax", 32, 4, torch.bfloat16), ("sigmoid", 8, 2, torch.float32)]
    layout = Layout([(metric_by_name("monitor/router_margin_local"), layer, microbatch, 8)
                     for layer in range(len(routers)) for microbatch in (0, 1)])
    ring = ShmRing.create(f"ms_test_{uuid.uuid4().hex}", layout, slot_count=1)
    sampler = MetricSampler(directory)
    monitors = sampler.monitor_capture
    sampler.recorder = RankRecorder(layout, ring, torch.device("cuda"))
    native_margins = monitors._native_router_margins() if native else None
    gates = []
    for layer, (scoring_func, experts, top_k, dtype) in enumerate(routers):
        torch.manual_seed(layer)
        gate = types.SimpleNamespace(gate=torch.nn.Linear(64, experts, bias=False, device="cuda", dtype=dtype),
                                     scoring_func=scoring_func, n_routed_experts=experts, top_k=top_k)
        if native_margins is None:
            hook = functools.partial(monitors._observe_router_margin, layer, gate, 3)
        else:
            hook = native_margins.router(layer, scoring_func, top_k, 3, experts,
                                         functools.partial(monitors._record_router_margin, layer, gate, 3))
        gate.gate.register_forward_hook(hook)
        gates.append(gate)
    sampler._capture_forward_active = sampler._grad_forward = True
    return sampler, gates, ring


@requires_extension
def test_the_extensions_router_margin_hooks_write_every_word_the_python_ones_write(tmp_path):
    runs = []
    for native in (True, False):
        sampler, gates, ring = _margin_capture(str(tmp_path / f"native-{native}"), native)
        steps = []
        try:
            for step in range(2):
                sampler.recorder.begin_step(False)
                for microbatch in range(2):
                    sampler.microbatch = microbatch
                    generator = torch.Generator(device="cuda").manual_seed(10 * step + microbatch)
                    for layer, gate in enumerate(gates):
                        # The third router sees an empty microbatch once: its statistics are NaN.
                        rows = 0 if layer == 2 and microbatch == 1 else 96 + 32 * layer
                        x = torch.randn(rows, 64, device="cuda", generator=generator).to(gate.gate.weight.dtype)
                        gate.gate(x)
                    sampler.monitor_capture._flush_router_margins()
                torch.cuda.synchronize()
                steps.append((sampler.recorder.buffer.cpu().clone(), sampler.recorder.recorded_flags().clone()))

            sampler.microbatch = 0
            x = torch.randn(4, 64, device="cuda")
            gates[0].gate(x)
            with pytest.raises(ValueError) as twice:
                gates[0].gate(x)
            sampler.monitor_capture._flush_router_margins()
            before = sampler.recorder.buffer.clone()
            gates[0].gate.eval()
            gates[0].gate(torch.randn(4, 64, device="cuda"))
            gates[0].gate.train()
            sampler._capture_forward_active = False
            gates[1].gate(torch.randn(4, 64, device="cuda").bfloat16())
            torch.cuda.synchronize()
            steps.append((sampler.recorder.buffer.cpu().clone(), str(twice.value)))
            assert torch.equal(sampler.recorder.buffer, before) and not sampler.monitor_capture._router_margin_layers
        finally:
            sampler.recorder.close()
            ring.unlink()
        runs.append(steps)

    for (native_words, native_flags), (python_words, python_flags) in zip(*runs):
        assert torch.equal(native_words, python_words)
        if isinstance(native_flags, str):
            assert native_flags == python_flags == "router margin of layer 0 was observed twice in microbatch 0"
        else:
            assert torch.equal(native_flags, python_flags) and bool(native_flags.all())
