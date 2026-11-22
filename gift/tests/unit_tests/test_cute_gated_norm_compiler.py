import operator

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx import Graph

from torchtitan.models.common.norms.cute_gated_norm import compiler
from torchtitan.models.common.norms.cute_gated_norm.op import (
    _forward,
    _forward_residual,
    _forward_stats,
)
from torchtitan.models.common.norms.cute_gated_norm.pair import (
    PAIR_SHAPE,
    forward_pair,
)


def _make_norm_graph(*, keep_output, barrier=None, shape=PAIR_SHAPE):
    graph = Graph()
    mode = FakeTensorMode()
    device = torch.device("cuda", 0)
    hidden = shape[-1]

    def value(shape, dtype=torch.bfloat16):
        return FakeTensor(mode, torch.empty(shape, device="meta", dtype=dtype), device)

    def input_node(name, shape):
        node = graph.placeholder(name)
        node.meta["val"] = value(shape)
        return node

    def norm_outputs():
        return (
            value(shape),
            value(shape[:-1], torch.float32),
            value((*shape[:-1], 16), torch.float32),
            value((hidden,), torch.float32),
            value((hidden, 16)),
            value((16, hidden)),
            value((hidden, 16)),
        )

    def item(node, index):
        result = graph.call_function(operator.getitem, (node, index))
        result.meta["val"] = node.meta["val"][index]
        return result

    x = input_node("x", shape)
    residual = input_node("residual", shape)
    weight = input_node("weight", (hidden,))
    up = input_node("up", (16, hidden))
    down = input_node("down", (hidden, 16))
    next_weight = input_node("next_weight", (hidden,))
    next_up = input_node("next_up", (16, hidden))
    next_down = input_node("next_down", (hidden, 16))
    first = graph.call_function(
        _forward_residual._opoverload, (x, weight, up, down, 1e-6, 1.0, residual)
    )
    first.meta["val"] = norm_outputs()
    activation = item(first, 0)
    first_stats = item(first, 1)
    if barrier == "call":
        graph.call_function(torch.ops.aten.neg.default, (x,))
    if barrier == "earlier_result_use":
        graph.call_function(torch.ops.aten.alias.default, (activation,))
    second = graph.call_function(
        _forward._opoverload, (activation, next_weight, next_up, next_down, 1e-6, 1.0)
    )
    second.meta["val"] = norm_outputs()
    next_result = item(second, 0 if keep_output else 1)
    graph.output((activation, first_stats, next_result))
    return graph, first, second


@pytest.mark.parametrize("stats_only", (False, True))
def test_norm_pair_keeps_post_aot_output_indices(monkeypatch, stats_only):
    monkeypatch.setattr(compiler, "supports_pair", lambda device: True)
    graph, first, second = _make_norm_graph(keep_output=not stats_only)
    old_values = (first.meta["val"], second.meta["val"])
    compiler.GatedNormStatsOnlyPass()(graph)
    compiler.GatedNormPairPass()(graph)
    graph.lint()

    (pair,) = graph.find_nodes(op="call_function", target=forward_pair._opoverload)
    (output,) = graph.find_nodes(op="output")
    assert pair.args[-1] is stats_only
    assert not graph.find_nodes(op="call_function", target=_forward_residual._opoverload)
    assert not graph.find_nodes(op="call_function", target=_forward._opoverload)
    assert not graph.find_nodes(op="call_function", target=_forward_stats._opoverload)
    (results,) = output.args
    assert tuple(result.args for result in results) == ((pair, 0), (pair, 1), (pair, 7))
    assert results[-1].meta["val"] is old_values[1][1 if stats_only else 0]
    assert len(pair.meta["val"]) == (13 if stats_only else 14)


@pytest.mark.parametrize("barrier", ("call", "earlier_result_use", "shape", "device"))
def test_norm_pair_preserves_ineligible_graph(monkeypatch, barrier):
    monkeypatch.setattr(compiler, "supports_pair", lambda device: barrier != "device")
    shape = (2, 2048, 4096) if barrier == "shape" else PAIR_SHAPE
    graph, first, second = _make_norm_graph(keep_output=True, barrier=barrier, shape=shape)

    compiler.GatedNormPairPass()(graph)
    graph.lint()

    assert first in graph.nodes and second in graph.nodes
    assert not graph.find_nodes(op="call_function", target=forward_pair._opoverload)
