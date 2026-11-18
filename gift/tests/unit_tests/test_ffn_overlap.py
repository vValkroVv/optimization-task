"""Local compute and wrapping contracts for the two FFN overlap boundaries."""

import copy

import pytest
import torch
from torch._dynamo.backends.common import aot_autograd

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
)
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.compile import apply_compile, EagerWindow
from torchtitan.distributed.ffn_overlap import enable_ffn_overlap
from torchtitan.distributed.gfsdp._param_group import GFSDPBufferContext
from torchtitan.distributed.gfsdp._state import _alias_buffer_region
from torchtitan.protocols.module import Module, ModuleDict
from tests.unit_tests.test_mtp import _block_config, _module_config


@pytest.fixture(autouse=True)
def _isolate_compile_cache():
    # This matrix deliberately changes checkpoint policy and backend; those
    # variants must not exhaust the process-global cache of a previous case.
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


class _Model(Module):
    def __init__(self, mtp):
        super().__init__()
        if mtp:
            self.mtp = Module()
            self.mtp.layers = ModuleDict({"0": _module_config().build()})
        else:
            self.layers = ModuleDict({"0": _block_config().build()})

    @property
    def unit(self):
        return self.mtp.layers["0"] if hasattr(self, "mtp") else self.layers["0"]

    def forward(self, *inputs):
        return self.unit(*inputs, rope_cache=None)


def _model(mtp):
    torch.manual_seed(2048)
    model = _Model(mtp).double()
    with torch.no_grad():
        for param in model.parameters():
            if param.ndim == 1:
                param.fill_(1)
            else:
                param.normal_(std=0.05)
    return model


def _inputs(mtp):
    torch.manual_seed(87)
    return [
        torch.randn(1, 3, 64, dtype=torch.float64, requires_grad=True)
        for _ in range(2 if mtp else 1)
    ]


def _run(model, inputs):
    output = model(*inputs)
    gradient = torch.linspace(-0.5, 0.5, output.numel(), dtype=output.dtype)
    output.backward(gradient.reshape_as(output))
    return (
        output.detach(),
        [value.grad.clone() for value in inputs],
        {
            key.replace("_checkpoint_wrapped_module.", ""): value.grad.clone()
            for key, value in model.named_parameters()
        },
    )


@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("mode", ["none", "selective", "full", "save:mm"])
def test_compiled_checkpoint_regions_match_original(mtp, mode):
    control = _model(mtp)
    candidate = copy.deepcopy(control)
    original_parameters = dict(candidate.named_parameters())
    original_modules = dict(candidate.named_modules())
    enable_ffn_overlap(candidate.unit)
    ac = ActivationCheckpointConfig(mode=mode, selective_save=[], early_stop=False)
    if mode != "none":
        apply_ac(control, ac, model_compile_enabled=True)
        apply_ac(candidate, ac, model_compile_enabled=True)
    apply_compile(control, CompileConfig(enable=True, backend="aot_eager"))
    apply_compile(candidate, CompileConfig(enable=True, backend="aot_eager"))
    assert candidate.unit._compiled_call_impl is None
    assert all(
        region._compiled_call_impl for region in candidate.unit._ffn_overlap_regions
    )
    assert dict(candidate.named_parameters()) == original_parameters
    assert dict(candidate.named_modules()) == original_modules
    left = _run(control, _inputs(mtp))
    right = _run(candidate, _inputs(mtp))
    torch.testing.assert_close(right[0], left[0], rtol=1e-12, atol=1e-12)
    for got, expected in zip(right[1], left[1], strict=True):
        torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-12)
    assert right[2].keys() == left[2].keys()
    for key in left[2]:
        torch.testing.assert_close(right[2][key], left[2][key], rtol=1e-12, atol=1e-12)


def test_sac_policy_controls_actual_compiled_recomputation():
    backward_mm_counts = []
    for mode in ("none", "selective", "save:mm"):
        torch._dynamo.reset()
        graphs = []

        def compiler(graph, _example_inputs):
            return graph.forward

        def backward_compiler(graph, _example_inputs):
            graphs.append(graph)
            return graph.forward

        model = _model(False)
        enable_ffn_overlap(model.unit)
        if mode != "none":
            apply_ac(model, ActivationCheckpointConfig(mode=mode, selective_save=[]))
        backend = aot_autograd(fw_compiler=compiler, bw_compiler=backward_compiler)
        apply_compile(model, CompileConfig(enable=True, backend=backend))
        _run(model, _inputs(False))
        assert len(graphs) == 3
        backward_mm_counts.append(
            sum(
                str(node.target) == "aten.mm.default"
                for graph in graphs
                for node in graph.graph.nodes
            )
        )
    # Recomputing the compiled regions must add matrix multiplications; keeping
    # the outer eager checkpoint would save the opaque compiled calls instead.
    assert backward_mm_counts[1] > backward_mm_counts[0]
    assert backward_mm_counts[2] == backward_mm_counts[0]


@pytest.mark.parametrize("mtp", [False, True])
def test_inductor_checkpointed_region_gradients(mtp):
    control = _model(mtp)
    candidate = copy.deepcopy(control)
    enable_ffn_overlap(candidate.unit)
    ac = ActivationCheckpointConfig(mode="selective", selective_save=[])
    apply_ac(control, ac, model_compile_enabled=True)
    apply_ac(candidate, ac, model_compile_enabled=True)
    apply_compile(control, CompileConfig(enable=True))
    apply_compile(candidate, CompileConfig(enable=True))
    left = _run(control, _inputs(mtp))
    right = _run(candidate, _inputs(mtp))
    torch.testing.assert_close(right[0], left[0], rtol=1e-12, atol=1e-12)
    for got, expected in zip(right[1], left[1], strict=True):
        torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-12)
    for key in left[2]:
        torch.testing.assert_close(right[2][key], left[2][key], rtol=1e-12, atol=1e-12)


def test_eager_window_includes_unregistered_prefix_and_suffix():
    model = _model(False)
    enable_ffn_overlap(model.unit)
    apply_compile(model, CompileConfig(enable=True, backend="aot_eager"))
    regions = model.unit._ffn_overlap_regions
    slots = [region._compiled_call_impl for region in regions]
    window = EagerWindow(iter([model]), owner="test")
    assert len(window) == 3
    window.open()
    assert all(region._compiled_call_impl is None for region in regions)
    window.close()
    assert [region._compiled_call_impl for region in regions] == slots


def test_regions_alias_existing_storage_and_detect_all_overlaps():
    ctx = GFSDPBufferContext(GFSDPBufferContext.BufferType.ALL_GATHER)
    ctx.lazy_init({None: 128, torch.float8_e4m3fn: 64}, torch.device("cpu"))
    parent = ctx.region({None: 48}, {None: 0})
    ffn = ctx.region(
        {None: 64, torch.float8_e4m3fn: 64}, {None: 48, torch.float8_e4m3fn: 0}
    )
    assert not parent.overlaps(ffn)
    assert ctx.overlaps(parent) and ctx.overlaps(ffn)
    for region in (parent, ffn):
        for dtype, view in region.buffers.items():
            assert (
                view.untyped_storage().data_ptr()
                == ctx.buffers[dtype].untyped_storage().data_ptr()
            )
            assert view.untyped_storage().nbytes() == ctx.buffers[dtype].numel()
    owner = object()
    ffn.owner = owner
    assert ctx.conflicting_owner(object()) is owner
    assert parent.conflicting_owner(object()) is None
    assert ffn.conflicting_owner(owner) is None
    ffn.owner = None
    ctx.owner = owner
    assert parent.conflicting_owner(object()) is owner
    assert ffn.conflicting_owner(object()) is owner


def test_nested_transition_alias_uses_absolute_storage_ranges_and_borrow_registry():
    ctx = GFSDPBufferContext(GFSDPBufferContext.BufferType.ALL_GATHER)
    ctx.lazy_init({None: 128}, torch.device("cpu"))
    source = ctx.region({None: 96}, {None: 16})
    source.buffers[None].copy_(torch.arange(96, dtype=torch.uint8))

    target = _alias_buffer_region(source, {None: 64})

    assert target._storage_ctx is ctx
    assert target._ranges[None] == (16, 80)
    assert target.buffers[None].storage_offset() == 16
    torch.testing.assert_close(target.buffers[None], source.buffers[None][:64])
    assert source.overlaps(target)

    owner = object()
    target.owner = owner
    assert ctx.conflicting_owner(object()) is owner
    with pytest.raises(RuntimeError, match="live owner"):
        with ctx.borrow(None):
            pass


def test_flags_default_off_and_repeated_enable_rejected():
    config = ParallelismConfig()
    assert not config.gfsdp_split_mtp_ffn
    assert not config.gfsdp_split_layer0_ffn
    assert not config.gfsdp_mtp_transition_prefetch
    assert not config.gfsdp_row_owner_embedding_optimizations
    model = _model(False)
    enable_ffn_overlap(model.unit)
    with pytest.raises(ValueError, match="already configured"):
        enable_ffn_overlap(model.unit)


def test_ordinal_mm_half_policy_is_explicitly_rejected_for_split_regions():
    model = _model(False)
    enable_ffn_overlap(model.unit)
    with pytest.raises(ValueError, match="mm_half"):
        apply_ac(model, ActivationCheckpointConfig(mode="save:mm_half"))


@pytest.mark.parametrize(
    "split_layer0,split_mtp,mtp_transition",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (True, True, False),
        (False, True, True),
    ],
)
def test_nested_wrapping_and_prefetch_entry_order(
    monkeypatch, split_layer0, split_mtp, mtp_transition
):
    import torchtitan.models.gift_moe._base.parallelize as parallelize
    from tests.unit_tests.test_mtp import _make_tiny_decoder

    class RecordedGFSDP:
        def set_buffer_region_child(self, child):
            self.buffer_region_child = child

        def set_buffer_region_transition(self, **modules):
            self.buffer_region_transition = modules

        def set_own_grad_buffer(self):
            self.private_grad = True

        def set_gradient_divide_factor(self, value):
            self.divide_factor = value

        def set_shard_order(self, order):
            self.shard_order = order

        def set_modules_to_forward_prefetch(self, modules):
            self.forward_prefetch = modules

        def set_modules_to_backward_prefetch(self, modules):
            self.backward_prefetch = modules

    sharded = []
    managed_ids = set()
    ownership = {}

    def shard(module, **kwargs):
        assert kwargs["mp_policy"].param_dtype is torch.bfloat16
        assert kwargs["mp_policy"].reduce_dtype is torch.float32
        module.__class__ = type("RecordedUnit", (RecordedGFSDP, type(module)), {})
        ids = {id(param) for param in module.parameters()} - managed_ids
        ownership[module] = ids
        managed_ids.update(ids)
        sharded.append(module)
        return module

    monkeypatch.setattr(parallelize, "GFSDPModule", RecordedGFSDP)
    monkeypatch.setattr(parallelize, "g_fully_shard", shard)
    model = _make_tiny_decoder(1)
    if split_layer0:
        enable_ffn_overlap(model.layers["0"])
    if split_mtp:
        enable_ffn_overlap(model.mtp.layers["0"])
    # The second layer is dense in this fixture. ep_degree=2 selects the
    # production explicit prefetch wiring without executing any collectives.
    parallelize.apply_gfsdp(
        model,
        object(),
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        pp_enabled=False,
        ep_degree=2,
        edp_mesh=object(),
        mtp_transition_prefetch=mtp_transition,
    )
    units = [
        model.tok_embeddings,
        *model.layers.values(),
        model.norm,
        model.mtp.layers["0"],
        model.lm_head,
    ]

    def expand(unit):
        if getattr(unit, "_ffn_overlap_regions", None) is None:
            return [unit]
        block = getattr(unit, "block", unit)
        ffn = block.feed_forward
        assert sharded.index(ffn) < sharded.index(unit)
        assert unit.buffer_region_child is ffn
        assert not getattr(unit, "private_grad", False)
        assert ownership[unit].isdisjoint(ownership[ffn])
        assert ownership[ffn] == {id(param) for param in ffn.parameters()}
        return [unit, ffn]

    forward = [entered for unit in units for entered in expand(unit)]
    backward = [entered for unit in reversed(units) for entered in expand(unit)]
    assert model.shard_order == forward
    for source, target in zip(forward, forward[1:]):
        assert source.forward_prefetch == [target]
    for source, target in zip(backward, backward[1:]):
        if source is model.lm_head and mtp_transition:
            mtp_layer = model.mtp.layers["0"]
            assert source.backward_prefetch == [
                mtp_layer,
                mtp_layer.block.feed_forward,
            ]
        else:
            assert source.backward_prefetch == [target]
    assert all(unit.divide_factor == 1 for unit in sharded)
    if mtp_transition:
        mtp_layer = model.mtp.layers["0"]
        assert model.buffer_region_transition == {
            "source": mtp_layer.block.feed_forward,
            "data_target": model.layers[list(model.layers.keys())[-1]],
            "grad_target": model.norm,
        }
        assert mtp_layer.backward_prefetch == [mtp_layer.block.feed_forward]
