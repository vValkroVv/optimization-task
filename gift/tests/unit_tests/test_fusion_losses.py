"""Focused numerical and native-path checks for the independent loss patches."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from torchtitan.components.loss import MTPLoss, cross_entropy_loss as baseline_ce
from torchtitan.config import CompileConfig
from torchtitan.experiments.four_fusion.linear_ce import native_linear_ce
from torchtitan.experiments.four_fusion.stable_ce import cross_entropy_loss
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _compile(fn, compiled):
    return torch.compile(fn, fullgraph=True) if compiled else fn


@CUDA
@pytest.mark.parametrize("compiled", [False, True])
def test_ce_large_offset_and_ignored_rows(compiled):
    vocab = 128256
    x = torch.full(
        (3, vocab), 2**28, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    labels = torch.tensor([5, -100, vocab - 1], device="cuda")
    loss = _compile(cross_entropy_loss, compiled)(x, labels)
    (loss * (0.3 / 17)).backward()
    torch.testing.assert_close(
        loss, loss.new_tensor(2 * math.log(vocab)), rtol=2e-6, atol=2e-6
    )
    expected = torch.full_like(x, (0.3 / 17) / vocab)
    expected[1] = 0
    expected[0, 5] = ((1 / vocab) - 1) * (0.3 / 17)
    expected[2, -1] = expected[0, 5]
    torch.testing.assert_close(x.grad, expected, rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize("compiled", [False, True])
def test_unadapted_donor_fails_large_offset(compiled):
    donor_path = (
        Path(__file__).resolve().parents[3]
        / "references/kindulov/source/torchtitan/ops/_softcap_cross_entropy.py"
    )
    spec = importlib.util.spec_from_file_location("kindulov_row_donor", donor_path)
    donor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(donor)

    def unadapted(x, y):
        losses = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
        lse = torch.empty_like(losses)
        donor.forward_kernel[(x.shape[0],)](
            x,
            y,
            losses,
            lse,
            x.shape[1],
            0.0,
            4096,
            num_warps=4,
            enable_fp_fusion=False,
            debug=True,
        )
        grad = torch.empty_like(x)
        scale = torch.ones((), device=x.device)
        donor.backward_kernel[(x.shape[0],)](
            x,
            y,
            lse,
            scale,
            grad,
            x.shape[1],
            0.0,
            4096,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return losses.sum(), grad

    x = torch.full((3, 128256), 2**28, device="cuda", dtype=torch.bfloat16)
    y = torch.tensor([5, -100, 128255], device="cuda")
    loss, grad = _compile(unadapted, compiled)(x, y)
    assert loss.item() == 0.0
    assert grad[0, 0].item() == 1.0
    assert grad[0, 5].item() == 0.0
    assert torch.count_nonzero(grad[1]).item() == 0


@CUDA
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("all_ignored", [False, True])
def test_ce_partial_vocab_nonunit_upstream(compiled, all_ignored):
    torch.manual_seed(71)
    x = (torch.randn(5, 517, device="cuda") * 9).to(torch.bfloat16).requires_grad_()
    labels = torch.tensor([0, 516, -100, 29, 400], device="cuda")
    if all_ignored:
        labels.fill_(-100)
    reference = x.detach().float().requires_grad_()
    expected = F.cross_entropy(reference, labels, ignore_index=-100, reduction="sum")
    (expected * (-0.3 / 19)).backward()
    actual = _compile(cross_entropy_loss, compiled)(x, labels)
    (actual * (-0.3 / 19)).backward()
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(
        x.grad, reference.grad.to(x.dtype), rtol=0.008, atol=2e-6
    )


@CUDA
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("all_ignored", [False, True])
def test_linear_ce_known_head_scales_and_all_gradients(compiled, all_ignored):
    torch.manual_seed(17)
    hidden = torch.randn(
        32, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    weight = torch.randn(
        517, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    labels = torch.randint(517, (32,), device="cuda")
    labels[3] = -100
    if all_ignored:
        labels.fill_(-100)
    denominator = torch.tensor(53.0, device="cuda")
    h_ref = hidden.detach().clone().requires_grad_()
    w_ref = weight.detach().clone().requires_grad_()
    logits = F.linear(h_ref, w_ref)
    terms = torch.stack(
        [
            F.cross_entropy(x.float(), y, ignore_index=-100, reduction="sum")
            / denominator
            * scale
            for x, y, scale in zip(logits.chunk(2), labels.chunk(2), (1.0, 0.3))
        ]
    )
    terms.sum().backward()
    actual = _compile(native_linear_ce, compiled)(
        hidden, weight, labels, denominator, [1.0, 0.3]
    )
    actual.sum().backward()
    torch.testing.assert_close(actual, terms, rtol=2e-6, atol=2e-6)
    # Tile GEMMs keep FP32 dW accumulation before their one final BF16 cast.
    torch.testing.assert_close(hidden.grad, h_ref.grad, rtol=0.016, atol=3e-5)
    torch.testing.assert_close(weight.grad, w_ref.grad, rtol=0.02, atol=6e-5)


@CUDA
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    ("invalid_metadata", "message"),
    [
        ("dtype", "dtype int64"),
        ("shape", r"input\.shape\[:-1\]"),
        ("labels_device", "share a device"),
        ("weight_device", "share a device"),
        ("denominator_device", "share a device"),
    ],
)
def test_native_linear_ce_rejects_invalid_metadata(compiled, invalid_metadata, message):
    device = torch.device("cuda")
    hidden = torch.randn(2, 2, 4, device=device, dtype=torch.bfloat16)
    weight = torch.randn(17, 4, device=device, dtype=torch.bfloat16)
    labels = torch.zeros(2, 2, device=device, dtype=torch.int64)
    denominator = torch.tensor(4.0, device=device)

    if invalid_metadata == "dtype":
        labels = labels.to(torch.int32)
    elif invalid_metadata == "shape":
        labels = torch.zeros(2, 1, device=device, dtype=torch.int64)
    elif invalid_metadata == "labels_device":
        labels = torch.zeros(2, 2, device="cpu", dtype=torch.int64)
    elif invalid_metadata == "weight_device":
        weight = torch.randn(17, 4, device="cpu", dtype=torch.bfloat16)
    elif invalid_metadata == "denominator_device":
        denominator = torch.tensor(4.0, device="cpu")

    # Fullgraph compilation reports Python validation errors as Unsupported
    # while tracing; the message still identifies the rejected metadata.
    expected_error = (
        (ValueError, torch._dynamo.exc.Unsupported) if compiled else ValueError
    )
    with pytest.raises(expected_error, match=message):
        _compile(native_linear_ce, compiled)(hidden, weight, labels, denominator, [1.0])


@CUDA
@pytest.mark.parametrize("compiled", [False, True])
def test_linear_ce_near_certain_target_subtracts_before_cast(compiled):
    hidden = torch.ones(2, 1, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.zeros(
        517, 1, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    with torch.no_grad():
        weight[0] = 18
    labels = torch.zeros(2, dtype=torch.int64, device="cuda")
    denominator = torch.tensor(31.0, device="cuda")
    actual = _compile(native_linear_ce, compiled)(
        hidden, weight, labels, denominator, [1.0, 0.3]
    )
    actual.sum().backward()
    assert (
        weight.grad[0, 0].item() < 0
    )  # Rounded BF16 p_target then subtraction gives zero.
    x = F.linear(hidden.detach(), weight.detach()).float().requires_grad_()
    expected = sum(
        F.cross_entropy(t, y, reduction="sum") / denominator * s
        for t, y, s in zip(x.chunk(2), labels.chunk(2), (1.0, 0.3))
    )
    expected.backward()
    expected_dw = x.grad.t() @ hidden.detach().float()
    torch.testing.assert_close(
        weight.grad, expected_dw.to(weight.dtype), rtol=0.02, atol=1e-8
    )


@CUDA
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    "denominator_kind", ["tensor", "native_float", "fractional_float"]
)
def test_native_mtp_two_chunks_loss_hidden_weight_and_identity(
    compiled, denominator_kind
):
    torch.manual_seed(9)
    from torchtitan.models.common.embeddings.token_debeddings import Unembedding

    head = Unembedding(
        Unembedding.Config(in_features=32, out_features=517, bias=False)
    ).to(device="cuda", dtype=torch.bfloat16)
    original = {k: v.detach().clone() for k, v in head.state_dict().items()}
    parameter = head.weight
    loss = MTPLoss(
        MTPLoss.Config(num_chunks=2, loss_weights=[0.3]),
        compile_config=CompileConfig(enable=compiled, components=["loss"]),
    )
    original_fn = getattr(loss.fn, "_torchdynamo_orig_callable", loss.fn)
    if not hasattr(loss, "_linear_ce") and original_fn is not cross_entropy_loss:
        pytest.skip("Apply one loss patch for native integration checks")
    loss.set_lm_head(head)
    assert head.weight is parameter
    assert head.state_dict().keys() == original.keys()
    for key, value in head.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)
    torch.manual_seed(9)
    h = torch.randn(
        2, 2, 8, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    labels = torch.randint(517, (2, 8), device="cuda")
    labels[0, 3] = -100
    positions = torch.tensor(
        [[0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7]], device="cuda"
    )
    targets = loss.pack_targets(labels, {"positions": positions})
    if denominator_kind == "native_float":
        from torchtitan.distributed.utils import dist_sum

        # This is the native DP producer's return type/value; no synthetic Tensor adapter.
        denom = dist_sum(torch.tensor(2096640, dtype=torch.int64, device="cuda"))
        assert type(denom) is float and denom == 2096640.0
    elif denominator_kind == "fractional_float":
        denom = 29.5
    else:
        denom = torch.tensor(29.0, device="cuda")
    h_ref = h.detach().clone().requires_grad_()
    ref_head = torch.nn.Linear(32, 517, bias=False, device="cuda", dtype=torch.bfloat16)
    ref_head.load_state_dict(original)
    ref = MTPLoss(MTPLoss.Config(num_chunks=2, loss_weights=[0.3]))
    ref.set_lm_head(ref_head)
    # Keep native labels/metrics/chunking, with the original logits path as control.
    ref.fn = _compile(baseline_ce, compiled)
    if hasattr(ref, "_linear_ce"):
        ref._linear_ce.chunk_loss = (
            lambda owner, head, hidden, labels, denominator, **kw: owner._chunk_loss(
                head(hidden), labels, denominator, num_heads=kw["num_heads"]
            )
        )
    expected = ref(h_ref, targets, denom)
    expected.backward()
    selected = []
    if hasattr(loss, "_linear_ce"):
        original_fused = loss._linear_ce.fn

        def observed(*args, **kwargs):
            selected.append(args[3])
            return original_fused(*args, **kwargs)

        loss._linear_ce.fn = observed
    actual = loss(h, targets, denom)
    actual.backward()
    if hasattr(loss, "_linear_ce"):
        assert len(selected) == (0 if denominator_kind == "fractional_float" else 2)
        if denominator_kind == "native_float":
            assert all(type(value) is int and value == denom for value in selected)
        else:
            assert all(value is denom for value in selected)
    magnitude = 29.0 / denom if isinstance(denom, float) else 1.0
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-6 * magnitude)
    torch.testing.assert_close(h.grad, h_ref.grad, rtol=0.016, atol=3e-5 * magnitude)
    torch.testing.assert_close(
        head.weight.grad, ref_head.weight.grad, rtol=0.02, atol=6e-5 * magnitude
    )
    assert head.weight is parameter
    assert loss.components().keys() == ref.components().keys()
    for key in loss.components():
        torch.testing.assert_close(
            loss.components()[key],
            ref.components()[key],
            rtol=3e-6,
            atol=3e-6 * magnitude,
        )
    torch.testing.assert_close(
        head(h.detach()[0]), F.linear(h.detach()[0], head.weight), rtol=0, atol=0
    )


@CUDA
@pytest.mark.parametrize("denominator_kind", ["tensor", "native_float"])
def test_native_hooks_eval_and_probe_fall_back(denominator_kind):
    from torchtitan.models.common.embeddings.token_debeddings import Unembedding

    head = Unembedding(
        Unembedding.Config(in_features=4, out_features=17, bias=False)
    ).to(device="cuda", dtype=torch.bfloat16)
    loss = MTPLoss(MTPLoss.Config(num_chunks=2, loss_weights=[0.3]))
    original_fn = getattr(loss.fn, "_torchdynamo_orig_callable", loss.fn)
    if not hasattr(loss, "_linear_ce") and original_fn is not cross_entropy_loss:
        pytest.skip("Apply one loss patch for native integration checks")
    loss.set_lm_head(head)
    h = torch.randn(2, 8, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = torch.randint(17, (2, 8), device="cuda")
    shapes = []
    hook = head.register_forward_hook(
        lambda module, args, out: shapes.append(out.shape)
    )
    denominator = (
        2096640.0
        if denominator_kind == "native_float"
        else torch.tensor(16.0, device="cuda")
    )
    loss(h, y, denominator).backward()
    assert shapes == [torch.Size([2, 4, 17])] * 2
    hook.remove()
    head.weight.grad = None
    result = loss(h.detach(), y, denominator)
    assert not result.requires_grad
    assert head.weight.grad is None
    total, grad = loss._chunked_loss_and_grad(h, y, denominator, shape_probe=True)
    assert grad is not None and torch.isfinite(total)
    assert head.weight.grad is None


@CUDA
def test_compiled_native_mtp_accepts_changing_host_denominators_without_recompile():
    from torch._dynamo.backends.registry import lookup_backend
    from torchtitan.distributed.utils import dist_sum
    from torchtitan.models.common.embeddings.token_debeddings import Unembedding

    torch.manual_seed(81)
    head = Unembedding(
        Unembedding.Config(in_features=8, out_features=37, bias=False)
    ).to(device="cuda", dtype=torch.bfloat16)
    loss = MTPLoss(
        MTPLoss.Config(num_chunks=2, loss_weights=[0.3]),
        compile_config=CompileConfig(enable=True, components=["loss"]),
    )
    if not hasattr(loss, "_linear_ce"):
        pytest.skip("Apply linear-ce.patch for changing native denominator checks")
    loss.set_lm_head(head)
    compilations = []
    inductor = lookup_backend("inductor")

    def backend(graph, inputs):
        compilations.append(graph)
        return inductor(graph, inputs)

    loss._linear_ce.fn = torch.compile(
        native_linear_ce, backend=backend, fullgraph=True, dynamic=True
    )
    hidden = torch.randn(
        2, 2, 4, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    labels = torch.randint(37, (2, 4), device="cuda")
    losses = []
    compilation_counts = []
    for count in (2096640, 2096632, 2096624):
        denominator = dist_sum(torch.tensor(count, device="cuda", dtype=torch.int64))
        head.weight.grad = None
        hidden.grad = None
        result = loss(hidden, labels, denominator)
        result.backward()
        assert (
            torch.isfinite(hidden.grad).all() and torch.isfinite(head.weight.grad).all()
        )
        losses.append(result.detach() * denominator)
        compilation_counts.append(len(compilations))
    for value in losses[1:]:
        torch.testing.assert_close(value, losses[0], rtol=3e-6, atol=3e-6)
    assert compilation_counts[0] > 0
    assert compilation_counts == [compilation_counts[0]] * 3
    print("native denominator compile counts:", compilation_counts)


def test_native_host_count_requires_exact_int64_round_trip():
    from torchtitan.experiments.four_fusion.linear_ce import LinearCE

    assert LinearCE._host_count(2096640.0) == 2096640
    assert LinearCE._host_count(float(-(1 << 63))) == -(1 << 63)
    for value in (29.5, float(1 << 63), float("inf"), float("nan")):
        assert LinearCE._host_count(value) is None


class _NoOpParallelDims:
    """Resolve configured dense placements without constructing a device mesh."""

    def __init__(self, *, full_dtensor=False):
        self.full_dtensor = full_dtensor
        self.active_axes = set()

    def resolve_mesh(self, axes):
        names = {getattr(axis, "value", axis) for axis in axes}
        return object() if names & self.active_axes else None

    def resolve_shared_mesh(self, placements):
        present = [placement for placement in placements if placement is not None]
        if not present:
            return None
        axes = present[0].keys()
        if any(placement.keys() != axes for placement in present[1:]):
            raise AssertionError("input placements name different mesh axes")
        return self.resolve_mesh(axes)


def _parallelized_linear_config(*, boundary="all", local_map=None):
    from torchtitan.config.schema.models.common.nn_modules import LinearConfig
    from torchtitan.config.schema.protocols.sharding import (
        LocalMapConfig,
        Replicate,
        Shard,
        ShardingConfig,
    )
    from torchtitan.models.common.decoder_blocks.sharding import (
        dense_activation_placement,
        dense_param_placement,
    )

    state = dense_param_placement(tp=Shard(0))
    activation = dense_activation_placement(tp=Replicate())
    in_src = {"input": activation} if boundary in ("all", "input") else None
    in_dst = (
        {"input": dense_activation_placement(tp=Replicate())}
        if boundary in ("all", "input", "local_map")
        else None
    )
    out_dst = (
        dense_activation_placement(tp=Replicate())
        if boundary in ("all", "output")
        else None
    )
    out_src = activation if local_map is not None else None
    if boundary == "local_map" and local_map is None:
        local_map = LocalMapConfig(in_grad_placements=(activation,))
        out_src = activation
    return LinearConfig(
        in_features=32,
        out_features=517,
        bias=False,
        sharding_config=ShardingConfig(
            state_shardings={"weight": state if boundary in ("all", "state") else {}},
            in_src_shardings=in_src,
            in_dst_shardings=in_dst,
            out_src_shardings=out_src,
            out_dst_shardings=out_dst,
            local_map=local_map,
        ),
    )


def test_flce_recognizes_canonical_noop_parallel_forward():
    from torchtitan.experiments.four_fusion.linear_ce import LinearCE

    dims = _NoOpParallelDims()
    head = _parallelized_linear_config().build()
    parameter = head.weight
    state_keys = tuple(head.state_dict())
    inputs = torch.randn(2, 32)
    expected = F.linear(inputs, parameter)

    head.parallelize(dims)
    wrapper = head.forward
    metadata = head._parallelized_forward_contract
    assert isinstance(metadata, tuple) and len(metadata) == 3
    assert metadata[0] is wrapper and metadata[2] is dims
    assert metadata[1].__self__ is head
    assert metadata[1].__func__ is torch.nn.Linear.forward
    torch.testing.assert_close(wrapper(inputs), expected, rtol=0, atol=0)

    adapter = LinearCE()
    adapter.install(head)

    assert adapter.supported_head
    assert adapter.original_forward is wrapper
    assert head.weight is parameter
    assert tuple(head.state_dict()) == state_keys
    torch.testing.assert_close(head(input=inputs), expected, rtol=0, atol=0)


class _StubTPMesh:
    mesh_dim_names = ("tp",)

    def size(self):
        return 2


def _parallel_dims_with_active_tp(*, full_dtensor=False):
    from torchtitan.distributed import ParallelDims

    parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=1,
        cp=1,
        tp=2,
        pp=1,
        ep=1,
        world_size=2,
        full_dtensor=full_dtensor,
    )
    parallel_dims._single_axis_meshes = {"tp": _StubTPMesh()}
    return parallel_dims


@pytest.mark.parametrize("boundary", ["state", "input", "output"])
def test_flce_rejects_parallel_forward_with_active_dense_mesh(boundary):
    from torchtitan.experiments.four_fusion.linear_ce import LinearCE

    dims = _parallel_dims_with_active_tp()
    head = _parallelized_linear_config(boundary=boundary).build()
    # No tensor operation is needed: the real resolver sees a size-two TP
    # mesh stub, and the test only checks the adapter's conservative guard.
    head.register_parameter("weight", None)
    if boundary != "state":
        head._sharding_config.state_shardings = {}
    head.parallelize(dims)
    wrapper = head.forward

    adapter = LinearCE()
    adapter.install(head)

    assert not adapter.supported_head
    assert adapter.original_forward is wrapper


def test_flce_rejects_full_dtensor_and_local_map_contracts():
    from torchtitan.experiments.four_fusion.linear_ce import LinearCE

    head = _parallelized_linear_config().build()
    head.register_parameter("weight", None)
    head.parallelize(_parallel_dims_with_active_tp(full_dtensor=True))
    adapter = LinearCE()
    adapter.install(head)
    assert not adapter.supported_head

    local_map_head = _parallelized_linear_config(boundary="local_map").build()
    local_map_wrapper_dims = _NoOpParallelDims()
    local_map_head.parallelize(local_map_wrapper_dims)
    local_map_adapter = LinearCE()
    local_map_adapter.install(local_map_head)
    assert not local_map_adapter.supported_head


def test_flce_rejects_user_replaced_parallel_wrapper():
    from torchtitan.experiments.four_fusion.linear_ce import LinearCE

    head = _parallelized_linear_config().build()
    head.parallelize(_NoOpParallelDims())
    original_wrapper = head.forward

    def user_forward(input):
        return original_wrapper(input)

    head.forward = user_forward
    adapter = LinearCE()
    adapter.install(head)

    assert not adapter.supported_head
    assert adapter.original_forward is user_forward


def test_flce_rejects_copied_contract_from_another_module():
    from torchtitan.experiments.four_fusion.linear_ce import LinearCE

    dims = _NoOpParallelDims()
    first = _parallelized_linear_config().build()
    second = _parallelized_linear_config().build()
    first.parallelize(dims)
    second.parallelize(dims)
    second.forward = first.forward
    second._parallelized_forward_contract = first._parallelized_forward_contract

    adapter = LinearCE()
    adapter.install(second)

    assert not adapter.supported_head
    assert adapter.original_forward is first._parallelized_forward_contract[0]


def test_flce_rejects_primitive_subclass_overriding_forward():
    from torchtitan.experiments.four_fusion.linear_ce import LinearCE
    from torchtitan.models.common.nn_modules import Linear

    class CustomLinear(Linear):
        def forward(self, input):
            return super().forward(input)

    config = _parallelized_linear_config()
    head = CustomLinear(config)
    head._sharding_config = config.sharding_config
    head.parallelize(_NoOpParallelDims())

    adapter = LinearCE()
    adapter.install(head)

    assert not adapter.supported_head


class TestLinearCEParallelizedGFSDP(DTensorTestBase):
    @property
    def world_size(self):
        return 1

    def _run_parallelized_gfsdp_head(self, *, compiled):
        if self.device_type != "cuda":
            self.skipTest("The native linear CE integration requires CUDA")
        from torchtitan.distributed import ParallelDims
        from torchtitan.distributed.gfsdp import (
            MixedPrecisionPolicy,
            fully_shard as g_fully_shard,
        )

        torch.manual_seed(9)
        device = torch.device(self.device_type)
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=self.world_size,
        )
        parallel_dims.build_mesh()

        head = (
            _parallelized_linear_config()
            .build()
            .to(device=device, dtype=torch.bfloat16)
        )
        initial_weight = head.weight.detach().clone()
        head.parallelize(parallel_dims)
        head = g_fully_shard(
            head,
            mesh=parallel_dims.get_mesh("fsdp"),
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                cast_forward_inputs=False,
            ),
        )
        head.set_own_grad_buffer()
        head.set_gradient_divide_factor(1.0)
        head.set_requires_gradient_sync(False, recurse=False)
        head.set_is_last_backward(False)
        parameter = head.weight
        state_keys = tuple(head.state_dict())

        compile_config = CompileConfig(enable=compiled, components=["loss"])
        loss = MTPLoss(
            MTPLoss.Config(num_chunks=2, loss_weights=[0.3]),
            compile_config=compile_config,
        )
        if not hasattr(loss, "_linear_ce"):
            self.skipTest("Apply runs/four-fusion/linear-ce.patch for this check")
        loss.set_lm_head(head)
        assert loss._linear_ce.supported_head
        assert (
            loss._linear_ce.original_forward is head._parallelized_forward_contract[0]
        )
        assert not loss._linear_ce._has_logits_hooks(head)

        ref_head = torch.nn.Linear(
            32, 517, bias=False, device=device, dtype=torch.bfloat16
        )
        ref_head.load_state_dict({"weight": initial_weight})
        ref_loss = MTPLoss(MTPLoss.Config(num_chunks=2, loss_weights=[0.3]))
        ref_loss.set_lm_head(ref_head)
        ref_loss.fn = _compile(baseline_ce, compiled)
        if hasattr(ref_loss, "_linear_ce"):

            def baseline_chunk_loss(
                owner, module, hidden, labels, denominator, *, num_heads, **kwargs
            ):
                return owner._chunk_loss(
                    module(hidden), labels, denominator, num_heads=num_heads
                )

            ref_loss._linear_ce.chunk_loss = baseline_chunk_loss

        positions = torch.tensor(
            [[0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7]],
            device=device,
        )
        denominator = 29.0
        selected = []
        native_fn = loss._linear_ce.fn

        def observed(*args, **kwargs):
            selected.append(args[3])
            return native_fn(*args, **kwargs)

        loss._linear_ce.fn = observed
        actual_losses = []
        expected_losses = []
        hidden_inputs = []
        reference_hiddens = []
        fsdp_state = head._get_fsdp_state()
        for microbatch in range(2):
            torch.manual_seed(90 + microbatch)
            hidden = torch.randn(
                2, 2, 8, 32, device=device, dtype=torch.bfloat16, requires_grad=True
            )
            labels = torch.randint(517, (2, 8), device=device)
            labels[0, 3] = -100
            targets = loss.pack_targets(labels, {"positions": positions})
            reference_hidden = hidden.detach().clone().requires_grad_()

            expected = ref_loss(reference_hidden, targets, denominator)
            expected.backward()
            expected_losses.append(expected.detach())
            reference_hiddens.append(reference_hidden)

            if microbatch == 1:
                head.set_requires_gradient_sync(True, recurse=False)
                head.set_is_last_backward(True)
            actual = loss(hidden, targets, denominator)
            actual.backward()
            actual_losses.append(actual.detach())
            hidden_inputs.append(hidden)

            if microbatch == 0:
                assert not fsdp_state._state_ctx.is_last_backward
                assert all(
                    not group.reduce_grads for group in fsdp_state._fsdp_param_groups
                )
                assert all(
                    fsdp_param.unsharded_accumulated_grad is not None
                    for group in fsdp_state._fsdp_param_groups
                    for fsdp_param in group.fsdp_params
                )

        assert selected == [29, 29, 29, 29]
        for actual, expected in zip(actual_losses, expected_losses, strict=True):
            torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-6)
        for hidden, reference_hidden in zip(
            hidden_inputs, reference_hiddens, strict=True
        ):
            torch.testing.assert_close(
                hidden.grad, reference_hidden.grad, rtol=0.016, atol=3e-5
            )
        actual_weight_grad = head.weight.grad
        assert actual_weight_grad is not None
        if hasattr(actual_weight_grad, "to_local"):
            with torch.no_grad():
                actual_weight_grad = actual_weight_grad.to_local()
        torch.testing.assert_close(
            actual_weight_grad.reshape_as(ref_head.weight.grad),
            ref_head.weight.grad,
            rtol=0.02,
            atol=6e-5,
        )
        assert loss.components().keys() == ref_loss.components().keys()
        for key in loss.components():
            torch.testing.assert_close(
                loss.components()[key],
                ref_loss.components()[key],
                rtol=3e-6,
                atol=3e-6,
            )
        assert head.weight is parameter
        assert tuple(head.state_dict()) == state_keys
        saved_weight = head.state_dict()["weight"]
        if hasattr(saved_weight, "to_local"):
            with torch.no_grad():
                saved_weight = saved_weight.to_local()
        torch.testing.assert_close(
            saved_weight.reshape_as(initial_weight), initial_weight, rtol=0, atol=0
        )
        fsdp_state = head._get_fsdp_state()
        assert fsdp_state._own_grad_buffer
        assert fsdp_state._state_ctx.is_last_backward
        assert all(group.reduce_grads for group in fsdp_state._fsdp_param_groups)
        assert all(
            fsdp_param.unsharded_accumulated_grad is None
            for group in fsdp_state._fsdp_param_groups
            for fsdp_param in group.fsdp_params
        )
        assert all(
            group.gradient_divide_factor == 1.0
            for group in fsdp_state._fsdp_param_groups
        )
        assert all(
            group.reshard_after_backward for group in fsdp_state._fsdp_param_groups
        )

    @with_comms
    def test_dtensor_input_and_labels_keep_logits_fallback(self):
        if self.device_type != "cuda":
            self.skipTest("The native linear CE integration requires CUDA")
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import Replicate
        from torchtitan.config.schema.models.common.nn_modules import LinearConfig
        from torchtitan.experiments.four_fusion.linear_ce import LinearCE

        device = torch.device(self.device_type)
        mesh = init_device_mesh(self.device_type, (1,), mesh_dim_names=("tp",))

        def as_dtensor(value):
            from torch.distributed.tensor import DTensor

            return DTensor.from_local(value, mesh, (Replicate(),), run_check=False)

        class LossRecorder:
            def __init__(self):
                self.observed = []

            def _chunk_loss(self, logits, labels, denominator, *, num_heads):
                self.observed.append(
                    (hasattr(logits, "to_local"), hasattr(labels, "to_local"))
                )
                local_logits = (
                    logits.to_local() if hasattr(logits, "to_local") else logits
                )
                return local_logits.float().sum()

        for dtensor_input in (True, False):
            head = (
                LinearConfig(in_features=4, out_features=17, bias=False)
                .build()
                .to(device=device, dtype=torch.bfloat16)
            )
            if dtensor_input:
                head.weight = torch.nn.Parameter(as_dtensor(head.weight.detach()))
            adapter = LinearCE()
            adapter.install(head)
            adapter.fn = lambda *args, **kwargs: pytest.fail(
                "DTensor inputs and labels must retain the logits fallback"
            )
            hidden = torch.randn(
                2, 4, device=device, dtype=torch.bfloat16, requires_grad=True
            )
            labels = torch.randint(17, (2,), device=device)
            if dtensor_input:
                hidden = as_dtensor(hidden)
            else:
                labels = as_dtensor(labels)
            recorder = LossRecorder()
            adapter.chunk_loss(
                recorder,
                head,
                hidden,
                labels,
                None,
                num_heads=1,
                argmax_sink=None,
                shape_probe=False,
            )
            assert recorder.observed == [(dtensor_input, not dtensor_input)]

    @with_comms
    def test_parallelized_gfsdp_native_eager(self):
        self._run_parallelized_gfsdp_head(compiled=False)

    @with_comms
    def test_parallelized_gfsdp_native_compiled(self):
        self._run_parallelized_gfsdp_head(compiled=True)
