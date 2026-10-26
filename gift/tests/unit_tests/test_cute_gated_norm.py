"""Numerics, registration and model compilation on Hopper; CPU coverage is in test_gated_norm."""

import copy
import math

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from torchtitan.models.common.norms import GatedRMSNorm
from torchtitan.models.common.norms.cute_gated_norm.op import _forward
from torchtitan.models.common.norms.gated_norm import _gated_rms_norm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="CuTe gated norm requires Hopper (SM90)",
)


@pytest.fixture(autouse=True)
def _deterministic():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    yield
    torch.use_deterministic_algorithms(previous)


def _norm(hidden, zero_centered, dtype):
    norm = (
        GatedRMSNorm.Config(
            normalized_shape=hidden,
            norm_backend="cute",
            zero_centered=zero_centered,
            eps=1e-6,
            gating_weight=1.7,
        )
        .build()
        .to(device="cuda", dtype=dtype)
    )
    with torch.no_grad():
        norm.weight.normal_(0.0 if zero_centered else 1.0, 0.1)
        norm.gate_up_weight.normal_(std=1 / math.sqrt(hidden))
        norm.gate_down_weight.normal_(std=0.25)
    return norm


def _compare(result, reference):
    for index, (actual, expected) in enumerate(zip(result, reference)):
        actual, expected = actual.float(), expected.float()
        assert torch.isfinite(actual).all()
        relative_l2 = (actual - expected).norm() / expected.norm().clamp_min(1e-30)
        assert relative_l2 < (0.01 if index < 2 else 0.02)
        if index < 2:
            atol = 0.02 * expected.square().mean().sqrt().item()
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=atol)


@pytest.mark.parametrize(
    "shape", [(1, 128), (63, 512), (64, 512), (65, 1536), (129, 4096), (2, 3, 5, 128)]
)
@pytest.mark.parametrize("zero_centered", [False, True])
@pytest.mark.parametrize("param_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_numerics(shape, zero_centered, param_dtype):
    norm = _norm(shape[-1], zero_centered, param_dtype)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn_like(x)
    y = norm(x)
    grads = torch.autograd.grad(y, (x, *norm.parameters()), dy)
    # FP32 reference, with exactly the same BF16-valued inputs/parameter values.
    reference_x = x.detach().float().requires_grad_()
    params = [p.detach().float().requires_grad_() for p in norm.parameters()]
    reference_y = _gated_rms_norm(
        reference_x,
        *params,
        eps=norm.eps,
        gating_weight=norm.gating_weight,
        zero_centered=zero_centered,
    )
    reference_grads = torch.autograd.grad(
        reference_y, (reference_x, *params), dy.float()
    )
    _compare((y, *grads), (reference_y, *reference_grads))
    assert all(g.dtype == param_dtype for g in grads[1:])
    # Fixed inputs reproduce every bit, including the parameter reductions.
    repeated = norm(x)
    repeated_grads = torch.autograd.grad(repeated, (x, *norm.parameters()), dy)
    for first, second in zip((y, *grads), (repeated, *repeated_grads)):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_noncontiguous_and_unaligned_inputs():
    norm = _norm(128, True, torch.float32)
    for x in (
        torch.randn(3, 2, 128, device="cuda", dtype=torch.bfloat16).transpose(0, 1),
        torch.randn(6 * 128 + 1, device="cuda", dtype=torch.bfloat16)[1:].view(
            2, 3, 128
        ),
    ):
        x = x.requires_grad_()
        dy = torch.randn_like(x)
        actual = norm(x)
        grads = torch.autograd.grad(actual, (x, *norm.parameters()), dy)
        reference = copy.deepcopy(norm)
        reference.norm_backend = "compile"
        expected = reference(x)
        expected_grads = torch.autograd.grad(expected, (x, *reference.parameters()), dy)
        _compare((actual, *grads), (expected, *expected_grads))


def test_empty_and_zero_weight():
    norm = _norm(128, False, torch.float32)
    with torch.no_grad():
        norm.weight.zero_()
    for rows in (0, 65):
        x = torch.randn(
            rows, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        y = norm(x)
        grads = torch.autograd.grad(y.sum(), (x, *norm.parameters()))
        assert torch.count_nonzero(y) == 0
        assert all(torch.isfinite(g).all() for g in grads)
        assert torch.count_nonzero(grads[2]) == torch.count_nonzero(grads[3]) == 0
        if rows == 0:
            assert all(torch.count_nonzero(g) == 0 for g in grads)


def test_opcheck():
    norm = _norm(128, False, torch.float32)
    x = torch.randn(65, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    torch.library.opcheck(
        _forward, (x, *norm.parameters(), norm.eps, norm.gating_weight)
    )


@pytest.mark.parametrize("activation_checkpoint", [False, True])
def test_compile_model_forward_backward(activation_checkpoint):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = _norm(128, True, torch.float32)
            self.proj = nn.Linear(128, 128, device="cuda", dtype=torch.bfloat16)
            self.second = _norm(128, False, torch.float32)

        def forward(self, x):
            x = self.proj(self.first(x))
            if activation_checkpoint:
                return checkpoint(self.second, x, use_reentrant=False)
            return self.second(x)

    model = Model()
    x = torch.randn(2, 65, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    expected = model(x)
    expected_grads = torch.autograd.grad(
        expected.float().sum(), (x, *model.parameters())
    )
    compiled = torch.compile(model, fullgraph=True)
    actual = compiled(x)
    actual_grads = torch.autograd.grad(actual.float().sum(), (x, *model.parameters()))
    for a, b in zip((actual, *actual_grads), (expected, *expected_grads)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_cuda_graph_replay():
    norm = _norm(128, True, torch.float32)
    x = torch.randn(65, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def step():
        y = norm(x)
        return y, torch.autograd.grad(y.sum(), (x, *norm.parameters()))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, grads = step()
    for _ in range(10):
        with torch.no_grad():
            x.normal_()
        graph.replay()
        expected, expected_grads = step()
        for a, b in zip((output, *grads), (expected, *expected_grads)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def _dtensor_worker(rank, rendezvous):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor, Replicate, Shard

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("sequence",))
        torch.manual_seed(42)
        reference = _norm(128, True, torch.float32)
        norm = copy.deepcopy(reference)
        for name, param in list(norm.named_parameters()):
            setattr(
                norm, name, nn.Parameter(distribute_tensor(param, mesh, (Replicate(),)))
            )
        # Uneven row shards exercise both a full and a partial 64-row tile.
        x = torch.randn(
            2, 129, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        dy = torch.randn_like(x)
        expected = reference(x)
        expected_grads = torch.autograd.grad(expected, (x, *reference.parameters()), dy)
        for compile_model in (False, True):
            distributed_x = distribute_tensor(
                x.detach(), mesh, (Shard(1),)
            ).requires_grad_()
            distributed_dy = distribute_tensor(dy, mesh, (Shard(1),))
            model = torch.compile(norm, fullgraph=True) if compile_model else norm
            actual = model(distributed_x)
            grads = torch.autograd.grad(
                actual, (distributed_x, *norm.parameters()), distributed_dy
            )
            _compare(
                (actual.full_tensor(), *(g.full_tensor() for g in grads)),
                (expected, *expected_grads),
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two Hopper GPUs")
def test_dtensor_sequence_sharding(tmp_path):
    torch.multiprocessing.spawn(
        _dtensor_worker, args=(f"file://{tmp_path / 'nccl'}",), nprocs=2
    )
