"""Check direct dW writes through grouped GEMM with CPU kernel doubles."""

import sys

import pytest
import torch

from torchtitan.config import ActivationCheckpointConfig
from torchtitan.distributed.activation_checkpoint import _apply_ac_to_block
from torchtitan.ops.float8.deep_gemm_handle import PreparedExpertWeight
from torchtitan.ops.float8.fp8_tensor import Float8BlockwiseQTensor
from torchtitan.ops.float8.quantize_expert_weights import quantize_expert_weight

from . import test_gfsdp_prepared_fp8_consumer as prepared_consumer

grouped_kernels = prepared_consumer.grouped_kernels


@pytest.fixture
def slot_writes(grouped_kernels):
    writes = []
    projection_size = 128 * 256

    def write(a, a_scale, b, b_scale, slot, offset, ks, ks_cpu, ks_tensor):
        writes.append((slot.data_ptr(), offset, None))
        rows = sum(ks if ks is not None else ks_cpu.tolist())
        grad = a.float().view(128, rows) @ b.float().view(256, rows).T
        slot.narrow(0, offset, projection_size).copy_(grad.to(torch.bfloat16).flatten())

    def zero(slot, offset, grad_output):
        writes.append((slot.data_ptr(), offset, grad_output.dtype))
        slot.narrow(0, offset, projection_size).zero_()

    lib = torch.library.Library("torchtitan", "IMPL")
    lib.impl("write_direct_expert_dw", write, "CPU")
    lib.impl("zero_direct_expert_dw", zero, "CPU")
    try:
        yield writes
    finally:
        lib._destroy()


class _TwoProjections(torch.nn.Module):
    def __init__(self, gemm, prepared, empty):
        super().__init__()
        self.gemm = gemm
        self.prepared = prepared
        self.empty = empty
        self.slot = None
        self.weight_planes = (
            torch.full((1, 128, 256), 0.125).to(torch.float8_e4m3fn),
            torch.ones(1, 1, 2),
        )
        self.weights = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.empty((), dtype=torch.bfloat16).expand(1, 128, 256)
                )
                for _ in range(2)
            ]
        )

    def forward(self, x):
        outputs = []
        for i, weight in enumerate(self.weights):
            rows = 0 if i == 1 and self.empty else 128
            counts = torch.tensor([rows])
            gemm_weight = (
                quantize_expert_weight(weight, self.weight_planes)
                if self.prepared is None
                else weight
            )
            out, _ = self.gemm.group_gemm(
                x[:rows],
                [rows] if self.slot is None else None,
                gemm_weight,
                m_indices=torch.zeros(rows, dtype=torch.int32),
                tensor_groups_sizes_padded=counts,
                tensor_groups_sizes_cpu=counts.clone(),
                triton_row2col=True,
                prepared_weight=self.prepared,
                direct_dw_slot=self.slot,
                direct_dw_offset=i * weight.numel(),
            )
            outputs.append(out.square().sum())
        return sum(outputs)


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("mode", ["eager", "inductor_sac"])
@pytest.mark.parametrize("weight_mode", ["prepared", "wrapped"])
def test_direct_dw_matches_bf16_path_and_writes_original_slot(
    grouped_kernels, slot_writes, empty, mode, weight_mode, request
):
    if (
        mode != "eager"
        and getattr(sys.modules.get("triton"), "__version__", None) == "0.0.0"
    ):
        pytest.skip("Run without the conftest Triton stub for Inductor coverage")

    prepared = None
    if weight_mode == "prepared":
        provider = prepared_consumer._Provider()
        provider.acquire(0.125)
        prepared = PreparedExpertWeight(provider, "w13")
    model = _TwoProjections(grouped_kernels, prepared, empty)
    x = torch.full((128, 256), 0.0625, dtype=torch.bfloat16, requires_grad=True)
    expected = model(x)
    expected.backward()
    expected_dx = x.grad
    expected_dw = torch.cat([w.grad.flatten().float() for w in model.weights])
    x.grad = None
    model.zero_grad()

    candidate = model
    if mode == "inductor_sac":
        request.addfinalizer(torch._dynamo.reset)
        candidate = _apply_ac_to_block(
            candidate, ActivationCheckpointConfig(selective_save=[]), mode="selective"
        )
        candidate = torch.compile(candidate, backend="inductor", fullgraph=True)
    slot = model.slot = torch.full_like(expected_dw, float("nan"))
    actual = candidate(x)
    assert not slot_writes
    assert slot.isnan().all()
    actual.backward()

    assert sorted(slot_writes) == [
        (slot.data_ptr(), 0, None),
        (slot.data_ptr(), model.weights[0].numel(), torch.bfloat16 if empty else None),
    ]
    assert all(w.grad is None for w in model.weights)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, expected_dx, rtol=0, atol=0)
    torch.testing.assert_close(slot, expected_dw, rtol=0, atol=0)


def test_empty_direct_dw_uses_the_packed_cotangent(grouped_kernels, slot_writes):
    class Combine(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, edge):
            return value.clone()

        @staticmethod
        def backward(ctx, grad):
            return None, Float8BlockwiseQTensor(
                grad.to(torch.float8_e4m3fn),
                grad.new_empty((0, 1), dtype=torch.float32),
                grad.dtype,
                scale_block_first=False,
            )

    x = torch.empty(0, 256, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.empty(1, 128, 256, dtype=torch.bfloat16, requires_grad=True)
    slot = torch.ones(weight.numel(), dtype=torch.float32)
    out, edge = grouped_kernels.group_gemm(
        x.detach(), [0], weight, grad_edge=x, fp8_grad_output=True, direct_dw_slot=slot
    )

    Combine.apply(out, edge).sum().backward()

    assert slot_writes == [(slot.data_ptr(), 0, torch.float8_e4m3fn)]
    assert weight.grad is None
    assert x.grad.shape == x.shape
    torch.testing.assert_close(slot, torch.zeros_like(slot), rtol=0, atol=0)
