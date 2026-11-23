"""Check the prepared expert-weight handoff with CPU replacements for GPU kernels."""

import sys
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend

from torchtitan.config import ActivationCheckpointConfig
from torchtitan.distributed.activation_checkpoint import _apply_ac_to_block
from torchtitan.ops.float8 import deep_gemm_handle as dg


class _Provider:
    def __init__(self):
        self.generation = 0
        self.planes = {}
        self.reads = []

    def acquire(self, value):
        self.planes["w13"] = (
            torch.full((1, 128, 256), value).to(torch.float8_e4m3fn),
            torch.ones(1, 1, 2),
        )

    def release(self):
        # Overwrite the borrowed storage as if another GFSDP group reused it.
        for data, scale in self.planes.values():
            data.view(torch.uint8).zero_()
            scale.zero_()
        self.planes.clear()

    def resolve_fp8_weight(self, projection, expected_generation=None):
        if projection not in self.planes:
            raise RuntimeError("test slot is idle")
        if expected_generation is not None and expected_generation != self.generation:
            raise RuntimeError("test weight generation changed")
        self.reads.append(expected_generation)
        return *self.planes[projection], self.generation


@pytest.fixture
def grouped_kernels(monkeypatch):
    from torchtitan.ops.float8 import grouped_gemm as grouped

    def grouped_cpu(a, a_scale, b, b_scale, indices):
        x = a.float() * a_scale.repeat_interleave(128, dim=-1)
        w = b[0].float() * b_scale[0].repeat_interleave(128, dim=-2).repeat_interleave(
            128, dim=-1
        )
        return (x @ w.T).to(torch.bfloat16)

    def wgrad_cpu(a, a_scale, b, b_scale, d, ks, ks_cpu, ks_tensor):
        rows = sum(ks)
        n, k = d.shape[-2:]
        d.copy_((a.float().view(n, rows) @ b.float().view(k, rows).T).unsqueeze(0))

    def quantize(x):
        return x.to(torch.float8_e4m3fn), x.new_ones(
            (x.shape[0], x.shape[1] // 128), dtype=torch.float32
        )

    def row2col(x, indices, sizes, scale):
        return x.T.contiguous().reshape(-1), torch.ones((x.shape[1], x.shape[0] // 128))

    def transpose(t):
        return t.transpose(-1, -2).contiguous()

    monkeypatch.setattr(dg, "_m_grouped_fp8_gemm", grouped_cpu)
    monkeypatch.setattr(
        dg, "_transpose_fp8_weight", lambda d, s: (transpose(d), transpose(s))
    )
    monkeypatch.setattr(grouped, "_cached_handle", object.__new__(dg._DeepGemmHandle))
    monkeypatch.setattr(grouped, "row_quantization_1x128_fn", quantize)
    monkeypatch.setattr(grouped, "row2col_requantization_deepgemm_layout_fn", row2col)
    monkeypatch.setattr(grouped, "transpose_last_two_dims_fn", transpose)
    lib = torch.library.Library("torchtitan", "IMPL")
    lib.impl("m_grouped_fp8_gemm_nt_contiguous", grouped_cpu, "CPU")
    lib.impl("k_grouped_fp8_gemm_contiguous", wgrad_cpu, "CPU")
    try:
        yield grouped.GroupedGemmFp8Wrapper()
    finally:
        lib._destroy()


class _ExpertUp(torch.nn.Module):
    def __init__(self, gemm, prepared_weight):
        super().__init__()
        self.gemm = gemm
        self.prepared_weight = prepared_weight
        self.weight = torch.nn.Parameter(
            torch.full((), float("nan"), dtype=torch.bfloat16).expand(1, 2, 64, 256)
        )

    def forward(self, x):
        weight = self.weight.view(1, 128, 256)
        sizes = torch.tensor([128], dtype=torch.int64)
        ids = torch.zeros(x.shape[0], dtype=torch.int32)
        out, _ = self.gemm.group_gemm(
            x,
            [128],
            weight,
            m_indices=ids,
            tensor_groups_sizes_padded=sizes,
            tensor_groups_sizes_cpu=sizes.clone(),
            triton_row2col=True,
            prepared_weight=self.prepared_weight,
        )
        return out.square()


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    "mode,reads", [(None, 1), ("selective", 2), ("save:experts_up", 1)]
)
def test_prepared_fp8_weight_rebinds_across_generations(
    grouped_kernels, compiled, mode, reads
):
    if compiled and getattr(sys.modules.get("triton"), "__version__", None) == "0.0.0":
        pytest.skip("Run without the conftest Triton stub for Dynamo coverage")

    provider = _Provider()
    block = _ExpertUp(grouped_kernels, dg.PreparedExpertWeight(provider, "w13"))
    model = block
    if mode is not None:
        model = _apply_ac_to_block(
            model, ActivationCheckpointConfig(selective_save=[]), mode=mode
        )
    backend = CompileCounterWithBackend("aot_eager")
    if compiled:
        model = torch.compile(model, backend=backend, fullgraph=True)

    for value in (0.125, -0.25):
        provider.generation += 1
        provider.reads.clear()
        provider.acquire(value)
        x = torch.full((128, 256), 0.0625, dtype=torch.bfloat16, requires_grad=True)
        block.weight.grad = None
        out = model(x)
        provider.release()
        provider.acquire(value)
        out.sum().backward()
        provider.release()

        x_ref = x.detach().clone().requires_grad_()
        w_ref = torch.full_like(block.weight, value).requires_grad_()
        expected = (x_ref @ w_ref.view(128, 256).T).square()
        expected.sum().backward()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, x_ref.grad, rtol=0, atol=0)
        torch.testing.assert_close(block.weight.grad, w_ref.grad, rtol=0, atol=0)
        assert provider.reads == [None] + [provider.generation] * (reads + 1)

    if compiled:
        assert backend.frame_count == 1


@pytest.mark.parametrize(
    "failure", ["test slot is idle", "test weight generation changed"]
)
@pytest.mark.parametrize("compiled_sac", [False, True])
def test_prepared_fp8_weight_rejects_invalid_backward_slot(
    grouped_kernels, failure, compiled_sac
):
    if (
        compiled_sac
        and getattr(sys.modules.get("triton"), "__version__", None) == "0.0.0"
    ):
        pytest.skip("Run without the conftest Triton stub for Dynamo coverage")
    provider = _Provider()
    provider.acquire(0.125)
    model = _ExpertUp(grouped_kernels, dg.PreparedExpertWeight(provider, "w13"))
    if compiled_sac:
        model = _apply_ac_to_block(
            model, ActivationCheckpointConfig(selective_save=[]), mode="selective"
        )
        model = torch.compile(model, backend="aot_eager", fullgraph=True)
    x = torch.full((128, 256), 0.0625, dtype=torch.bfloat16, requires_grad=True)
    out = model(x)
    provider.release()
    if failure == "test weight generation changed":
        provider.generation += 1
        provider.acquire(0.125)
    with pytest.raises(RuntimeError, match=failure):
        out.sum().backward()


def test_empty_prepared_gemm_uses_the_bf16_parameter_and_activation_edges(
    grouped_kernels,
):
    provider = _Provider()
    model = _ExpertUp(grouped_kernels, dg.PreparedExpertWeight(provider, "w13"))
    x = torch.empty((0, 256), dtype=torch.bfloat16, requires_grad=True)
    out = model(x)
    out.sum().backward()
    assert out.shape == (0, 128)
    assert x.grad.shape == x.shape
    torch.testing.assert_close(model.weight.grad, torch.zeros_like(model.weight))
    assert not provider.reads


def test_prepared_experts_keep_routing_and_skip_dispatch_quantization(
    grouped_kernels, monkeypatch
):
    from torchtitan.distributed.deepep.fp8_state import FP8DispatchState
    from torchtitan.models.common.moe.experts.gmm import GroupedExperts
    from torchtitan.ops.float8.fp8_tensor import Float8BlockwiseQTensor
    from torchtitan.ops.float8.quantize_expert_weights import quantize_expert_weight
    from torchtitan.ops.float8.triton_kernels import utils

    def block_planes(w):
        e, n, k = w.shape
        return w.detach().to(torch.float8_e4m3fn), torch.ones(e, n // 128, k // 128)

    def row_payload(x):
        return Float8BlockwiseQTensor(
            x.detach().to(torch.float8_e4m3fn),
            torch.ones(x.shape[0], x.shape[1] // 128),
            x.dtype,
            scale_block_first=False,
        )

    class Combine(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, edge):
            return value.clone()

        @staticmethod
        def backward(ctx, grad):
            return None, row_payload(grad)

    class Dispatcher(torch.nn.Module):
        def dispatch(self, x, scores, expert_ids, sizes, **kwargs):
            assert kwargs["use_fp8_dispatch"]
            weights = kwargs.get("expert_weights")
            self.quantized = weights is not None
            assert weights is None or len(weights) == 2
            state = FP8DispatchState(
                permuted_probs=scores.flatten(),
                padded_group_sizes_t=sizes,
                padded_group_sizes_list=[128],
                padded_tokens=row_payload(x),
                uses_gradient_edges=True,
                fp8_expert_weights=(
                    [quantize_expert_weight(w, block_planes(w)) for w in weights]
                    if weights is not None
                    else None
                ),
            )
            return x, sizes, SimpleNamespace(state=SimpleNamespace(fp8=state))

        def combine(self, value, metadata, x):
            return Combine.apply(value, metadata.state.fp8.output_fp8_edge)

    def m_indices(sizes, num_rows):
        return torch.zeros(num_rows, dtype=torch.int32)

    monkeypatch.setattr(utils, "build_m_indices", m_indices)
    grouped_kernels.group_gemm = partial(
        grouped_kernels.group_gemm, triton_row2col=True
    )
    experts = GroupedExperts(
        GroupedExperts.Config(dim=384, hidden_dim=128, num_experts=1)
    )
    experts.to(torch.bfloat16)
    experts.token_dispatcher = Dispatcher()
    experts._use_float8_grouped_gemm = True
    experts._sparse_fused_swiglu_quant = False
    experts.handle1 = experts.handle2 = grouped_kernels
    with torch.no_grad():
        experts.w13_E2FD.fill_(0.125)
        experts.w2_EDF.fill_(0.25)

    def run():
        experts.zero_grad()
        x = torch.full(
            (1, 128, 384), 0.015625, dtype=torch.bfloat16, requires_grad=True
        )
        scores = torch.full((1, 128, 1), 0.25, dtype=torch.float32, requires_grad=True)
        out = experts(
            x, scores, torch.zeros_like(scores, dtype=torch.long), torch.tensor([128])
        )
        out.sum().backward()
        return (
            out.detach(),
            x.grad,
            scores.grad,
            experts.w13_E2FD.grad,
            experts.w2_EDF.grad,
        )

    expected = run()
    assert experts.token_dispatcher.quantized
    provider = _Provider()
    projections = ("w13", "w2")
    provider.planes.update(
        zip(projections, map(block_planes, experts._local_expert_weights()))
    )
    experts._gfsdp_fp8_weights = tuple(
        dg.PreparedExpertWeight(provider, p) for p in projections
    )
    anchor = torch.full((), float("nan"), dtype=torch.bfloat16)
    experts.w13_E2FD = torch.nn.Parameter(anchor.expand(experts.w13_E2FD.shape))
    experts.w2_EDF = torch.nn.Parameter(anchor.expand(experts.w2_EDF.shape))
    actual = run()
    assert not experts.token_dispatcher.quantized
    generation = provider.generation
    assert provider.reads == [
        None,
        generation,
        None,
        generation,
        generation,
        generation,
    ]
    for value, target in zip(actual, expected):
        torch.testing.assert_close(value, target, rtol=0, atol=0)
