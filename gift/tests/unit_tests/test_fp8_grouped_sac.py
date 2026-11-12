"""CPU dispatcher doubles exercise the real grouped-FP8/SAC boundary.

The doubles replace only CUDA kernels and quantizers. They establish activation
lifetime and recomputation contracts, not CUDA numerics or performance.
"""

import sys

import pytest
import torch

from torchtitan.config import ActivationCheckpointConfig
from torchtitan.distributed.activation_checkpoint import _apply_ac_to_block
from torchtitan.ops.float8.deep_gemm_handle import (
    _DeepGemmHandle,
    _m_grouped_fp8_gemm_nt_contiguous,
)


@pytest.fixture
def kernels():
    calls = []

    def grouped_cpu(a, a_scale, b, b_scale, indices):
        calls.append((a.shape[-1], b.shape[1]))
        return (a.float() @ b[0].float().T).to(torch.bfloat16)

    def wgrad_cpu(a, a_scale, b, b_scale, d, ks, ks_cpu, ks_tensor):
        rows = sum(ks)
        calls.append(("wgrad", rows))
        n, k = d.shape[-2:]
        d.copy_((a.float().view(n, rows) @ b.float().view(k, rows).T).unsqueeze(0))

    lib = torch.library.Library("torchtitan", "IMPL")
    lib.impl("m_grouped_fp8_gemm_nt_contiguous", grouped_cpu, "CPU")
    lib.impl("k_grouped_fp8_gemm_contiguous", wgrad_cpu, "CPU")
    try:
        yield calls
    finally:
        lib._destroy()


def test_grouped_operator_has_fresh_output_and_fake_contract(kernels):
    a = torch.ones(3, 256).to(torch.float8_e4m3fn)
    b = torch.ones(1, 128, 256).to(torch.float8_e4m3fn)
    scale = torch.ones(3, 2)
    b_scale = torch.ones(1, 1, 2)
    ids = torch.zeros(3, dtype=torch.int32)
    torch.library.opcheck(
        _m_grouped_fp8_gemm_nt_contiguous,
        (a, scale, b, b_scale, ids),
        # CPU float8 lacks the equality kernel used by test_schema.
        test_utils=(
            "test_autograd_registration",
            "test_faketensor",
            "test_aot_dispatch_dynamic",
        ),
    )
    handle = object.__new__(_DeepGemmHandle)
    out = handle.m_grouped_fp8_gemm_nt_contiguous((a, scale), (b, b_scale), ids)
    assert out.shape == (3, 128)
    assert out.dtype == torch.bfloat16
    assert out.is_contiguous()
    assert out.data_ptr() not in (a.data_ptr(), b.data_ptr())
    torch.testing.assert_close(out, torch.full((3, 128), 256.0, dtype=torch.bfloat16))


@pytest.mark.parametrize("trainable", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    "mode,expected_calls", [(None, 2), ("selective", 3), ("save:experts_up", 2)]
)
def test_real_grouped_forward_backward_survives_saved_sac(
    kernels, monkeypatch, compiled, mode, expected_calls, trainable
):
    if compiled and getattr(sys.modules.get("triton"), "__version__", None) == "0.0.0":
        pytest.skip(
            "Unit-test Triton stub blocks Dynamo; run with --noconftest plus real import shims"
        )
    from torchtitan.ops.float8 import grouped_gemm as grouped
    from torchtitan.ops.float8.fp8_tensor import Float8BlockwiseQTensor

    def quantize(x):
        return x.to(torch.float8_e4m3fn), x.new_ones(
            (x.shape[0], x.shape[1] // 128), dtype=torch.float32
        )

    monkeypatch.setattr(grouped, "row_quantization_1x128_fn", quantize)
    monkeypatch.setattr(
        grouped,
        "transpose_last_two_dims_fn",
        lambda t: t.transpose(-1, -2).contiguous(),
    )
    monkeypatch.setattr(grouped, "_cached_handle", object.__new__(_DeepGemmHandle))

    def row2col(x, indices, counts, scale):
        return x.T.contiguous().reshape(-1), torch.ones((x.shape[1], x.shape[0] // 128))

    monkeypatch.setattr(grouped, "row2col_requantization_deepgemm_layout_fn", row2col)
    # Retain real autograd and early storage release in both wgrad branches.
    data = torch.full((1, 128, 256), 0.125).to(torch.float8_e4m3fn)
    weights = Float8BlockwiseQTensor(
        data, torch.ones(1, 1, 2), torch.bfloat16, scale_block_first=False
    )
    weights.requires_grad_(trainable)
    ids = torch.zeros(128, dtype=torch.int32)
    sizes = torch.tensor([128], dtype=torch.int64)
    sizes_cpu = sizes.clone()

    class GroupedBlock(torch.nn.Module):
        def forward(self, x):
            return grouped._GroupedGemm.apply(
                None, x, [128], weights, ids, sizes, sizes_cpu, True, False
            ).sigmoid()

    def run(selected_mode, do_compile):
        model = GroupedBlock()
        if selected_mode:
            model = _apply_ac_to_block(
                model, ActivationCheckpointConfig(selective_save=[]), mode=selected_mode
            )
        if do_compile:
            model = torch.compile(model, backend="aot_eager", fullgraph=True)
        weights.grad = None
        x = torch.full((128, 256), 0.0625, dtype=torch.bfloat16, requires_grad=True)
        out = model(x)
        loss = out.square().sum()
        loss.backward()
        result = [out.detach(), loss.detach(), x.grad]
        if trainable:
            result.append(weights.grad.detach().clone())
        return result

    reference = run(None, False)
    kernels.clear()
    result = run(mode, compiled)
    for actual, expected in zip(result, reference):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(kernels) == expected_calls + int(trainable)
    assert kernels[-1 - int(trainable)] == (128, 256)
    if trainable:
        assert kernels[-1] == ("wgrad", 128)
        assert result[3].abs().sum() > 0
    assert result[2].abs().sum() > 0
