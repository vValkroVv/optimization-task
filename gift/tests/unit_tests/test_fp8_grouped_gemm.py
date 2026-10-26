# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The fp8 grouped GEMM against a reference, and the group sizes it rejects.

Run:
    pytest tests/unit_tests/test_fp8_grouped_gemm.py

Skipped without a suitable GPU, so the CPU CI job passes over it -- the same
convention as the other GPU-gated tests in this directory (``test_compile_moe``,
``test_flash_attention``, ``test_fsdp_moe_sharding``).

Every comparison here is against the *dequantized* operands, in fp32. That
isolates what the GEMM does from what the quantization lost: the only
difference left is accumulation order, which is why the tolerances are tight
for the forward and looser for the gradients, where the operands are quantized
a second time on the way in.
"""

import pytest
import torch
import os

os.environ["DG_CACHE_FORCE_SKIP"] = "1"

pytest.importorskip("triton")
pytest.importorskip("deep_gemm")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="DeepGEMM's blockwise fp8 grouped GEMM is SM90-only",
)

from torchtitan.ops.float8 import grouped_gemm as gg  # noqa: E402
from torchtitan.ops.float8.fp8_tensor import (  # noqa: E402
    Float8BlockwiseQTensor,
    InputLayoutType,
)
from torchtitan.ops.float8.quantize_expert_weights import (  # noqa: E402
    quantize_expert_weight,
)
from torchtitan.ops.float8.triton_kernels.block_quantization import (  # noqa: E402
    block_quantization_128x128_fn,
)
from torchtitan.ops.float8.triton_kernels.row_quantization import (  # noqa: E402
    row_quantization_1x128_fn,
)
from torchtitan.ops.float8.triton_kernels.utils import build_m_indices  # noqa: E402

BLOCK = 128
DEV = "cuda"
EXPERTS, GROUP_ROWS = 2, BLOCK
ROWS = EXPERTS * GROUP_ROWS
DIM, OUT = 2 * BLOCK, 2 * BLOCK
GROUPS = [GROUP_ROWS] * EXPERTS


def _layout():
    counts = torch.tensor(GROUPS, dtype=torch.int32, device=DEV)
    return counts, build_m_indices(counts, num_rows=ROWS)


def _quantized_weights(w: torch.Tensor) -> Float8BlockwiseQTensor:
    data, scale = block_quantization_128x128_fn(w)
    return Float8BlockwiseQTensor(
        data,
        scale,
        torch.bfloat16,
        InputLayoutType.ROWWISE,
        2,
        scale_block_first=False,
    )


def _dequantize_rowwise(data: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return data.float() * scale.repeat_interleave(BLOCK, dim=-1)


def _dequantize_blockwise(weights: Float8BlockwiseQTensor) -> torch.Tensor:
    scale = weights._scale.repeat_interleave(BLOCK, dim=-2).repeat_interleave(
        BLOCK, dim=-1
    )
    return weights._data.float() * scale


def _reference(x: torch.Tensor, w: torch.Tensor, m_indices: torch.Tensor):
    """``out[i] = x[i] @ w[m_indices[i]].T``, one group at a time."""
    out = torch.empty(x.shape[0], w.shape[1], dtype=torch.float32, device=x.device)
    for expert in range(w.shape[0]):
        rows = m_indices == expert
        out[rows] = x[rows] @ w[expert].t()
    return out


@pytest.fixture
def wrapper():
    return gg.GroupedGemmFp8Wrapper(num_sms=116)


# --- forward ---


def test_the_grouped_gemm_matches_a_per_group_matmul(wrapper):
    torch.manual_seed(0)
    counts, m_indices = _layout()
    x = torch.randn(ROWS, DIM, dtype=torch.bfloat16, device=DEV)
    w = torch.randn(EXPERTS, OUT, DIM, dtype=torch.bfloat16, device=DEV)
    weights = _quantized_weights(w)

    out, edge = wrapper.group_gemm(
        tensor_groups=x,
        weights=weights,
        tensor_groups_sizes=GROUPS,
        m_indices=m_indices,
        tensor_groups_sizes_padded=counts,
        triton_row2col=True,
    )

    expected = _reference(
        _dequantize_rowwise(*row_quantization_1x128_fn(x)),
        _dequantize_blockwise(weights),
        m_indices,
    )
    assert out.shape == (ROWS, OUT) and out.dtype == torch.bfloat16
    assert edge is None
    torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=8e-3)


def test_an_already_quantized_payload_is_not_quantized_again(wrapper):
    """The dispatch hands the experts an fp8 payload; re-quantizing it would
    lose another mantissa's worth for nothing."""
    torch.manual_seed(0)
    counts, m_indices = _layout()
    x = torch.randn(ROWS, DIM, dtype=torch.bfloat16, device=DEV)
    data, scale = row_quantization_1x128_fn(x)
    payload = Float8BlockwiseQTensor(
        data,
        scale,
        torch.bfloat16,
        InputLayoutType.ROWWISE,
        1,
        scale_block_first=False,
    )
    w = torch.randn(EXPERTS, OUT, DIM, dtype=torch.bfloat16, device=DEV)
    weights = _quantized_weights(w)

    from_packed, _ = wrapper.group_gemm(
        tensor_groups=payload,
        weights=weights,
        tensor_groups_sizes=GROUPS,
        m_indices=m_indices,
        tensor_groups_sizes_padded=counts,
        triton_row2col=True,
    )
    from_bf16, _ = wrapper.group_gemm(
        tensor_groups=x,
        weights=weights,
        tensor_groups_sizes=GROUPS,
        m_indices=m_indices,
        tensor_groups_sizes_padded=counts,
        triton_row2col=True,
    )

    assert torch.equal(from_packed, from_bf16)


# --- backward ---


def test_the_gradients_match_a_per_group_matmul(wrapper):
    torch.manual_seed(0)
    counts, m_indices = _layout()
    x = torch.randn(ROWS, DIM, dtype=torch.bfloat16, device=DEV, requires_grad=True)
    w = torch.randn(
        EXPERTS, OUT, DIM, dtype=torch.bfloat16, device=DEV, requires_grad=True
    )
    # Through the quantizing Function, as the experts build it: the wgrad
    # belongs to the bf16 parameter, and the fp8 planes are values on the way.
    weights = quantize_expert_weight(w)
    grad_out = torch.randn(ROWS, OUT, dtype=torch.bfloat16, device=DEV)

    out, _ = wrapper.group_gemm(
        tensor_groups=x,
        weights=weights,
        tensor_groups_sizes=GROUPS,
        m_indices=m_indices,
        tensor_groups_sizes_padded=counts,
        tensor_groups_sizes_cpu=counts.cpu(),
        triton_row2col=True,
    )
    # Before the backward: it frees the quantized planes it is done with.
    g = _dequantize_rowwise(*row_quantization_1x128_fn(grad_out))
    x_q = _dequantize_rowwise(*row_quantization_1x128_fn(x.detach()))
    w_q = _dequantize_blockwise(weights)

    dgrad, wgrad = torch.autograd.grad(out, [x, w], grad_out)

    # dgrad[i] = grad_out[i] @ w[m_indices[i]]
    expected_dgrad = torch.empty(ROWS, DIM, dtype=torch.float32, device=DEV)
    expected_wgrad = torch.empty(EXPERTS, OUT, DIM, dtype=torch.float32, device=DEV)
    for expert in range(EXPERTS):
        rows = m_indices == expert
        expected_dgrad[rows] = g[rows] @ w_q[expert]
        expected_wgrad[expert] = g[rows].t() @ x_q[rows]

    assert dgrad.shape == x.shape and wgrad.shape == w.shape
    torch.testing.assert_close(dgrad.float(), expected_dgrad, rtol=5e-3, atol=8e-3)
    torch.testing.assert_close(wgrad.float(), expected_wgrad, rtol=5e-3, atol=8e-3)


def test_frozen_weights_cost_no_wgrad(wrapper):
    """The k-grouped GEMM is the expensive half of this backward.

    ``out.sum()`` rather than a random cotangent on purpose: its gradient is a
    stride-0 expand, and the quantization kernel asserts contiguity. A loss that
    ends in a sum is ordinary, so the backward has to take one.
    """
    torch.manual_seed(0)
    counts, m_indices = _layout()
    x = torch.randn(ROWS, DIM, dtype=torch.bfloat16, device=DEV, requires_grad=True)
    w = torch.randn(EXPERTS, OUT, DIM, dtype=torch.bfloat16, device=DEV)
    weights = _quantized_weights(w)  # requires_grad stays False

    out, _ = wrapper.group_gemm(
        tensor_groups=x,
        weights=weights,
        tensor_groups_sizes=GROUPS,
        m_indices=m_indices,
        tensor_groups_sizes_padded=counts,
        triton_row2col=True,
    )
    out.sum().backward()

    assert x.grad is not None
    assert weights.grad is None


# --- the group sizes the wgrad refuses ---


def _k_grouped(ks, ks_cpu, ks_tensor, rows):
    a_data = torch.zeros(rows * OUT, dtype=torch.float8_e4m3fn, device=DEV)
    a_scale = torch.zeros(1, rows // BLOCK, dtype=torch.float32, device=DEV)
    b_data = torch.zeros(rows * DIM, dtype=torch.float8_e4m3fn, device=DEV)
    b_scale = torch.zeros(1, rows // BLOCK, dtype=torch.float32, device=DEV)
    d = torch.zeros(EXPERTS, OUT, DIM, dtype=torch.bfloat16, device=DEV)
    torch.ops.torchtitan.k_grouped_fp8_gemm_contiguous(
        a_data, a_scale, b_data, b_scale, d, ks, ks_cpu, ks_tensor
    )
    return d


def test_group_sizes_off_the_128_row_grid_are_refused():
    ks = [BLOCK, BLOCK - 1]
    with pytest.raises(ValueError, match="multiples of 128"):
        _k_grouped(ks, None, torch.tensor(ks, dtype=torch.int32, device=DEV), 2 * BLOCK)


def test_group_sizes_that_overrun_the_operand_are_refused():
    ks = [BLOCK, BLOCK]
    with pytest.raises(ValueError, match="more than"):
        _k_grouped(ks, None, torch.tensor(ks, dtype=torch.int32, device=DEV), BLOCK)


def test_every_group_empty_leaves_the_zero_weight_gradient():
    """Which is what the caller allocated, so the GEMM has nothing to do."""
    ks = [0, 0]
    d = _k_grouped(ks, None, torch.tensor(ks, dtype=torch.int32, device=DEV), 2 * BLOCK)
    assert (d == 0).all()


def test_the_host_copy_of_the_group_sizes_is_used_when_there_is_no_list():
    """A compiled blocking dispatch cannot pass a Python list through the graph,
    but the counts did reach the host, so the backward must not sync for them.

    The device copy holds sizes that would pass, so only reading the host one
    raises.
    """
    with pytest.raises(ValueError, match="multiples of 128"):
        _k_grouped(
            None,
            torch.tensor([BLOCK, BLOCK - 1], dtype=torch.int32),
            torch.tensor([BLOCK, BLOCK], dtype=torch.int32, device=DEV),
            2 * BLOCK,
        )


# --- a rank whose token groups are all empty ---


def _empty_payload() -> Float8BlockwiseQTensor:
    """The padded fp8 payload of a rank that received no tokens at all."""
    return Float8BlockwiseQTensor(
        torch.zeros(0, DIM, dtype=torch.float8_e4m3fn, device=DEV),
        torch.zeros(0, DIM // BLOCK, dtype=torch.float32, device=DEV),
        torch.bfloat16,
        InputLayoutType.ROWWISE,
        1,
        scale_block_first=False,
    )


@pytest.mark.parametrize("fp8_grad_output", [False, True], ids=["value", "edge"])
def test_an_empty_rank_returns_the_pair_the_general_path_returns(
    wrapper, fp8_grad_output
):
    """It takes a separate Function, so the caller would otherwise read a
    different shape of result depending on the data."""
    out, edge = wrapper.group_gemm(
        tensor_groups=_empty_payload(),
        weights=_quantized_weights(
            torch.randn(EXPERTS, OUT, DIM, dtype=torch.bfloat16, device=DEV)
        ),
        tensor_groups_sizes=[0] * EXPERTS,
        m_indices=torch.empty(0, dtype=torch.int32, device=DEV),
        tensor_groups_sizes_padded=torch.zeros(EXPERTS, dtype=torch.int32, device=DEV),
        fp8_grad_output=fp8_grad_output,
    )

    assert out.shape == (0, OUT) and out.dtype == torch.bfloat16
    if not fp8_grad_output:
        assert edge is None
        return
    assert isinstance(edge, Float8BlockwiseQTensor)
    assert edge._data.dtype == torch.float8_e4m3fn
    assert edge._data.shape == (0, OUT)


def test_an_empty_rank_still_produces_the_gradients_it_owes(wrapper):
    """Zeros, in the slot the non-empty path would have used: with an edge the
    payload is off the graph and the data gradient belongs to the edge."""
    w = torch.randn(
        EXPERTS, OUT, DIM, dtype=torch.bfloat16, device=DEV, requires_grad=True
    )
    edge_in = torch.zeros(0, DIM, dtype=torch.bfloat16, device=DEV).requires_grad_()

    out, _ = wrapper.group_gemm(
        tensor_groups=_empty_payload(),
        weights=quantize_expert_weight(w),
        tensor_groups_sizes=[0] * EXPERTS,
        m_indices=torch.empty(0, dtype=torch.int32, device=DEV),
        tensor_groups_sizes_padded=torch.zeros(EXPERTS, dtype=torch.int32, device=DEV),
        grad_edge=edge_in,
    )
    out.sum().backward()

    assert edge_in.grad is not None and edge_in.grad.shape == (0, DIM)
    assert w.grad is not None and (w.grad == 0).all()
