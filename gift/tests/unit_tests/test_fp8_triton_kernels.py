# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What the float8 triton kernels compute, against a reference in torch.

Run:
    pytest tests/unit_tests/test_fp8_triton_kernels.py

Skipped without a suitable GPU, so the CPU CI job passes over it -- the same
convention as the other GPU-gated tests in this directory (``test_compile_moe``,
``test_flash_attention``, ``test_fsdp_moe_sharding``).

These are the kernels the fp8 MoE path runs on every token. Everything else
reaches them through an opaque grouped GEMM or a compiled graph, so this is the
only place their arithmetic is checked directly.

The quantizations are compared exactly rather than loosely. Every scale is a
power of two (``exp2(ceil(log2(amax / 448)))``), so multiplying by its inverse
is exact and the e4m3 rounding is the same round-to-nearest-even on both sides:
a mismatch here is a real difference, not accumulated float error. The fused
SwiGLU is the exception -- it scales by a precomputed ``1 / 448``, which is not
exactly representable -- so it is compared within the e4m3 step.
"""

import pytest
import torch

pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the float8 kernels are CUDA-only"
)

from torchtitan.ops.float8.triton_kernels.block_quantization import (  # noqa: E402
    block_quantization_128x128_fn,
    quantize_fp32_to_blockwise_fp8_,
)
from torchtitan.ops.float8.triton_kernels.block_transpose import (  # noqa: E402
    transpose_last_two_dims_fn,
)
from torchtitan.ops.float8.triton_kernels.fused_swiglu_quantization import (  # noqa: E402
    fused_swiglu_quantized_fn,
)
from torchtitan.ops.float8.triton_kernels.row_quantization import (  # noqa: E402
    row_quantization_1x128_fn,
)
from torchtitan.ops.float8.triton_kernels.utils import build_m_indices  # noqa: E402

FP8 = torch.float8_e4m3fn
BLOCK = 128
DEV = "cuda"


def _pow2_scale(amax: torch.Tensor) -> torch.Tensor:
    return torch.exp2(torch.ceil(torch.log2((amax / 448.0).clamp_min(1e-30))))


# --- the per-row expert index the grouped GEMMs are driven by ---


def test_padded_counts_cover_every_row():
    """Production feeds the padded counts, so nothing is left unassigned."""
    counts = torch.tensor([128, 256], dtype=torch.int32, device=DEV)

    m_indices = build_m_indices(counts, num_rows=384)

    expected = torch.tensor([0] * 128 + [1] * 256, dtype=torch.int32, device=DEV)
    assert torch.equal(m_indices, expected)


def test_the_capacity_tail_is_marked_invalid():
    """A fixed-capacity payload is longer than its groups, and the rows past
    them are never written -- every kernel has to be told to skip them."""
    counts = torch.tensor([128, 128], dtype=torch.int32, device=DEV)

    m_indices = build_m_indices(counts, num_rows=384)

    expected = torch.tensor(
        [0] * 128 + [1] * 128 + [-1] * 128, dtype=torch.int32, device=DEV
    )
    assert torch.equal(m_indices, expected)


def test_unpadded_counts_are_aligned_and_their_slack_marked():
    """Each group still starts on a 128-row boundary, so the rows between a
    group's count and its alignment are not part of it."""
    counts = torch.tensor([100, 200], dtype=torch.int32, device=DEV)

    m_indices = build_m_indices(counts, num_rows=384).tolist()

    assert m_indices[:100] == [0] * 100
    assert m_indices[100:128] == [-1] * 28
    assert m_indices[128:328] == [1] * 200
    assert m_indices[328:] == [-1] * 56


def test_an_expert_with_no_tokens_occupies_no_rows():
    counts = torch.tensor([0, 128], dtype=torch.int32, device=DEV)
    m_indices = build_m_indices(counts, num_rows=128)
    assert torch.equal(m_indices, torch.ones(128, dtype=torch.int32, device=DEV))


# --- rowwise 1x128 activation quantization ---


@pytest.mark.parametrize("rows", [256, 100])
def test_rowwise_quantization_matches_the_reference(rows):
    torch.manual_seed(0)
    x = torch.randn(rows, 2 * BLOCK, dtype=torch.bfloat16, device=DEV) * 7.0

    data, scale = row_quantization_1x128_fn(x)

    blocks = x.float().reshape(rows, 2, BLOCK)
    expected_scale = _pow2_scale(blocks.abs().amax(-1))
    expected_data = (blocks / expected_scale[..., None]).to(FP8).reshape(rows, -1)

    assert data.shape == (rows, 2 * BLOCK) and data.dtype == FP8
    assert scale.shape == (rows, 2) and scale.dtype == torch.float32
    assert torch.equal(scale, expected_scale)
    assert torch.equal(data.view(torch.uint8), expected_data.view(torch.uint8))


# --- blockwise 128x128 weight quantization ---


def test_blockwise_quantization_matches_the_reference():
    torch.manual_seed(0)
    experts, out_dim, in_dim = 2, 2 * BLOCK, 3 * BLOCK
    w = torch.randn(experts, out_dim, in_dim, dtype=torch.bfloat16, device=DEV)

    data, scale = block_quantization_128x128_fn(w)

    flat = w.float().reshape(experts * out_dim, in_dim)
    blocks = flat.reshape(experts * out_dim // BLOCK, BLOCK, in_dim // BLOCK, BLOCK)
    expected_scale = _pow2_scale(blocks.abs().amax(dim=(1, 3)))
    expected_data = (
        (blocks / expected_scale[:, None, :, None])
        .to(FP8)
        .reshape(experts, out_dim, in_dim)
    )

    assert data.shape == w.shape and data.dtype == FP8
    assert scale.shape == (experts, out_dim // BLOCK, in_dim // BLOCK)
    assert torch.equal(
        scale, expected_scale.reshape(experts, out_dim // BLOCK, in_dim // BLOCK)
    )
    assert torch.equal(data.view(torch.uint8), expected_data.view(torch.uint8))


def _preallocated_blockwise_outputs(weight: torch.Tensor):
    data = torch.empty(weight.shape, dtype=FP8, device=weight.device)
    scales = torch.empty(
        (
            weight.shape[0],
            weight.shape[1] // BLOCK,
            weight.shape[2] // BLOCK,
        ),
        dtype=torch.float32,
        device=weight.device,
    )
    return data, scales


@pytest.mark.parametrize(
    "shape",
    [
        (2, 128, 256),
        (1, 3072, 128),  # w13 row axis
        (1, 128, 4096),  # w13 column axis
        (1, 4096, 128),  # w2 row axis
        (1, 128, 1536),  # w2 column axis
    ],
)
def test_fp32_blockwise_quantization_matches_explicit_bf16_cast(shape):
    """Keep production matrix axes while limiting each case to one expert."""
    torch.manual_seed(0)
    weight = torch.randn(shape, dtype=torch.float32, device=DEV) * 16.0
    data, scales = _preallocated_blockwise_outputs(weight)

    expected_data, expected_scales = block_quantization_128x128_fn(
        weight.to(torch.bfloat16)
    )
    returned_data, returned_scales = quantize_fp32_to_blockwise_fp8_(
        weight, data, scales
    )

    assert returned_data is data
    assert returned_scales is scales
    assert torch.equal(scales, expected_scales)
    assert torch.equal(data.view(torch.uint8), expected_data.view(torch.uint8))


def test_fp32_blockwise_quantization_accepts_contiguous_local_matrix_slice():
    torch.manual_seed(1)
    source = torch.randn((4, 256, 384), dtype=torch.float32, device=DEV)
    weight = source[1:3]
    assert weight.is_contiguous()

    data, scales = _preallocated_blockwise_outputs(weight)
    expected_data, expected_scales = block_quantization_128x128_fn(
        weight.to(torch.bfloat16)
    )
    returned_data, returned_scales = quantize_fp32_to_blockwise_fp8_(
        weight, data, scales
    )

    assert returned_data is data
    assert returned_scales is scales
    assert torch.equal(scales, expected_scales)
    assert torch.equal(data.view(torch.uint8), expected_data.view(torch.uint8))


def test_fp32_blockwise_quantization_matches_bf16_rounding_boundaries():
    weight = torch.zeros((1, 2 * BLOCK, 2 * BLOCK), dtype=torch.float32, device=DEV)
    midpoint = 1.0625  # halfway between adjacent E4M3 values 1.0 and 1.125
    near_midpoint = 2.0**-20

    # This tile also keeps its scale at one while exercising BF16 rounding
    # immediately above and below the E4M3 midpoint.
    weight[0, 0, 0] = 448.0
    weight[0, 0, 1] = midpoint + near_midpoint
    weight[0, 0, 2] = midpoint - near_midpoint
    weight[0, 0, 3] = -(midpoint + near_midpoint)
    weight[0, 0, 4] = 1.0 + 2.0**-8  # BF16 tie rounds to the even lower value
    weight[0, 0, 5] = 1.0 + 3.0 * 2.0**-8  # tie rounds to the even upper value

    # A zero tile checks the 1e-30 scale floor. The next tile maxima test
    # power-of-two scale selection on opposite sides of BF16 rounding.
    weight[0, BLOCK, 0] = 448.5  # BF16 tie rounds to 448, so scale remains 1
    weight[0, BLOCK, BLOCK] = 449.75  # rounds to 450, so scale becomes 2
    weight[0, BLOCK, BLOCK + 1] = 2.0 * (midpoint + near_midpoint)

    data, scales = _preallocated_blockwise_outputs(weight)
    expected_data, expected_scales = block_quantization_128x128_fn(
        weight.to(torch.bfloat16)
    )
    returned_data, returned_scales = quantize_fp32_to_blockwise_fp8_(
        weight, data, scales
    )

    assert returned_data is data
    assert returned_scales is scales
    assert torch.equal(scales, expected_scales)
    assert torch.equal(data.view(torch.uint8), expected_data.view(torch.uint8))


def test_fp32_blockwise_quantization_rejects_non_tile_aligned_matrices():
    weight = torch.zeros((1, BLOCK + 1, BLOCK), dtype=torch.float32, device=DEV)
    data = torch.empty(weight.shape, dtype=FP8, device=DEV)
    scales = torch.empty((1, 2, 1), dtype=torch.float32, device=DEV)

    with pytest.raises(ValueError, match="multiples of 128"):
        quantize_fp32_to_blockwise_fp8_(weight, data, scales)


# --- the transpose the dgrad needs ---


@pytest.mark.parametrize("dtype", [FP8, torch.float32])
def test_the_last_two_dims_are_transposed_exactly(dtype):
    """A pure data movement: the dgrad reads the weights the other way round."""
    torch.manual_seed(0)
    source = torch.randn(2, 2 * BLOCK, 3 * BLOCK, device=DEV)
    x = source.to(dtype)

    out = transpose_last_two_dims_fn(x)

    assert out.shape == (2, 3 * BLOCK, 2 * BLOCK) and out.dtype == dtype
    expected = x.transpose(-2, -1).contiguous()
    if dtype == FP8:
        assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))
    else:
        assert torch.equal(out, expected)


# --- the fused SwiGLU between the two grouped GEMMs ---

HID = 2 * BLOCK
GROUPS = (BLOCK, BLOCK)
IN_GROUP = sum(GROUPS)


def _swiglu_inputs(rows: int):
    torch.manual_seed(0)
    fc1 = torch.randn(rows, 2 * HID, dtype=torch.bfloat16, device=DEV)
    probs = torch.rand(rows, dtype=torch.float32, device=DEV) + 0.5
    return fc1, probs


def _swiglu_reference(fc1: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    gate, up = fc1.float().chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up * probs[:, None]


def test_the_fused_swiglu_matches_the_reference():
    fc1, probs = _swiglu_inputs(IN_GROUP)

    _edge, payload = fused_swiglu_quantized_fn(None, fc1, probs)

    dequantized = payload._data.float() * payload._scale.repeat_interleave(
        BLOCK, dim=-1
    )
    torch.testing.assert_close(
        dequantized, _swiglu_reference(fc1, probs), rtol=0.06, atol=3e-5
    )
    assert payload._data.dtype == FP8
    assert payload._scale.shape == (IN_GROUP, HID // BLOCK)
