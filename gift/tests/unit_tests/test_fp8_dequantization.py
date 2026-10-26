# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FP8 weight dequantization through the monitoring decoder on CUDA."""

import importlib.util

import pytest
import torch

from torchtitan.callbacks.metrics.stats import dequantize_fp8

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FP8 quantization/dequantization requires CUDA",
)


@pytest.mark.parametrize("shape", [(1, 256, 256), (2, 256, 512)])
def test_quantized_expert_weight_round_trip(shape):
    from torchtitan.ops.float8.quantize_expert_weights import quantize_expert_weight

    torch.manual_seed(42)
    original = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    value = quantize_expert_weight(original)

    actual = dequantize_fp8(value)

    scale = value._scale.repeat_interleave(128, dim=-2).repeat_interleave(128, dim=-1)
    expected = (value._data.float() * scale).bfloat16()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, original, rtol=0.065, atol=1e-3)


def test_cudacherry_weight_round_trip():
    if importlib.util.find_spec("cudacherry") is None:
        pytest.skip("cudacherry is not installed")
    from torchtitan.ops.float8._cudacherry import quantize_blk128x128

    torch.manual_seed(42)
    original = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
    value = quantize_blk128x128(original)
    scale = (
        value._scale[:2, :2]
        .repeat_interleave(128, dim=-2)
        .repeat_interleave(128, dim=-1)
    )
    actual = dequantize_fp8(value)

    expected = (value._data.float() * scale).bfloat16()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, original, rtol=0.08, atol=3e-3)
