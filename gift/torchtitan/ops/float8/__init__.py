# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Blockwise float8 (DeepSeek-style 1x128 / 128x128) MoE training kernels.

This package hosts the fp8 tensor wrapper and triton kernels used by the fp8
MoE training path: rowwise/blockwise quantization, row-to-column
requantization for wgrad, fused SwiGLU + quantization, and padded expert-major
token permutation. Requires CUDA and triton; import of this package is safe
on CPU-only hosts, kernel launches are not.
"""

from torchtitan.ops.float8.fp8_tensor import (
    clear_tensor_data,
    Float8BlockwiseQTensor,
    Float8OperandTuple,
    InputLayoutType,
    is_fp8_operand_tuple,
)

__all__ = [
    "Float8BlockwiseQTensor",
    "Float8OperandTuple",
    "InputLayoutType",
    "clear_tensor_data",
    "is_fp8_operand_tuple",
]
