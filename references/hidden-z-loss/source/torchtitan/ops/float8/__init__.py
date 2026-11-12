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

import torch

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
    "make_zloss_state",
]


def make_zloss_state(
    scales: tuple[torch.Tensor | None, ...],
    totals: tuple[torch.Tensor | None, ...],
) -> object:
    """Give the cudacherry z-loss state of one fused SwiGLU layer.

    ``wire_aux_loss`` calls it two times for each fused site, outside each
    compiled region: one state has the totals, and one state has no totals for
    ``suppress_aux_recording``. The import of cudacherry occurs here, thus a
    model that declares a fused site does not import cudacherry at its build.
    """
    from torchtitan.ops.float8._cudacherry import zloss_state

    return zloss_state(scales, totals)
