# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Blockwise 128x128 FP8 weight quantization for (E, F, D) matrices."""

import torch
import triton
import triton.language as tl

from torchtitan.ops.float8.triton_kernels.kernel_params import params_to_kernel_kwargs

from torchtitan.ops.float8.triton_kernels.utils import tensor_to_kernel_args

_BLOCK_SIZE: int = 128

STATIC_TRITON_PARAMS = {
    7168: {
        "num_warps": 8,
        "maxnreg": 255,
    },
    4096: {
        "num_warps": 8,
        "maxnreg": 255,
    },
    2048: {
        "num_warps": 8,
        "maxnreg": 255,
    },
    1536: {"num_warps": 8, "maxnreg": 128},
    1280: {"num_warps": 8, "maxnreg": 255},
}


@triton.jit
def _block_quantization_kernel(
    # tensor
    tensor_ptr,
    tensor_stride_row,
    tensor_stride_col,
    tensor_size_row,
    tensor_size_col,
    # out
    out_ptr,
    out_stride_row,
    out_stride_col,
    out_size_row,
    out_size_col,
    # scale
    scale_ptr,
    scale_stride_row,
    scale_stride_col,
    scale_size_row,
    scale_size_col,
    # constexpr
    BLOCK_SIZE: tl.constexpr,
    ROUND_FP32_TO_BF16: tl.constexpr,
    POWER_TWO_MAX_ROUND: tl.constexpr,
):
    pid_token = tl.program_id(axis=0)
    pid_hid = tl.program_id(axis=1)

    offset_row = pid_token * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset_col = pid_hid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    tensor_ptr = (
        tensor_ptr
        + offset_row[:, None] * tensor_stride_row
        + offset_col[None, :] * tensor_stride_col
    )
    tensor_mask = (offset_row[:, None] < tensor_size_row) & (
        offset_col[None, :] < tensor_size_col
    )
    tensor_block = tl.load(tensor_ptr, mask=tensor_mask, other=0.0)
    tensor_block = tensor_block.to(tl.float32)
    if ROUND_FP32_TO_BF16:
        # Match the existing BF16-source path without materializing a BF16 copy.
        tensor_block = tensor_block.to(
            tl.bfloat16, fp_downcast_rounding="rtne"
        ).to(tl.float32)

    scale = tl.maximum(tl.max(tl.abs(tensor_block)) / 448.0, 1e-30)
    if POWER_TWO_MAX_ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))

    scale_inv = 1.0 / scale
    tensor_block = tensor_block * scale_inv

    # save out
    out_ptr = (
        out_ptr
        + offset_row[:, None] * out_stride_row
        + offset_col[None, :] * out_stride_col
    )
    tl.store(out_ptr, tensor_block.to(out_ptr.dtype.element_ty), mask=tensor_mask)

    # save scale inv
    scale_ptr = scale_ptr + pid_token * scale_stride_row + pid_hid * scale_stride_col
    scale_mask = (pid_token < scale_size_row) & (pid_hid < scale_size_col)
    tl.store(scale_ptr, scale.to(scale_ptr.dtype.element_ty), mask=scale_mask)


def block_quantization_128x128_fn(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    global _BLOCK_SIZE

    weight_shape = weight.shape

    num_groups, int_dim, hid_size = weight.shape
    assert int_dim % 128 == 0 and hid_size % 128 == 0

    weight = weight.view(-1, hid_size)

    assert weight.is_contiguous()

    weight_quantized = torch.empty(
        (num_groups * int_dim, hid_size),
        device=weight.device,
        dtype=torch.float8_e4m3fn,
    )

    scale_shape = (num_groups * int_dim // _BLOCK_SIZE, hid_size // _BLOCK_SIZE)
    scale_inv = torch.empty(scale_shape, device=weight.device, dtype=torch.float32)

    _block_quantization_kernel[scale_shape](
        *tensor_to_kernel_args(weight, 2),
        *tensor_to_kernel_args(weight_quantized, 2),
        *tensor_to_kernel_args(scale_inv, 2),
        BLOCK_SIZE=_BLOCK_SIZE,
        ROUND_FP32_TO_BF16=False,
        POWER_TWO_MAX_ROUND=True,
        **params_to_kernel_kwargs(
            STATIC_TRITON_PARAMS, hid_size, num_warps=8, maxnreg=255
        ),
    )
    weight_quantized = weight_quantized.reshape(weight_shape)
    scale_inv = scale_inv.reshape(num_groups, -1, scale_shape[-1])
    return weight_quantized, scale_inv


def quantize_fp32_to_blockwise_fp8_(
    weight: torch.Tensor,
    data: torch.Tensor,
    scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize FP32 expert matrices into caller-owned blockwise FP8 buffers.

    The input is rounded to BF16 (round-to-nearest, ties-to-even) in registers,
    widened to FP32, and then processed by the same scale and FP8 math as
    :func:`block_quantization_128x128_fn`. This avoids a full-sized BF16
    intermediate and allocates no output storage. The returned tensors are the
    same objects supplied through ``data`` and ``scales``.

    ``weight`` must contain contiguous ``(E, F, D)`` matrices with positive,
    128-aligned ``F`` and ``D``. Each matrix is quantized independently, so
    callers may pass complete matrices or a contiguous rank-local selection of
    whole matrices. ``data`` must be a contiguous E4M3FN tensor with the same
    shape; ``scales`` must be contiguous FP32 with shape
    ``(E, F // 128, D // 128)``. The output buffers are filled and returned.
    """
    if not isinstance(weight, torch.Tensor):
        raise TypeError("weight must be a torch.Tensor")
    if not isinstance(data, torch.Tensor):
        raise TypeError("data must be a torch.Tensor")
    if not isinstance(scales, torch.Tensor):
        raise TypeError("scales must be a torch.Tensor")
    if weight.device.type != "cuda":
        raise ValueError("weight must be a CUDA tensor")
    if data.device != weight.device or scales.device != weight.device:
        raise ValueError("weight, data, and scales must be on the same device")
    if weight.dtype != torch.float32:
        raise ValueError("weight must have dtype torch.float32")
    if data.dtype != torch.float8_e4m3fn:
        raise ValueError("data must have dtype torch.float8_e4m3fn")
    if scales.dtype != torch.float32:
        raise ValueError("scales must have dtype torch.float32")
    if weight.ndim != 3:
        raise ValueError("weight must have shape (E, F, D)")
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous")
    if not data.is_contiguous():
        raise ValueError("data must be contiguous")
    if not scales.is_contiguous():
        raise ValueError("scales must be contiguous")

    num_groups, int_dim, hid_size = weight.shape
    if num_groups <= 0 or int_dim <= 0 or hid_size <= 0:
        raise ValueError("weight dimensions must be positive")
    if int_dim % _BLOCK_SIZE != 0 or hid_size % _BLOCK_SIZE != 0:
        raise ValueError("weight matrix dimensions must be multiples of 128")
    if data.shape != weight.shape:
        raise ValueError("data must have the same shape as weight")
    expected_scale_shape = (
        num_groups,
        int_dim // _BLOCK_SIZE,
        hid_size // _BLOCK_SIZE,
    )
    if scales.shape != expected_scale_shape:
        raise ValueError(f"scales must have shape {expected_scale_shape}")

    flat_weight = weight.view(-1, hid_size)
    flat_data = data.view(-1, hid_size)
    flat_scales = scales.view(-1, expected_scale_shape[-1])
    scale_grid = (flat_weight.shape[0] // _BLOCK_SIZE, hid_size // _BLOCK_SIZE)

    _block_quantization_kernel[scale_grid](
        *tensor_to_kernel_args(flat_weight, 2),
        *tensor_to_kernel_args(flat_data, 2),
        *tensor_to_kernel_args(flat_scales, 2),
        BLOCK_SIZE=_BLOCK_SIZE,
        ROUND_FP32_TO_BF16=True,
        POWER_TWO_MAX_ROUND=True,
        **params_to_kernel_kwargs(
            STATIC_TRITON_PARAMS, hid_size, num_warps=8, maxnreg=255
        ),
    )
    return data, scales
