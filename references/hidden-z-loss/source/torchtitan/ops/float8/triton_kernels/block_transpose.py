# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tiled batched transpose of the last two dims: (G, R, C) -> (G, C, R)."""

import torch
import triton
import triton.language as tl

from torchtitan.ops.float8.triton_kernels.utils import tensor_to_kernel_args


@triton.jit
def _transpose_last_two_kernel(
    # inp (num_groups, rows, cols)
    inp_ptr,
    inp_stride_g,
    inp_stride_row,
    inp_stride_col,
    inp_size_g,
    inp_size_row,
    inp_size_col,
    # out (num_groups, cols, rows)
    out_ptr,
    out_stride_g,
    out_stride_row,
    out_stride_col,
    out_size_g,
    out_size_row,
    out_size_col,
    # constexpr
    BLOCK_ROW: tl.constexpr,
    BLOCK_COL: tl.constexpr,
):
    pid_group = tl.program_id(axis=0)
    pid_row = tl.program_id(axis=1)
    pid_col = tl.program_id(axis=2)

    offset_row = pid_row * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    offset_col = pid_col * BLOCK_COL + tl.arange(0, BLOCK_COL)

    inp_ptrs = (
        inp_ptr
        + pid_group * inp_stride_g
        + offset_row[:, None] * inp_stride_row
        + offset_col[None, :] * inp_stride_col
    )
    inp_mask = (offset_row[:, None] < inp_size_row) & (
        offset_col[None, :] < inp_size_col
    )
    block = tl.load(inp_ptrs, mask=inp_mask, other=0)
    block = tl.trans(block)

    out_ptrs = (
        out_ptr
        + pid_group * out_stride_g
        + offset_col[:, None] * out_stride_row
        + offset_row[None, :] * out_stride_col
    )
    out_mask = (offset_col[:, None] < out_size_row) & (
        offset_row[None, :] < out_size_col
    )
    tl.store(out_ptrs, block, mask=out_mask)


def transpose_last_two_dims_fn(tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.dim() == 3, tensor.shape
    fp8_dtype = None
    if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        fp8_dtype = tensor.dtype
        tensor = tensor.view(torch.uint8)

    num_groups, rows, cols = tensor.shape
    out = torch.empty(
        (num_groups, cols, rows), dtype=tensor.dtype, device=tensor.device
    )

    max_block = 128 if tensor.element_size() == 1 else 64
    block_row = min(max_block, triton.next_power_of_2(rows))
    block_col = min(max_block, triton.next_power_of_2(cols))
    grid = (num_groups, triton.cdiv(rows, block_row), triton.cdiv(cols, block_col))
    _transpose_last_two_kernel[grid](
        *tensor_to_kernel_args(tensor, 3),
        *tensor_to_kernel_args(out, 3),
        BLOCK_ROW=block_row,
        BLOCK_COL=block_col,
        num_warps=8 if block_row * block_col * tensor.element_size() >= 8192 else 4,
    )
    return out.view(fp8_dtype) if fp8_dtype is not None else out
