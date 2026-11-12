# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared helpers for the float8 triton kernels."""

import torch
import triton
import triton.language as tl
from torch.fx.experimental.symbolic_shapes import guard_or_false


def maybe_contiguous(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is not None and not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor


def maybe_convert_to_float8_e4m3fn(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.uint8:
        tensor = tensor.view(torch.float8_e4m3fn)
    assert tensor.dtype == torch.float8_e4m3fn
    return tensor


def tensor_to_kernel_args(tensor: torch.Tensor | None, dims: int):
    return (
        [tensor, *tensor.stride(), *tensor.size()]
        if tensor is not None
        else [None] * (1 + 2 * dims)
    )


@triton.jit
def _build_m_indices_kernel(
    # group_sizes
    group_sizes_ptr,
    group_sizes_stride_row,
    group_sizes_size_row,
    # output
    output_ptr,
    output_stride_row,
    output_size_row,
    # constants
    NUM_EXPERTS: tl.constexpr,
    ALIGN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_block = tl.program_id(0).to(tl.int32)
    output_offset = pid_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int32)

    # Start with the fixed-capacity tail marked invalid, then overwrite rows
    # belonging to an expert. Keeping the prefix sum on the GPU is required by
    # HybridEP's non-blocking path: reading group sizes on the host would undo
    # the D2H-sync avoidance that fixed-capacity dispatch provides.
    output_values = tl.full((BLOCK_SIZE,), -1, tl.int32)
    group_start = tl.full((), 0, tl.int32)
    for expert_num in tl.range(0, NUM_EXPERTS):
        group_size = tl.load(group_sizes_ptr + expert_num * group_sizes_stride_row).to(
            tl.int32
        )
        group_size_padded = ((group_size + ALIGN_SIZE - 1) // ALIGN_SIZE) * ALIGN_SIZE
        in_group = (output_offset >= group_start) & (
            output_offset < group_start + group_size
        )
        output_values = tl.where(in_group, expert_num, output_values)
        group_start += group_size_padded

    out_ptrs = output_ptr + output_offset * output_stride_row
    tl.store(out_ptrs, output_values, mask=output_offset < output_size_row)


def build_m_indices(
    group_sizes_tensor: torch.Tensor,
    *,
    num_rows: int,
    block_size: int = 128,
) -> torch.Tensor:
    m_indices = torch.empty(
        (num_rows,), device=group_sizes_tensor.device, dtype=torch.int32
    )

    if guard_or_false(num_rows == 0):
        return m_indices

    grid = (triton.cdiv(num_rows, block_size),)

    _build_m_indices_kernel[grid](
        *tensor_to_kernel_args(group_sizes_tensor, dims=1),
        *tensor_to_kernel_args(m_indices, dims=1),
        BLOCK_SIZE=block_size,
        NUM_EXPERTS=group_sizes_tensor.numel(),
        ALIGN_SIZE=128,
    )
    return m_indices
