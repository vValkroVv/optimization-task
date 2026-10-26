import typing as tp

import torch
import triton
import triton.language as tl


def maybe_contiguous(tensor: tp.Optional[torch.Tensor]) -> tp.Optional[torch.Tensor]:
    if tensor is not None and not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor


def maybe_convert_to_float8_e4m3fn(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.uint8:
        tensor = tensor.view(torch.float8_e4m3fn)
    assert tensor.dtype == torch.float8_e4m3fn
    return tensor


def tensor_to_kernel_args(tensor: tp.Optional[torch.Tensor], dims: int):
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
    pid_expert = tl.program_id(0).to(tl.int32)
    pid_block  = tl.program_id(1).to(tl.int32)

    start = tl.full((), 0, tl.int32)
    previously_seen = tl.full((), 0, tl.int32)

    for expert_num in tl.range(0, NUM_EXPERTS):
        expert_size = tl.load(group_sizes_ptr + expert_num * group_sizes_stride_row).to(tl.int32)
        expert_size_padded = ((expert_size + ALIGN_SIZE - 1) // ALIGN_SIZE) * ALIGN_SIZE

        start += tl.where((previously_seen == 0) & (expert_num < pid_expert), expert_size_padded, 0)
        previously_seen = tl.where(expert_num == pid_expert, 1, previously_seen)

    group_size = tl.load(group_sizes_ptr + pid_expert * group_sizes_stride_row).to(tl.int32)
    group_size_padded = ((group_size + ALIGN_SIZE - 1) // ALIGN_SIZE) * ALIGN_SIZE

    output_offset = pid_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int32)
    output_mask = output_offset < group_size_padded

    output_values = tl.where(output_offset < group_size, pid_expert, -1).to(tl.int32)

    out_ptrs = output_ptr + (start + output_offset) * output_stride_row
    tl.store(out_ptrs, output_values, mask=output_mask)


def build_m_indices(
    group_sizes_list: tp.List[int],
    group_sizes_tensor: torch.Tensor,
    BLOCK_SIZE: int = 128,
):
    group_sizes_padded = [(m + 127) // 128 * 128 for m in group_sizes_list]
    total = sum(group_sizes_padded)
    m_indices = torch.empty((total,), device=group_sizes_tensor.device, dtype=torch.int32)
    if total == 0:
        return m_indices

    grid = (len(group_sizes_list), triton.cdiv(max(group_sizes_padded), BLOCK_SIZE))

    _build_m_indices_kernel[grid](
        *tensor_to_kernel_args(group_sizes_tensor, dims=1),
        *tensor_to_kernel_args(m_indices, dims=1),
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_EXPERTS=len(group_sizes_list),
        ALIGN_SIZE=128,)
    return m_indices


