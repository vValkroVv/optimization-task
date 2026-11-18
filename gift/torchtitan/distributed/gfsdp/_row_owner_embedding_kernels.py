"""Narrow CUDA kernels for the opt-in row-owner embedding path.

This module is imported only when a selected optimization reaches a supported
CUDA tensor. Keep these kernels small and specific to the row-owner contracts.
"""

import triton
import triton.language as tl


@triton.jit
def _validate_clamp_ids_kernel(
    ids_ptr,
    safe_ids_ptr,
    valid_ptr,
    num_ids,
    num_embeddings,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK).to(tl.int64)
    mask = offsets < num_ids
    ids = tl.load(ids_ptr + offsets, mask, other=0).to(tl.int64)
    valid_ids = (ids >= 0) & (ids < num_embeddings)
    invalid_count = tl.sum((mask & ~valid_ids).to(tl.int64), 0)
    tl.store(valid_ptr, (invalid_count == 0).to(tl.int64))
    safe_ids = tl.minimum(tl.maximum(ids, 0), num_embeddings - 1)
    tl.store(safe_ids_ptr + offsets, safe_ids, mask)


def validate_and_clamp_ids(ids, safe_ids, valid, num_embeddings: int) -> None:
    """Validate signed int64 IDs and clamp them before routing arithmetic."""
    num_ids = ids.numel()
    block = triton.next_power_of_2(max(1, num_ids))
    _validate_clamp_ids_kernel[(1,)](
        ids,
        safe_ids,
        valid,
        num_ids,
        num_embeddings,
        block,
        num_warps=4 if block <= 2048 else 8,
    )


@triton.jit
def _routing_metadata_kernel(
    unique_ptr,
    valid_ptr,
    metadata_ptr,
    num_unique,
    rows_per_owner: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK).to(tl.int64)
    mask = offsets < num_unique
    unique_ids = tl.load(unique_ptr + offsets, mask, other=0).to(tl.int64)
    for owner in tl.static_range(0, world_size):
        lower = owner * rows_per_owner
        upper = (owner + 1) * rows_per_owner
        count = tl.sum(
            (mask & (unique_ids >= lower) & (unique_ids < upper)).to(tl.int64), 0
        )
        tl.store(metadata_ptr + owner, count)
    tl.store(metadata_ptr + world_size, tl.load(valid_ptr).to(tl.int64))


def build_routing_metadata(
    unique, valid, metadata, rows_per_owner: int, world_size: int
) -> None:
    """Count each contiguous owner interval without intermediate tensors."""
    num_unique = unique.numel()
    block = triton.next_power_of_2(max(1, num_unique))
    _routing_metadata_kernel[(1,)](
        unique,
        valid,
        metadata,
        num_unique,
        rows_per_owner,
        world_size,
        block,
        num_warps=4 if block <= 2048 else 8,
    )


@triton.jit
def _owner_lookup_bf16_kernel(
    weight_ptr,
    local_rows_ptr,
    output_ptr,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    local_row = tl.load(local_rows_ptr + row).to(tl.int64)
    values = tl.load(weight_ptr + local_row * width + columns, columns < width, other=0)
    output_offset = row * width + columns
    tl.store(output_ptr + output_offset, values.to(tl.bfloat16), columns < width)


def owner_lookup_bf16(weight, local_rows, output) -> None:
    """Gather FP32 owner rows and round once into the BF16 send buffer."""
    num_rows, width = output.shape
    if not num_rows:
        return
    _owner_lookup_bf16_kernel[(num_rows, triton.cdiv(width, 1024))](
        weight,
        local_rows,
        output,
        width,
        1024,
        num_warps=4,
    )


@triton.jit
def _owner_add_bf16_kernel(
    gradient_ptr,
    local_rows_ptr,
    values_ptr,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    local_row = tl.load(local_rows_ptr + row).to(tl.int64)
    values = tl.load(values_ptr + row * width + columns, columns < width, other=0)
    destination = gradient_ptr + local_row * width + columns
    previous = tl.load(destination, columns < width, other=0.0)
    tl.store(destination, previous + values.to(tl.float32), columns < width)


def owner_add_bf16_(gradient, local_rows, values) -> None:
    """Add one source's unique BF16 rows to an FP32 gradient in place.

    Rows within the source must be unique; the caller launches each source on
    one stream in rank order, so each destination has one writer per launch.
    """
    num_rows, width = values.shape
    if not num_rows:
        return
    _owner_add_bf16_kernel[(num_rows, triton.cdiv(width, 1024))](
        gradient,
        local_rows,
        values,
        width,
        1024,
        num_warps=4,
    )
