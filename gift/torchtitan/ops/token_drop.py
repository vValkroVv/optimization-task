"""Fused capacity masks without host reads or a global sort.

Selection programs own individual experts; copy programs handle other routes.
Only overloaded experts run radix selection. The stable tie scan
uses original token/slot order, independently of program scheduling.
"""

from collections.abc import Callable
from typing import cast

import torch
import triton
import triton.language as tl


# Bound the per-expert register working set. Larger inputs use the existing
# PyTorch implementation until a tiled selector has been measured on hardware.
MAX_RADIX_ROUTES = 65536


@triton.jit
def _capacity_mask_kernel(
    weights_ptr,
    buckets_ptr,
    valid_ptr,
    counts_ptr,
    needs_drop_ptr,
    keep_ptr,
    NUM_ROUTES: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
    COPY_BLOCK: tl.constexpr,
    RADIX_PASSES: tl.constexpr,
):
    program = tl.program_id(0)
    needs_drop = tl.load(needs_drop_ptr)
    if program < NUM_EXPERTS:
        if needs_drop and CAPACITY > 0:
            count = tl.load(counts_ptr + program).to(tl.int32)
            if count > CAPACITY:
                offsets = tl.arange(0, BLOCK)
                buckets = tl.load(buckets_ptr + offsets, offsets < NUM_ROUTES, other=-1)
                owned = (offsets < NUM_ROUTES) & (buckets == program)
                scores = tl.load(weights_ptr + offsets, owned, other=0).to(tl.float32)
                bits = scores.to(tl.uint32, bitcast=True)
                keys = tl.where(
                    (bits & 0x80000000) != 0, bits ^ 0xFFFFFFFF, bits ^ 0x80000000
                )
                active = owned
                # Locate the capacity-th largest score via its ascending rank.
                rank = count - CAPACITY + 1
                bins = tl.arange(0, 256)
                for digit_index in tl.static_range(RADIX_PASSES):
                    shift = 24 - 8 * digit_index
                    digits = ((keys >> shift) & 255).to(tl.int32)
                    histogram = tl.histogram(digits, 256, mask=active)
                    cumulative = tl.cumsum(histogram, 0)
                    selected = tl.min(tl.where(cumulative >= rank, bins, 256), 0)
                    rank -= tl.sum(tl.where(bins < selected, histogram, 0), 0)
                    active = active & (digits == selected)
                # Promoted fp16/bf16 keys have constant low bits inside a prefix,
                # so they need fewer radix passes without reducing precision.
                threshold = tl.max(tl.where(active, keys, 0), 0)
                above = owned & (keys > threshold)
                tied = owned & (keys == threshold)
                remaining = CAPACITY - tl.sum(tl.where(above, 1, 0), 0)
                tie_rank = tl.cumsum(tl.where(tied, 1, 0), 0)
                keep = above | (tied & (tie_rank <= remaining))
                tl.store(keep_ptr + offsets, keep, owned)
    else:
        copy_offsets = (program - NUM_EXPERTS) * COPY_BLOCK + tl.arange(0, COPY_BLOCK)
        in_bounds = copy_offsets < NUM_ROUTES
        copy_valid = tl.load(valid_ptr + copy_offsets, in_bounds, other=False)
        if needs_drop and CAPACITY > 0:
            copy_buckets = tl.load(
                buckets_ptr + copy_offsets, in_bounds, other=NUM_EXPERTS
            )
            copy_counts = tl.load(counts_ptr + copy_buckets)
            # Selection programs own overloaded routes. Disjoint writes avoid
            # races without atomics or a separate output-initialization kernel.
            in_bounds = in_bounds & (~copy_valid | (copy_counts <= CAPACITY))
        copy_keep = copy_valid & (~needs_drop | (CAPACITY > 0))
        tl.store(keep_ptr + copy_offsets, copy_keep, in_bounds)


def capacity_mask(
    weights: torch.Tensor,
    buckets: torch.Tensor,
    valid: torch.Tensor,
    counts: torch.Tensor,
    needs_drop: torch.Tensor,
    capacity: int,
) -> torch.Tensor:
    """Select stable per-expert top scores and apply the device overload gate.

    Inputs are flattened, contiguous route arrays. ``counts`` includes the last
    inactive bucket. The caller handles unsupported devices/dtypes/input sizes.
    """
    num_routes = weights.numel()
    if num_routes == 0 or capacity >= num_routes:
        return valid
    if num_routes > MAX_RADIX_ROUTES:
        raise ValueError(
            f"Triton capacity selection supports at most {MAX_RADIX_ROUTES} routes"
        )
    if weights.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Triton capacity selection requires fp16, bf16 or fp32 scores")
    num_experts = counts.numel() - 1
    keep = torch.empty_like(valid)
    block = triton.next_power_of_2(num_routes)
    copy_block = 1024
    radix_passes = {torch.bfloat16: 2, torch.float16: 3, torch.float32: 4}[
        weights.dtype
    ]
    # Triton launchers accept host values for constexprs and extra launch options.
    launch = cast(
        Callable[..., object],
        _capacity_mask_kernel[(num_experts + triton.cdiv(num_routes, copy_block),)],
    )
    launch(
        weights.contiguous(),
        buckets,
        valid,
        counts,
        needs_drop,
        keep,
        num_routes,
        num_experts,
        capacity,
        block,
        copy_block,
        radix_passes,
        num_warps=4 if block <= 4096 else (8 if block <= 16384 else 16),
    )
    return keep
