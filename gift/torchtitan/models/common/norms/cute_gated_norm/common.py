"""Tile constants, gating math, compilation and scratch allocation for CuTe GatedNorm."""

import math

import cutlass
import cutlass.cute as cute
import torch
from cutlass.utils import LayoutEnum

from cutlass.cute.runtime import make_fake_stream

BLK_M = 64  # rows per tile (one wgmma M)
BLK_H = 64  # columns per sub-tile: 128 B of bf16 = one SW128 atom = one TMA box
WG = 128  # threads per warpgroup
RANK = 16  # gate rank r (one wgmma K step)
ROW = LayoutEnum.ROW_MAJOR
COL = LayoutEnum.COL_MAJOR
BF16 = cutlass.BFloat16
F32 = cutlass.Float32


# =============================================================================
# device helpers
# =============================================================================
def sigmoid(z):
    return cute.arch.rcp_approx(1.0 + cute.math.exp(-z, fastmath=True))


# =============================================================================
# host
# =============================================================================
COMPILED = {}
PLANS = {}
RESERVED_SMS = 16


def resolve_num_sms(device, requested=None):
    """Leave SMs available for concurrent collectives, as in the original kernel."""
    available = torch.cuda.get_device_properties(device).multi_processor_count
    budget = max(1, available - RESERVED_SMS) if requested is None else requested
    if not 1 <= budget <= available:
        raise ValueError(f"num_sms must be in [1, {available}], got {budget}")
    return budget


def to_cute_dtype(dtype):
    return {torch.bfloat16: BF16, torch.float32: F32, torch.float16: cutlass.Float16}[
        dtype
    ]


def compile_op(key, make_op, args):
    """Compile symbolic shapes without example allocations; use PyTorch's stream."""
    device = torch.cuda.current_device()
    key = (device, *key)
    if key not in COMPILED:
        COMPILED[key] = cute.compile(
            make_op(),
            *args,
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    return COMPILED[key]


def _carve(shapes, align=64):
    """Pack `shapes` back to back in one fp32 buffer, each at a multiple of `align` elements
    (64 fp32 = 256 B, enough for every kernel argument). Returns (total, offsets, sizes)."""
    offs, sizes, total = [], [], 0
    for sh in shapes:
        n = math.prod(sh)
        offs.append(total)
        sizes.append(n)
        total += -(-n // align) * align
    return total, offs, sizes


def workspace(device, shapes):
    """Per-call scratch: the allocator owns its lifetime, including graph capture.

    Global stream-keyed scratch can be shared by independently captured graphs
    replayed on different streams. Allocating here avoids that hidden aliasing.
    """
    total, offs, sizes = _carve(shapes)
    buf = torch.empty(total, device=device, dtype=torch.float32)
    return [buf[o : o + n].view(sh) for o, n, sh in zip(offs, sizes, shapes)]
