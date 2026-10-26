"""Shared pieces of the CuTe GatedNorm kernels (Hopper / SM90).

Tile constants, device-side helpers (ldmatrix / stmatrix copies in the wgmma accumulator
layout, TMA atoms, the TMA mbarrier ring, the trivial tiled MMA) and the host-side compile /
plan / workspace caches that forward.py and backward.py share.
"""

import math
import os

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass.utils import LayoutEnum

try:
    from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
    FAKE_TENSOR_API = hasattr(cute, "sym_int")
except ImportError:
    FAKE_TENSOR_API = False
if not FAKE_TENSOR_API:
    import cuda.bindings.driver as cuda
    from cutlass.cute.runtime import from_dlpack

LOG2E = 1.4426950408889634
BLK_M = 64          # rows per tile (one wgmma M)
BLK_H = 64          # columns per sub-tile: 128 B of bf16 = one SW128 atom = one TMA box
WG = 128            # threads per warpgroup
RANK = 16           # gate rank r (one wgmma K step)
ROW = LayoutEnum.ROW_MAJOR
COL = LayoutEnum.COL_MAJOR
BF16 = cutlass.BFloat16
F32 = cutlass.Float32


# =============================================================================
# device helpers
# =============================================================================
def _operand_smem_layout(layout, shape_mnk, dtype, stages, operand):
    """Staged smem layout of a wgmma operand (A: M x K, B: N x K) built from the swizzle atom,
    as hopper_helpers.make_smem_layout_a/_b do in newer DSL releases."""
    k_major = layout.sm90_mma_major_mode() == cute.nvgpu.warpgroup.OperandMajorMode.K
    if operand == "a":
        shape, mn_size = cute.slice_(shape_mnk, (None, 0, None)), shape_mnk[0]
    else:
        shape, mn_size = cute.slice_(shape_mnk, (0, None, None)), shape_mnk[1]
    major_size = shape_mnk[2] if k_major else mn_size
    atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(layout, dtype, major_size), dtype)
    return cute.tile_to_shape(atom, cute.append(shape, stages),
                              order=(0, 1, 2) if k_major else (1, 0, 2))


def make_smem_layout_a(layout, shape_mnk, dtype, stages):
    if hasattr(sm90_utils, "make_smem_layout_a"):
        return sm90_utils.make_smem_layout_a(layout, shape_mnk, dtype, stages)
    return _operand_smem_layout(layout, shape_mnk, dtype, stages, "a")


def make_smem_layout_b(layout, shape_mnk, dtype, stages):
    if hasattr(sm90_utils, "make_smem_layout_b"):
        return sm90_utils.make_smem_layout_b(layout, shape_mnk, dtype, stages)
    return _operand_smem_layout(layout, shape_mnk, dtype, stages, "b")


def make_smem_layout_epi(dtype, layout, tile_mn, stages):
    """Staged smem layout of an epilogue / plain 2-D tile (row-major = N-major)."""
    if hasattr(sm90_utils, "make_smem_layout_epi"):
        return sm90_utils.make_smem_layout_epi(dtype, layout, tile_mn, stages)
    n_major = layout == LayoutEnum.ROW_MAJOR
    atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(layout, dtype, tile_mn[1] if n_major else tile_mn[0]), dtype)
    return cute.tile_to_shape(atom, cute.append(tile_mn, stages),
                              order=(1, 0, 2) if n_major else (0, 1, 2))


def sigmoid(z):
    return cute.arch.rcp_approx(1.0 + cute.math.exp2(z * (-LOG2E), fastmath=True))


def make_s2r_acc(tiled_mma, thr_idx, s_staged):
    """ldmatrix copy of a staged bf16 smem tile straight into the wgmma accumulator layout.

    Returns (copy, smem partition with a trailing stage mode, bf16 register fragment shaped
    like the accumulator, its copy view); r[i] pairs with acc[i].
    """
    atom = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), BF16)
    s2r = cute.make_tiled_copy_C(atom, tiled_mma)
    thr = s2r.get_slice(thr_idx)
    r = cute.make_rmem_tensor(tiled_mma.get_slice(thr_idx).partition_shape_C((BLK_M, BLK_H)), BF16)
    return s2r, thr.partition_S(s_staged), r, thr.retile(r)


def make_r2s_acc(tiled_mma, thr_idx, s_staged, acc):
    """stmatrix epilogue copy: fp32 accumulator -> bf16 staged smem tile (row-major, SW128).

    Returns (copy, smem partition with a trailing stage mode, accumulator view, fp32 and
    bf16 register staging tensors).
    """
    r2s_atom = sm90_utils.sm90_get_smem_store_op(ROW, elem_ty_d=BF16, elem_ty_acc=F32)
    c_atom = cute.make_copy_atom(cute.nvgpu.warp.StMatrix8x8x16bOp(False, 4), BF16)
    r2s = cute.make_tiled_copy_S(r2s_atom, cute.make_tiled_copy_C_atom(c_atom, tiled_mma))
    thr = r2s.get_slice(thr_idx)
    shape = cute.make_layout(cute.shape(thr.partition_S(s_staged))[:3]).shape
    return (r2s, thr.partition_D(s_staged), r2s.retile(acc), cute.make_rmem_tensor(shape, F32),
            cute.make_rmem_tensor(shape, BF16))


def tma_load_atom(tensor, layout, tile):
    return cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(), tensor, cute.slice_(layout, (None, None, 0)), tile)


def tma_store_atom(tensor, layout, tile):
    return cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), tensor, cute.slice_(layout, (None, None, 0)), tile)


def tma_part(atom, s, g):
    return cute.nvgpu.cpasync.tma_partition(atom, 0, cute.make_layout(1), cute.group_modes(s, 0, 2),
                                            cute.group_modes(g, 0, 2))


def make_mma(a_major, b_major, tiler_mn):
    return sm90_utils.make_trivial_tiled_mma(BF16, BF16, a_major.sm90_mma_major_mode(),
                                             b_major.sm90_mma_major_mode(), F32, (1, 1, 1),
                                             tiler_mn=tiler_mn)


def make_ring(storage_mbar, stages, tx_bytes):
    """TMA producer / consumer mbarrier ring; one consumer warpgroup (4 warps) frees a stage."""
    return pipeline.PipelineTmaAsync.create(
        barrier_storage=storage_mbar.data_ptr(), num_stages=stages,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 4),
        tx_count=tx_bytes, cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)))


# =============================================================================
# host
# =============================================================================
COMPILED = {}
PLANS = {}
_WS = {}
_NUM_SMS = {}
RESERVED_SMS = 16
_SM_BUDGET = None


def set_sm_budget(n):
    """Override the persistent-kernel CTA budget; ``None`` restores the default.

    A GatedNorm CTA occupies a whole SM. By default ``RESERVED_SMS`` SMs are left free for
    collectives running on another stream. The override is process-global and should be set
    before kernel compilation or CUDA graph capture.
    """
    global _SM_BUDGET
    if n is not None:
        n = int(n)
        if n < 1:
            raise ValueError(f"sm budget must be >= 1, got {n}")
    _SM_BUDGET = n


def get_sm_budget():
    """Return the process-wide SM-budget override, or ``None`` for the default policy."""
    return _SM_BUDGET


def num_sms(device):
    idx = torch.device(device).index or 0
    if idx not in _NUM_SMS:
        _NUM_SMS[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _NUM_SMS[idx]


def resolve_num_sms(device, requested=None):
    """Resolve and validate the CTA budget for ``device``."""
    available = num_sms(device)
    budget = requested
    if budget is None:
        budget = _SM_BUDGET
    if budget is None:
        budget = max(1, available - RESERVED_SMS)
    budget = int(budget)
    if budget < 1 or budget > available:
        raise ValueError(
            f"num_sms must be in [1, {available}] for {torch.device(device)}, got {budget}"
        )
    return budget


if os.environ.get("CUTE_GN_SM_BUDGET"):
    set_sm_budget(os.environ["CUTE_GN_SM_BUDGET"])


class _SymM:
    """The symbolic token count on the older DSL path (a marker in `fake` shapes)."""

    def __repr__(self):
        return "M"


_SYM_M = _SymM()
_M_EXAMPLE = 128            # token count of the example tensors on the older DSL path


def sym_m():
    """Symbolic token count to put in `fake` shapes."""
    return cute.sym_int() if FAKE_TENSOR_API else _SYM_M


def fake(dtype, shape, align=16):
    """Compile-time tensor description: row-major, any symbolic dims stay dynamic."""
    if FAKE_TENSOR_API:
        return make_fake_compact_tensor(dtype, shape,
                                        stride_order=tuple(range(len(shape) - 1, -1, -1)),
                                        assumed_align=align)
    return (dtype, tuple(shape), align)


_TORCH_DTYPE = {BF16: torch.bfloat16, F32: torch.float32, cutlass.Float16: torch.float16}


def to_cute_dtype(dt):
    return {torch.bfloat16: BF16, torch.float32: F32, torch.float16: cutlass.Float16}[dt]


def _as_cute(t, dyn_modes, align):
    """torch tensor -> cute tensor with the token-count modes dynamic (older DSL path)."""
    ct = from_dlpack(t, assumed_align=align)
    for mode in dyn_modes:
        ct = ct.mark_compact_shape_dynamic(mode=mode)
    return ct


def _current_stream(t):
    return cuda.CUstream(torch.cuda.current_stream(t.device).cuda_stream)


def compile_op(key, make_op, args):
    """cute.compile once per key; the result is called with torch tensors on the current stream."""
    fn = COMPILED.get(key)
    if fn is not None:
        return fn
    if FAKE_TENSOR_API:
        try:
            import tvm_ffi  # noqa: F401
        except ImportError as exc:
            raise ImportError("gated_norm needs apache-tvm-ffi next to nvidia-cutlass-dsl") from exc
        fn = cute.compile(make_op(), *args, make_fake_stream(use_tvm_ffi_env_stream=True),
                          options="--enable-tvm-ffi")
    else:
        meta = [(tuple(i for i, d in enumerate(sh) if d is _SYM_M), al) for _, sh, al in args]
        examples = [torch.empty([_M_EXAMPLE if d is _SYM_M else d for d in sh],
                                dtype=_TORCH_DTYPE[dt], device="cuda") for dt, sh, _ in args]
        compiled = cute.compile(make_op(), *[_as_cute(t, dm, al) for t, (dm, al) in zip(examples, meta)],
                                _current_stream(examples[0]))

        def fn(*tensors):
            return compiled(*[_as_cute(t, dm, al) for t, (dm, al) in zip(tensors, meta)],
                            _current_stream(tensors[0]))
    COMPILED[key] = fn
    return fn


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


def workspace(key, device, shapes):
    """fp32 scratch that lives only inside one call, cached per (key, device, stream)."""
    total, offs, sizes = _carve(shapes)
    k = (key, device.index, torch.cuda.current_stream(device).cuda_stream)
    buf = _WS.get(k)
    if buf is None or buf.numel() < total:
        buf = torch.empty(total, device=device, dtype=torch.float32)
        _WS[k] = buf
    return [buf[o:o + n].view(sh) for o, n, sh in zip(offs, sizes, shapes)]
