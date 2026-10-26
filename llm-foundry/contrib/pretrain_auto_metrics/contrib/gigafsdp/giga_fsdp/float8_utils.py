import typing as tp
from typing import NamedTuple, Sequence

import torch
import triton
import triton.language as tl

FP8_TILE_SIZE = 128


_EXPERT_MODULE_NAME = "experts"
_EXPERT_PARAM_NAMES = ("weight1", "weight2")


def is_expert_param(fqn: str) -> bool:
    """True only for ``<...>.experts.weight1`` / ``<...>.experts.weight2``.

    Matching is anchored on whole FQN segments, NOT a substring: a substring test
    also accepts ``shared_experts.weight1``, which is a DENSE module whose forward
    never reads the ``_<name>_quantized`` handoff.  Enabling fp8 all-gather on it
    would silently hand its GEMM the bf16 NaN carrier.
    """
    parts = fqn.split(".")
    if len(parts) < 2:
        return False
    return parts[-2] == _EXPERT_MODULE_NAME and parts[-1] in _EXPERT_PARAM_NAMES


class Fp8ParamGeom(NamedTuple):
    """Per-parameter geometry driving the fused fold+quantize kernel.
    """
    fqn: str
    elem_offset: int   # param start in the SuperTensor's flat element array
    tile_offset: int   # elem_offset // FP8_TILE_SIZE
    n_experts: int     # 1 for a plain 2-D [N, K] param, E for a 3-D [E, N, K] param
    n_row_elem: int    # N (non-contraction dim), % FP8_TILE_SIZE == 0
    n_cols: int        # K (contraction dim),     % FP8_TILE_SIZE == 0


def fp8_param_shape_is_supported(shape: Sequence[int]) -> bool:
    """2-D or 3-D, with both TRAILING dims divisible by FP8_TILE_SIZE.
    """
    if len(shape) not in (2, 3):
        return False
    return shape[-2] % FP8_TILE_SIZE == 0 and shape[-1] % FP8_TILE_SIZE == 0


def _tensor_to_kernel_args(tensor: tp.Optional[torch.Tensor], dims: int) -> list:
    return ([tensor, *tensor.stride(), *tensor.size()]
            if tensor is not None
            else [None] * (1 + 2 * dims))


@triton.jit
def _compute_local_tile_amaxes_triton_kernel(
    shard_ptr,
    shard_stride_row, shard_stride_col,
    shard_size_row, shard_size_col,
    out_ptr,
    out_stride_row,
    out_size_row,
    BLOCK_M: tl.constexpr,
    TILE_SIZE: tl.constexpr,
) -> None:
    pid = tl.program_id(axis=0)
    row_offs = pid.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    col_offs = tl.arange(0, TILE_SIZE).to(tl.int64)
    in_ptr = shard_ptr + row_offs[:, None] * shard_stride_row + col_offs[None, :] * shard_stride_col
    mask = (row_offs[:, None] < shard_size_row) & (col_offs[None, :] < shard_size_col)
    block = tl.load(in_ptr, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(block), axis=1)
    out_p = out_ptr + row_offs * out_stride_row
    tl.store(out_p, amax, mask=row_offs < out_size_row)


def compute_local_tile_amaxes(shard: torch.Tensor, out: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
    """Per-rank 1x128 (flat) tile amaxes of a contiguous shard.
    """
    assert shard.dim() == 1
    assert shard.numel() % FP8_TILE_SIZE == 0
    assert shard.is_contiguous()
    n_tiles = shard.numel() // FP8_TILE_SIZE
    shard_2d = shard.view(n_tiles, FP8_TILE_SIZE)
    if out is None:
        out = torch.empty(n_tiles, device=shard.device, dtype=torch.float32)
    BLOCK_M = 32
    grid = (triton.cdiv(n_tiles, BLOCK_M),)
    _compute_local_tile_amaxes_triton_kernel[grid](
        *_tensor_to_kernel_args(shard_2d, 2),
        *_tensor_to_kernel_args(out, 1),
        BLOCK_M=BLOCK_M,
        TILE_SIZE=FP8_TILE_SIZE,
        num_warps=4,)
    return out


@triton.jit
def _fused_fold_quantize_kernel(
    gathered_ptr,
    shard_ptr,
    out_fp8_ptr,
    out_scale_ptr,
    scale_block_offset,
    packed_scale_stride,
    local_scale_blocks,
    tile_global_offset,
    n_row_elem,
    n_col_tiles,
    n_block_rows,
    elem_global_offset,
    global_start,
    shard_size,
    fp8_max,
    TILE: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    """Fused: compute block scale from gathered_scales + quantize shard rows in one pass.

    Grid: (E * n_block_rows * n_col_tiles,) — one program per 128x128 block.
    Phase 1: reads TILE amaxes from gathered_scales (same strided gather as fold kernel),
             reduces to block_amax, computes scale, stores to out_scale_ptr.
    Phase 2: loops over TILE/ROW_BLOCK groups of ROW_BLOCK rows, loads ROW_BLOCK*TILE
             source elements from shard, quantizes using scale_inv, stores FP8.
             Rows outside the shard are masked (no-op loads/stores).

    For a plain 2-D matrix E == 1, so ``n_row_elem == n_rows`` and the ``e``
    dimension collapses.
    """
    pid = tl.program_id(0)
    bc = pid % n_col_tiles
    tmp = pid // n_col_tiles
    br = tmp % n_block_rows
    e = tmp // n_block_rows

    # --- Phase 1: block scale from gathered_scales ---
    base_flat = (tl.cast(tile_global_offset, tl.int64)
                 + tl.cast(e, tl.int64) * tl.cast(n_row_elem, tl.int64) * tl.cast(n_col_tiles, tl.int64)
                 + tl.cast(br, tl.int64) * TILE * tl.cast(n_col_tiles, tl.int64)
                 + tl.cast(bc, tl.int64))
    r_arange = tl.arange(0, TILE).to(tl.int64)
    flat_tiles = base_flat + r_arange * tl.cast(n_col_tiles, tl.int64)
    n_local = tl.cast(local_scale_blocks, tl.int64)
    ranks = flat_tiles // n_local
    local_blks = flat_tiles - ranks * n_local
    gathered_idxs = (ranks * tl.cast(packed_scale_stride, tl.int64)
                     + tl.cast(scale_block_offset, tl.int64) + local_blks)
    amaxes = tl.load(gathered_ptr + gathered_idxs).to(tl.float32)
    block_amax = tl.max(amaxes)
    block_amax_safe = tl.maximum(block_amax, 1e-12)
    scale = tl.exp2(tl.ceil(tl.log2(block_amax_safe / fp8_max)))
    scale_inv = 1.0 / scale
    tl.store(out_scale_ptr + pid, scale)

    # --- Phase 2: quantize shard rows belonging to this block ---
    H = tl.cast(n_col_tiles * TILE, tl.int64)
    col_offs = tl.arange(0, TILE).to(tl.int64)
    g_start = tl.cast(global_start, tl.int64)
    g_size = tl.cast(shard_size, tl.int64)
    # base element offset for expert e, block-row br, block-col bc
    e_base = (tl.cast(elem_global_offset, tl.int64)
              + tl.cast(e, tl.int64) * tl.cast(n_row_elem, tl.int64) * H
              + tl.cast(br, tl.int64) * TILE * H
              + tl.cast(bc, tl.int64) * TILE)

    for rb in range(TILE // ROW_BLOCK):
        row_offs = (rb * ROW_BLOCK + tl.arange(0, ROW_BLOCK)).to(tl.int64)
        g_elem = e_base + row_offs[:, None] * H + col_offs[None, :]
        local_elem = g_elem - g_start
        in_shard = (local_elem >= 0) & (local_elem < g_size)
        src_data = tl.load(shard_ptr + local_elem, mask=in_shard, other=0.0).to(tl.float32)
        fp8_data = (src_data * scale_inv).to(tl.float8e4nv)
        tl.store(out_fp8_ptr + local_elem, fp8_data, mask=in_shard)


def fold_and_quantize_param(
    gathered_scales: torch.Tensor,
    scale_block_offset: int,
    packed_scale_stride: int,
    local_scale_blocks: int,
    shard: torch.Tensor,
    out_fp8: torch.Tensor,
    out_block_scale: torch.Tensor,
    n_experts: int,
    n_row_elem: int,
    n_cols: int,
    elem_offset: int,
    tile_offset: int,
    rank: int,
    world_size: int,
) -> None:
    """Generic single-param fused fold + FP8 quantize.

    Drives ``_fused_fold_quantize_kernel`` once for ONE weight tensor: either a
    plain 2-D matrix (``n_experts == 1``) or a 3-D stacked-expert tensor
    ``[E, n_row_elem, n_cols]`` (``n_experts == E``), treated as E contiguous
    2-D matrices — the kernel's native ``e`` dimension.

    Args:
        gathered_scales: rank-major all-gathered 1x128 tile amaxes
            (shape [world_size * packed_scale_stride]).
        scale_block_offset: this SuperTensor's offset inside a rank's packed amax region.
        packed_scale_stride: total local tiles per rank across all fp8 SuperTensors.
        local_scale_blocks: this SuperTensor's per-rank tile count (shard_size // 128).
        shard: this rank's shard of the flat parameter buffer (1-D, contiguous).
        out_fp8: persistent fp8 shard buffer (same numel as ``shard``).  Padding
            beyond the actual weight elements must be pre-zeroed by the caller —
            this kernel only writes elements belonging to this parameter's range.
        out_block_scale: this param's 128x128 block-scale grid, length
            ``n_experts * (n_row_elem // 128) * (n_cols // 128)``, laid out
            row-major (expert, block_row, block_col).
        n_experts: number of stacked experts (1 for a plain 2-D param).
        n_row_elem, n_cols: the per-expert 2-D shape (both must be % 128 == 0).
        elem_offset: param start in the SuperTensor's flat element array.
        tile_offset: ``elem_offset // 128`` (param start in the flat tile-amax array).
        rank, world_size: fsdp rank / world size.
    """
    assert gathered_scales.is_cuda and gathered_scales.is_contiguous()
    assert shard.dim() == 1 and shard.is_contiguous()
    assert out_fp8.numel() == shard.numel() and out_fp8.dtype == torch.float8_e4m3fn
    assert n_row_elem % FP8_TILE_SIZE == 0 and n_cols % FP8_TILE_SIZE == 0
    assert elem_offset % FP8_TILE_SIZE == 0
    assert 0 <= rank < world_size

    n_col_tiles = n_cols // FP8_TILE_SIZE
    n_block_rows = n_row_elem // FP8_TILE_SIZE
    n_blocks = n_experts * n_block_rows * n_col_tiles
    assert out_block_scale.numel() == n_blocks, (
        f"{out_block_scale.numel()=} != {n_blocks=} for param {n_experts}x{n_row_elem}x{n_cols}"
    )

    if n_blocks == 0:
        return

    shard_size = shard.numel()
    global_start = rank * shard_size
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    ROW_BLOCK = 8

    _fused_fold_quantize_kernel[(n_blocks,)](
        gathered_scales, shard, out_fp8, out_block_scale.view(-1),
        scale_block_offset, packed_scale_stride, local_scale_blocks,
        tile_offset, n_row_elem, n_col_tiles, n_block_rows,
        elem_offset, global_start, shard_size,
        fp8_max,
        TILE=FP8_TILE_SIZE,
        ROW_BLOCK=ROW_BLOCK,
    )
