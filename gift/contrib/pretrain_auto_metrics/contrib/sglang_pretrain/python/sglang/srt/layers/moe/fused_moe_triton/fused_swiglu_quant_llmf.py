"""Fused SwiGLU + FP8 quantization kernel for inference (triton_llmf backend).

Adapted from the forward kernel in the training framework:
  llmfoundry/models/ops/float8/triton_kernels/fused_swiglu_quantization.py

This is a **forward-only**, **inference-only** variant:
  - No autograd Function wrapper
  - No Float8BlockwiseQTensor dependency (raw tensors)
  - No TE / clip_counts dependency
  - No backward pass
  - Compile-time code elimination: APPLY_CLAMP, MUL_PROBS

Forward pipeline:
  load BF16 → cast f32 → optionally clamp(gate ≤ limit, up ∈ [-limit, limit])
  → silu(gate) * up * optional_probs → per-row-block FP8 quantize
  → store FP8 data + f32 scale

Output format is directly compatible with sglang's per_token_group_quant_fp8 output
  (fp8_e4m3fn data, float32 row-wise scales with group_size=128).
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

FP8_MAX: float = 448.0
INV_FP8_MAX: tl.constexpr = tl.constexpr(1.0 / FP8_MAX)
MIN_SCALE: tl.constexpr = tl.constexpr(1e-30)

# Default row-block size for forward kernel.
# 64 rows × 128 threads × 160 regs = 20480 ≪ 65536 → ~3 blocks/SM on H100.
# Can be overridden via `row_block` parameter for tuning.
_DEFAULT_ROW_BLOCK = 64

# Static tuning table for forward kernel (same as training).
# Keys are intermediate_size (hid_dim), values are triton launch parameters.
STATIC_TRITON_PARAMS_FWD: dict[int, dict[str, int]] = {
    4096: {"num_warps": 4, "maxnreg": 160},
    2048: {"num_warps": 4, "maxnreg": 160},
    1536: {"num_warps": 4, "maxnreg": 160},
    1280: {"num_warps": 4, "maxnreg": 160},
}


# ── helper ────────────────────────────────────────────────────────────────────

@triton.jit
def _fp8_scale_inv_from_amax(amax):
    amax_f32 = amax.to(tl.float32)
    scale_pre = tl.maximum(amax_f32 * INV_FP8_MAX, MIN_SCALE)
    e = tl.ceil(tl.log2(scale_pre))
    return tl.exp2(e), tl.exp2(-e)  # scale, inv_scale


# ── forward kernel ────────────────────────────────────────────────────────────

@triton.jit
def _swiglu_quant_fwd_llmf_kernel(
    # input (packed gate|up, BF16/FP16) — shape [M, 2N]
    input_tensor_ptr,
    input_tensor_stride_row,
    input_tensor_stride_col,
    input_tensor_size_row,
    input_tensor_size_col,
    # output FP8 data — shape [M, N]
    fp8_data_ptr,
    fp8_data_stride_row,
    fp8_data_stride_col,
    fp8_data_size_row,
    fp8_data_size_col,
    # output FP8 scales (f32) — shape [M, N // 128]
    fp8_scales_ptr,
    fp8_scales_stride_row,
    fp8_scales_stride_col,
    fp8_scales_size_row,
    fp8_scales_size_col,
    # hid_dim = N (half of input_tensor last dim)
    hid_dim: tl.constexpr,
    ROW_BLOCK: tl.constexpr,  # 64
    COL_BLOCK: tl.constexpr,  # 128
    APPLY_CLAMP: tl.constexpr,
    swiglu_limit: tl.constexpr,
    MUL_PROBS: tl.constexpr,
    # probs — only used when MUL_PROBS=True; pass a null/zero pointer otherwise
    probs_ptr,
    probs_stride_row,
    probs_size_row,
):
    pid_token = tl.program_id(axis=0)
    pid_hid = tl.program_id(axis=1)

    row_indices = (pid_token * ROW_BLOCK + tl.arange(0, ROW_BLOCK)).to(tl.int64)
    row_mask = row_indices < input_tensor_size_row

    gate_col_offsets = pid_hid * COL_BLOCK + tl.arange(0, COL_BLOCK)
    up_col_offsets = hid_dim + gate_col_offsets

    gate_block = tl.load(
        input_tensor_ptr
        + row_indices[:, None] * input_tensor_stride_row
        + gate_col_offsets[None, :] * input_tensor_stride_col,
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    up_block = tl.load(
        input_tensor_ptr
        + row_indices[:, None] * input_tensor_stride_row
        + up_col_offsets[None, :] * input_tensor_stride_col,
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    if APPLY_CLAMP:
        up_block = tl.minimum(tl.maximum(up_block, -swiglu_limit), swiglu_limit)
        gate_block = tl.minimum(gate_block, swiglu_limit)

    # silu(gate) * up
    gate_sigm = tl.sigmoid(gate_block)
    swiglu = (gate_block * gate_sigm) * up_block

    if MUL_PROBS:
        probs_vec = tl.load(
            probs_ptr + row_indices * probs_stride_row,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        swiglu = swiglu * probs_vec[:, None]

    # per-row-block quantisation to FP8 (group_size == COL_BLOCK == 128)
    # Zero-out invalid rows so they don't pollute the amax/scale computation
    swiglu = tl.where(row_mask[:, None], swiglu, 0.0)
    swiglu_amax = tl.max(tl.abs(swiglu), axis=1)
    swiglu_sf, swiglu_inv_sf = _fp8_scale_inv_from_amax(swiglu_amax)
    swiglu_q = swiglu * swiglu_inv_sf[:, None]

    tl.store(
        fp8_data_ptr
        + row_indices[:, None] * fp8_data_stride_row
        + gate_col_offsets[None, :] * fp8_data_stride_col,
        swiglu_q.to(fp8_data_ptr.dtype.element_ty),
        mask=row_mask[:, None],
    )
    tl.store(
        fp8_scales_ptr
        + row_indices * fp8_scales_stride_row
        + pid_hid * fp8_scales_stride_col,
        swiglu_sf.to(fp8_scales_ptr.dtype.element_ty),
        mask=row_mask,
    )


# ── public API ────────────────────────────────────────────────────────────────

def _get_triton_params(hid_dim: int) -> dict:
    """Look up triton launch parameters for the given hidden dimension."""
    default = {"num_warps": 4, "maxnreg": 160}
    if hid_dim in STATIC_TRITON_PARAMS_FWD:
        return STATIC_TRITON_PARAMS_FWD[hid_dim]
    # Find the closest known size — use largest available if none match.
    for known in sorted(STATIC_TRITON_PARAMS_FWD.keys()):
        if hid_dim <= known:
            return STATIC_TRITON_PARAMS_FWD[known]
    return default


def fused_swiglu_quant_llmf(
    gateup_output: torch.Tensor,
    swiglu_limit: float = 0.0,
    probs: Optional[torch.Tensor] = None,
    group_size: int = 128,
    row_block: int = _DEFAULT_ROW_BLOCK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused SwiGLU + per-row-block FP8 quantization (inference-only).

    Takes packed gate|up output from GEMM1, applies:
      1. Optional clamping  (APPLY_CLAMP compiled away when swiglu_limit=0.0)
      2. silu(gate) * up * optional_probs
      3. Per-row-block FP8 E4M3 quantization (group_size along hidden dim)

    Args:
        gateup_output: [M, 2N] packed gate|up tensor in BF16/FP16.
        swiglu_limit: 0.0 = no clamping; > 0 enables DeepSeek-style clamp
            (gate ≤ limit, up ∈ [-limit, limit]).
        probs: Optional [M] routing weights (None for dense/inference path).
        group_size: Scale granularity along the hidden dimension (default 128).
            Must match block_k from the downstream GEMM's block_shape[1].
        row_block: Row tiling size (default 64). Performance tuning parameter;
            does not affect correctness.

    Returns:
        tuple (fp8_data, fp8_scales):
            fp8_data:  [M, N]     torch.float8_e4m3fn
            fp8_scales:[M, N // group_size] torch.float32 row-wise scales
        These are directly compatible with the output format of
        ``sglang_per_token_group_quant_fp8``.
    """
    total_tokens, hidden2 = gateup_output.shape
    hid_dim = hidden2 // 2

    device = gateup_output.device

    # Shape constraints: hidden dim must be divisible by group_size.
    # Rows can be arbitrary — the kernel masks invalid rows at boundaries.
    assert hid_dim % group_size == 0, (
        f"Hidden dim {hid_dim} must be divisible by group_size={group_size}."
    )

    blocks_per_row = hid_dim // group_size

    fp8_data = torch.empty(
        (total_tokens, hid_dim),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    fp8_scales = torch.empty(
        (total_tokens, blocks_per_row),
        device=device,
        dtype=torch.float32,
    )

    grid = (triton.cdiv(total_tokens, row_block), blocks_per_row)
    triton_params = _get_triton_params(hid_dim)

    has_probs = probs is not None and probs.numel() > 0
    probs_contig = probs.reshape(-1).contiguous() if has_probs else torch.empty(0, device=device)

    _swiglu_quant_fwd_llmf_kernel[grid](
        # input tensor strides
        gateup_output,
        gateup_output.stride(0),
        gateup_output.stride(1),
        gateup_output.shape[0],
        gateup_output.shape[1],
        # fp8 data strides
        fp8_data,
        fp8_data.stride(0),
        fp8_data.stride(1),
        fp8_data.shape[0],
        fp8_data.shape[1],
        # fp8 scales strides
        fp8_scales,
        fp8_scales.stride(0),
        fp8_scales.stride(1),
        fp8_scales.shape[0],
        fp8_scales.shape[1],
        # constexpr / runtime constants
        hid_dim,
        ROW_BLOCK=row_block,
        COL_BLOCK=group_size,
        APPLY_CLAMP=swiglu_limit > 0,
        swiglu_limit=float(swiglu_limit),
        MUL_PROBS=has_probs,
        probs_ptr=probs_contig,
        probs_stride_row=probs_contig.stride(0) if has_probs else 0,
        probs_size_row=probs_contig.shape[0] if has_probs else 0,
        **triton_params,
    )

    return fp8_data, fp8_scales