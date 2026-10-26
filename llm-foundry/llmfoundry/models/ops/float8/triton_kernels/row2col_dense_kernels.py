import typing as tp

import torch
import triton
import triton.language as tl

from llmfoundry.models.ops.triton_utils import params_to_kernel_kwargs


BLOCK_SIZE: int = 128
_ROW2COL_BUCKET_SIZE: int = 2048

STATIC_ROW2COL_TRITON_PARAMS = {
    36864: {
        "num_warps": 4,
        "maxnreg": 128,
    },
    22016: {
        "num_warps": 4,
        "maxnreg": 128,
    },
    18432: {
        "num_warps": 4,
        "maxnreg": 128,
    },
    17920: {
        "num_warps": 4,
        "maxnreg": 128
    },
    11008: {
        "num_warps": 4,
        "maxnreg": 128,
    },
    8960: {
        "num_warps": 4,
        "maxnreg": 128,
    },
    7168: {
        "num_warps": 4,
        "maxnreg": 128,
    },
    4096: {
        "num_warps": 4,
        "maxnreg": 128,
    },
    3072: {
        "num_warps": 4,
        "maxnreg": 128,
    },
    2560: {
        "num_warps": 4,
        "maxnreg": 128,
    },
    2048: {
        "num_warps": 8,
        "maxnreg": 255,
    },
    1536: {
        "num_warps": 4,
        "maxnreg": 192,
    },
    1280: {
        "num_warps": 4,
        "maxnreg": 255,
    }
}

STATIC_BLOCK_TRITON_PARAMS = {
    (36864, 7168): {
        "num_warps": 4,
        "maxnreg": 255,
    },
    (7168, 36864): {
        "num_warps": 4,
        "maxnreg": 255,
    },
    (7168, 18432): {
        "num_warps": 16,
        "maxnreg": 64,
    },
    (18432, 7168): {
        "num_warps": 16,
        "maxnreg": 64,
    },
    (4096, 7168): {
        "num_warps": 16,
        "maxnreg": 128,
    },
    (7168, 4096): {
        "num_warps": 16,
        "maxnreg": 255,
    },
    (7168, 2048): {
        "num_warps": 16,
        "maxnreg": 255,
    },
    (2048, 7168): {
        "num_warps": 16,
        "maxnreg": 255,
    },

    (22016, 4096): {
        "num_warps": 16,
        "maxnreg": 64,
    },
    (4096, 22016): {
        "num_warps": 16,
        "maxnreg": 64,
    },
    (4096, 11008): {
        "num_warps": 16,
        "maxnreg": 128,
    },
    (11008, 4096): {
        "num_warps": 16,
        "maxnreg": 64,
    },
    (3072, 4096): {
        "num_warps": 16,
        "maxnreg": 128,
    },
    (4096, 3072): {
        "num_warps": 16,
        "maxnreg": 128,
    },
    (4096, 1536): {
        "num_warps": 16,
        "maxnreg": 255,
    },
    (1536, 4096): {
        "num_warps": 16,
        "maxnreg": 128,
    },

    (17920, 1536): {
        "num_warps": 16,
        "maxnreg": 128,
    },
    (1536, 17920): {
        "num_warps": 16,
        "maxnreg": 128,
    },
    (1536, 8960): {
        "num_warps": 16,
        "maxnreg": 255,
    },
    (8960, 1536): {
        "num_warps": 16,
        "maxnreg": 255,
    },
    (2560, 1536): {
        "num_warps": 16,
        "maxnreg": 255,
    },
    (1536, 2560): {
        "num_warps": 16,
        "maxnreg": 128,
    },
    (1536, 1280): {
        "num_warps": 16,
        "maxnreg": 128,
    },
    (1280, 1536): {
        "num_warps": 16,
        "maxnreg": 255,
    }
}

@triton.jit
def _rowwise_to_columnwise_inplace_kernel(
    # tensor
    tensor_ptr,
    tensor_stride_row,
    tensor_stride_col,
    tensor_size_row,
    tensor_size_col,
    # out tensor
    out_tensor_ptr,
    out_tensor_stride_row,
    out_tensor_stride_col,
    out_tensor_size_row,
    out_tensor_size_col,
    # scale
    scale_ptr,
    scale_stride_row,
    scale_stride_col,
    scale_size_row,
    scale_size_col,
    # scale
    scale_col_ptr,
    scale_col_stride_row,
    scale_col_stride_col,
    scale_col_size_row,
    scale_col_size_col,
    # block
    BLOCK_SIZE: tl.constexpr,
    POWER_TWO_MAX_ROUND: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    offset_row = pid_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset_col = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # load tensor block of size 128x128
    tensor_ptr = (
        tensor_ptr
        + offset_row[:, None] * tensor_stride_row
        + offset_col[None, :] * tensor_stride_col
    )
    out_tensor_ptr = (
        out_tensor_ptr
        + offset_row[:, None] * out_tensor_stride_row
        + offset_col[None, :] * out_tensor_stride_col
    )
    tensor_mask = (offset_row[:, None] < tensor_size_row) & (offset_col[None, :] < tensor_size_col)
    out_tensor_mask = (offset_row[:, None] < out_tensor_size_row) & (offset_col[None, :] < out_tensor_size_col)
    tensor_block = tl.load(tensor_ptr, mask=tensor_mask)
    tensor_block = tensor_block.to(tl.float8e4nv, bitcast=True)  # fp8e4m3

    # load scale
    scale_ptr = scale_ptr + offset_row[:, None] * scale_stride_row + pid_col * scale_stride_col
    scale_mask = (offset_row[:, None] < scale_size_row) & (pid_col < scale_size_col)
    scale = tl.load(scale_ptr, mask=scale_mask, other=0.0)

    # dequantize
    tensor_block = tensor_block.to(tl.float32) * scale.to(tl.float32)

    # colwise quantization
    scale_col = tl.maximum(tl.max(tl.abs(tensor_block), 0) / 448.0, 1e-30)
    if POWER_TWO_MAX_ROUND:
        scale_col = tl.exp2(tl.ceil(tl.log2(scale_col)))
    scale_inv_col = 1 / scale_col
    tensor_block = tensor_block * scale_inv_col[None, :]

    # save block inplace
    tl.store(
        out_tensor_ptr,
        tensor_block.to(tl.float8e4nv).to(tl.uint8, bitcast=True),
        mask=out_tensor_mask,
    )

    # save scale col
    scale_col = scale_col[None, :]
    scale_col_ptr = (
        scale_col_ptr
        + pid_row * scale_col_stride_row
        + offset_col[None, :] * scale_col_stride_col
    )
    scale_col_mask = (pid_row < scale_col_size_row) & (offset_col[None, :] < scale_col_size_col)
    tl.store(scale_col_ptr, scale_col.to(scale_col_ptr.dtype.element_ty), mask=scale_col_mask)


def rowwise_to_columnwise_inplace(
    tensor: torch.Tensor,
    scales_inv: torch.Tensor,
) -> torch.Tensor:
    assert tensor.is_cuda and scales_inv.is_cuda
    assert tensor.ndim == 2 and scales_inv.ndim == 2

    seq_len, hidden_size = tensor.shape
    assert hidden_size % 128 == 0
    assert seq_len % 128 == 0

    BLOCK_SIZE: int = 128
    blocks_per_row = hidden_size // BLOCK_SIZE
    assert scales_inv.shape == (seq_len, blocks_per_row), (
        f"{scales_inv.shape} != {(seq_len, blocks_per_row)}"
    )

    scales_inv_col = torch.empty_like(scales_inv).view(seq_len // 128, hidden_size)
    output_tensor = torch.empty_like(tensor)

    args = [
        tensor,
        *tensor.stride(),
        *tensor.size(),
        output_tensor,
        *output_tensor.stride(),
        *output_tensor.size(),
        scales_inv,
        *scales_inv.stride(),
        *scales_inv.size(),
        scales_inv_col,
        *scales_inv_col.stride(),
        *scales_inv_col.size(),
    ]

    grid = (
        (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE,
        (hidden_size + BLOCK_SIZE - 1) // BLOCK_SIZE,
    )
    _rowwise_to_columnwise_inplace_kernel[grid](
        *args,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=4,
        POWER_TWO_MAX_ROUND=True,
    )
    return output_tensor, scales_inv_col


@triton.jit
def _rne_shift_right_u32(value, shift):
    """Unsigned dynamic right shift with round-to-nearest, ties-to-even."""
    # Triton evaluates both sides of tl.where. Clamp first so inactive lanes
    # cannot execute a negative or word-width shift.
    safe_shift = tl.minimum(tl.maximum(shift, 1), 31)
    one = value * 0 + 1
    quotient = value >> safe_shift
    remainder_mask = (one << safe_shift) - 1
    remainder = value & remainder_mask
    half = one << (safe_shift - 1)
    round_up = (remainder > half) | (
        (remainder == half) & ((quotient & 1) != 0)
    )
    return quotient + round_up.to(tl.uint32)


@triton.jit
def _scaling_aware_fp8_transpose_kernel(
    # input pointers
    rowwise_data_ptrs,
    rowwise_scale_inv_ptrs,
    columnwise_data_ptrs,
    columnwise_scale_inv_ptrs,
    # sizes
    rows,
    cols,
    rsi_cols,
    # strides
    stride_rowwise_data_r,
    stride_rowwise_data_c,
    stride_rsi_r,
    stride_rsi_c,
    # metas
    BLOCK_SIZE: tl.constexpr,
    COMPUTE_SUBNORMALS: tl.constexpr,
):

    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    r_start = pid_row * BLOCK_SIZE
    c_start = pid_col * BLOCK_SIZE
    r_offsets = r_start + tl.arange(0, BLOCK_SIZE) # [pid_r*128, .., pid_r*128 + 127]
    c_offsets = c_start + tl.arange(0, BLOCK_SIZE) # [pid_c*128, .., pid_c*128 + 127]
    valid_r = r_offsets < rows
    valid_c = c_offsets < cols

    data = tl.load(
        rowwise_data_ptrs + (r_offsets[:, None] * stride_rowwise_data_r + \
                             c_offsets[None, :] * stride_rowwise_data_c),
        mask=valid_r[:, None] & valid_c[None, :],
        other=0,
        cache_modifier=".cg"
    )

    rsi_c_offsets = pid_col + tl.arange(0, 1) # [pid_c]
    valid_rsi_c = rsi_c_offsets < rsi_cols
    si = tl.load(
        rowwise_scale_inv_ptrs + r_offsets[:, None] * stride_rsi_r + \
                                 rsi_c_offsets[None, :] * stride_rsi_c,
        mask=valid_r[:, None] & valid_rsi_c[None, :],
        other=0.0,
        cache_modifier=".cg"
    )

    # For the current block-row (128 rows), take the per-channel max of rowwise_scale_inv
    # This max value becomes the columnwise scaling factor for this block
    target_si = tl.max(si, axis=0)
    tl.store(columnwise_scale_inv_ptrs + (pid_row * cols + c_offsets),
             target_si,
             mask=valid_c,
             cache_modifier=".wb")

    if not COMPUTE_SUBNORMALS:
        # Fast FTZ path: all scales are powers of two, so requantization is an
        # exact exponent decrement. Keep the work in integer registers and
        # preserve raw bytes when the exponent shift is zero.
        sign = data & 0x80
        exp = (data >> 3) & 0xF
        mant = data & 0x7

        bits_target = tl.cast(target_si, tl.uint32, bitcast=True)
        bits_si = tl.cast(si, tl.uint32, bitcast=True)
        exp_target = tl.cast((bits_target & 0x7F800000) >> 23, tl.int32) - 127
        exp_source = tl.cast((bits_si & 0x7F800000) >> 23, tl.int32) - 127
        shift = tl.cast(exp_target[None, :] - exp_source, tl.int32)
        shifted_exp = exp - shift

        shift_needed = shift > 0
        underflow = shift_needed & ((shifted_exp <= 0) | (exp == 0))
        safe_exp = tl.maximum(shifted_exp, 0)
        new_data = sign | (safe_exp << 3) | mant
        new_data = tl.where(underflow, 0, new_data)
    else:
        # Fully integer gradual-underflow path. Normal values remain a cheap
        # exponent decrement; only underflow lanes need integer RNE.
        data_u32 = data.to(tl.uint32)
        sign = data_u32 & 0x80
        exp = (data_u32 >> 3) & 0xF
        mant = data_u32 & 0x7
        magnitude = data_u32 & 0x7F

        bits_target = tl.cast(target_si, tl.uint32, bitcast=True)
        bits_si = tl.cast(si, tl.uint32, bitcast=True)
        exp_target = tl.cast((bits_target & 0x7F800000) >> 23, tl.int32) - 127
        exp_source = tl.cast((bits_si & 0x7F800000) >> 23, tl.int32) - 127
        shift = tl.cast(exp_target[None, :] - exp_source, tl.int32)
        exp_new = exp.to(tl.int32) - shift

        input_is_subnormal = exp == 0
        significand = 0x8 | mant
        sub_shift = 1 - exp_new

        # One vectorized RNE candidate covers both cases:
        #   existing subnormal -> RNE(mant / 2^shift), zero for shift >= 4;
        #   normal underflow   -> RNE((8 + mant) / 2^sub_shift), zero for
        #                         sub_shift >= 5.
        round_value = tl.where(input_is_subnormal, mant, significand)
        round_shift = tl.where(input_is_subnormal, shift, sub_shift)
        rounded_mant = _rne_shift_right_u32(round_value, round_shift)
        zero_shift_limit = tl.where(input_is_subnormal, 4, 5)
        gradual_result = tl.where(
            rounded_mant >= 8,
            sign | 0x08,
            sign | rounded_mant,
        )
        gradual_result = tl.where(
            (round_shift >= zero_shift_limit) | (rounded_mant == 0),
            0,
            gradual_result,
        )

        normal_exp = tl.maximum(exp_new, 0).to(tl.uint32)
        normal_result = sign | (normal_exp << 3) | mant
        stays_normal = (~input_is_subnormal) & (exp_new > 0)
        new_data = tl.where(stays_normal, normal_result, gradual_result)
        new_data = tl.where(shift <= 0, data_u32, new_data)
        new_data = tl.where(magnitude == 0, 0, new_data).to(tl.uint8)

    # Canonicalize both +0 and -0 to match the CUDA result.
    new_data = tl.where((new_data & 0x7F) == 0, 0, new_data)

    # The scale ABI pads the 128-row-block dimension to a multiple of four.
    # Zero only those at-most-three rows from this launch instead of memset-ing
    # the complete scale tensor before overwriting its valid region.
    num_row_blocks = (rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_row_blocks_pad4 = (num_row_blocks + 3) // 4 * 4
    pad_rows = num_row_blocks + tl.arange(0, 4)
    pad_ptrs = (
        columnwise_scale_inv_ptrs
        + pad_rows[:, None] * cols
        + c_offsets[None, :]
    )
    pad_mask = (
        (pid_row == 0)
        & (pad_rows[:, None] < num_row_blocks_pad4)
        & valid_c[None, :]
    )
    tl.store(pad_ptrs, 0.0, mask=pad_mask, cache_modifier=".wb")

    # write columnwise_data (uint8) to [K,M] (c, r)
    tl.store(
        columnwise_data_ptrs + (c_offsets[:, None] * rows + r_offsets[None, :]),
        new_data.T,
        mask=valid_c[:, None] & valid_r[None, :],
        cache_modifier=".wb"
    )


def blockwise_scaling_aware_fp8_transpose(
    rowwise_data: torch.Tensor,
    rowwise_scale_inv: torch.Tensor,
    block_size: int = 128,
    compute_subnormals: bool = False,
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    """
    Scaling-aware FP8 transpose that converts row-wise quantized FP8 tensors to a
    column-wise layout in the FP8 domain.
    """

    device = rowwise_data.device
    data_dtype = rowwise_data.dtype
    scale_dtype = rowwise_scale_inv.dtype

    assert rowwise_data.is_cuda and rowwise_scale_inv.is_cuda
    assert rowwise_data.ndim == 2 and rowwise_scale_inv.ndim == 2
    assert data_dtype in (torch.uint8, torch.float8_e4m3fn)
    assert scale_dtype == torch.float32
    assert block_size == BLOCK_SIZE

    rows = rowwise_data.shape[0]
    cols = rowwise_data.shape[1]
    rsi_cols = rowwise_scale_inv.shape[1]

    assert rows % block_size == 0, f"rows % block_size != 0: {rows} % {block_size} = {rows % block_size}"
    assert cols % block_size == 0, f"cols % block_size != 0: {cols} % {block_size} = {cols % block_size}"
    assert rowwise_scale_inv.shape == (rows, cols // block_size), (
        f"{rowwise_scale_inv.shape} != {(rows, cols // block_size)}"
    )

    if not rowwise_data.is_contiguous():
        rowwise_data = rowwise_data.contiguous()
    if not rowwise_scale_inv.is_contiguous():
        rowwise_scale_inv = rowwise_scale_inv.contiguous()

    nbrows = (rows + block_size - 1) // block_size
    nbcols = (cols + block_size - 1) // block_size

    rowwise_data_u8 = rowwise_data.view(torch.uint8)
    columnwise_data_u8 = torch.empty((cols, rows), dtype=torch.uint8, device=device)
    nbrows_mult_4 = (nbrows + 4 - 1) // 4 * 4
    columnwise_scale_inv = torch.empty((nbrows_mult_4, cols), dtype=scale_dtype, device=device)

    grid = (nbrows, nbcols)
    _scaling_aware_fp8_transpose_kernel[grid](
        rowwise_data_u8,
        rowwise_scale_inv,
        columnwise_data_u8,
        columnwise_scale_inv,
        rows,
        cols,
        rsi_cols,
        *rowwise_data_u8.stride(),
        *rowwise_scale_inv.stride(),
        BLOCK_SIZE=block_size,
        COMPUTE_SUBNORMALS=compute_subnormals,
        num_warps=4,
        num_stages=3,
        # NOTE (Sbr, fedorovgv): we have non effective args here
        # **params_to_kernel_kwargs(STATIC_ROW2COL_TRITON_PARAMS, cols,
        #                           num_warps=4, maxnreg=128)
    )

    return columnwise_data_u8.view(data_dtype), columnwise_scale_inv


@triton.jit
def _block_quantize_kernel(
    # tensor weight (int_dim, hid_size)
    tensor_ptr,
    tensor_stride_row, tensor_stride_col,
    tensor_size_row, tensor_size_col,
    # out (hid_size, int_dim)
    out_ptr,
    out_stride_row, out_stride_col,
    out_size_row, out_size_col,
    # scale (hid_blocks, seq_blocks_per_group)
    scale_ptr,
    scale_stride_row, scale_stride_col,
    scale_size_row, scale_size_col,
    # constexpr
    BLOCK_SIZE: tl.constexpr,
    POWER_TWO_MAX_ROUND: tl.constexpr
):
    pid_token = tl.program_id(axis=0)
    pid_hid = tl.program_id(axis=1)

    # Row in weight for this block: global row = token_block * BLOCK_SIZE
    offset_row = pid_token * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset_col = pid_hid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    tensor_ptr = tensor_ptr + offset_row[:, None] * tensor_stride_row + offset_col[None, :] * tensor_stride_col
    tensor_mask = (offset_row[:, None] < tensor_size_row) & (offset_col[None, :] < tensor_size_col)
    tensor_block = tl.load(tensor_ptr, mask=tensor_mask, other=0.0)
    tensor_block = tensor_block.to(tl.float32)

    scale = tl.maximum(tl.max(tl.abs(tensor_block)) / 448.0, 1e-30)
    if POWER_TWO_MAX_ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))

    scale_inv = 1.0 / scale
    tensor_block = tensor_block * scale_inv
    tensor_block = tl.trans(tensor_block)

    # save out: output[offset_col, pid_token*BLOCK_SIZE + arange]
    out_row = pid_token * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_ptr = (
        out_ptr
        + offset_col[:, None] * out_stride_row
        + out_row[None, :] * out_stride_col
    )
    out_mask = (offset_col[:, None] < out_size_row) & (out_row[None, :] < out_size_col)
    tl.store(out_ptr, tensor_block.to(out_ptr.dtype.element_ty), mask=out_mask)

    # save scale inv: scale[pid_token, pid_hid]
    scale_ptr = (
        scale_ptr
        + pid_token * scale_stride_col
        + pid_hid * scale_stride_row
    )
    scale_mask = (pid_hid < scale_size_row) & (pid_token < scale_size_col)
    tl.store(scale_ptr, scale.to(scale_ptr.dtype.element_ty), mask=scale_mask)


def block_quantization_transpose(weight: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    weight_shape = weight.shape
    int_dim = weight_shape[0]
    hid_size = weight_shape[1]
    assert weight.is_contiguous()

    BLOCK_SIZE: int = 128

    # Single allocation: (hid_size, int_dim)
    output = torch.zeros(
        (hid_size, int_dim),
        device=weight.device,
        dtype=torch.float8_e4m3fn,
    )

    assert int_dim % BLOCK_SIZE == 0, f"int_dim % BLOCK_SIZE != 0: {int_dim} % {BLOCK_SIZE} = {int_dim % BLOCK_SIZE}"
    assert hid_size % BLOCK_SIZE == 0, f"hid_size % BLOCK_SIZE != 0: {hid_size} % {BLOCK_SIZE} = {hid_size % BLOCK_SIZE}"

    seq_blocks_per_group = (int_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
    hid_blocks = (hid_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # \brief Swizzling scaling factors into the required interleaved layout for GEMM
    #
    #  \param[in]     input        Input tensor with non-swizzled scale_inv.
    #  \param[in,out] output       Output tensor which hosts swizzled scale_inv.
    #  \param[in]     stream       CUDA stream used for the operation.
    #
    #  Requirements:
    #  - scale_inv is stored in row-major.
    #  - scale_inv size is padded to 128x4 for row-scale and 4x128 for col-scale.
    #  - data is quantitized along K-dimension, i.e. 1D-scaling block lies along the K-dimension.
    # code: https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/include/transformer_engine/swizzle.h
    seq_blocks_per_group_mult_4 = (seq_blocks_per_group + 4 - 1) // 4 * 4
    scale_inv = torch.zeros(
        (hid_blocks, seq_blocks_per_group_mult_4),
        device=weight.device,
        dtype=torch.float32,
    )

    args = [
        weight,
        weight.stride(0),
        weight.stride(1),
        int_dim,
        hid_size,
        output,
        output.stride(0),
        output.stride(1),
        hid_size,
        int_dim,
        scale_inv,
        scale_inv.stride(0),
        scale_inv.stride(1),
        hid_blocks,
        seq_blocks_per_group,
    ]

    grid = (seq_blocks_per_group, hid_blocks)

    _block_quantize_kernel[grid](
        *args,
        BLOCK_SIZE=BLOCK_SIZE,
        POWER_TWO_MAX_ROUND=True,
        **params_to_kernel_kwargs(STATIC_BLOCK_TRITON_PARAMS,
                                  (int_dim, hid_size),
                                  num_warps=16, maxnreg=255)
    )

    return output, scale_inv
