import typing as tp

import torch
from torch.autograd.profiler import record_function as record_fn
import triton
import triton.language as tl

from llmfoundry.models.ops.float8.triton_kernels.utils import tensor_to_kernel_args
from llmfoundry.models.ops.triton_utils import params_to_kernel_kwargs
from llmfoundry.models.ops.float8.fp8_tensor import Float8BlockwiseQTensor, InputLayoutType
import llmfoundry.models.ops.float8.triton_kernels.fixed_transformer_engine_permutation as triton_permutation


_PERMUTE_TOKEN_BUCKET_SIZE: int = 2048

STATIC_PERMUTE_TRITON_PARAMS_FWD = {
    7168: {
        "num_warps": 1,
        "num_stages": 1,
        "maxnreg": 128,
        "BLOCK_SIZE": 1024,
        "BLOCKS_NUM_TO_PROCESS_IN_ROW": 8,
    },
    4096: {
        "num_warps": 1,
        "num_stages": 4,
        "maxnreg": 192,
        "BLOCK_SIZE": 1024,
        "BLOCKS_NUM_TO_PROCESS_IN_ROW": 8,
    },
    1536: {
        "num_warps": 1,
        "num_stages": 1,
        "maxnreg": 64,
        "BLOCK_SIZE": 1024,
        "BLOCKS_NUM_TO_PROCESS_IN_ROW": 8,
    },
}

STATIC_PERMUTE_TRITON_PARAMS_BWD = {
    7168: {
        "num_warps": 2,
        "num_stages": 2,
        "maxnreg": 128,
        "BLOCK_SIZE": 1024,
        "BLOCKS_NUM_TO_PROCESS_IN_ROW": 8,
    },
    4096: {
        "num_warps": 2,
        "num_stages": 4,
        "maxnreg": 32,
        "BLOCK_SIZE": 1024,
        "BLOCKS_NUM_TO_PROCESS_IN_ROW": 8,
    },
    1536: {
        "num_warps": 2,
        "num_stages": 1,
        "maxnreg": 64,
        "BLOCK_SIZE": 1024,
        "BLOCKS_NUM_TO_PROCESS_IN_ROW": 8,
    },
}

STATIC_UNPERMUTE_TRITON_PARAMS_FWD = {
    7168: {
        "num_warps": 1,
        "num_stages": 2,
        "maxnreg": 128,
        "BLOCK_SIZE": 1024
    },
    4096: {
        "num_warps": 2,
        "num_stages": 1,
        "maxnreg": 32,
        "BLOCK_SIZE": 1024
    },
    1536: {
        "num_warps": 2,
        "num_stages": 1,
        "maxnreg": 32,
        "BLOCK_SIZE": 1024
    },
}

STATIC_UNPERMUTE_TRITON_PARAMS_BWD = {
    7168: {
        "num_warps": 1,
        "num_stages": 4,
        "maxnreg": 64,
        "BLOCK_SIZE": 1024
    },
    4096: {
        "num_warps": 1,
        "num_stages": 2,
        "maxnreg": 64,
        "BLOCK_SIZE": 1024
    },
    1536: {
        "num_warps": 1,
        "num_stages": 2,
        "maxnreg": 64,
        "BLOCK_SIZE": 1024
    },
}

# NOTE (fedorovgv): in feature we have to replace row id map preparation
#   from te with more effective implementation;
@triton.jit
def _row_to_padded_row_id_kernel(
    group_sizes_ptr,
    out_indices_ptr,
    num_experts: tl.constexpr,
    align_size: tl.constexpr,
):
    pid = tl.program_id(0)
    prefix = 0
    padded_prefix = 0
    new_idx = 0
    for expert in tl.range(0, num_experts):
        expert_size = tl.load(group_sizes_ptr + expert).to(tl.int32)
        start = prefix
        end = start + expert_size
        in_range = (pid >= start) & (pid < end)
        local_idx = pid - start
        padded_size = ((expert_size + align_size - 1) // align_size) * align_size
        cand = padded_prefix + local_idx
        new_idx = tl.where(in_range, cand, new_idx)
        prefix = end
        padded_prefix = padded_prefix + padded_size
    tl.store(out_indices_ptr + pid, new_idx)


@triton.jit
def _zero_expert_padding_kernel(
    ptr,
    stride_row,
    size_col,
    pad_starts_ptr,  # [num_experts] int32: first padding row for each expert in padded space
    pad_counts_ptr,  # [num_experts] int32: number of padding rows per expert
    BLOCK_COL: tl.constexpr,
):
    """Zero only the alignment-padding rows for each expert.

    Grid: (num_experts, cdiv(size_col, BLOCK_COL))

    For 1-D tensors call with stride_row=1, size_col=1, BLOCK_COL=1 so that
    the column offset is always 0 and each store hits exactly one element.

    Each thread-group is responsible for one expert's padding rows and one
    column tile.  The inner loop over padding rows is short (< ALIGN_SIZE=128)
    so it stays in registers with no shared-memory pressure.
    """
    pid_e = tl.program_id(0)
    pid_c = tl.program_id(1)

    pad_start = tl.load(pad_starts_ptr + pid_e).to(tl.int64)
    pad_count = tl.load(pad_counts_ptr + pid_e).to(tl.int32)

    col = pid_c * BLOCK_COL + tl.arange(0, BLOCK_COL)
    col_mask = col < size_col
    zeros = tl.zeros((BLOCK_COL,), dtype=ptr.dtype.element_ty)
    for i in tl.range(pad_count):
        tl.store(
            ptr + (pad_start + i) * stride_row + col,
            zeros,
            mask=col_mask,
        )


def _launch_zero_expert_padding(
    tensor: torch.Tensor,
    pad_starts_t: torch.Tensor,
    pad_counts_t: torch.Tensor,
    BLOCK_COL: int = 1024,
) -> None:
    """Launch _zero_expert_padding_kernel for a 1-D or 2-D tensor.

    1-D tensors are handled by passing stride_row=1, size_col=1, BLOCK_COL=1
    so the column offset is always 0 — equivalent to scalar stores but using
    the same block-based kernel path (avoids the scalar-pointer / block-value
    mismatch that Triton rejects).
    """
    num_experts = pad_starts_t.shape[0]
    if tensor.dim() == 1:
        # Treat as (N, 1) with stride_row=1 so store hits ptr + row + 0.
        _zero_expert_padding_kernel[(num_experts, 1)](
            tensor, 1, 1,
            pad_starts_t, pad_counts_t,
            BLOCK_COL=1,
        )
    else:
        size_col = tensor.size(1)
        grid = (num_experts, triton.cdiv(size_col, BLOCK_COL))
        _zero_expert_padding_kernel[grid](
            tensor, tensor.stride(0), size_col,
            pad_starts_t, pad_counts_t,
            BLOCK_COL=BLOCK_COL,
        )


@triton.jit
def _remap_rows_kernel(
    src_ptr,
    src_stride_row,
    dst_ptr,
    dst_stride_row,
    index_ptr,  # [n_index] int32: per-program row mapping
    n_cols,
    BLOCK_COL: tl.constexpr,
    SCATTER: tl.constexpr,
):
    """Copy whole rows between two layouts using a row-index map.

    Grid: (n_index, cdiv(n_cols, BLOCK_COL)).

    SCATTER=True : dst[index[pid_row]] = src[pid_row]        (compact -> padded)
    SCATTER=False: dst[pid_row]        = src[index[pid_row]]  (padded -> compact)

    For 1-D tensors pass src_stride_row=dst_stride_row=1, n_cols=1, BLOCK_COL=1
    so the column offset is always 0 and each store hits exactly one element.
    """
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    idx = tl.load(index_ptr + pid_row).to(tl.int64)
    if SCATTER:
        src_row = pid_row.to(tl.int64)
        dst_row = idx
    else:
        src_row = idx
        dst_row = pid_row.to(tl.int64)

    col = pid_col * BLOCK_COL + tl.arange(0, BLOCK_COL)
    mask = col < n_cols
    vals = tl.load(src_ptr + src_row * src_stride_row + col, mask=mask)
    tl.store(dst_ptr + dst_row * dst_stride_row + col, vals, mask=mask)


def _launch_remap_rows(
    src: torch.Tensor,
    dst: torch.Tensor,
    index: torch.Tensor,
    scatter: bool,
    BLOCK_COL: int = 1024,
) -> None:
    """Launch _remap_rows_kernel for a 1-D or 2-D tensor (see _launch_zero_expert_padding)."""
    n_index = index.shape[0]
    if src.dim() == 1:
        _remap_rows_kernel[(n_index, 1)](
            src, 1, dst, 1, index, 1,
            BLOCK_COL=1, SCATTER=scatter,
        )
    else:
        n_cols = src.size(1)
        grid = (n_index, triton.cdiv(n_cols, BLOCK_COL))
        _remap_rows_kernel[grid](
            src, src.stride(0), dst, dst.stride(0), index, n_cols,
            BLOCK_COL=BLOCK_COL, SCATTER=scatter,
        )


@triton.jit(do_not_specialize=[
    "input_size_row",
    "output_size_row",
    "row_id_map_size_row",
    "probs_size_row",
    "scale_size_row",
    "permuted_probs_size_row",
    "permuted_scale_size_row",
    "dst_to_padded_dst_size_row"
])
def _permute_and_pad_kernel(
    input_ptr,  # tokens x hs
    input_stride_row, input_stride_col,
    input_size_row, input_size_col,
    #
    output_ptr,  # total_tokens x hs
    output_stride_row, output_stride_col,
    output_size_row, output_size_col,
    #
    row_id_map_ptr,  # tokens x num_experts
    row_id_map_stride_row, row_id_map_stride_col,
    row_id_map_size_row, row_id_map_size_col,
    #
    probs_ptr,  # tokens x num_experts
    probs_stride_row, probs_stride_col,
    probs_size_row, probs_size_col,
    #
    scale_ptr,  # tokens x (hs // 128)
    scale_stride_row, scale_stride_col,
    scale_size_row, scale_size_col,
    #
    permuted_probs_ptr,  # tokens x num_experts
    permuted_probs_stride_row,
    permuted_probs_size_row,
    #
    permuted_scale_ptr,  # total_tokens x (hs // 128)
    permuted_scale_stride_row, permuted_scale_stride_col,
    permuted_scale_size_row, permuted_scale_size_col,
    #
    dst_to_padded_dst_ptr,  # num_experts
    dst_to_padded_dst_stride_row,
    dst_to_padded_dst_size_row,
    #
    num_experts: tl.constexpr,
    # Bucketed token count for the autotune key: tokens // _PERMUTE_TOKEN_BUCKET_SIZE.
    # Not used in the kernel body; only drives autotuner config selection.
    total_token_rank_bucket,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_NUM_TO_PROCESS_IN_ROW: tl.constexpr,
    PERMUTE_SCALE: tl.constexpr,
    PERMUTE_PROBS: tl.constexpr,
):
    # Keep row indices 64-bit before multiplying by tensor strides. Large SFT
    # batches can otherwise overflow int32 at token_index * hidden_size.
    pid_row = tl.program_id(0).to(tl.int64)
    pid_col = tl.program_id(1)

    input_offset_col = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    input_ptr = (
        input_ptr
        + pid_row * input_stride_row
        + input_offset_col * input_stride_col
    )
    mask_input = (input_offset_col < input_size_col)
    input_block = tl.load(input_ptr, mask=mask_input) # 1 x block_col

    if PERMUTE_SCALE:
        scale_offset_col = (
            pid_col * BLOCKS_NUM_TO_PROCESS_IN_ROW
            + tl.arange(0, BLOCKS_NUM_TO_PROCESS_IN_ROW)
        )
        scale_ptr = (
            scale_ptr
            + pid_row * scale_stride_row
            + scale_offset_col * scale_stride_col
        )
        mask_scale = (scale_offset_col < scale_size_col)
        scale_block = tl.load(scale_ptr, mask=mask_scale)  # 1 x (block_col // 128)

    n_routed = tl.load(
        row_id_map_ptr
        + pid_row * row_id_map_stride_row
        + num_experts * 2 * row_id_map_stride_col
    )

    for idx in tl.range(n_routed):
        dst_row = tl.load(
            row_id_map_ptr
            + pid_row * row_id_map_stride_row
            + idx * row_id_map_stride_col
        )
        padded_dst_row = tl.load(
            dst_to_padded_dst_ptr
            + dst_row * dst_to_padded_dst_stride_row
        ).to(tl.int64)
        output_ptr_offset = (
            padded_dst_row * output_stride_row
            + input_offset_col * output_stride_col
        )
        tl.store(output_ptr + output_ptr_offset, input_block, mask=mask_input)

        if PERMUTE_SCALE:
            permuted_scale_offset = (
                padded_dst_row * permuted_scale_stride_row
                + scale_offset_col * permuted_scale_stride_col
            )
            mask_permuted_scale = (scale_offset_col < scale_size_col)
            tl.store(permuted_scale_ptr + permuted_scale_offset,
                     scale_block,
                     mask=mask_permuted_scale)

        if PERMUTE_PROBS:
            if pid_col == 0:
                expert_idx = tl.load(
                    row_id_map_ptr
                    + pid_row * row_id_map_stride_row
                    + (num_experts + idx) * row_id_map_stride_col
                )
                probs_offset = (
                    pid_row * probs_stride_row
                    + expert_idx * probs_stride_col
                )
                prob = tl.load(probs_ptr + probs_offset)
                permuted_prob_offset = padded_dst_row * permuted_probs_stride_row
                tl.store(permuted_probs_ptr + permuted_prob_offset, prob)


@triton.jit(do_not_specialize=[
    "input_size_row",
    "output_size_row",
    "row_id_map_size_row",
    "dst_to_padded_dst_size_row",
    "permuted_probs_size_row",
    "unpermute_probs_size_row"
])
def _unpad_and_unpermute_kernel(
    input_ptr,  # tokens x hs
    input_stride_row, input_stride_col,
    input_size_row, input_size_col,
    #
    output_ptr,  # total_tokens x hs
    output_stride_row, output_stride_col,
    output_size_row, output_size_col,
    #
    row_id_map_ptr,  # tokens x num_experts
    row_id_map_stride_row, row_id_map_stride_col,
    row_id_map_size_row, row_id_map_size_col,
    #
    dst_to_padded_dst_ptr,  # num_experts
    dst_to_padded_dst_stride_row,
    dst_to_padded_dst_size_row,
    #
    permuted_probs_ptr,  # total_tokens
    permuted_probs_stride_row,
    permuted_probs_size_row,
    #
    unpermute_probs_ptr,  # tokens x num_experts
    unpermute_probs_stride_row, unpermute_probs_stride_col,
    unpermute_probs_size_row, unpermute_probs_size_col,
    #
    num_experts: tl.constexpr,
    # Bucketed token count for the autotune key: tokens // _PERMUTE_TOKEN_BUCKET_SIZE.
    # Not used in the kernel body; only drives autotuner config selection.
    total_token_rank_bucket,
    BLOCK_SIZE: tl.constexpr,
    UNPERMUTE_PROBS: tl.constexpr,
):
    # Keep row indices 64-bit before multiplying by tensor strides. Large SFT
    # batches can otherwise overflow int32 at token_index * hidden_size.
    pid_t = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1)

    n_routed = tl.load(
        row_id_map_ptr
        + pid_t * row_id_map_stride_row
        + num_experts * 2 * row_id_map_stride_col
    )

    input_offset = pid_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    input_mask = (input_offset < input_size_col)

    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for idx in tl.range(n_routed):
        src_row = tl.load(
            row_id_map_ptr
            + pid_t * row_id_map_stride_row
            + idx * row_id_map_stride_col
        ).to(tl.int64)
        padded_src_row = tl.load(
            dst_to_padded_dst_ptr
            + src_row * dst_to_padded_dst_stride_row
        ).to(tl.int64)
        input_row_block = tl.load(
            input_ptr + padded_src_row * input_stride_row + input_offset * input_stride_col,
            mask=input_mask
        )
        input_row_block = input_row_block.to(tl.float32)
        accumulator = accumulator + input_row_block

        if UNPERMUTE_PROBS:
            if pid_h == 0:
                expert_idx = tl.load(
                    row_id_map_ptr
                    + pid_t * row_id_map_stride_row
                    + (num_experts + idx) * row_id_map_stride_col
                )

                permuted_prob_offset = padded_src_row * permuted_probs_stride_row
                prob = tl.load(permuted_probs_ptr + permuted_prob_offset)

                unpermuted_prob_offset = (
                    pid_t * unpermute_probs_stride_row
                    + expert_idx * unpermute_probs_stride_col
                )
                tl.store(unpermute_probs_ptr + unpermuted_prob_offset, prob)

    output_offset = pid_t * output_stride_row + input_offset * output_stride_col
    output_mask = (input_offset < input_size_col)
    tl.store(output_ptr + output_offset,
             accumulator.to(output_ptr.dtype.element_ty),
             mask=output_mask)


class _permute_and_pad(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                recv_x: torch.Tensor,
                local_map: torch.Tensor,
                probs: torch.Tensor,
                tokens_per_expert: torch.Tensor,) -> tp.Tuple[torch.Tensor, ...]:
        with record_fn("_permute_and_pad_fwd"):
            if not recv_x.numel():
                ctx.probs = probs
                empty = torch.tensor([], device=recv_x.device)
                return (recv_x, empty, empty, empty, empty, empty)

            num_tokens = recv_x.shape[0]
            num_experts = local_map.size(1)
            row_id_map = triton_permutation.make_row_id_map(
                local_map, num_tokens, num_experts)

            ALIGN_SIZE: int = 128

            tokens_per_expert_list = tokens_per_expert.cpu().tolist()
            num_experts = len(tokens_per_expert_list)
            tokens_per_expert_list_padded = [
                (m + ALIGN_SIZE - 1) // ALIGN_SIZE * ALIGN_SIZE for m in tokens_per_expert_list]

            total_tokens_padded = sum(tokens_per_expert_list_padded)
            row_to_padded_row_id = torch.empty((total_tokens_padded,),
                                            device=recv_x.device,
                                            dtype=torch.int32)

            # we need to make mapping from row in unpadded list to list with group paddings
            _row_to_padded_row_id_kernel[(total_tokens_padded,)](
                tokens_per_expert,
                row_to_padded_row_id,
                num_experts=num_experts,
                align_size=ALIGN_SIZE,)

            # Compute per-expert padding info on CPU (no extra GPU sync: tokens_per_expert
            # was already synced via .cpu().tolist() above).
            padded_offset = 0
            pad_starts: tp.List[int] = []
            pad_counts: tp.List[int] = []
            for t, p in zip(tokens_per_expert_list, tokens_per_expert_list_padded):
                pad_starts.append(padded_offset + t)
                pad_counts.append(p - t)
                padded_offset += p
            pad_starts_t = torch.tensor(pad_starts, device=recv_x.device, dtype=torch.int32)
            pad_counts_t = torch.tensor(pad_counts, device=recv_x.device, dtype=torch.int32)

            tokens_on_rank, hidden_size = recv_x.size()
            output_dtype = recv_x.dtype if not isinstance(recv_x, Float8BlockwiseQTensor) else torch.uint8

            # Use torch.empty; padding rows are zeroed by _zero_expert_padding_kernel.
            # _permute_and_pad_kernel only writes real-token rows (disjoint from padding rows),
            # so the two launches are order-independent on the same stream.
            output = torch.empty((total_tokens_padded, hidden_size),
                                device=recv_x.device,
                                dtype=output_dtype)
            _launch_zero_expert_padding(output, pad_starts_t, pad_counts_t)

            permuted_scale = None
            scale = None
            if isinstance(recv_x, Float8BlockwiseQTensor):
                scale = recv_x._scale
                _, scale_hidden_size = scale.size()
                permuted_scale = torch.empty((total_tokens_padded, scale_hidden_size),
                                            device=scale.device,
                                            dtype=scale.dtype)
                _launch_zero_expert_padding(permuted_scale, pad_starts_t, pad_counts_t)

            permuted_probs = torch.empty((total_tokens_padded,),
                                        device=probs.device,
                                        dtype=probs.dtype,
                                        requires_grad=probs.requires_grad)
            _launch_zero_expert_padding(permuted_probs, pad_starts_t, pad_counts_t)

            data = (
                recv_x if not isinstance(recv_x, Float8BlockwiseQTensor)
                else recv_x._data.view(torch.uint8)
            )

            total_token_rank_bucket = tokens_on_rank // _PERMUTE_TOKEN_BUCKET_SIZE
            grid = lambda META: (tokens_on_rank, triton.cdiv(hidden_size, META["BLOCK_SIZE"]))
            _permute_and_pad_kernel[grid](
                *tensor_to_kernel_args(data, dims=2),
                *tensor_to_kernel_args(output, dims=2),
                *tensor_to_kernel_args(row_id_map, dims=2),
                *tensor_to_kernel_args(probs, dims=2),
                *tensor_to_kernel_args(scale, dims=2),
                *tensor_to_kernel_args(permuted_probs, dims=1),
                *tensor_to_kernel_args(permuted_scale, dims=2),
                *tensor_to_kernel_args(row_to_padded_row_id, dims=1),
                num_experts=num_experts,
                total_token_rank_bucket=total_token_rank_bucket,
                PERMUTE_SCALE=(permuted_scale is not None),
                PERMUTE_PROBS=(probs is not None),
                **params_to_kernel_kwargs(STATIC_PERMUTE_TRITON_PARAMS_FWD, hidden_size,
                                          num_warps=1, num_stages=1, 
                                          maxnreg=128, BLOCK_SIZE=1024, BLOCKS_NUM_TO_PROCESS_IN_ROW=8)
            )

            ctx.num_experts = num_experts
            ctx.row_to_padded_row_id = row_to_padded_row_id
            ctx.row_id_map = row_id_map
            ctx.input_shape_before_permute = recv_x.shape
            ctx.probs_shape = probs.shape
            ctx.total_token_rank_bucket = total_token_rank_bucket

            if output.dtype == torch.uint8:
                output = Float8BlockwiseQTensor(
                    data=output,
                    scale=permuted_scale,
                    orig_dtype=torch.bfloat16,
                    input_layout_type=InputLayoutType.ROWWISE,
                    block_scaling_dim=1,
                )

        return output, permuted_probs, row_id_map, row_to_padded_row_id, pad_starts_t, pad_counts_t

    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor,
                 grad_permuted_probs: torch.Tensor,
                 grad_row_id_map: torch.Tensor,
                 grad_row_to_padded_row_id: torch.Tensor,
                 grad_pad_starts: torch.Tensor,
                 grad_pad_counts: torch.Tensor,):

        with record_fn("_permute_and_pad_bwd"):
            assert grad_output.dtype in [torch.bfloat16, torch.float32]

            if not grad_output.numel():
                return grad_output, None, None, None

            # Group A: _unpad_and_unpermute_kernel writes every row of grad_input
            # (grid covers all pid_t in [0, num_tokens)), so torch.empty is safe.
            grad_input = torch.empty(
                ctx.input_shape_before_permute,
                device=grad_output.device,
                dtype=grad_output.dtype)

            # Group C: only routed (token, expert) pairs are written → must stay zeros.
            grad_input_probs = torch.zeros(
                ctx.probs_shape,
                device=grad_permuted_probs.device,
                dtype=grad_permuted_probs.dtype)

            _n, _h = ctx.input_shape_before_permute[0], ctx.input_shape_before_permute[1]
            grid = lambda META: (_n, triton.cdiv(_h, META["BLOCK_SIZE"]))

            _unpad_and_unpermute_kernel[grid](
                *tensor_to_kernel_args(grad_output, dims=2),
                *tensor_to_kernel_args(grad_input, dims=2),
                *tensor_to_kernel_args(ctx.row_id_map, dims=2),
                *tensor_to_kernel_args(ctx.row_to_padded_row_id, dims=1),
                *tensor_to_kernel_args(grad_permuted_probs, dims=1),
                *tensor_to_kernel_args(grad_input_probs, dims=2),
                num_experts=ctx.num_experts,
                total_token_rank_bucket=ctx.total_token_rank_bucket,
                UNPERMUTE_PROBS=True,
                **params_to_kernel_kwargs(STATIC_UNPERMUTE_TRITON_PARAMS_BWD, _h,
                                          num_warps=1, num_stages=4,
                                          maxnreg=64, BLOCK_SIZE=1024)
            )

        return (grad_input,
                None,
                grad_input_probs,
                None,)


class _unpad_and_unpermute(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                tensor: torch.Tensor,
                row_map_id: torch.Tensor,
                row_to_padded_row_id: torch.Tensor,
                restore_shape: tp.Tuple[int, ...],
                pad_starts_t: torch.Tensor,
                pad_counts_t: torch.Tensor,) -> torch.Tensor:

        with record_fn("_unpad_and_unpermute_fwd"):
            if not tensor.numel():
                return tensor

            BLOCK_SIZE: int = 1024
            ALIGN_SIZE: int = 128
            ctx.BLOCK_SIZE = BLOCK_SIZE
            ctx.ALIGN_SIZE = ALIGN_SIZE

            num_experts = (row_map_id.size(1) - 1) // 2

            # Group A: _unpad_and_unpermute_kernel writes every row of output
            # (grid = (restore_shape[0], ...) covers all tokens), so torch.empty is safe.
            output = torch.empty(restore_shape, device=tensor.device, dtype=tensor.dtype)

            _n, _h = restore_shape[0], restore_shape[1]
            total_token_rank_bucket = _n // _PERMUTE_TOKEN_BUCKET_SIZE

            grid = lambda META: (_n, triton.cdiv(_h, META["BLOCK_SIZE"]))
            _unpad_and_unpermute_kernel[grid](
                *tensor_to_kernel_args(tensor, dims=2),
                *tensor_to_kernel_args(output, dims=2),
                *tensor_to_kernel_args(row_map_id, dims=2),
                *tensor_to_kernel_args(row_to_padded_row_id, dims=1),
                *tensor_to_kernel_args(None, dims=1),
                *tensor_to_kernel_args(None, dims=2),
                num_experts=num_experts,
                total_token_rank_bucket=total_token_rank_bucket,
                UNPERMUTE_PROBS=False,
                **params_to_kernel_kwargs(STATIC_UNPERMUTE_TRITON_PARAMS_FWD, _h,
                                          num_warps=1, num_stages=2,
                                          maxnreg=128, BLOCK_SIZE=1024)
            )

            # NOTE (fedorovgv): should we do it with save_tensors ?
            ctx.row_map_id = row_map_id
            ctx.row_to_padded_row_id = row_to_padded_row_id
            ctx.pad_starts_t = pad_starts_t
            ctx.pad_counts_t = pad_counts_t
            ctx.total_token_rank_bucket = total_token_rank_bucket
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        with record_fn("_unpad_and_unpermute_bwd"):
            if not grad_output.numel():
                return grad_output, None, None, None, None, None

            total_tokens_padded = ctx.row_to_padded_row_id.size(0)
            hidden_size = grad_output.size(1)
            num_experts = (ctx.row_map_id.size(1) - 1) // 2

            if isinstance(grad_output, Float8BlockwiseQTensor):
                data, scale = grad_output._data, grad_output._scale
                scale_hidden_size = scale.size(1)
                # Group B: only real-token rows are written; zero padding rows via kernel.
                permuted_scale = torch.empty((total_tokens_padded, scale_hidden_size),
                                            device=scale.device,
                                            dtype=scale.dtype)
                _launch_zero_expert_padding(permuted_scale, ctx.pad_starts_t, ctx.pad_counts_t)
                grad_output_dtype = torch.uint8
            else:
                data, scale = grad_output, None
                permuted_scale = None
                grad_output_dtype = grad_output.dtype

            # Group B: _permute_and_pad_kernel only writes real-token rows;
            # zero padding rows explicitly so consumers see clean zeros there.
            grad_input = torch.empty((total_tokens_padded, hidden_size),
                                    device=grad_output.device,
                                    dtype=grad_output_dtype)
            _launch_zero_expert_padding(grad_input, ctx.pad_starts_t, ctx.pad_counts_t)

            # grid = (data.size(0), (hidden_size + ctx.BLOCK_SIZE - 1) // ctx.BLOCK_SIZE)
            _n = data.size(0)
            grid = lambda META: (_n, triton.cdiv(hidden_size, META["BLOCK_SIZE"]))

            _permute_and_pad_kernel[grid](
                *tensor_to_kernel_args(data, dims=2),
                *tensor_to_kernel_args(grad_input, dims=2),
                *tensor_to_kernel_args(ctx.row_map_id, dims=2),
                *tensor_to_kernel_args(None, dims=2), # probs
                *tensor_to_kernel_args(scale, dims=2), # scales
                *tensor_to_kernel_args(None, dims=1), # permuted probs
                *tensor_to_kernel_args(permuted_scale, dims=2), # permuted scales
                *tensor_to_kernel_args(ctx.row_to_padded_row_id, dims=1),
                num_experts=num_experts,
                total_token_rank_bucket=ctx.total_token_rank_bucket,
                PERMUTE_SCALE=(scale is not None),
                PERMUTE_PROBS=False,
                **params_to_kernel_kwargs(STATIC_PERMUTE_TRITON_PARAMS_BWD, hidden_size,
                                          num_warps=2, num_stages=2,
                                          maxnreg=128, BLOCK_SIZE=1024,
                                          BLOCKS_NUM_TO_PROCESS_IN_ROW=8)
            )

            if isinstance(grad_output, Float8BlockwiseQTensor):
                grad_input = Float8BlockwiseQTensor(
                    data=grad_input,
                    scale=permuted_scale,
                    orig_dtype=torch.bfloat16,
                    input_layout_type=InputLayoutType.ROWWISE,
                    block_scaling_dim=1,
                )
        return (grad_input,
                None,
                None,
                None,
                None,
                None,)


def permute_and_pad_fn(
    recv_x: torch.Tensor,
    local_map: torch.Tensor,
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor) -> tp.Tuple[torch.Tensor, ...]:
    """Perform fused permute and pad operation.

    Returns:
        (output, permuted_probs, row_id_map, row_to_padded_row_id,
         pad_starts_t, pad_counts_t)

        pad_starts_t / pad_counts_t: int32 tensors of shape [num_experts]
            describing the padding rows per expert in the padded output space.
            Pass these to unpad_and_unpermute_fn so its backward can avoid a
            full torch.zeros memset on the gradient buffer.
    """
    return _permute_and_pad.apply(
        recv_x,
        local_map,
        probs,
        tokens_per_expert)


def unpad_and_unpermute_fn(
    tensor: torch.Tensor,
    row_map_id: torch.Tensor,
    row_to_padded_row_id: torch.Tensor,
    restore_shape: tp.Tuple[int, ...],
    pad_starts_t: torch.Tensor,
    pad_counts_t: torch.Tensor,) -> torch.Tensor:
    """Perform fused unpad and unpermute operation.

    pad_starts_t / pad_counts_t must be the tensors returned by the matching
    permute_and_pad_fn call; they are stored in the autograd context and used
    in the backward pass to zero only the alignment-padding rows instead of
    issuing a full torch.zeros memset.
    """
    return _unpad_and_unpermute.apply(
        tensor,
        row_map_id,
        row_to_padded_row_id,
        restore_shape,
        pad_starts_t,
        pad_counts_t,)


def pad_permuted_for_grouped_gemm(
    hidden_states: torch.Tensor,
    probs: torch.Tensor,
    tokens_per_expert_list: tp.List[int],
    groups_padded_sizes_list: tp.List[int],
) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand per-expert-grouped tokens to 128-aligned groups for DeepGEMM.

    Used on the eval / inference FP8 path: there the dispatcher does not pad
    (``is_float8_training`` is False outside training), but the grouped GEMM
    still requires each expert block padded to a multiple of 128.  The input is
    already grouped by expert; this just inserts zero rows after each group.

    Returns ``(padded_hidden, padded_probs, u2p)`` where ``u2p[u]`` is the
    padded-space row for unpadded row ``u`` — reuse it with
    :func:`unpad_grouped_gemm_output` to invert the mapping on the GEMM output.
    """
    device = hidden_states.device
    num_experts = len(tokens_per_expert_list)
    total_tokens = hidden_states.shape[0]
    total_padded = sum(groups_padded_sizes_list)
    hidden_size = hidden_states.shape[1]
    ALIGN_SIZE = 128

    # u2p: unpadded row -> padded row (built on GPU from the per-expert sizes).
    tokens_per_expert_t = torch.tensor(
        tokens_per_expert_list, device=device, dtype=torch.int32)
    u2p = torch.empty((total_tokens,), device=device, dtype=torch.int32)
    _row_to_padded_row_id_kernel[(total_tokens,)](
        tokens_per_expert_t, u2p,
        num_experts=num_experts, align_size=ALIGN_SIZE)

    # Per-expert padding rows in the padded layout (CPU arithmetic; no GPU sync).
    pad_starts: tp.List[int] = []
    pad_counts: tp.List[int] = []
    off = 0
    for m, mp in zip(tokens_per_expert_list, groups_padded_sizes_list):
        pad_starts.append(off + m)
        pad_counts.append(mp - m)
        off += mp
    pad_starts_t = torch.tensor(pad_starts, device=device, dtype=torch.int32)
    pad_counts_t = torch.tensor(pad_counts, device=device, dtype=torch.int32)

    padded_hidden = torch.empty(
        (total_padded, hidden_size), device=device, dtype=hidden_states.dtype)
    _launch_zero_expert_padding(padded_hidden, pad_starts_t, pad_counts_t)
    _launch_remap_rows(hidden_states, padded_hidden, u2p, scatter=True)

    padded_probs = torch.empty(
        (total_padded,), device=device, dtype=probs.dtype)
    _launch_zero_expert_padding(padded_probs, pad_starts_t, pad_counts_t)
    _launch_remap_rows(probs, padded_probs, u2p, scatter=True)

    return padded_hidden, padded_probs, u2p


def unpad_grouped_gemm_output(
    output: torch.Tensor,
    u2p: torch.Tensor,
    total_tokens: int,
) -> torch.Tensor:
    """Inverse of :func:`pad_permuted_for_grouped_gemm` for the GEMM output.

    Gathers the real (non-padding) rows back into the compact per-expert layout
    the dispatcher expects.  ``u2p`` and ``total_tokens`` are those returned by
    the matching pad call.
    """
    out = torch.empty(
        (total_tokens, output.shape[1]), device=output.device, dtype=output.dtype)
    _launch_remap_rows(output, out, u2p, scatter=False)
    return out
