from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _adamw_parameter_update_kernel(
    param_ptr,
    exp_avg_ptr,
    variance_ptr,
    numel,
    step_size,
    sqrt_bias_correction2,
    eps,
    weight_decay,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply an Adam update with cautious weight decay without storing a mask."""
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    param = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.load(variance_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    update = exp_avg / (tl.sqrt(variance) / sqrt_bias_correction2 + eps)
    cautious_decay = tl.where(param * exp_avg >= 0.0, weight_decay * param, 0.0)
    param = param - cautious_decay - step_size * update
    tl.store(param_ptr + offsets, param, mask=mask)


def adamw_cautious_parameter_update(
    param: torch.Tensor,
    exp_avg: torch.Tensor,
    variance: torch.Tensor,
    step_size: float,
    sqrt_bias_correction2: float,
    eps: float,
    weight_decay: float,
) -> None:
    """Apply an in-place Adam update with allocation-free cautious weight decay."""
    if not (param.is_cuda and exp_avg.is_cuda and variance.is_cuda):
        raise RuntimeError('Cautious weight decay requires CUDA tensors.')
    if not (param.is_contiguous() and exp_avg.is_contiguous() and variance.is_contiguous()):
        raise RuntimeError('Cautious weight decay requires contiguous tensors.')
    if param.numel() != exp_avg.numel() or param.numel() != variance.numel():
        raise ValueError('Parameter, first moment, and variance tensors must have the same number of elements.')
    if param.numel() == 0:
        return

    block_size = 256
    grid = (triton.cdiv(param.numel(), block_size),)
    _adamw_parameter_update_kernel[grid](
        param,
        exp_avg,
        variance,
        param.numel(),
        step_size,
        sqrt_bias_correction2,
        eps,
        weight_decay,
        BLOCK_SIZE=block_size,
    )


def _get_autotune_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": 8,
                "LOWER_UPPER": 1,
            },
            num_stages=stages,
            num_warps=warps,
        )
        for bm in [64, 128]
        for bn in [64, 128]
        for bk in [64, 128]
        for stages, warps in [(2, 4), (3, 4), (3, 8)]
        if bm // bn <= 2 and bn // bm <= 2
    ]

@triton.jit
def _pid_to_block(
    pid,
    M,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Split output matrix into blocks of size (BLOCK_SIZE_M, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)

    # Map PID to a single matrix in batch
    batch_idx = pid // (num_pid_m * num_pid_n)
    pid = pid % (num_pid_m * num_pid_n)

    # Map PID to 2D grid of blocks
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    m_idx = pid_m * BLOCK_SIZE_M
    n_idx = pid_n * BLOCK_SIZE_N
    return batch_idx, m_idx, n_idx

@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "K", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def XXT_kernel(
    A_ptr, C_ptr,
    M, K,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)

def XXT(A: torch.Tensor, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = A @ A.T
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert out.size(-2) == M, "Output matrix has incorrect shape"
    assert out.size(-1) == M, "Output matrix has incorrect shape"

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = lambda meta: (
        batch_size * triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
    )
    XXT_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        K=K,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
    )
    return out

@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def ba_plus_cAA_kernel(
    A_ptr, C_ptr,
    M,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    # This is mostly duplicated from XXT_kernel, but also loads and adds a block of A
    # Performance is slightly slower than XXT_kernel, so we use two separate kernels
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < M - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < M - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    # Load block of A to add (corresponds to the current block of C)
    offs_am = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_an = n_idx + tl.arange(0, BLOCK_SIZE_N)
    a_add_ptrs = A_ptr + (offs_am[:, None] * a_stride_r + offs_an[None, :] * a_stride_c)
    a_add_mask = (offs_am[:, None] < M) & (offs_an[None, :] < M)
    a_add = tl.load(a_add_ptrs, mask=a_add_mask, other=0.0).to(tl.float32)

    # Apply alpha and beta
    accumulator *= alpha
    accumulator += a_add * beta

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)

def ba_plus_cAA(A: torch.Tensor, alpha: float, beta: float, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = alpha * A @ A.T + beta * A
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert M == K, "Input matrix must be square"
    assert out.size(-2) == M
    assert out.size(-1) == M

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = lambda meta: (
        batch_size * triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
    )
    ba_plus_cAA_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
        alpha=alpha,
        beta=beta,
    )
    return out


# Triton's libdevice sqrt_rn and div_rn flush subnormal operands and results to zero, which torch
# does not; Adam's second moments reach the subnormal range, so these use the IEEE instructions.
# A product torch rounds in a kernel of its own is never fused into a following addition here:
# mul.rn is not contracted.
@triton.jit
def sqrt_ieee(x):
    return tl.inline_asm_elementwise("sqrt.rn.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def div_ieee(a, b):
    return tl.inline_asm_elementwise("div.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def mul_ieee(a, b):
    return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1)


def float32_reciprocal(value: float) -> float:
    """The float32 reciprocal of ``value`` rounded to float32: CUDA torch divides a float32 tensor
    by a Python scalar as a product with it."""
    return float(np.float32(1.0) / np.float32(value))


# Elements reduced by one program of the metric kernels below, in tiles of _METRIC_TILE.
_METRIC_CHUNK = 65536
_METRIC_TILE = 2048

_TRITON_DTYPES = {torch.float32: tl.float32, torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}

# The metric kernels reduce the chunks of several tensors in one launch. A tensor's row of the int64
# table holds the addresses of its operands, then its element count and its first chunk; its row of
# the float32 table holds its step terms.


@triton.jit
def _chunk_owner(table_ptr, chunk, owners, ROW: tl.constexpr, FIRST: tl.constexpr, OWNERS_PAD: tl.constexpr):
    # The last tensor whose first chunk is at most `chunk`.
    rows = tl.arange(0, OWNERS_PAD)
    firsts = tl.load(table_ptr + rows * ROW + FIRST, mask=rows < owners, other=2**62)
    return tl.sum((firsts <= chunk).to(tl.int32)) - 1


def _device_tables(rows: list, scalars: list, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """The int64 and float32 tables of a launch, staged in pinned memory so that their copies do not
    wait for the queued step."""
    return (torch.tensor(rows, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True),
            torch.tensor(scalars, dtype=torch.float32, pin_memory=True).to(device, non_blocking=True))


@triton.jit
def _adam_metric_partials_kernel(
    table_ptr, scalars_ptr, partials_ptr, owners,
    CAUTIOUS: tl.constexpr, CHUNK: tl.constexpr, TILE: tl.constexpr, OWNERS_PAD: tl.constexpr,
    PARAM_DTYPE: tl.constexpr, GRAD_DTYPE: tl.constexpr, STATE_DTYPE: tl.constexpr,
):
    # One chunk's sums of squares of the moments, the parameter, the reconstructed step and the
    # gradient, and the maxima of |exp_avg| and exp_avg_sq. The step is rebuilt with the operations
    # and roundings of the torch expression it replaces. Table row: [param, grad, exp_avg,
    # exp_avg_sq, numel, first chunk]; scalars: [step size, 1 / sqrt(bias correction 2), eps, decay].
    chunk = tl.program_id(0)
    owner = _chunk_owner(table_ptr, chunk, owners, 6, 5, OWNERS_PAD)
    row = table_ptr + owner * 6
    param_ptr = tl.load(row).to(tl.pointer_type(PARAM_DTYPE))
    grad_ptr = tl.load(row + 1).to(tl.pointer_type(GRAD_DTYPE))
    exp_avg_ptr = tl.load(row + 2).to(tl.pointer_type(STATE_DTYPE))
    exp_avg_sq_ptr = tl.load(row + 3).to(tl.pointer_type(STATE_DTYPE))
    numel = tl.load(row + 4)
    start = (chunk - tl.load(row + 5)) * CHUNK
    step_size = tl.load(scalars_ptr + owner * 4)
    inverse_sqrt_bias_correction2 = tl.load(scalars_ptr + owner * 4 + 1)
    eps = tl.load(scalars_ptr + owner * 4 + 2)
    decay = tl.load(scalars_ptr + owner * 4 + 3)
    sums_m = tl.zeros([TILE], dtype=tl.float32)
    sums_v = tl.zeros([TILE], dtype=tl.float32)
    sums_p = tl.zeros([TILE], dtype=tl.float32)
    sums_s = tl.zeros([TILE], dtype=tl.float32)
    sums_g = tl.zeros([TILE], dtype=tl.float32)
    max_m = tl.zeros([TILE], dtype=tl.float32)
    max_v = tl.full([TILE], float('-inf'), dtype=tl.float32)
    # Loads of the next tiles are issued while one is reduced (the sums are unchanged).
    for tile in tl.range(0, CHUNK, TILE, num_stages=3):
        offsets = start + tile + tl.arange(0, TILE)
        mask = offsets < numel
        param = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        denom = mul_ieee(sqrt_ieee(exp_avg_sq), inverse_sqrt_bias_correction2) + eps
        step = mul_ieee(step_size, div_ieee(exp_avg, denom))
        if CAUTIOUS:
            step = step + decay * tl.where(step * param >= 0.0, param, 0.0)
        else:
            step = step + decay * param
        step = tl.where(mask, step, 0.0)
        sums_m += exp_avg * exp_avg
        sums_v += exp_avg_sq * exp_avg_sq
        sums_p += param * param
        sums_s += step * step
        sums_g += grad * grad
        max_m = tl.maximum(max_m, tl.abs(exp_avg))
        max_v = tl.maximum(max_v, tl.where(mask, exp_avg_sq, float('-inf')))
    base = partials_ptr + chunk * 7
    sum_m = tl.sum(sums_m)
    sum_v = tl.sum(sums_v)
    tl.store(base + 0, sum_m)
    tl.store(base + 1, sum_v)
    tl.store(base + 2, tl.sum(sums_p))
    tl.store(base + 3, tl.sum(sums_s))
    tl.store(base + 4, tl.sum(sums_g))
    # tl.max drops NaN, which torch's amax propagates; a sum of squares is NaN exactly when an element is.
    tl.store(base + 5, tl.where(sum_m != sum_m, sum_m, tl.max(max_m)))
    tl.store(base + 6, tl.where(sum_v != sum_v, sum_v, tl.max(max_v)))


def adam_metric_sums(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step_size: float,
    sqrt_bias_correction2: float,
    eps: float,
    decay: float,
    cautious: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Float64 sums of squares of exp_avg, exp_avg_sq, the parameter, the Adam step (with its
    decoupled decay ``decay * param``, only where step and parameter agree in sign if
    ``cautious``) and the gradient, and the float32 maxima of |exp_avg| and exp_avg_sq, in one
    pass; chunk partials are combined in float64. The step is computed in float32, as torch
    computes it for float32 optimizer state."""
    sums, maxima = adam_metric_sums_batch(
        [(param, grad, exp_avg, exp_avg_sq, step_size, sqrt_bias_correction2, eps, decay)], cautious,
    )
    return sums[0], maxima[0]


def adam_metric_sums_batch(entries: Sequence[tuple], cautious: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """``adam_metric_sums`` of each (param, grad, exp_avg, exp_avg_sq, step_size,
    sqrt_bias_correction2, eps, decay) entry as rows of [entries, 5] and [entries, 2], every entry's
    chunks reduced in one launch per dtype combination. Each entry's chunk partials are combined on
    their own, so its row is what a batch of it alone gives."""
    device = entries[0][0].device
    sums = torch.empty(len(entries), 5, dtype=torch.float64, device=device)
    maxima = torch.empty(len(entries), 2, dtype=torch.float32, device=device)
    groups = {}
    for index, entry in enumerate(entries):
        groups.setdefault((entry[0].dtype, entry[1].dtype, entry[2].dtype), []).append(index)
    for (param_dtype, grad_dtype, state_dtype), indices in groups.items():
        rows, scalars, counts = [], [], []
        for index in indices:
            param, grad, exp_avg, exp_avg_sq, step_size, sqrt_bias_correction2, eps, decay = entries[index]
            numel = param.numel()
            rows.append([param.data_ptr(), grad.data_ptr(), exp_avg.data_ptr(), exp_avg_sq.data_ptr(), numel,
                         sum(counts)])
            scalars.append([step_size, float32_reciprocal(sqrt_bias_correction2), eps, decay])
            counts.append(triton.cdiv(numel, _METRIC_CHUNK))
        table, scalar_table = _device_tables(rows, scalars, device)
        partials = torch.empty(sum(counts), 7, dtype=torch.float32, device=device)
        _adam_metric_partials_kernel[(sum(counts),)](
            table, scalar_table, partials, len(indices),
            CAUTIOUS=cautious, CHUNK=_METRIC_CHUNK, TILE=_METRIC_TILE, OWNERS_PAD=triton.next_power_of_2(len(indices)),
            PARAM_DTYPE=_TRITON_DTYPES[param_dtype], GRAD_DTYPE=_TRITON_DTYPES[grad_dtype],
            STATE_DTYPE=_TRITON_DTYPES[state_dtype],
        )
        lengths = torch.tensor(counts, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
        rows_of = torch.tensor(indices, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
        sums[rows_of] = torch.segment_reduce(partials[:, :5].double(), 'sum', lengths=lengths, unsafe=True)
        maxima[rows_of] = torch.segment_reduce(partials[:, 5:], 'max', lengths=lengths, unsafe=True)
    return sums, maxima


@triton.jit
def _slice_square_partials_kernel(
    table_ptr, scalars_ptr, starts_ptr, ends_ptr, partials_ptr, owners,
    HAS_UPDATE: tl.constexpr, TILE: tl.constexpr, OWNERS_PAD: tl.constexpr,
    PARAM_DTYPE: tl.constexpr, GRAD_DTYPE: tl.constexpr, STATE_DTYPE: tl.constexpr,
):
    # One chunk of one slice: sums of squares of the parameter, the full Adam update (with its
    # decoupled decay) and the gradient. Table row: [param, grad, exp_avg, exp_avg_sq, has grad,
    # first chunk]; scalars: [step size, 1 / sqrt(bias correction 2), eps, decay].
    chunk = tl.program_id(0)
    owner = _chunk_owner(table_ptr, chunk, owners, 6, 5, OWNERS_PAD)
    row = table_ptr + owner * 6
    param_ptr = tl.load(row).to(tl.pointer_type(PARAM_DTYPE))
    grad_ptr = tl.load(row + 1).to(tl.pointer_type(GRAD_DTYPE))
    exp_avg_ptr = tl.load(row + 2).to(tl.pointer_type(STATE_DTYPE))
    exp_avg_sq_ptr = tl.load(row + 3).to(tl.pointer_type(STATE_DTYPE))
    has_grad = tl.load(row + 4) != 0
    step_size = tl.load(scalars_ptr + owner * 4)
    inverse_sqrt_bias_correction2 = tl.load(scalars_ptr + owner * 4 + 1)
    eps = tl.load(scalars_ptr + owner * 4 + 2)
    decay = tl.load(scalars_ptr + owner * 4 + 3)
    start = tl.load(starts_ptr + chunk)
    end = tl.load(ends_ptr + chunk)
    sums_p = tl.zeros([TILE], dtype=tl.float32)
    sums_u = tl.zeros([TILE], dtype=tl.float32)
    sums_g = tl.zeros([TILE], dtype=tl.float32)
    # Loads of the next tiles are issued while one is reduced (the sums are unchanged).
    for tile in tl.range(start, end, TILE, num_stages=3):
        offsets = tile + tl.arange(0, TILE)
        mask = offsets < end
        param = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sums_p += param * param
        if HAS_UPDATE:
            exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            denom = mul_ieee(sqrt_ieee(exp_avg_sq), inverse_sqrt_bias_correction2) + eps
            update = tl.where(mask, mul_ieee(step_size, div_ieee(exp_avg, denom)) + decay * param, 0.0)
            sums_u += update * update
        if has_grad:
            grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            sums_g += grad * grad
    base = partials_ptr + chunk * 3
    tl.store(base + 0, tl.sum(sums_p))
    tl.store(base + 1, tl.sum(sums_u))
    tl.store(base + 2, tl.sum(sums_g))


def slice_square_sums(
    param: torch.Tensor,
    grad: Optional[torch.Tensor],
    exp_avg: Optional[torch.Tensor],
    exp_avg_sq: Optional[torch.Tensor],
    chunks: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    step_size: float,
    sqrt_bias_correction2: float,
    eps: float,
    decay: float,
) -> torch.Tensor:
    """Per slice, float64 [sum param^2, sum update^2, sum grad^2] in one pass, where the update is
    ``step_size * exp_avg / (sqrt(exp_avg_sq) / sqrt_bias_correction2 + eps) + decay * param``
    (zero without optimizer state) and the gradient term is zero without a gradient. ``chunks`` are
    the slices' ``slice_chunks``."""
    return slice_square_sums_batch(
        [(param, grad, exp_avg, exp_avg_sq, chunks, step_size, sqrt_bias_correction2, eps, decay)],
    )


def slice_square_sums_batch(entries: Sequence[tuple]) -> torch.Tensor:
    """``slice_square_sums`` of each (param, grad, exp_avg, exp_avg_sq, chunks, step_size,
    sqrt_bias_correction2, eps, decay) entry, the slices of all entries in order as rows of one
    [slices, 3] tensor; every entry's chunks are reduced in one launch per dtype combination, and
    each slice's chunk partials are combined on their own, as for the entry alone. Either every
    entry has optimizer state or none has."""
    device = entries[0][0].device
    has_update = entries[0][2] is not None
    groups = {}
    for index, (param, grad, exp_avg, *_) in enumerate(entries):
        groups.setdefault((param.dtype, param.dtype if grad is None else grad.dtype,
                           param.dtype if exp_avg is None else exp_avg.dtype), []).append(index)
    slices = [entry[4][2].numel() for entry in entries]
    firsts = [sum(slices[:index]) for index in range(len(entries))]
    sums = torch.empty(sum(slices), 3, dtype=torch.float64, device=device)
    for (param_dtype, grad_dtype, state_dtype), indices in groups.items():
        rows, scalars, chunk_count = [], [], 0
        for index in indices:
            param, grad, exp_avg, exp_avg_sq, chunks, step_size, sqrt_bias_correction2, eps, decay = entries[index]
            assert (exp_avg is not None) == has_update, 'every entry has optimizer state or none has'
            address = param.data_ptr()
            rows.append([address, address if grad is None else grad.data_ptr(),
                         address if exp_avg is None else exp_avg.data_ptr(),
                         address if exp_avg_sq is None else exp_avg_sq.data_ptr(), int(grad is not None), chunk_count])
            scalars.append([step_size, float32_reciprocal(sqrt_bias_correction2), eps, decay])
            chunk_count += chunks[0].numel()
        table, scalar_table = _device_tables(rows, scalars, device)
        starts, ends, chunks_per_slice = (torch.cat([entries[index][4][part] for index in indices]) for part in range(3))
        partials = torch.empty(chunk_count, 3, dtype=torch.float32, device=device)
        _slice_square_partials_kernel[(chunk_count,)](
            table, scalar_table, starts, ends, partials, len(indices),
            HAS_UPDATE=has_update, TILE=_METRIC_TILE, OWNERS_PAD=triton.next_power_of_2(len(indices)),
            PARAM_DTYPE=_TRITON_DTYPES[param_dtype], GRAD_DTYPE=_TRITON_DTYPES[grad_dtype],
            STATE_DTYPE=_TRITON_DTYPES[state_dtype],
        )
        group_sums = torch.segment_reduce(partials.double(), 'sum', lengths=chunks_per_slice, unsafe=True)
        rows_of = [row for index in indices for row in range(firsts[index], firsts[index] + slices[index])]
        sums[torch.tensor(rows_of, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)] = group_sums
    return sums


def slice_chunks(bounds: Sequence[Tuple[int, int]], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The ``(start, end)`` slices as chunks of at most _METRIC_CHUNK elements on ``device``: their
    starts and ends, ordered by slice, and the chunk count per slice."""
    starts, ends, counts = [], [], []
    for start, end in bounds:
        chunk_starts = range(start, end, _METRIC_CHUNK)
        starts.extend(chunk_starts)
        ends.extend(min(chunk + _METRIC_CHUNK, end) for chunk in chunk_starts)
        counts.append(len(chunk_starts))
    return tuple(torch.tensor(values, dtype=torch.int64, device=device) for values in (starts, ends, counts))
