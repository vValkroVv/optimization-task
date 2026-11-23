"""Replay-only donor reduction matching the pinned Hopper FP32 TMA add."""

import triton
import triton.language as tl


@triton.jit
def _owner_add_ftz_kernel(
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
    # CUDA 13.0 Hopper bulk FP32 reduction uses RN and flushes subnormals.
    updated = tl.inline_asm_elementwise(
        "add.rn.ftz.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[previous, values.to(tl.float32)],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(destination, updated, columns < width)


def owner_add_bf16_(gradient, local_rows, values) -> None:
    """Add one donor's unique BF16 rows, with callers launching in rank order."""
    num_rows, width = values.shape
    if not num_rows:
        return
    _owner_add_ftz_kernel[(num_rows, triton.cdiv(width, 1024))](
        gradient, local_rows, values, width, 1024, num_warps=4
    )
