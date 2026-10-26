---
description: Rules for integer counters, accumulators and kernel atomics
globs: torchtitan/**
---

# Integer Overflow Rules

int32 tops out at 2,147,483,647 (~2.1e9), and production sizes get there
fast. One MoE expert rank can receive ~256k routed rows per forward (seen on
SFT traces), and 262,144 rows x 8,192 hidden is exactly 2**31. Summing over
ranks, microbatches, steps or experts multiplies that further. An integer
overflow does not raise: the value wraps, and the metric or index just becomes
wrong.

## Default To int64
- Counters and accumulators are int64: anything that counts elements, tokens
  or events, and anything summed over ranks, microbatches, steps or experts.
- Use int32 only where an API requires it (grouped-GEMM offsets, kernel index
  arguments, an external kernel's contract). When you do, state the worst-case
  value in a comment next to it.

## Prove The Bound Before Choosing int32
Write the worst case as a product of production sizes, e.g.
`rows_per_rank x hidden x microbatches x ranks`. If it is unknown, or not at
least ~10x below 2**31, use int64. "Fits in the debug model" proves nothing.

## Silent-Narrowing Traps
- `x.sum().to(torch.int32)` narrows. A bool/int sum is already int64, so write
  `x.sum(dtype=torch.int64)` and never cast down.
- In-place ops keep the destination dtype: `int32_buf.add_(int64_tensor_0d)`
  wraps silently, because a 0-dim tensor takes part in type promotion like a
  scalar.
- `dist.all_reduce` sums in the buffer's dtype: `num_ranks x local_count` in
  int32 overflows. Reduce counts in int64 (or float64).
- float32 is exact only up to 2**24 (~16.7M). Convert counts to float64 when
  an exact count or ratio matters.
- `torch.cumsum(..., dtype=torch.int32)` for offsets is fine only while the
  total row count stays below 2**31. Check it where rows can grow.

## Kernels (Triton / CUDA)
- The pointer's dtype sets the atomic's width. Counters use int64 atomics:
  `tl.atomic_add` on an int64 pointer, `atomicAdd(unsigned long long*)` in CUDA.
- Triton specializes on whatever buffer it is handed and would compile an int32
  atomic for an int32 buffer, so validate counter dtypes in the Python wrapper
  and raise.
- Compute row/element offsets in int64 (`.to(tl.int64)`) whenever
  `rows x stride` can pass 2**31.
- Prefer widening an external kernel to int64 over working around it. If it
  must stay int32 for now, pass a per-call int32 scratch buffer, raise if one
  call can exceed 2**31, accumulate into an int64 buffer, and leave a TODO to
  widen the kernel.

## Tests
- Pre-fill the buffer at `torch.iinfo(torch.int32).max` and assert the running
  total stays exact after the op.
- Assert that kernel wrappers refuse a counter buffer of the wrong dtype.

## Review
For every new integer tensor, buffer or collective in a diff: what is its
worst-case value at production sizes, and is that written down? Grep the diff
for `int32`, `.to(torch.int32)`, `tl.int32`, `int*` in kernels, and in-place
adds into integer buffers.
