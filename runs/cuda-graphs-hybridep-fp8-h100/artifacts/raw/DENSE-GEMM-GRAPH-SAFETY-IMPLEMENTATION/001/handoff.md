# Dense FP8 GEMM graph-safety implementation

## Result and scope

Added a task-local adapter around the unchanged public CudaCherry GEMM and
switched the existing dense FP8 alias to it. The CUDA-only functional and
mutable custom ops both carry `torch.Tag.cudagraph_unsafe`. The functional op
always passes `out=None`; the mutable op declares `out` mutation and returns
`None`, while the Python adapter returns the original `out` object. Both paths
unpack public QTensor/tuple inputs before the opaque boundary, then reconstruct
the vendor four-item tuples inside the body. No SAC save rule or `FP8Gemm`
autograd implementation changed.

Changed files:

- `gift/torchtitan/ops/float8/_cudacherry_graph_safe.py`
- `gift/torchtitan/ops/float8/_cudacherry.py`
- `gift/tests/unit_tests/test_cudacherry_gemm_graph_safety.py`

The production helper and alias are frozen for independent review. Root's
reviewer reported no material production finding and passed an FX/Inductor
unsafe-node predicate probe, bound-QTensor FakeTensor probe, and CPU unsupported
device probe with CUDA uninitialized. Those probes are reviewer evidence, not
part of my pytest command.

## Verification

Source revision before this change: `446431dfa191ef0300f116257ae918bc85e38c06`.
The focused suite used `/opt/torchtitan-venv/bin/python3.12`, PyTorch
`2.15.0.dev20260822+cu130`, and installed CudaCherry `0.1.0`.

Command, from `gift/`:

```bash
CUDA_VISIBLE_DEVICES= CUDACHERRY_DISABLE_JIT=1 \
  /opt/torchtitan-venv/bin/python3.12 -m pytest \
  tests/unit_tests/test_cudacherry_gemm_graph_safety.py -q --tb=short
```

Result: **8 passed** in 4.06 seconds. The test fixture forces CUDA availability
false, device count zero, and makes both `torch.cuda.init` and `_lazy_init` fail.
Fake CUDA tensors exercise the vendor Meta path without initializing a device;
temporary CPU kernels exercise the actual outer op and public vendor call. They
are removed after each test and are not production fallbacks. Pytest emitted one
installed `pynvml` deprecation warning.

The tests cover:

- Full public signature/default parity; row/row, row/column, and column/column
  forwarding; vendor invalid-layout and accumulate-without-output errors.
- Fresh functional outputs and eager output identity, overwrite/accumulate,
  noncontiguous output strides, and mutation version changes.
- FakeTensor symbolic dimensions, input/output strides, dtype, both unsafe tags,
  and the mutable no-return schema.
- A compiled functional call whose captured FX target is the tagged outer op.
- Actual `FP8Gemm` manual dgrad/wgrad ownership, residual gradients, frozen-weight
  and disabled-activation-gradient paths, and compiled `aot_eager` loss/gradient
  parity with controlled quantizer and GEMM doubles.
- The actual selective-checkpoint policy: dense GEMM remains outside
  `_get_save_ops`, the forward GEMM is recomputed once, and dgrad/wgrad follow in
  the backward pass with the expected layouts and matching forward/recompute
  shapes.

Checks on the final source/tests:

```bash
ruff check --select F821,F811,F841 --no-cache \
  gift/torchtitan/ops/float8/_cudacherry.py \
  gift/torchtitan/ops/float8/_cudacherry_graph_safe.py \
  gift/tests/unit_tests/test_cudacherry_gemm_graph_safety.py
git diff --check
```

Both passed. A separate whitespace scan of the untracked test and helper files
found no trailing whitespace.

## Failed attempts and log retention

Three earlier focused pytest invocations failed before the final pass:

1. `-q`: **1 passed, 7 failed**. The temporary CPU implementation lacked schema
   defaults; dispatcher calls omitted default-valued trailing arguments.
2. `-q` after adding defaults: **2 passed, 6 failed**. The CPU implementation
   signature still included the internal `out` placeholder, which is absent from
   the public functional-op schema and shifted arguments.
3. `-q --tb=short` after matching that schema: **5 passed, 3 failed**. The test
   quantizer double did not transpose columnwise payloads; the residual expected
   value read the buffer after its intended in-place update; and direct
   `torch.func.functionalize` rejected the mutable custom op's alias schema in
   this installed PyTorch.

The CPU test wrapper and the two incorrect test oracles were corrected. The
direct `torch.func.functionalize` execution assertion was removed; eager mutable
behavior and the accurate mutation schema/tag remain covered. The full stdout of
those three failed invocations was not redirected and the tool view was
truncated, so only the commands, counts, and failure causes above are retained.
They were not rerun to reconstruct logs. The final successful command output is
saved in `cpu-tests.log`.

## Runtime cost and limits

The adapter adds Python `unpack_operand` calls, two short tuple reconstructions,
and one outer custom-op dispatch per dense GEMM. The unchanged vendor function
unpacks the reconstructed tuples again. These are metadata/host operations; no
device-value read, transfer, synchronization, scratch allocation, or change to
the cuBLASLt call is intended. Functional output allocation and the existing
workspace cache remain vendor-owned. The mutable route retains the supplied
buffer.

In CUDA Graph runs, the unsafe tag can partition each dense GEMM from captured
regions. That can add host submissions/graph launches, change intermediate
liveness and overlap, and reduce graph replay benefit. Calls occur in dense
forward, backward, and SAC recomputation. No production timing or memory
measurement was run, so magnitude and speedup remain unknown. Communication,
optimizer, checkpoint state, and data paths were not changed.

Compiled `out=` behavior is **not claimed**. The installed vendor GEMM documents
compiled `out=` as unsupported; the selected production preset follows the
functional route. Native FP8/cuBLASLt parity, CUDA Graph capture/replay, all-rank
execution, continuation, and performance remain root-owned runtime gates.
CPU/FakeTensor and `aot_eager` results do not close those gates.
