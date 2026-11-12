# Independent dense FP8 GEMM safety review

## Result and acceptance

Source-level integration and runtime-cost review is complete. No material
production-source finding identified in the bounded review. The
adapter follows the accepted [design](../../DENSE-GEMM-GRAPH-SAFETY-DESIGN/001/handoff.md).
This is a source/integration review with CPU/FakeTensor evidence, not production
acceptance or a capture, numerical-equivalence, or speedup claim.

Finite checklist: inspect adapter/vendor argument and precision contracts;
trace actual FP8Gemm manual backward and SAC consumers; verify unsafe-op target
recognition and QTensor/FakeTensor/CPU behavior; review mutation and runtime cost;
reconcile final owner tests and remaining native gates. Unrelated campaign code
is outside this review.

Repository HEAD: `446431dfa191ef0300f116257ae918bc85e38c06` (dirty task tree).
Interpreter: `/opt/torchtitan-venv/bin/python3.12`; Torch
`2.15.0.dev20260822+cu130`, reported revision
`8b353065ce260336c5fdfa74d754e8bd9b91b062`.

## Integration and numerical contracts

- `_cudacherry.py` changes only its lazy GEMM binding. The QTensor-class guard
  and quantizer aliases remain intact. Registration is confined to the sibling
  helper; no vendor operator, library registration, or global dispatcher alias
  is replaced. Non-FP8 imports retain the existing lazy path.
- `fp8_gemm` preserves the vendor parameter names/defaults and uses the vendor's
  exported `unpack_operand`. Data, scale, integer layout and block dimensions
  cross the opaque boundary; the body reconstructs the same tuple ABI and calls
  the separately bound original public vendor function. Its validation,
  transpose selection, default dtype, alpha/beta, TF32 and workspace request
  remain active. Required split accumulation remains true in all actual callers;
  the vendor CUDA body still rejects false. No arithmetic, cast, quantization,
  scale rounding, accumulation precision or index width is changed.
- The functional op always supplies `out=None`; vendor output is fresh. The
  separate mutable op declares exactly `out` mutable and returns None, avoiding
  a custom-op output alias. The shim returns the caller's original out.
  `torch.Tag.out` is correctly absent because accumulation reads that buffer.
  The inherited vendor schema's incorrect out declaration is hidden beneath the
  truthful outer mutation declaration; eager overwrite/accumulate is retained.
  Compiled residual/out support was already unsupported and is not an added
  production contract for this preset (all actual preset GEMMs have out=None).
- Fake registrations delegate shape/dtype/stride semantics to the installed
  public vendor Meta path. They add no real allocation or CPU fallback. This
  deliberately preserves inherited Meta behavior rather than independently
  inventing layout or symbolic-shape rules.
- `cuda_gemm.py:FP8Gemm` remains the autograd owner. Its forward, dgrad and wgrad
  all import the same adapter. Save/restore metadata, padding, frozen weights,
  activation-gradient decisions, residual gradient and clear behavior are
  unchanged. No new raw-FP8/scale autograd formula competes with that Function.
  Optimizer, model/optimizer checkpoint schemas, parameter ownership, evaluation
  and persisted state have no changed consumer or producer in this diff.
- `activation_checkpoint.py:_get_save_ops` contains neither old nor new dense
  GEMMs; its counter recognizes only aten.mm/aten.linear. All these dense ops
  retain PREFER_RECOMPUTE. The real selected SAC policy must remain covered by
  owner behavior tests, including nested vendor events. There is no MUST_SAVE
  workaround, quantizer change, or communication-policy change.
- Actual `make_fx` tracing retains one outer
  `torchtitan.fp8_dense_gemm.default` node. The installed
  `torch._inductor.utils.is_cudagraph_unsafe_fx_node` recognizes that actual node.
  `scheduler.py:should_partition` uses this predicate for GPU fallback/extern
  nodes. Both task ops carry the public unsafe tag. This establishes the source
  route to the real consumer, not a claim that native partition capture ran.

## Final runtime-cost review

The original vendor GEMM was already opaque. The new production path adds one
outer custom-op dispatch, unpacking metadata before it and rebuilding tuples
inside it; vendor unpacking and kernel selection remain. No extra GEMM,
quantization pass, tensor clone, intermediate tensor, device-value read,
synchronization, transfer, collective, logging or conditional precision path is
introduced. Functional output allocation and cached vendor workspace remain
owned by the vendor. There is no new per-token numerical computation.

The material intended cost is graph partitioning: dense GEMMs leave replay,
adding host submissions/launches and potentially changing intermediate lifetimes,
workspace stream-cache entries and overlap. This occurs for forward, required
manual backward calls and SAC recomputation, across each accumulation microbatch.
No total launch count is inferred from the first block. Unused compiled mutable
routes may introduce functionalization copies; they are not the target preset.
No measured claim is made for device compute, host work, waits, copies/communication,
integrated latency, or peak memory. Those scopes remain native runtime gates.

The proposed exclusion is appropriately limited to the address-linked
CudaCherry workspace owner. The backend cache still allocates hidden persistent
scratch on stream-key misses, and its capture guard remains unchanged. Only
actual runtime partition evidence can establish that these allocations now occur
outside private-pool warmup/capture. Removing all graphs or bypassing the guard
would not satisfy this contract.

## Verification and failures

Independent command, exit 0:

```sh
PYTHONPATH=gift /opt/torchtitan-venv/bin/python3.12 runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/DENSE-GEMM-GRAPH-SAFETY-REVIEW/001/probe.py
```

[probe.py](probe.py) and [probe.log](probe.log) show real FX target/unsafe
classification, actual bound QTensor FakeTensor inputs producing `(128, 256)`
fp32 output, and unsupported production CPU dispatch. CUDA availability/count
are overridden and init/lazy-init forbidden before task imports. CUDA remained
uninitialized. A package deprecation warning is retained. No GPU query, launch,
training, install or benchmark was performed by this reviewer.

`git diff --check -- gift/torchtitan/ops/float8/_cudacherry.py` passed. Initial
read commands used the wrong evidence-relative path and package module layout;
they failed without mutation, then were corrected to the supplied task evidence
root and installed package paths. These were inspection errors, not test results.

The retained owner [cpu-tests.log](../../DENSE-GEMM-GRAPH-SAFETY-IMPLEMENTATION/001/cpu-tests.log)
reports **8 passed, 1 warning in 4.06s**. Inspected final test SHA-256 is
`0f76f342b00b7a7d7e0ad0d6ce7482ea68d5ba652f73f60052606eefdf9f6cef`.
The suite covers public signature/default/layout forwarding, eager mutable
overwrite/accumulate and version/identity, FakeTensor symbolic/strided metadata,
actual compiled outer target, eager FP8Gemm elementwise gradients, residual and
frozen/disabled-gradient branches, actual selected SAC recomputation, and
eager/AOT FP8Gemm loss and elementwise gradient parity. Temporary CPU backend and
quantizer doubles prove plumbing, not real FP8 kernels. The final oracle fixes
transpose payloads as required by the actual columnwise ABI and clone the
residual baseline before inherited in-place accumulation. Production hashes
remained unchanged. The compiled-target check would fail for the original direct
vendor alias, though an explicit before-patch red run was not executed/reported.

Owner reported earlier counts of 1 passed/7 failed, 2 passed/6 failed and
5 passed/3 failed. Their first three commands were not redirected and tool output
was truncated: complete raw failed-run logs are unavailable. Do not imply they
were retained or recreate those failures. The known final three causes were two
test-oracle errors and installed `torch.func.functionalize` rejecting the nested
vendor op's mutable-alias behavior. That proposed functionalize test was removed
because compiled out was already unsupported; eager mutation remains tested.
There is no new compiled-out acceptance claim. Owner's exact historical command
and failure-accounting handoff, and owner lint results, were not present when
this review was finalized; the parent must reconcile them before its final
implementation report. No further tests were repeated for this review.

## Remaining acceptance gates

1. Parent reconciliation of owner's final handoff/lint and historical failed-run
   command accounting. The source-level behavior review itself is complete;
   focused tests and independent probe passed, with the retention limit above.
2. Native same-source graph0/graph1, all eight ranks, exact preset (Flex, SAC,
   FP8, HybridEP, FSDP2, EP8, model/loss compile, local/global batch 8/256,
   S4096, accumulation 4), at least 25 real optimizer completions and update
   transitions, no guard bypass. Confirm unsafe dense ops outside captured
   private-pool regions and eligible partitions actually capture/replay.
3. Actual FP8 loss and elementwise pre/post-clipping gradients, clipping
   norm/scale, optimizer moments and parameter updates against baseline with
   agreed tolerances and measured A/A variation. Test doubles do not close this.
4. Required checkpoint/resume/RNG/data-position continuation and the separately
   retained prior update33/native71 illegal-instruction finding.
5. Representative unprofiled A/A and alternating controls, stable step series,
   paired memory, and separated host/device/wait/copy/communication/integrated
   scopes. No performance or production-readiness claim before those gates.

## Content pins inspected

SHA-256 values pin this review's production and inherited-path files:

| File | SHA-256 |
| --- | --- |
| `gift/torchtitan/ops/float8/_cudacherry_graph_safe.py` | `6ec39b228463b10af34ec0c4431b502b61bceae42bda75060c2fa65458d235c6` |
| `gift/torchtitan/ops/float8/_cudacherry.py` | `df1d07bd9e5733b93ea6931ed0022d7fa356bc4f3f3543eb70a90f294fbbfb50` |
| `gift/torchtitan/ops/float8/cuda_gemm.py` | `e4f8e9249ee13951ee4ff7de9f87c641a272157a3eb8fbbb59de1d4ec6304f13` |
| `gift/torchtitan/distributed/activation_checkpoint.py` | `2beca1d3bed90a98178658a02b7a5b93b659fbd2d7a305d53426341a2e2f99de` |
| Installed `cudacherry/gemm/gemm.py` | `52c407671ebb68895aaa1e60bb5e9435221dba81f9c1863554eec8ad730a7bd1` |
| Installed `cudacherry/tensor/binding.py` | `019855ebfea56f2bd2ee00e75a1432841bcfd0ee2d81363624be835fcbbe6f69` |
| Installed `cudacherry/csrc/gemm/gemm.cu` | `8edc23dd605285473fa28324312a5dc7c33cca735f24e391e911430d770c51bd` |
| Installed `cudacherry/csrc/gemm/bindings.cpp` | `23def151275dc625efb339304c61d07bf1d4121fee2048d1b262662e30eb4b35` |
| Installed `torch/_inductor/utils.py` | `439806f149f955d1f9123e0a0862dd0425dd734f230ab16f8d4bb53b089409ad` |
| Installed `torch/_inductor/scheduler.py` | `541cd92e71740b411a1d2f6935c551a50bf819deb650f3e45181e3ab3888603b` |

Source hashes are not a source-to-binary attestation; native evidence must identify
the actual loaded extension. No unrelated campaign code was re-audited.

## Coordinator reconciliation after finalization

The final implementation handoff is now available: all eight focused tests,
required Ruff F821/F811/F841 and whitespace checks passed. Its three earlier
failed commands/counts/causes and missing raw-output retention are explicit.
Production review hashes remain unchanged. No further review or test repeat
was needed; native capture and other runtime gates remain open.
