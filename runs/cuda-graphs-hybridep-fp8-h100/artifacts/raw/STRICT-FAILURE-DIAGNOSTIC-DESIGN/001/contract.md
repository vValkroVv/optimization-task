# One genuine sparse-wgrad replay diagnostic

## Verdict and finite acceptance

A task-entry-only interception is feasible from the inspected APIs. Implement one
sample per rank in the **first native training microbatch**, recording the first
paired cuDACHERRY row2col calls and their actual sparse wgrad consumer. Duplicate
row2col independently, then duplicate GEMM using the same immutable actual column
operands. Never substitute a replay result into native training.

Acceptance for this design: identify an exact genuine runtime hook; preserve
compile/SAC/native outputs and operand lifetimes; specify immutable inputs,
precision, memory, frequency, decisions, and implementation verification. These
items are addressed below. No implementation, CPU behavior test, GPU query,
CUDA initialization or run was performed. Strict A/A remains **FAIL**, including
after the corrected scan header: updates 2–10 differ in both required float32
metrics. See [comparison](../../SCAN-FIX-DETERMINISTIC-AA/comparison.json).

Reuse [source localization](../../DETERMINISTIC-AA/source-localization/handoff.md)
and [fault research](../../BASELINE-AA/fault-research/001/handoff.md). No new defect
or unordered reduction is established here. This experiment changes localization
only; it cannot replace native strict A/A or establish production correctness.

## Pinned boundaries and selected approach

The source identity is [SOURCE-SNAPSHOT/002](../../SOURCE-SNAPSHOT/002/manifest.json),
GIFT upstream `6fd9a3230ab6361f11bbbf0b6ec5a5f125b78fc2`. Additional inspected files
and hashes are in `source-pins.json` beside this note. Installed text is not proof
of complete extension-binary provenance.

Relevant task source (paths relative to repository root):

- `gift/torchtitan/trainer.py:613`: native `Trainer.forward_backward_step(self,
  *, input_dict, labels, global_valid_tokens)`. The microbatch loop calls this
  outside compiled model/loss. Trainer construction calls
  `setup_deep_gemm_cache` at line 154 **before model build**.
- `gift/torchtitan/ops/warmup/deep_gemm_fast_warmup.py:190–208`: startup warmup
  calls the same cuDACHERRY row2col and k-grouped wrapper on synthetic operands.
  A first-dispatch observer without an explicit training arm is invalid.
- `gift/torchtitan/ops/float8/grouped_gemm.py:204–259`: backward calls row2col
  on saved activation then output gradient, converts data to FP8 views and scale
  to `.mT`, allocates BF16 zero wgrad, and invokes k-grouped GEMM. Early
  `clear_tensor_data` follows each native operand's use. Clones must occur inside
  its runtime kernel, before returning to that caller.
- `gift/torchtitan/ops/float8/deep_gemm_handle.py:37–105`: custom op mutates only
  `d`; its Python body checks layouts/counts, reads already-host counts, handles
  empty input, then calls `deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
  (a_data,a_scale), (b_data,b_scale), d, ks, ks_tensor, d)`.

Installed package root is `/opt/torchtitan-venv/lib/python3.12/site-packages`:

- `cudacherry/quantization/row2col_dq.py`: actual wrapper normalizes rowwise data
  to contiguous uint8, scales to stride(0)=1, m-indices to contiguous int32,
  allocates FP8 flat output plus FP32 scales, then invokes
  `torch.ops.cudacherry.row2col_dq(X,S_in,group_sizes,m_indices,Y,S)`.
- `cudacherry/csrc/quantization/row2col_dq/bindings.cpp`: **both** the CUDA
  dispatcher implementation and the extension's direct pybind `row2col_dq`
  call the same C++ function. Direct call bypasses the experimental dispatcher
  shim without recursion. `row2col_kernel.cu:325` launches on current CUDA stream.
- `torch/library.py:441`: `Library.impl(..., dispatch_key="CUDA",
  allow_override=True)` can replace that CUDA implementation while retaining
  its schema and Meta kernel. Keep the Library alive for the process.
- `torch/_library/custom_ops.py:295,392–505`: CustomOpDef retains `_init_fn`
  and supports `register_kernel("cuda", fn)`. Save the original
  `_k_grouped_fp8_gemm_contiguous._init_fn` before registration; call that saved
  function directly for native and duplicate work. `_init_fn` is a **private,
  pinned implementation detail**: assert callable and signature/hash before use;
  fail closed if absent. Do not copy the wrapper body or call the overridden
  `torch.ops` recursively. Public kernel registration is documented in
  [PyTorch torch.library](https://docs.pytorch.org/docs/stable/library.html).

No new op, fake implementation, backend, precision recipe, dependency or source
workaround is needed. Existing op boundaries preserve compiled graphs while the
CUDA dispatch implementation performs runtime observation. Python monkeypatching
only the high-level `allow_in_graph` row2col wrapper would not reliably observe
compiled runtime execution.

## Exact task-entry contract

Proposed entry module, not created: `torchtitan.experiments.cuda_graph_campaign.train_wgrad_replay`.
It installs the existing scan correction, installs the experiment only when an
explicit output directory/toggle is supplied, then calls `torchtitan.train.main`.
Native non-experiment entry and disabled toggle install **nothing**.

1. Save `Trainer.forward_backward_step` (including its existing logging wrapper).
   A thin method shim arms process-local state only during its first invocation,
   calls the saved method with the exact keyword arguments, and clears the arm
   in `finally`. Store native `self.step` and microbatch ordinal, rank and PID.
   It must not modify inputs, return value, RNG, configuration or callbacks.
   After that microbatch require a complete sample or record NOT_OBSERVED and
   fail the diagnostic, rather than sampling a later synthetic/validation path.
2. Before native main/compilation, load only the installed prebuilt requantize
   extension through the existing loader (reject forced-JIT/missing-prebuilt
   rather than building another kernel); save its direct `row2col_dq`. Install
   CUDA shim with `Library("cudacherry", "IMPL").impl("row2col_dq", shim,
   "CUDA", allow_override=True)`. Save the existing sparse CustomOpDef `_init_fn`
   and register its CUDA shim with `.register_kernel("cuda", shim)`.
   Fake/Meta implementations remain unchanged and cannot increment counters.
3. Unarmed calls delegate exactly once. Armed first two row2col CUDA calls each
   delegate the real call exactly once to the saved pybind function, retain
   its real output storage identity, then run two scratch calls from one
   immutable snapshot of `(X,S_in,group_sizes,m_indices)`. All work stays on
   the caller's current CUDA device/stream; no auxiliary stream or collective.
   Take snapshots before the native call so an invalid mutating kernel is also
   detectable. Preserve dtype, shape and strides using explicit strided copies
   where needed; never retain only `.detach()` aliases. Snapshot group counts
   and indices too. Hold originals until copies have been ordered on that stream.
4. The first armed k-grouped CUDA call must match the two recorded real row2col
   output storages: `a` is the second (gradient), `b` is the first (activation);
   scales are their `.mT` views. Match data pointers, shapes, scale storage,
   device and group metadata, not just matching dimensions. An unexpected order,
   third row2col before the pair, concurrent/reentrant invocation or mismatch
   is INVALID_SAMPLE, never an inferred layer name. Record ordinal and actual
   widths. This is one genuine layer occurrence, not coverage of all layers.
5. Snapshot the actual k-grouped input planes/counts before calling the saved
   native body once with the **original arguments and original `d`**. After
   that call, invoke the same saved body twice with the immutable column planes
   and two separately zeroed BF16 scratch `d`s, identical copied host `ks` /
   pinned `ks_cpu` and cloned `ks_tensor`. Preserve `ks=None` versus list and
   the original host-count route. Do not use fresh row2col outputs to feed these
   duplicate GEMMs: differing row2col values must not confound the GEMM test.
6. Compare bytes, not floating tolerances: FP8 via uint8; FP32 scales via byte
   views after CPU contiguous conversion; BF16 wgrad via uint8/int16 bytes.
   Copy logical tensors to CPU on the current stream, then wait for that copy
   before hashing/comparing; preserve raw bit patterns including signed zero
   and NaN payloads. Record SHA-256, mismatch count/first coordinates, complete
   metadata and before/after immutable input hashes. GPU scalar extraction in
   Python is unnecessary. Compare real output versus scratch as an additional
   separately labeled observation, never overwrite it. Release scratch and
   retained references after completion; DONE then delegates remaining calls.

cuDACHERRY explicitly uses the current stream. DeepGEMM's installed binary imports
`c10::cuda::getCurrentCUDAStream` (static symbol inspection), and the native ABI
has no stream parameter. This supports the intended stream use but does not prove
every installed kernel path. Runtime output-copy/event ordering and input
immutability are validation gates; an unexplained race or late write invalidates
the sample. Do not insert device-wide synchronization into backward or silently
claim a side-stream lifetime was verified. No CUDA Graph capture is supported by
this host observer: keep the existing strict-run graph policy `0`.

## Dtypes, counts, cost and budget

Preserve row data FP8 e4m3fn/raw uint8, FP32 row/column scales and existing
integer-width metadata. Do not reconstruct scales, requantize BF16 sources,
rebalance counts, replace padding, downcast counts, or synthesize an m-index map.
Require the actual exact layout: `T == sum(ks)`, four local groups, each count
multiple of 128, indices match the real group prefixes, no unused `-1` tail.
Validate values after snapshot/copy, not via new per-call GPU reads. Invalid
layout is evidence to retain and makes this replay sample invalid.

Native output remains BF16. Scratch output is also BF16 with the original
accumulation call (`c=d`, initially zero). No output FP32 experiment or changed
recipe is permitted. The shipped SM90 1d1d header declares FP32 `accum` and
`final_accum` at line 284; using the identical installed call preserves that
accumulation path, without certifying binary-source identity.

Let `R(T,H)=T*H+4*T*H/128` bytes (FP8 data plus FP32 scales),
`Q=R(T,M)+R(T,N)`, `D=2*G*M*N`, and `J=bytes(indices)+bytes(counts)`.
Conservative additional live device storage is bounded by
`max(3*R(T,max(M,N))+2*J, Q+2*D+2*J)` plus allocator/workspace overhead:
row2col retains one input snapshot and two output pairs, one operand at a time;
GEMM retains both column snapshots and two zero outputs. CPU staging/comparison
can add several times those bytes; free each completed operand's host/device
snapshots before proceeding. Account for input post-hash copies and any strided
storage holes in implementation's byte planner. At expected first w2 widths
M=1536,N=1024,G=4 and T=32768, `Q=82.5 MiB`, `2*D=24 MiB`; this is an
illustration, **not a measured row count or headroom**. Routing may be unbalanced.

Before snapshot allocation, use actual shapes/strides to compute bytes and check
rank-local allocator/device headroom with a reviewed reserve for native live
allocations and backend workspace. If the conservative budget is unsafe, report
MEMORY_BUDGET_UNAVAILABLE and stop the diagnostic; do not shrink the workload.
No hardware/headroom was queried for this design. Do not reset native peak stats.

Frequency: at most two observed native row2col calls and one native wgrad per
rank; exactly four extra row2col launches and two extra GEMM calls per sampled
rank, plus snapshots/CPU copies/comparisons. Startup/warmup and fake calls do not
consume the budget. One-time synchronization for CPU evidence and extra allocation
perturb overlap, cache state and memory. There are no new communication calls,
optimizer operations or parameter writes; collective timing may still shift.
Installed shims add Python delegation to affected CUDA calls for this diagnostic
process, even after DONE. Disabled mode has no hooks, copies or runtime branch.
Do not use diagnostic timings as throughput evidence; no speedup is claimed.

## Required implementation checks and execution route

Before GPU execution, CPU behavior checks must exercise the actual registration
pattern with toy CPU mutable custom ops and a direct saved implementation: no
recursion, exactly one native write, scratch nonaliasing, disabled passthrough,
first-microbatch arming, startup/fake exclusion, exception/finally restoration,
DONE frequency, output-pair order validation and deliberate mismatch detection.
Exercise byte comparison with signed zero and distinct NaN payloads, strided
FP32 scales, and empty/count-layout failures; validate the conservative byte
planner against actual tensor storage sizes. Use injected toy native functions
rather than importing CUDA-only extensions or asserting source text. CPU compiled
custom-op smoke must show runtime observation with the same graph/op schema;
this does not prove CUDA/Inductor integration. Restore patched methods, kernels,
environment and registration-library lifetime in tests. Run changed-file lint.

GPU integration must confirm original preset hash and rank-zero resolved config,
all eight ranks' sample activation and original/source package hashes, first
training microbatch selection, no fake/startup sampling, actual pair linkage,
unchanged native return tensors, immutable inputs, and completed per-rank reports.
A CPU test or absent deterministic exception cannot close this gate.

Use the [existing strict launch](../../SCAN-FIX-DETERMINISTIC-AA/001/launch.sh)
with its environment, eight-rank torchrun, caches and all native arguments,
changing only output paths and the module to the proposed diagnostic entry. Add
one explicit output-directory environment toggle consumed only by that entry.
The command tail is:

```bash
-m torchtitan.experiments.cuda_graph_campaign.train_wgrad_replay \
  configs/experiments/gift_moe_3b_hybridep_fp8_compile.yaml \
  --training.steps 10 --lr_scheduler.total_steps 30000 \
  --debug.seed 42 --debug.deterministic \
  --debug.deterministic_warn_only false --checkpoint.enable false \
  --metrics.enable_tensorboard true --metrics.log_freq 1 \
  --dump_folder "$ATTEMPT_DIR/native"
```

Keep `TORCHINDUCTOR_CUDAGRAPHS=0`, existing scan correction and strict-run
checkpoint setting. Preserve EP8, FSDP2, FP8, SAC, model+loss compile,
local/global batch 8/256, S4096 and four accumulation microbatches. Do not launch
while Lane A owns GPUs or change frozen GIFT. The entry/module above does not
exist yet and this is an implementation contract, not an executable launch.

## Decision table and stop

| Valid observation on identical immutable inputs | Decision |
| --- | --- |
| Either row2col data/scale duplicate differs | Localize a failure to the installed row2col boundary before GEMM; preserve binary/source/stream provenance, do not blame GEMM for downstream differences. |
| Row2col matches; duplicate wgrad differs | Localize repeatability failure to the installed sparse GEMM boundary under this operand/stream context; investigate that boundary, with strict prerequisite still failed. |
| Both duplicate boundaries match | This one sample is repeatable. Native strict A/A remains failed; upstream input generation, cross-process algorithm choice, other layers/dense gradients, routing state and optimizer remain unresolved. |
| Input hashes change, pair mismatch, missing rank, memory/ordering issue or absent activation | Inconclusive/invalid diagnostic. Repair that concrete issue before expanding; never report a component pass. |

Independent GEMM duplicates remain interpretable if row2col duplicates fail,
because they use one immutable copy of the actual native column operands.
A mismatch proves a boundary problem in the instrumented context; it does not
alone establish the cause or prevalence of cross-process A/A divergence.
Identical same-process replay cannot exclude cross-process algorithm selection
or rare scheduling-dependent failures. Stop after this single diagnostic and
report its valid outcome; do not automatically expand to a universal observer,
new policy controls, tolerance acceptance or another A/A campaign.
