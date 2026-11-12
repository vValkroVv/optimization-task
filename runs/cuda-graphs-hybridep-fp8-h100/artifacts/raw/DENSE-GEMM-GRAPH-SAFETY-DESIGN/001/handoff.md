# Dense FP8 GEMM graph safety design

## Result and finite acceptance

Select a task-owned opaque GEMM adapter registered through public
`torch.library.custom_op(..., tags=(torch.Tag.cudagraph_unsafe,))`, retaining the
installed CudaCherry GEMM implementation. Keep Inductor graph partitioning enabled
so only actual dense GEMM dispatches execute outside CUDA Graphs. Do not patch
installed registration or Trees. The inspected installation supports this API;
end-to-end compatibility remains an implementation/runtime gate.

Acceptance of this design: establish the offending owner; inspect the installed
public API and compiler consumers; name the smallest integration and numerical,
autograd, fake, mutation and SAC contracts; define finite behavior/runtime gates
and expected cost. These design items are complete. No production source edits,
package changes, GPU calls, tests, training or benchmarks were performed. Only
this directory was written. This is an implementation contract, not a verified
implementation or speedup claim.

## Corrected ownership evidence and scope

A003 failed during the first dense block's forward warmup, before HybridEP.
A004's interleaved supervisor excerpt was inconclusive, but its retained rank-3
`torchrun/01a36e84-c7c0-45b8-aa67-bebf178f6694_cpo7bqkf/attempt_0/3/error.json`
under `LANE-A/attempts/004` contains the full allocation history. The history for
bad address `597168226304` explicitly names
`cuda_gemm::(anonymous namespace)::get_cached_workspace` followed by
`cuda_gemm::gemm`. This is address-linked producer evidence. It supersedes the
preliminary statement that A004 had only unsymbolized frames; no second history
run is needed. This design inspected rank 3, not an independently verified
all-eight allocation-history census.

Installed CudaCherry `gemm.cu:234` retains a Tensor in a thread-local
(device, stream) cache and allocates `workspace_size + 256` bytes on a miss.
The default request is 33554688 bytes. Its capture check does not reject Trees'
noncapturing private-pool warmup. The persistent scratch is absent from GEMM
outputs. Excluding GEMM from capture avoids allocating this hidden persistent
scratch in captured partitions; it does not change or expose the cache itself.

Preserve exact preset SHA-256
`b341701bd1eef7774c110fbf7ed69cf6280ba8691100b05be457dee3e7d50a7b`:
3B MoE, FlexAttention, FP8 recipe, HybridEP, EP8/FSDP2, selective AC, model and
loss compile, local/global batch 8/256, S4096, four accumulation microbatches,
loss/data/optimizer and 30000-step scheduler horizon. Preserve existing native
full-update `torch.compiler.cudagraph_mark_step_begin` placement. Frozen source003
remains unchanged by this design. FlashAttention and extra fusion are unrelated.

## Established APIs and selected boundary

[PyTorch Trees documentation](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html#cudagraph-unsafe-custom-ops)
describes unsafe custom-op tags and partitioning. Installed
`torch/_inductor/utils.py:is_cudagraph_unsafe_fx_node` recognizes tagged
`OpOverload` targets; `scheduler.py` excludes them from captured partitions.
Installed `config.py:598` enables graph partitioning by default outside fbcode,
with `TORCHINDUCTOR_GRAPH_PARTITION`; verify the resolved value in execution.
Without partitioning the containing graph can be skipped, which does not satisfy
the requested narrow exclusion.

[The public custom-op API](https://docs.pytorch.org/docs/2.15/library.html#torch.library.custom_op)
accepts tags, fake implementations and mutation declarations. Installed
`custom_ops.py` has the same `tags` parameter; inspecting the actual interpreter
confirmed `torch.Tag.cudagraph_unsafe`. The public registration API defines a
new operator; no inspected public API adds tags to an already registered vendor
operator. Do not mutate `.tags`, private registration/manager objects, or replace
`torch.ops.cudacherry.gemm` globally.

The task's sole dense entry point is
`gift/torchtitan/ops/float8/_cudacherry.py:56`, currently an alias of public
`cudacherry.gemm.gemm`. Its only task callers are `FP8Gemm.forward` and
`FP8Gemm.backward` in `cuda_gemm.py` (forward, dgrad and wgrad). Grouped experts
use DeepGEMM/HybridEP, not this alias. Replace that alias with a thin shim, placing
registration in one sibling FP8 helper if needed. Keep the existing lazy import
and QTensor binding check. No trainer, attention, expert or compiler fork is
required. This is a backend capture-safety declaration in both graph0 and graph1;
use the same candidate source for the controls. No new generic environment switch
or graph-dependent arithmetic branch is needed.

### Exact adapter contract

1. Keep the full existing `fp8_gemm(a, b, out=None, out_dtype=None, alpha=1.0,
   accumulate=False, use_split_accumulator=True, allow_tf32=True,
   workspace_size=32*1024*1024)` signature and defaults. Use exported
   `cudacherry.tensor.unpack_operand` before the opaque boundary for each operand.
   It supports the configured QTensor and the four-item tuple ABI and normalizes
   layout/block dimensions. Pass raw data and scale Tensors plus int layout/block
   values through the custom-op schema. Do not put a QTensor wrapper or arbitrary
   Python object in an opaque schema.
2. Inside the backend body, reconstruct the same two four-item tuples and call a
   separately bound original public `cudacherry.gemm.gemm` with all original
   arguments. This preserves vendor validation, transpose selection, alpha/beta,
   dtype/default dtype, split accumulation, TF32 flag, workspace size, descriptor
   and algorithm selection and actual `torch.ops.cudacherry.gemm` CUDA dispatch.
   Do not call a pybind GEMM entry point, vendor private `_transpose_flags`, or
   reimplement precision/quantization. The vendor body remains nested behind the
   opaque outer op; it is not traced away into an untagged raw GEMM.
3. Register functional `torchtitan::fp8_dense_gemm` with `mutates_args=()`,
   `device_types='cuda'`, the unsafe tag, and a fresh Tensor return. This variant
   always delegates with `out=None`. Preserve the vendor's error when
   `accumulate=True` lacks an out buffer. Every actual preset forward/dgrad/wgrad
   uses this functional route.
4. Preserve supplied-output callers with a second opaque mutable variant,
   `torchtitan::fp8_dense_gemm_mutate`, `mutates_args={'out'}`, same unsafe tag,
   explicit nonoptional Tensor out, and `-> None`. Delegate with the original out
   and accumulate flag, ignore the aliased vendor return, and return None. The
   Python shim then returns its original out Tensor. This preserves eager buffer
   identity and beta=0/beta=1 mutation. Do not describe this as a functional op or
   return an input alias from an ordinary custom_op. Do not use `torch.Tag.out`:
   [its public contract](https://docs.pytorch.org/docs/2.15/library.html#torch.library.custom_op)
   prohibits reading output tensors, whereas beta=1 reads the residual. A second
   variant is needed only to retain this existing API truthfully; it adds no
   functional-preset kernel or copy. The installed vendor already documents
   compiled out as unsupported because its own schema omits mutation. Do not
   silently narrow the eager API or claim this design has verified compiled
   residual support. The new accurate outer mutation schema can support that path
   only after the listed behavior gates; functionalization may insert copies.
5. Functional fake registration must call the original vendor public GEMM on the
   reconstructed FakeTensor tuples, returning its Meta-produced fake result. This
   reuses installed shape/stride/dtype semantics, including symbolic dimensions,
   without workspace allocation. For mutable fake registration, call the vendor
   fake path with out and return None; mutation metadata belongs to the outer
   schema. Never allocate CUDA scratch or manually substitute CPU fake tensors.
6. Keep existing `FP8Gemm` autograd.Function, quantizers, save/restore metadata,
   padding/unpadding, residual gradient, frozen-weight and activation-gradient
   decisions unchanged. They own FP8 training's manual backward and invoke the
   same adapter for dgrad/wgrad. Do not invent gradients for raw quantized data or
   scales, add detach, cast, copy, fallback GEMM, straight-through gradient, or a
   second competing autograd formula. The raw vendor GEMM has no native registered
   autograd formula. The opaque adapter is an internal no-grad backend primitive
   beneath the existing Function, not a newly differentiable public GEMM. Tests
   must prove AOTAutograd preserves this ownership; unsupported higher-order
   differentiation is not established by this design.

### Why tags and SAC can remain correct

A functional custom_op is an opaque dispatcher target under AOT/Inductor. The tag
is attached to that outer target; the inner vendor op does not need to inherit it.
Fake code is used only for metadata. Generated forward and backward code must
show the outer op, and the partition plan must exclude it. Do not accept a source
string/tag assertion as proof that an actual compiled call is protected.

For the mutable variant, installed `post_grad.py:1585`
`decompose_auto_functionalized` decomposes functionalization wrappers into clones
and the underlying mutable op; its end asserts the auto-functionalized HOPs are
removed. This supports eventual recognition of the tagged underlying target,
but compiled mutation/tag preservation still needs a behavior test.

The target `activation_checkpoint.py:_get_save_ops` contains neither
`cudacherry.gemm.default` nor the proposed ops. Its mm counter only recognizes
aten.mm/aten.linear. Both old and new dense boundaries therefore fall through to
PREFER_RECOMPUTE and do not increment mm counters. Keep that policy; do not add
the new op to MUST_SAVE. Quantization and communication save rules retain their
identities. Confirm the SAC observed forward/recompute streams agree, because
nested vendor dispatch visibility must not create a second cached event. Reuse
the actual selected policy/context, not a newly invented simplified save policy.

## Pre-implementation runtime cost review

The opaque vendor GEMM was already a compiler fusion boundary. The adapter adds
Python/custom-op dispatcher overhead per dense GEMM and retains the same cuBLASLt
kernel and scratch size. There are four first-dense-block forward GEMMs, with
corresponding required backward calls and SAC recomputation; other dense output
projections also use this entry point. Do not infer a total step count from the
first block alone. Graph partition boundaries add host submissions/graph launches
and may materialize intermediates or change live memory and overlap. Removing
GEMMs from replay can reduce CUDA Graph benefit substantially. Retain compiled
surrounding quantization, SwiGLU, attention and other eligible kernels. No direct
new collective, device-value host read, synchronization or transfer is intended.

Functional adapter itself requires no extra Tensor allocation beyond the original
vendor output; the stream-keyed workspace persists outside captured pools. Actual
stream keys/cache entries and graph-boundary staging may change memory. Mutable
functionalization may clone residual storage and requires its own evidence if
used. No steady-state timing, memory, launch or speedup estimate is verified.
Checkpoint model/optimizer schemas and parameter ownership are untouched by the
design; continuation still requires native acceptance rather than source inference.

## Finite verification and remaining gates

These are implementation acceptance checks, not tests executed in this design.
Before each further run, identify the unresolved gate and stop on its result.

1. **CPU behavior regression:** import the actual task shim/helper with a scoped
   vendor backend double and actual tuple/QTensor inputs, restoring all state.
   Observe calls through the real outer operator, not an AST/text match. Broken
   alias routes directly to an untagged vendor target; fixed route exposes the
   tagged task op to tracing/compilation. Verify original argument values and
   defaults for row/row, row/column and column/column, alpha, beta, split flag,
   workspace, dtype, invalid layouts and accumulate-without-out. No fake CPU
   kernel becomes a production CUDA fallback. These checks prove plumbing only.
2. **Schema/fake/mutation:** use actual FakeTensor tracing and symbolic/strided
   shapes, delegating installed vendor Meta where possible without CUDA. Assert
   fresh functional output, dtype/shape/stride, mutable out identity outside the
   None-return op, overwrite and accumulate behavior, invalid-call failure and
   alias/version semantics. CPU stand-ins can prove the mutable contract, not
   cuBLASLt numerics. Inspect compiled post-functionalization target/tag; test
   must fail if the outer adapter/tag is removed. Do not require a new GEMM CPU
   implementation to pretend to test CUDA.
3. **Autograd/SAC integration:** exercise actual FP8Gemm and actual selected SAC
   policy with controlled backend/quantizer doubles where feasible, comparing
   elementwise dgrad/wgrad and residual grad, saved state, frozen weights and
   disabled activation grads in eager and compiled paths. The bare adapter must
   not silently provide a fake raw-FP8 backward. Assert unchanged recomputation
   policy and forward/recompute event consistency. This isolates integration;
   it cannot close actual FP8 CUDA parity.
4. **Authorized native runtime gate:** run exact-preserved graph0/graph1 controls
   on the same new source snapshot, without allocator-history instrumentation,
   first-error stop and no guard bypass. Require all eight ranks, 25 real optimizer
   completions with four microbatches, initial and next update transitions,
   forward/backward/SAC paths, and capture/replay evidence for eligible partitions.
   Confirm dense unsafe nodes execute outside captured/private-pool regions and
   no hidden-workspace pool error persists. A later failure is a new recorded
   blocker, not permission for blanket exclusions. Preserve the former update33
   native71 illegal-instruction finding as a separate outstanding gate.
5. **Numerics/continuation/performance:** compare loss and elementwise gradients
   before clipping, clipping norm/scale, clipped gradients, optimizer moments and
   parameter updates with agreed FP8 tolerances and measured A/A nondeterminism.
   Preserve native checkpoint/resume/RNG/data-position acceptance where required.
   After runnable capture and scoped execution checks, use production A/A and alternating unprofiled controls,
   stable-step series and matched memory; full numerical/continuation acceptance may remain explicitly open while measuring performance. Separate host work, device compute,
   synchronization/idle, copies/communication and integrated step latency. Traces
   explain partition mechanism, not speedup magnitude. CPU results do not establish
   CUDA/NCCL, graph-pool safety, distributed lifecycle or production readiness.

## Rejected alternatives and provenance limits

No installed public API accepts an explicit workspace Tensor, so caller-owned
scratch requires a vendor schema/kernel change outside this task-only remedy.
Ordinary prewarm uses the wrong stream key. Private manager-stream prewarm,
workspace_size=0, guard suppression and backend/algorithm/precision changes do
not satisfy this contract. `allow_in_graph` alone does not attach safety tags.

`source-pins.json` records installed file hashes and task revision. The installed
Torch reported revision is metadata, and CudaCherry's local-wheel metadata does
not provide a reproducible source-to-binary build attestation. Runtime still must
record the actually loaded GEMM binary. Public documentation informs the design;
installed pinned source determines the exact version-specific contract.
