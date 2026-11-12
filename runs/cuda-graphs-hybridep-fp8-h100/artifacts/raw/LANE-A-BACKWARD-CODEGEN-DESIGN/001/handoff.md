# A006 backward codegen diagnosis and next decision

## Result and finite acceptance

A006 fails in generated Python for the first dense decoder block's compiled AOT
backward, before invoking its final eligible partition. A split mixed-order
RMSNorm backward reduction is renamed during codegen without a consistent
partition boundary: the outer runner requests undefined `buf66`, while the real
fp32 result `buf67` is local to the preceding partition and is not returned.
This is an Inductor codegen/signature defect. The Python exception detects it;
there is no evidence here of a failing CUDA kernel.

**Next decision:** prepare one reduced-coverage candidate by adding `layers.0` to
the existing exclusions, so all 18 model blocks remain fullgraph compiled with
CUDA Graphs off, while the independently compiled chunked CE retains global
CUDA Graph eligibility. Use the same exclusions in graph0 and graph1 controls.
Do not change another compiler/codegen option. This is a finite native viability
candidate, not an accepted fix or a throughput recommendation. Its useful outcome
is proving actual loss forward/backward capture and replay across full updates.
If that surviving region cannot satisfy this gate, stop this Lane A candidate;
do not continue removing regions or sweeping flags.

Acceptance for this diagnosis was finite: identify first failure/phase and actual
region; trace definition/use/free and producer/consumer mismatch; inspect matching
installed code and primary guidance; select one legal next contract with costs
and open gates; write and check these two artifacts. These items are complete.
No GPU calls, runs, source edits, package changes, tests, harnesses, agents or
publication were performed. Prior tests and accepted adapter/MoE policy reviews
were reused. The code-rules and perf-experiments skills and applicable GIFT
CONTRIBUTING/REVIEW/scoped manuals were read; supplied AGENTS rules were reused.

## Exact evidence and phase

All eight retained `LANE-A/attempts/006/torchrun/**/error.json` files contain the
same `UnboundLocalError` involving `buf66`; their bytes are individually pinned in
`source-pins.json`. Rank 3 supplies the first complete printed stack at
`LANE-A/attempts/006/supervisor.log:36032–36084`; print order is not rank failure
ordering. The path is `Trainer.train_step -> forward_backward_step:701 ->
loss.backward -> AOT CompiledFunction backward -> Inductor output_code ->
Runner.call:2831`. Immediately preceding log entries show backward warmup of
eligible partitions 0 through 4 at 00:56:19.662–.666 UTC on rank 3. This is step 1,
with zero completed optimizer updates (campaign owner's observation).

The supervisor end marker was reported at 00:56:25.847Z; retained
`exit-status.json` records exit code 1 at `2026-11-12T00:56:25.887353+00:00`.
These are separate timing boundaries. Exact task processes gone and all eight
H100s idle at 00:57:19 are the owner's cleanup observation, not a query made here.

The inspected rank-3 generated module is:
`NATIVE-SMOKE/001/cache/torchinductor/jy/cjyedefy543up2dllmundwwr54j5nu6ekdpavamvw4brzozee3uv.py`.
It declares `AOT ID: ['1_backward']` at line 1. Graph `[1/0]` scheduler source
stacks point through SAC's checkpoint wrapper, `transformer_block.py:93`, MLA
`softmax/mla.py:158`, and dense `cuda_gemm.py:319`. The module contains attention,
dense SwiGLU and their backward/recompute work, not HybridEP dispatch or CE.
The preset has one dense block and 17 MoE blocks. Activation logs at supervisor
lines 37148–37151 confirm loss compilation, 18 compiled blocks and graph-off
`layers.1` through `layers.17`; therefore the surviving model region is `layers.0`.
This region assignment also agrees with its shapes and dense FFN operations.

Source snapshot 005's manifest SHA256 is
`7192818038565d3c5f6d4c39273f3f9fc21a951f981fc158e834e890f3f1162a`,
with 88 changed files and 25 prior files unchanged. The inspected config policy,
compile policy and `_cudacherry.py` hashes match their snapshot-005 entries.
Current inspected files and runtime artifacts are pinned separately. Unrelated
callback diagnostics owned elsewhere do not change this diagnosis's source scope.
The preserved preset SHA256 remains
`b341701bd1eef7774c110fbf7ed69cf6280ba8691100b05be457dee3e7d50a7b`.

## Buffer and partition chain

| Location in generated module | Observed behavior |
| --- | --- |
| `partition_4:516–530` | Allocates fp32 `workspace_0` with 3,145,728 elements (12 MiB), runs mixed-order RMSNorm backward kernel, then reduces 2,048 partials over 1,536 channels into fp32 `buf67: [1536]`. |
| `partition_4:529–570` | Deletes workspace after reduction. It neither defines `buf66` nor returns `buf67`; its return tuple has 14 other values. |
| `Runner.call:2819–2821` | Calls partition 4 and unpacks those 14 values; neither name becomes an outer-runner binding. |
| `Runner.call:2824` | Calls public unsafe dense GEMM outside captured partition 4, producing `buf76`. |
| `Runner.call:2831` | Tries to construct partition-5 arguments including `buf66`; this is the observed first exception. |
| `Runner.call:2832` | Would delete `buf66` after argument staging; this statement has not executed and is not the cause. |
| `partition_5:2494` | Declares incoming `buf66` in its argument unpack. There is no effective use of it in the body. |
| `partition_5:2635–2639` | Instead reads `buf67`, casts its fp32 norm-weight gradient to bf16 `buf68`, then deletes `buf67`. No `buf67` argument/global binding exists. |
| `Runner.call:2865` | Would return `buf68` among parameter gradients, after the unreachable final work. |

There is no allocation, assignment, alias, or earlier free of `buf66` anywhere in
the entire module. This is not use-after-free or a request to retain a tensor
longer. Substituting `buf67` at line 2831 would also be wrong: it is not an outer
binding. Even an outer-signature-only repair would leave the inner argument-name
mismatch. Do not patch the generated cache.

Installed `codegen/simd.py:_codegen_mix_order_reduction:2898–2927` renames a split
reduction intermediate to the final reduction's buffer name, marks the old buffer
removed and its final reduction op removed, and rewrites saved partial result
names. Lines 2957–3009 emit the final workspace reduction under that final name.
The emitted `buf67` and missing `buf66` fit this exact rename mechanism.

Installed `scheduler.py:10172` builds partition signatures from scheduler
read/write dependencies before codegen. At 10614–10635 it cleans only buffers
removed during the current partition's codegen from that partition's signature.
`codegen/wrapper.py:4774–4805` emits input names, deletes, output bindings and
calls from the resulting signature. In this artifact the producer output and
later input signature never agree with the rename. The generated source proves
the mismatch; identifying every transient scheduler object would require compiler
instrumentation and is unnecessary to select the bounded next action.

## Adapter relationship and rejected remedies

The accepted unsafe adapter is active. Scheduler logs at 33769 onward describe
12 scheduler partitions (6 eligible, 6 excluded), with exclusions explicitly
attributed to `torchtitan.fp8_dense_gemm.default`. The six generated eligible
callables are numbered 0–5; excluded GEMMs execute in the outer runner. Do not
confuse these callable numbers with all 12 scheduler partition indices.

The adapter's boundary makes this broken split-reduction signature observable.
It does not implement the rename or allocate the missing buffer. Its unchanged
vendor delegate and required fp32 reduction semantics must remain intact.
A004 hidden-workspace and A005 nested-tuple pool diagnoses remain resolved at
source level under their accepted contracts; A006 does not justify bypassing
pool checks, retagging unsafe GEMMs as safe, or undoing MoE exclusions.

[Official Trees guidance](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html)
describes unsafe tags and partitioning; neither provides a public way to correct
this cross-partition reduction rename. The
[pinned upstream SIMD source](https://raw.githubusercontent.com/pytorch/pytorch/8b353065ce260336c5fdfa74d754e8bd9b91b062/torch/_inductor/codegen/simd.py)
contains the same rename strategy. The
[current upstream scheduler](https://raw.githubusercontent.com/pytorch/pytorch/main/torch/_inductor/scheduler.py)
still has per-partition removed-buffer cleanup. Retrieved bytes are hash-pinned;
`main` is not represented as an immutable commit. Installed Python is separately
hashed because reported Torch revision does not attest pristine source; installed
Trees includes local generation-cloning support.

Bounded official-source/issue searches, recorded in `source-pins.json`, established
no applicable maintained known fix or documented public option that repairs this
case. This is a bounded negative result, not proof no fix exists anywhere.
`triton.mix_order_reduction=False` and `TORCHINDUCTOR_MIX_ORDER_REDUCTION=0` exist
in installed config, but mere availability does not establish a supported remedy
or numerical/performance fit. Turning them off would alter reduction scheduling
and fusion in both controls without a matching known fix. Do not introduce that
experiment or a flag sweep. `allow_buffer_reuse=False` does not address an absent
producer binding. Generation cloning and the batch marker do not repair codegen.

No installed Torch/vendor patch, private manager, custom compiler wrapper,
exception suppression, guard bypass, extra fusion, backend/precision/Flex change,
or global graph disable is proposed.

## Selected candidate contract and costs

The existing GIFT policy already calls public `Module.compile` with
`fullgraph=True` and selected `options={'triton.cudagraphs': False}`. The
[public torch.compile documentation](https://docs.pytorch.org/docs/main/generated/torch.compile.html)
documents backend options and `triton.cudagraphs`. Add only `layers.0` to the
runtime exclusion list, retaining the exact original preset file. Keep all 18
blocks and loss compiled, global graph1, graph partition policy, FP8 recipe,
selective AC, Flex, HybridEP, FSDP2/EP8, batch8/global256/S4096/accum4, full-update
marker, data/loss/optimizer/schedule and checkpoint semantics. Keep the existing
unsafe adapters unchanged and apply identical exclusions to graph0 controls.
This requires no new training-source feature or production-source edit.

Installed `scheduler.py:9950–9955` keeps all kernels in the call function when
that compile invocation has `triton.cudagraphs=False`, avoiding this region's
captured partition signature route. This source mechanism supports the candidate;
only native execution can establish it succeeds with all inherited settings.

`ChunkedCELoss.__init__:560–561` independently compiles `cross_entropy_loss` via
`BaseLoss._maybe_compile:155–162`. The eight chunks per microbatch each call
`self.fn` at 789 and `chunk_loss.backward()` at 758. The compiled CE excludes
lm_head, normalization by global valid-token count, GradAccumulator, FSDP flags,
and decoder backward. Those surrounding operations retain their existing paths.
At four microbatches, there are 32 CE forward/backward opportunities per update,
with equal chunk shapes for S4096. CE uses fp32 logits conversion and the same
sum/ignore-index semantics. Source supports genuine surviving loss eligibility.

A006 `[2/0]` logs show forward/backward warmup around 00:56:09.723/.826 and later
`Running eager function=partition_0` calls. This is consistent with the independently
compiled loss reached before decoder backward, but graph-ID mapping to CE must
be proved from its generated source in native acceptance. Warmup and config
eligibility are not capture/replay evidence. Repeated same-update eager calls can
be part of Trees warmup lifecycle; they do not prove future replay impossible.

Documentation adds no production runtime cost. This candidate adds only one name
to startup matching. It removes dense attention/FFN graph coverage and its graph
pools, while preserving compiled kernels, SAC, quantization and communication.
Expected tradeoff: more ordinary host submission for that block relative to a
hypothetical working graph route, fewer graph launch boundaries/private pools,
and changed compilation/cache identity. No extra tensor pass, copy, device-value
read, synchronization or collective is introduced at the source boundary. Native
allocator peaks, compilation/startup time and numerical behavior remain unmeasured.
Existing mixed-order reduction costs remain in ordinary compiled execution.

The supplied baseline trace has no material exposed host gap. Loss-only coverage
therefore supplies no evidence of a useful speedup; do not invent a benefit or
launch a performance matrix before viability is established. Device compute,
host/wait, copy/communication, integrated latency and paired-memory magnitudes
are all unavailable from A006's failed startup.

## Native acceptance and stop conditions

Under existing execution authorization, the next native attempt must prove the
resolved list is exactly `layers.0`–`layers.17`, all model blocks remain fullgraph
compiled and loss remains compiled/graph-eligible. Complete the existing 25-update
all-rank gate with four microbatches and next-update transitions, without compiler,
allocator or pool-error suppression. Retain exact activation, failure and skipped
region evidence. Map loss generated functions to CE forward and backward and
show actual capture and replay after warmup on all ranks; mask-only capture does
not establish the surviving loss candidate. A failure to obtain loss replay, or a
new regional failure, ends this finite candidate pending a new explicit decision.
Do not count graph-off compiled model execution as graph coverage.

Successful viability would still leave numerical loss/elementwise-gradient,
clipping/optimizer/update parity, checkpoint continuation, historical
update-33/native-71 fault gate, and representative performance/memory acceptance
open. Stable unprofiled A/A and alternating A/B measurements are required before
any throughput claim. No gate is closed by this diagnosis, source inspection,
or the earlier 8 helper + 13 policy + 110 config checks.

## Decision correction: selected mixed-order-disable viability attempt

This section explicitly supersedes the earlier rejection of disabling mixed-order
reduction and the earlier selection of loss-only coverage as the next action.
The coordinator accepted one causally targeted attempt with
`TORCHINDUCTOR_MIX_ORDER_REDUCTION=0`. Loss-only coverage remains a conditional
fallback requiring a later decision; it is not the selected next run.

The earlier rejection imposed too strong a requirement for a documented known
fix. The inspected source supports disabling the precise optimization responsible
for this generated rename mismatch. A single bounded intervention at that gate
is justified provisionally without claiming it is an upstream-established fix.
This is not a search over compiler flags.

### Source evidence and precision

- Installed `scheduler.py:339` returns false from `MixOrderReduction.can_fuse`
  when `config.triton.mix_order_reduction` is false. This avoids forming the
  mixed-order fusion that invokes the implicated SIMD generator.
- Installed `config.py:2288–2291` initializes that bool from
  `TORCHINDUCTOR_MIX_ORDER_REDUCTION`; the exact string `0` selects false.
  Set the variable before Torch import on every rank. `apply_compile` sets the
  separate `mix_order_reduction_non_strict_mode` field; it does not re-enable
  `mix_order_reduction`.
- Ordinary `codegen/triton.py:5346–5378` explicitly promotes bf16/fp16 reduction
  operands to fp32. Lines 5563–5564 choose the accumulator type from the promoted
  source dtype. The affected RMSNorm weight-gradient sum is fp32 in A006's IR.
  Disabling this scheduling optimization does not request lower precision.
- The public `torch.compile(options=...)` interface supports backend options.
  Installed `torch/__init__.py:_TorchCompileInductorWrapper.apply_options`
  validates existing option keys/types, including this bool, and forwards them
  as `config_patches`; `_inductor/__init__.py:42–61` likewise forwards options.
  A narrow per-block public call could pass
  `options={'triton.mix_order_reduction': False}` while retaining fullgraph and
  inherited graph policy. Existing GIFT config has no such per-block option
  field, so that route would require a small explicit integration. It is not
  needed for the selected attempt-local environment route.

The environment variable is explicitly supported by the installed config source;
it is not represented as a separately documented public API guarantee. It needs
no framework/vendor patch, private manager, generic compiler interception or
training-source edit. The newly inspected installed files are hash-pinned in
`source-pins.json`; no new runtime introspection was performed for this correction.

The source supports retaining required fp32 accumulation, SAC/autograd and every
selected preset feature. It does not establish identical floating-point values:
ordinary reduction scheduling can change reduction order, fusion boundaries,
intermediate materialization and rounding. Numerical loss/elementwise-gradient,
clipping and optimizer-update parity remain acceptance gates. FP8 primitives and
their accumulation controls are unchanged.

### Exact selected contract

For one next Lane A viability attempt only:

1. Export `TORCHINDUCTOR_MIX_ORDER_REDUCTION=0` before Torch import to every rank;
   reserve the identical false setting for both graph0 and graph1 controls.
   Retain the original baseline and all failed attempts as evidence.
2. Keep exclusions exactly `layers.1` through `layers.17`. Keep `layers.0` and
   independently compiled loss graph-eligible with global graph1. All 18 blocks
   remain fullgraph compiled. Do not add `layers.0` to exclusions for this run.
3. Preserve graph partitioning, unsafe GEMM adapters, the full-update marker,
   selective AC, FP8 recipe, Flex, HybridEP, FSDP2/EP8, batch8/global256/S4096/
   accum4, loss/data/optimizer/schedule and checkpoint behavior. Change no other
   compiler option. Do not alter the original preset file.
4. Record the effective false value on every rank and verify that the affected
   generated dense backward no longer uses mixed-order reduction codegen. The
   environment applies to all compiled regions, including MoE blocks and loss;
   this wider scope must be recorded rather than described as dense-only.
5. Require the existing native 25-update all-rank viability gate, four
   microbatches, and next-update transitions, plus actual mapped dense/loss
   capture and replay after warmup. Eligibility, warmup or mask-only replay is
   insufficient. Preserve all error detectors and retain the first failure.

The expected cost changes are reduction order/fusion, kernel launches,
intermediate storage, memory traffic and compilation/cache identity. No extra
communication, explicit host/device synchronization or precision downgrade is
introduced by setting this gate. Memory and time magnitude remain unmeasured.
The supplied baseline's lack of a material exposed host gap still prevents a
benefit claim. Identical settings in controls make later graph comparisons
interpretable but do not by themselves establish compatibility with the original
mixed-order-enabled baseline; numerical and runtime effects remain to be checked.

If this one attempt fails, inspect its first new failure before selecting another
action. Loss-only coverage is conditional, not automatic. Success establishes
viability only: numerical parity, checkpoint continuation, historical
update-33/native-71 fault gate and controlled performance/memory acceptance remain
open. This correction adds no research, tests, runs or source edits.
