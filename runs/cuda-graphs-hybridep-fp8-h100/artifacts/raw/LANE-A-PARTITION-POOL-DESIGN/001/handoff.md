# A005 partition input and pool ownership diagnosis

## Result and finite acceptance

A005 exposes an Inductor partition / Trees input representation mismatch. The
failing captured partition receives HybridEP's whole nested return tuple as one
input, then exposes eight of its existing Tensor leaves as outputs. Installed
Trees checks only top-level Tensor inputs when excluding pre-existing external
storage from newly created private-pool outputs. Those nested leaves escape that
exclusion. The post-warmup detector consequently expects their external storage
in pool `(0, 1)` and fails. This is a different error direction from A004's hidden
CudaCherry workspace allocated inside the pool but absent from tracked outputs.

Finite acceptance for this diagnosis: inspect first stack and phase; identify
actual unsafe placement and failing input/output ownership; inspect exact installed
Trees and pinned allocator semantics; research maintained primary guidance; state
one bounded next decision and verification limits. These items are complete.
No source edits, GPU calls, runs, tests, package changes or agents were used.
Only this evidence directory was written. Prior adapter tests were not repeated.
The code-rules and perf-experiments skills and applicable GIFT manuals were read.

**Next decision (coordinator-selected narrower coverage):** prepare and review the
minimal opt-in per-block policy described below, retaining model/loss compilation
while disabling graphs only for the 17 HybridEP MoE blocks. Keep A005 failed.
This is a supported source-level compatibility route; it is not runtime-verified.
No public fix of the internal partition boundary itself has been established.

## First failure and phase

All eight retained `error.json` files under `LANE-A/attempts/005/torchrun/` report
the same pool-error class, each with eight missing addresses, at timestamp
`1794442958` (2026-11-12 00:22:38 UTC, second resolution). The first traceback
printed in `supervisor.log:4529` is rank 2; print order is not failure ordering.
Rank 3's stack follows native `Trainer.train_step -> forward_backward_step ->
model -> Decoder.forward -> layer -> checkpoint wrapper -> AOT forward ->
Runner.call:2120 -> self.partitions[2] -> deferred_cudagraphify -> manager._run
-> run_eager -> CUDAWarmupNode.run:844 -> check_memory_pool:2177`.

This is forward warmup, not backward, optimizer, replay, or the old Flex blockmask
failure. Logs show initial `[1/0]` forward partitions warm up, then `[1/1]`
recompilation (8 scheduler partitions: 4 eligible / 4 excluded) and
`[1/1] Running warmup function=partition_2` at 00:22:34.770–34.826 UTC on all ranks.
The tree callable numbering includes only four captured candidates, so its
`partition_2` is scheduler partition 4, after the second excluded region.
No optimizer completed (owner-provided campaign fact; timing window is incomplete).
Exit-status records code 1, finished `00:22:43.514761+00:00`; the supervisor finish
marker was reported at `00:22:43.475Z`. Process/GPU cleanup at 00:23:19 is the
extraction owner's observation, not a new query made by this diagnosis.

Rank 3 reports `{311685218816, 800411615232, 311475504128, 43012648960,
801082638336, 801084385792, 43012647424, 43012644352}`. Every rank's complete
error file and generated module are SHA-pinned in `source-pins.json`.
Repeated numeric addresses across ranks are separate device address spaces,
not evidence that ranks share a storage object.

## Actual generated order and ownership

Inspect rank 3 module `NATIVE-SMOKE/001/cache/torchinductor/gg/`
`cggm6pihsnqn4arbhccry2xrcpwzwzakwwi4kohh2l4rfjwfmy5u.py`:

1. `Runner.call:2071` onward runs eligible partition 0.
2. `fp8_dense_gemm` creates `buf32` directly in the outer runner, outside the
   captured callable. Eligible partition 1 follows.
3. Outer runner line 2098 calls `hybridep.dispatch`, producing `buf47`, then
   derives dynamic row count `u3 = buf47[0].size(0)`. Two shared-expert
   `fp8_dense_gemm` calls create `buf84`, `buf88` in that same excluded region.
4. Lines 2118–2120 pass `[buf47, buf84, buf88, u3]` to eligible partition 2.
   The nested dispatch tuple is not flattened into Tensor input slots.
5. `partition_2:1920` extracts `buf48/49/50/51 = buf47[0/1/2/3]` and
   `buf53/54/55/56 = buf47[5][0/1/2/3]`. These are dispatch-produced hidden,
   scale, probabilities, counts, and four quantized-expert-weight tensors.
   It performs grouped FP8 GEMMs, SwiGLU quantization and combine, then line 1991
   returns those eight existing Tensor leaves plus fresh results and `buf47`.
6. A final dense GEMM remains outside capture, followed by eligible partition 3.

Thus the new adapter/tag works at this inspected forward boundary: scheduler
logs explicitly label `torchtitan.fp8_dense_gemm.default` unsafe, and generated
calls are outside captured callables. This does not prove all backward/recompute
boundaries or steady-state capture succeed.

HybridEP's production `_dispatch_impl` in
`gift/torchtitan/distributed/deepep/hybridep.py:243` returns a tuple containing
Tensor outputs, `DispatchHandle`, and a `list[Tensor]` of quantized weights.
Its implementation creates these outputs in dispatch/comm-stream/quantization
work before the eligible partition. `_dispatch_fake:417` declares the same
nested arity and a dynamic hidden row count. It declares fresh Tensor outputs;
this failure alone is not evidence of a dishonest fake alias annotation.
Caller `dispatch_tokens:939` and autograd `_dispatch_setup_context:746` depend on
that return structure, handle and non-differentiable weight/scale ownership.

## Framework mechanism: detector versus cause

Installed `CUDAWarmupNode.run:773–789` builds external-input storage exclusions
from `itertools.chain(new_inputs, constants)` with `isinstance(t, Tensor)`.
It does not traverse tuples/lists. For this generated call it recognizes
`buf84` and `buf88`, but not any leaf inside `buf47`. Following execution under
`_use_cuda_memory_pool_manager`, `add_ref:827–834` treats a CUDA output with a
nonzero pointer and no excluded storage identity as newly pool-owned. The eight
extracted dispatch tensors meet that predicate despite predating this partition.
Line 844 calls the pool checker with those refs included.

`check_memory_pool:2137–2184` compares expected live storage pointers against
active allocator blocks belonging to the requested pool; after GC/synchronization
it reports expected pointers absent from that pool. Pinned upstream allocator
`checkPoolLiveAllocations` at reported Torch revision
`8b353065ce260336c5fdfa74d754e8bd9b91b062`, lines 1601–1626, tests actual active
pool blocks and exact expected-set cardinality. It is a consistency detector,
not the producer of the misclassification. Installed header and Python Trees
are pinned separately: the installed tree has local generation-cloning support,
so the reported revision is not asserted to attest pristine installed Python or
allocator binary source.

The eight missing-address count matches exactly the eight extracted external
Tensor leaves. The source establishes the ownership error directly. Individual
address-to-leaf identity and allocation stacks were not logged by A005; assigning
a specific reported address to a named leaf would be an unsupported inference.
No allocation-history rerun is needed merely to select the next decision.
Generation cloning cannot repair this first-warmup shallow input exclusion.
Recorded `CUDAGraphNode` input classification also operates on top-level Tensor
slots; patching only the warmup check would not establish replay correctness.

## Maintained guidance and legal remedy limits

[Official PyTorch Trees guidance](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html)
describes unsafe tags and automatic graph partitioning. That supported mechanism
is already active and correctly excludes dense GEMMs. The inspected guidance
provides no public option to flatten partition boundaries or move tuple extraction
into the preceding excluded region.

[Maintained upstream Trees source](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/cudagraph_trees.py)
was searched for the nested-input handling and known-fix question; its indexed
`get_non_cudagraph_inps` still uses the shallow Tensor filter. Targeted official
PyTorch source/issue searches did not identify a maintained, applicable fix.
That is bounded search evidence, not proof that no fix exists anywhere.
[The pinned allocator source](https://raw.githubusercontent.com/pytorch/pytorch/8b353065ce260336c5fdfa74d754e8bd9b91b062/c10/cuda/CUDACachingAllocator.cpp)
supports the detector interpretation above.

Changing HybridEP's tuple to another multi-output schema does not itself prove
Inductor will emit flat boundary arguments: codegen `WrapperCodeGen.codegen_partition_call`
passes named input nodes, and current getitem extraction is scheduled after the
excluded dispatch. Replacing its registered ABI affects autograd, SAC, fake
metadata and opaque handle lifetime; do not recommend that as a verified remedy.
A Python unpack in the existing caller is already present and did not flatten
this generated boundary. Cloning eight tensors adds large data/weight copies and
hides ownership symptoms without a demonstrated legal scheduling mechanism.

`apply_compile:59` compiles blocks with `fullgraph=True`. Public
`torch.compiler.disable` around dispatch introduces an incompatible graph break.
Global graph disable, graph_partition=False, preset changes, private Trees
patch/prewarm, guard bypass and installed vendor/framework edits do not establish
the selected surviving capture coverage. A public opaque op cannot truthfully return
input aliases as fresh functional outputs simply to force tuple unpacking.

## Runtime cost and open gates

This diagnosis changes no production runtime. Existing adapter overhead and
extra excluded-region/graph-launch boundaries remain as documented in the accepted
adapter design. The new failure supplies no throughput, kernel, host-wait,
communication, integrated-step or paired-memory magnitude; no speedup is claimed.

### Selected public integration contract: compiler regions

[Public torch.compile options](https://docs.pytorch.org/docs/2.15/generated/torch.compile.html)
include the per-call triton.cudagraphs setting. Installed nn.Module.compile:3057
passes kwargs directly to torch.compile, and Inductor compile:38–61 passes options
as compile_fx config_patches. Existing apply_compile already compiles each block
independently with fullgraph=True. This supplies a supported route around the
incompatible MoE Trees region without a graph break, package edit or new backend.

Minimal proposed source contract (no implementation in this diagnosis):

- Reuse existing CompileConfig in gift/torchtitan/config/configs.py:474 and its
  existing YAML/override mechanism. It currently has only enable, components and
  backend, with no options/exclusion field. Add one narrowly scoped torch-free
  field: cudagraph_exclude_blocks: list[str], default empty via default_factory.
  Values are exact existing named_segments FQNs, not regexes or a generic compiler
  policy registry. Empty preserves the exact canonical compile call/defaults.
- At gift/torchtitan/distributed/compile.py:apply_compile, match these names only
  among existing is_block segments; fail for unmatched/duplicate names and reject
  non-Inductor backend if exclusions were requested. Keep the existing backend
  and fullgraph=True. For matched blocks pass options={"triton.cudagraphs": False};
  for others preserve the current call verbatim. If compiler kwargs/options exist
  by implementation time, preserve all and merge only that key, without mutating
  a shared options dictionary. No mode replacement, eager fallback or global
  config mutation. Do not change the independent loss compile call.
- Existing entrypoint remains experiments.cuda_graph_campaign.train_scan_fix.main
  calling native main. Its existing campaign config/CLI override supplies the
  same compile.cudagraph_exclude_blocks list in graph0 and graph1, selecting
  layers.1 through layers.17 for this preset after checking actual segment FQNs.
  Do not duplicate or alter the production preset file. Only global graph policy
  differs between controls; both use identical source and exclusions.
- Actual call chain is native trainer construction ->
  models/gift_moe/_base/parallelize.py:parallelize_gift_moe after apply_ac ->
  distributed.compile.apply_compile. The caller is parallelize_gift_moe:36;
  the inspected compile call is line 84.
  Its named_segments walks via protocols.blocks.walk_segments, which yields the
  SAC wrapper as segment.module. Use existing protocols.blocks.peel to inspect
  the underlying block; inspect its descendants for an actual
  models.common.moe.dispatch.hybridep.HybridEPTokenDispatcher (inside MoE.experts).
  This proves the selected 17 MoE blocks, rather than assuming wrappers forward
  arbitrary type identity. Compile segment.module itself, preserving SAC and
  FSDP wrapping, module ownership and state keys. One dense block must inherit
  graph policy. Log names/counts and options once at construction; no per-step
  traversal or device-value reads.

The policy leaves all 18 blocks compiled, the MoE blocks' complete forward and
AOT backward outside Trees, and dense/loss regions eligible according to inherited
global policy and their existing op exclusions. It excludes MoE attention and
shared FFN too: this is the existing compilation-region granularity, not an
op-level repair. It introduces no new arithmetic, tensors, copies, collectives or
host/device synchronization. It forfeits graph benefit in most model blocks;
extra host launches there are expected versus a hypothetical working full-model
graph policy. No magnitude or speedup is established. Loss capture and surviving
dense capture remain execution gates, not guaranteed by eligibility.

No established public no-copy partition-flatten fix was found. compile_fx:3273–3276
flattens nested top-level graph inputs, but the failing tuple is an internal op
result at a scheduler-generated boundary. WrapperCodeGen.codegen_partition_call
passes named values. A traced Python tree_flatten becomes getitems whose placement
still belongs to the scheduler; the existing caller already unpacks this return.
Do not introduce custom compiler helpers to force placement or falsify aliases.

Baseline trace BASELINE-TRACE/analysis/001/handoff.md reports all-stream idle
56.267–70.887 ms in profiled 3283.130–3331.349 ms updates and dispatch sync overlap
with that idle of 8.512–9.132 ms. Its largest host-late gaps are submillisecond;
Lane B has no positive trace-supported material attention/shared-FFN/loss target.
These intrusive profile observations neither bound graph benefit nor justify a
speedup claim for narrowed coverage. Compatibility and actual replay coverage
come first; performance benefit may be immaterial.

Finite required checks for that later implementation:

1. Actual supported preset through SAC wrappers selects exactly 17 excluded
   HybridEP blocks and one inherited dense block, with loss compilation unchanged;
   invalid/duplicate names and wrong backend fail explicitly. Verify options,
   backend, fullgraph and all inherited kwargs survive. Default empty exclusion
   must reproduce canonical behavior. Use the real selection/compile call path.
2. If config changes, run config torch-free/default/roundtrip validation and
   affected configuration fingerprint gates required by the checkout. Review
   source/runtime cost once; no unrelated benchmark is needed for this policy's
   source-level design.
3. Authorized native gate: 25 completed optimizer updates on every rank with the
   exact preset, four microbatches, forward/backward/SAC and next-update transition.
   Prove MoE compilation with graphs excluded and actual dense/loss capture/replay
   where supported; report each skipped region. Preserve pool asserts. Use the
   identical exclusions for graph0/1, keeping original A005 as failure evidence.
   A surviving-region failure requires targeted assessment, not automatic further
   exclusions. No new experiment or runtime check was executed here.

The single next action is implement and independently review this opt-in policy
under explicit source authority, then use the existing native gate if execution
is authorized. No additional framework reproducer is required before that action.

Any future repair must establish correct flat Tensor staging and storage/opaque
handle lifetime through warmup, capture, replay, SAC recomputation and backward,
without changing FP8 primitives or communication. Exact all-rank native optimizer
completion, next-update transition, numerical/gradient/update parity, checkpoint
continuation, the historical update-33/native-71 illegal-instruction gate, and
controlled performance/memory acceptance remain open. A005 closes none of them.
