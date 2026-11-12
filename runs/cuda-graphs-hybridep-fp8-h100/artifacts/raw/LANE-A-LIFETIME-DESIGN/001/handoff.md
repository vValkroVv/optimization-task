# Lane A mask lifetime: bounded diagnosis and implementation contract

## Acceptance correction — owner update, 2026-11-11

The owner subsequently removed independent-run bitwise loss/norm equality as an
acceptance gate. Retain strict A/A differences as diagnostics; they do not block
capture or performance research. Establish floating criteria from production A/A
variability and precision before interpreting graph-off/on numerical results.
Exact structural, gradient-accumulation and checkpoint/resume requirements remain.
Keep FlexAttention and fix its mask lifetime without removing SAC, FP8 or HybridEP.
This correction supersedes the historical gate-4 bitwise wording below; the mask
lifetime diagnosis and selected public boundary remain applicable. See the
[current acceptance contract](../../../../report.md#current-acceptance-contract--owner-update).

## Result and acceptance

A public-API boundary fix is justified for a separate candidate. The smallest first
candidate is **one explicit CUDA Graph Trees generation per complete optimizer
update**, beginning before its first compiled mask producer. Preserve all four
microbatches, their backward passes, clipping, optimizer and logging within that
generation. This is a source-supported design, not an executed fix or proof of
replay. No source or installed-package changes and no GPU commands were performed.

Finite diagnosis acceptance: (1) identify first failing consumer and producing
storage; (2) trace native mask lifetime through compilation/SAC/FSDP; (3) compare
public remedies and their costs; (4) specify one isolated candidate and finite
runtime gates. These design items are complete. GPU gates below remain open.

## Pins and first causal error

Repository HEAD `446431dfa191ef0300f116257ae918bc85e38c06`, frozen candidate source
`SOURCE-SNAPSHOT/002`, patch SHA-256
`b7c02e66f19d7ca2558960660dee91e9771b2d06b75a1556416838e27ed05465`.
Installed torch `2.15.0.dev20260822+cu130`, reported upstream revision
`8b353065ce260336c5fdfa74d754e8bd9b91b062`. [Hashes](source-pins.json)
pin inspected local files, including the installed Trees and Flex implementations.
The installed Trees file contains generation-cloning support; rely on inspected
local behavior as well as the reported revision, rather than assuming pristine
upstream bytes.

[Complete rank-2 error](first-error.json), copied from LANE-A/002, retains the
first full Python stack. Ranks 2, 6 and 7 reported the same failure class. Rank-2
cudagraph logs establish this order:

1. Compiled mask producer `[0/0]`: 17 nodes, 1 CUDA-graphable partition, 8 outputs;
   INFERENCE warmup at 22:38:02.636 UTC.
2. First compiled decoder partition `[1/0]`: 49 nodes, 18 inputs, 23 outputs;
   FORWARD warmup at 22:38:05.111 UTC.
3. Trees `CUDAWarmupNode.run`, `get_non_cudagraph_inps`, evaluates
   `t.untyped_storage().data_ptr()` and raises the overwritten-output error.
   Its stored allocation traceback identifies Flex
   `create_block_mask -> _create_sparse_block_from_block_mask ->
   _dense_to_ordered(partial_blocks) -> num_blocks_in_row.to(int32)`.

The failure is an invalid mask-output storage being inspected as a decoder input.
The stored Flex traceback names its producer; it is **not** the current execution
stack failing inside mask construction. No optimizer update completed. No actual
attention kernel, backward, or replay coverage is established by this attempt.
The exact invalid tensor field is consistent with `kv_num_blocks`; the error does
not print the input index, so this field identification remains an inference.

## Affected path and lifetime

- Native `Trainer.train_step` fetches four microbatches and discovers global valid
  tokens before forward. Each is transferred before `forward_backward_step`.
- `post_dataloading_process` builds positions and calls
  `Decoder.get_attention_masks` before decoder invocation. `_get_flex_attention_masks`
  composes causal/document mask mods and calls shared `create_attention_mask`.
  Shared `_compiled_create_block_mask` caches `torch.compile(create_block_mask)`
  with default options. Therefore the global graph-on option also captures this
  integer, non-autograd producer.
- Installed Flex produces partial/full KV counts and indices and their Q transpose
  metadata in `BlockMask.from_kv_blocks`. The output records all those tensors,
  block sizes, sequence lengths and mask callable. A document-mask callable also
  closes over eagerly produced `doc_ids`. None may be frozen across changing data.
- `Decoder.forward` passes the same mask object to every layer, not one new mask
  per layer. It must remain valid across the entire 18-layer forward and its
  SAC backward. A compiled layer does not own or shorten this lifetime.
- `parallelize_gift_moe` applies EP setup, SAC, per-block fullgraph compilation,
  then FSDP2. Native Flex attention has its own explicit
  `options['triton.cudagraphs']=False`; this policy is distinct from the shared
  mask producer's default compile policy and remains unchanged.
- Native chunked loss runs head/chunk backward then decoder backward. Selective
  checkpoint saves costly attention/communication operations and recomputes other
  parts; `skip_fwd_side_effects_in_bwd_under_checkpoint` stays unchanged. All mask
  metadata remains live until native backward is finished. FSDP2 gather/reshard and
  communication occur through the original wrappers. The diagnosis does not prove
  their capture safety.
- Four forward/backward pairs accumulate `.grad`; `accumulated_losses` retains
  detached loss tensors until update metrics. Gradients survive until clipping and
  AdamW. A generation boundary per microbatch can invalidate graph-owned gradients
  or loss values even after that microbatch's backward has finished.

## Why the heuristic is implicated

Installed Trees `get_curr_generation` uses Dynamo's generation counter until the
first explicit mark; afterwards it uses `MarkStepBox.mark_step_counter` exclusively.
`can_start_new_generation` permits a changed automatic generation when no forward
backward is pending. The integer mask producer is INFERENCE and does not create a
pending backward. The next separately compiled decoder invocation can therefore
end its warmup generation and deallocate still-needed mask output storage.
`try_end_curr_warmup` calls `dealloc_current_path_weakrefs` on such a transition.
This mechanism fits the recorded INFERENCE-to-FORWARD failure precisely. It does
not imply stale external model inputs: input tokens, positions and document IDs
are eagerly created/transferred, while the reported allocation belongs to the
compiled mask. Nor does it prove other graph lifetimes will succeed later.

The selected API makes the compiler invocations within the full update share one
explicit generation, preventing that particular automatic transition. It does not
make arbitrary saved activations or gradients safe across explicit marks.

## Established remedies and selected tradeoff

Primary references inspected on 2026-11-11:

- [PyTorch 2.15 Trees limitations](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html#limitations)
  documents iteration heuristics and out-of-compile clones; its public examples
  support managing a sequence of calls with an explicit generation boundary.
- [Public mark API](https://docs.pytorch.org/docs/2.15/generated/torch.compiler.cudagraph_mark_step_begin.html)
  defines the boundary as the start of a new iteration.
- [Compile options](https://docs.pytorch.org/docs/2.15/generated/torch.compile.html)
  documents the graph-only `triton.cudagraphs` option.
- [Pinned upstream Trees](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/torch/_inductor/cudagraph_trees.py)
  and inspected installed implementation establish generation ownership.
- [Upstream issue 169545](https://github.com/pytorch/pytorch/issues/169545)
  records separate gradient-accumulation lifetime failures. Its closure references
  improved diagnostics; closure does not establish this native training path safe.

| Remedy | Fit and cost | Decision |
| --- | --- | --- |
| One explicit boundary before first mask of complete update | Public API, one CPU counter update, no added tensor compute/copies/sync/collectives; may retain more masks/activations in a longer path and increase graph pool/recording footprint | First candidate; directly addresses the observed generation transition and retains all four accumulated gradients/losses |
| Mark immediately before decoder after mask | Frees the newly produced graph-owned mask at the wrong point | Reject placement |
| Mark each microbatch | Protects mask-to-decoder chain but can discard graph-owned accumulated gradients and detached losses needed at update end | Do not select without a separate gradient ownership solution |
| Clone mask outside compiled producer | Public remedy for persistent outputs; must clone every tensor field including full/Q metadata, preserve optional deterministic dQ metadata and callable; extra device allocations/copies/launches each microbatch; still leaves decoder-to-decoder boundaries unmanaged | Available fallback for the exact mask ownership problem; do not clone only the named count field or put clone inside compile |
| Graph exclusion only for mask producer | `torch.compile(create_block_mask, options={'triton.cudagraphs': False})` preserves compilation and computes fresh eager-owned outputs, no clone launch; leaves per-update boundary question for decoder/SAC/gradients | Small fallback if explicit-boundary candidate fails specifically on mask lifetime; requires separate graph-off/on prerequisite controls and capture attribution |
| Global `cudagraph_trees_generation_cloning='user_visible'` | Wider private compiler policy; can clone unrelated outputs and does not cover internal saved activations/gradients | Do not select for this scoped fix |

No maintained drop-in fix was found that proves exact EP8/FSDP2/FP8/SAC native
correctness. The public marker needs no package patch, custom op or new dependency.
There is no evidence for a speedup. This diagnosis document has no production
runtime effect; the candidate's material open costs are pool memory, recording
path length/recapture and existing distributed/wrapper capture behavior.

## Exact implementation contract (not implemented)

Keep global-graph-only attempt 002 unchanged as the original candidate/control.
**Integration correction:** `CampaignTrainer` unconditionally binds exactly one
RNG continuation callback; its schema also enforces the exact continuation loader.
Selecting it for marker-only performance runs would add unrelated prerequisites.
Keep that trainer, its schema, continuation YAML and validation unchanged.

Use one experiment-only `Configurable, Observer` callback instead:
`CampaignCUDAGraphStepBoundary`, with a torch-free `CallbackConfig` subclass
registered as `@config(type="campaign_cudagraph_step_boundary")`. Set the schema's
`_owner_spec` to its experiment runtime class. Schema discovery already walks all
`config.schema` modules; no trainer registry addition is needed. Update the exact
callback type set pinned by `test_config_torch_free.py`.

The callback is **off by omission**, following `CallbackManager`'s existing
composition rule: no enable boolean, trainer selection, dependencies, checkpoint
payload or extra RNG callback. A scoped overlay adds only:

```yaml
callbacks:
  campaign_cudagraph_step_boundary:
    type: campaign_cudagraph_step_boundary
```

At `fit_start`, reject `state.parallel_dims.pp_enabled`; the target is PP1 and a
different schedule needs its own design. Reject duplicate instances through the
schema's `validate_dependencies(previous)` so a duplicate key cannot cause two
marks. No compiler API call during construction, config validation or fit_start.
At `batch_start`, call `torch.compiler.cudagraph_mark_step_begin()` exactly once.
No marks at mask creation, individual layers, loss chunks, SAC recomputation or
microbatches; no per-step counter or device reads are needed.

This is compatible with both native Trainer and existing CampaignTrainer because
both use native `train_step`, whose first statement emits `Event.batch_start`
before zero_grad, fetch, transfers, mask construction and all four forwards and
backwards. Continuation's deferred RNG restore remains in its original post-fetch
location; changing compiler generation does not consume RNG.

Callback contract assessment: `callbacks/README.md:96` reserves `Mutator` for
training-semantic changes; supported mutation axes contain no compiler/runtime
axis. This callback changes only compiler iteration ownership, uses no trainer
reference, and never writes through ObserverState, modules, parameters, gradients,
data, optimizer or scheduler. Classify it as Observer, explicitly documenting its
runtime-control purpose and the required numerical equivalence gate. The base
Observer description about owning its own state is not permission to mutate
training objects or hide semantic changes; if actual numerical behavior changes,
this route fails acceptance. Do not invent `Mutation.routing`/`grads` attribution
for a compiler generation counter or expand the global callback API.

The callback handler runs observers in configuration order, after mutators and
before aggregators. Append the marker after the existing timing observer, keeping
that relative order identical in graph-off/on controls: the timing observer
records the prior-update closing event/current opening event before the marker;
its initial synchronization and rank alignment remain unchanged. Current timing
`batch_start` only observes events/memory and runs its existing window operations;
monitoring sources reset their own counters/hooks and do not invoke the compiled
mask/model there. The marker still precedes the first compiled training call.
The final closing boundary may mark once before the existing timing observer's
stop is acted upon; it must not trigger an extra training update or replay.

Runtime cost beyond the earlier generation analysis is one existing event-dispatch
handler call and a CPU counter update per update. If an observer snapshot is
already built by timing/monitoring, this adds no new snapshot; otherwise enabling
an Observer creates the handler's existing metadata snapshot at that event. No new
tensor pass, kernel, transfer, scalar device read, sync or collective is needed.
Use the same callback and ordering in the graph-off prerequisite control; graph-off
still increments compiler iteration state, but creates no CUDA Graph work.

Retain EP8, FSDP2, selective AC, both model/loss compilation, FP8, native Flex option
policy, batching, schedule, valid-token normalization, all optimizers and callbacks.
Do not alter graph partitioning, autotuning, reduction/fusion or strict numerical
settings. Do not allocate or reset gradients manually in this fix. Preserve previous
snapshot and produce a new candidate diff/hash before execution.

## Finite verification and decision gates

1. CPU config/behavior tests: torch-free registry discovery/type pin and config
   round-trip accept the callback; absence builds none; duplicate callbacks and PP
   fail clearly. Construct actual CallbackManager/CallbackHandler with the callback
   and timing double to establish one call per emitted batch_start in configured
   order, none at fit_start or any of four microbatch events, and exceptions
   propagate. Run a lightweight actual native Trainer.train_step path with mocked
   compute/optimizer to prove marker precedes zero_grad, fetch and mask/forward,
   all four forwards/backwards occur before clipping/optimizer, and next update
   marks once again. Verify both native trainer config (no RNG callback needed)
   and unchanged continuation config plus marker resolve; existing continuation
   binding/loader rejection checks remain intact. Preserve any monkeypatched
   compiler function/config state after tests. Lint changed Python with
   F821/F811/F841 and inspect call sites once. These tests cannot prove CUDA
   memory ownership.
2. One actual-target graph-on 25-update diagnostic, unchanged production settings,
   all eight ranks, four microbatches and 18 layers, native backward and optimizer.
   Retain first-error stacks and graph logs; prove mask/decoder first-call failure
   is removed and identify real FORWARD/BACKWARD capture/replay versus partition
   skips. Count captures/recaptures and inspect steady-state/capture memory on every
   rank. Stop on first genuine error; no silent fallback or parameter reduction.
   If a new failure arises, classify its owner from the stack before choosing an
   additional fix; this design does not authorize a broad rewrite.
3. Graph-off identical prerequisite control and changing-batch mask equivalence:
   compare every BlockMask tensor, dtype, shape, block size/sequence metadata and
   active callable semantics against original graph-off output; all four masks
   must remain valid through backward. Diagnostic snapshots/clones stay outside
   performance runs. Confirm accumulated pre-clip gradients persist through all
   four backwards, then native clipping/AdamW execute; no stale routing/data or
   FP8-state reuse. Include beginning of second update to exercise generation
   transition and zero_grad. A mask-only fixture cannot satisfy this gate.
4. Success at gate 2 establishes a runnable scoped candidate only. The campaign's
   required matched ten-update elementwise gradient/update/FP8-state comparisons,
   bitwise loss/norm gate, full-state fresh-process continuation, A/B/A/B timing,
   paired memory and 500-update sustained runs remain mandatory for finalist
   acceptance. Existing strict baseline A/A fails at updates 2–10 and remains
   independently unresolved; this marker cannot waive or explain that failure.

If the candidate fails, retain the original failure and new result separately.
Only a mask-specific continued ownership failure justifies trying the narrow
mask-producer exclusion next; it would change capture scope and needs explicit
attribution. Neither outcome proves universal CUDA Graph incompatibility.

## Design amendment record

The initial trainer-override wiring was withdrawn after integration inspection
identified its unconditional RNG-continuation requirements. The selected callback
route above supersedes that wiring; generation scope, source freeze, runtime
acceptance and unresolved strict-baseline gate are unchanged.
