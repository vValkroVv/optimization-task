# Independent integration and runtime-cost review

Reviewed 13–14 November 2026 against task HEAD
`657e77a4f8ddb116b75925e5671b00642a0e4d32` plus the current working diff.
This is the assignment's one consolidated independent review. No GPU command,
installation, allocation, publication or production-source edit was performed
by the reviewer. The review findings below received targeted correction
inspection; both the ordinary-stream and nondefault-writer actual-GFSDP checks
passed.

## Finite acceptance and verdict

Review acceptance: inspect the actual configuration-to-consumer path, ownership
and stream lifecycle, supported writers, inherited precision/optimizer behavior,
fallback and diagnostic runner; assess meaningful fixture coverage and final
runtime costs; consolidate material findings and their acceptance gates.

**Verdict: review findings resolved; focused runtime correctness checks passed.**
The initial three findings received scoped source/test corrections. Subsequent
real-EP execution exposed an inherited fallback-input invalidation issue on
the same `_apply` path, described under R3 below. Its correction passed ordinary
stream run 007. The completed release-event join passed main run 008 with a
nondefault writer and immediate default-stream consumer: 1 passed, 2 deselected,
19.68 seconds. At review handoff, native activation, mechanism and performance
were separate gates. Both authorized native runs and mechanism analysis later
passed; the [completed report](report.md) owns that evidence and its limits.
The focused checks alone do not establish production performance or FP8,
compile and SAC integration.

Final dispositions: R1 exact-set activation is resolved by source inspection and
the activated actual-GFSDP fixture; R2 initialization is resolved by the faithful
backend initialization and passing parity/lifecycle checks; R3 identity,
fallback freshness and writer/reader ordering are resolved by the scoped source
corrections and runs 007/008. The DCP failure in run 005 was a fixture topology
error; true EP produced nonoverlapping checkpoint chunks and passed exact FP32
restoration. No review correctness finding remains open.

## Material findings

### R1 — P1: exact expert matching happens after the original Parameters vanish

- Location: `gift/torchtitan/models/gift_moe/_base/parallelize.py:192` and
  `gift/torchtitan/distributed/gfsdp/_param_group.py:532` in the initial diff.
- Failure: `g_fully_shard` constructs the parameter group eagerly. Its
  constructor calls `GFSDPParam.init_sharded_param` (`_param_group.py:416`),
  which deletes `fsdp_param.param` (`_param.py:93`). The new matcher checks only
  that deleted attribute. Every nonempty group therefore fails to match;
  no expert cache is requested or activated, despite the opt-in flag.
- Required fix: compare the exact captured expert set with each group's original
  parameter list before construction replaces Parameters, then enable the matched
  group before lazy initialization. Do not retain whole original expert tensors
  merely to perform this comparison.
- Targeted verification: actual GIFT `apply_gfsdp` must activate exactly the
  singleton expert groups while dense/default-off/extension groups retain the
  original output path. The drafted fixture's enabled-row assertion detects
  the original failure.
- Disposition: source correction inspected. `fully_shard` now accepts the
  optional exact `bf16_cache_params` set and matches it against `group_params`,
  whose original Parameter references remain available in that caller even
  after group construction. The corresponding group is enabled before lazy
  init; the model boundary retains PP1/EP>1/BF16 restrictions. The obsolete
  matcher is removed. This stores no original full-weight cache.

### R2 — P1: the initial parity fixture trains uninitialized model and EMA storage

- Location: `gift/tests/unit_tests/test_gfsdp_bf16_expert_cache.py`,
  `_build_small_model` in the initial draft.
- Failure: `.model.build().to(device)` allocates weights but does not initialize
  them. For example, `GroupedExperts.__init__` creates `torch.empty` Parameters
  at `models/common/moe/experts/gmm.py:121` and `:131`. The builder never calls
  `init_weights`/`init_states`; reseeding alone cannot make allocator contents
  deterministic. Also, `setup_ema_buffer` allocates `empty_like` shadows, then
  the fixture immediately applies EMA `lerp_` to those uninitialized values.
- Required fix: follow Trainer's real initialization order: parallelize,
  `to_empty`, then `init_weights` under `torch.no_grad()` and
  `GFSDP2Backend.weight_init_context`; seed the EMA from initialized main
  weights, as `BackendBufferStore.__init__` does. Preserve deterministic state.
- Targeted verification: confirm matched finite initial masters, then run the
  actual-GFSDP fixture and retain loss, input/weight gradient, updated FP32
  master and exact fresh-BF16-cast assertions. Restore any borrowed deterministic
  runtime settings.
- Disposition: source correction inspected. The fixture now follows the real
  backend `to_empty` / seeded `init_weights` context and explicitly seeds EMA
  shadows from main weights. Exact tensor comparisons reject NaNs by default.
  Actual distributed execution remains pending.
- Failed-run method reassessment (14 November): the second focused attempt,
  `artifacts/raw/pytest-cache-fixture-002.log`, reports two failed tests and one
  pass. The main test has **no recorded numerical mismatch**. At the 300-second
  timeout rank 0 is inside the test's unconditional `finally: dist.barrier()`;
  rank 1 is inside `torch.testing.assert_close`'s `if torch.all(matches)` host
  value check during the first post-DCP forward. Thus the cleanup barrier masks
  rank 0's original exception and leaves its peer waiting in CUDA work. This
  artifact cannot establish a cache-freshness defect or successful parity.
  The smallest method correction is success-only collective cleanup: preserve
  and propagate an exception immediately, retain its checkpoint artifact, and
  barrier/delete only after successful completion on all ranks. The existing
  freshness hook is registered after GFSDP's prepended unshard/wait pre-hook;
  no source stream change or relaxed comparison is justified by this timeout.
  Re-run only the failed required fixture after its independent extension-test
  correction and exception-reporting repair; keep the failed log as evidence.
  Execution reaching the post-DCP forward retains the earlier completed
  activation, accumulation/parity and checkpoint-stage assertions as partial
  evidence; it does not close later lifecycle assertions or the final fixture
  gate. GFSDP's prepended hook calls `pre_forward`, `unshard` and
  `wait_for_unshard` before this test hook; SAC pre-backward also unshards/waits
  before recomputation. Keep that valid consumer boundary and exact criterion.
- Third main-only attempt (`artifacts/raw/pytest-cache-fixture-003-main.log`)
  still has no reported numerical mismatch and times out at 90 seconds. Its
  rank-0 stack is now `common_dtensor.py:900`, calling
  `dist.destroy_process_group` from `with_comms`'s **exception handler**; the
  successful path is `self.destroy_pg()` at line 903. Therefore rank 0 has not
  established body completion: its original exception is again hidden during
  teardown. The success-only fixture cleanup was implemented, but no explicit
  exception traceback was printed before `with_comms` attempted destruction.
  The smallest decision-changing diagnostic is to print/flush the original
  exception inside the fixture before re-raising to that decorator, then rerun
  only the failed main test once if the parent authorizes it. Rank 1's blocked
  host equality read is secondary evidence of incomplete peer progress, not a
  freshness mismatch or proof of an event cycle. Preserve the actual GFSDP
  consumer hook, exact comparisons and production stream implementation.
- The exception-reporting attempt
  (`artifacts/raw/pytest-cache-fixture-005-main.log`) exposes the actual failure:
  rank 0's **FP32 master roundtrip comparison immediately after DCP load**,
  before the next BF16 consumer. Of 1,572,864 expert elements, 1,572,665 differ;
  maximum absolute difference is `0.003996892832219601`. This is not a
  BF16-cast comparison. Rank 1 proceeds to the next forward and stalls because
  rank 0 has already failed.
- Focused root cause: the initial fixture passes `ep_degree=2` into GFSDP but
  omits actual `model.parallelize` and GIFT sharding-config preparation.
  Consequently expert tensors are complete ordinary tensors on both ranks,
  with distinct singleton efsdp groups but no EP DTensor partition metadata.
  Rank-specific inputs make those unsynchronized complete replicas diverge.
  `_FlatCheckpointTensor._partition_offset` correctly returns zero when no
  fixed EP/TP `Shard` exists; both ranks advertise the same full logical
  `(FQN, offset=0)` checkpoint item. PyTorch's default
  `dedup_save_plans` saves only one such replica. CPU-only `FileSystemReader`
  inspection of `artifacts/raw/focused-dcp-a9f17d709cd84320adac17f40f9481de`
  confirms `layers.0.moe.experts.w13_E2FD` is one full `[1048576]` chunk at zero
  and `w2_EDF` one full `[524288]` chunk at zero, both from `__1_0.distcp`.
  Loading therefore restores the saved rank-1 replica to both ranks, correctly
  for the declared checkpoint layout but incorrectly for this fixture's
  divergent-replica assumption. This is a fixture topology error, not evidence
  for changing production DCP semantics, FP32 precision or cache events.
- Smallest faithful correction: prepare the GIFT sharding config for EP2, build
  one `ParallelDims` with dense FSDP2/EP2 and reuse its fsdp/efsdp/ep meshes;
  call `model.parallelize` before AC/GFSDP, preserving Trainer ordering.
  `GroupedExperts.parallelize` uses the existing dispatcher selection to turn
  local dispatch into maintained PyTorch all-to-all when EP exists. Assert
  actual EP `Shard(0)` DTensor ownership and nonoverlapping checkpoint chunks.
  Preserve rank-specific inputs, FP32 master equality and the first-consumer
  BF16 assertion. Source or event changes are not indicated by this failure.

### R3 — P1: backing replacement replaces optimizer-owned Parameter identities

- Location: `gift/torchtitan/distributed/gfsdp/fully_shard.py:583`, invoking
  unchanged `GFSDPParam.reset_sharded_param` (`_param.py:352`).
- Failure: every reset constructs a new sharded `nn.Parameter`. A previously
  built optimizer still owns the old Parameter; backward fills the new
  Parameter's gradient. Subsequent optimizer zeroing/stepping can therefore
  skip updates or consume stale gradients. The new cache hooks still invalidate
  the group, so apparently valid generations do not reveal the lost update.
  This behavior is inherited, but lies on the explicitly requested supported
  backing-replacement and identity path. The initial fixture only runs a
  `no_grad` consumer after `_apply`, which cannot detect it.
- Required fix: preserve the stable sharded Parameter when restoring replaced
  data into its existing FP32 flat allocation, or establish a concrete existing
  supported-boundary restriction without claiming continued training there.
- Targeted verification: perform a same-device/same-dtype value-changing `_apply`
  after optimizer construction; assert identity and optimizer ownership, then
  run backward/zeroing/update and compare fresh cache values and updated masters.
- Disposition: source correction inspected. Reset copies replacement data into
  the persistent flat master, reconstructs the proper RaggedShardDTensor wrapper
  and swaps it into the original sharded Parameter. It also retains a reference
  to the original flat gradient slice and restores the original gradient wrapper
  to that backing. Existing optimizer references and Parameter identity survive.
  The installed Torch 2.15 one-rank Gloo identity/spec/data-pointer probe and
  scoped lint are source-owner evidence; they do not replace the pending
  two-GPU fixture. That fixture now asserts IDs and compares an additional
  backward/update with the existing cached and uncached optimizers after `_apply`.
- Targeted real-EP run 006
  (`artifacts/raw/pytest-cache-fixture-006-main.log`) passes initial two-update
  parity, the corrected two-chunk EP checkpoint layout and FP32 master restore,
  post-DCP BF16 freshness, EMA swap/restoration and freshness, then stable `_apply`
  Parameter IDs and equal expert masters before the next optimizer step. Its
  next exact comparison fails at `post_apply.loss`; later gradient/master
  comparisons are not established. This is a genuine failed required check.
- Concrete affected inherited path: the cached model has several extra
  `no_grad` consumers during its load/EMA lifecycle; the uncached reference ends
  at backward. Those consumers mark **dense fallback** BF16 gather inputs
  valid. `_apply(+0.125)` then changes dense FP32 masters, but its only
  invalidation calls `invalidate_bf16_weight_cache`, which immediately returns
  on cache-disabled dense groups. Their validity remains set, so
  `_collectives.all_gather` skips the cast and uses pre-mutation dense weights.
  The reference's final backward cleared its validity, so its next consumer
  recasts the changed dense masters. Expert casts can remain correct while
  full-model loss differs. No loss tolerance change is justified.
- Minimal correction contract: clear ordinary gather-input validity for every
  affected group at `_apply`, retaining eligible cache-generation invalidation.
  After all reset/master copies, publish writer completion on each affected
  initialized shared data-buffer context using its existing `release_event`;
  selected preparation streams already wait/consume that event before recast.
  Deduplicate shared contexts. Existing compute release events predate these
  writes, and an old root `post_optim_event` does not establish their completion.
  This correction needs no new host synchronization or steady-state branch.
  Verify all managed FP32 main masters match before full-loss comparison and
  assert dense BF16 freshness at the actual first post-`_apply` consumer, then
  retain subsequent backward/optimizer parity checks. Extra nonpersistent
  `tokens_per_expert_E` accounting from the lifecycle forwards is distinct from
  consumed weights; do not silently align histories to conceal stale inputs.
- Targeted correction inspected: `_apply` now clears every affected group's
  gather-input validity before transformation. After all reset copies it records
  one current-stream event and publishes it to the deduplicated initialized
  data-buffer contexts. Their existing unshard release waits establish writer
  completion even when root pre-forward has an older optimizer event. Private
  cache events remain intact. The fixture now checks dense and expert
  first-consumer BF16 freshness and matched dense/expert layer FP32 masters
  before its unchanged exact post-`_apply` comparison. Scoped diff check passed.
  Only the affected main GPU check remains necessary for this correction;
  the independently repaired extension test already passed run 004.
- Main run 007 passes the complete corrected actual-GFSDP lifecycle/parity
  fixture on the ordinary stream. During final patch integration, one precise
  dependency issue was identified before source freeze: replacing the shared
  context's old release event with the new writer event must preserve the old
  readers' completion. Host-quiescent mutation can still have CUDA work pending.
  The corrected writer stream now waits each unique initialized context's
  non-null prior release event **before** `_apply` transformations, then publishes
  the post-reset event afterward. Exact source inspection accepted this join.
  Eligible private-reader waits alone do not
  cover dense fallback readers. This adds event dependencies only at `_apply`;
  it adds no host synchronization. Main run 008 passed the same fixture with
  a nondefault writer and immediate default-stream consumer, retaining the
  older root optimizer event. Freshness, stable Parameter IDs, all-group master
  equality before the step, and exact post-step loss/gradient/master comparisons
  passed. This resolves R3 and its targeted fallback/event-ordering extension.
  Failed run 006 remains evidence of the pre-fix fallback freshness regression.

## Integration inspected

- Eligibility is narrow: GIFT GFSDP, PP1/EP>1 and exact expert set at the model
  boundary; FP32 original/master data, BF16 input, singleton shard process
  group, ordinary `None` bucket, equal padded and per-parameter layouts, no
  all-gather extensions, no YCCL handle at group finalization. Other layouts
  retain normal gathering. R1 prevents this intended boundary from working in
  the initial diff.
- Eligible output aliases the already allocated private per-group BF16 input.
  Other pool users still write the shared output allocation, not that private
  input. The existing shared ownership/release checks remain conservative.
  Refresh does not construct new unsharded Parameters.
- The original FP32-to-BF16 `copy_` remains the refresh operation. No new
  rounding stage, custom kernel, compiler wrapper or FP8 cache is introduced.
  FP8 quantization in `GroupedExperts._quantize_local_expert_weights` remains
  active. SAC and compilation policy, gradient routing, FP32 accumulation,
  reduce-scatter, MTP and optimizer mathematics are unchanged by the cache.
- Inner optimizer pre/post hooks cover direct inner steps as well as container
  steps. Each requested trainable group's managed Parameter IDs must be covered
  by those optimizers; optimizer-in-backward retains the ordinary path.
- Ordinary load invalidates before writes and records write completion afterward.
  DCP invalidates before direct target-view loading and records completion in
  `finally`, including partial failure. Cache state is Python runtime state and
  is absent from persisted model/optimizer data.
- EMA swap/restoration invalidates active values. Refresh uses the active
  `_padded_sharded_param_data`, not the main-only per-parameter backing slice.
  Shadow seeding/updating does not invalidate main-active cache generations;
  active-EMA mutations do invalidate. The production callback/store reaches
  these group methods. R2 concerns the initial test's missing shadow seeding.
- The cache keeps readiness and per-group reader/write events. Refresh waits
  for readiness, readers and completed master writes; hit preparation waits for
  readiness; `AllGatherResult.wait` orders compute before consumers. Invalidation
  orders master writes after prior refresh/read work. These are stream waits,
  not host synchronizations. An unused speculative preparation retains its
  ready event even after `finalize_backward` waits and clears its transient
  result; optimizer invalidation also waits that retained event.

## Final runtime-cost assessment

| Scope | Source-derived effect | Required execution evidence |
| --- | --- | --- |
| Device compute / copies | One existing FP32-to-BF16 cast per consumed weight generation; none on hits. Eligible size-one input-to-output gather/copy is bypassed. | Per-group counters across all accumulation microbatches and boundaries; rank-0/1 diagnostic trace of the actual weight-preparation region. |
| Device waits / host sync | Additional readiness/reader/write stream dependencies; no new `.item()`, D2H read or host `synchronize`. Existing shared-buffer serialization remains. | Trace consumer waits and overlap; no estimate of recovered optimizer time from overlapping kernel sums. |
| Host work | Python generation/validity checks and disabled-counter method calls on cached unshards; one additional reader-event record per reshard; selected-group optimizer hooks per update and readiness/write handling. | Native unprofiled update window measures integrated effect; trace contextualizes launch/event cost. |
| Memory | Reuse existing persistent input; no second full per-layer BF16 tensor. Shared outputs remain allocated. Constant-sized events/integers/counters are added. | Matched-lifecycle peak allocated/reserved memory, retries/OOMs. No source-level memory-saving claim. |
| Communication / gradients | Only proven singleton expert gather is bypassed; dense/fallback gather and all gradient reductions retain their implementation and precision. | Focused true-GFSDP gradients and native configuration/activation receipts. |
| Compile / SAC / FP8 | Stable unsharded compute Parameters and unchanged quantization/metadata consumers. No new compile region or custom kernel. | Native FP8+compile+SAC run; the small BF16 fixture does not establish these conditions. |

The R3 correction adds wrapper reconstruction/swaps and, when a gradient exists,
one gradient-backing copy only at `_apply`/reset boundaries. It adds no training
step, microbatch, GPU synchronization, full-gradient allocation or communication
pass. The flat-gradient reference does not own additional tensor storage.
The fallback-input correction clears existing validity metadata and joins prior
shared-context release events on the writer stream before mutation, then records
one event after reset copies. These operations occur only at `_apply`; they add
no steady-state microbatch branch, full tensor allocation or host synchronization.

The default remains off. Default-off forward/backward still follows the original
allocation and collective route; the additional optimizer construction walk is
host-only startup work. The diagnostic API reads Python metadata only.

## Runner and verification boundaries

The existing runner explicitly forwards 60/20/40 for unprofiled timing and
21/20/0 with profile update 21 / at most three complete microbatches for ranks
0/1. It preserves accumulation 32 from local/global batch and topology, LR
65000/2000, V2 prefetch 32 and logical epoch 51200. Previous fusion patches
remain unapplied. The native configuration and resolved-config guards retain
FP8, compile, selective SAC, EMA and MTP; no canonical recipe is changed here.

Timing counters are off and only one activation-metadata snapshot is collected
after training, outside the measurement window. Diagnostic counters enable
before training and snapshot initially and after every complete optimizer update,
so their coverage is independent of the three-microbatch trace cap. The parent
will also use existing diagnostic `--capture-microbatches` for explicit all-rank
32-microbatch receipts. Timing leaves that host increment off; the historical
control's small instrumentation mismatch must remain explicit.

The fixture uses actual GIFT blocks, GFSDP parameters, mixed-precision buffers,
backward hooks, a two-rank dense fallback and singleton expert groups, three
checkpointed layers with groups 0/2 sharing the pool, and ordinary plus actual
DCP load. It does not mock the GFSDP path. Exact BF16 equality to a fresh cast
is justified. Exact mini-fixture parity is acceptable only with established
deterministic settings and matched initialization; it is not a whole-model
FP8 equality criterion and must not motivate tolerance relaxation.

Source-owner scoped normal Ruff, F821/F811/F841, import and py_compile results,
and the runner owner's 76 unique focused checks, are recorded in
[implementation](implementation.md) and [native plumbing](native-plumbing.md).
The reviewer did not duplicate those executions. Focused coverage is complete:
the prior default-off check, extension stand-in run 004, ordinary-stream main
run 007 and final nondefault-writer main run 008 passed. Failed logs remain
retained. The two native candidate launches remain open, together with their
activation, mechanism, integrated timing and memory evidence.

The compatible saved V2 control covers updates 21–60; it is historical rather
than a contemporary paired run. No new control, repeat/noise estimate, confidence
interval, convergence claim or PP4/EP32 extension is authorized by this review.
