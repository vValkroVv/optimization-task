# BF16 expert-weight reuse

Completed: 14 November 2026. The
[finite checklist](../../planning/README.md#active-assignment-bf16-expert-weight-reuse)
is closed. The implementation, focused correctness checks, independent review,
two authorized native launches and bounded mechanism analysis passed. The
feature remains opt-in and disabled by default. Throughput was 1.99% above the
saved V2 control; this historical comparison cannot establish a causal speedup.
This report owns experiment results; [compact results](results.json) retain
exact values and artifact pointers.

## Scope and source

The 14 November 2026 objective authorizes opt-in task-local implementation and
bounded checks on the existing allocation. Starting task revision:
`657e77a4f8ddb116b75925e5671b00642a0e4d32` (`main`, initially clean).
The original workspace GIFT checkout is outside scope. No installation,
allocation, commit, push or fresh V2 control is authorized.

The workload is the [accepted crop](../../docs/baseline.md), with explicit V2
prefetch 32 and logical epoch 51,200. All four previous fusion candidates stay
disabled. Preserve FP32 master weights and gradient accumulation, FP8 compute,
compile, SAC, EMA, MTP, accumulation 32, LR horizon 65,000 and LR warmup 2,000.
This experiment does not extend to production PP4/EP32.

## Question and selected approach

Reuse the existing per-group BF16 input storage across microbatches while its
FP32 source generation is unchanged. Eligible size-one expert shard groups
consume that persistent storage directly. Ineligible groups retain the existing
working-buffer path. The source hypothesis and observed cast/copy costs are in
[profile research](../../docs/profile-my-ideas.md#reuse-bf16-expert-weights-across-microbatches).

The reference is Megatron-LM
`5b069b1f76890bb322c6fdb656419999c0ba073e`, optimizer
`step_with_ready_grads` and distributed optimizer
`_copy_main_params_to_model_params`. Adapt only its master-to-compute-weight
lifecycle to GFSDP. Explicit mutation boundaries must cover optimizer updates,
checkpoint loading, EMA swap/restoration and supported storage replacement.
Tensor version alone is insufficient. BF16 values are derived runtime state.

The canonical flag is
`parallelism.gfsdp_reuse_bf16_expert_weights: false`. GIFT selects the exact expert
Parameter set before eager GFSDP sharding; eligible groups require PP1, EP>1,
size-one expert sharding, FP32 masters, BF16 compute, matching layouts and no
custom gather extension/YCCL path. Initialization assigns compute views to the
existing private BF16 input. Generation changes invalidate it, and the first
subsequent consumer refreshes it. Cache state is not persisted in checkpoints.
The [implementation record](implementation.md) and
[source pointers](../../docs/candidates.md#bf16-expert-weight-reuse) own details.

The inspected lifecycle reference supplies the master/compute-weight separation;
the GFSDP adaptation additionally needs shared-pool ownership and asynchronous
reader/writer ordering. Existing shared output storage remains available to
fallback groups. There is no additional full-size BF16 allocation or new host
synchronization. Ready, reader and writer events add stream ordering. Their cost
is included in native measurements rather than assumed free.

## Required verification

The deterministic fixture must exercise actual GFSDP preparation and consumers,
accumulation across at least two optimizer updates, recomputation, two groups
sharing the original working pool, changed-weight load, EMA swap/restoration,
eligible activation and fallback. Compare fresh BF16 casts, losses, input/weight
gradients and updated FP32 masters against uncached execution. No whole-model
bitwise FP8 requirement or full-model state export is part of this experiment.
Scoped lint and one independent integration/runtime-cost review are required.

## Native protocol

| Candidate run | Optimizer updates | Warmup | Measurement / capture |
| --- | --- | --- | --- |
| Unprofiled | 60 | 1–20 | 21–60 inclusive, 40 complete updates; counters off |
| Diagnostic | 21 | 1–20 | Update 21, at most 3 complete microbatches, ranks 0/1; counters on |

All ranks complete normal 32-microbatch accumulation. Profiling cannot supply
TPS magnitude. Verify actual child command and parsed arguments before launch.
Reuse saved V2 evidence with its actual measurement window. A single candidate
does not establish statistical significance or a general speedup. Ambiguous
performance ends this experiment without automatic repeats.

## Correctness and integration

Implementation and focused verification are complete. The
[actual-GFSDP checks](focused-checks.md) passed exact deterministic local
loss/input-gradient/expert-gradient/FP32-master comparisons across updates,
recomputation and shared-pool groups, plus ordinary/DCP loading, EMA swaps,
default-off/extension fallback and nondefault-stream backing replacement.
All failures are retained, including the post-replacement regression repaired
by clearing fallback input validity and preserving reader/writer dependencies.
The [independent review](review.md) has no unresolved material finding.
[Runner checks](native-plumbing.md) passed all 76 unique affected checks across
the documented invocations. Source and runner bytes were frozen before both
native launches.

The final main fixture uses real EP2/FSDP2 partitioning, seeded backend weight
initialization, fused AdamW and two accumulated optimizer updates. It compares
exact fresh BF16 casts at actual consumers and deterministic cached/uncached
losses, input/expert gradients and updated FP32 masters. It covers recomputation,
two groups sharing the original pool, ordinary and direct-view DCP loading,
changed weights, EMA swaps/restoration and value-changing backing replacement.
The nondefault writer/default consumer regression also passes without a caller
stream wait. Default-off and extension fallback pass separately.

The consolidated review found activation matching, fixture initialization and
backing-replacement identity/order defects. All were corrected and verified.
The inherited `_apply` repair preserves optimizer-owned Parameter identity,
clears ordinary input validity for dense and expert groups, joins prior readers
before writes and publishes completion after reset copies. The original failing
post-replacement behavior and subsequent passing runs are retained. Early harness
timeouts and the fixture's incorrect initial EP checkpoint topology are recorded
as failures, not reclassified as passing checks. No material correctness finding
remains unresolved within this scope.

## Native results

Exactly two candidate launches completed with exit 0 and unchanged source/config:
`bf16-cache-tps-001` (60 updates) and `bf16-cache-trace-001` (21 updates), on all
eight H100 80GB ranks. No control or repeat was launched. The
[native execution record](native-execution.md) owns exact commands, actual child
argument checks, resolved-config hashes, runtime versions and per-rank receipts.
Timing counters, profiler and microbatch capture were off. The diagnostic
captures exactly three complete microbatches on ranks 0/1; every rank completes
the normal 32-microbatch update and optimizer. LR remains 65,000/2,000.

Both timing rows below use 60 total updates, warmup 1–20 and **updates 21–60**
inclusive: 40 complete updates and 83,886,080 global physical input tokens.
The saved V2 row is the depth-32 `v2-long-epoch-candidate-perf-001` in the
[long-epoch receipt](../v2-prefetch-baseline/pair-long-epoch-001.json), not its
depth-zero baseline. No older 10+10 window is relabeled.

| Metric | Saved V2, cache off | BF16 reuse candidate |
| --- | ---: | ---: |
| Global physical input tokens/s | 140,933.05 | 143,736.69 |
| Per-GPU physical input tokens/s | 17,616.63 | 17,967.09 |
| Global supervised-target tokens/s | 140,898.64 | 143,701.60 |
| Full eight-rank window envelope, s | 595.219380370 | 583.609374266 |
| Rank-0 host update median, s | 14.971230144 | 14.625412892 |
| Rank-0 host update min–max, s | 13.804412458–15.560705662 | 13.536819126–15.438273734 |
| Maximum peak allocated/active, GiB | 70.839288 | 70.813498 |
| Maximum peak reserved, GiB | 71.998047 | 71.939453 |
| Allocator retry / OOM deltas, every rank | 0 / 0 | 0 / 0 |

TPS divides tokens by the complete synchronized eight-rank window envelope. It
includes normal loading, transfers, forward/backward, communication, optimizer
and lightweight logging; startup, compilation/warmup and teardown are excluded.
Rank-0 cadence is a host callback interval, without synchronization per update;
it is not device-only step time. Allocator peaks are reset/sampled at matching
window boundaries and maximized across ranks, not whole-process memory.
The allocated/reserved changes are small and establish no material memory saving.

The throughput difference is **+1.99% relative to historical V2**. These runs
have different source identities and execution times, no contemporary paired
control, repeat, identical-control noise estimate or confidence interval. The
saved control also had optional integer microbatch counting enabled; the timing
candidate did not. Within-run cadence ranges are descriptive, not run-to-run
uncertainty. The observed difference is provisional and its causal attribution
is inconclusive. This closes the bounded experiment without another run or
promotion of the default.

## Mechanism and final runtime costs

The [Perfetto analysis](trace-analysis.md) and
[machine-readable evidence](trace-analysis.json) import complete device events
with launch-flow links and no importer errors. For each rank's captured three
microbatches, the seven expert groups change from **21 to 7 FP32→BF16 casts**
and **42 to 0 local input→output copies**. Each removed copy is 603,979,776 bytes.
Every group refreshes once in microbatch 1; microbatches 2/3 reuse it. The active
FP8 path still launches 78 weight-quantization kernels inside 39 dispatches at
the same forward/recompute frequencies and grids. Its source bytes are unchanged.

Across all eight ranks and all 21 diagnostic updates, every eligible group
records one refresh, 63 hits, 64 direct uses and one invalidation per generation.
At update 21 the totals are 21 refreshes, 1,323 hits and 1,344 direct uses.
Noneligible groups' cache counters remain zero. Direct profile receipts record
32 microbatch calls, one train-step call and one optimizer-step call at update
21 on every rank. The generic microbatch map is empty in profile mode: earlier
updates' full accumulation is supported by 64 per-group gather uses and the
unchanged native loop, rather than separate per-microbatch hook records.

All 42 expert consumer wait submissions remain in each trace; preparation wait
submissions increase from 42 to 98. There is no host event/stream synchronization
or event query inside the affected consumer regions. Exact device wait time and
producer-event joins are unavailable because the profiler omits event handles;
they are not assumed zero. Source ordering and the focused cross-stream fixture
provide complementary lifetime evidence.

Affected cast/copy device unions over the sampled prefix fall from
29.355 to 4.118 ms on rank 0 and 29.343 to 4.126 ms on rank 1. Control work partly
overlaps other kernels and EP/NCCL, while remaining candidate refreshes also
overlap computation. Host spans, device sums/unions, overlapping communication
and enclosing three-F/B spans are reported separately in the analysis.
Overlapping durations cannot be added into optimizer-step savings. The prefix
excludes the other 29 microbatches and optimizer; profiled historical captures
do not establish throughput magnitude.

## Reproducibility, limits and completion

Candidate source is starting HEAD
`657e77a4f8ddb116b75925e5671b00642a0e4d32` plus the frozen patch SHA256
`4634d6ac225cc0993567de491605cb31b44121965ab633ea292179fa7436cb26`.
[Source identity](source-identity.json) records the patch and individual source,
runner, test and canonical-config hashes. The native source snapshot digest is
`3dfde3ee2b0d9b207d042522bbc4882d4bcbaec6d07250efdde24ae6dc24e42c`.
Saved V2 used HEAD `3b6b8928ec4b15a529d07ee99e672f2fc75ef73a`, snapshot
`ac22aae7ced85f3970b89e420010d87076b1ed0fa58b162b3aa245215aae35b7`.
The canonical experiment/model YAML bytes remain unchanged. Native records retain
seed 2048, data/runtime identities, resolved configurations and complete step
series. Original traces and failed logs remain outside tracked source.

This verifies scoped actual-GFSDP numerical/lifecycle behavior and native
FP8/compile/SAC/MTP activation and execution in the eight-GPU crop. It does not
establish whole-model FP8 numerical equivalence, convergence, long-run stability,
exact native checkpoint continuation, or production PP4/EP32 performance/support.
No full-model numerical capture or convergence run was requested. The feature
remains disabled by default. No required item in this finite assignment is open.

Both owned training groups exited without orphans; task scratch was removed and
all eight GPUs were idle at release. Existing allocation services and analysis
servers were preserved. No original-workspace GIFT edit, installation,
allocation, commit or publication was performed.

Final artifact checks passed: 15 maintained Markdown files, 241 local links,
balanced fences, three JSON documents, 17 frozen file hashes, the patch checksum
and `git diff --check`. The initial document check found a missing historical
display-copy link; it now points to the preserved import provenance, retaining
its recorded hash and identifying the unavailable display copy. Raw native logs,
traces and the frozen patch remain ignored by Git.

## Publication authorization

After experiment completion, the owner authorized committing and pushing this
implementation, tests and compact evidence to `origin/main`, updating the local
checkout, and downloading both candidate traces. Earlier no-publication
statements describe the measurement assignment. Raw traces remain outside Git;
no additional training or performance comparison is part of publication.
