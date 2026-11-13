# Monitoring-free baseline measurement — 13 November 2026

Status: unprofiled TPS PASS (provisional single run); complete-update traces PASS.
Owned-cache cleanup is complete. The final results/documentation commit publishes
this record to `origin/main` using the existing repository-local identity.
The [results](results.json) own exact values, identities and artifact hashes.
The original candidate campaign remains cancelled; no candidate or numerical
comparison was run. All new scratch/cache/output paths stay inside the task root.

## Unprofiled result

`monitorfree-baseline-w10-m10-002` completed 10 warmup updates and measured
updates 11–20, each with 32 microbatches. The complete eight-rank window was
**144.467936499 s** for **20,971,520 physical input tokens**:
**145,163.837099 global input tokens/s**, **18,145.479637 per GPU** (global/8).
The native supervised-token counter gives 145,128.396709 global tokens/s;
physical input and supervised tokens have distinct denominators.

Rank-0 per-update host cadence had median **14.562491119 s** and range
**13.542409329–15.803554488 s**. These are intervals between native
`Profiler.step` returns, with no adapter synchronization each update; the last
interval includes the end drain. The primary TPS uses max(rank end) minus
min(rank start) on the verified single host, with two CUDA boundary drains.
Normal MDS loading, transfers, compute, communication, optimizer and native
lightweight logging are included. Initialization, cold compilation, warmup and
teardown are excluded. Profiling, validation, checkpoints, numerical capture,
monitoring and snapshots were off.

Warmup was not extended. Update 2 took 17.303 s; warmup updates 3–10 were
13.514–14.907 s. The measured window has ordinary variation and no recorded
compilation-scale stall. Optional compiler observation/guard stayed off; this
is not an exhaustive proof of compilation absence or a repeatability estimate.

Maximum sampled allocator peaks across eight ranks were **69.667068 GiB
allocated/active** and **70.181641 GiB reserved**. The sampler observes native
reset epochs at start, before resets and at end; a boundary epoch can include
adjacent warmup. These are not simultaneous device-wide peaks. Every rank had
zero allocation-retry and OOM counter deltas. Rank-0 native composite memory
was **72.088745–72.244995 GiB** in updates 11–20: reserved peak since its reset
plus report-time non-PyTorch memory (**2.555542 GiB**). This sum is also not a
simultaneous device-wide peak. The all-rank native console maximum composite
was **72.73 GiB** (rounded). All eight ranks logged finite loss/gradient values
for all 20 updates; rank-0 TensorBoard scalars retain full logged precision.
No numerical equivalence or convergence claim is made. GiB means 2^30 bytes.

## Reproduction and scope

Both runs use execution revision `16b13c2548317945712ded2e66f136b169130d95`.
GIFT/config/data bytes are unchanged from cleanup `eadcc6ac`; the necessary
runner repairs serialize already-captured wall timestamps with observation off,
keep scratch in short `.tmp/run-*` paths, and isolate Triton writers by global
rank within each attempt. No timed-window operation was added. Rank isolation
can duplicate artifacts and cold startup work; NFS scratch is required by the
user. The first attempt failed with Triton `ESTALE` before any complete update;
its compact failure receipt remains. The cause is unproven.

The baseline resolved-config SHA256 is
`94e32fa61596e4c3dd2e5f9d5582988fc9c3ef641c6b737e61194c72781fdb39`;
GIFT manifest SHA256 is
`3257ca6fd225668045ce3f2838c82da72bed121d8e9d43a4315dae028c8f365e`.
Metadata, commands, exact runtime/package identities, helper/config hashes,
all-rank sidecars, startup receipts and native scalars are retained under
`artifacts/raw/monitorfree-baseline-w10-m10-002/`. The existing MDS index SHA256
is `7cc8d53152c801450fdae44ba84c182eb8dcb9106d8bba9e3c3d69ebdaa8a8c2`.
FP8, compile, CuTe, SAC, MTP, EMA, fused AdamW, EP8, batch/accumulation, seed and
65,000/2,000 LR schedule are preserved. See [baseline defaults](../../docs/baseline.md).

This is one short run on a repeated 896-row local fixture and an eight-layer,
single-node crop. Noise confidence, production scale/data, convergence and
continuation are untested. Historical monitoring-inclusive TPS is retained below
and in explicitly historical results keys; it is not directly comparable and no
candidate optimization gain is claimed.

## Fresh traces and cleanup

`monitorfree-profile-update11-001` passed at the same execution revision.
All eight ranks completed 11 updates; only global ranks **0 and 1** constructed
actual profilers. The supported schedule used wait 0, warmup 1, active 1,
repeat 1, skip-first 9 and skip-first-wait 0: updates 1–9 were skipped, update 10
warmed Kineto and **optimizer update 11** was active. The sole trace annotation
`ProfilerStep#10` is zero-based; target task markers and `iteration_11` identify
the actual optimizer update. Profiled TPS was not used as the benchmark.

Each trace parsed and contains exactly 32 complete CPU microbatch markers,
one train-step marker and one optimizer marker, nested within that single
ProfilerStep. CUDA kernels are present (94,436 / 94,438), including 17 Adam
kernels on each rank. All eight target receipts record 32 microbatches and one
optimizer call. Shapes, stacks, profiler memory and snapshots were off; traces
contain no nonempty CPU input-shape or stack records or profiler memory events.
No checkpoint or numerical payload was created.

The profile resolved-config SHA256 is
`c1ac05654c4dfe72721b110dddb2221aab8bad016e001e3a18d53920994da19f`.
The [resolved-config difference receipt](artifacts/raw/monitorfree-profile-update11-001/resolved-config-differences.json)
confirms only profiler controls, training stop and output path differ from the
unprofiled configuration. Source/runtime/data identities are retained in each
run's metadata and the [results](results.json).

Canonical traces, relative to this campaign directory:

| Rank | Canonical trace | Compressed SHA256 |
| --- | --- | --- |
| 0 | [rank0_trace.json.gz](artifacts/raw/monitorfree-profile-update11-001/native/profiling/traces/iteration_11/rank0_trace.json.gz) | `f8532db191edb226385409ef5f707769db58c6db0bac47b07d1b24252d7c6360` |
| 1 | [rank1_trace.json.gz](artifacts/raw/monitorfree-profile-update11-001/native/profiling/traces/iteration_11/rank1_trace.json.gz) | `42a074fc3b1b9910d1057fda3d2bd4f0506a8756a03f4dbde9a2468917295111` |

The [verification/compression receipt](artifacts/raw/monitorfree-profile-update11-001/trace-verification-compression.json)
records original hashes/sizes and coverage. Gzip level 1, zero timestamp and no
filename header produced 30,393,413 / 30,383,635 bytes. Decoding reproduced the
original 297,079,146 / 297,064,044 bytes and SHA256 values exactly. Both redundant
plain JSON copies were removed after their parser consumers exited; one canonical
copy per rank remains. The original parse summary is retained unchanged and maps
to canonical gzip files through this receipt.

Owned training controllers/process groups have exited, and all eight GPUs show
0 MiB with no compute applications. The [cleanup receipt](artifacts/raw/monitorfree-cleanup.json) records removal of
75,313 obsolete cache/helper-bytecode files (2,339,828,158 logical bytes), both
controller scratch directories and the already-verified redundant plain traces
(594,143,190 logical bytes). These are not NFS reclaimed-space claims. The exact
run scratch paths were removed by the runner after group exit; no task temporary
children remain. A scoped process/environment check found no remaining owned
training, loader or compiler-cache consumer. MDS index identity is unchanged.
Compact evidence, candidate sources, the existing untracked V2 helper and the
historical canonical G3 traces are preserved. Stop here; no candidate experiment,
full-model numerical comparison or new bottleneck analysis was performed during
the measurement/cleanup assignment.

## Baseline input preparation analysis

Subsequent read-only Perfetto inspection of the monitoring-free baseline trace
`monitorfree-profile-update11-001`, rank 0, confirms exposed input preparation
before forward. The CPU train step lasts **14.337954319 s**. From its start to
the first CPU token-count all-reduce call, **0.972840740 s** elapses; underlying
DataLoader `__next__` calls account for **0.959384510 s** of that interval.
The [trainer](../../gift/torchtitan/trainer.py) collects all 32 CPU microbatches
and counts valid targets before reducing the global count and starting forward.
Existing two-worker prefetch does not materialize that whole burst in advance.

The first GPU `nccl:all_reduce` (Perfetto slice 666) starts at 0.973804118 s
and lasts 36.300705 ms. It sums one int64 valid-token count across eight ranks
for loss normalization; the host waits through `.item()` in
[`dist_sum` / `_dist_reduce`](../../gift/torchtitan/distributed/utils.py).
This communication remains necessary under V2.

This is a real overlap opportunity, **not a measured speedup**. Hiding the entire
pre-reduction interval with everything else unchanged would give a **7.3% TPS
increase for this traced step**. One DataLoader call alone lasts 0.786874459 s
(slice 306), at the expected epoch boundary; worker-side causality is not captured.
If that call remains exposed and all other preparation is hidden, the same
arithmetic gives only **1.3%**. Neither scenario predicts sustained V2 gain.
The [V2 contract](reference-coverage.md#queue-depth-and-what-the-trace-can-establish)
keeps epoch entry synchronous and cannot prefetch across that boundary. The
[first planned experiment](../../planning/README.md#first-future-experiment-v2-input-prefetch)
therefore starts with V2 and a queue of 32 CPU microbatches, with actual benefit
and queue memory still to be measured.

Follow-up results: the [completed V2 comparison](../v2-prefetch-baseline/report.md#long-logical-epoch-comparison--complete)
uses a long logical epoch and measures updates 21–60. V2 gained 2.12% input TPS
in that single matched pair. The owner subsequently
[accepted V2 at depth 32](../v2-prefetch-baseline/report.md#mcp-perfetto-comparison-and-acceptance)
based on that result and the MCP trace comparison; the report owns exact values
and limitations.

## Update 11: candidate analysis with Perfetto MCP

**Finding:** streaming prefetch has the clearest directly exposed opportunity.
The head/linear-CE boundary is the largest identified candidate compute block.
Capacity checks cause repeated host synchronization, but their host duration
mostly overlaps GPU work. No candidate speedup has been measured.

This read-only follow-up answers the new trace-analysis request. Acceptance:
verify the captured workload and coverage; reconcile the Perfetto import; measure
candidate costs and exposed waits; map every listed direction to evidence and
limits. No GIFT/config/runner changes, training, new capture, candidate A/B or
numerical block experiment were performed.

The [SQL](monitorfree-update11-perfetto.sql) and
[query results](monitorfree-update11-perfetto.json) preserve 53 actual MCP queries,
including failed and superseded queries. Final rank-0 GPU results use handle
`trace_fb8d93262d864d34`. CPU findings and complete CE counts from rank 1 provide
a narrow cross-check; there is no rank-1 whole-device budget claim.

### Workload and import verification

Source is execution revision `16b13c2548317945712ded2e66f136b169130d95`;
inspection HEAD is `b6610e60193319169f4349a41292dbac46964e41`, with no GIFT diff
between them. The resolved configuration and rank-0 native log confirm EP8 on
eight H100 80GB GPUs, dense/grouped FP8, compiled blocks/loss, CuTe norms, fixed
per-layer SAC, MTP1 weight 0.3, B2/S4096 and 32-way accumulation. Detailed
monitoring, evaluation, saving and graphs are off. All eight sidecars completed
11 updates and record 32 microbatches/one optimizer at the target update.
Runtime metadata pins PyTorch `2.15.0.dev20260822+cu130`, CUDA 13.0,
NCCL 2.30.7, DeepGEMM `2.3.0+f093a4c`, DeepEP `1.2.1+1b8f467`
and streaming 0.13.0.

Evidence is under
[the captured run](artifacts/raw/monitorfree-profile-update11-001/):
`native/config/resolved_config.yaml`, `metadata.json`,
`measurement/rank-00.json` through `rank-07.json`, and rank-0/1
`logs/torchrun/686f3b1b-6505-4fdf-b7c4-7ab1b64f08fa_o4sawdbv/attempt_0/*/stdout.log`.
Rank-0 stdout lines 128–149 record compile, SAC and HybridEP activation;
line 180 records finite update-11 loss 14.43901 and gradient norm 7.8928.

Direct import of the original rank-0 gzip silently omitted 417 CuTe kernels:
94,019 imported versus 94,436 raw, with
`slice_drop_overlapping_complete_event=417`. Raw same-stream CuTe timestamps
have small crossing overlaps. Removing GPU user annotations alone did not fix
this; that unsuccessful analysis copy and its receipt are retained.

The [preparation script](monitorfree-perfetto-prepare.py) places overlapping device
events on separate display lanes and moves their matching flow endpoints.
Actual `args.stream`, timestamps, durations, names, arguments, CPU events and
source gzip bytes remain unchanged; GPU display ranges are omitted from the
analysis copy. Reproduce with:

```bash
nice -n 10 python runs/overnight-streaming-perf/monitorfree-perfetto-prepare.py \
  runs/overnight-streaming-perf/artifacts/raw/monitorfree-profile-update11-001/native/profiling/traces/iteration_11/rank0_trace.json.gz \
  runs/overnight-streaming-perf/artifacts/raw/monitorfree-profile-update11-001/perfetto-analysis/rank0.device-lanes.json.gz
```

The [receipt](artifacts/raw/monitorfree-profile-update11-001/perfetto-analysis/rank0.device-lanes.json.receipt.json)
records hashes, 420 relocated device events and 420 relocated flow endpoints.
MCP then imported **all 94,436 kernels, 2,839 copies and 9,184 memsets**, with zero
error/data-loss statistics, no incomplete slices and a runtime/driver flow for
every device event. Stream statistics group by original `args.stream`, not
synthetic display lanes. The original rank-1 import drops 537 events; only its
unchanged CPU records and intact 128-per-name CE census are used here.

### Timing scopes

All values below describe **rank 0, one profiled optimizer update**, clipped to
the CPU train-step annotation. Interval unions count overlapping streams once.
Compute includes unclassified noncommunication kernels; HybridEP mixes transport,
permutation, reductions and peer waits.

| Scope | Time |
| --- | ---: |
| Integrated CPU train-step span | 14,337.954 ms |
| Device busy union: kernels, copies, memsets | 12,911.264 ms |
| Device idle inside that span | 1,426.690 ms |
| Compute/other kernel union | 10,791.650 ms |
| NCCL union | 2,328.472 ms |
| HybridEP mixed union | 1,327.828 ms |
| Memcpy union | 191.583 ms |
| NCCL with no other device work | 701.699 ms |
| NCCL or HybridEP without classified compute | 2,034.346 ms |

These rows overlap and must not be added. Inclusive CPU-operator totals also
nest and do not measure independent CPU work. Python stacks and CPU scheduling
samples are unavailable; host evidence consists of operator/API spans and
correlated launches.

The train-step span is within the separate unprofiled baseline's observed
13.542–15.804 s cadence range. This checks plausibility, not profiler neutrality.
The profile sidecar's update-11 return interval is **17.918 s**, including capture/
export effects outside the train-step timer. Use the existing unprofiled
145,163.84 global input tokens/s result for baseline magnitude. There is no new
monitoring-off A/A noise estimate.

### Ranked opportunities

| Candidate in docs/candidates.md | Observed evidence | Decision |
| --- | --- | --- |
| Streaming prefetch | **972.483 ms** initial all-device gap; loader calls total **959.385 ms** | First comparison to authorize |
| Linear projection + CE | Head GEMMs **2,193.849 ms** plus CE **195.220 ms** | Larger compute/memory boundary; investigate compatible fused implementations |
| MoE metadata/capacity | **448** predicate checks, **6,611.223 ms** nested host synchronization; **92.149 ms** coincident device idle | Concrete synchronization target; establish exposed effect before estimating gain |
| Kindulov CE algorithm | CE backward **102.887 ms**; all CE **195.220 ms** | Small budget; retain required numerical adaptation |
| GFSDP buffers/prefetch | AG kernel sum **1,138.830 ms**; **180.373 ms** without non-NCCL device work | Existing prefetch is active; narrow the remaining wait |
| GDN recomputation | Original/SAC forwards plus VJP and native-backward rebuilds visible | Proven repeated work; memory and compile contracts limit the change |
| MLA backend | cuDNN SDPA forward/backward **322.712 ms** | Lower priority; no faster backend demonstrated |

Kernel sums in this table are cost budgets, not removable elapsed time.
The detailed dispositions below cover the remaining listed directions.

**Streaming prefetch.** The GPU finishes its initial scalar fill, then has no
recorded work for 972.483 ms. The next HtoD launch starts 972.449 ms into that gap,
so this is not an already-submitted copy waiting behind GPU work. All 33
DataLoader calls occur before the first microbatch; call 18 alone takes
786.874 ms. The infinite loader can cross an epoch boundary during the trainer's
up-front collection of 32 microbatches; the extra call and long reset are
consistent with that mechanism, but no worker stack establishes its internal
cause. Rank 1 also records 33 calls, 838.785 ms total and a 685.991 ms maximum.
The MDS fixture has only 896 repeated full-length rows; do not generalize this
epoch/reset exposure to production data.

The trainer collects all microbatches before GPU forward/backward, so the
preserved bounded CPU producer could prepare across this boundary while the
previous update computes. DataLoader already has two persistent workers: adding
workers is not evidence that this trainer-level boundary disappears. Removing
the entire observed gap would reduce this trace's latency by **6.78%**, equivalent
to **7.28% more TPS** if every other cost stayed fixed. This is an ideal ceiling
for eliminating this gap alone, not a V2 result or a bound on all input-related
effects. Preserve ordering, exception propagation, queue
memory, loader/RNG cursor, Gloo/DCP ownership and exact resume; the V2 runtime/
performance gates remain open. HtoD device copies total only 0.885 ms, so transfer
bandwidth does not explain the nearly one-second input gap.

**MoE capacity/metadata.** Launch ancestry identifies
`select_capacity_routes -> conditional_capacity_mask -> is_nonzero -> item
-> _local_scalar_dense -> cudaStreamSynchronize`.
There are 448 checks, 6,656.050 ms inclusive host time and 6,611.223 ms nested
synchronization. Rank 1 has the same 448 checks and 6,612.258 ms inclusive time.
Only **92.149 ms** of rank-0 all-device idle intersects those predicate calls;
that coincidence is not a causal savings estimate. Much of the host wait is
previous GPU work draining. The observed host total therefore cannot support a
46% speedup claim. This matches the general synchronization behavior described
in [PyTorch's tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#avoid-unnecessary-cpu-gpu-synchronization).

There is a source-derived invariant for this exact crop: at most
`8 * 8192 * 8 = 524288` routed pairs exist across the EP group, while the
destination-group overload threshold is `(50000 // 8) * 128 = 800000`.
A destination's count cannot exceed the total, so the configured group-overload
gate is always false at this bound. This is an inference from the inspected
capacity implementation and fixed layout, not captured predicate tensor values.
It does **not** hold generally at EP32 or larger token bounds.

An existing default-off `branchless_capacity_drop` path accepts the crop's
65,536 routes. A future isolated comparison could evaluate that path while
retaining the same mask semantics; a construction-time invariant fast path would
require its own correctness review. Keep capacity 50,000, group gating, score
ordering and stable ties. Do not simply remove production token dropping.
HybridEP also has 416 dispatch synchronizations totaling 107.538 ms; those are a
separate dynamic-output-size dependency. Existing nonblocking dispatch requires
capacity/buffer proofs and does not automatically eliminate the capacity check.

**Loss candidates.** Correlated launch ancestry and the active loss loop identify
64 head forward GEMMs (759.772 ms) and two sets of 64 backward GEMMs
(715.958/718.120 ms). MTP folds two heads into the batch. There are 128 calls of
each of the three ordinary CE kernels: row statistics 91.580 ms, loss reduction
0.753 ms and backward 102.887 ms. Thus CE alone accounts for **1.36%** of the
14.338 s trace; even eliminating the entire backward kernel would cover only
**0.72%**. A replacement still has to calculate gradients. The remaining Kindulov
backward simplification is applicable, but the existing
[compiled contract](k1a-compiled-contract.md) already establishes fused BF16
logits/gradients, FP32 row statistics and a stability defect in the unchanged
donor's combined-LSE formula. This trace does not resolve that defect.

The larger linear+CE boundary is 2,389.069 ms, **16.66%** of trace time in summed
kernel cost, with matrix products that remain mathematically necessary.
This budget motivates research; it does not predict a 16.66% time saving.
Any implementation must retain FP32 accumulation, compiled loss, main/MTP
weights and masks, hidden-state gradient accumulation and head GFSDP hooks.
From the actual folded loop and resolved dimensions, each BF16 logits chunk is
`[4,2048,128256]`, **1.957 GiB**. Shapes were not captured. Completed-chunk
release can shorten this lifetime in source; no memory snapshot measures its
peak benefit, and the trace does not prove a speed benefit.

**GFSDP.** There are 736 NCCL all-gathers and 384 reduce-scatters.
Only 180.373 ms of AG is free of non-NCCL device work, versus 1,138.830 ms summed
AG kernel duration. Reduce-scatter has 374.683 ms in that exposure definition
and other NCCL 146.643 ms. These sum to the strict 701.699 ms NCCL-only union;
they are not all independently removable waits. AG and RS do not overlap in
this trace. RS is required by optimizer/buffer lifetime, not necessarily the
next layer, so exposed RS is not proof of a forward/backward stall.

The active parallelizer already wires explicit forward and reverse backward
prefetch chains for EP. CPU gather markers show 64 calls per dense unit and
32 for the head: **one head gather per microbatch**, across two chunks and two
heads. Expert-FSDP1 marker activity is not an eight-rank expert-weight collective.
There is no evidence for a redundant per-chunk head all-gather fix.
Evaluate a specific remaining consumer wait and retained-buffer memory cost
before porting donor buffer/prefetch code.

**GDN.** Six GDN layers times 32 microbatches give 192 logical calls.
Trace context shows 384 `gdn_chunk` forwards (original plus SAC replay), another
192 full forwards inside `fla_vjp`, then 192 native FLA backwards that rebuild
`w/u/h`. This agrees with `vjp_via_recompute` and the vendored FLA backward.
Three representative kernel families account for **62.300 ms** in VJP forward
rebuild and **43.388 ms** in native-backward rebuild. These are partial costs,
not the whole GDN block or a measured removable budget. Avoiding repeated work
requires a compatible saved-state/direct-backward boundary; do not disable SAC
or expose the FLA integer indexing stream to compile. A broader all-GDN
range-join query failed with an MCP RPC decode error; a concurrent aggregate
then failed to send. Neither supplies a measurement. The completed bounded
parent-context breakdown is retained; no further retry was needed for this
candidate disposition.

**Remaining directions.** cuDNN flash SDPA is already selected: 192 forward and
96 backward kernels total 322.712 ms for the two trunk MLA layers and MTP layer,
including replay. Conversion costs have not been uniquely attributed; do not
assign generic concat/copy kernels to MLA. CuTe norms are active and cost
546.187 ms over 13,664 kernels; enabling CuTe is not a new optimization.
Residual-boundary fusion still needs an identified unfused operation.

Dense and routed SwiGLU/FP8 fusion is already active. The gate/up z-loss
coefficients are zero; only norm-input regularization is selected. The hidden
z-loss donor therefore changes the objective if enabled unchanged, and its
pinned external CudaCherry source remains missing. Local dense/shared FFN
recomputation trades memory for extra GEMMs under existing SAC; routed donor
recomputation rejects required FP8. Neither has demonstrated speed benefit.
CompactCE rejects MTP; the actual baseline's main labels are already 99.9756%
supervised (only 1/4096 ignored). All eight rank sidecars report zero frozen
parameters. Frozen storage and metric-accounting changes have no demonstrated
new saving here.

### Decision and remaining evidence

The first comparison worth running is a matched short monitoring-off baseline/V2
pair if this fixture remains the intended target. In further research, prioritize
a compatible linear+CE boundary and the exact MoE synchronization dependency. This ranking
does not authorize source edits or compute. Keep candidate numerical, peak-memory,
distributed/resume and repeated unprofiled timing gates open under
[validation](../../planning/validation.md).

The final source/config/runner bytes are unchanged. Diagnostic normalization was
verified through the actual MCP import and full device-flow counts. The original
gzip hashes remain canonical. No candidate is accepted and no speedup is claimed.

Verification passed: original rank-0 gzip SHA256; event-by-event equality of all
retained payloads and trace metadata except exactly 420 device/420 matching flow
display tids; full normalized MCP census/launch coverage and empty importer error
statistics; both maintained JSON files; 126 maintained local links/fragments;
`ruff check --select F821,F811,F841 --no-cache` on the preparation script;
`git diff --check`; and an empty diff for `gift/` and the training runner.
The two optional GDN aggregate RPC failures remain in the query evidence.

---

# Historical cleanup and original campaign record

# Streaming performance campaign: cancellation and cleanup

Status: ORIGINAL_CAMPAIGN_CANCELLED_INCOMPLETE; cleanup assignment COMPLETE.
The [active plan](../../planning/overnight-streaming-perf.md) owns the finite
cleanup checklist. No candidate is accepted. Do not resume G0–G7 or launch any
training, benchmark, trace or local numerical block check during cleanup.

## Revised protocol and cleanup receipt

The last campaign execution source was `e9048f65`. Remove the campaign full-model
numerical modules and exclusive integrations/tests/config, and disable detailed
monitoring callbacks/hooks in the performance configuration. Ordinary two-chunk
loss, lightweight loss/time/TPS/errors, EMA and training losses remain. Preserve
the fixed MDS/model, FP8, compile, CuTe, SAC, MTP, optimizer, routing and batch.
Validation, checkpoint saves, snapshots, numerical capture and CUDA Graphs stay off.

Future separately authorized measurements start with 10 warmup + 10 measured
optimizer updates, accumulation 32 and LR horizon 65,000/warmup 2,000 unchanged.
Extend warmup only for actual compilation/runtime settling affecting the window;
a cache entry alone is insufficient. No mandatory 110/50, exhaustive-cache or large
A/A or A/B/A/B matrix remains. Begin with a matched short baseline/candidate pair and
label it provisional; repeat promising/ambiguous effects when needed.

Performance comes first. A candidate worth retaining then receives a local check
of the smallest affected block on matched boundary inputs and needed weights/
state, plus affected backward with matched incoming gradients if changed. Assess
accumulation precision, casts, rounding, reductions and fusion, using a higher-
precision local reference when needed. Baseline output is not assumed exact;
there is no universal tolerance, assumed bitwise equality or whole-model
parameter/gradient/Adam capture. See [validation](../../planning/validation.md).

Historical monitoring-inclusive G2 median input TPS was 124454.81175967702 global
and 15556.851469959627 per GPU. **It is not directly comparable to future
monitoring-off measurements.** MDS preparation, G2 measurements, G3 Perfetto
SQL/results/compressed traces, candidate code and useful compact evidence remain.

C1 B2 passed; compact history is retained. C2 A reached terminal PASS export
at 08:34:48 UTC; child 418189/process group stopped and controller absent at 08:45.
C2 B and C50 were cancelled without launch. Full-model validation was abandoned,
not completed. V2 source `21c1891d`/`vkropotin-v2-prefetch` remains preserved and
default 0; actual eight-owner Gloo/DCP and performance gates were not executed or
completed. Its source/CPU handoff is not runtime acceptance.

| Current cleanup acceptance | Receipt |
| --- | --- |
| Numerical removal/import/parser checks | PASS: modules, full-model probe and exclusive tests/config removed; imports/help and removed-mode rejection verified |
| Short window/disabled monitoring/preserved config checks | PASS: 100 scoped CPU/config tests plus 1 canonical golden; one unrelated native-DCP test deselected; independent integration/runtime-cost review passed |
| Exact stopped jobs/heavy numerical payload removal | PASS: [receipt](cancellation-cleanup.json); zero owned numerical processes and reference blobs; 40,650 files / 381,915,112,855 logical bytes removed, plus two obsolete clean worktrees; no NFS reclaimed-byte claim |
| Affected document links | PASS: 176 local links across 20 affected documents resolve; 2 fragment links pass; 3 unchanged upstream root-relative links are excluded and recorded in results; no new broken link; JSON/whitespace checks pass |
| Commit/push | Cleanup commit `aa45e3cd8d0c8856d383e72a9ad5f5d60c0530ae` pushed to `origin/main`; this receipt update records completion |

Verification commands, intermediate failed checks and their resolutions are retained
in `artifacts/raw/cleanup-verification/runner/CHECKS.md`; the independent review is
`artifacts/raw/cleanup-verification/integration-review.md`. The new configuration
regression fails on the original monitored YAML and passes on the final YAML.
No full suite or GPU execution is claimed. The final path removes numerical
exports and diagnostic hooks/reductions; it retains native logging, EMA and two
timing-boundary synchronizations, with no new model computation or communication.
The GIFT operating instructions also replace their universal bitwise-loss rule
with the local, precision-aware post-performance policy.

The unchanged historical smoke launcher requires its recorded original source
and configuration; current MDS/monitoring-off defaults do not reproduce its
receipt. MDS index and both canonical compressed G3 trace hashes still match;
V2 source remains clean at `21c1891d`. The existing untracked V2 loader-check helper
is preserved without adding it to this cleanup change set.

## Archived original campaign record

**Everything below is dated historical evidence, including its proposed next
steps, numerical prerequisites, monitoring-inclusive policy and old 110/50/cache
method. Those instructions are retired and do not authorize or constrain future
runs. The current cleanup plan and validation policy above replace them.**
Original failures and incomplete gates are preserved; the archive does not claim
the original campaign finished. Heavy numerical payload locations may describe
removed historical files; compact evidence is retained where useful.

## Entry and environment

Execution started 13 November 2026 02:18 Europe/Moscow (12 November 23:18 UTC)
at clean task `main`, revision `a2a59e14545b55ca777f94a28e332759e7c06087`.
The existing single master exposes eight idle H100 80 GB GPUs. The task path
is on SR008 NFS with approximately 2.5 TiB free; no new allocation was made.
Runtime preflight verified `/opt/torchtitan-venv/bin/python3.12`, torch
`2.15.0.dev20260822+cu130`, CUDA 13.0, NCCL 2.30.7, streaming 0.13.0,
transformers 4.57.1, CUTLASS DSL 4.4.1 and TVM FFI 0.1.11 after sourcing
`/etc/profile.d/wm.sh`. Shared packages remain unchanged.

Perfetto MCP successfully loaded a synthetic Chrome JSON at the actual task
absolute path and executed `SELECT name, ts, dur FROM slice`; it returned
`campaign_route_check`, 1000000 ns, 100000 ns. This verifies tool/path access,
not the later required training-trace analysis. No private server is needed.

## Scope and interpretation

Preserve the fixed crop, precision, SAC, MTP1, EMA, optimizer, seed and scheduler.
MDS is a new data baseline because HF and MDS packing differ. Performance will
include input and monitoring at interval 50, excluding validation and checkpoint
saves. Short runs remain in optimizer LR warmup. No convergence, continuation,
multi-node or production-scale performance conclusion is implied.

## Results

Native MDS smoke retry `g1-streaming-smoke-002` passed at source `4d7689aa`:
ten optimizer updates and 320 successful forward/backward calls on each rank,
finite metrics and native monitoring at 5/10. Final loss was 14.57432 and
gradient norm 7.6163; maximum native composite memory was 72.57 GiB.
The [separate receipt](g1-smoke-002-receipt.json) verifies unchanged source/config,
clean teardown, no diagnostic outputs and released GPUs. That smoke has no
performance measurement; the G2 A/A TPS baseline is reported below. First
attempt001 remains a failed harness attempt, not a
candidate verdict. Source applicability is recorded in the
[reference evidence](reference-coverage.md); candidate runtime gates remain open.

The smoke's rank0 host boundary intervals were about 13.5–14.8 s for ordinary
updates and 173.5/166.7 s for monitoring updates5/10. These are unsynchronized
smoke cadence observations, not baseline TPS or CUDA kernel time. They reinforce
the required normal-cadence calibration through update50 before choosing W;
the 32-chunk shape and monitoring work cannot be omitted from the controls.

## MDS fixture and real loader contract

The [data receipt](mds-data.json) owns corpus/tokenizer, installed streaming
implementation, row and shard hashes and exact check commands. The fixture is
896 full 4096-token uint32 rows in 29 uncompressed shards, made from the first
3,670,016 BOS/text/EOS tokens of the same local C4/tokenizer as the HF smoke.
Index SHA256: `7cc8d53152c801450fdae44ba84c182eb8dcb9106d8bba9e3c3d69ebdaa8a8c2`.
The [helper](prepare_mds.py) refuses existing output before calling the destructive
packer. The [loader fragment](mds-dataloader.yaml) selects installed streaming0.13,
two persistent workers/rank, shuffle/FIM off, canonical nodes1, infinite iteration
and no pinning. After native smoke passed, the canonical experiment's data
mapping was promoted, with all non-data values preserved. Its affected golden
test passed (1 passed,60 deselected) after regeneration and again in ordinary
check mode. Relative `local` resolves from the normal `gift/` working directory
to the same index and all29 shard hashes. Native002 used the equivalent absolute
external fragment; canonical bytes have decode/path evidence, without another
GPU smoke.

Two fresh actual eight-rank Gloo runs each validated 58 batches/rank, including
the 56-batch epoch wrap. Both passed exact row hashes/IDs, all16 worker partitions,
global coverage, stable worker PIDs, shapes/dtypes, EOS-derived positions, label
shift and final ignored labels. Each microbatch has 8190 valid main targets;
global per-update numerators are 2,097,152 input and 2,096,640 valid main targets.
No MTP token multiplier is applied. The separate native GPU smoke passed; this
does not establish parity with the differently packed historical HF input.

Pass1 exited0 but retained 16 Python multiprocessing finalizer Errno16 warnings
from the provider's NFS TMPDIR. Pass2 used local tmpfs and had none. The runner
will create and remove only its uniquely owned local temp directory; all data,
compiler caches and persistent logs stay in the task NFS area. No shared temp
cleanup was performed. Final helper, packer, data-world, loader-fragment and index
hashes were checked against the receipt after worker handoff.

Native resolved/source correlation identifies an important position distinction:
the fixed SDPA attention selects causal, non-varlen behavior and propagates that
mask type to GDN. `Trainer.prepare_inputs` replaces the collator's EOS-derived
positions with int32 per-row `arange` for this configuration. Thus the data check
proves the collator contract, while actual model attention/scan state spans
documents within each packed row. It is causal within rows, without EOS document
isolation. Keep this existing model behavior; switching to block-causal/varlen
would change the identified baseline. Exact native per-batch partition counters
are unavailable; the prior actual eight-rank Gloo loader check plus native mesh
and source correlation support rank partitioning.

## G2 calibration

`g2-calibration-001` passed60 native updates/1920 microbatches on every rank,
with exactly32 calls/update, finite loss/gradient logs, clean teardown and released
GPUs. Source/config stayed unchanged; no checkpoint/trace/snapshot was emitted.
Raw evidence is under `artifacts/raw/g2-calibration-001/`; resolved config SHA256
is `b88ffc3da42a7507fa47bb33f6f54fb22d023fa5bc0978338fce6fa9d3e9b27a`.
Helper bytes used by calibration are preserved at `02cfb427`.

Rank0 host boundary cadence medians (seconds):

| Updates | Median |
| --- | ---: |
| 2–10 | 13.7441 |
| 11–20 | 13.8981 |
| 21–30 | 15.3566 |
| 31–40 | 15.3071 |
| 41–49 | 14.9610 |
| 50, monitoring | 173.9149 |
| 51–55 | 14.9717 |
| 56–60 | 15.0141 |

Updates51–60 range14.5722–15.5668s without continuing monotonic settling;
maximum same-update host-interval spread across ranks was1.843ms. Recommend W60,
then measure all updates61–110 including monitor100. This is not TPS evidence:
the calibration has no synchronized50-update window. Step1 took259.828s,
including245.228s in its first forward/backward. Monitor50's first microbatch
took8.477s, then median4.8147s; its173.9s cannot all be assigned to compilation.
Diagnostic compiler logs were disabled, so later compilation is unknown.
Monitor100 and surrounding updates must also be checked directly in A1/A2.

### Compilation gate for measured controls

The task-only observer samples existing CPU state at the two synchronized
window boundaries, outside each local timer. It reads unbounded Dynamo compile
timing lists, existing Inductor async-cache counters, the active CuTe compiled
object map and existing Triton JIT device caches. It never indexes a missing
Triton device cache, inserts a defaultdict key, logs per step or synchronizes
the GPU beyond the two existing window boundaries. Boundary scans can still
contribute to rank-start skew in the global elapsed envelope. Runtime-only
Dynamo timing growth is allowed; new compilation/cache entries invalidate the
window. This is scoped evidence, not visibility into every asynchronous worker.

Each rank retains raw snapshots, source pins and the computed comparison;
the parser recomputes it and rejects missing or inconsistent results. Independent
review found and resolved two evidence bugs: identical malformed snapshots
could pass, and rank receipts initially lacked their comparison. Both original
reproductions now fail closed. Final observer36 and existing measurement26 tests pass,
with the task `gift/` explicitly on `PYTHONPATH`; the initial default-image import
test failure is retained as a harness error. Actual CPU snapshots, scoped Ruff,
compilation and whitespace checks pass. `run.py` pins the observer helper hash.

The installed DeepGEMM extension has no exposed compilation counter. Its actual
binary and active warmup code identify `DG_CACHE_FORCE_SKIP=0` as a cache-miss
rejection guard; the binary reads that environment value on each compiler call.
A serial GPU probe passed on13 November02:21 UTC. Both actual `m_grouped` and
`k_grouped` paths compiled A with flag1, reused A with flag0 without new cache
entries, rejected a different absent configuration B with literal `CACHE MISS`
and unchanged cache inventories, then compiled B with flag1. The installed build
was `2.3.0+f093a4c`, extension SHA256
`99b7afad2e2fe7ebabd26c22afc99dc652e2e6935fcbf37324359eb5aeeefe26`.
The frozen [probe](probe_deepgemm_guard.py) used serial GPU0 children and fresh
private caches; both exited0 and all eight GPUs were released. Raw receipt/logs:
`artifacts/raw/g0/deepgemm-guard-probe-001/`. This proves guard/cache behavior,
not numerical parity or speed. Matching full fork source is unavailable, so
upstream behavior is not substituted for installed-binary/runtime evidence.
The thin adapter sets literal0 at W60 after synchronization and before the
timer, preserves it through measurement and records both boundaries on every rank.
It verifies the actual extension build/hash before training and restores the
activation-time environment value on adapter exit, including errors. The profile
run uses the same guard policy with one explicit synchronization at W, before the
target update; this supersedes the earlier capture-only zero-sync cost statement.
Plain smoke/calibration mode does not import or activate the guard. Parser checks
require the manifest policy and complete, matching guard receipts from all ranks.
Independent integration/runtime-cost review accepted this path; its final finding
was missing `previous_value` being accepted as explicit null. The validator now
requires presence; the regression reproduced the false pass and all5 affected
cases pass. Root inspected the small fix. Post-run probe lint found E731 in a
planner lambda; an equivalent local function now passes Ruff and produces the
identical CPU plan. Exact GPU-tested script bytes remain with the raw receipt.
The first staged whitespace check found trailing blank lines in two newly added
files, which the earlier unstaged check had not included. They are removed;
executable statements are unchanged and the final helper hash is in results.
A1/A2 and TPS remain open.

### First measured control: invalid window

`g2-baseline-a1`, source `72a814f2`, completed110 native updates, but its runner
failed required acceptance at03:24:19 UTC. The direct observer found new Triton
cache state on ranks0,2,5 during61–110; all eight DeepGEMM guard receipts were
complete with literal0 at both boundaries. A2 was not launched. This window is
not an accepted baseline or speed claim. Exact changed-key and settling diagnosis
is active; raw failure, summary and rank snapshots remain under
`artifacts/raw/g2-baseline-a1/`. Rank0 monitor100 took169.400s; ordinary intervals
also drifted downward over the window, so source/cache interpretation and actual
warmup selection remain open rather than selecting fast steps after the fact.

Read-only diagnosis resolved the only additions as forward/backward fused FP8
SwiGLU specializations in the padded routed-expert path: the runtime token bucket
`total_tokens // 2048` changed from an unaligned int32 signature to Triton's
divisibility16 attribute `D`. Hidden1536, launch options and all other signature
fields match. Dynamo/Inductor/CuTe state and Triton registry/function/cache
identities stayed stable. The run's shared disk metadata, TTIR, PTX and cubin for
both aligned kernels predate the window; installed Triton returns `CompiledKernel`
on a disk-metadata hit. Disk loading is strongly supported; actual new device
compilation during the window is unproven. No compiler listener was recorded.
The before/after snapshot is inside training, before exit; stop/export
contamination is unsupported. Keep the conservative cache-stability criterion.

All eight ranks hold both bucket-alignment classes by110. Rank0 ordinary medians
were14.7735s at61–70,14.2267 at71–80,13.9576 at81–90,13.7936 at91–99 and13.8516
at101–110. The trend's routing/clock/warmup cause is unresolved, while the final
tail is stable. Reuse this run as additional calibration and select **W110,
updates111–160, monitor150** symmetrically for fresh controls. No kernel/observer
change or feature removal is required; first-control failure stops before A2.
All110 native updates and finite metric pairs completed on all ranks; source and
config hashes match, retries/OOMs were zero, owned temp/group stopped and all GPUs
were released. Its unaccepted diagnostic timing/memory remain in results with
the explicit failure label. No baseline TPS has been accepted.

After that diagnosis, the obsolete calibration001 compiler cache was removed:
9,217 files /602,133,276 logical bytes. Stop/ownership/path/symlink checks passed;
all non-cache artifacts remain unchanged. Its cleanup receipt is
`artifacts/raw/g2-calibration-001/obsolete-cache-cleanup.json`. The failed A1's
cache evidence and the new control's independent fresh caches remain retained.

### Accepted W110 baseline and identical-control variation

`g2-baseline-a1-w110` PASS: all eight ranks completed160 finite updates; raw
snapshots show stable observed compiler/cache state and guard0after110 through160.
All50 updates111–160 include monitor150. Complete-window global inputTPS is
**124,500.985** (15,562.623/GPU), supervisedTPS124,470.589 (15,558.824/GPU), using
104,857,600 physical and104,832,000 valid tokens over842.2231s. Source/config
hashes match; temporary resources and GPUs were released. Root inspected the raw
all-rank comparison/guard/counter records, not just the launch settings.

Ordinary host median13.7997s, range13.4381–14.4012; monitor150 median167.1280s.
Rank0 successive ordinary-bin medians111–120/121–130/131–140/141–149/151–160 are
13.8784/13.8527/13.6500/13.7850/13.7915s: no continuing monotonic settling.
Allocator peak maxima are67.401GiB allocated and68.467GiB reserved, native reset
scope including the101–110 tail, with zero retries/OOMs. These exclude non-Torch
device memory. Start/end rank skew1.698/0.596ms; primary elapsed uses the global
envelope, with only two boundary synchronizations.

The identical `g2-baseline-a2-w110` also PASSed, finishing05:23:46 UTC with all160
finite updates on every rank. Source `87230b5b`, initial helper/config/data hashes,
runtime and fresh-cache policy match A1. Root verified all eight raw window,
counter, observer and guard records. Both runs include all50 updates111–160 and
monitor150; ordinary cadence shows no continuing monotonic settling.
The actual `native/config/resolved_config.yaml` files differ only in each run's
`dump_folder`. Their SHA256 values are
`aa2c792ce61e88717c2171f6fad8e8d4f20f8a5b54675de2887524b5305a7007` (A1) and
`099734af6018ecbe8d7b2b67337a43b82f68628e5c7651de592bfc995768f2e1` (A2), matching
the completion receipts. `source_config.yaml` is a separate artifact.

| Complete-window metric | A1 | A2 |
| --- | ---: | ---: |
| Global input tokens/s | 124,500.985 | 124,408.639 |
| Global valid-target tokens/s | 124,470.589 | 124,378.266 |
| Global elapsed seconds | 842.223058 | 842.848222 |
| Ordinary host median seconds | 13.799678 | 13.789450 |
| Ordinary host range seconds | 13.438092–14.401202 | 13.423635–14.463464 |
| Monitor150 host median seconds | 167.127963 | 167.607337 |
| Maximum allocated/reserved GiB | 67.401 / 68.467 | 67.387 / 68.494 |
| Allocator retries / OOMs | 0 / 0 | 0 / 0 |

The fixed MDS baseline is the run-level median: **124,454.812 global input
tokens/s (15,556.851/GPU)** and **124,424.427 valid-target tokens/s
(15,553.053/GPU)**. A2 versus A1 is−0.074173%; the two-run range divided by
their mean is0.074200%. These are descriptive A/A observations, not a confidence
interval or guaranteed bound on future variation. No candidate gain is accepted.
Memory uses native allocator reset epochs spanning101–160, including the warmup
tail; it is not an exact-window or simultaneous device-wide peak. Validation,
checkpointing, profiling and numerical payload capture are excluded from TPS.

A2 rank0 ordinary-bin medians111–120/121–130/131–140/141–149/151–160 are
13.930675/13.805102/13.638698/13.816832/13.787526s. Start/end rank skew is
4.619/0.774ms. Both runs have stable observed compiler state and tested DeepGEMM
guard coverage; asynchronous compiler-worker completion and native JIT paths
outside the probed grouped APIs remain unverified. Both process groups stopped,
owned temporary directories were removed and GPUs released; private run caches
were subsequently removed with verified receipts; raw run evidence remains.

## Native profiler controls

Implemented generic opt-in `profile_ranks` and `record_shapes` controls in the
native typed profiler. Defaults remain all ranks and shape capture enabled;
this campaign selects ranks 0/1 and disables shape capture. Nonselected ranks
return before profiler or postprocessor construction. Native schedule, exports
and full-update callsite remain unchanged. With profiling disabled, no new
per-step runtime work is introduced. Stacks and profiler-memory capture are
explicitly false; memory-history snapshots remain a separate disabled feature.

Focused checks from `gift/`, using the existing interpreter and task source:

```bash
python -m pytest tests/unit_tests/test_profiler.py tests/unit_tests/test_config_torch_free.py::test_profiler_rank_and_shape_controls_decode_torch_free -q
python -m pytest tests/unit_tests/test_experiment_parity.py tests/unit_tests/test_model_arch_presets.py -q
```

Results: 21 and 112 passed respectively; scoped F821/F811/F841, default Ruff,
formatting and whitespace checks passed. Goldens were regenerated through the
native update mechanism, then checked without update mode. The CPU native
schedule test exported only update 6, with 32 microbatch markers and optimizer
coverage. It does not establish CUDA tracing. A broader torch-free run retained
one pre-existing registry mismatch (`campaign_numerical_probe` is registered but
missing from the expected callback set); no registry files changed.

The [independent review](profiler-review.md) accepted the source handoff and
verified all 56 golden files contain only the two expected default additions.
The selected-rank baseline trace integrity and Perfetto analysis are now accepted
at G3. Numerical equivalence and candidate/lifecycle acceptance remain open at
later gates. Full-tree autofixing hooks were not run across
the flattened task tree because their source-root path filters do not protect
its byte-preserved reference and contrib snapshots; changed-file lint was used.

## Native runner review and prelaunch repairs

No GPU run was started before review. The independent reviewer found two
material integration defects: a missing measured-window completion attribute,
and inherited DeepGEMM startup overriding private cache variables with hardcoded
`/temp/.cache/rank_N/deepgemm` paths. The task adapter now redirects the actual
warmup helper before Trainer construction and fails closed on alternate shared
cache paths, retaining native warmup and source/package bytes. Five focused
tests passed, including actual setup/share-path sentinels for ranks0–7 and patch
restoration, full W..W+50/two-sync timing, incomplete-window rejection and smoke
cadence/microbatch counting. The parser passed a synthetic eight-rank global
envelope/counter check; actual measured-window acceptance remains at G2.
The host CPU core limit was unlimited; task child processes set it to zero
and clear inherited diagnostic/core-dump flags. The
[review](runner-review.md) owns findings and their final resolution. These are
prelaunch harness/inherited-path repairs, not failed candidate measurements.

### First native attempt: census failure

The [runtime receipt](g1-runtime-receipt.json) records exit1 on all eight ranks:
`parameter_census` calls `RaggedShardDTensor.to_local()` with grad enabled,
which this wrapper explicitly rejects. Its same method returns the local shard
under no_grad. Zero native updates or microbatch calls occurred; at that failed
attempt, finite loss, monitoring and execution gates were still open. The scoped repair was to run metadata
census under no_grad, retaining original trainability and restoring caller grad
mode, then test that real method guard before retrying.

The correction is implemented and accepted by targeted independent review.
Two affected tests pass, including the actual ragged-shard method guard and
success/error grad-mode restoration. The exact historical census loaded from
Git into memory reproduces the old exception; the fixed implementation counts
the existing local shard. No model arithmetic or training-source bytes changed.
After owned resources were released, the obsolete failed-run compiler cache was
removed (23,453,549 logical file bytes); logs and compact evidence are retained.

Initialization did verify the resolved fixed contract, all-rank FP8/SAC/compile/
GFSDP/streaming/MTP/EMA builds and native DeepGEMM warmup (39 kernels/node),
with all rank caches under the attempt's private tree and child core limits0.
These are initialization observations, not executed CuTe/model kernels or
training correctness. Source/config/helper hashes stayed unchanged; the owned
group stopped, local temp directory was removed, GPUs were released, and no
checkpoint/trace/snapshot was emitted. Logs and compact failure evidence remain.

### Active compiler-cache scope

TorchInductor, Triton, DeepGEMM rank caches and HybridEP use the run's fresh NFS
cache tree. CUTLASS DSL4.4.1 additionally caches generated MLIR under
`CUTE_DSL_CACHE_DIR`, or by default `$TMPDIR/$USER/cutlass_python_cache`
(`cutlass/base_dsl/cache_helpers.py:66–82,235–306`). Root verified that
`CUTE_DSL_CACHE_DIR` and `TVM_FFI_CACHE_DIR` are unset after the image profile;
thus active CuTe IR cache uses each run's unique private tmpfs directory and is
removed with that directory. Its `COMPILED`/`PLANS` maps are process-local.
The active CuTe JIT exports a function pointer to TVM-FFI rather than invoking
`tvm_ffi.cpp.extension`, so that extension builder's persistent default cache
is not the norm path. This qualifies the cache policy without changing kernels
or introducing another experiment. Shared/personal unrelated caches are intact.

### G3 baseline capture and Perfetto analysis

Run `g3-baseline-update111-001` completed111 native updates on all eight ranks
with finite metrics, all guard values0, source/helpers frozen and clean teardown;
all GPUs were idle afterward. The profiler was constructed only on ranks0/1.
Sidecars on all ranks attest one target train step,32 microbatches and one
optimizer call. No throughput measurement or numerical diagnostic mode was
active. Source revision was `6123a24f`; the runtime training code matches the
G2 control pin `87230b5b`.

The original runner result is still recorded as exit1: its strict parser counted
CPU logical annotations and GPU projections by name alone, reporting2/64/1 on
each trace instead of the sidecar's1/32/1. This parser failure is immutable
history; no retraining or trace recapture was done. The category-aware post-hoc
reparse passed: CPU `user_annotation` contains exactly1 train step,32 microbatches
and1 optimizer step per selected rank; separate `gpu_user_annotation` projections
contain1 train step,32 microbatches and no optimizer projection. Neither trace
has incomplete slices or a microbatch outside its CPU train frame. The original
summary/failure and the corrected [reparse summary](artifacts/raw/g3-baseline-update111-001/summary.reparsed.json)
and [receipt](artifacts/raw/g3-baseline-update111-001/reparse-receipt.json) remain.

Actual Perfetto MCP loaded both traces and completed the accepted analyses. The
[canonical result](g3-baseline-perfetto.json) preserves the original loaded paths
and trace IDs; its current artifact map points to the two gzip files after full
stream roundtrip size/hash checks. The pair was reduced from602,500,095 raw bytes
to45,573,265 compressed bytes, with556,926,830 logical bytes removed. The
compression receipt records identical pre/post hashes for all other raw-run
artifacts and zero immediately observed NFS available-byte/inode change. The
approximately604,978,569-byte,9217-file profile run cache was preserved when
the compression receipt was written. The four minimal compiled CE wrapper
sources are now preserved for the K1a contract. Separate [cache cleanup](artifacts/raw/g3-baseline-update111-001/obsolete-cache-cleanup.json)
completed:9217 files/604,978,569 logical bytes removed; the four exact wrappers
retain73,504 bytes plus their manifest. Raw evidence and canonical traces stayed
unchanged; observed shared NFS capacity/inode deltas were0.

The trace scopes are rank-local. Every kernel, memcpy and memset event falls
inside the exact CPU train-step frame; GPU busy unions were13,044.218/12,996.754ms
on ranks0/1. Optimizer flow analysis links17 fused AdamW kernels per rank.
A607.371/643.908ms interior gap precedes the input H2D copy and
overlaps host loading. Update111 crosses the 56-batch local epoch after8 old
batches; 33 underlying loader calls are consistent with32 deliveries plus the
epoch restart. This makes safe streaming prefetch the first candidate to inspect,
but does not confirm a gain or show that the full gap is removable. Strict
NCCL-exclusive union is714.427/694.566ms after excluding memset overlap; the
historical broad result remains in the JSON. There is no cross-rank clock mapping
or complete stream-dependency DAG, so neither union proves critical-path time.

The compiled CE already avoids full-vocabulary FP32 intermediates. Its ordinary
baseline kernels total about201/198ms per rank in overlapping kernel-duration
sums. Removing the baseline's one-hot-gradient reduction remains an independent
applicable donor algorithm candidate; no port or gain is established. See the
[compiled CE contract](k1a-compiled-contract.md). Candidate order and remaining
checks are recorded in the canonical JSON and reference table. One profiled
update provides mechanism evidence only, not TPS, monitoring cadence cost,
convergence or candidate acceptance.

## Reused validation interfaces and limits

The historical `CampaignNumericalProbe` is not an unchanged validation route for
this campaign: its trainer schema requires `hf_text/c4_test`, zero workers and
synchronous full checkpointing (`gift/torchtitan/config/schema/experiments/cuda_graph_campaign/trainer.py`).
It also persists multiple full tensor families. Do not switch this streaming
campaign to that trainer or generate its full-model payloads merely to reuse
the old probe. Candidate checks must exercise their actual affected loss, state
or lifetime path with matched inputs and elementwise gradients/state, then native
integration; any unresolved full-workload numerical gate stays explicit.

### K1 implementation contract (design only; no candidate enabled)

Keep completed-chunk release and fused CE independently selectable, default off.
The release point is the existing chunk-loop tail after backward/gradient
collection, preserving head calls, argmax, components, probes and flag restoration.
For CE, select a cap-zero callable before `_maybe_compile`, retain current
normalization and weighted MTP, and route DTensor to the current distributed CE.
Do not copy the donor loss class or change Unembedding. The pinned donor's
softcap tests do **not** test fused CE directly, so its CUDA/Inductor autograd
path needs new evidence, including V128256, signed int64 ignored labels,
all-ignored rows, noncontiguous inputs and explicit higher-order rejection.

Release tests must prove output weakrefs expire before the next projection and
compare the actual current chunked A/B paths through gradients and two AdamW
updates (parameters, moments and steps), ordinary/monitoring chunks, MTP,
evaluation/probes, FSDP protocol and exception restoration. For fused CE, first
freeze baseline error/variation against an FP64 CE/analytic-gradient oracle at
the same quantized logits and output cast; compare elementwise downstream
gradients/state through the same projection. Different reduction rounding is
not subject to an invented percentage or unjustified bitwise gate. Any
uncalibrated difference stays unresolved and the feature stays disabled.
Tiny tests do not replace fixed-crop native GPU/SAC/GFSDP integration.

### V3 partial-backward contract (design only)

The isolated donor guard belongs at the start of the group loop in
`GFSDPState._root_post_backward_final_callback`. During nonfinal inner backward,
skip groups outside PRE_BACKWARD/POST_BACKWARD; preserve final cleanup. Current
MTP already folds its hidden states into the chunk loop and sets shared
`is_last_backward=False`, so no loss-class transplant is needed. Prefetch borrows
PRE_BACKWARD temporarily and restores IDLE: an untouched IDLE group can still own
a gathered buffer/pending all-gather. Regression coverage must preserve that
ownership through repeated head chunks until the group's actual backward.

Keep this default-off guard independently selectable from arena/tail retention.
It adds Python state checks per group/inner backward and may remove premature
reshard/regather, while extending existing buffer lifetimes; collision/overlap
safety needs native evidence. Existing explicit EP8 chains, double buffering and
head private gradients are already present. Startup shard slicing is an
applicable initialization optimization with no warmed-runtime effect from the
inspected mechanism; do not credit it as stable-step TPS or silently drop its
initialization-equality gate.

Current loss restores incoming `is_last_backward`/reduce-grads flags, but its
finally hardcodes both reshard policies to true. This inherited limitation is
outside the fixed crop's required policy and must not be described as arbitrary
policy restoration. A guard port must preserve the current policy and verify
normal/exception paths, MTP0.3, chunks2/32, actual owner/event lifetimes and
elementwise losses/gradients/two optimizer updates. Exact callback/state checks
are justified; native FP8/HybridEP floating-point repeatability is not yet
established. Calibrate unchanged numerical A/A before setting native tolerances;
timing A/A and rounded norms do not establish numerical equivalence.

### Native numerical receipt: integration verified, calibration open

#### Observation path and limits

The opt-in receipt observes native `after_dataloader` callback-mutated CPU
inputs and labels; `forward_backward_step` metadata, denominator and loss;
`dist_utils.clip_grad_norm_` before and after clipping; the optimizer after
actual AdamW; and `after_load`/`batch_end` state. Preserve arguments, returns,
hooks and wrapper restoration. Adaptive load balancing is an optimizer-container
pre-hook: capture incoming counts/bias before clipping, then updated bias/reset
counts after the step. Its counts are FP32 and nonpersistent; `ntokens_seen` is
int64. Neither dtype is changed.

Optimizer parameter groups own the sharded tensors consumed by fused AdamW.
Capture each RaggedShardDTensor local shard under `no_grad`, without padding,
gathering or casting; retain identity/FQN/alias, shape, dtype and shard metadata.
One-tensor-at-a-time CPU hashes establish exact equality only when unchanged
native A/A repeats exactly. A mismatch requires bounded temporary payloads and
elementwise error metrics, not a tolerance inferred from hashes. The receipt
covers pre/post-clip gradients, parameters, optimizer moments/steps, losses and
components, normalization, and routing/counter state. The clipping return is the
norm; a separately computed clip coefficient is derived rather than observed.
Post-clip tensors attest the applied result.

The first isolated observer copied each microbatch to CPU and could serialize
the next microbatch, masking asynchronous buffer-lifetime faults. The corrected
path hashes CPU batches after `after_dataloader` (all 32 are fetched before GPU
compute), retains detached native scalar losses and one immutable denominator
reference through forward/backward, then hashes those at preclip. It samples
native loss-component totals and mutable auxiliary totals/current scale once at
preclip, before the next-update reset. It needs no per-microbatch D2H, auxiliary
clones or graph-bearing references. Native loss provides one normalized scalar
per microbatch; component tests own elementwise CE evidence. This preserves
microbatch overlap but still perturbs update boundaries, so hash runs alone do
not validate V3 asynchronous ownership. The uninstrumented native integration
and separate representative trace remain required.

The mode is default-off and separate from TPS runs. With it disabled, the runner
does not import or install the observer, add native flags or hashes, or perform
receipt work. Enabling it bounds captures to selected updates; tensor snapshots
copy local values to CPU and exported references can create substantial I/O.
These are numerical diagnostics, never throughput measurements.

#### Integration and parser status

The original helper review found that `dist_sum` can return a Python-float
denominator while the first observer required a tensor. A CPU probe reproduced
the TypeError; support for the scalar without tensor conversion fixed it, and
the original 11 CPU tests remain valid. The earlier isolated runs (61 affected
tests plus two targeted tests) are historical evidence: their executable and
ambient `PYTHONPATH` were not captured. Main integration now closes that check
with an explicit campaign interpreter and task `PYTHONPATH`.

Main integration is at `f29581c4dcae78928cb940313b13893826f4e10`. It applies
the accepted helper revisions `84b72524fd274830c0bcb8b49afd3c196eabd118`
(hash mode) and `be8d75ad6d90cc3de9175993be638d2f8bdd3fa4` (payload
extension). Independent review accepted the source changes. The combined main
suite passed 110 tests, including CPUAdamW observation-neutrality checks; scoped
Ruff and `git diff --check` passed. The exact command, interpreter and package
versions, outputs and integration details are recorded in
[the main integration evidence](artifacts/raw/native-numerics-integration-evidence/INTEGRATION.md).
The prior worktree test counts do not replace this main-checkout result.

The G3 category-aware parser fix is preserved in the combined main source:
CPU logical `user_annotation` markers and GPU projections are counted
separately, and unsupported categories fail validation. The G3 assertions still
pass. GIFT source and canonical config bytes are unchanged.

The four parser gaps found in independent review are resolved. Receipts bind the
selected metric files, counts and digests to the A/B identities. Only selected
mathematical float fields are whitelisted; learning rate, scheduler state,
optimizer step and counters remain exact, and any nonfinite value invalidates a
receipt. Selected FP32 `tokens_per_expert_E` and `expert_bias_E` are compared
numerically at three router phases; initial routing and integer counters remain
exact. Original post-flush record-count handling was also corrected by wrapping
the actual `MetricEmitter.emit` publication callback.

#### Calibration state and acceptance gates

C1A's one-update export completed on all eight ranks, covering32 microbatches
per rank and approximately47.5 GB per rank of numerical payloads. C1B failed before
update 1: all eight ranks raised
`ValueError('initial parameter, optimizer, scheduler, buffer, RNG, or counter gate differs')`
at `numerics_payload.py` (removed during cleanup) line 580. The runner exited, owned
cleanup completed, and all GPUs are idle. Native A is complete, but the A/B
initial-state gate did not pass because live tuple mesh coordinates differed
from their serialized list representation. The original B live hashes were not
persisted, so fresh exact-state verification is still required. This failed
control supplies no candidate numerical verdict or error-floor measurement. No tensors were
compared elementwise and no tolerance was calibrated. C2 and monitor50 remain
open. The monitor wrapper now observes
actual `MetricEmitter.emit` publications, but native monitor50 inventory/equality is still
pending. EMA is inactive through update 50, so this evidence makes no EMA claim.
The selected online scalar records retain native reduced CPU mappings, order,
multiplicity and metadata before clearing, without another reduction, GPU copy
or per-microbatch drain. The actual inventory must still establish whether any
non-scalar or offline records are uncovered. The campaign does not configure
`MtpAccLenMonitor`; MTP coverage is its actual 0.3-weighted loss/state and the
32-chunk monitoring path.

The normal W110 controls already differ in rounded first-update values:
A1 loss 15.31946/norm 9.5378 versus A2 loss 15.31943/norm 9.5374. These values
do not establish elementwise errors or their cause, but exact native
repeatability cannot be assumed. No tolerance is inferred from them or from
hashes.

The selected extension exports temporary exact local reference bytes and
compares one tensor at a time in bounded CPU blocks. Acceptance uses three fresh
matched prefix pairs, stopping and capturing at updates 1, 2 and 50. That is
106 native updates and six cold starts (four extra starts over the hash-only
route). One update's five large tensor families occupy about 380.122 GB across
eight ranks. Across the three reference runs, writes and comparison reads are
about 1.140 TB each; both sides transfer about 2.281 TB to CPU, plus about
456 GB of initial-parameter traffic across six starts. Check free bytes/inodes,
host RAM and the largest local shard before execution. Copies synchronize and
can change scheduling; these costs are diagnostic, never TPS.

The thin task adapter uses PyTorch
[`assert_close`](https://docs.pytorch.org/docs/main/testing.html) for tolerance
and metadata checks, while recognizing that it does not calibrate an error floor
or manage streamed payload lifetimes. PyTorch
[`frombuffer`](https://docs.pytorch.org/docs/stable/generated/torch.frombuffer)
interprets owned CPU bytes with an explicit dtype. Distributed
[Checkpoint](https://docs.pytorch.org/docs/main/distributed/checkpoint.html)
adds state-dictionary/resharding machinery and possible collectives this
comparison does not require. No dependency or checkpoint path is added.

Before comparison, require matching source/runtime/config/data, initialization,
ownership, optimizer/scheduler/counter state, and hashes for every prefix CPU
input batch. Capture RNG hashes after native `fit_start` without advancing or
changing RNG or determinism settings. Initial-state mismatches fail the gate;
they are not numerical error. Report per-tensor maximum absolute error and
index, RMS error and reference/current RMS, relative L2 with explicit
zero-reference behavior, nonfinite counts and exact agreement. Post-optimizer
parameters are values, not parameter deltas. One A/A pair per capture measures
an observed error floor, not a statistical upper bound. Freeze family-specific
acceptance budgets from valid precision references and A/A evidence before
candidate comparison; unsupported tolerances stay open.

Reference namespaces must be immutable and complete. Verify byte hashes,
identity/shape/dtype/shard metadata and observation coverage; durably write each
compact comparison result before removing its temporary payload. Retain bounded
failure evidence when comparison or evidence writing fails.

Earlier V5b source review corrected one premise: `int(global_valid_tokens)`
at the counter update is a CPU conversion, not a new GPU read. Removing the
producer `.item()` while preserving the baseline clamp, normalization precision
and callback/counter order remains an unmeasured retained-device hypothesis; no
V5 speedup is established. Numerical receipts also do not establish checkpoint
continuation or convergence correctness.

The [C1 diagnosis](native-c1-initial-gate-diagnosis.md) isolates tuple/list mesh
coordinate serialization. The two-line producer fix keeps the exact comparator
and all value/metadata gates intact. Its regression failed on the old producer;
14 observer,12 payload and2 default-path checks pass. Root approved reusing the
immutable A bytes with explicit old/new adapter hashes in `results.json` because
serialized metadata, training mathematics and payload format are unchanged.
Fresh B must still pass every exact initial/input gate and verify unchanged
GIFT/config/runtime/data; the original failed B remains failed.

## Runtime applicability: V4

The native002 parameter census found 243 trainable tensors and 2,375,764,304
local FP32 elements per rank, with zero frozen tensors/elements on every rank.
EMA storage and buffers are not frozen model parameters. V4 frozen BF16 storage
is therefore INAPPLICABLE to this fixed crop; no precision change is made.

## Cleanup

Both native smoke processes have stopped and released the GPUs. Their logs,
resolved configs and compact receipts are retained; no checkpoint, trace or
snapshot was produced. Attempt001's obsolete compiler cache was removed as
recorded above. Attempt002's obsolete compiler cache was removed after smoke
acceptance (604,228,268 logical file bytes); calibration/controls use fresh
per-process caches. Compact native/cache-path evidence remains.

After G2 acceptance, the rejected W60 and both accepted W110 caches were removed:
27,651 files /1,806,310,969 logical bytes. Each run retains an
`obsolete-cache-cleanup.json` receipt; stop/ownership/path/symlink/reference gates
passed, non-cache artifacts matched their pre-cleanup hashes, and active G3 used
an independent cache. Immediate `statvfs` available-byte/inode deltas were zero;
logical deletion is not reported as a measured free-capacity increase.
The rejected run retains six SwiGLU specializations/48 exact artifacts
(2,161,925 bytes) under
`artifacts/raw/g2-baseline-a1/rejected-w60-fused-fp8-swiglu-evidence/`.
Its manifest SHA256 is
`2f29bf1bb64b926bda95e4c0eda464b5a0b7f5785d17a25ec90858727496a2b3`;
original paths, mtimes and content hashes preserve the pre-window cache evidence.
