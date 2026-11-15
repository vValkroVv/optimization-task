# FFN overlap and row-owner embeddings

## Input baseline and local artifacts

This completed campaign used the historical 2009-ID test tokenizer and the
low-diversity C4 fixture. Its timing and memory values describe those inputs.
The owner's [global real-data/tokenizer requirement](../../docs/baseline.md#global-requirement-real-data-and-production-tokenizer)
requires remeasurement on the corrected production input baseline. The
[production-data A/B/C/D campaign](../production-data-abcd/report.md) completed
that cumulative comparison and supplies the current speed and memory results.
Applicable correctness and lifecycle evidence is retained, and the accepted
MTP/FP8 A/A decision is unchanged.

The server implementation, reports and complete campaign artifacts were copied
to the local task checkout on 15 November 2026. File SHA256 checks matched the
server snapshot. Canonical trace bytes and recorded source identities are
preserved. Native update-21 traces cover three complete microbatches:

- Original traces: [rank 0](artifacts/raw/ffn-row-owner-diagnostic-001/native/profiling/traces/iteration_21/rank0_trace.json.gz)
  and [rank 1](artifacts/raw/ffn-row-owner-diagnostic-001/native/profiling/traces/iteration_21/rank1_trace.json.gz).
- Analysis derivatives with all imported kernels: [rank 0](artifacts/analysis/native-perfetto/rank0.normalized.json.gz)
  and [rank 1](artifacts/analysis/native-perfetto/rank1.normalized.json.gz).
  These omit GPU annotation ranges, including GPU-side task and GFSDP scopes,
  and move overlapping device events and matching flow endpoints to synthetic
  display lanes. Use the original traces above for interactive browsing with
  the complete annotation hierarchy. See the
  [capture/import explanation](native-trace-results.md#capture-and-import).

## Scope and source

Started 14 November 2026 from clean task `main`,
`2ada87125e433e1f76c4cb2bfad244b641a78a93`.
The goal attachment is
`/workspace-SR008.nfs2/vkropotin/.sr008-gpu/.codex/attachments/2bcb7b91-6254-4fea-a8fe-69727809b412/goal-objective.md`.
The [planning entry](../../planning/README.md#active-assignment-ffn-overlap-and-row-owner-embeddings)
owns finite acceptance. The [baseline recipe](../../docs/baseline.md#current-final-task-baseline)
owns the control configuration. Only measurement duration changes to 10+10;
training schedule and accumulation remain unchanged.

The standard-autograd owner and both FFN components have completed focused,
eight-rank component evidence. The all-three 32-microbatch native update,
initialized-pool accounting, common-baseline memory cap, and bounded native
mechanism/activation checks now pass. The matched A/B primary timing record is
complete; all finite acceptance items are closed, with the A/B difference
remaining provisional under the measured cadence limits.

### Current verification scope

The owner accepted the investigated MTP/FP8 A/A variability and stopped further
MTP and layer-0 numerical investigations. Those measured differences remain
in the receipts and do not block the campaign; no new tolerance policy is
introduced. The embedding component output/gradient check passes exactly. The
all-three native update also passes compiled/SAC integration without reopening
MTP/layer-0 A/A investigation.

The standard-autograd owner returns an owned 262,668,288-byte FP32
`RaggedShardDTensor` local gradient to ordinary autograd; backward scratch
borrows up to 704 MiB from the resident 2004 MiB private `lm_head` bucket. No
manual `.grad` mutation or engine/hook restrictions remain. Native BF16 source
coalescing, FP32 owner SUM and normalization are preserved. Eight-rank NCCL checks cover
routing, rounding, accumulation-32/reset, hooks/VJP, clipping, AdamW, EMA and
continuation; its [full-shape numerical result](embedding-numeric-results.md)
has exact outputs and first/later/reset gradients. The final [six-case component table](embedding-component-results.md)
and [matched-cache memory check](embedding-numeric-results.md) cover native
coalescing scratch and routing metadata. The [source-derived storage budget](owner-storage.md)
is complemented by the passing [initialized native pool and peak comparison](native-memory-results.md).

Both FFN groups completed eight-rank component checks and one paired component
measurement each. Their [primary results](ffn-component-results.md) record
8.56% lower isolated layer0 latency and 6.08% lower isolated MTP latency with
unchanged unique pool bytes; [MCP Perfetto](ffn-component-perfetto.md) confirms
early reduce-scatter overlapping parent compute in all 48 captured rank/MB
cases. These measurements do not establish native whole-step speedup. The owner
accepted the measured MTP/FP8 A/A variability, which remains in its receipts;
no further MTP/layer0 numerical investigation blocks this campaign.

## Execution authorization and environment

Read-only inspection finds the current shell inside
`lm-mpi-job-23027817-4bfe-4730-926f-5c2de590b32b-mpimaster-0`, with eight idle
H100 80GB GPUs (no compute applications, 0 MiB and 0% at inspection). Provider
metadata reports binary job, one worker, SR008 and allocation
`alloc-sberdevices-giga-vision-h100-sr008`, but user identity is `unknown`.
The owner subsequently confirmed that this is their allocation and explicitly
authorized the specified component benchmarks, focused GPU checks, one-update
integration run, native 10+10 A/B comparison and bounded traces. This closes the
current allocation ownership/compute-authorization gate. GPU work is serialized
after source integration and the corresponding correctness prerequisites.
Historical closed campaigns are not used as permission.

The existing interpreter is `/opt/torchtitan-venv/bin/python`, resolving to
Python 3.12.3. Site packages are at
`/opt/torchtitan-venv/lib/python3.12/site-packages`; torch identifies as
`2.15.0.dev20260822+cu130`, source
`8b353065ce260336c5fdfa74d754e8bd9b91b062`; Triton identifies as
`3.8.0+git3f6e4113`. These metadata facts do not establish import/runtime
compatibility. The historical native plan's task root is `opt-fq`, whereas
this assignment's actual root is `optimization-task`; new commands must use
the latter. Its recorded backend/runner hashes match the current files.

No additional allocation, installation, shared-environment change,
original-source write, commit, push or publication is authorized.

## Design and expected costs

The FFN design preserves existing module/parameter identities and FQNs, wraps
each selected `feed_forward` child before its remainder parent, and keeps eager
block orchestration across checkpoint-bearing compiled prefix, FFN and suffix
callables. The resolved SAC policy must apply inside those compiled regions:
an outer eager checkpoint around compiled children can save opaque compiled
results because the existing policy always saves `inductor_compiled_code`.
The initial implementation used private parent data and gradient contexts;
that implementation was removed to meet the owner's buffer requirement. The
current FFN and remainder groups use separately tracked contiguous,
non-overlapping regions within existing weight and gradient allocations. The
eight-rank component receipts confirm unchanged unique pool bytes and measured
component peaks; existing pending-reader tests cover their tested lifecycle.
The bounded component trace observes early reductions overlapping parent
backward on all ranks. The initialized native pool and measured memory phases
pass the common-baseline cap; the later [native combined trace](native-trace-results.md)
confirms combined scheduling within the two-rank, three-microbatch capture.

Expected costs are added collective startup/packing, autograd/compile boundaries,
lost fusion, additional saved boundary tensors (64 MiB per BF16 crop activation),
and possibly missed prefetch or extra all-gather work. Buffer-region handling
must add no GPU storage or staging allocation.
The child reduction must read all three complete wgrads only after its input
gradient is ready; parent backward then supplies the proposed overlap.

Embedding model and Adam state use flat checkpoint tensors and
RaggedShardDTensor. The standard-autograd owner retains that representation and
integrates wrapper exclusion, clipping, dense AdamW and EMA/state lifecycle;
the eight-rank focused checks cover continuation. Native dense BF16 embedding
backward still creates a 1002 MiB gradient plus native scratch; some
low-parallelism branches can use a 2004 MiB FP32 full-table temporary. The
full-shape component check counts its actual branch scratch; final native
initialized-pool/peak behavior is recorded in the passing
[native memory receipt](native-memory-results.md).

Existing research and pinned donor code are reused; the older optimistic
3.3–3.6% trace estimate is not a current-baseline prediction. The accepted
`external_gate` control trace records six full-table embedding gathers and
three reductions per rank over three captured microbatches. These are
control-path counts, not changed-path timing. The owner component trace instead
shows routed transports and no full-table embedding gather/reduction kernels;
its metadata AllGather and three SendRecv transports are documented in the
[owner component report](embedding-component-results.md). Native combined
scheduling remains separate.

## Component measurements

The current owner table uses the final standard-autograd implementation. Its
six eight-rank cases cover fixture, duplicates, skew, empty requester, empty
all and boundaries; integrated fixture F+B median latency is 13.040 ms baseline
versus 2.554 ms owner. The table records all per-rank samples and the
allocator-history limits. In a separate equal-cache full-shape check, exact
outputs and first/later/reset gradients match, with owner allocated/reserved
peaks below baseline; neither fixture establishes the full native pool or
whole-training performance. See the [primary owner table](embedding-component-results.md),
[numerical and memory receipt](embedding-numeric-results.md), and its linked
JSON. The current owner source SHA256 is
`f3a01c6cf99b981db7b4ef66866fe46259af84e9d2c21e2044e9ae18ff1696fd`.

Both FFN component pairs are also complete on eight H100/NCCL ranks. One
unprofiled control/split pair per component measured layer0 23.58189→21.56324 ms
(8.56% lower) and MTP 31.41067→29.49934 ms (6.08% lower), with identical unique
data/gradient storage bytes. The [FFN table](ffn-component-results.md) records
per-rank samples, peaks and numerical comparisons for both components; these
isolated one-pair results do not estimate a full-model gain. The owner's
accepted MTP/FP8 A/A variability remains visible in the numeric records and is
not a remaining campaign gate.

The [bounded FFN device trace](ffn-component-perfetto.md) links launches to
device kernels and observes early FFN reduce-scatter overlapping parent compute
in all 48 rank/microbatch cases (96.99–98.61% compute coverage). It also confirms
disjoint parent/child regions with unchanged unique bytes. Event/stream handles
are absent, and the trace synchronizes between microbatches, so it does not
prove which release event each wait consumes or exercise native combined
scheduling. Existing pending-reader tests retain their separate lifecycle
scope. Earlier manual-gradient owner results remain historical in
`embedding-component-results-manual-history.md`; they are not current acceptance
measurements.

## Implementation and focused checks

The [FFN implementation record](ffn-implementation.md) owns both default-off
controls, callable/checkpoint/compile and GFSDP boundaries, exact commands,
readiness checks and buffer costs. Nineteen focused local checks include CPU
Inductor/SAC output, input and parameter-gradient parity. The two-rank GFSDP
check used CUDA/NCCL with BF16/AOT/SAC and covers accumulation 32, fused AdamW,
flat-layout DCP/EMA restoration, the next update and exactly-once post-backward
ordering. The later eight-rank component receipts exercise the selected groups
with native FP8, model compilation and selective SAC. The superseded private
remainder data/gradient buffers totaled 959.793 MiB raw and have been removed.
The replacement passed targeted CUDA/NCCL checks with equal unique pool bytes,
disjoint views, pending-collective reuse and accumulation/state parity. Its
component peaks are measured; the full native inventory/peak now passes the
common-baseline cap in the [native memory receipt](native-memory-results.md).

The [owner implementation record](embedding-implementation.md) owns routing,
native BF16 coalescing, FP32 Ragged gradients and EMA/DCP lifecycle tests. Its
final two-rank CPU/Gloo check passed with `CUDA_VISIBLE_DEVICES=''`, both ranks
reporting CPU meshes and CUDA uninitialized. It covers duplicates, skew, empty
and invalid traffic, multiple invocations, shared-output derivatives,
accumulation 32, mixed clipping, dense AdamW including untouched-row state,
baseline flat-state loading, EMA and the next update. The revised borrowed
storage path then passed the explicit eight-rank CUDA/NCCL suite, including
pending reuse, persistent-gradient ownership and resets. Independent integration
and runtime-cost review found no material defect in the supported normal
`.backward()` path. Full-shape numerical parity and combined native integration
are complete; the bounded native trace confirms combined scheduling. The
matched A/B primary timing record is linked in the final verdict below.

The [native plan](native-plan.json) and [runner checks](runner-checks.md) retain
actual hook-generated A/B, one-update and diagnostic commands. The initial
runner suite passed 77 checks plus the post-plan roundtrip. All three controls
default off; the candidate selects them together above the accepted recipe.

### Execution and evidence corrections

The FFN worker incorrectly assumed that `DTensorTestBase` selected CPU. Its
distributed tests auto-selected CUDA/NCCL during an assignment restricted to
CPU execution, before a serialized GPU lease was issued. The implementation
record retains exact commands, known timestamps, failed attempts and timing
uncertainty relative to the owner's grant. Earlier CPU labels are withdrawn.
All test commands exited. A subsequent root check found all eight GPUs at
0 MiB/0% with no compute applications or task test/training processes; existing
Perfetto analysis services were preserved.

Earlier owner tests used CPU tensors and Gloo but DCP implicitly initialized a
CUDA context. The final forced-CPU rerun verifies the same affected route with
CUDA hidden and uninitialized. No production DCP behavior was changed for this
test isolation. Future CPU commands set visibility per process explicitly.

### Revised FFN buffer acceptance

The owner requires minimal changes, no additional GPU storage, no buffer-pool
growth and no duplicate staging tensors. Inspection of `GFSDPBufferContext`,
`_init_shared_state`, `_init_unsharded_params` and prefetch/release informed the
final implementation. It partitions existing dtype-bucket storage into aligned
contiguous FFN/remainder views, preserves rank layouts and alternating weight
slots, and tracks region ownership/completion so overlapping users wait for
relevant readers and writers. Allocation and reuse occur outside compiled
compute.

Component evidence covers shared backing/non-overlap, pending-reader reuse,
numerical/state regression, unprofiled allocation/peak comparison and bounded
Perfetto overlap. The all-three native update confirms initialized pool
capacities and allocation lifetime within the common-baseline memory cap; equal
unique pool bytes and the no-growth requirement are preserved. The measured
first/warm phases do not establish a constructor-inclusive full-lifecycle or
long-run peak.

## Native comparison and mechanism

The all-three native update passed on all eight ranks with the accepted recipe,
32 microbatches, SAC, and model/loss compilation. The initialized deduplicated
GFSDP pool is 8,719,958,016 B/rank in A and 6,973,565,952 B/rank in B. All
candidate ranks fit the common baseline worst-rank allocated/reserved cap; the
candidate global combined peak is lower. The [native memory receipt](native-memory-results.md)
owns per-rank phases, raw artifacts, validator recovery and retained allocator
caveats. The first-update B reserved premium is 127,926,272–253,755,392 B
(122–242 MiB; 128–254 decimal MB) versus A002, the only baseline that reached
GPU execution. In the separate warm update-2 pair, candidate rank 2 is
62,914,560 B (60 MiB) above its own A control but remains under the common
worst-rank cap; allocator cause is unproven. These windows do not establish a
constructor-inclusive full-lifecycle or long-run peak.

The matched native 10-warmup/10-measured A/B pair completed on all eight ranks;
both runs used the same source revision and accepted base recipe, with all three
new controls enabled only in B. Both completed cleanly with child `0` stopped,
no errors, and zero OOM/retry deltas. Candidate EMA remains enabled
(`callbacks.ema.use_ema=true`, decay `0.99`, update period `10`). Timing kept
profiling, detailed monitoring, validation, checkpoint saves, memory snapshots
and numerical capture disabled; state continuation remains covered by the
integration harness.

The [primary native-performance report](native-performance-results.md) and
[JSON results](native-performance-results.json) own exact values, cadence,
per-rank memory, source/config and raw receipts. The single serial pair measured
164,737.68→166,825.08 global physical tokens/s (+1.27%) and
164,697.46→166,784.35 supervised tokens/s, with measured elapsed time
127.303→125.710 s. Rank-zero cadence rose from 12.606650 to 13.016546 s for A
and from 12.448897 to 12.820990 s for B, about 3% in each window; the full
per-rank series remains in the primary record and raw receipts. There is no A/A
noise estimate, so this fixed-window difference is provisional and does not
establish a general or causal speedup. An auxiliary formatter first stopped
because an optional BF16 snapshot was absent; the canonical raw summaries and
per-rank series remain complete, and no A/B or diagnostic run failed.

The [canonical native trace report](native-trace-results.md) confirms candidate
activation and the combined mechanism. Its update-21 capture contains three of
32 complete F+B microbatches on ranks 0/1 after an all-eight-rank, 21-update
diagnostic. In all 12 layer0/MTP rank/microbatch cases, the 513 MiB FP32 FFN
reduce-scatter precedes parent reduce-scatter and overlaps 96.20–98.32% with
remaining parent compute. The owner lookup/backward runs once per microbatch;
one small count AllGather and three routed SendRecv calls remain, with no
embedding-table GFSDP collectives. Three full-table BF16 gathers and three FP32
reductions per rank remain for the untied `lm_head`. Across four natural
cross-microbatch boundaries, prior head reduce-scatter ends 233.413–236.354 ms
before the next owner scratch write, and the final owner scratch consumer is
118.283–119.267 ms before the next head pack. Event/stream handles are absent,
so these intervals and the trace do not prove which wait consumes which event
or establish a full-update critical path.

## Final verdict

The independent review found no material source defect in the reviewed crop
path. The final standard-autograd owner supersedes manual `.grad` accumulation
and its hook/VJP restrictions; its eight-rank checks, full-shape numeric result
and component tables are linked above. FFN components have actual split-group
measurements and all-rank Perfetto overlap evidence. Those component results
alone did not close combined native behavior; the passing all-three native
update and memory evidence above now close integration and the measured budget.

All finite assignment gates are complete: source integration and focused
correctness, full-shape embedding numerics, initialized native memory and the
common-baseline budget, the all-three 32-microbatch update, the matched 10+10
A/B pair, and bounded native mechanism/activation evidence. The three controls
remain opt-in and default-off. The +1.27% fixed-window difference is provisional;
the pair has within-window cadence drift and no A/A noise estimate. The native
trace covers only three of 32 microbatches on two ranks and cannot identify
which event each wait consumes or establish a full-update critical path.

Long-run convergence, exact production checkpoint/resume, PP4/EP32 behavior and
generalized throughput remain outside this eight-rank crop experiment.

MTP/FP8 A/A variability is retained as accepted campaign variability; no more
MTP or layer0 numerical investigation is a gate. PP4/EP32, convergence and
long-run stability remain outside the tested crop claim.

## Revised owner memory acceptance

The implemented owner borrows the existing private `lm_head` gradient bucket,
adds no pool or fallback, and returns an owned FP32 local gradient to ordinary
autograd. It preserves native source-local BF16 coalescing, per-source FP32
addition and one persistent accumulated owner gradient. The eight-rank tests
cover pending readers, first/later/reset accumulation, hooks/VJP, clipping,
AdamW, EMA and flat-DCP continuation. The source-derived ordinary FP32 bucket
shrinks from 2004 MiB to 893.673828125 MiB after owner exclusion; the private
head remains 2004 MiB. The full-shape owner fixture includes the allowed native
coalescing scratch and route metadata and has exact output/gradient parity with
lower allocated and equal-cache reserved peaks. The component table has higher
reserved high-water in its distinct allocator-cache history; both scopes are
retained in [the primary owner table](embedding-component-results.md) and
[matched-cache check](embedding-numeric-results.md). Actual initialized
full-model capacities and measured native phases pass the common-baseline cap;
phase and lifecycle limits are recorded in
[native memory results](native-memory-results.md).

## Current control trace: FFN and embedding opportunities

The accepted `external_gate` control still has candidate overlap windows;
it supplies no current speedup estimate. [Control evidence](control-trace.json)
records MCP SQL/results, canonical and normalized hashes, source pins and each
of three microbatches on ranks 0/1. Ownership follows CPU scope → descendant
runtime/driver launch → actual flow endpoint → GPU kernel. Current selected
device spans are 1233.485/1232.961 ms, excluding the other 29 microbatches and
optimizer. The [FlashQLA report](../flash-qla/trace-results.md#mechanism-and-ownership)
owns capture and backend attribution.

| Group / rank | Candidate remainder elapsed, ms | Main-stream union in remainder, ms | Whole-group RS duration, ms | No main kernel during RS, sum over 3 MB, ms |
| --- | ---: | ---: | ---: | ---: |
| MTP / 0 | 8.763–9.105 | 8.594–8.941 | 2.799–3.956 | 6.793 |
| MTP / 1 | 8.621–8.955 | 8.461–8.788 | 4.718–6.631 | 13.913 |
| layers.0 / 0 | 5.414–5.490 | 5.236–5.312 | 2.408–2.565 | 4.903 |
| layers.0 / 1 | 5.374–5.379 | 5.198–5.204 | 2.975–3.080 | 6.889 |

The candidate boundary follows the final FFN-like wgrad and fused input-gradient
add/norm preparation (MTP ordinals 96/97; layers.0 ordinals 71/72). The FP8
[backward](../../gift/torchtitan/ops/float8/cuda_gemm.py#L201) computes dInput
before wgrad; the [FFN](../../gift/torchtitan/models/common/ffn/__init__.py#L104)
contains w1/w3, SwiGLU and w2. This historical control trace did not record
the exact tensor operands or w1-versus-w3 identity. The subsequent changed
component captures identify the executed w1/w2/w3 groups and their reduction
ordering in the [FFN result table](ffn-component-results.md).
The shared AG/RS stream 24 has no kernel throughout any candidate remainder
on either rank. Its prior operation ends 4.458–7.923 ms before the MTP boundary
and 5.131–5.620 ms before the layers.0 boundary. Thus current local queue
occupancy did not rule out early RS. Changed component execution now confirms
all-rank FFN reduction ordering, compilation/SAC activation and device overlap;
the combined [native trace](native-trace-results.md) confirms that scheduling
mechanism within its capture. It lacks event handles, so exact wait attribution
is unavailable, and the capture does not establish contention-free execution.

Kernel metadata confirms FP32 reduction inputs: MTP 893.674 MiB and layers.0
772.188 MiB. Each selected FFN is 513 MiB (three 171 MiB weights); residuals
are 380.674/259.188 MiB. These are logical buffers, not measured wire traffic.
Current whole-group packing kernel sums are about 0.500 ms for MTP and
0.434–0.443 ms for layers.0 per MB. The component timing and diagnostic receipts
now measure split packing, boundaries, forward/recompute/backward, all-gathers,
prefetch and per-component memory. The [buffer-release contract](../../gift/torchtitan/distributed/gfsdp/_param_group.py#L935)
waits before packing and publishes reduction completion; an early FFN reduction
cannot safely reuse an ancestor's unreleased gradient storage.

Each rank has six full-table embedding gathers and three reductions. Forward
AG takes 2.657–3.371 ms, backward AG 3.661–3.888 ms, and FP32 RS
5.198–5.242 ms. AG input/output is 125.25/1002 MiB BF16; RS input/output is
2004/250.5 MiB FP32 for dense group size eight. Forward AG plus RS totals
24.527 ms on rank 0 and 23.666 ms on rank 1 over three MB, with no main-stream
kernel overlap. Backward AG overlaps main kernels except 0.339/0.289 ms total;
give its removal no direct critical-path credit. Dense embedding gradient
packing remains about 1.12 ms/MB; a staging native BF16 coalescer retains its
1002 MiB temporary, so removing transport does not remove that work.

FlashQLA shortens the layers.0 remainder from the historical 6.330–6.371 ms
to roughly 5.37–5.49 ms. The old zero-startup, payload-linear 3.3–3.6% scenario
therefore has different overlap and tail assumptions. Current MTP durations
also differ materially between the two ranks. No proportional-payload timing
or additive combined saving is inferred. In particular, layers.0 reduction
precedes embedding packing through shared-buffer release; row-owner routing
changes that dependency and requires a combined tail measurement.

All overlap values above use clipped interval unions. Small same-stream trace
timestamp overlaps made raw duration sums slightly overcount, so exact unions
were recomputed. No-main-kernel intervals and adjacent completion times prove
neither a wait nor contention; NCCL durations can contain peer waiting.
The owner component table covers request sorting/dedup/count synchronization,
exchange, lookup/casts/reconstruction, source-local BF16 coalescing, gradient
exchange, FP32 owner SUM and dense accumulation across fixture, duplicate,
skew, empty and boundary traffic. Full-shape owner numeric and eight-rank state
checks pass, and FFN component overlap is measured across all ranks. The
combined native update and initialized peak pass, and the bounded native trace
confirms activation and mechanisms. The linked [native performance report](native-performance-results.md)
and its JSON complete the measured A/B record, with cadence and single-pair
limits retained. Allocation ownership and compute permission are confirmed.
