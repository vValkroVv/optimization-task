# Combined four: native diagnostic trace

**All four mechanisms execute together in the accepted eight-H100 crop.**
Ranks 0/1 traces verify queue use, BF16 reuse, fused routing metadata and stable
CE through compiled two-chunk main/MTP loss. All eight ranks complete 21 updates
and the profiled update's 32 microbatches plus optimizer. This establishes
mechanism, without a trace-based TPS, optimizer-savings or general-speedup claim.

## Identity and capture

The completed run is combined-four-diagnostic in existing job
lm-mpi-job-23027817-4bfe-4730-926f-5c2de590b32b. Summary is complete/errors[],
child exit 0/process-group-stopped, completion PASS/source-and-config-unchanged.
Nonfatal finalizer Err16/.nfsbusy warnings follow all 21 updates; no ESTALE116.
Earlier launcher failure and the initial unavailable reports remain in ignored
artifacts; subsequent user-authorized execution supersedes their verdict.

Executed HEAD is 6c6cd9c0e41a0a5d3d00e1e252b3c120052d3507, frozen source digest
4b563032ba334d6497b7d658a3568e9d5560293be8f2457dae02c941dd5916cb.
[Core patch](candidate-core.patch) enables only routing and CE.
[Source identities](source-identities.json) pin existing helpers.
[Result JSON](trace-analysis.json) pins actual metadata, final source, resolved
YAML and later authorized runner/cache hashes, rather than assuming the older
planned runner bytes. Resolved YAML SHA256:
34441b3231b1618e64b6f715474351f71ec02dfe40a715d14111e2d812ed6c63.

Resolved settings preserve B2/S4096/GB512/accumulation32, EP8,
dense-FSDP8/expert-FSDP1, FP32 masters/accumulation, dense+sparse FP8, model/loss
compile, selective AC, EMA, MTP1 weight0.3 and two loss chunks; prefetch 32/reuse
true/epoch 102400/seed 2048. Capture: 21 updates, 20 warmup, no timing window;
update 21's first three complete F/B microbatches on ranks 0/1. Compiler artifacts
reuse the authorized baseline003 cache. Runtime: Torch 2.15.0.dev20260822+cu130,
CUDA 13, Triton 3.8, NCCL 2.30.7, eight H100 80GB.

Saved control v2-long-epoch-baseline-trace-001 has update 21/three microbatches,
depth 0/reuse false, epoch 51200 and HEAD 3b6b8928ec4b15a529d07ee99e672f2fc75ef73a.
Its source/allocation/runtime provenance differs. Matching local shapes and
original routing/CE support mechanism comparisons; elapsed differences are
historical observations, not contemporary estimates of the four-feature effect.

| Canonical trace | SHA256 |
| --- | --- |
| Saved control rank0 gzip | 5b2a23c893e59d7317dca46468b9d8416435245ebbcb865dfa3f81344c760576 |
| Saved control rank1 gzip | 1a09edab1e9bbdbd55deea3ad8e5280c3e523e0fa60b3474ed02f88ef9783ae5 |
| Combined rank0 JSON | 3aba2d4ce7fa0389ba68fd0848e4782dbbe92c663dbb390d9ef9fc394132e82b |
| Combined rank1 JSON | 1680cb0874eefa2637b4f725dffe01b46f54b631ce1f98cee03420e4eccfdfb4 |

Complete paths, prepared hashes, lane mappings and exact MCP queries are in JSON.

| Variant, each rank | Kernels | Copies | Memsets | Unique launch links | Import errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Saved control | 8795 | 274 | 861 | 9930/9930 | 0 |
| Combined | 8391 | 193 | 861 | 9445/9445 | 0 |

Actual Perfetto imports match canonical census/duration sums, with zero ambiguous
links. Existing lane repair preserves device fingerprints, actual timestamps,
durations, names/args and CUDA streams; it moves colliding display lanes and flow
endpoints, removing GPU annotation projections. Canonical bytes remain unchanged.
Synthetic display tids are not streams; dropped collisions are not removed work.
Expert preparation uses actual CUDA stream 24; metadata and CE use stream 7.

## Activation and consumers

- Queue: all 32 batches for update 21 are collected before its first F/B.
  Each selected rank records 32 pops in that captured input-collection interval,
  producer alive, initial readiness 32, minimum 4; those 32 host spans total
  0.370/0.402 ms. These cover full-update input collection, separately from
  the subsequent three-microbatch F/B trace prefix. Ranks 2–7 do not collect queue
  observations. Producer work before capture and its overlap remain unmeasured.
- BF16: each layers.1[efsdp] through layers.7[efsdp] has one cast in microbatch 1,
  none in 2/3 and no singleton copy; totals 7/0 versus control 21/42 per rank.
  On every rank/update 1–21, each eligible group refreshes 1, hits 63, directly
  uses 64 and invalidates/increments generation 1 per update. Final totals:
  21/1323/1344/21/21. Noneligible counters stay zero.
- FP8: both retain 78 expert-weight quantization kernels/rank: grids 384×32×1
  and 512×12×1 each have 21 forward/18 recompute launches.
- Routing: 39 metadata kernels/custom ops/rank, 21 forward/18 SAC recompute.
  Seven MoE layers run forward; six recompute because layer 5 saves communication.
  Each original helper owns 11 kernels+one DtoD copy; each candidate owns one
  metadata kernel. Census difference exactly matches 390 routing kernels,
  14 expert casts, 39 routing copies and 42 expert copies.
- CE: 12 forward_kernel+12 backward_kernel/rank, covering three microbatches,
  two outer chunks and main/MTP. Four actual compiled graph IDs, unique flows
  and rank-specific generated modules join kernels to stable_ce_forward.default
  and stable_ce_backward.default at BF16 [4096,128256]. Forward CompiledFunction
  and backward CompiledFunctionBackward ancestry, FP32 SUM and head GEMMs remain.
  Cached graph bytes were inspected without deserializing. CPU custom-op labels
  alone are not activation proof.
- Exclusions: the exact activation patch changes neither input preparation nor
  Linear+CE bindings. Prepare kernel/custom-op and native_linear_ce observables
  are absent in canonical captures. Native input work, capacity policy and
  communication remain.

## Timing scopes and waits

Milliseconds, each rank independently over the sampled prefix. Host spans
include dispatch/submission, scheduling and waits, not CPU utilization. Device
sums, unions and spans stay separate; rows overlap and must not be added.
Raw results retain per-call/chunk streams, launches, copies, collective families,
overlap, gaps and integrated completions.

| Scope | Control r0 | Combined r0 | Control r1 | Combined r1 |
| --- | ---: | ---: | ---: | ---: |
| Expert gather CPU-span sum | 7.125 | 1.980 | 6.207 | 1.918 |
| Expert cast+copy device union | 29.460 | 4.125 | 29.255 | 4.121 |
| Expert overlap with other kernels | 11.268 | 0.179 | 10.838 | 0.179 |
| Expert overlap with EP/NCCL | 1.187 | 0 | 1.981 | 0 |
| Metadata CPU-span sum, 39 calls | 11.118 | 4.150 | 10.183 | 4.196 |
| Metadata owned host-launch sum | 2.921 | 0.191 | 2.267 | 0.184 |
| Metadata kernel sum | 0.946 | 0.249 | 0.903 | 0.248 |
| Metadata device busy union | 0.995 | 0.249 | 0.952 | 0.248 |
| Metadata per-call device-span sum | 9.464 | 0.249 | 8.519 | 0.248 |
| Wider CE CPU-span sum, 6 chunks | 10.553 | 7.251 | 7.192 | 7.117 |
| Wider CE owned host-launch sum | 0.910 | 0.614 | 0.769 | 0.600 |
| Wider CE device union | 27.266 | 22.599 | 27.405 | 22.520 |
| Wider CE per-chunk device-span sum | 27.542 | 22.881 | 27.666 | 22.794 |

Original metadata CPU boundary runs from first aten::ge through final returned
view; candidate is its complete custom op, excluding eligibility/Python prelude.
Enclosing dispatch retains quantization, EP, copies and synchronization: mean
CPU-to-owned-device completion 1.687/1.699 ms control versus 1.490/1.492 ms
combined. Per-call spans and their integrated region unions are separately in
JSON. All 39 dispatches retain cudaStreamSynchronize; host sums 8.061/11.258 ms
control versus 10.251/9.964 ms combined. Synchronization does not uniformly fall.

Wider CE runs from compiled forward through SplitBackward0, including both
heads, normalization/scaling, scalar work and concatenation. Smaller CE blocks
do not establish uniform enclosing improvement: one rank0 full head/loss guard
grows 94.537→100.838 ms, with communication overlap 12–31 ms across guards.
Input-to-first-F/B is 675.782/677.895 ms control versus 7.986/8.514 ms combined;
local all-device idle 660.440/650.265 versus 7.958/7.766 ms. These historical
observations do not isolate the combined effect or quantify producer overlap.

All 42 expert consumer regions/rank retain stream-wait submissions; host sums
0.046/0.043 ms control versus0.041/0.042 ms combined. Gather wait submissions
increase 42→98, retaining cache lifetime ordering. No host synchronization/query
API occurs inside those consumer regions. CUDA API args omit concrete event
and stream handles: exact producer/consumer device wait and exposed critical
path are unavailable, not zero. Existing source/lifecycle checks remain needed.
EP/NCCL device duration can include spin/peer waits; other-kernel classification
can retain unclassified waits.
Both ranks/variants retain 42 capacity predicates and the same actual NCCL kernel
counts: 40 uint64 all-reduces, 69 all-gathers and 36 FP32 reduce-scatters.
CPU all-gather annotations fall 222→138 with bypassed singleton gathers;
that does not mean 84 device collectives were removed.

## Acceptance and reproduction

The finite trace gate is complete within those observability limits.
[Integration assessment](integration-assessment.md) and [review](review.md)
reconcile unchanged loader/cache lifecycle and routing/CE numerical evidence;
none was rerun here. Generic per-step microbatch hooks remain empty by design:
update 21 directly records 32 calls+optimizer1 on all ranks; earlier updates have
64-use group counters, not independent microbatch-hook records.

Ignored evidence is under runs/combined-four/artifacts/raw/trace-analysis/:
raw attribution, region details, preparation/integrity receipts, exact MCP
queries, generated-code pins and analysis scripts. Reproduction uses the existing
BF16 raw helper, then analyze-combined.py and finalize-combined.py; inputs and
hashes are in JSON. One retained analysis assertion initially expected queue
observations on all eight ranks; it was corrected to selected ranks 0/1, with
native results unchanged. Initial unavailable report copies preserve the failure.

The device prefix omits remaining 29 microbatches and optimizer. This trace alone
establishes no cross-rank aligned critical path, new numerical/convergence/native
continuation result, attributable memory saving or PP4/EP32 conclusion.
The parent report owns the measured-window memory comparison and the
unprofiled fixed-window result and decreasing-cadence/repeat/noise limits.
No additional native/GPU/test/install experiment or source/runner edit occurred
for this analysis.
