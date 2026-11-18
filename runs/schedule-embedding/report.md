# GFSDP transitions and row-owner embedding fusion

## Revised measurement assignment

The latest owner objective supersedes the unfinished short-window campaign:
run A, B, C, D0, S, SE and SE-fused serially, each with exactly 40 warmup and
100 measured complete updates, followed by separate D0/S/SE/SE-fused update-21
profiles. New SE selects `compact`, `owner_add` and `owner_lookup`; SE-fused
also selects `routing`. Historical receipts named SE retain their original
four-stage meaning. Their reconciled [native results](native-results.md) remain
historical evidence and do not complete the revised campaign.

Finite completion checklist for the revised assignment:

- [x] Add minimal native stage selection; verify parser round trips, resolved
  stages, fallback/fused metadata equivalence and all seven dry-run commands.
- [x] Freeze source/configuration/harness, then complete all seven 140-update
  timings with the fixed updates 41–140 window and full all-rank evidence.
- [x] Complete four separate 21-update diagnostics and reconcile their original
  traces through Perfetto MCP, retaining semantic comparisons and timelines.
- [x] Report incremental/cumulative throughput and memory, the direct
  SE-fused/SE comparison, full cadence plots and bounded current-data drift
  findings in this report and the existing timing-drift record.

Resumption inspection found no campaign/training children and all eight H100s
idle, with zero used memory and no compute applications. Existing Perfetto
services and contributor edits remain preserved. Reuse the validated saved-D
cache and original-capacity overlay. No new allocation, installation,
publication, kernel change or additional optimization is authorized.

The [selector preflight](selector-integration.md) records passing native config,
actual two-rank factory wiring, CUDA routing-metadata equivalence and all eleven
timing/diagnostic command checks. The [independent launch review](selector-review.md)
passes after targeted regression verification of diagnostic sequencing and
inspection of the corrected constructor, capacity freeze and CUDA visibility.
Final combined freeze is
`3b3244ae1e759111b8240af929715074b751c946881202a9ff95e9f3b9918ffd`.
Execution followed the live all-eight-GPU preflight and reused prior arithmetic
and lifecycle evidence. Current measurements and trace findings are below.

The first timing run exposed a reporting defect after native training: resolved
YAML omits the new selector when its value is `None`, while the existing runner
and campaign guards require every expected field to be present. A nevertheless
completed 140 updates on all eight ranks, with intact 40→140 drained-window
records, the required token totals, clean child exit and matching final source/
configuration hashes. Its original wrapper `FAIL` remains preserved. The matrix
continued under the unchanged freeze; separate offline adjudication checked
every acceptance contract and this exact nullable default. All seven timing
proofs passed before profiles began.

The recovery implementation then performed full-source validation scans during
B-003's measured window. This violated the concurrent-work protocol; B-003 is
excluded from the primary performance comparison despite complete native and
correctness records. C-003 was stopped at update 27, before measurement began.
The [repair plan](artifacts/raw/newlaunchrecords/repair-004-plan.json) retains
clean A-003 and selects fresh B/C/D0/S/SE/SE-fused attempts ending `140-004` in
that order. Their commands differ only in run IDs from the reviewed frozen
commands. Raw and recovery records remain intact; the
[protocol exclusions](artifacts/raw/newlaunchrecords/protocol-exclusions-003.json)
record the overlap and its unknown performance impact. Subsequent offline checks
were deferred until all timing attempts were terminal.

The [repair adjudication review](repair-adjudication-review.md) passes after
restoring authentication of all 14 explicit runtime, parser and bridge files
before offline imports. The checker pins the repair plan and retained freeze,
preserves native before/final identities and original wrapper failures, and
refuses actual adjudication until every timing attempt is terminal. It does
not walk the training-source tree. The
[diagnostic acceptance contract](diagnostic-acceptance.md) keeps native capture
proof separate from the later original-trace GPU-tail and Perfetto checks.

The final [diagnostic checker review](diagnostic-adjudication-review.md) and
[timing analyzer review](timing-analysis-review.md) pass. The reviewed offline
method is pinned at
`594fb7db1ff5403e6bcb9ac901fe98f85c35d10e09da36dcdf1c5322810780f4`;
its bytes and tests are retained under
`artifacts/raw/newlaunchrecords/offline-method-004/`. The
[analysis input manifest](timing-runs-140.json) selects retained A-003 and the
six repair-004 runs, each requiring a final `adjudication-140-004` proof from
that same method. Actual adjudication and analysis ran after all timings
finished. Diagnostic parsing was also gated on all four captures being terminal;
its native-parser proof deliberately leaves GPU-tail acceptance to the separate
Perfetto receipt.

## Revised full-window timing results

All seven final proofs are `ADJUDICATED_PASS` under the same reviewed method.
The raw A/B/C/D0/S wrapper failures remain preserved; SE and SE-fused have raw
`PASS` status. The [execution receipt](artifacts/raw/analysis-execution-004/receipt.json)
records each command and status. The numerical analysis and ten plot files
completed before a Markdown-only renderer error; the
[recovery receipt](artifacts/raw/analysis-execution-004/markdown-recovery.json)
records the targeted fix and rendering from the unchanged JSON. No timing,
adjudication, numerical analysis or plot was repeated for that repair.

The fixed window is updates 41–140, including 209,715,200 physical and
209,664,000 supervised global tokens. Primary elapsed time is
`max(rank end) - min(rank start)` after the existing CUDA drains. All eight
ranks retain updates 1–140, and every measured allocator retry/OOM delta is
zero. The [independent reconstruction](artifacts/raw/analysis-execution-004/primary-envelope-check.json)
matches all seven envelopes, token totals and maximum memory values.

Variant links below open the actual command metadata. A is the historical
FLA/prefetch-0 recipe. B adds prefetch 32, BF16 expert-weight reuse, fused MoE
routing indices and stable CE/cap-zero activation; embedding routing is a
separate later control.

| Variant and command | Added to preceding configuration | Physical tokens/s | Mean update, s | Max allocated / reserved, GiB |
| --- | --- | ---: | ---: | ---: |
| [A](artifacts/raw/schedule-embedding-a-timing-140-003/metadata.json) | Historical A | 149,636.307 | 14.014994 | 70.387 / 71.670 |
| [B](artifacts/raw/schedule-embedding-b-timing-140-004/metadata.json) | Four fixes | 156,741.317 | 13.379701 | 68.520 / 69.807 |
| [C](artifacts/raw/schedule-embedding-c-timing-140-004/metadata.json) | FlashQLA `external_gate` | 160,204.861 | 13.090439 | 68.554 / 69.746 |
| [D0](artifacts/raw/schedule-embedding-d0-timing-140-004/metadata.json) | Shared MTP/layer-0 FFNs and ordinary row-owner embedding | 162,271.645 | 12.923712 | 66.963 / 69.201 |
| [S](artifacts/raw/schedule-embedding-s-timing-140-004/metadata.json) | GFSDP transition repairs | 163,221.371 | 12.848514 | 66.954 / 69.201 |
| [SE](artifacts/raw/schedule-embedding-se-timing-140-004/metadata.json) | `compact`, `owner_add`, `owner_lookup` | 163,228.763 | 12.847932 | 66.924 / 68.334 |
| [SE-fused](artifacts/raw/schedule-embedding-se-fused-timing-140-004/metadata.json) | Embedding `routing` | 163,154.589 | 12.853773 | 66.962 / 68.352 |

The [timing tables](timing-results-140.md) contain every requested incremental
and cumulative comparison, supervised throughput and memory deltas. The
[full JSON](timing-results-140.json) retains all-rank cadence, medians, p10/p90,
ten-update bins, first/last windows, memory, exact stages, activation/input/cache
identities and compiler observations. Plots show
[all 140 records](plots-140/all-updates-cadence.svg),
[measured cadence](plots-140/per-update-cadence.svg),
[ten-update bins](plots-140/ten-update-bins.svg),
[first versus last 20](plots-140/first20-vs-last20.svg), and
[warmup bins](plots-140/warmup-bins-11-40.svg); PNG versions are beside the SVGs.
The first cadence record has no preceding interval and remains null rather than
being invented. This does not affect the declared measured window.

### Direct embedding routing comparison

SE-fused versus SE is **−0.0454% physical throughput**, **+5.841 ms per update**,
**+38.49 MiB maximum allocated** and **+18 MiB maximum reserved** in this one
fixed sequence. SE versus S is effectively flat (+0.0045% throughput).
These small observations do not establish a repeatable routing gain or penalty.
The full window retains the SE update-106 excursion and all GC intervals.
There is no tail-window substitution or automatic repeat matrix.

### Current-data drift and compiler coverage

The [updated drift investigation](../flash-qla/timing-drift.md) finds the
hump/recovery in all seven corrected-data variants. Mean host cadence in the
last 20 measured updates is 6.77–8.08% shorter than in the first 20, with the
same direction on all 56 rank/configuration pairs. Host callback synchrony
cannot identify a compute or communication straggler. The cause remains open;
the record separates measured facts, source-supported mechanisms and hypotheses,
and specifies the missing discriminator and an unexecuted bounded diagnostic.

A through SE have stable observed compiler state under partial coverage.
SE-fused adds 13 `_routing_metadata_kernel` entries on ranks 0, 3, 4, 5, 6 and 7;
their exact keys and block/divisibility signatures are retained in the
[drift evidence](../flash-qla/current-data-drift-evidence.json). First observation
is only known to lie in `(40,140]`; compile versus cache load and its duration
are unknown. Forty warmups therefore do not close the fused specialization
gate. Separate update-21 captures have passed their Perfetto mechanism and
GPU-tail checks; see [revised Perfetto profile findings](#revised-perfetto-profile-findings).
The fused specialization first-use and settling questions remain unresolved.

## Revised diagnostic execution

All four fresh diagnostics completed 21 updates on every rank. Ranks 0/1 each
collected the first three complete F+B microbatches of update 21; ranks 2–7
finished the full update without creating a profiler. All eight ranks completed
the remaining microbatches and optimizer. Stage receipts match the corresponding
timings and report zero device reads. Original traces remain under each run's
`native/profiling/traces/iteration_21/` directory.

| Variant | Native capture proof | Actual stages | Original wrapper |
| --- | --- | --- | --- |
| D0 | [Proof](artifacts/raw/adjudication-diagnostic-004/D0/adjudication.json) | Empty | Known nullable guard, child exit 0 |
| S | [Proof](artifacts/raw/adjudication-diagnostic-004/S/adjudication.json) | Empty | Known nullable guard, child exit 0 |
| SE | [Proof](artifacts/raw/adjudication-diagnostic-004/SE/adjudication.json) | `compact`, `owner_add`, `owner_lookup` | PASS |
| SE-fused | [Proof](artifacts/raw/adjudication-diagnostic-004/SE-fused/adjudication.json) | Same three plus `routing` | PASS |

The [diagnostic proof execution](artifacts/raw/diagnostic-proof-execution-004/execution-receipt.json)
records four successful native-parser checks under the same reviewed method.
Their status is deliberately `DIAGNOSTIC_PARSER_PASS_TAIL_PENDING`: real
capture structure and full-update counters pass, while final launch/device-tail
acceptance is recorded separately in the immutable
[final-005 receipt](trace-analysis/capture-acceptance-final-005.json). All eight
actual MCP imports, event and annotation counts, matched and inverse launch
flows, semantic comparisons and asynchronous tails pass. The raw native proofs
and wrapper outcomes remain unchanged. Every task training process is stopped,
the telemetry logger exited cleanly and all eight GPUs are idle. No additional
GPU work is scheduled.

## Revised Perfetto profile findings

The [comparison report](trace-analysis/comparison-004.md),
[machine-readable comparisons](trace-analysis/comparison-004.json), final-005
receipt, four native proofs above, and [rank 0](trace-analysis/four-way-004-rank0.html)
and [rank 1](trace-analysis/four-way-004-rank1.html) timelines retain the
profile evidence. All eight actual MCP imports pass original-category
reconciliation; annotations and launch/device correlations pass, inverse
work-launch coverage has no unmatched launches, and drain bounds cover the
asynchronous device tails. Each native proof retains 21 updates on all eight
ranks, all 32 F+B microbatches and optimizer work. The independent
[final integration review](final-campaign-review.md) passes with no unresolved
material reporting finding.

Across all 24 semantic cases, actual layers.7 compiled-kernel overlap with the
MTP-parent reduce-scatter is **zero**. Candidate layers.7 kernels begin
2.336–56.065 microseconds after that reduce-scatter ends; the remaining wait
cause is unknown. Both MTP parameter all-gathers finish before head reduce-scatter
in all 18 S/SE/SE-fused cases, and norm packing starts before parent
reduce-scatter ends in those 18 cases. Early FFN reduce-scatter overlaps
generic local compute,
but overlap with positively identified remaining MTP-prefix compiled compute is
**UNESTABLISHED** for both early FFN reduce-scatter and layers.7 dense all-gather.
The generic local-kernel overlaps do not establish MTP-prefix ownership.

The inherited `head_to_ffn_gap_ms` is an AG-adjacent original-CUDA-stream-7
inactivity proxy within the same host-launch-owned microbatch: the gap from the
latest stream-7 device event ending at or before backward FFN all-gather end to
the earliest stream-7 event starting at or after that end. It is not a
head-reduce-scatter-to-first-compiled-FFN-kernel interval. The measured proxy
ranges are D0 5.214–11.012 ms, S 0.874–11.863 ms, SE 0.845–12.343 ms, and
SE-fused 0.888–12.466 ms. Earlier weight readiness does not prove all FFN
compute waits are gone; these ranges support no causal regression claim.

Both SE variants retain a compact BF16 fill; the fill is smaller, not removed,
and the FP32 dense owner-gradient zero remains. Each forward uses direct BF16
lookup and each backward has eight ordered owner adds. SE has zero routing-metadata
executions; SE-fused has three per rank. Fusion reduces local kernels from 30 to
22 per forward, removes one DtoD copy and retains two DtoH copies and two host
stream synchronizations per forward. Across three forwards, local-kernel sums
fall by only 0.035359 ms on rank 0 and 0.028099 ms on rank 1; this is not TPS
evidence. The
separate unprofiled SE-fused/SE pair measured −0.04544% physical TPS,
+5.841 ms/update, +38.49 MiB maximum allocated and +18 MiB maximum reserved;
this single sequential observation remains provisional.

All seven corrected-data timing variants retain the hump/recovery, with a
6.77–8.08% mean decrease from the first to last 20 measured updates. The 13
SE-fused routing specializations are first observed only in `(40,140]`; compile
versus cache load and settling duration remain unresolved. The capture comparison
does not identify the cause of timing drift or the remaining dependency wait.
The measurement campaign is complete, but the original layers.7 overlap
implementation objective is **unmet**; no production-implementation-complete
claim follows from this result.

## Previous implementation and short-window campaign

Historical scope and implementation evidence from the original 15 November
assignment follow. Its source, focused checks, components and short timings are
integrated; historical SE enabled all four stages. The revised measurements
above supersede the old timing/profile gates. The original layers.7 compiled
compute overlap expectation remains unmet in the new captures.

## Scope and finite acceptance

The source starts at `e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f` in this
task's flattened `gift/` snapshot. D0 is current production-data D: the four
fixes, FlashQLA `external_gate`, shared-storage MTP/layer-0 FFN splits and
standard-autograd row-owner embeddings. S adds both GFSDP transitions; SE adds
the retained precision-preserving embedding optimizations. Historical A/C are
not performance controls.

- [x] Review active paths, lifetimes, precision, upstream implementations and
  expected runtime costs before implementation; preserve contributor edits.
- [x] Implement multi-target head prefetch, existing-region remapping and the
  real-completed-FFN post-backward transition, without new GPU pool storage,
  changed collective counts/payloads/normalization or expert-group remapping.
- [x] Implement compact source-local coalescing, ordered owner accumulation,
  exact owner lookup conversion and compatible routing fusion; preserve normal
  returned gradients/autograd and introduce no peak-memory regression.
- [x] Pass focused GPU numerical/autograd, storage/event, unused-callback,
  compile/SAC, accumulation/reset and affected state-lifecycle regressions.
- [x] Measure representative eight-rank embedding components before native
  comparisons: ten warmup/ten samples, forward/backward, joined CUDA events,
  synchronized wall/host time, all-rank distributions, kernels and memory.
- [x] Complete an integrated native full-update smoke at accumulation 32.
- [x] Execute matched D0/S/SE ten-warmup/ten-measured complete-update timings;
  retain step series, rank peaks/retries/OOMs, actual activation/configuration
  and cache/settling evidence. Repeat only to resolve promising/ambiguous effects.
- [x] Supersede the historical short-window profile gate with the four fresh
  update-21 captures and their MCP comparisons, original traces, counts, flows,
  tails and timelines; see the
  [revised Perfetto profile findings](#revised-perfetto-profile-findings).
  The original layers.7 overlap expectation remains unmet.
- [x] Resolve one independent combined integration/runtime-cost review and
  verify its affected findings; consolidate evidence and update status owners.

## Reconciled execution and controls

The shell is already on the authorized
`lm-mpi-job-23027817-4bfe-4730-926f-5c2de590b32b-mpimaster-0` server. Initial
live inspection found eight idle H100 80GB GPUs, zero used device memory and no
training process. Existing Perfetto services are preserved. The task directory
is on the verified SR008 NFS mount; use `/opt/torchtitan-venv/bin/python3.12`
directly and the existing native runner, without SSH or environment changes.

The unrelated capacity-group campaign has uncommitted source/test and
status/planning changes, including an incomplete native pair. Preserve those
bytes and its evidence. This campaign's measurement bootstrap will import the
byte-exact original capacity module from that campaign's existing control
snapshot, verifying its hash against HEAD. This excludes the unrelated change
from every D0/S/SE process without modifying contributor source.

Reuse the validated production tokenizer and real `data-002` inputs described
in the [baseline](../../docs/baseline.md#selected-production-tokenizer-and-real-data-inputs)
and [input receipts](../production-data-abcd/report.md#execution-route-and-input-discovery).
Keep B2/S4096, EP8/FSDP8, eight trunk layers/MTP1, accumulation 32/global batch
512, FP8 computation/BF16 parameter communication/FP32 gradients, selective SAC,
model/loss compilation, seed 2048, LR 65000/2000 and logical epoch 102400.
Reuse the completed D compiler cache at
`runs/production-data-abcd/artifacts/raw/proddata-d-timing-001/cache` for all
variants; the cache contains rank-local Triton, TorchInductor, DeepGEMM,
HybridEP and TileLang artifacts. New kernels may populate that same validated
cache during warmup. Inspect cache/step receipts for actual settling.

## Selected approach and preimplementation cost review

The existing GFSDP explicit prefetch list, idempotent unshard and region release
events provide the scheduling primitives. Add one default-off typed scheduling
control, declare a narrow MTP-FFN-to-final-layer/norm transition before lazy
binding, and reuse bounded aliases of the FFN's existing data/gradient regions.
Only the final layer's matching dense group moves; its expert group and the
layer-0 split retain their existing paths. A completed-backward guard excludes
unused-group root callbacks. Extend retained-owner validation to overlapping
contexts, and keep whole-storage embedding borrowing aware of every alias.
Expected added runtime work is bounded Python checks and existing event waits;
aliases are construction-time views. No communication storage, tensor copies,
device-value reads, host synchronization or collectives are added.

For embeddings, retain the installed PyTorch implementation at
`8b353065ce260336c5fdfa74d754e8bd9b91b062`. Its
[embedding dispatch](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/aten/src/ATen/native/cuda/Embedding.cu)
and [backward kernels](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/aten/src/ATen/native/cuda/EmbeddingBackwardKernel.cu)
allow compact inverse-index coalescing when the original and compact calls use
the same reduction algorithm. Guard the dispatch using actual shape/device
metadata; retain the original full-table path when equivalence is unproven.
The production ordered path accumulates sequential groups of ten in FP32,
sums FP32 partials and rounds once to BF16. Save the inverse in place of the
original IDs, send the compact native result directly and retain the ordinary
owned zero-initialized local FP32 gradient returned to autograd.

Narrow Triton kernels replace per-source BF16-to-FP32 copies plus owner adds,
FP32 gather temporaries plus BF16 copies, and compatible routing preparation.
Sources remain serialized in rank order; each source contributes unique IDs.
The installed CUDA 13 `index_add_` path may use FP32 global atomic arithmetic,
whose subnormal behavior must be matched in the fused path. Direct lookup must
match native GPU BF16 conversion bits, including ties, subnormals and nonfinites.
Signed int64 validation/counts and coordinated invalid-ID rejection remain.
Unique, fixed count exchange, host-visible variable splits and payload exchanges
are preserved. No new general routing protocol or dependency is introduced.

Expected memory reductions are the full-vocabulary BF16 coalesced table and
native partial scratch, per-source FP32 conversion scratch and FP32 selected
rows. The inverse replaces an equally sized saved ID tensor. Borrowed views
must remain within their existing event-protected lifetime; temporary compact
outputs, saved tensors and allocator peaks are explicit measurement gates.
The model, loss, gradient normalization, optimizer, EMA and checkpoint layouts
remain unchanged; affected state checks and native compile/SAC integration are
still required. The embedding device span was about 0.69% of the supplied
three-microbatch window, which bounds expectations but predicts no TPS gain.

## Findings retained during implementation

The first GPU owner-add check rejects the atomic candidate: adding BF16
subnormal bits `0x0001` produced FP32 `0x00010000` in the actual native
conversion-plus-`index_add_` path, while Triton atomic addition produced zero.
The pinned source's possible atomic dispatch was therefore insufficient to
establish the executed arithmetic. Use the unique-within-source invariant and
serialized source launches to test a single-writer FP32 load/add/store kernel;
its boundary and nonfinite parity subsequently passed, including width 4096.
The already-passing compact
coalescing and direct-lookup conversion checks do not excuse this discrepancy.
The embedding implementation record retains the exact failed and corrected
commands/results; no tolerance or accumulation precision is relaxed.

The two unsuccessful one-rank NCCL probes used relative checkpoint directories.
The existing [owner-storage record](../ffn-row-owner/owner-storage.md#historical-cudanccl-receipt-for-manual-accumulation)
documents the same `file://` URL trap: parsing a relative path loses its netloc
and waits at the wrong root path. The supported eight-rank rerun used a fresh
absolute checkpoint directory and passed. No production code or NCCL settings
were changed to address those launch failures.

## Focused correctness results

| Check | Result and scope |
| --- | --- |
| GFSDP focused CPU/Gloo | 69 passed, 1 skipped, 1 deselected; the unchanged Gloo baseline version-counter failure is retained in the implementation record. |
| GFSDP CUDA/NCCL | [71 passed](gfsdp-cuda-nccl.log), including actual delayed readers, alias/pool checks and 32-microbatch AOT/SAC/DCP continuation. Tiny control/split unique pool storage is 530,560 bytes per rank in both variants; this is not the full-crop memory gate. |
| Owner-add final widths 16/4096 | [2 passed](owner-add-full-width.log), raw FP32 bit parity with serialized native additions, including subnormals and nonfinites. |
| Other final embedding kernels | Compact native/full-table BF16 parity, exact lookup conversion bits, int64 routing limits/empty cases and final dynamic-scalar signatures pass as recorded in [the implementation evidence](embedding-implementation.md). Real production IDs also pass compact parity with a synthetic matched derivative. |
| Optimized eight-rank embedding NCCL | [Exit 0 on all eight ranks](embedding-cuda-nccl.log), using `--optimize-embeddings` for routing and state checks: ordinary backward/hooks/`autograd.grad`, accumulation-32/reset, scratch/reader lifetime, clipping, AdamW, EMA, flat DCP and next-update continuation. Checkpoints are under ignored `artifacts/raw/embedding-nccl-state-001`. |
| Native adapter | Actual wrapper-main/FlashQLA-main hooks and seven parser recipes pass with CUDA hidden, including one-update buffer inventory and round-2 IDs; original capacity overlay matches HEAD and contributor bytes remain unchanged. The initial inventory smoke recipe failed parsing at window 0 and was corrected to the required window 1 before GPU execution. |
| Independent combined review | [R1 resolved](review.md#r1-resolution): enabled transition explicitly requires MTP1 in public/direct guards; 9 focused validation cases pass. No other confirmed source correctness defect. |
| Final source checks before native execution | Scoped `ruff check --select F821,F811,F841 --no-cache` passes for all 13 owned source/test Python files; `git diff --check` passes. The unrelated capacity source/test hashes still match the initial-state receipt. |

These focused checks alone do not establish full-crop memory, component/native
performance or the intended production schedule; the measurements below supply
those separate gates.

## Embedding component measurements

The [component comparison](component-results.md) and
[compact machine-readable result](component-results.json) retain all-rank
distributions, source/cache/input identities, exact commands, kernel counts,
temporary accounting and hash-verified raw sample/profile references. All five
staged prefixes passed. The source-derived native partial tensor sizes are
distinguished from uninstrumented allocation lifetime and measured allocator
peaks. Multi-second first-use compilation, later settling and Quack cache
fallback warnings are retained. This single fixed-order ladder supports
carrying all four stages into native validation; it establishes no causal
component or training-speed claim.

## Control profile reconciliation

The existing annotation-preserving D rank-0/1 derivatives were loaded through
Perfetto MCP. Each imports 8,230 kernels, 202 copies, 885 memsets and 330 GPU
annotations, with no nonzero error/data-loss counters in the executed query.
[Control compatibility](trace-analysis/control-compatibility.json) compares the
current D0 resolved configuration and all eight bootstrap receipts with that
diagnostic: shared inputs, configuration hashes, active features, runtime and
schedule match. Differences are output/profiler cadence and the new explicit
false controls. Default-off Python/source revisions differ, so the saved trace
serves as a mechanism reference; the new matched D0 timings own magnitude.
[The raw MCP receipt](control-mcp-import.json) retains trace handles and SQL.
The subsequent [control analysis](trace-analysis/README.md) reconciles all
9,317 device events and 330 GPU annotations per rank, all launch correlations,
original fields and added stream metadata. It reproduces the objective's head
gaps and embedding totals. All six MTP-parent reductions have zero overlap
with layers.7 compiled compute; earlier norm/auxiliary overlap is reported
separately. New S/SE captures and before/after comparisons were open when this
historical control analysis was written; the revised four-variant captures and
current limits are recorded above.

## Integrated native smoke and communication storage

Both `schedule-embedding-se-smoke-001` and
`schedule-embedding-d0-smoke-001` completed one full accumulation-32 optimizer
update on all eight ranks. Native/campaign completion, unchanged source/config,
original-capacity overlay and process cleanup checks pass. Losses were finite
(15.32932 / 15.32930 respectively); these values are not a new numerical-parity
criterion. Allocator retry/OOM deltas were zero.

All eight D0/SE inventory pairs contain the same seven backing storages and
6,973,565,952 unique communication-pool bytes per rank. Exactly two view remaps
occur: layers.7 dense data occupies 170,460,160 bytes at offset 199,582,720 in
the existing 268,959,744-byte FFN data region; norm gradient input occupies
540,672 bytes at offset 399,165,440 in the existing 537,919,488-byte FFN gradient
region. All other views, including layers.7 expert and the layer-0 split, retain
their contexts and offsets. The parent independently inspected the actual SE
rank-0 inventory; the native execution record checks all ranks. Inventory JSON
files are under each ignored run's `measurement/` directory. This proves pool
storage parity; full-update performance/memory and schedule evidence remain
separate gates.

## Historical component and smoke results

Both implementations and their focused GPU correctness checks are complete;
the independent source review finding is resolved. All five eight-rank
component prefixes completed before native timing, with ten warmups and ten
samples per scope. The final prefix's rank-max median integrated CUDA span was
1.915 ms versus 2.722 ms for D0 in this single fixed-order ladder. Integrated
peak allocated/reserved bytes fell from 5,929,826,816 / 18,941,476,864 to
4,824,576,512 / 17,049,845,760; retry/OOM deltas were zero. These component
observations do not establish a native-training speedup or production peak.
Combined native smoke and full-crop communication-storage parity pass. The
historical [D0/S/SE short timings](native-results.md) completed. Current full-window
timings, memory and fresh four-variant trace findings above own the revised
measurement verdict, including the unmet layers.7 overlap.
