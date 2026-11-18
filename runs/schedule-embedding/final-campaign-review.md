# Revised campaign final integration review

Reviewed 15 November 2026 against the revised seven-timing/four-profile/drift
objective. **PASS for the revised measurement campaign; the sole reporting
finding is resolved below.** The prior implementation's
layers.7 compute-overlap expectation remains unmet.

## Finite acceptance and reused evidence

1. Minimal native selection distinguishes three-stage SE from four-stage
   SE-fused, preserving existing defaults, precision and production controls.
2. Seven fresh-start runs retain the declared 140 updates, warmup 1–40 and
   complete measured window 41–140, authenticated identities, all-rank cadence,
   token denominators, memory and original attempt outcomes.
3. Four separate update-21 diagnostics complete 32 microbatches and the optimizer
   on all ranks; ranks 0/1 preserve the first three complete F+B regions and
   asynchronous tails. Actual Perfetto MCP evidence establishes import fidelity,
   flows, stage execution and the requested schedule/embedding comparisons.
4. Direct SE-fused/SE results and the six current-data drift questions are
   reported with measured facts, mechanisms and causal unknowns separated.

Reused accepted [source/correctness review](review.md),
[selector review](selector-review.md), [original adjudication review](adjudication-review.md),
[repair review](repair-adjudication-review.md),
[diagnostic adjudication review](diagnostic-adjudication-review.md), and
[timing analyzer integration review](timing-analysis-review.md). No MTP/FP8 A/A,
broad source audit, benchmark, native execution, package installation, source
write or cache change was repeated. The relevant code-rules, perf-experiments
and trace-analysis instructions apply to this review.

## Stable timing evidence

Inspected [manifest](timing-runs-140.json), [numerical results](timing-results-140.json),
[timing tables](timing-results-140.md), the current section of
[campaign report](report.md#revised-measurement-assignment), and retained execution
and Markdown-recovery receipts. Independently hashed the numerical JSON:
`a50c7bca5149d5f27c099b6f25c1a9944955fd713d75e475766261fff4dc3a05`.

The final manifest selects retained raw A003 and six fresh 004 rows, each with a
new proof under `adjudication-140-004/<variant>`. All use final adjudicator
`594fb7db1ff5403e6bcb9ac901fe98f85c35d10e09da36dcdf1c5322810780f4`,
plan `1a193606963e2f3c7086a6afb1d6d56ad78692dae54121e257c05c18fc8a5968`
and freeze `3b3244ae1e759111b8240af929715074b751c946881202a9ff95e9f3b9918ffd`.
Original A/B/C/D0/S wrapper FAIL/1 and SE/fused PASS/0 remain distinct. Accepted
reviews establish the narrow nullable-selector recovery and complete parser
and producer-schema boundaries; native exit zero alone is not the criterion.

Reused the coordinator's [independent all-56-rank reconstruction](artifacts/raw/analysis-execution-004/primary-envelope-check.json),
which matches the seven CUDA-drained single-host envelopes, physical/supervised
token totals, cadence coverage, drains and memory counters. The full declared
window retains 209,715,200 physical and 209,664,000 supervised tokens per run;
MTP does not multiply the denominator. Update 1 has no preceding cadence and
remains null. Host cadence is not device-only compute. No tail window replaces
the primary envelope, and the SE update-106 excursion and GC intervals remain.

SE-fused/SE is −0.04544% physical TPS, +5.841 ms per complete update,
+38.49 MiB maximum allocated and +18 MiB maximum reserved. These are one fixed
sequential observation; the report correctly claims no repeatable gain or
penalty. The required incremental and versus-A comparisons are present.

The original analysis exit 1 is preserved: numerical JSON and plots completed
before a Markdown renderer KeyError. The [recovery receipt](artifacts/raw/analysis-execution-004/markdown-recovery.json)
records a renderer-only fix and unchanged numerical JSON; it does not claim a
successful original command or repeat native/analysis measurements.

## Stable drift evidence

Inspected [current investigation](../flash-qla/timing-drift.md) and its
[compact evidence](../flash-qla/current-data-drift-evidence.json). All seven
corrected-data curves exhibit the hump/recovery; all 56 rank/configuration
first/last mean comparisons decrease. Reported mean decreases are 6.77–8.08%.
The investigation distinguishes these means from historical medians and keeps
the old repeated-fixture/cache observations historical. Contaminated B003 is
excluded; historical failed/stopped attempts remain evidence.

The record separates shared callback timing from local rank-critical work;
strong rank correlation does not rule out a propagated straggler. Compiler
coverage is explicitly partial. Fused adds 13 routing kernel entries on ranks
0/3/4/5/6/7 in `(40,140]`; exact keys are retained, exact first-use updates and
compile-versus-load duration are unknown, and the settling gate remains open.
Cache entries do not establish execution of pre-existing kernels.

GC joins are labeled inferred temporal alignment; telemetry is coarse sampled
evidence with nonzero clock-event masks. The current capacity/padding bounds
constrain dropping and route volume without equating volume with runtime.
Changing MoE balance remains plausible and unproven. Missing joint early/late
expert/destination/readiness evidence is named, along with a bounded unexecuted
diagnostic and the decisions its outcomes would change. Update-21 traces cannot
establish what changes between updates 41 and 140. Causal attribution remains
open without preventing completion of the requested bounded investigation.

## Four-profile integration and independent checks

Inspected the actual [comparison](trace-analysis/comparison-004.md),
[24 semantic rows and exact operation differences](trace-analysis/comparison-004.json),
[identities](trace-analysis/identities-004.json), native proofs, saved actual MCP
load/query responses, and the bounded capture/semantic/closure methods. Final
[capture acceptance](trace-analysis/capture-acceptance-final-004.json) hashes to
`6451cc05cb8ff0404f31794d3dd76bbd30b65b78c8cf0ef3903a115162f65bff`.
Independently rehashed all **33** final receipt path/SHA references, including
native proofs, actual MCP responses, comparison/detail, methods and timelines;
all match. This is a metadata/derived-evidence check, not a repeated raw-trace
conversion or broad trace analysis.

Checked **144** saved actual MCP query responses: none report an error or
truncation. Reused the recorded original-versus-derivative permitted-change
verification and actual event censuses rather than rereading all original traces.
All eight identities agree with the final receipt and their exact native-proof
original trace hashes. All complete categories match, including 330 GPU
annotations each. Every kernel/copy/memset has a correlation-matched launch flow;
inverse work-launch coverage has zero unmatched launches. SQL pages and final
partial pages preserve coverage beyond the server's per-result cap.

All four native proofs use the same frozen source and final adjudicator. Each
retains 21 completed updates and all-eight-rank 32 F+B/one optimizer counts.
Their initialized stage receipts distinguish empty D0/S, three-stage SE and
four-stage fused, consistent with timing activation. Original D0/S nullable
FAILs and SE/fused PASSs remain preserved. The native proofs remain
`DIAGNOSTIC_PARSER_PASS_TAIL_PENDING`; the separate immutable final receipt
closes that gate without rewriting originals or native acceptance.

Each third microbatch retains asynchronous device work extending
17.999651–19.708998 ms beyond its CPU annotation. Final device completion lies
4.314–6.078 microseconds before the recorded device-drain return, followed by
more than 30 ms before capture closure. Flow/correlation, inverse launch census,
original hashes and drain bounds jointly support complete tails; CPU annotation
or GPU projection counts alone would not.

Independently checked all 24 semantic cases: actual layers.7 compiled-kernel
overlap with MTP-parent RS is **zero**. Both MTP parameter AGs precede head RS
and norm packing starts before parent RS ends in all 18 S/SE/fused cases.
All 24 cases retain positive FFN-RS overlap with local compute. Candidate
layers.7 kernels start 2.336–56.065 microseconds after parent RS ends. Earlier
CPU submission, annotations, auxiliary kernels or stream occupancy are not
evidence of layers.7 compute overlap. The remaining dependency cause is unknown.

Collective identity checks compare exact kernel family, counts, payloads,
BF16 parameter/FP32 gradient dtype and group across matched microbatches.
Storage lifetime/alias safety reuses the accepted source and targeted lifetime
tests; trace scheduling does not independently prove alias safety. Later
layers.6/backward gaps and both available next-microbatch boundaries are retained.
No full-update or remaining-29-microbatch absence-of-stall claim is made.

Embedding comparisons follow owner-operation launch ancestry: D0/S device
identities are unchanged; both SE variants show compact BF16 fill geometry,
ordered owner-add and direct FP32-to-BF16 lookup. The required FP32 gradient
zero and smaller BF16 compact fill remain. SE has no routing-metadata execution;
fused has three per rank. Exact per-forward differences remove eight net local
kernels and one DtoD copy while preserving backward and non-owner device
identity/counts. Copies, NCCL, local-kernel sums, device union/gaps, elapsed owner
spans and host scopes remain separate; peer waiting and nested host durations
are not added into an inferred compute or fusion speedup.

Both static timelines exist. Independently XML-parsed all **72** SVG panels,
36 per rank. No browser screenshot is available; this is a presentation limit,
not a device-tail gate. The initial immutable capture receipt is preserved and
explicitly superseded after later-backward evidence extended derived references.

## Consolidated ownership-scope clarification — resolved

`capture004.py` computes `l7_ag_overlap_mtp_remaining_local_compute_ms` from
all local compute whose ancestry does not contain layers.7. This is not a
positive identification of remaining MTP-prefix compiled compute. The FFN-RS
overlap metric similarly uses all local compute. The report table accurately
says local kernels, but neither metric establishes the more specific MTP-prefix
ownership requested in the original mechanism expectation. State that narrower
overlap as **unestablished**, unless positive module/compiled-block ownership is
demonstrated from existing evidence. This required a reporting clarification,
not a source fix or another GPU run. Generic local overlap remains observed.

The targeted follow-up resolves this finding. Both generic union fields now
explicitly name local compute; all 24 semantic rows separately mark the two
MTP-prefix-specific ownership claims `UNESTABLISHED`. The report explains why
the later overlapping anonymous compiled block cannot be positively assigned
to the remaining MTP prefix. It does not replace missing ownership with norm,
attention-recompute or other stream activity.

The retained `head_to_ffn_gap_ms` is now defined exactly as an AG-adjacent
original-stream-7 inactivity proxy, not a proven head-to-FFN dependency stall.
Independently checked the retained ranges: D0 5.214–11.012 ms, S 0.874–11.863,
SE 0.845–12.343 and fused 0.888–12.466. Candidate outliers remain visible.
Earlier parameter readiness does not establish elimination of all FFN compute
waits; no causal regression or absence of regression is inferred.

The fresh [005 receipt](trace-analysis/capture-acceptance-final-005.json), SHA256
`5c5d01039f979033963ddfbd98a4b570e0bd90c5476bc91e9295c93f5d40e0d3`,
explicitly supersedes immutable final004. Independently verified its 34 current
path/SHA references and all eight prior-derived-file archive hashes. All six
affected derived JSON files equal their archived predecessors exactly after
normalizing only the two metric names, two explicit ownership statuses and
the compact detail hash reference. Numeric values, counts, flows, tails,
payloads and operation inventories are unchanged. The corrected report SHA256 is
`631241184270d3f94a25bdb96276a8a06bc612637b6af0c9fb3e79c31adf4f76`;
compact comparison SHA256 is
`9c640dacf11d72eda3a960b2007748fa65f4b961f26c6d0a86f73bbc96586d53`.
The previously accepted eight-capture/144-response/72-SVG checks were reused;
no fresh MCP queries, broad analysis or GPU work was needed.

## Verdict and remaining limits

The revised measurement campaign's four finite outcomes have support: minimal
selector integration, seven fixed 40+100 measurements, four faithful separate
MCP profile comparisons with complete tails, and bounded current-data drift
investigation. Negative or unestablished mechanism outcomes must remain explicit.
The prior layers.7 overlap implementation objective is **not accepted** and no
production-implementation-complete claim follows from this measurement review.

No unresolved material finding remains in this bounded review. Causal drift,
fused-specialization settling, the missing remaining dependency, generalizable
speed magnitude, convergence, full-update/no-stall behavior and production-scale
behavior remain unproven. None is silently promoted by profile acceptance.
No GPU/native/profile run, installation, training-source/config/cache write,
publication, or additional benchmark was performed by this reviewer.
