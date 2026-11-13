# Current state

Updated: 13 November 2026. This file owns the integrated project state;
[the plan](../planning/README.md) owns future priorities. Update this file when
new evidence changes a conclusion, not after each diagnostic command.

## Baseline: selected; campaign cancelled

The [canonical baseline](baseline.md) is a GIFT 3.5 100B-dev crop: eight trunk
layers, D4096, one dense FFN and seven MoE layers, 128 routed experts, expert
width 1536, top-8, one shared expert and MTP1. It preserves production layer
widths rather than substituting the small L8 architecture.

EP8, dense FSDP8, expert FSDP1, PP/TP/CP1; GFSDP2, dense+sparse FP8, model/loss
compile, fused AdamW, CuTe norms and EMA. Microbatch 2, sequence 4096, global
batch 512, accumulation 32. Per-layer SAC and token capacity 50,000 are fixed.
The [YAML](../gift/configs/experiments/gift_100b_crop8_ep8_fp8_compile_cute.yaml)
and [model preset](../gift/configs/presets/models/gift_moe/gift_3_5/100b_crop8.yaml)
are the machine-readable defaults. The baseline document owns the full settings.

Task GIFT is based on upstream `6fd9a3230ab6361f11bbbf0b6ec5a5f125b78fc2` with
selected development mechanisms from `cute_norm_prodrun@ec47fb8e40a9a1e55d3c18dfa704ca104ae773af`
and its 100B ancestor. Implemented mechanisms include per-layer SAC, adaptive
expert-bias balancing, typed production initialization, monitoring loss chunks
and a functional grouped-FP8 output compatible with SAC result caching.
[Provenance](../references/provenance.json) pins the source and dependency history.
The runtime uses native `torchtitan.train`, installed DeepGEMM `2.3.0+f093a4c`
and DeepEP `1.2.1+1b8f467`.

## Historical HF smoke verification

| Item | Outcome | Evidence / limit |
| --- | --- | --- |
| Eight-H100 smoke | PASS: 10 optimizer updates, 320 microbatches per rank, exit 0 | [Report](../runs/100b-crop8-ep8-smoke/report.md), [receipt](../runs/100b-crop8-ep8-smoke/result.json) |
| Monitoring | Steps 5 and 10 completed | Smoke interval 5; then-default interval 50. Ordinary/monitoring loss chunks 2/32 supported by resolved config and active source/hook path |
| Observed metrics | Final rank-0 loss 14.57517, gradient norm 7.6661; finite logged metrics | These do not establish numerical equivalence or convergence |
| Memory | Maximum native console metric 72.61 GiB | Reserved peak plus report-time non-PyTorch usage; not a simultaneous device-wide peak |
| Lifecycle | All ranks tore down, GPUs released; no checkpoint, profiler trace or memory snapshot | No checkpoint continuation was tested in this smoke |
| Source checks | Focused initialization, routing, SAC, loss, FP8 regressions and configuration goldens passed | Detailed counts and CPU test-double scope are in the report; 112 golden checks passed |
| Known local test limitation | Combined local CPU harness: 12 compiled-router failures after fake-Triton preload, plus 55 passes, one skip and one deselection | Isolated compiled regressions passed. The combined suite is not declared passing |

Smoke data are bundled `c4_test` and the test tokenizer. Production data,
long-run stability, convergence, stable throughput, numerical candidate parity,
and production-scale PP/inter-node behavior are not established by this result.
The eight-GPU crop does not reproduce EP32 expert count per rank, expert-weight
collectives or production PP overlap. Token-drop pressure remains a future study.

## Candidate inventory: preserved, not integrated

[Candidates](candidates.md) maps mechanisms to 109 byte-preserved upstream files,
licenses, tests and hashes under `references/`.

- Kindulov: softcap/CE fusion, chunk lifetime reduction and FFN recomputation.
  The crop has no positive softcap; routed recomputation in the reference rejects FP8.
- Hidden z-loss: SwiGLU/FP8/gate-up integration is preserved. The crop's existing
  z-loss is on final norm input, so this is not a direct replacement. External
  CudaCherry kernel sources at the recorded pin are still missing.
- Vision-related work: CompactCE, streaming prefetch, GFSDP, frozen storage and
  metrics. This collection is not confirmed as the promised vision team list.
  CompactCE rejects MTP; the baseline has MTP1.
- Additional hypotheses retained from earlier source reviews: MLA backend,
  GFSDP prefetch, MoE metadata/communication, GDN recomputation and linear CE.
  Their former 700B-specific settings are not crop defaults.

No candidate has been promoted or benchmarked on the current crop.
Production 100B traces and the exact remaining production MR list have not been
established by the preserved research. The small L8 config remains a donor
example, not a confirmed future test setup or our baseline.

## CUDA Graphs: low priority, not globally incompatible

The historical 3B campaign demonstrated persistent loss-only forward/backward
replay. Broader capture encountered tensor/workspace lifetime issues and repeated
dense backward recording. A valid repeated speed comparison was not completed.
A later graph-off trace showed about 99% of main-stream kernels already queued
before prior work finished, reducing evidence for a large CPU-submission benefit.
See [the conclusions and limits](cuda-graphs.md).

Decision: keep CUDA Graphs off in the default. This is not proof that all graph
boundaries fail, nor that the 3B trace predicts the current crop's performance.
Revisit only with a concrete crop bottleneck and a finite comparison plan.

## Cleanup-only assignment and preserved campaign results

The original [overnight campaign](../planning/overnight-streaming-perf.md) is
cancelled and incomplete. The current assignment removes full-model numerical
machinery, disables detailed performance monitoring and records the revised
short-run/local-block protocol. Cleanup is complete. Do not resume candidate experiments, training,
benchmarks, new traces or numerical block comparisons. The active plan owns the
finite cleanup checklist. Numerical machinery is removed. Canonical/generated
configuration and short-run behavior passed 101 scoped CPU/config checks and an
independent integration/runtime-cost review. The [cleanup receipt](../runs/overnight-streaming-perf/cancellation-cleanup.json)
confirms no owned numerical processes or reference tensor blobs remain; 40,650
files totaling 381,915,112,855 logical bytes were deleted, without attributing NFS
free-space changes. Compact history, MDS, candidate source and Perfetto evidence
remain. No training, benchmark, new trace or local numerical block run occurred.

Preserved completed evidence:

- The 896-row, full-length MDS preparation and real eight-rank loader checks,
  plus the ten-update/320-microbatch native smoke, remain valid within their
  recorded scope. The [MDS receipt](../runs/overnight-streaming-perf/g1-smoke-002-receipt.json)
  also records zero frozen parameters on every rank. Canonical data selection
  and packing/partitioning remain unchanged.
- Historical monitoring-inclusive G2 controls measured median input throughput
  **124454.81175967702 global tokens/s (15556.851469959627 per GPU)** over
  updates 111–160 including monitor 150. These are historical results, **not
  directly comparable to future monitoring-off measurements**. Their W110/cache
  acceptance method is archived, not a future prerequisite.
- G3 update 111 trace acceptance was recovered post hoc by category-aware parsing;
  the original parser failure stays recorded. Both ranks were analyzed using
  actual Perfetto MCP. [Canonical results](../runs/overnight-streaming-perf/g3-baseline-perfetto.json),
  SQL, compact source evidence and compressed trace pair are retained. The input
  wait is a V2 hypothesis, not a measured gain.
- C1 B2 passed; compact history is retained. C2 A reached terminal PASS export
  at 08:34:48 UTC; child 418189 and its process group stopped, and its controller
  was absent at 08:45 UTC. C2 B and C50 were cancelled without launch. The
  full-model validation programme is abandoned, not completed; there is no
  replacement whole-model parameter/gradient/Adam acceptance gate.
- V2 source variant `21c1891dda531a1222e6c350802f2765758794f9` on
  `vkropotin-v2-prefetch` is preserved with default 0 and its
  [source/CPU review](../runs/overnight-streaming-perf/v2-prefetch-review.md).
  Actual eight-owner Gloo/DCP and native/performance gates were not executed
  or completed. No candidate has been accepted.

Future separately authorized comparisons start with **10 warmup + 10 measured
optimizer updates**, accumulation 32, with LR horizon 65,000/LR warmup 2,000
unchanged. Disable detailed monitoring callbacks/hooks rather than increasing an
interval; ordinary loss uses two chunks. Retain lightweight native loss/time/TPS/
errors, EMA and training losses. Validation, checkpoint saves, snapshots,
numerical capture and CUDA Graphs remain off. Repeat promising or ambiguous
short-pair results when needed; one pair is provisional. Extend warmup only for
actual compilation/runtime settling that affects the window, not a cache entry
alone. No mandatory 110/50, exhaustive-cache gate or large control matrix remains.

After performance identifies a candidate worth retaining, compare the smallest
meaningful affected block on matched boundary inputs and needed weights/state;
compare affected backward with matched incoming gradients if it changes. Assess
precision, casts, rounding, reduction order and fusion, with a higher-precision
local reference when needed. Do not assume baseline exactness, universal tolerance
or bitwise equality. See [validation](../planning/validation.md). No such checks
are run during cleanup.

The source execution pin before cleanup is `e9048f65`. The
[report](../runs/overnight-streaming-perf/report.md) and
[results](../runs/overnight-streaming-perf/results.json) retain the original
measurements, failures and cancellation status. Original G4–G7 incompleteness is
not an instruction to continue the campaign. Source/config/protocol and cleanup
acceptance are verified; cleanup commit `aa45e3cd8d0c8856d383e72a9ad5f5d60c0530ae`
was pushed to `origin/main`. Stop after recording completion; no experiments
are authorized by this completed assignment.

## Working layout and historical consolidation

Current documents, editable planning, active flattened `gift/`, inert candidate
references and compact run evidence remain at their existing paths. Historical
consolidation records are recoverable from `9d56011df976017972d8f2993b910bde03487e2a`
using [recovery instructions](../references/README.md). That earlier consolidation
had no source/runtime effect and passed its documented link/JSON/hash checks;
those historical checks do not substitute for this cleanup's affected checks.
