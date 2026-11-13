# Current state

Updated: 13 November 2026. This file owns the integrated project state;
[the plan](../planning/README.md) owns future priorities. Update this file when
new evidence changes a conclusion, not after each diagnostic command.

## Baseline: selected and running

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
| Monitoring | Steps 5 and 10 completed | Smoke interval 5; canonical interval 50. Ordinary/monitoring loss chunks 2/32 supported by resolved config and active source/hook path |
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

## Next decisions and working layout

The authorized [overnight MDS/performance campaign](../planning/overnight-streaming-perf.md)
is running. Entry checks verified eight idle H100s, the existing runtime and NFS,
and actual Perfetto MCP path/query access. Reference applicability is reconciled
in the [campaign evidence](../runs/overnight-streaming-perf/reference-coverage.md);
no candidate gain is established. The 896-row full-length MDS fixture passed
two real eight-rank loader checks; native profiler/measurement support passed
scoped tests and review. Native smoke attempt001 failed in a metadata census
before any model update; its scoped fix passed regression/review. Attempt002
passed ten updates and 320 microbatches on every rank, including finite metrics
and monitoring at 5/10, with clean teardown and released GPUs. The
[MDS smoke receipt](../runs/overnight-streaming-perf/g1-smoke-002-receipt.json)
also records zero frozen parameters on all ranks, making frozen-storage V4
inapplicable to this crop. The canonical YAML now selects that MDS fixture;
its affected decode/golden and path/shard-equivalence checks passed. Only the
data mapping changed. Calibration completed60 finite native updates per rank,
including normal monitor50. The first W60 control completed110 native updates
but failed measurement acceptance because Triton cache state grew on three ranks;
the second control is paused for a bounded warmup/cache diagnosis.
The reviewed task adapter records existing compiler state at window boundaries
and rejects DeepGEMM cache misses after W, using a guard validated on both active
grouped GEMM paths. Native measured-window acceptance and A/A remain open; no
performance baseline exists yet. The execution ledger owns
progress; the [report](../runs/overnight-streaming-perf/report.md)
and its results file own measurements and remaining evidence gates.

The working tree now keeps four current documents, an editable planning area,
the active GIFT source, candidate references and compact run evidence. Duplicate
scope/status scaffolding, the second framework and intermediate campaign files
were removed from the tracked tree. Their complete tracked versions remain at
Git revision `9d56011df976017972d8f2993b910bde03487e2a` with
[recovery instructions](../references/README.md). Raw traces/logs outside Git
are not included in that recovery guarantee and were not deleted by this cleanup.
The GIFT source, configuration, candidate source bytes and smoke launcher are
unchanged; this reorganization has no production-runtime effect.

Consolidation checks: all maintained local Markdown links resolve; all retained
JSON parses; all 109 candidate source checksums and eight graph-evidence hashes
match their manifests; the baseline config hashes match the successful receipt;
`bash -n` and staged whitespace checks pass. Git confirms no change under `gift/`
or to the smoke launcher/result. No GPU rerun was needed for document moves.
Tracked files decreased from 57,849 to 27,768; task Markdown outside training and
reference snapshots decreased from 150 to 13 at that consolidation commit (before
the later overnight plan and prompt were added). Git history is retained, so this
reduces the working tree, not the size of already published historical objects.
