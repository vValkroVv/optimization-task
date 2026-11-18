# Agreed 100B-dev crop baseline

**State:** This is the accepted default workload contract. Its canonical task-local YAML is [gift_100b_crop8_ep8_fp8_compile_cute.yaml](../gift/configs/experiments/gift_100b_crop8_ep8_fp8_compile_cute.yaml), with the model preset [100b_crop8.yaml](../gift/configs/presets/models/gift_moe/gift_3_5/100b_crop8.yaml). The 15 November rerun replaces the input baseline with real diverse data and the production tokenizer described below. Its [campaign report](../runs/production-data-abcd/report.md) owns current acceptance and measurements. The previous [MDS smoke](../runs/overnight-streaming-perf/report.md) and [HF smoke](../runs/100b-crop8-ep8-smoke/report.md) remain historical evidence for their original inputs.

## Global requirement: real data and production tokenizer

**Owner decision, 15 November 2026:** replace the test-tokenizer fixture with
real, diverse data encoded by the correct production tokenizer, and remeasure
all optimization results on that input baseline. This applies across the task:
training runs, profiling, native comparisons and data-dependent component
benchmarks, including the FFN/embedding campaign and earlier V2, BF16
cache, fusion, combined-four and FlashQLA comparisons.

The historical preparation and runner selected `gift/tests/assets/tokenizer`: 2009 IDs
and no BPE merges. The 896-row C4 fixture contains only 94 distinct input IDs;
its B2/S4096 rank microbatches have a median of 74. The model still has 128256
embedding rows. These inputs do not supply the token diversity needed for the
requested performance evaluation. A real-text corpus name or a full-size model
vocabulary does not establish correct tokenization.

The source production 100B preset selects
`/mnt/nfs/07-share/models_templates/dclm-7b-128k_digit_tokenizer_FIM`
(`gift_3_5_100b_pretrain_2t_tokens`, donor revision `ec47fb8e40a9a1e55d3c18dfa704ca104ae773af`).
Use the corresponding verified assets on the execution host; matching only
the vocabulary size does not establish tokenizer identity. The checked-in
crop YAML, active runner and MDS preparation must agree on the selected
tokenizer and newly encoded data. The completed
[production-data campaign](../runs/production-data-abcd/report.md) verifies this
replacement and measures the four requested cumulative variants plus the three
affected components. The old hard-coded tokenizer CLI override is removed.

Before measurement, record dataset/tokenizer identities and hashes, actual
input-ID bounds, unique-ID counts and owner request distributions on
representative rank microbatches, and sequence/packing/padding rules. Inspect
the actual resolved configuration and loaded batches. Preserve the selected
model, precision, SAC, compile, MTP, EMA, optimizer and accumulation settings.

Preserve previous raw runs and reports as historical evidence for their exact
inputs. Recompute timing, TPS, peak-memory comparisons and applicable trace
budgets with fresh matched controls on the corrected input baseline; do not
compare new candidates with old-fixture controls. Keep the agreed initial
10-warmup + 10-measured full-update protocol unless the active assignment
explicitly selects another duration. Existing implementation selections and
applicable correctness/lifecycle evidence remain in force; this does not reopen
the accepted MTP/FP8 A/A investigation. Synthetic edge cases can still test
correctness, but cannot replace the real-data performance measurements.

The [work plan](../planning/README.md#global-priority-rebaseline-all-runs-on-real-data)
records completion of the requested cumulative rerun. Any future comparison
must use this input contract; this requirement does not itself launch
additional compute.

## Current final task baseline

**Owner decision, 18 November 2026:** the selected production recipe is
**4fix + FlashQLA + embeddings/GFSDP + GPU capacity selector**. It extends S,
the control in the selector comparison. The 14 November selection of
combined-four and FlashQLA `external_gate` is the foundation of this recipe.

| Part | Selected setting |
| --- | --- |
| Workload | Microbatch 2, sequence length 4,096, global batch 512, accumulation 32; EP8, dense FSDP8, expert FSDP1; TP/PP/CP1. Preserve FP8 dense/sparse paths, model/loss compilation, per-layer SAC, MTP1 with weight 0.3, and configured EMA. |
| Existing accepted V2 | CPU materialized-batch prefetch depth 32. This is the already accepted V2 setting, reused as part of the combined recipe rather than an additional fifth change. |
| Three combined-four additions beyond V2 | BF16 expert-weight reuse; the explicit routing-metadata kernel using `indices_to_map`; and stable cap-zero cross-entropy through the existing `CrossEntropyLoss` and `ChunkedCELoss` path. GPU input-preparation fusion and fused Linear+CE remain excluded. |
| GDN backend | Select `external_gate` explicitly with `runs/flash-qla/run.py --gdn-backend external_gate`. The choice is recipe-scoped; the generic backend default remains separate. |
| Embeddings and GFSDP (S) | Split the MTP and layer-0 FFN groups, use row-owner input embeddings, and enable the S GFSDP scheduling repairs. Use S embedding settings, with the SE stages off. |
| GPU capacity selector | Set `branchless_capacity_drop: true` under `model.config.blocks.<block>.moe.experts.token_dispatcher.token_drop` for `gdn_moe` and `mla_moe`. Keep fixed capacity 50,000, group-capacity dropping, and the original 128-counter all-reduce. |
| Short comparison window | Logical epoch size 102,400; measure optimizer updates 11–20 after 10 warmup updates. Preserve the 65,000-update LR horizon and 2,000 LR-warmup updates used in A/B/C/D and the selector comparison. |

The [selector report](../runs/capacity-selector/report.md) records the tested S
launch path, the production flag, and the evidence behind this selection. The
flag is an explicit setting for the recipe. The shared library default is off.

The historical FlashQLA invocation and per-run values are retained in the
[FlashQLA native plan](../runs/flash-qla/native-plan.json). It selects the
backend with `--gdn-backend external_gate` and records `--steps 140
--warmup 40 --window 100 --num-prefetch-batches 32 --epoch-size 102400
--gfsdp-reuse-bf16-expert-weights`. The plan also records the operational
compiler-cache path used by those runs; cache reuse is not an additional model
change. Those 140-update old-fixture values do not override the current rerun.
Its [command plan](../runs/production-data-abcd/campaign-plan.json) uses fresh
separate timing caches and a corresponding completed timing cache for each
diagnostic.

The combined-four pair measured 149,386.56 versus 156,094.27 global physical
input tokens/s, a provisional +4.49% fixed-window difference from one sequential
pair with continuing cadence settling. The FlashQLA sequence measured 156,452.36
(FLA), 160,090.72 (`external_gate`), and 159,735.21 (`fused_gate`) global physical
tokens/s in fixed order; roughly 10.4% within-window cadence drift prevents a
general or causal throughput conclusion. The owner selected the recipe; those
measurements are not a numerical selection rule or convergence result. See the
[combined-four report](../runs/combined-four/report.md) and
[FlashQLA report](../runs/flash-qla/report.md) for evidence and limits.

Full-scale validation remains open for long-run convergence, checkpoint/resume,
EMA updates, CP warmup, and production PP4/EP32 behavior.

## Identity and scope

- Architecture reference: 100B-dev source revision `f15d33f3fd9204e5fef2adbd625b3caed2241052`.
- CuTe/runtime donor reference: `cute_norm_prodrun` at `ec47fb8e40a9a1e55d3c18dfa704ca104ae773af`, a descendant of the 100B-dev revision.
- Training source base: task-local GIFT `main` at `6fd9a3230ab6361f11bbbf0b6ec5a5f125b78fc2`, recorded in the [source provenance](../references/provenance.json). The selected implementation route is to port only the required 100B-dev mechanisms into this base; it does not replace the base with the donor branch.
- The ports include per-layer SAC, adaptive routing, development packed initialization, and loss-monitoring support. A functional grouped-FP8 output also fixes the donor's eager SAC result-caching defect; the smoke report records its regressions and source review.
- This is an eight-trunk-layer crop of the referenced model. It is a controlled single-node development workload. Current throughput evidence and its limits are recorded in [current state](status.md).
- Keep architecture, donor and trainer-base identities distinct. Historical [CUDA Graph conclusions](cuda-graphs.md) are from another workload.

## Model and training configuration

The crop has eight trunk layers with the pattern `GDN, GDN, GDN, MLA` repeated twice. The model width is `D=4096`; the vocabulary size is 128,256 with untied input/output embeddings; the MTP depth is one; and GIFT 3.5 zero-centered norms with the CuTe norm backend are retained. The config uses the MTP objective with `loss_weights: [0.3]`.

Layer 0 is dense with hidden size 10,944. Layers 1–7 are MoE layers with 128 routed experts, expert hidden size 1,536, top-8 routing, and one shared expert.

| Setting | Agreed default |
| --- | --- |
| Parallelism | GFSDP2 backend; EP8, dense FSDP8, expert FSDP1; PP/TP/CP1 |
| Precision and compile | FP8 for dense and sparse paths; compile model and loss |
| Optimizer | Fused AdamW; CuTe applies to the norm backend |
| Offload and routing | No CPU offload; adaptive load balancing; no forced routing |
| EMA | Keep enabled |
| Loss and monitoring | Performance runs disable detailed monitoring callbacks/hooks; ordinary loss uses two chunks; retain lightweight native loss/time/TPS/errors |
| Batch | Microbatch 2, sequence length 4,096, global batch 512, gradient accumulation 32 |
| Scheduler | 65,000-update horizon, 2,000-update LR warmup; short performance stop does not compress it |
| Performance window | 10 warmup + 10 measured optimizer updates; extend only for actual compilation/runtime settling in the window |
| Disabled performance features | Validation, checkpoint saves, memory snapshots, numerical capture, profiling and CUDA Graphs |

Use the agreed microbatch and accumulation values. Do not copy the development preset's local batch 64 or its prior PP microbatch 2 setting into this EP8/PP1 baseline. Confirm the effective mesh and batch semantics from the rank-zero resolved configuration before treating the configuration as verified.

## Selective activation checkpointing

Keep the layer policy explicit:

| Trunk layer | Policy |
| --- | --- |
| 0–1 | Recompute |
| 2–4, 6–7 | Save `experts_up` |
| 5 | Save `experts_up` and `comm` |

Do not replace this policy with full activation checkpointing or silently alter individual layer choices to make the smoke fit.

## Token-drop policy

Preserve the production token-drop policy at a fixed capacity of 50,000 and leave `group_gate` unchanged. The canonical config expresses this as `strategy: fixed_capacity`, `expert_capacity_tokens: 50000`, and `drop_by_group_capacity: true`. With the production EP32 layout, the source quota is 1,562; at EP8 it becomes 6,250. This topology change motivates a later pressure study, but does not authorize changing the baseline capacity. A 12,500-cap run is a future diagnostic proxy, not the default.

For context, the expected balanced expert GEMM `M` is 16,384 in the production EP32 setup and 4,096 in this EP8 crop at `B=2, S=4096`; local experts per rank change from 4 to 16, while `K/N` remain the same. Treat this as a workload-shape comparison, not evidence that the crop reproduces production routing pressure.

## Memory estimate and fallback

The planning calculation is not a measurement: it records `P8=19,006,114,432` and estimates about 56.8 GiB per GPU for model state and GFSDP buffers, before activations, compiler workspaces, NCCL, and FP8-related overhead. This estimate already includes about 8.85 GiB per GPU for EMA. A 12-layer crop with EMA is estimated above 80 GiB per GPU.

The completed smoke's maximum native console memory metric was 72.61 GiB across
ranks/steps. That metric combines reserved peak and report-time non-PyTorch
memory; it is not a simultaneous device-wide maximum or a bound for other data
and routing distributions.

If the eight-layer crop cannot run, a four-layer fallback requires an explicit, recorded selection. Do not shrink the model silently or present a fallback run as the agreed eight-layer baseline.

## Topology and transfer limits

The hypothetical production target is 512 GPUs with PP4 and EP32, with dense FSDP128 and expert FSDP4 when TP, CP and DP replication are one. The single-node, eight-GPU crop uses dense FSDP8 and expert FSDP1. It does not exercise production PP, inter-node communication, or expert-weight collectives, so smoke results cannot establish those behaviors or their performance.

## Smoke and runtime status

### Selected production tokenizer and real-data inputs

The canonical tokenizer is
`/home/jovyan/shares/SR008.nfs1/models_templates/dclm-7b-128k_digit_tokenizer_FIM`,
the execution-host location of the named production 100B asset. The supplied
donor reference is `ec47fb8e40a9a1e55d3c18dfa704ca104ae773af` and historical asset
spelling is `/mnt/nfs/07-share/models_templates/dclm-7b-128k_digit_tokenizer_FIM`.
Current tokenizer file identities and actual loaded behavior belong to the
[new campaign](../runs/production-data-abcd/report.md#execution-route-and-input-discovery).
The completed [data receipt](../runs/production-data-abcd/data-002.json) records
the hashes of all three tokenizer files, vocabulary and special-token IDs, and
the native wrapper behavior.
Do not substitute the bundled tokenizer or another 128k tokenizer.

The `streaming_text` input targets 102400 distinct full 4096-token uint32 rows,
prepared from existing GigaSets 5.3.1 English reference text, Russian web,
mathematics and Python code. Each domain contributes 25600 rows in a deterministic
row round-robin stream. This is an identified diverse comparison mixture, not a
claim of the production mixture. Canonical local path is
`../runs/production-data-abcd/artifacts/raw/data-002` relative to `gift/`, with
split `train`; the runner fragment selects the same resolved path.

Keep two persistent workers, no pinning, null prefetch factor, shuffle/FIM off,
one canonical node, original partitioning, drop-last and infinite iteration.
Logical epoch and physical row count are both 102400, avoiding oversampling.
Documents are encoded with the native wrapper's BOS/EOS behavior, concatenated
and split into nonoverlapping full rows. Main labels shift within each row,
ignore its final position and retain 4095 valid targets. The native trainer uses
causal per-row positions without EOS document isolation; loader EOS-reset
positions are not the effective model positions for this crop.

Preparation and CPU/Gloo actual-loader acceptance are complete. The
[loader receipt](../runs/production-data-abcd/data-002-loader-summary.json)
passes both prefetch depths (0 and 32) on all eight ranks, with 672 batches per
rank, identical opening order, and exact loaded inputs, labels and loader
positions. Across the 21-update census, each arm covers 10752 unique global
rows and 44029440 valid main targets; the receipt owns bounds, per-rank
uniqueness and owner-skew details. The base runner's actual
`_verify_ready_inputs(TASK_ROOT)` check passes, as recorded in
[input-readiness.json](../runs/production-data-abcd/input-readiness.json).
These receipts establish CPU/Gloo input acceptance, not GPU training behavior.
Native component acquisition and all four serial A/B/C/D
timing runs are complete; their single fixed-order performance ratios remain
provisional. All eight timing/diagnostic rows completed without failure; the
source/config guard passed, and the final check found the GPUs idle. MCP Perfetto analysis and
report integration are complete. The [campaign report](../runs/production-data-abcd/report.md)
owns its results, parser limitations and execution identities.
`data-001` is a failed partial and remains excluded. Historical fixture smokes
and performance values do not establish acceptance for these new inputs.

<a id="selected-mds-fixture"></a>
### Historical MDS fixture

The previous canonical loader was `streaming_text` over 896 full 4096-token uint32 rows
prepared from the bundled C4 corpus and tokenizer. It uses two persistent workers,
no pinning, null prefetch factor, shuffle/FIM off, one canonical node, original
partitioning, drop-last and infinite iteration. The canonical relative path
`../runs/overnight-streaming-perf/artifacts/raw/mds4096` assumes the normal
`gift/` training working directory. The [data receipt](../runs/overnight-streaming-perf/mds-data.json)
records the callable packing recipe, identities, hashes and all-rank loader checks.
This repeated low-diversity fixture and its 2009-ID tokenizer are excluded from
the current rerun. Its results cannot serve as controls for new-data candidates.

CPU materialized-batch prefetch retains the loader API and canonical control
default `num_prefetch_batches: 0`. On 13 November 2026 the owner accepted the
[V2 variant](../runs/v2-prefetch-baseline/report.md#mcp-perfetto-comparison-and-acceptance)
with the explicit override `num_prefetch_batches: 32` for the current long-epoch
workload, based on the TPS pair and MCP Perfetto comparison. Consumer-owned epoch
entry and accumulation 32 remain part of the contract. The acceptance record
does not change source or canonical configuration bytes.

The [native MDS receipt](../runs/overnight-streaming-perf/g1-smoke-002-receipt.json)
passed the ten-update/monitor5 smoke on all eight ranks. That run used the
equivalent external loader fragment; the subsequent canonical YAML promotion
passed its decode/golden test, non-data semantic comparison and path/shard-hash
equivalence check. These new YAML bytes were not independently GPU-rerun.
All model, precision, parallelism, SAC, optimizer and scheduler settings remain
unchanged. Full rows provide 4095 valid main targets per sample; the trainer
uses causal per-row positions without EOS document isolation. HF packing differs,
so this is a new identified data baseline, without an HF/MDS parity claim.

### Historical HF smoke

The authorized basic smoke passed on eight NV18-connected H100s using PyTorch
2.15 development with CUDA 13.0, CUTLASS DSL 4.4.1 and TVM FFI 0.1.11. The launcher
initializes the existing image CUDA compatibility profile. That config used bundled
`c4_test` data and the repository tokenizer as smoke inputs, not as a production
data selection. The [smoke report](../runs/100b-crop8-ep8-smoke/report.md) records the
verified runtime and successful execution.

The executed smoke completed 10 optimizer updates with 32 microbatches per update,
preserving the 65,000-step scheduler horizon. Its validation-only monitoring
interval was 5 instead of the default 50. Both ordinary and monitoring paths
completed; the 2/32 loss-chunk selection is supported by resolved settings and
native source/hook correlation, not a direct chunk counter. No checkpoint,
profiler trace or memory snapshot was emitted, and all training processes released
the GPUs. Those historical monitoring settings do not apply to future performance runs:
detailed callbacks/hooks are disabled there, with ordinary two-chunk loss. No
performance benchmark or convergence study was performed in that HF smoke.
The original MDS performance campaign is cancelled/incomplete; its unfinished
gates are not smoke acceptance claims.

The baseline uses the installed DeepGEMM `2.3.0+f093a4c` and DeepEP
`1.2.1+1b8f467`, with the native `torchtitan.train` entrypoint. The launcher verifies
package paths and source hashes. See the [result receipt](../runs/100b-crop8-ep8-smoke/result.json).

## Historical performance boundary

The monitoring-inclusive G2 median input TPS was 124454.81175967702 global and
15556.851469959627 per GPU. It is historical and not directly comparable with
future monitoring-off measurements. A new matched short baseline/candidate pair
is required for a future authorized comparison; single-pair results remain
provisional. Performance precedes local assessment of a candidate worth retaining.
The active [work plan](../planning/README.md) owns current execution authorization.

## Future work

The editable [work plan](../planning/README.md) owns experiment priorities and
status. [Candidates](candidates.md) owns mechanisms and source pointers;
[validation](../planning/validation.md) defines how to test and interpret results.
