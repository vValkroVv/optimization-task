# Agreed 100B-dev crop baseline

**State:** This is the accepted default workload contract. Its canonical task-local YAML is [gift_100b_crop8_ep8_fp8_compile_cute.yaml](../gift/configs/experiments/gift_100b_crop8_ep8_fp8_compile_cute.yaml), with the model preset [100b_crop8.yaml](../gift/configs/presets/models/gift_moe/gift_3_5/100b_crop8.yaml). The full-length MDS configuration is now selected after its eight-H100 smoke passed. Every rank completed ten updates and 320 microbatches, including monitoring. The [campaign report](../runs/overnight-streaming-perf/report.md) owns MDS evidence and performance gates; the [historical HF smoke](../runs/100b-crop8-ep8-smoke/report.md) remains unchanged.

## Identity and scope

- Architecture reference: 100B-dev source revision `f15d33f3fd9204e5fef2adbd625b3caed2241052`.
- CuTe/runtime donor reference: `cute_norm_prodrun` at `ec47fb8e40a9a1e55d3c18dfa704ca104ae773af`, a descendant of the 100B-dev revision.
- Training source base: task-local GIFT `main` at `6fd9a3230ab6361f11bbbf0b6ec5a5f125b78fc2`, recorded in the [source provenance](../references/provenance.json). The selected implementation route is to port only the required 100B-dev mechanisms into this base; it does not replace the base with the donor branch.
- The ports include per-layer SAC, adaptive routing, development packed initialization, and loss-monitoring support. A functional grouped-FP8 output also fixes the donor's eager SAC result-caching defect; the smoke report records its regressions and source review.
- This is an eight-trunk-layer crop of the referenced model. It is a smoke and development baseline, not a production-scale run or a measured throughput baseline.
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
| Loss and monitoring | Two ordinary loss chunks, 32 on monitoring steps; monitoring interval 50 |
| Batch | Microbatch 2, sequence length 4,096, global batch 512, gradient accumulation 32 |
| Scheduler | 65,000-step horizon |

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

### Selected MDS fixture

The canonical loader is `streaming_text` over 896 full 4096-token uint32 rows
prepared from the bundled C4 corpus and tokenizer. It uses two persistent workers,
no pinning, null prefetch factor, shuffle/FIM off, one canonical node, original
partitioning, drop-last and infinite iteration. The canonical relative path
`../runs/overnight-streaming-perf/artifacts/raw/mds4096` assumes the normal
`gift/` training working directory. The [data receipt](../runs/overnight-streaming-perf/mds-data.json)
records the callable packing recipe, identities, hashes and all-rank loader checks.
This repeated local fixture supports a controlled performance study, not a
production data mixture or quality evaluation.

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
the GPUs. The canonical interval remains 50. No full performance benchmark,
convergence study was performed in that historical smoke. The authorized MDS
performance campaign is tracked separately; its unfinished gates are not smoke
acceptance claims.

The baseline uses the installed DeepGEMM `2.3.0+f093a4c` and DeepEP
`1.2.1+1b8f467`, with the native `torchtitan.train` entrypoint. The launcher verifies
package paths and source hashes. See the [result receipt](../runs/100b-crop8-ep8-smoke/result.json).

## Future work

The editable [work plan](../planning/README.md) owns experiment priorities and
status. [Candidates](candidates.md) owns mechanisms and source pointers;
[validation](../planning/validation.md) defines how to test and interpret results.
