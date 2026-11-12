# Agreed 100B-dev crop baseline

**State:** This is the accepted default workload contract. Its canonical task-local YAML is [gift_100b_crop8_ep8_fp8_compile_cute.yaml](../gift/configs/experiments/gift_100b_crop8_ep8_fp8_compile_cute.yaml), with the model preset [100b_crop8.yaml](../gift/configs/presets/models/gift_moe/gift_3_5/100b_crop8.yaml). The required runtime ports, torch-free configuration checks and independent source review are complete; the GPU smoke remains pending. See the [smoke report](../runs/100b-crop8-ep8-smoke/report.md) for executed checks and their limits.

## Identity and scope

- Architecture reference: 100B-dev source revision `f15d33f3fd9204e5fef2adbd625b3caed2241052`.
- CuTe/runtime donor reference: `cute_norm_prodrun` at `ec47fb8e40a9a1e55d3c18dfa704ca104ae773af`, a descendant of the 100B-dev revision.
- Training source base: task-local GIFT `main` at `6fd9a3230ab6361f11bbbf0b6ec5a5f125b78fc2`, recorded in the [source refresh receipt](../additional/baseline/gift-main-2026-11-11.md). The selected implementation route is to port only the required 100B-dev mechanisms into this base; it does not replace the base with the donor branch.
- The ports include per-layer SAC, adaptive routing, development packed initialization, and loss-monitoring support. A functional grouped-FP8 output also fixes the donor's eager SAC result-caching defect; the smoke report records its regressions and source review.
- This is an eight-trunk-layer crop of the referenced model. It is a smoke and development baseline, not a production-scale run or a measured throughput baseline.
- Keep the architecture, donor, and trainer-base identities distinct. The current task priority is the agreed crop, not the historical 3B CUDA Graph stability campaign in [OPT-004](../backlog/tasks/opt-004-cuda-graphs-hybridep-fp8.md). Preserve that campaign record and its evidence without extending it.

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

If the eight-layer crop cannot run, a four-layer fallback requires an explicit, recorded selection. Do not shrink the model silently or present a fallback run as the agreed eight-layer baseline.

## Topology and transfer limits

The hypothetical production target is 512 GPUs with PP4 and EP32, with dense FSDP128 and expert FSDP4 when TP, CP and DP replication are one. The single-node, eight-GPU crop uses dense FSDP8 and expert FSDP1. It does not exercise production PP, inter-node communication, or expert-weight collectives, so smoke results cannot establish those behaviors or their performance.

## Smoke and runtime status

The owner authorized basic GPU training verification only. The eight idle H100s with NV18 connectivity passed availability preflight, but training has not launched. The observed preflight runtime is PyTorch 2.15 development with CUDA 13.0, CUTLASS DSL 4.4.1, and TVM FFI 0.1.11; verify the runtime again for the actual smoke. The config uses bundled `c4_test` data and the repository tokenizer as smoke inputs, not as a production data selection. The [smoke report](../runs/100b-crop8-ep8-smoke/report.md) records execution evidence and its limits.

The bounded smoke plan is 10 optimizer updates with 32 microbatches per update. Preserve the configured 65,000-step scheduler horizon. Use a validation-only monitoring interval of 5 rather than the baseline interval of 50 so the smoke exercises both the two normal loss chunks and the 32-chunk monitoring path without running 50 updates. Keep this override local to the smoke. Disable checkpoint saving, profiling, and memory snapshots. The smoke must not create checkpoints or traces. No full performance benchmark or optimization campaign is authorized in this scope. Record the resolved rank-zero configuration, active feature markers, basic training result, and any failure in the task evidence; until then, these remain pending and this document makes no runtime-success claim.

The smoke reuses the already available isolated DeepGEMM descriptor repair and pinned HybridEP scan correction on the inspected runtime. The repaired header hash was checked during preflight. Shared packages remain unchanged; the crop's actual runtime behavior still requires its own smoke result.

## Finite future experiments

These are research candidates for later authorization, not current work:

1. Measure the eight-layer, 128-expert default with repeated unprofiled controls before making performance claims; production-scale claims require a matching production check.
2. Run a separate 32-expert/EP8/microbatch-2 diagnostic: four experts per rank and expected balanced `M=16384`. This changes the model and is a shape diagnostic only.
3. Run a 128-expert/EP8/microbatch-8 diagnostic only with a separate memory and compute review; it has higher memory pressure and four times the per-layer token work of the default.
4. Test token-drop capacity 12,500 as a diagnostic proxy while retaining the 50,000 production default until evidence and a separate decision support a change.
5. Revisit CUDA Graphs only after the baseline is established. No graph implementation or comparison is part of the current scope.
