# Four kernel experiments on the fixed Hopper training recipe

This follow-up first used the eight-GPU [production recipe](../../docs/baseline.md#current-final-task-baseline) with all four backward fixes from `ab62730a9`. The CuTe candidate ran before promotion. The canonical launcher then enabled it by default, and the full production recipe was compared with the original A baseline. The component and single-change runs preserve S, the GPU capacity selector, the five Hopper fusions, folded CE, the three expert GFSDP switches, EP8/eFSDP1, dense FSDP8, accumulation 32, compile, SAC and the real training data. HybridEP dispatch and combine both use 16 SMs.

## Original A versus promoted production

Each protocol ran production first and A second on the same eight H100 80 GB GPUs on `sr008-gpu1`, with the same `data-002` input and seed 2048. A used `e9b7cdbb4`, with native FLA, original routing and original CE. Production used `2c67c182e`, with the canonical switches, including CuTe statistics-only SAC and SwiGLU row16 disabled. Both ran the eight-layer crop with global batch 512, sequence length 4,096 and accumulation 32. The timer covers complete measured updates, with 20,971,520 physical tokens for updates 11–20 and 209,715,200 for updates 41–140. Memory peaks are the maximum across all eight ranks. The [comparison receipt](results/original-a-vs-production.json) preserves both raw receipt locations and hashes, configuration and source identities, and per-rank checks.

| Protocol | A tokens/s | Production tokens/s | TPS change | Peak allocated A → production, GiB | Peak reserved A → production, GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10+10 | 154,918.74 | 180,325.21 | +16.40% | 69.149 → 60.309 | 69.732 → 62.186 |
| 40+100 | 148,462.22 | 172,685.52 | +16.32% | 68.545 → 61.390 | 69.805 → 64.041 |

In the longer pair, mean update time fell from 14.126 to 12.144 seconds and peaks fell by 7.155 GiB allocated and 5.764 GiB reserved. Both arms passed, reused their own compiler cache, and had zero allocator retries or OOMs. The compiler observer was stable in its covered scope. In 10+10, A used a fresh cache and production reused its cache. An earlier A launch stopped during loader startup because its Unix socket path was too long and supplied no timing samples. Both pairs ran in fixed order. Their gains describe the full recipe against A on this topology.

### Depth scaling estimate

For a 40-layer model with the same repeated GDN/MLA pattern, token shapes, microbatch count, and per-layer execution costs, the planning estimate is **about +15% TPS versus A**, with a sensitivity range of **+14–16%**. The estimates below use separate timings and trace scopes to allocate the 1.981 seconds saved on the measured eight-layer step.

| Depth-independent change | Estimated saving per eight-layer update, seconds | Evidence |
| --- | ---: | --- |
| CPU prefetch | 0.20–0.32 | The lower end uses observed host loader waits on current inputs. The upper end uses a separately timed historical pair. [Current trace](../production-data-abcd/trace-analysis.md#ba-the-combined-four-changes) · [Prefetch timing](../v2-prefetch-baseline/report.md#matched-tps-results) |
| D0 and S | 0.24 | The embedding, first FFN and MTP boundary occur once per microbatch. [C through S timings](../schedule-embedding/timing-results-140.md) · [Schedule scope](../schedule-embedding/report.md#selected-approach-and-preimplementation-cost-review) |
| Tail retention | 0.07–0.11 | It avoids gathers at the end of the decoder and around MTP. [Timing receipts](../efsdp-optimizations/tail-receipts.json) · [Tail mechanism](../../docs/profile-my-ideas.md#keep-tail-weights-live-across-chunked-lm-head-backward) |
| Stable CE, folded CE, staged head dW, direct dHidden and hidden z-loss | About 0.23 | The same output heads and loss scopes execute at the same shapes. [Stable CE trace](../four-fusion/ce-trace-analysis.md#complete-headloss-guard-and-downstream-work) · [Folded CE timing](../backward-components/report.md#native-1010-comparison) · [Backward components](../fusion-pack/report.md#profile-budget-and-candidate-selection) |

The total depth-independent saving is approximately 0.73–0.91 seconds. In the central scenario, 0.82 seconds stay fixed and the remaining 1.16 seconds grow by 5.56×, based on the measured GDN/MLA mix as the MoE count grows from 7 to 39. The model uses about 4.2 seconds of fixed production work, estimated from the first dense layer, embeddings, MTP, the heads and final norm in the historical trace, adjusted for the later fixes. Thus `production_40 = 4.2 + 5.56 × (12.144 − 4.2) ≈ 48.37 s` and `A_40 = production_40 + 0.82 + 5.56 × (1.981 − 0.82) ≈ 55.65 s`. Their ratio gives about +15%. The sensitivity range varies fixed production time between 4.1 and 4.4 seconds and the fixed savings across the table's range. These are modeled times and combine evidence from different timing scopes. [Layer timings](../compile-baseline/matched-blocks.md#full-layer-coverage) · [Layer pattern](../../gift/configs/presets/models/gift_moe/gift_3_5/100b_crop8.yaml)

If only direct expert dW is unavailable, removing its isolated 0.21-second crop saving before scaling puts the same model at about +12% TPS. The current direct dW path requires a singleton reduce-scatter group and the Hopper layout with 16 local experts. FP8 expert cache and tail retention need separate timing on a multi-rank expert mesh, where collective traffic and overlap change. [Standalone GFSDP receipts](../efsdp-optimizations/four-standalone-receipts.json) · [Direct dW guard](../../gift/torchtitan/distributed/gfsdp/_param_group.py#L899) · [Topology follow-up](../../planning/README.md#follow-up-ep-and-gfsdp-topology-audit)

## Native 10+10 results

Each arm of this eight-rank 10+10 comparison ran once on the same H100s on `sr008-gpu1`, in table order, at revision `42a4ae25a253b865912512b60f502b74f36cfdd5`. The timer covered 20,971,520 global physical input tokens across updates 11–20. The resolved training configurations had identical normalized SHA-256 `5f88935814db6d86660a41eee655f067eff5ceec2a9c73c8506dbba2cce39b0d`. Compilation was stable during measurement, with zero allocator retries or OOMs on every rank. The experiment and the four existing fixes produced execution receipts on each applicable rank. [Timing and all 20 update scalars](results/native.json) · [Per-rank kernel and allocator receipts](results/native-ranks.json)

| Single change from the four-fix control | Physical tokens/s | Change | Median update, s | Peak allocated, GiB | Peak reserved, GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Four-fix control (historical, before CuTe promotion) | 179,761.44 | — | 11.6514 | 60.3085 | 62.1855 |
| CuTe statistics-only SAC replay (candidate, later promoted) | 180,393.28 | +0.351% | 11.6192 | 60.3078 | 62.2051 |
| SwiGLU row16, 16×128/four warps (experimental, off) | 179,990.03 | +0.127% | 11.6525 | 60.3073 | 62.1191 |

The CuTe candidate measured +631.84 tokens/s (+0.351%) against the four-fix control. Peak allocated memory fell 0.0007 GiB (0.70 MiB), while reserved peak rose 0.0195 GiB (20 MiB). The owner selected this change for the canonical production launch. This result does not establish a repeatable throughput gain. SwiGLU row16 used 1.22 MiB less peak allocated and 68 MiB less reserved. Its median update was 1.1 ms longer despite higher whole-window TPS, so the small throughput difference remains uncertain; it stays experimental and disabled. Their combined performance and memory have not been measured.

Across all 20 updates, the largest absolute differences from the control in total loss and global gradient norm were 0.0004282 and 0.01553 for CuTe, and 0.0002499 and 0.01866 for SwiGLU. The fixed-input component checks establish bitwise parity within their documented scope. These short runs do not establish convergence.

From the synchronized checkout, `bash runs/fusion-followup/timing.sh NEW_TAG cute-replay-stats swiglu-row16` reproduces this historical comparison on the reserved single 8×H100 node. The canonical production launcher selects only CuTe stats-only SAC; SwiGLU row16 remains off. Run receipts are in `runs/fusion-followup/artifacts/raw/followup-native-001`, with full logs and TensorBoard events under `runs/kernel-fusion/artifacts/raw/followup-native-001-<arm>`.

## Component results

| Change | Control → candidate, H100 | Precision and memory | Decision for training |
| --- | --- | --- | --- |
| cuBLASLt for the staged LM head | Eager medians for two identical controls 48.038 and 48.055 ms; best confirmed pitched dW tile 47.256 ms. Paired saving 0.620 ms, control variation 0.636 ms. dX also finished within control variation. | Both BF16 chunks and the merged FP32 gradient matched. Peak allocated and reserved 7,690,256,384 bytes in each arm. | Retain native `torch.mm`. [Sweep, confirmation and traces](head-lt.md) |
| CuTe gated norm, statistics-only SAC replay | Complete compiled replay 364.8 → 303.9 µs plain and 409.4 → 331.1 µs with residual. Trace-to-wrapper matching found 9 eligible plain and 24 eligible residual calls in three native microbatches. | Six prepared inputs/statistics and tested gradients matched bitwise. Full component peaks were unchanged. Plain replay used its matched upstream tangent; residual replay used a same-shape tangent from another native call. | Selected for production after the one-shot native 10+10. [Liveness, parity and replay](cute-sac.md) |
| GDN short-conv dQKV → FP8 | Final exact producer through projection backward 1.778 → 1.850 ms. | Exact FP8 payload, scales, conv dW and projection dX/dW. Eager peak allocated/reserved fell 64/44 MiB. | Keep separate production kernels. [Exact patch and traces](gdn-qkv-fp8.md) |
| Staged expert SwiGLU row16 | Four warps: producer 342.1 → 336.8 µs. Through the native grouped dgrad, control 1.764 / 1.778 ms and candidate 1.757 / 1.791 ms in forward/reverse arm order. | Four warps matched FP8 bytes, FP32 scales, routing dP and downstream BF16 dgrad on valid rows. Identical allocated/reserved peaks, 1,551,876,096 / 1,566,572,544 bytes. Eight warps changed routing dP. | Remains experimental and disabled in production. [Per-arm traces and timing](results/swiglu.json) |

The SwiGLU fixture contains 63,616 rows of 3,072 BF16 inputs, real routing probabilities and gradients, grouped weights/scales, expert indices and the native dgrad output. The replay reproduced the captured payload, scales and valid dgrad rows before the sweep. Four warps reduced registers per thread from 128 to 95. The two-warp variant kept 128 and slowed the consumer chain. The two DeepGEMM cache-preflight attempts stopped before measurement; the recorded run used a task-local cache with compilation outside timing. The tuning keeps the 128-channel FP8 quantum and changes only the number of independent rows scheduled in one program. [SwiGLU harness](swiglu_component.py) · [Triton discussion of registers, occupancy and warp count](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)

Raw component traces and logs are under `runs/fusion-followup/artifacts/raw` on `sr008-gpu1`. The three newly captured fixtures come from rank 0 of `runs/kernel-fusion/artifacts/raw/followup-fixtures-001`, a native eight-rank update. LM-head replay reuses the earlier captured inputs documented in the sweep. Each component timing covers its stated boundary and does not predict end-to-end TPS.
