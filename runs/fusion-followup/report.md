# Four kernel experiments on the fixed Hopper training recipe

The control is the eight-GPU [production recipe](../../docs/baseline.md#current-final-task-baseline) with all four backward fixes from `ab62730a9`. The experiments preserve S, the GPU capacity selector, the five Hopper fusions, folded CE, the three expert GFSDP switches, EP8/eFSDP1, dense FSDP8, accumulation 32, compile, SAC and the real training data. HybridEP dispatch and combine both use 16 SMs.

## Native 10+10 results

Each arm ran once on the same eight H100s on `sr008-gpu1`, in table order, at revision `42a4ae25a253b865912512b60f502b74f36cfdd5`. The timer covered 20,971,520 global physical input tokens across updates 11–20. The resolved training configurations had identical normalized SHA-256 `5f88935814db6d86660a41eee655f067eff5ceec2a9c73c8506dbba2cce39b0d`. Compilation was stable during measurement, with zero allocator retries or OOMs on every rank. The experiment and the four existing fixes produced execution receipts on each applicable rank. [Timing and all 20 update scalars](results/native.json) · [Per-rank kernel and allocator receipts](results/native-ranks.json)

| Single change from the four-fix control | Physical tokens/s | Change | Median update, s | Peak allocated, GiB | Peak reserved, GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current production | 179,761.44 | — | 11.6514 | 60.3085 | 62.1855 |
| CuTe statistics-only SAC replay | 180,393.28 | +0.351% | 11.6192 | 60.3078 | 62.2051 |
| Expert SwiGLU 16×128, four warps | 179,990.03 | +0.127% | 11.6525 | 60.3073 | 62.1191 |

CuTe had the larger throughput difference. Its peak live allocation fell 0.70 MiB, while the allocator reserved 20 MiB more at peak. The single window does not establish a memory-neutral production improvement under a strict limit on reserved memory. SwiGLU used 1.22 MiB less peak allocated and 68 MiB less reserved. Its median update was 1.1 ms longer despite higher whole-window TPS, so the small throughput gain remains uncertain. Both switches are opt-in. Their combined performance and memory have not been measured.

Across all 20 updates, the largest absolute differences from the control in total loss and global gradient norm were 0.0004282 and 0.01553 for CuTe, and 0.0002499 and 0.01866 for SwiGLU. The fixed-input component checks establish bitwise parity within their documented scope. These short runs do not establish convergence or the repeatability of either throughput difference.

From the synchronized checkout, `bash runs/fusion-followup/timing.sh NEW_TAG cute-replay-stats swiglu-row16` reproduces the campaign on the reserved single 8×H100 node. Run receipts are in `runs/fusion-followup/artifacts/raw/followup-native-001`, with full logs and TensorBoard events under `runs/kernel-fusion/artifacts/raw/followup-native-001-<arm>`.

## Component results

| Change | Control → candidate, H100 | Precision and memory | Decision for training |
| --- | --- | --- | --- |
| cuBLASLt for the staged LM head | Eager medians for two identical controls 48.038 and 48.055 ms; best confirmed pitched dW tile 47.256 ms. Paired saving 0.620 ms, control variation 0.636 ms. dX also finished within control variation. | Both BF16 chunks and the merged FP32 gradient matched. Peak allocated and reserved 7,690,256,384 bytes in each arm. | Retain native `torch.mm`. [Sweep, confirmation and traces](head-lt.md) |
| CuTe gated norm, statistics-only SAC replay | Complete compiled replay 364.8 → 303.9 µs plain and 409.4 → 331.1 µs with residual. Trace-to-wrapper matching found 9 eligible plain and 24 eligible residual calls in three native microbatches. | Six prepared inputs/statistics and tested gradients matched bitwise. Full component peaks were unchanged. Plain replay used its matched upstream tangent; residual replay used a same-shape tangent from another native call. | Advance the graph rewrite to native 10+10. [Liveness, parity and replay](cute-sac.md) |
| GDN short-conv dQKV → FP8 | Final exact producer through projection backward 1.778 → 1.850 ms. | Exact FP8 payload, scales, conv dW and projection dX/dW. Eager peak allocated/reserved fell 64/44 MiB. | Keep separate production kernels. [Exact patch and traces](gdn-qkv-fp8.md) |
| Staged expert SwiGLU 16×128 | Four warps: producer 342.1 → 336.8 µs. Through the native grouped dgrad, control 1.764 / 1.778 ms and candidate 1.757 / 1.791 ms in forward/reverse arm order. | Four warps matched FP8 bytes, FP32 scales, routing dP and downstream BF16 dgrad on valid rows. Identical allocated/reserved peaks, 1,551,876,096 / 1,566,572,544 bytes. Eight warps changed routing dP. | Advance only four warps to native 10+10. [Per-arm traces and timing](results/swiglu.json) |

The SwiGLU fixture contains 63,616 rows of 3,072 BF16 inputs, real routing probabilities and gradients, grouped weights/scales, expert indices and the native dgrad output. The replay reproduced the captured payload, scales and valid dgrad rows before the sweep. Four warps reduced registers per thread from 128 to 95. The two-warp variant kept 128 and slowed the consumer chain. The two DeepGEMM cache-preflight attempts stopped before measurement; the recorded run used a task-local cache with compilation outside timing. The tuning keeps the 128-channel FP8 quantum and changes only the number of independent rows scheduled in one program. [SwiGLU harness](swiglu_component.py) · [Triton discussion of registers, occupancy and warp count](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)

Raw component traces and logs are under `runs/fusion-followup/artifacts/raw` on `sr008-gpu1`. The three newly captured fixtures come from rank 0 of `runs/kernel-fusion/artifacts/raw/followup-fixtures-001`, a native eight-rank update. LM-head replay reuses the earlier captured inputs documented in the sweep. Each component timing covers its stated boundary and does not predict end-to-end TPS.
