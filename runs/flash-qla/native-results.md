# FlashQLA native integrated training results

All three authorized native runs completed and passed artifact validation. The results below describe one fixed-order sequence (FLA → external gate → fused gate), not a demonstrated general speedup. No additional runs were performed for this analysis.

## Recipe and evidence

- Frozen active worktree: `/workspace-SR008.nfs2/vkropotin/opt-fq`; source HEAD `0f32623d5e0e7544b20c1f047d94236cb6697758`. Full source/config identities, launch argv, runtime, activation records and step series are retained in [native-results.json](native-results.json). The eight plan source hashes match unchanged files after the sequence. Resolved configuration hashes match [native-config-comparison.json](native-config-comparison.json); configurations are identical after normalizing only run output paths. See [native-plan.json](native-plan.json) and [analysis-preparation.md](analysis-preparation.md).
- One node, eight NVIDIA H100 80GB HBM3 GPUs; PyTorch `2.15.0.dev20260822+cu130`, CUDA13.0, NCCL2.30.7, Triton `3.8.0+git3f6e4113`. FLA upstream `1d2281ff1b7988d8070f00dba5ffa9747ae3a60e`; FlashQLA upstream `a14dbf3474514e1674526431e960325f72081e9f`. Exact package details are in JSON runtime receipts.
- GIFT `gift_3_5/100b_crop8`: eight main layers plus MTP; model dimension4096; native GDN local microbatch2 × sequence4096, key heads16/value heads32, key/value dimensions128. Global batch512, accumulation32, GFSDP2 shard8, EP8, TP/PP/CP1. Model+loss compilation, selective activation checkpointing, FP8 dense/sparse paths and BF16 expert-weight reuse remain configured. Master parameters/reductions are FP32, mixed-precision parameters BF16. Actual gathered gate leaves A_log/dt_bias are BF16; FP32 masters/intermediate arithmetic do not make those leaves FP32.
- Each run starts fresh with seed2048; deterministic mode is false. The same local MDS `c4_perf896` stream, sequence4096, shuffle disabled, epoch_size102400, prefetch32 and original optimizer/scheduler recipe are used. Scheduler warmup2000/total65000 is distinct from measurement warmup. Common completed untimed capture caches are reused; TileLang caches remain attempt/rank local. Cache warmth and asynchronous compilation costs are not independently quantified.
- All140 optimizer updates run; updates1–40 are excluded and41–140 form the predeclared100-update window. Checkpointing, validation, profiler and memory snapshots are disabled. EMA is configured with start_step10000 and is not exercised here. FLA retains the combined-four foundation, including TileLang `common.chunk_bwd_dqkwg`; it is not an all-Triton control.

## Integrated latency and throughput

The primary timer is host monotonic elapsed with CUDA-drained boundaries after updates40 and140: max(rank end) − min(rank start), valid for this single host. It includes training, host submission, data loading, optimizer, logging and window instrumentation. Start-side allocator/token reads follow the start timestamp; end-side reads follow the end timestamp. It is neither a sum of kernel durations nor isolated device compute.

Each rank processes26,214,400 physical tokens in the window; summing eight ranks gives209,715,200. The already-global supervised count209,664,000 is counted once (2,096,640/update versus2,097,152 physical/update). Physical and supervised TPS divide their own count by the same envelope seconds. Per-GPU figures divide global TPS by8; MTP does not multiply the input-token denominator.

| Run | Envelope s | Physical global TPS | Physical TPS/GPU | Supervised global TPS | Supervised TPS/GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| fqla-native-fla-001 | 1340.441264351 | 156452.36 | 19556.55 | 156414.16 | 19551.77 |
| fqla-native-external-001 | 1309.977240186 | 160090.72 | 20011.34 | 160051.64 | 20006.45 |
| fqla-native-fused-001 | 1312.892723662 | 159735.21 | 19966.90 | 159696.22 | 19962.03 |

- external_gate: observed physical throughput +2.326% relative to FLA (FLA TPS denominator); elapsed -2.273% relative to FLA (FLA seconds denominator).
- fused_gate: observed physical throughput +2.098% relative to FLA (FLA TPS denominator); elapsed -2.055% relative to FLA (FLA seconds denominator).
- Fused versus external: physical throughput -0.222%, elapsed +0.223%, using external as denominator. These are ratios of single integrated observations, not paired statistics.

## Cadence, tails and drift

Cadence is rank-local host time between native Profiler.step boundaries; there is no per-update adapter drain, and the last interval includes the final drain. All eight rank series are preserved in JSON. Rank0 summaries below describe the entire declared window, not a proven stationary window. Native displayed TPS is a separate rounded per-GPU logging-interval metric and must not replace integrated throughput.

| Variant | Median s | p10–p90 s | Min–max s | MAD s | First20 median s | Last20 median s | Last/first −1 | Displayed TPS/GPU median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fla | 13.178094 | 12.865413–14.317043 | 12.822396–15.043007 | 0.293255 | 14.357223 | 12.864839 | -10.395% | 19891.5 |
| external_gate | 12.847254 | 12.591889–13.985210 | 12.546298–14.705261 | 0.240361 | 14.041859 | 12.587410 | -10.358% | 20404.5 |
| fused_gate | 12.908920 | 12.603895–14.051485 | 12.541914–14.812290 | 0.291434 | 14.089885 | 12.604501 | -10.542% | 20306.5 |

All three windows continue to shorten by about10.4–10.5% between their first and last20 intervals. This drift exceeds the roughly2% between-run integrated difference and confounds an interpretation as a stationary backend effect; its cause is not attributed here. Last20 intervals (updates121–140) are descriptive tails selected for a fixed summary, not replacement acceptance windows. No identical-control pair, alternating order, or repetition exists, so run-to-run noise and order effects cannot be estimated. Within-run MAD/quantiles are not confidence intervals on a backend effect.

## Memory and allocator pressure

Values are allocator active/allocated/reserved peaks sampled at window boundaries, each update and before native peak resets. Units are GiB (2^30 bytes). They are allocator observations, not NVML total-device peaks. Active equals allocated in these records; per-rank values below show all observed peaks. All measured allocator retry and OOM counter deltas are zero on every rank in every run.

| Variant | Rank | Active/allocated peak GiB | Reserved peak GiB |
| --- | ---: | ---: | ---: |
| fla | 0 | 67.415422 | 67.945312 |
| fla | 1 | 68.003489 | 68.613281 |
| fla | 2 | 67.829949 | 68.261719 |
| fla | 3 | 68.099291 | 69.937500 |
| fla | 4 | 67.863855 | 68.320312 |
| fla | 5 | 67.440467 | 67.919922 |
| fla | 6 | 67.267522 | 67.851562 |
| fla | 7 | 67.309961 | 67.863281 |
| external_gate | 0 | 67.386829 | 67.925781 |
| external_gate | 1 | 67.892100 | 68.455078 |
| external_gate | 2 | 67.787388 | 68.259766 |
| external_gate | 3 | 68.080414 | 69.859375 |
| external_gate | 4 | 67.862168 | 68.339844 |
| external_gate | 5 | 67.476156 | 68.017578 |
| external_gate | 6 | 67.289329 | 68.005859 |
| external_gate | 7 | 67.388394 | 67.843750 |
| fused_gate | 0 | 67.384616 | 67.863281 |
| fused_gate | 1 | 67.889238 | 68.396484 |
| fused_gate | 2 | 67.825955 | 68.337891 |
| fused_gate | 3 | 68.034949 | 70.035156 |
| fused_gate | 4 | 67.855766 | 68.300781 |
| fused_gate | 5 | 67.407448 | 67.960938 |
| fused_gate | 6 | 67.228237 | 67.888672 |
| fused_gate | 7 | 67.391041 | 68.039062 |

- fla maximum across ranks: allocated 68.099291GiB (+0.000% versus FLA maximum), reserved 69.937500GiB (+0.000%). Maxima are reported separately and need not represent one simultaneous allocation state.
- external_gate maximum across ranks: allocated 68.080414GiB (-0.028% versus FLA maximum), reserved 69.859375GiB (-0.112%). Maxima are reported separately and need not represent one simultaneous allocation state.
- fused_gate maximum across ranks: allocated 68.034949GiB (-0.094% versus FLA maximum), reserved 70.035156GiB (+0.140%). Maxima are reported separately and need not represent one simultaneous allocation state.

## Numerical trajectory and activation

Rank0 has exactly one finite loss and finite aggregate grad_norm record for each update1–140 in every run. Values below are rounded native stdout metrics. Each variant reduces its early loss over this short trajectory, with later fluctuations and different gradients; this does not establish convergence, equal updates, or elementwise parity.

| Update | FLA loss / grad_norm | External loss / grad_norm | Fused loss / grad_norm |
| ---: | ---: | ---: | ---: |
| 1 | 15.32036 / 9.9607 | 15.32038 / 9.9403 | 15.32043 / 9.9526 |
| 40 | 7.49004 / 16.0799 | 7.49044 / 16.0720 | 7.49001 / 16.0733 |
| 41 | 7.28230 / 15.6737 | 7.28284 / 15.6605 | 7.28245 / 15.6650 |
| 80 | 3.28671 / 2.8375 | 3.27270 / 3.5467 | 3.28774 / 3.7341 |
| 120 | 2.62990 / 2.4160 | 2.61348 / 2.8654 | 2.68379 / 6.0081 |
| 140 | 2.70112 / 5.0579 | 2.70210 / 4.4642 | 2.68365 / 3.2517 |

All24 rank/run logs contain only their requested backend and both grad_enabled=False/True entry markers. The marker is emitted at `backend.py:135–138` in `run_gdn`, so it proves invocation rather than counting backward kernels. The inspected selected autograd route (including fused `_FusedGate.backward`) and successful140-update backward/optimizer training with finite aggregate gradients support native forward+VJP execution; no dedicated backward counter or elementwise gradient comparison is available in these native artifacts. Captured block parity is separate evidence, not a substitute for whole-training parity.

## Timing scope and limits

| Scope | Availability / method |
| --- | --- |
| Integrated latency | Available: single-host rank-envelope host seconds with drained CUDA boundaries, full100-update window. |
| Device compute | Unavailable from these unprofiled sidecars; no isolated per-operation kernel cost. |
| Host work | Included in integrated latency; Python/dispatcher/autograd/allocation/submission attribution unavailable. |
| Host synchronization and GPU idle/waits | Boundary drains known; separate duration/dependency attribution unavailable. |
| Copies and communication | Included where on the critical path; separate stream/rank durations and attribution unavailable. |

These scopes may overlap and cannot be reconstructed by subtracting unmatched timers. No kernel-cost, communication-saving, scaling, causal explanation or general full-training speedup claim follows from this sequence. Additional profiling is outside these native results. Production convergence, checkpoint/resume, validation and EMA-update behavior remain unverified by these140-update runs.

## Completion and stop decision

The existing analyzer accepted all three runs: PASS base completion, exit0, accepted native parser summary, unchanged source/config receipts, expected resolved-config hashes, eight matching bootstraps and contiguous140-update sidecars, complete unique rank0 numerical series, correct token geometry and actual requested backend invocation. Fused completed2026-11-14T18:04:28.010819+00:00. Terminal verification found no live owned process or process-group rows and no GPU compute applications. The detached fused controller receipt alone was updated with terminal verification; the earlier sequence controller BrokenPipeError receipt remains historical evidence, after its first two runs had already been verified.

Files: [machine-readable results](native-results.json), [recipe plan](native-plan.json), [config comparison](native-config-comparison.json), [timer/source notes](analysis-preparation.md). Raw evidence is under `/workspace-SR008.nfs2/vkropotin/opt-fq/runs/flash-qla/artifacts/raw/`; the terminal receipt is `controllers/fqla-native-fused-001.json` there. The native comparison deliverable is complete; no more native runs or source changes are authorized or required for this descriptive report.
