# Matched recapture pair handoff

Status: C0 is a valid graph-off control. C1 exited with code 0 and its five-update observer window is complete and valid, but the run-level structured-event completeness gate failed after logger flush errors and persisted JSONL sequence gaps and duplicates. The pair supports a descriptive window comparison; it does not pass the full event-completion gate. No retry was performed.

## Primary window result

- C0 graph off: **318,557.08 useful tokens/s**, from 5,242,880 already-global valid tokens divided by 16.458212 seconds of slowest-rank completed-window wall time.
- C1 graph on: **36,055.46 useful tokens/s**, from the same 5,242,880 tokens divided by 145.411524 seconds.
- In this single diagnostic pair, graph-on throughput was 11.32% of graph-off throughput and its window was 8.84 times longer. This is an observed pair, not an A/A noise estimate or a general speedup claim.

Each observer window covers updates 21–25 and closes at boundary 26. The global valid-token count is already reduced; it is counted once, not summed over ranks. The completed-window timer starts after the pre-window device drain and rank barrier and stops after the closing device drain. It includes queued device tail and excludes final aggregation and persistence.

## Configuration and activation

Both rank-zero resolved configurations have the same SHA-256 after normalizing the attempt output path: c93d1885757ba6d35be3302b26bfd03d057ec202ec8fa3c1998c4ae87c1908bc. Both runs use source snapshot 006 and manifest 3a606c81d6b13692e422da04c333252620e504df9f1c4ba59c4bcf0c1a51f603. The graph option is the only planned runtime difference. All eight compiler-option records in both runs report mix_order_reduction=false and graph_partition=true; triton.cudagraphs is false for C0 and true for C1. Both use Torch 2.15.0.dev20260822+cu130.

The resolved preset is 3B on c4_test, sequence length 4096, global batch 256, local batch 8, accumulation 4, float32 training with bfloat16 parameters and float32 reduction, fused AdamW, selective activation checkpointing, Inductor compilation of model and loss, FSDP2 and EP8, FP8 dense shared quantization and sparse grouped GEMM, seed 42, and no checkpointing. Seventeen MoE layers are excluded from graph capture. Batch, EP, FSDP, loss, and world groups contain ranks 0–7.

Rank-zero observer markers report 18 activation-checkpoint wrappers, 18 installed model callables, 72 dense FP8 linear modules, 18 shared quantizers, 17 HybridEP dispatchers, and 17 sparse grouped-expert modules. The observer leaves loss compile execution and FP8 custom-op execution pending; it does not observe CUDA Graph replay. These are configuration and constructed-component markers, not full runtime execution proof.

## Per-rank timing

Host values are medians of host_update_submission_spans_seconds between batch-start callbacks. They include native post-update work and are not pure Python CPU time. CUDA values are medians of asynchronous current-stream boundary event spans. They are not device-compute time and may omit independent side-stream work.

| Rank | C0 window wall (s) | C0 host span median (s) | C0 CUDA boundary span median (s) | C1 window wall (s) | C1 host span median (s) | C1 CUDA boundary span median (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 16.458207 | 3.263578 | 3.274959 | 145.407203 | 28.798083 | 28.767172 |
| 1 | 16.458212 | 3.263623 | 3.274954 | 145.407148 | 28.755274 | 28.767033 |
| 2 | 16.458178 | 3.265531 | 3.274945 | 145.407138 | 28.766820 | 28.767006 |
| 3 | 16.458194 | 3.265728 | 3.274966 | 145.408370 | 28.831313 | 28.768578 |
| 4 | 16.458204 | 3.264539 | 3.274975 | 145.407146 | 28.755286 | 28.762648 |
| 5 | 16.458190 | 3.265588 | 3.274972 | 145.411524 | 28.893228 | 28.771758 |
| 6 | 16.458192 | 3.264480 | 3.274988 | 145.407174 | 28.823209 | 28.767340 |
| 7 | 16.458191 | 3.265901 | 3.274985 | 145.407187 | 28.641758 | 28.767361 |

Aggregate medians of persisted trainer-event records, in milliseconds: C0 fwd_bwd_end was 579.878 over 832 records and optim_end was 42.251 over 208. C1 fwd_bwd_end was 6,988.220 over 822 persisted records and optim_end was 12.266 over 200. C1 event records are incomplete and include duplicated entries; these raw-log medians are not complete operation timings or completed-window latency.

## Allocator boundary values

Values are GiB at the observer's opening and closing boundaries. The high-water counters are reported as-is; the observer does not reset them, so they are not isolated window peaks.

| Rank | C0 allocated current start→end | C0 reserved current start→end | C0 end high-water allocated/reserved | C1 allocated current start→end | C1 reserved current start→end | C1 end high-water allocated/reserved |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6.242→6.242 | 32.549→32.549 | 25.603/32.549 | 6.180→6.180 | 11.572→11.574 | 25.867/31.748 |
| 1 | 6.242→6.242 | 31.143→33.273 | 25.836/33.273 | 6.180→6.180 | 11.592→11.498 | 26.287/33.777 |
| 2 | 6.242→6.242 | 31.576→31.576 | 25.185/31.576 | 6.180→6.180 | 11.590→11.479 | 25.580/30.107 |
| 3 | 6.242→6.242 | 30.871→31.379 | 25.916/31.379 | 6.180→6.180 | 11.436→11.514 | 25.978/31.203 |
| 4 | 6.242→6.242 | 33.486→33.486 | 26.561/33.486 | 6.180→6.180 | 11.379→11.221 | 26.441/31.850 |
| 5 | 6.242→6.242 | 31.086→31.086 | 25.744/31.086 | 6.180→6.180 | 11.594→11.533 | 26.101/30.773 |
| 6 | 6.242→6.242 | 31.301→31.301 | 25.267/31.301 | 6.180→6.180 | 11.572→11.592 | 25.577/30.088 |
| 7 | 6.242→6.242 | 29.131→29.680 | 25.317/29.680 | 6.180→6.180 | 11.455→11.160 | 25.590/31.107 |

## Whole-run sampled device memory

device-memory.csv samples nvidia-smi memory.used nominally every 500 ms across startup, execution, and shutdown. These are sampled maxima, not exact peaks. The observer records step bounds and durations but no absolute window timestamps, so a window-only NVML peak cannot be aligned or claimed.

| GPU index | C0 sampled max (MiB) | C0 sample time (UTC) | C1 sampled max (MiB) | C1 sample time (UTC) |
|---:|---:|---|---:|---|
| 0 | 35,799 | 2026-11-12 02:14:21.554 | 36,329 | 2026-11-12 02:31:19.209 |
| 1 | 36,535 | 2026-11-12 02:14:51.198 | 37,265 | 2026-11-12 02:31:57.016 |
| 2 | 34,797 | 2026-11-12 02:14:27.099 | 35,705 | 2026-11-12 02:26:35.398 |
| 3 | 34,595 | 2026-11-12 02:14:39.681 | 35,345 | 2026-11-12 02:34:38.869 |
| 4 | 36,753 | 2026-11-12 02:14:26.599 | 38,975 | 2026-11-12 02:30:23.337 |
| 5 | 34,297 | 2026-11-12 02:14:52.701 | 35,327 | 2026-11-12 02:29:06.052 |
| 6 | 34,515 | 2026-11-12 02:14:27.100 | 35,815 | 2026-11-12 02:22:53.186 |
| 7 | 32,857 | 2026-11-12 02:14:49.697 | 34,329 | 2026-11-12 02:33:02.494 |

## C1 event integrity and acceptance

C1 exited 0 and logged training completion at step 26, but the required event records are not fully persisted. Each rank has 25 optim_end records with step 17 absent. Ranks 0, 1, 3, 5, 6, and 7 have 103 fwd_bwd_end records; ranks 2 and 4 have 102. Every rank has only two persisted fwd_bwd_end records at step 17. Step 18 has five entries on ranks 0, 1, 3, 5, 6, and 7 because seq_id 1278 is byte-identically duplicated, and four entries on ranks 2 and 4. C1 therefore fails the 26/104 per-rank event-completion gate. A missing optim_end marker is missing evidence, not proof that an optimizer update was skipped; these logs cannot independently certify tensor semantics.

The storage diagnosis found the common JSONL sequence gap 1254–1276 on all ranks, additional missing 1277 and duplicate byte-identical 1278 on ranks 0, 1, 3, 5, 6, and 7, and Python logging flush OSError [Errno 28] No space left on device in stderr ranks 0, 4, 6, and 7. Six stderr files and supervisor.log contain NUL bytes; all eight native JSONL files parse and contain no NUL. A post-run mount check showed 225,582,514,176 bytes available and 9% inode use, but it cannot establish capacity or backend behavior at the time of the failure. See [storage evidence](../../../C1-STORAGE-LOGGING/001/evidence.json) and [diagnosis](../../../C1-STORAGE-LOGGING/001/diagnosis.md).

No numerical loss or gradient comparison was interpreted. No profiler or TORCH_LOGS traces were produced. This pair is not an A/A noise estimate, finalist, or full-training speedup claim. C1's observer window remains a valid descriptive measurement; its run-level structured-event completion gate remains failed. The reviewed plan permits no retry. Any later finalist still needs the existing 50-update/500-update, numerical, continuation, and quality gates.

## Artifacts

- C0 detailed result: [attempt001 result](../001/result.json); GPU release: [attempt001 release](../001/gpu-release-check.json).
- C1 detailed result: [attempt002 result](result.json); GPU release: [attempt002 release](gpu-release-check.json).
- Raw observer, structured-log, compiler-option, and sampled-memory records remain in each attempt directory.
