# A2: corrected graph-off A/A control

Status: **prepared, not launched**. Root release of the GPU lease and fresh eight-GPU preflight are required before launch.

This is one of two identical corrected graph-off controls. The only intended differences between A1 and A2 are attempt-specific dump, log, and rank-local runtime staging paths. Both use the exact pinned 3B HybridEP/FP8 preset, scan-fix entrypoint, seed 42, graph policy off, native optimizer path, and a 50-update observer window over updates 21–70, closed by native update 71. The first 20 updates are excluded by the requested protocol; logs must still confirm initialization and compilation have finished before the window is interpreted as stable.

## Question and decision

Can the corrected graph-off run complete all 50 measured updates and close cleanly at update 71, and what matched A/A timing and sampled-memory variation does this one pair show? A valid pair is calibration only; it is not a graph speedup estimate or a causal claim. Two runs provide only a single paired calibration, not a robust multi-pair statistical noise distribution.

Expected observer denominator is 52,428,800 global valid tokens per run (1,048,576 per update × 50), counted once per update. The observer timer is its max completed-window elapsed time across ranks, not device-compute time. The 500-ms NVML sampler supplies timestamped device-memory samples over each full run; maxima are sampled, not instantaneous. Observer allocator counters remain boundary/high-water counters, not window-only or NVML peaks.

## Validity gates

Accept this attempt as one member of a valid A/A calibration only if all eight ranks exit successfully, each rank emits activation metadata and one complete valid 21–70 timing record closed at batch-start 71, expected tokens and runtime source pins are consistent, no fatal runtime errors occur, and the sampled memory log is usable with the sampler stopped by the shell trap. Inspect resolved config and native markers for selective AC and model/loss compilation; constructed module metadata alone does not prove compiled loss execution or FP8 custom-op execution. Leave unsupported runtime proof open. Profiling, memory snapshots, checkpoints, and tensor dumps remain disabled.

Before each attempt, root must confirm lease release, intended one-node eight-H100 topology and idle ownership, and unchanged source/preset pins. Run A1 then A2 serially. If either fails any acceptance gate, retain its result without a throughput/noise verdict and stop for root review before another attempt.
