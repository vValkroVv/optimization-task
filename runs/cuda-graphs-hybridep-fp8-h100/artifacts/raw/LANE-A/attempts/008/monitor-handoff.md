# A008 monitor handoff

## Completion and release

- The single launch ran from 2026-11-12 02:39:15.723 UTC to 02:41:52.473 UTC and exited with code 0. The native log reports training completion and process-group destruction. No timeout or retry occurred.
- Source snapshot: `006`, manifest SHA-256 `3a606c81d6b13692e422da04c333252620e504df9f1c4ba59c4bcf0c1a51f603`.
- Controller 566556, runner 566557, sampler 566573, torchrun 566574 and ranks 566644–566651 were absent at release verification. All eight preflight H100 UUIDs showed 0 MiB, 0% utilization and no compute-app PIDs. See [gpu-release-check.json](gpu-release-check.json).

## Runtime and resolved activation

- Runtime metadata records Python 3.12.3, PyTorch `2.15.0.dev20260822+cu130`, TorchTitan 0.2.2 and world size 8.
- Resolved workload: `gift_moe` on `c4_test`, sequence length 4096, local batch 8, global batch 256, accumulation 4, 25 updates (scheduler horizon 30000), seed 42, and checkpointing disabled. Model dtype is float32, parameter mixed precision is bfloat16, and reduction is float32. FSDP2 and EP8 are active; TP, PP and CP are degree 1. The attention config is FlexAttention and the MoE dispatcher is HybridEP.
- Model and loss compilation are enabled with Inductor. Rank-zero startup confirms compiled loss, 18 compiled model blocks and selective activation checkpointing. All 18 blocks (`layers.0`–`layers.17`) were explicitly excluded from Inductor CUDA Graphs.
- All eight rank compiler-option records agree: `triton.cudagraphs=true`, `triton.mix_order_reduction=false`, `graph_partition=true`. Each rank's activation record reports 18 checkpoint wrappers, 18 compiled callables, 72 dense FP8 linear modules, 17 grouped-expert modules and 17 HybridEP dispatchers. These are constructed/installed markers; the callback says CUDA Graph replay was not observed and FP8 custom-op execution was not inferred from module presence (`runtime_execution_proof.status=pending`).

## Completed work and diagnostic windows

- Each rank's structured JSONL contains 1,576 valid records, 0 malformed lines and 0 NUL bytes. Each has exactly 25 `optim_end` events and 100 `fwd_bwd_end` events covering steps 1–25; all records are INFO and no error-like message was found. Rank stdout/stderr files also had no NUL bytes or traceback, ENOSPC, Errno 28, runtime-error or child-failure matches.
- The campaign timing observer was configured for steps 3–7 (`start_step=3`, five updates; closing boundary step 8). All ranks report a complete, valid window with no failure reasons. Its completed-window field is CPU wall time from the opening batch-start after a pre-window device drain/rank barrier through the closing device drain; it includes queued device tail and excludes final aggregation/persistence. CUDA-event spans are current-stream asynchronous spans, not device-compute time and may omit side-stream work. Host-submission spans cover successive batch-start callbacks and include native post-update metrics/checkpoint/validation cadence; the final span ends before closing synchronization. No timing values are interpreted here.
- `device-memory.csv` samples `nvidia-smi memory.used` in MiB every 500 ms: 310 samples per GPU (2,480 rows) from 02:39:16.010 to 02:41:51.914 UTC. The sampler stderr is empty.

## Warnings and trace inventory

- Nonfatal warnings included Inductor's complex-operator codegen fallback notice; untuned Triton launch-parameter defaults for key 1024 on each rank; Quack cache-lock timeouts that fell back to in-process compilation; and repeated `c4_test` re-loop notices. No A008 ENOSPC or fatal error was found. The profiler logged trace dumping at step 24 and completion in 2.00 seconds.
- Exactly eight non-empty traces were inventoried. Contents were not parsed in this handoff.

| Rank | Trace | Bytes |
|---:|---|---:|
| 0 | `native/profiling/traces/iteration_24/rank0_trace.json` | 245,107,177 |
| 1 | `native/profiling/traces/iteration_24/rank1_trace.json` | 245,207,100 |
| 2 | `native/profiling/traces/iteration_24/rank2_trace.json` | 245,290,255 |
| 3 | `native/profiling/traces/iteration_24/rank3_trace.json` | 245,420,149 |
| 4 | `native/profiling/traces/iteration_24/rank4_trace.json` | 245,329,767 |
| 5 | `native/profiling/traces/iteration_24/rank5_trace.json` | 245,213,459 |
| 6 | `native/profiling/traces/iteration_24/rank6_trace.json` | 245,456,202 |
| 7 | `native/profiling/traces/iteration_24/rank7_trace.json` | 245,440,892 |

No performance, replay or speedup conclusion is made here; those require the assigned trace review and separate evidence.
