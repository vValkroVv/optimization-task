# A006 run handoff

**Status:** FAILED, single launch, exit code 1. This was the 3B MoE/HybridEP FP8 attempt on one eight-H100 node; all eight GPUs were released.

The runner started at 2026-11-12 00:55:23.653668 UTC and finished at 00:56:25.847532 UTC (about 62.2 seconds). First failure was observed on rank 3 around 00:56:19.66 UTC in step 1, inside loss.backward() in trainer.py:701. AOT Autograd entered an Inductor-generated backward callable and raised UnboundLocalError: cannot access local variable 'buf66' where it is not associated with a value at generated cache file line 2831 while building partition5_args. All eight ranks reported the same exception. The complete rank-3 traceback remains in [error.json](torchrun/c96d02b9-8a68-4ba7-9fe1-e417a4c4b72c_akumtnph/attempt_0/3/error.json) and [supervisor.log](supervisor.log).

Each rank recorded four batch fetch start/end pairs, one forward/backward start/error/end sequence, and one step start/error/end sequence. The end events followed the error during exception unwind. No optimizer step completed (0 updates). The launch requested 25 steps, but exited during step 1. No root cause is assigned beyond the observed generated-backward exception.

The observer records 18 selective activation-checkpoint wrappers and 18 installed compiled model callables. The requested Inductor model/loss compile configuration remained enabled; loss compile execution proof is pending. Resolved config used FSDP2 with dp_shard degree 8 and EP degree 8; EP/FSDP/batch/loss/world groups each span global ranks 0–7; efsdp degree 1 at rank 0. CUDA graph exclusions were logged for layers 1–17. The combined log has 96 tree-record markers, 40 forward warmup markers, 48 backward warmup markers, and 88 CUDAGraph-unsafe custom-op markers. It has no explicit capture, replay, or skip marker. Tree recording and warmup do not establish successful replay.

The requested steps 3–8 timing window is incomplete on all ranks. No profiler trace file was produced. The 500 ms sampler output is sampled memory data, not an exact peak. There is no stable-step throughput or speedup result.

The live worker maps showed the CudaCherry GEMM extension at /opt/torchtitan-venv/lib/python3.12/site-packages/cudacherry/gemm/_C_gemm.cpython-312-x86_64-linux-gnu.so; SHA-256 9548e0fdb0be6033a4bbd0e159ee9b04e20fcc12874b0d968e497ea18c53aaa5 (13,384,960 bytes; inode 74482301).

At 00:57:19 UTC, runner, torchrun, eight worker PIDs, and sampler PID were absent; all eight preflight UUID-matched H100s reported 0 MiB, 0% utilization, and no compute apps. No retry or additional launch was made.

Evidence: [launch invocation](launch-invocation.json), [lifecycle](lifecycle.log), [exit status](exit-status.json), [resolved config](native/config/resolved_config.yaml), [rank-0 structured log](native/structured_logs/training.global_rank_0.20261112-005534-7D99Q5.jsonl), [rank-0 timing observer](native/timing_observer/rank-00000.jsonl), [memory samples](device-memory.csv), [failure summary](failure-summary.json), [manifest](manifest.json), and [command record](command-record.md).

**Open:** the generated backward exception's source-level cause is unproven. The timing window, CUDA Graph replay, stable throughput, and exact memory peak remain unverified. The attempt is terminal; no retry was launched.
