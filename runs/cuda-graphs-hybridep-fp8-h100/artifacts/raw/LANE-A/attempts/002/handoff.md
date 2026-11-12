# Lane A graph-only diagnostic, attempt 002

## Result

One actual-target attempt ran on the exact 3B MoE HybridEP FP8 workload through the corrected `train_scan_fix` entry using the frozen `SOURCE-SNAPSHOT/002` source. It exited 1 during update 1, in the compiler CUDA Graph Trees forward warmup, before any optimizer event. This is one failed attempt; it is not a repeated or universal incompatibility result. No retry, fallback, option change, or source edit was made.

The first rank0 runtime error in `supervisor.log` output order is preserved in `failure-context-rank0.txt`; the original interleaved logs remain in `supervisor.log` and `torchrun/`. The exception line itself has no wall-clock timestamp, so output order does not prove which rank failed first in time. The earliest timestamped forward tree marker is rank5 at `2026-11-11T22:38:05.059000Z` (`Recording cudagraph tree`), followed by `Running warmup function=partition_0, mode=FORWARD` at `22:38:05.060000Z`. Rank0's corresponding marker is at `22:38:05.086000Z`; its first RuntimeError appears at supervisor log line 287. The native structured wrapper records rank5 `fwd_bwd_end` at `22:38:05.061772Z` and rank0 at `22:38:05.088173Z`.

All ranks 0–7 logged exactly one matching runtime error: `RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run.` The stack enters compiled Flex `create_block_mask` / `_create_sparse_block_from_block_mask` / `_dense_to_ordered` and then Inductor `cudagraph_trees.py` input storage validation. Each rank has one `fwd_bwd_start`/`fwd_bwd_end` pair and one `step_end` record, but no `optim_start` or `optim_end`; the fatal stack is inside model forward, so no completed optimizer update is evidenced. `TORCH_LOGS=cudagraphs` records partition creation and inference/forward tree recording and warmup markers. Those logs do not establish a successful capture or replay. The failure preceded profiler skip-first 20, so no CUDA API trace was produced and graph-launch/replay coverage is unmeasured.

## Launch and cleanup evidence

- Detached launcher PID 488216; started `2026-11-11T22:37:42.002804Z`; exit 1 at `2026-11-11T22:38:09.803725Z`.
- Fresh post-unwind ownership check: `gpu-ownership-handoff.txt` and `.json`, checked `2026-11-11T22:44:36.824348Z`. All eight H100s had 81,090 MiB free and 0 MiB used; compute-app query was empty; known launcher, sampler, and worker PIDs were absent; no process held the Lane A attempt path.
- `exit-status.json`, `lifecycle.log`, `device-memory.csv`, `nvidia-smi.stderr`, all eight structured logs, and all per-rank torchrun stdout/stderr are retained.
- Two auxiliary process-list checks were ineffective and were not used as GPU-idle evidence: one heredoc scanner matched its own shell wrapper; one `ps -p` invocation received the labeled `runner.pid` text. Direct `/proc` checks of known PIDs and the post-unwind all-device/compute-app check passed.

## Frozen execution contract

The reviewed launcher used the exact preset SHA-256 `b341701bd1eef7774c110fbf7ed69cf6280ba8691100b05be457dee3e7d50a7b`, 25 training steps, scheduler horizon 30,000, seed 42, deterministic debug false, EP8/FSDP2, FP8, selective activation checkpointing, model/loss compilation, and the original batch/sequence/accumulation settings. Checkpointing was disabled for this short diagnostic. The only compiler option change was `TORCHINDUCTOR_CUDAGRAPHS=1`; `TORCH_LOGS=cudagraphs` was diagnostic logging. Profiling was configured with skip 20, warmup 1, active 3, repeat 1, frequency 4; memory sampling interval was 500 ms.

The graph-off corrected prerequisite `SCAN-FIX-SMOKE/attempts/001` exited 0 after 25 updates on all ranks; it is only a bounded smoke, not a stable/noise or numerical comparison. The graph-option probe established only that the environment toggle changes `triton.cudagraphs`; it did not establish runtime capture.

## Next gate

Root owns the next scoped source diagnosis. Do not rerun this lane or change graph settings before that review. This attempt supplies no graph replay coverage, numerical parity, speed, or production-readiness evidence.
