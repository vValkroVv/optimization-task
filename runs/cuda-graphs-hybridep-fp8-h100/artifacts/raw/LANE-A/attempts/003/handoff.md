# Lane A graph-on diagnostic, attempt 003

## Result

The one authorized attempt exited 1 during the first update’s Inductor CUDA Graph Trees forward warmup. No optimizer update completed on any rank. No retry, option change, source edit, fallback, or other lane launch occurred. This attempt establishes neither successful capture/replay nor a performance or numerical result.

The first timestamped and first supervisor-output error is rank 5 at `2026-11-11T23:22:18.177944Z`; the first matching supervisor RuntimeError is at line 253, after its traceback begins at line 153. All eight ranks reported:

```text
RuntimeError: Detected 1 tensor(s) in the cudagraph pool not tracked as outputs. All live allocations must be tracked for correctness.
```

Each rank records one `fwd_bwd_start`, `fwd_bwd_error`, and `fwd_bwd_end`, followed by `step_error` and `step_end`. No rank records `optim_start` or `optim_end`. The `step_end` event is emitted while unwinding and does not indicate an optimizer update.

## Resolved configuration and activation

The rank-zero resolved config confirms the pinned 3B HybridEP FP8 preset, EP8/FSDP2, selective activation checkpointing, model and loss compilation, Flex attention, local batch 8/global batch 256, sequence length 4096, and 25 requested steps. The source preset hash is `b341701bd1eef7774c110fbf7ed69cf6280ba8691100b05be457dee3e7d50a7b`; the final source manifest hash is `eae89a656a0e59efb2248dd921395d418afa2485ab2ae320d1e834a14ce97526` (22 files, archive hash `ea215abaac14d98a21c36e8684dbd8ce88a1d48f47cec5fe4f85f61be5dd4e27`).

The resolved callback order is timing observer then `campaign_cudagraph_step_boundary`. Supervisor line 2135 records the native `batch_start` dispatch in that order. The callback has no custom per-step log line. The timing observer starts at update 3, which this run never reached. `TORCH_LOGS=cudagraphs` contains 16 tree-recording lines and eight forward plus eight inference warmup lines. These do not prove successful capture or replay. No profiler trace was produced before the skip-first-20 window, so actual `cudaGraphLaunch` calls are unmeasured.

The attempt-local HybridEP runtime path produced eight runtime manifests and eight backend header copies. The 500 ms `nvidia-smi` sampler saved 432 rows; peak sampled device-wide memory use was 6,299 MiB. This is not PyTorch allocated or reserved memory.

## Launch, cleanup, and evidence

- Popen/runner PID `510132`; started `2026-11-11T23:21:56.091983Z`; runner exit code 1 at `2026-11-11T23:22:24.064243Z`.
- Rank worker PIDs were `510215`–`510222`; sampler PID was `510144`.
- Fresh post-unwind ownership check at `2026-11-11T23:24:46.061938Z`: all eight same-UUID H100s were at 0 MiB and 0% utilization, with no compute apps. The runner and sampler PIDs were absent.
- `preflight.json` records the fresh prelaunch idle, prior-PID, source, archive, and preset checks. `postrun-ownership.json` records release after unwind.
- `supervisor.log` preserves all-rank interleaving and CUDA Graph logs. Rank stacks are retained in `torchrun/*/attempt_0/{0..7}/error.json` and `stderr.log`; `failure-context-first-rank.txt` extracts the first rank-5 stack. All eight structured event logs, resolved config, per-rank runtime manifests and headers, memory samples, lifecycle log, and `exit-status.json` remain in this attempt directory.
- `failure-summary.json` provides the compact event, config, memory, and failure record.

The monitor controller hit a setup error after Popen: `TypeError: JSONEncoder.__init__() got an unexpected keyword argument 'flush'`. It failed before entering its wait loop. The existing PID was monitored once by exact PID, and the runner’s `exit-status.json` supplied the exit result. This controller error did not change training source or options and caused no relaunch.

## Acceptance status

The one diagnostic attempt failed. It did not complete 25 optimizer updates, produce all-rank optimizer events, reach the timing window, or produce traces with actual graph launches. It therefore supplies no replay coverage, numerical parity, stable timing, or speedup evidence. The separate graph-off control and further diagnosis remain outside this one-attempt scope.
