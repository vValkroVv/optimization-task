# A1 execution handoff

- Outcome: A1 exited 1 and is invalid for the requested 50-update A/A calibration.
- First material failure: rank 6, native step 33, forward/backward error: CUDA illegal instruction, CUDA_ERROR_ILLEGAL_INSTRUCTION, return 715 from cuLaunchKernel. The later ProcessGroupNCCL CUDA-event watchdog abort is downstream.
- No completed 21-70 timing record or update-71 close exists. The final rank-6 step_end is exception-path output and does not indicate successful work.
- A2 was not launched because its conditional release requires valid A1 completion.
- After trap completion, all known attempt PIDs were absent and all eight H100s were idle. The 19-file source guard and exact preset hash passed after the run.
- Full-run 500-ms device-memory samples and post-fault ECC/row-remapper query are retained. They are diagnostics, not matched-window memory evidence. Existing structured logs contain no scalar training-loss or norm series.

## Evidence

- result.json: compact outcome and per-rank phase summary.
- supervisor.log, torchrun/, native/structured_logs/: failure and per-rank records.
- device-memory.csv: raw sampled memory values.
- postfault-gpu-state.txt, postfault-source-check.json, postrun-pid-check.txt: GPU health, frozen source identity, and cleanup evidence.
- Prepared command was /bin/bash launch.sh through the detached subprocess.Popen method recorded in launch-invocation.json.
- Do not infer A/A noise, throughput, graph speedup, or hardware root cause from this failed single attempt. No retry or altered launch was performed.
