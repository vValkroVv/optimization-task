# BASELINE-AA attempt 001

**Status:** failed before the configured timing window completed; no throughput or A/A noise value.

The run used the original 3B HybridEP/FP8 preset with only the approved timing observer, 71-update stop, `lr_scheduler.total_steps=30000`, seed 42, checkpointing disabled, and no deterministic flag. The observer window was configured for updates 21–70 and closes at `batch_start(71)`.

All eight workers reached training step 22. Their last successful optimizer event was `optim_end` at step 21. At step 22, rank 2 completed three forward/backward microbatches, then logged `fwd_bwd_error: AcceleratorError: CUDA error: an illegal instruction was encountered` during the fourth. Other ranks had begun that fourth microbatch; no rank logged a step-22 optimizer event. The first error is in the rank-2 structured log. The later NCCL watchdog error came from querying a CUDA event after the asynchronous device error; its stack does not identify the originating kernel.

- Exit status: 1; run ended at 2026-11-11 21:28:41 UTC.
- Rank logs: `native/structured_logs/`; per-rank stdout/stderr: `torchrun/`; combined log: `supervisor.log`.
- The observer did not write a complete timing record; no `native/timing_observer/rank-*.jsonl` files exist.
- The 500 ms `nvidia-smi` sampler recorded 290 samples per GPU from 21:26:14.059 to 21:28:40.463 UTC. Sampled peaks in MiB were GPU 0: 35877, 1: 36375, 2: 34355, 3: 33915, 4: 37353, 5: 34755, 6: 34673, 7: 32399. These are interval samples, not instantaneous peaks; they show no evidence of OOM.
- Post-run `nvidia-smi` health query at 21:32:32 UTC showed zero ECC counters on all eight GPUs, no pending row remap or remap failure, and retired-page fields reported N/A. Temperatures were 27–31°C. No compute apps remained. No Xid log path was already known, so none was probed.
- The exact sampler PID 458910, launcher PID 458908, and worker PIDs 459016–459023 exited. All GPUs returned to 0 MiB used. Checkpointing was disabled; no checkpoint cleanup was needed.

Keep this failure and all raw logs and memory samples. Do not interpret it as evidence for or against CUDA Graphs. A/A attempt 002 was not run pending root review of the asynchronous kernel failure.
