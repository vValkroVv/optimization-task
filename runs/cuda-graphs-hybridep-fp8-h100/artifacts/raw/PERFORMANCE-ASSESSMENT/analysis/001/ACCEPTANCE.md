# Bounded native performance acceptance

1. Keep preparation002, Source010 and supplied workload/options frozen; no source or launcher writes.
2. Run native006–010 serially only while the prior attempt completes validly, owned processes are released, and the exact eight GPUs are idle. Stop on first CUDA, harness or native validity failure.
3. Require all eight native ranks: 41 optimizer/164 forward-backward completions, valid twenty-update timing window21–40, already-global20,971,520 useful tokens, finite native loss/norm, intact logs, compiler/config/environment proof and complete phase memory.
4. Extract integrated useful throughput, completed mean latency, labeled host/current-stream series, allocator intervals21–30/31–40, sampled NVML memory and setup separately.
5. Report A/A variation and alternating paired ratios if all runs pass; otherwise measurement blocked. Preserve failed/NOTRUN attempts and explicit replay/long-run limits.

Applying perf-experiments and titan-train-job skills. Reuse accepted preparation/review and A008 mechanism evidence; no new trace or diagnostics.
