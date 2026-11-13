# Native runner and measurement integration review

Scope: one combined review of `run.py`, `measure.py`, `parse_results.py` and
their actual native trainer, metrics, profiler, configuration and cache consumers.
No GPU execution by this reviewer. Final targeted fixes were reviewed after the
implementation owner froze scripts. **G1 launch approved**; executed GPU smoke
acceptance and final measured-parser verification remain separate gates.

Finite acceptance: safe single owned launch and teardown; fixed workload and
native CLI/configuration; complete all-rank G1 receipts; W..W+50 timing and token
denominators; memory resets and failure handling; final targeted owner tests.

## Material findings

1. **RESOLVED:** `Trainer.__init__` calls
   `setup_deep_gemm_cache`. Its default warmup path sets `DG_JIT_CACHE_DIR` to
   `/temp/.cache/rank_N/deepgemm/` through `_rank_cache_dir` in
   `gift/torchtitan/ops/warmup/deep_gemm_fast_warmup.py`. Both warmup and local
   kernel sharing write there. `run.py` only sets task-owned
   `DG_JIT_SHARED_CACHE_DIR`, so its fresh private-cache policy does not govern
   this active rank cache. The alternate `_deepgemm_cache_manager` also uses the
   hardcoded shared path. Use an opt-in task-owned rank-cache root consistently
   across both paths; preserve the canonical default and warmup behavior. Do not
   clean shared caches or disable warmup to close this finding. Smallest
   campaign-only remedy: patch the warmup module's `_rank_cache_dir` to return
   `<per-run task cache>/rank_N/deepgemm/` before native trainer construction.
   Guard the actual imported `trainer.setup_deep_gemm_cache` consumer against
   disabled warmup or nonempty `PROPAGATE_DG_CACHE` before delegating; either
   would select the alternate hardcoded manager. Restore both patches afterward.
   Focused verification should execute actual setup with a fake GPU warmup,
   check all eight output paths/env values, reject either fallback before writes,
   and exercise actual sibling kernel copying with tiny private sentinel files.
   Implemented `DeepGemmCacheRedirect` matches this design, installs before
   native construction, restores both symbols and rejects either alternate
   writer before setup. Owner's test executes actual setup and sibling sharing
   for all eight ranks under a private temporary task root with fake GPU warmup.
2. **RESOLVED:** `on_profiler_step` referenced
   nonexistent `window_finished`, failing measured runs at their start boundary.
   Owner added the missing property; complete W..W+50 and incomplete-window
   lifecycle tests now pass.
   Disabled-window G1 was unaffected.

## Integration and runtime-cost conclusions

- Native `Trainer.train` calls `Profiler.step` after `train_step`, checkpoint
  handling and validation. With saves/validation disabled, the adapter boundaries
  include input fetch, GC, transfers, all 32 microbatches, optimizer, scheduler,
  callbacks, monitoring and native metrics output. First-update timeout setup is
  after this boundary, but positive warmup excludes it from a measured window.
- Window synchronization occurs twice: after W and W+50. Endpoint `.item()`
  reads follow synchronization; the adapter adds no per-update device-value read
  or collective. Per-update host timestamps and allocator statistics cost O(1)
  per rank; all controls must use the same adapter. Census runs before training
  and uses local-shard metadata; it performs no parameter-value read.
- The physical numerator intercepts the actual `interval_metrics` reset and sums
  local label storage counts across ranks. The supervised numerator reads the
  replicated global `ntokens_seen` delta once, without multiplying by ranks/MTP.
  Full-row fixture/data-contract evidence remains the prerequisite for calling
  label-storage geometry physical input tokens.
- The memory hook patches the actual imported
  `torchtitan.components.metrics.reset_peak_stats` alias before native resets,
  preserving allocated/active/reserved maxima across resets. Native peak reset
  cadence remains unchanged. Retry/OOM differences retain their native lifecycle.
- Parser primary elapsed time is the conservative same-host global envelope
  `max(end)-min(start)`, explicitly labeled alongside local spans and rank skew.
  The investigation initially proposed maximum local span as primary; the final
  parser exposes both conventions and must be reported consistently. Host
  completion cadence is not device kernel time; tracing cannot enter TPS claims.
- The runner uses one new-session torchrun group, eight native ranks, all-rank
  logs, timeout cleanup restricted to that process group, unique owned local
  temporary storage, source/config hashes and persistent exit/failure receipts.
  It sources `wm.sh` then restores task TMPDIR, checks task imports and pinned
  dependencies, and does not dump full environment values or kill broad matches.
  The task-child shell sets `ulimit -c 0` after `wm.sh` and removes inherited
  TORCH tracing/compile debug, CUDA launch blocking, structured logger handlers
  and CUDA core-dump controls. Parent environment and limits remain untouched.
  Source/config edits must stop before launch and remain frozen during each run.
- G1 command after blockers/tests close, from the verified task root:
  `/opt/torchtitan-venv/bin/python3.12 runs/overnight-streaming-perf/run.py --run-id g1-streaming-smoke-UNIQUE --steps 10 --monitor-interval 5`.
  The runner infers microbatch capture. Actual all-rank 320-call receipts,
  finite native metrics, monitoring at 5/10, clean teardown and idle GPUs remain
  G1 acceptance gates. Source/CPU evidence does not establish those GPU results.

## Verification

Reused existing profiler and eight-rank data-fixture evidence; no redundant
reruns. Reviewed actual native `train.main(argv)`, YAML dot-path overrides,
resolved bundle location, trainer update ordering, physical counter reset,
supervised global counter update and native peak reset implementation.
Owner reports the following command passed all five adapter/cache tests:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/gift:$PWD/runs/overnight-streaming-perf" /opt/torchtitan-venv/bin/python3.12 -m pytest -q runs/overnight-streaming-perf/tests/test_measure.py
```

Owner also reports selective Ruff and AST checks passed. Reviewer inspected
the final fixes and their focused test behavior, without repeating passed tests.
Final measured-parser verification remains a G2 gate; it does not block the
disabled-window G1 smoke receipt.

Freeze `run.py`, `measure.py`, `parse_results.py`, `mds-dataloader.yaml`, generated
configuration, all `gift/` source/config/tokenizer bytes, MDS index/shards and
installed dependencies while the run is active. The runner hashes the parser
even in smoke mode. Documentation outside those inputs may be maintained.

Final helper SHA256 values:

| File | SHA256 |
| --- | --- |
| `run.py` | `b48a52a78149799c3159cad0fadaccef40a4ef9cafb954ad4376547ed414a2a8` |
| `measure.py` | `03d69be7aacd8e4d39990a0d71972ffd763c3141747672f4c6d793e51aaf8d05` |
| `parse_results.py` | `c26b2780a3d2b638be2a8ddc3573895f4ffcd22546a34c106606f517e6d5253c` |
| `mds-dataloader.yaml` | `9aa85c4e17d0fd813ab50e95ee14d5b6caba1b5c3e4ae9654fb28b8a047d49cc` |
