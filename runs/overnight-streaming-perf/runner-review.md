# Native runner and measurement integration review

Scope: one combined review of `run.py`, `measure.py`, `parse_results.py` and
their actual native trainer, metrics, profiler, configuration and cache consumers.
No GPU execution by this reviewer. Final targeted fixes were reviewed after the
implementation owner froze scripts. **G1 launch approved**; executed GPU smoke
acceptance and final measured-parser verification remain separate gates.
The first G1 attempt subsequently failed before training; the targeted census
correction below is accepted for a retry. No G1 execution pass is implied.

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
3. **RESOLVED after failed G1 attempt001:** the census called actual
   `RaggedShardDTensor.to_local` with grad enabled. Its native guard at
   `gift/torchtitan/distributed/gfsdp/_tensor.py:674` rejects this. All ranks
   failed before any update; the [runtime receipt](g1-runtime-receipt.json)
   retains the failure and parent-observed GPU release. `parameter_census` now
   wraps local metadata iteration in `torch.no_grad()`. Trainability is still
   read from each original parameter. Native `to_local` returns its existing
   `_local_tensor`; subsequent dtype/numel reads add no copy, collective or
   tensor-value read. The context restores incoming grad mode on success/error.
   The regression binds the real native guard, reproduces the old enabled-grad
   direct-call failure, validates trainable/frozen counts and dtypes, and verifies
   grad-mode restoration after both success and an injected shard exception.
   Owner executed targeted census and adapter-smoke tests: **2 passed**, one
   third-party pynvml deprecation warning; selective Ruff and `git diff --check`
   passed. No additional reviewer execution or broad re-audit was needed.

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
  These peaks are not reset at W: the first sampled native peak can include the
  tail after the preceding native monitoring reset (updates51–60 for W60), then
  preserves the maxima through measured update110. Report this same lifecycle
  for paired runs rather than calling it an exact61–110 allocator peak. Retry/OOM
  deltas use the actual start/end boundaries.
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

G1 validated helper SHA256 values (G3 changes below have separate run hashes):

| File | SHA256 |
| --- | --- |
| `run.py` | `b48a52a78149799c3159cad0fadaccef40a4ef9cafb954ad4376547ed414a2a8` |
| `measure.py` after census correction | `269e04fd570b745bfb59b96e8bbf7bf2b8593870aee9df9825d2e20dbfee6c83` |
| `parse_results.py` | `c26b2780a3d2b638be2a8ddc3573895f4ffcd22546a34c106606f517e6d5253c` |
| `mds-dataloader.yaml` | `9aa85c4e17d0fd813ab50e95ee14d5b6caba1b5c3e4ae9654fb28b8a047d49cc` |

Census regression test SHA256:
`4c0df5c0e49abcc95330bd43c59ede243a36ebdc4f46d37b20b13eaa392c293b`.

## G3 profile-only update capture adapter

The task-local runner now accepts `--profile-update N` for `window=0`,
`monitor_interval=50`, positive warmup `W`, `N >= 2`, `N > W`, `N % 50 != 0`,
and `steps == N`. The resolved profiler config selects global ranks 0 and 1,
uses the native `wait=0 / warmup=1 / active=1 / repeat=1` schedule with
`skip_first=N-2` and `skip_first_wait=0`, disables shapes and memory snapshots,
and writes under `profiling/traces/iteration_N`. Native source still hardcodes
`with_stack=False` and `profile_memory=False`.

The adapter observes native `Profiler.build_torch_profiler` on all ranks. In
profile mode it temporarily delegates the actual `Trainer.train_step`,
`Trainer.forward_backward_step`, and `OptimizersContainer.step`; successful
target-update calls are counted on all eight ranks. Kineto `record_function`
scopes are created only on ranks 0 and 1 for the target update: 32 microbatch
scopes, one `train_step` scope, and one optimizer scope. The train-step scope
excludes `Trainer.train` GC, checkpoint, validation, and outer `sl.step` work.
No `sl.log_trace_span` bridge is added because native `Profiler.step` exports
while that scope is open.

Runtime cost is opt-in. With `--profile-update` unset, no new delegation wrappers
or per-update checks are installed, the unprofiled profiler overrides remain
false, and sidecars do not gain profile fields. Profile mode adds a target-step
and rank check at these native calls on all ranks so the receipt can verify
successful calls; only ranks 0 and 1 open record scopes. It adds no CUDA
synchronization, tensor copy, or collective. Native profiling and trace export
still perturb the captured update, so profile runs do not produce TPS evidence.

Focused verification used the task interpreter: **22 passed**, with one existing
pynvml deprecation warning. The exact pytest, selective Ruff, `py_compile`, and
`git diff --check` commands passed. Tests cover TorchTitan CLI override parsing
and typed config decoding, each resolved profile field, invalid CLI cases,
wrapper arguments/returns/exceptions/restoration, selected-rank scopes and
all-rank counts. The CPU lifecycle test uses the native TorchTitan `Profiler`
and actual `OptimizersContainer`; it captures update 7 with 32 microbatch
markers plus one train-step and optimizer marker, exports exactly one native
trace, and makes no adapter synchronization. Parser tests require exactly the
selected-rank traces with hashes, sizes, markers, and all-rank call counts;
missing, extra, wrong-rank, wrong-count, and wrong-build receipts fail.

The real eight-GPU/NCCL profile run, selected-rank trace pair, and Perfetto
analysis remain campaign gates. CPU lifecycle evidence does not establish
CUDA/NCCL integration or profile-run performance.

### Independent G3 review: profile-only marker finding resolved

The independent review found no unprofiled regression. Native schedule,
selected-rank construction, actual delegation, scope boundaries, exception
restoration and lifecycle/config/source gates match the contract by inspection.
The reviewer inspected tests but did not repeat their reported execution.

P2: the frozen parser at `02cfb427` filtered only the target-update prefix and
checked required marker keys individually. I added two negative regressions, then
ran them against that parser: an extra previous-update marker and a wrong-rank
marker both produced a false `complete` summary, so both tests failed as intended.

**Resolved:** `parse_results.py` now collects every `task.profile_update.*` scope
and requires exact name/count equality with the three expected target-rank
markers. It also rejects markers whose Kineto phase is not a complete event.
The two previously failing cases now pass along with the existing profile-parser
cases: 9 passed. Selective Ruff, default Ruff, `py_compile`, and `git diff --check`
pass. The change is limited to this parser, its two regression cases, and this
review note; `run.py`, `measure.py`, GIFT, config, and HEAD are unchanged after
the calibration. Profile-only source acceptance is closed. GPU/NCCL capture and
Perfetto remain separate campaign gates.

During CPU test development, an initial mocked-distributed fixture failed because
Kineto requires a real process group. The owner replaced that test setup with a
single-rank native CPU lifecycle check. It is a retained test-fixture failure,
not a native training or candidate verdict.

## Measured-window compilation observer contract

Final guard integration: the measured path retains its two existing boundary
synchronizations and activates `DG_CACHE_FORCE_SKIP=0` after W synchronization,
before snapshot/timing. The separate profile path adds one synchronization at W,
before its target update, and activates the same guard. This supersedes the earlier
capture-only no-added-sync statement. Both paths verify the installed extension
build/hash before training, retain the flag through the target window, store raw
activation/end states and restore the activation-time value on adapter exit,
including errors. Smoke/calibration mode does not import or activate this guard.
The actual m/k-grouped miss/hit/positive controls passed; see the campaign report
and compact `results.json` for pins and raw evidence. Final source tests:36 observer
and26 existing measurement tests passed with explicit task imports.

Calibration disabled compiler diagnostics, so cadence alone cannot close the
no-compilation gate. The bounded design samples existing CPU state on every
rank after start synchronization/before the start timestamp, and immediately
after the end timestamp. Keep snapshots in memory and serialize with the existing
receipt; no per-step work, added synchronization, cache changes or logging.

- Torch2.15 `torch._dynamo.utils.compilation_time_metrics`: record each existing
  timing-list identity and length. Require zero growth of compile-only events
  (`_compile.compile_inner`, `compile_attempt_*`, forward/backward compiler,
  `fx_codegen_and_compile`, `CachingAutotuner.*`). Retain all deltas; missing,
  replaced or decreased state is inconclusive/failure. The list is unbounded;
  `get_compilation_metrics()` is a bounded64-entry deque and is unsuitable.
- Existing Inductor `async_compile_cache_miss`/`hit` counters: require unchanged
  values without creating missing defaultdict entries. They conservatively mark
  compile-path entry, not exact compiler misses.
- Active CuTe norm `common.COMPILED` existing keys/object identities: unchanged.
  Its source calls `cute.compile` only on absent keys and does not clear the map.
- Existing Triton JIT registry/function identities and existing device kernel-cache
  keys/object identities: unchanged. Never index a missing device (CUDA binder).
  New specializations can include disk loads, so a change conservatively opens
  acceptance rather than proving a native compiler invocation.

Pinned inspected source SHA256: Dynamo utils
`21217311dbe8022f3793e9dfd61a2be66c50ecb97078c83b4c414e538fc5e832`;
Triton jit `38b8151fcceec1a0210a696775b7c64d7fac5e9b69eee2dda7e93c1e2a377509`;
task CuTe common `ab2912468af5661b0b9f23ddaaf36a829d0924a64e25b7c2455cabe70e12a205`.
Asynchronous compiler workers can outlive parent timing scopes. DeepGEMM
`2.3.0+f093a4c` exposes no verified compilation counter in the installed Python
API/stubs; matching compiler/cache source is being sought to establish a
post-run persistence audit. Wall timestamps around the timed window are needed
for that audit. These snapshots alone cannot claim complete JIT coverage until
the remaining source contract is resolved.

### DeepGEMM fail-closed alternative under runtime verification

The installed wheel records only short revision `f093a4c`; its image build removes
the source. The helper names `gigallms/deepgemm`, but matching public source was
not retrievable. Persistent-artifact guarantees therefore remain unproven.
The actual extension SHA256 is
`99b7afad2e2fe7ebabd26c22afc99dc652e2e6935fcbf37324359eb5aeeefe26`.

Binary inspection provides a bounded alternative: `Compiler::build` loads
`DG_CACHE_FORCE_SKIP` at0xc5e39, calls `get_env<int>` at0xc5e60, compares with1
at0xc5e70 and otherwise branches to the CACHE MISS exception at0xc6ed7.
`get_env<int>` calls libc getenv each time (0x359f0/0x35a08), so this guard is
not construction-cached. The active helper documents the same flag semantics.
A small private-cache positive/negative probe for both active m/k-grouped paths
is being prepared before any control integration: known hit must run at0,
different planned miss must fail at0 and then run at1. If verified, set literal0
after normal warmup/before measurement on every rank and retain it through the
window. This preserves tensor computation and fails the run on a new cache miss;
it does not establish complete source provenance or arbitrary native paths.
