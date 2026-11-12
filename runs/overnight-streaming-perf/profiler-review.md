# Profiler controls: independent integration and runtime-cost review

Verdict: **accept source handoff for G1/G3; no blocking source findings**.
Eight-GPU trace/lifecycle acceptance remains open. This review did not launch
GPU work, modify training source, or repeat previously passing suites.

Reviewed HEAD `d5e3d71723dcf77dd8712994966f8560356ff35b` plus the working
GIFT diff (SHA256
`b2b33c3e616457718d3ab643f0360e6c3914a2142689d659dad1f5a93d003a9b`).
Scope: the two profiler modules, two unit-test modules, debugging documentation
and 56 configuration goldens. The measurement adapter/data helper are outside
this review. Acceptance checklist: typed/default configuration; global-rank
validation and exclusion; native full-update schedule/lifecycle; independent
memory history; scoped costs and honest evidence limits.

## Findings and dispositions

- **Configuration accepted.** Schema `ProfilerConfig.__post_init__` rejects empty,
  duplicate, negative, boolean and non-list rank selections without importing
  torch. Decoder handles field types and unknown fields. `profile_ranks=None`
  and `record_shapes=True` preserve initialized native defaults. Independent
  diff inspection verified all 56 golden files contain exactly the two expected
  default additions, with no removals or other changed values. An initial
  inspection script compared incorrect indentation and reported false mismatches;
  corrected whitespace handling verified the actual additions.
- **Rank selection accepted.** `tools/profiler.py:202` uses default-process-group
  global rank/world size. World validation runs on every profiling-enabled rank
  before selection; out-of-world entries therefore fail even on excluded ranks.
  The return at line 218 precedes directories, postprocessor construction,
  schedule construction and `torch.profiler.profile`. Excluded ranks retain
  normal training and cannot export a trace through this handle.
- **Standalone fallback is a nonblocking behavior expansion.** Lines 202–206
  now allow trace-only direct use without a distributed group as rank 0/world 1;
  previously `get_rank()` raised. Native `Trainer.__init__` initializes distributed
  execution at `trainer.py:119`, before profiler entry at line 938, so this does
  not mask missing initialization on the active training path. Memory snapshots
  still require a process group. No campaign dependency on standalone fallback
  is required; it serves the native CPU schedule fixture.
- **Schedule and lifecycle accepted at source/CPU scope.** Native `Trainer.train`
  advances once at line 978 after all accumulation, optimizer/scheduler,
  callbacks, metric output, checkpoint and validation work. Neither trainer nor
  entrypoint changed. Native FT/Forge callsites retain the same interface, and
  graph-trainer postprocessing still applies to selected exported traces. The
  new CPU test records update 6, all 32 microbatch marker names and the optimizer
  marker, verifies `iteration_6/rank0_trace.json`, and rejects extra exit exports.
  These markers contain no real model computation. The test stops at update 6;
  it does not independently exercise later updates after repeat exhaustion.
  Schedule/repeat arguments and implementation remain unchanged. The fresh-step
  `N-2` formula is supported; resumed schedule behavior is not newly established.
- **Memory independence accepted.** Trace selection does not bypass
  `build_memory_profiler` in `__enter__`. The disabled-memory test asserts no
  `_record_memory_history` call; the excluded-rank test proves explicit memory
  snapshots can still start. Campaign must resolve `enable_memory_snapshot=false`.

## Runtime cost and verification

With profiling disabled, the existing immediate return remains before new rank
queries. With profiling enabled, added work is process-group metadata and an
O(number of selected ranks) Python membership/range check once at construction;
no device value read, transfer, collective, barrier or numerical operation is
added. Per-update `step()` remains unchanged. Selected ranks retain native
CPU/CUDA activity collection and synchronous export costs. `record_shapes=false`
removes requested shape metadata/reference retention; the magnitude of time or
memory benefit is unmeasured. Explicit `with_stack=false` and
`profile_memory=false` match previous torch defaults. Excluded ranks avoid trace
collection/export, but selected-rank profiling can still delay distributed peers.
Model precision, autograd, optimizer, checkpoint state and communication ordering
have no changed source path.

Reused implementer-reported checks, executed from `gift/` with its absolute path
in `PYTHONPATH` and `/opt/torchtitan-venv/bin/python3.12`:

- `python -m pytest tests/unit_tests/test_profiler.py tests/unit_tests/test_config_torch_free.py::test_profiler_rank_and_shape_controls_decode_torch_free -q`: 21 passed.
- `python -m pytest tests/unit_tests/test_experiment_parity.py tests/unit_tests/test_model_arch_presets.py -q`: 112 passed after intentional golden update, then 112 passed without update mode.
- Scoped undefined-name/default Ruff and format checks reported passing.

No raw test logs were saved by the implementer. Test/source inspection agrees
with the claimed focused coverage; this review did not rerun those checks.
The broader torch-free registry test's existing `campaign_numerical_probe`
registry mismatch remains a known failed check, not a profiler pass.

Remaining acceptance: actual eight-rank native training completion and teardown;
only ranks 0/1 constructing profilers; exactly two readable CUDA traces for an
ordinary warmed update containing the real 32 microbatches and optimizer;
resolved shape/stack/history settings, trace provenance and Perfetto inspection.
No numerical parity, CUDA/NCCL lifecycle, profiling-overhead magnitude, throughput
improvement or production readiness is claimed by this source/CPU review.
