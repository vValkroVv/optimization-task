# Native trace attempt 003 independent review

## Acceptance and current verdict

**PASS: attempt003 satisfies the requested two-rank native diagnostic trace.** All five acceptance items below are closed by the final evidence section.

Finite acceptance checklist:

1. Task-only adapter preserves native trainer, profiler handle, schedule, callback, stepping and teardown; constructor settings restore on success and failure.
2. Accepted eight-rank launch, Source010 pins, descriptor-fix-001 precedence, seed42, scheduler horizon30000, exact production preset, graph0/mix0 and scan correction remain intact.
3. Live markers prove global ranks0/1 record CPU+CUDA with shapes/stacks/memory disabled; ranks2-7 never construct a profiler.
4. Exactly two parseable traces cover the same single complete optimizer update111, with all four microbatches, forward/backward, communication, clipping and optimizer work.
5. All eight ranks complete111 updates/444 microbatches with finite metrics, intact logs and clean teardown.

## Integration review

Reviewed `attempts/003/profile_entry.py`, launcher diff against accepted `BASELINE-STABILITY/attempts/004/launch.sh`, controller diff against accepted stability preparation003, and the inherited native profiler/trainer consumers.

The adapter captures `torch.profiler.profile` before patching, returns `None` for excluded global ranks and invokes the original native builder for selected ranks. A scoped `unittest.mock.patch` forwards every native constructor argument except the three explicitly false diagnostic flags. Context exit restores the constructor after success or failure. The native profiler object remains responsible for scheduling, trace export, stepping and teardown. This does not create a new profiler framework or change source/schema/defaults.

Native `gift/torchtitan/trainer.py` fetches all accumulation microbatches, executes four complete forward/backward calls, clips gradients and steps optimizers/schedulers inside `train_step`. The native loop advances the profiler once after that call and disabled checkpoint/validation handling. The wait109/warmup1/active1/repeat1 schedule selects update110 for warmup and111 for recording/export, then the loop ends. No partial microbatch step advances the profiler.

The launch diff preserves accepted environment, offline data/cache route, source and dependency byte gates, GPU identities, eight processes, descriptor-fix precedence and native scan correction. Changes are diagnostic entrypoint, output, stop111 and profiler schedule. Controller retains the accepted bounded owner with exact task-entrypoint matching and a fresh-output guard.

The preparation report mismatch about `profiler_skip_first_wait` was corrected to the actual explicit0 flag. No material findings remain.

## Runtime-cost review

Adapter additions occur once during profiler construction: global-rank selection, marker logging, constructor wrapping and three option overrides. They add no tensor computation, transfer, communication, optimizer state or per-microbatch work. Excluded ranks retain a `None` profiler handle. Native profiler step bookkeeping remains on selected ranks; warmup110, capture111 and trace export intentionally add host/device diagnostic overhead. Profiled ranks may delay unprofiled peers at collectives. Shapes/stacks/memory instrumentation is disabled, but CPU/CUDA collection remains perturbing.

No throughput, speedup, numerical equivalence or unprofiled runtime-cost magnitude is claimed. Those comparisons are outside this single diagnostic trace request.

## Prelaunch checks and then-open gates

Read-only integration inspection found no material prelaunch blocker. Focused CPU adapter success/failure/rank checks and syntax checks are owned by preparation; their results must be retained. At prelaunch, actual resolved configuration, adapter live markers, two trace JSON contents, all-rank completion and clean process-owner outcome were open gates. The final section below closes every one. No GPU job, monitor, source edit or repeated native200 check was run by this reviewer.

## Final execution and trace acceptance

Final verdict: PASS. Evidence is in `acceptance.json`; this is a diagnostic capture acceptance, not production performance or quality acceptance.

- Exactly two trace JSON files exist under attempt003/native, both at `profiling/traces/iteration_111`: rank0 is66,720,244 bytes and rank1 is66,754,593 bytes. Both parse successfully and include a closing `Record Window End` event. No trace exists for ranks2-7 or another update.
- Each trace contains60,500 CPU operators and15,781 CUDA kernels. Actual kernel device fields are exclusively device0 for rank0 and device1 for rank1. Eight entries in `deviceProperties` describe visible devices; they do not indicate eight captured ranks. Distributed trace metadata independently reports the selected rank, NCCL backend and world_size8.
- Each has exactly one CPU `ProfilerStep#110` annotation. This is zero-based profiler index110, corresponding to training update111. Native callback logs identify export step111 on ranks0/1 exactly once. The adapter live markers prove ranks2-7 disabled and selected-rank constructor options `record_shapes=false with_stack=false profile_memory=false`.
- Structured log epoch `time_us` was converted to trace timestamps using `trace.baseTimeNanoseconds /1000`. Every update111 fetching_batch interval (four), fwd_bwd interval (four) and optim interval (one) lies within the CPU profiler interval on each selected rank. Each fwd_bwd interval includes exactly one model entry `FSDP::pre_forward (tok_embeddings)`, backward CPU events, communication events and CUDA kernels. There are320 recognizable NCCL/HybridEP NVLink communication kernels per trace. The optim interval contains `aten::_foreach_norm` and `aten::_fused_adamw_`; inspected native source places clipping and optimizer/scheduler updates there. This closes whole-update accumulation and communication coverage without enabling shape or stack instrumentation.
- Neither trace contains input shape/type/stride fields, call-stack fields, Python-function collection or memory-allocation events. CUDA memcpy/memset activity is ordinary CUDA activity and does not imply allocation profiling.
- Active traces contain zero compilation/recompilation work events. Each has108 benign `TorchDynamo Cache Lookup` events plus cached compiled-region/function execution; no startup compiler work is captured. Native schedule warmup110 precedes recording111, consistent with the earlier accepted settled-window selection.
- All eight structured logs have zero `*_error` events and zero ERROR/CRITICAL records. They have contiguous sequence IDs, completed updates1..111 exactly,444 completed forward/backward microbatches and distributed teardown completion. All recorded numeric values and all displayed loss/gradient-norm metrics are finite. Native metrics retain their existing ten-update cadence, ending at110; no final111 loss display was added. All eight stdout logs report `Training completed` and `Process group destroyed`.
- `attempts/003/live-activation.json` confirms exact resolved-config differences against accepted native200 are only stop, profiler and output fields; all eight live workers used descriptor-fix-001 extension with expected hash and graph0/mix0/world8. Required3B HybridEP/FP8, FSDP2/EP8, selective AC, model/loss compilation, Flex, c4_test, batch8/256, seq4096, accumulation4, seed42 and scheduler30000/warmup2000 are preserved. Source/dependency post-run immutability and idle-GPU checks are owned by root in `attempts/003/root-post-run-release.json`.
- Controller completion is exit0. Process owner reports `natural_exit`, exit0, all691 recorded identities absent and no remaining identities. No checkpoint was requested or produced. The prelaunch syntax/Ruff/CPU rank/options/restoration/schedule checks passed; the unsupported `/dev/null` py_compile attempt remains recorded, superseded by successful in-memory syntax compilation.

No required diagnostic acceptance gate remains open. No throughput estimate, benchmark, new trace framework, source edit or GPU rerun was performed by this reviewer.
