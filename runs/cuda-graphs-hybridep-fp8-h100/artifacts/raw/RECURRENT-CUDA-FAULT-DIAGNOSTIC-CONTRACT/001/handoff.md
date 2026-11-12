# Recurrent CUDA fault: implementation contract

## Result, scope and acceptance

Use a task-only experiment adapter around the module-global `_run_on_comm_stream` helper. The actual registered CUDA body resolves that global at runtime, so the adapter can wait immediately after the native communication joins compute, before `combine_bwd` resumes. No core-source hook or framework/package modification is needed. This contract is preparation only; it does not schedule a run ahead of A006.

Finite acceptance: verify (1) actual dispatcher/helper binding; (2) exact boundary, counter and lifecycle ownership; (3) default-off costs and peer-progress limits; (4) finite CPU checks and one target attempt with decision outcomes. These design items are complete. Implementation, CPU behavior checks, independent implementation review and GPU execution remain open. No GPU work, source/package edits, installation, or new harness occurred here. Applied `code-rules`, `perf-experiments` and `titan-train-job` to the design.

Reuse [the prior diagnosis](../../RECURRENT-CUDA-FAULT/001/handoff.md), including its corrected E4 scan provenance and native-stream investigation. Original graph0 failed at update 22; corrected smoke25 passed; corrected long graph0 failed in the first microbatch of update 33 after 32 optimizer completions. Sticky 715 was noticed in a later compiled pointwise launch after `combine_bwd`; the originating kernel is unknown. This diagnostic localizes an executed prefix; it neither fixes the fault nor eliminates nondeterminism.

Production contract remains 3B MoE, one eight-H100 NVLink node, EP8/FSDP2, HybridEP fused communication, FP8, FlexAttention, selective AC, model and loss compile, local/global batch 8/256, S4096, accumulation 4, unchanged data/routing/counts, loss/optimizer/schedule and 30000-step scheduler horizon. Only graph0 is eligible. The bounded run retains the prior checkpoint-disabled fault-reproduction setting; it establishes no checkpoint-continuation claim.

## Binding verified against source

Task HEAD is `446431dfa191ef0300f116257ae918bc85e38c06`, branch `main`, with existing snapshot004 changes. [Source pins](source-pins.json) pin the files inspected here; they are not a replacement for the next integrated snapshot. The concurrent MoE graph-exclusion owner owns `config/configs.py` and `distributed/compile.py`; neither was inspected or modified for this contract.

- `gift/torchtitan/distributed/deepep/hybridep.py:630` decorates `_combine_bwd_impl` with `torch.library.custom_op("hybridep::combine_bwd", mutates_args=(), device_types="cuda")`. `_dispatch_backward` calls `torch.ops.hybridep.combine_bwd` at line 721. Rebinding the module name `_combine_bwd_impl` alone cannot intercept that registered callable.
- The decorated original Python body invokes global `_run_on_comm_stream` at line 650, supplying a nested lambda at line 651. A function retains its module globals dictionary; assigning `hybridep._run_on_comm_stream` changes that lookup in the captured body too. The compiled backward retains the custom op and invokes the registered runtime kernel. The helper is not independently inlined into the compiled graph; fake execution uses `_combine_bwd_fake` and does not enter the body.
- `_run_on_comm_stream(fn, *produced_on_compute)` at line 121 invokes `_launch_on_comm_stream`, then `join()`, then returns its exact `out`. The compute stream waits on the comm stream in `_join`; output tensors receive `record_stream(compute)`. The outer adapter runs immediately after this return, before output destructuring and the nullable-probability fallback at line 660. This preserves the requested boundary more precisely than wrapping the complete custom-op return.
- Installed Torch is `2.15.0.dev20260822+cu130`, commit `8b353065ce260336c5fdfa74d754e8bd9b91b062`. `torch.library.register_kernel(op, "cuda", wrapper)` is a real public API and forwards custom ops to `CustomOpDef.register_kernel`. That implementation updates the backend callable used by the dispatcher. However, the original callable is exposed here only through private `_init_fn`/`_backend_fns`; calling the custom op from its replacement would recurse. Do not use those private fields, disable kernels recursively, redefine the op, or register a new autograd/fake rule. The helper adapter avoids these problems without a new source hook.

## Exact proposed task-owned route

Later worker may add only these small files in the authorized task `gift/` tree, plus focused tests and maintained experiment documentation:

1. `torchtitan/experiments/cuda_graph_campaign/combine_bwd_completion.py`: adapter installer/restorer and `CampaignCombineBwdCompletion` observer.
2. `torchtitan/config/schema/experiments/cuda_graph_campaign/combine_bwd_completion.py`: torch-free `@config(type="campaign_combine_bwd_completion")` callback schema with `_owner_spec` pointing to that observer. Schema auto-discovery already imports modules beneath `torchtitan.config.schema`; no registry change is necessary.
3. `torchtitan/experiments/cuda_graph_campaign/train_combine_bwd_completion.py`: explicit opt-in entrypoint. Initialize logging; call existing `install_hybrid_ep_scan_fix()` first; import/install adapter second; call native `torchtitan.train.main(argv)` inside `try/finally`; restore adapter in `finally`.

Keep the ordinary `train_scan_fix` and native entrypoints unchanged. No adapter import, helper replacement, branch, tensor operation, event, synchronization or log is added to the ordinary production path. The new entrypoint and exactly one callback must be paired; either missing is a startup error. Do not introduce a generic interception framework.

### Adapter guards and ordering

Installer saves the original helper function object. Verify before replacing it: module physical path is the task GIFT file; its SHA-256 equals the pinned `hybridep.py` hash; helper belongs to that module, is a Python function with signature `(fn, *produced_on_compute)` and its source definition starts at line 121. Reject installation after `_buffer` construction, duplicate installation, a preexisting adapter, or a mismatched source pin. Do not rewrite a different wrapper. Updated source requires reviewing this contract and refreshing the exact pin, not relaxing the guards.

The installed adapter has the same call signature. Select only a Python function with all of:

- `fn.__module__ == "torchtitan.distributed.deepep.hybridep"`;
- `fn.__qualname__ == "_combine_bwd_impl.<locals>.<lambda>"`;
- resolved `fn.__code__.co_filename` equals the pinned task module path;
- `fn.__code__.co_firstlineno == 651`.

Use public Python function/code attributes, not stack walking, inspecting tensor values, reading closure contents, or private Torch dispatcher state. Pinned module contents make this selector exact for the inspected body. Other helper callers pass through unchanged; a lambda claiming this exact qualified name but failing filename/line guards is an adapter error. Check static assumptions at installation; preserve any error record and terminate on unexpected runtime binding. Do not silently report success without selected runtime calls.

For each selected call, allocate an integer process-local ordinal before calling the original helper. Call the saved original exactly once with the same `fn` and the same positional objects. No CUDA call, rank barrier, event wait or device-value read precedes native communication. On its normal return, keep `fn`, inputs and exact output object strongly referenced in this adapter frame until the checkpoint completes; the lambda closure retains the dispatch handle. During an active update:

1. Flush a rank-local JSONL `checkpoint_enter` record with native update, microbatch, global selected-call ordinal and per-update ordinal.
2. Read `torch.cuda.current_stream()` in this runtime context; create `torch.cuda.Event(enable_timing=False)` and call `event.record(stream)` explicitly.
3. Call only `event.synchronize()`. Do not call `torch.cuda.synchronize`, stream-wide synchronize, `dist.barrier`, `CUDA_LAUNCH_BLOCKING`, or a pre-call wait.
4. Flush `checkpoint_complete`, then return the original `out` object unchanged. Preserve tensor identities, aliases, dtype, layout, None values and handle ownership; no clone/cast/contiguous operation is added.

Log stage-specific `native_call_error`, `event_record_error` or `event_wait_error` with the same counters and exception text, then bare `raise`. A native-call error may itself report earlier asynchronous work. Logging failure must fail the attempt, never fabricate completion. Do not retry, clear sticky errors, reset devices, run CUDA inspection in exception cleanup, or start another attempt.

### Native counter and observer lifecycle

The schema needs only `output_dir` (default `combine_bwd_completion`); updates 21–40 are fixed for this diagnostic. Empty/invalid paths fail validation. The observer uses the same process-local adapter state as the entrypoint; it owns no tensors or checkpoint state and emits no distributed metrics.

`fit_start(ObserverState)` checks entrypoint pairing, rank/world size 8, PP1, EP8, graph0 environment (`TORCHINDUCTOR_CUDAGRAPHS=0`), expected preset/runtime and fresh loaded step 0. Reject capture settings; do not disable them silently. Record rank, source identity and enabled window in a separately opened `rank-00000.jsonl`-style file for every rank. Use ordinary file write/flush, with no per-record fsync or metric reduction. The entrypoint must fail before training if this callback is absent; use its loaded configuration validation without duplicating native training or mutating the config.

`batch_start` stores `int(state.step)` directly and resets the per-update count; the native trainer increments `self.step` before `train_step`, so this is the impending optimizer update, not a microbatch or completed-update count. `microbatch_start` stores native zero-based `state.microbatch_idx` (0–3). Arm waits only when 21 <= step <= 40. No prediction from local loop iteration, timing observer, loss values, or modulo arithmetic is acceptable. `after_optimizer_step` records host optimizer/scheduler submission completion for that update; it is not a device-wide completion measurement. `batch_end` disarms the state. Validation, fit end and close also disarm it. Global selected-call ordinals never reset; per-update counts reset at `batch_start`.

SAC recomputation and runtime routing can change call counts; log every selected invocation rather than assuming a fixed layer count or aligning ordinals across ranks. Record update and microbatch boundaries even when their selected count is zero. Missing selected calls in the active interval invalidate activation evidence. A selected call without valid native training context is an error in this diagnostic. Do not instrument fake tracing calls or infer runtime activation from schema values alone.

The installer returns an idempotent restoration action, which restores the saved helper only if the module still contains this exact adapter. Unexpected replacement is reported and not overwritten. Observer close flushes/closes its file without CUDA work and disarms state; entrypoint `finally` restores the helper even on native-main failure. Preserve the original error and report cleanup errors separately. Do not assume `fit_end` runs after a CUDA exception. Native trainer cleanup is unchanged.

## Safety and runtime-cost review

This is a host wait after the complete native call has been submitted and joined to compute. The existing one-node fused native launch submits expected-value update, combine and device-sync kernels on the supplied stream. Prior source investigation supports that their peer waits depend on the matching call, not a future call. Every rank uses the adapter and the same update window, and each rank submits its call before waiting. No earlier-rank host wait prevents a slower peer from launching its matching call. There is no new rank collective. This reasoning does not certify hidden closed-extension behavior, scheduling across different call orders, or multi-node progress; support is restricted to the exact one-node target. A hang remains a failed diagnostic, not evidence of a compute kernel fault.

Event completion includes the compute prefix plus the joined communication stream. It may report earlier asynchronous faults. It does not prove absence of silent corruption, establish an originating kernel, or fence unrelated streams. See the prior handoff's NVIDIA event-contract source. Success before a later fault narrows the executed interval only.

Default path cost is zero because the helper is only replaced by the new entrypoint. Within the diagnostic, all helper calls incur a Python adapter/selector; selected calls maintain integer counters. Updates 21–40 add one no-timing event, one record and blocking host wait, plus flushed CPU logs per selected backward call. This removes cross-boundary submission overlap and changes temporary/handle lifetime and allocator reuse. It can mask a timing race. No new tensor storage, copies, dtype conversion, communication, or computation is intended; live references persist across the wait. Compilation/fake/autograd/functionalization registrations, SAC save policy, actual native counts and route gradients remain unchanged. This run supplies no throughput, peak-memory optimization, convergence, or repair claim.

## Finite later verification

These checks are required after implementation; none was run in this preparation:

1. CPU behavior tests exercise installer pin/signature/duplicate/pairing failures and exact restore after success/native/record/wait errors. With doubles, check original is called once, join-return precedes current-stream/event-record/wait, return identity and handle/input liveness, wrong selectors pass unchanged, boundaries 20/21/40/41 and four microbatches use native counters, and inactive/default paths create no event or file. Preserve/restores global state. Explicitly check logging failure is not success.
2. One CPU dispatcher integration check uses a tiny test-only custom op whose registered original body resolves a module-global helper, with a fake implementation and registered autograd. Call through `torch.ops` eagerly and under `torch.compile(backend="aot_eager")`; replacing the helper must intercept actual runtime backward, not fake tracing, preserve output and elementwise gradient values, and restore cleanly. This validates the captured-global mechanism without claiming CUDA/SAC correctness. Do not replace the production op's device kernel to make a CPU test pass.
3. Run the checkout's scoped lint (`ruff check --select F821,F811,F841 --no-cache` on changed Python), relevant callback/config test module, whitespace/link checks and one independent integration/runtime-cost review. Inspect the implemented actual selector and no-op ordinary entrypoint; do not expand into another CUDA audit. Source-pin/read-only assertions complement, not replace, behavior tests.

One actual-target attempt, only after A006 release and normal owner authorization/preflight, closes runtime activation/localization. Freeze a new integrated source snapshot and fresh attempt directory. Reuse the reviewed scan-fix baseline launcher and verified environment/cache/data route, with fresh per-rank scan runtime/cache as required by that launcher; do not execute the historical launcher directly. Replace only its module/config/output and stop step as follows (proposed module does not exist yet):

```bash
/opt/torchtitan-venv/bin/python3.12 -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=8 --max-restarts=0 \
  --log-dir="$ATTEMPT_DIR/torchrun" --redirects=3 --tee=3 \
  -m torchtitan.experiments.cuda_graph_campaign.train_combine_bwd_completion \
  "$ATTEMPT_DIR/config/diagnostic.yaml" \
  --training.steps 40 --lr_scheduler.total_steps 30000 \
  --debug.seed 42 --checkpoint.enable false --dump_folder "$ATTEMPT_DIR/native"
```

`diagnostic.yaml` imports the exact pinned production preset and adds only `callbacks.combine_bwd_completion: {type: campaign_combine_bwd_completion, output_dir: combine_bwd_completion}`. Do not retain the incompatible 50-update timing-observer gate. Preserve graph0 via the inherited `TORCHINDUCTOR_CUDAGRAPHS=0`; preserve the scoped E4 scan runtime installation. Clear no other feature. Capture all eight ranks, actual resolved rank-zero config, same-node topology/group membership, native scan marker, adapter activation/counters, first exception and complete attempt status. Use inherited bounded job timeout and precise task-owned failure cleanup; add no barrier or in-process CUDA watchdog. Stop on first fault or after update 40; no automatic repeats or benchmark matrix.

## Runtime question and decisions

| One-attempt outcome | Decision |
| --- | --- |
| Native/helper or event checkpoint reports fault | Preserve last complete per-rank update/microbatch/ordinal; investigate that executed prefix, including preceding native MoE/custom ops, buffer lifetime and traps. Do not attribute it specifically to combine. |
| Checkpoint completes, later Flex/RoPE/pointwise path faults | Prioritize the interval after that boundary; earlier silent corruption and unrelated streams remain alternatives. |
| No fault through update 40 with every-rank activation | Report nonreproduction under timing perturbation; race masking remains open. No fix, nondeterminism-elimination or speed claim. |
| Missing activation, source/selector mismatch, counter error or peer hang | Attempt is inconclusive; preserve evidence and resolve that concrete instrumentation/progress failure before choosing any separately authorized next run. |

Required open gates: implementation and independent review, CPU mechanism checks, integrated source pin/launch preflight, actual target activation and peer progress, then one outcome classification. Originating PC/kernel, fault mechanism, exact numerical effects of altered scheduling and sustained production correctness remain unresolved. This preparation does not authorize scheduling or broader fixes.
