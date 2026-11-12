# Native profiler schedule preparation

## Acceptance and result

Finite acceptance: identify the active native profiler path and exposed controls; map one profiler interval to one full optimizer update; select a post-startup update from existing successful logs; specify one-export termination and output paths; identify unmet controls without launching GPU work or changing source.

The existing native schedule can capture exactly update 111 from a fresh run and export once. Existing native configuration cannot select global ranks or disable shapes: `record_shapes=True` is hardcoded. The coordinating agent selected a narrow launcher-local adaptation outside `gift/` to supply these controls; core/schema/defaults remain unchanged. No source change or GPU launch was performed for this report.

## Existing path and scope

- Revision inspected: `446431dfa191ef0300f116257ae918bc85e38c06`.
- Accepted launcher: `BASELINE-STABILITY/attempts/004/launch.sh`; entrypoint `torchtitan.experiments.cuda_graph_campaign.train_scan_fix` installs only the accepted scan correction and calls native `torchtitan.train.main`.
- Native `gift/torchtitan/trainer.py:924-964` builds `config.profiler` before the loop and calls `profiler.step()` once after `train_step`, checkpoint handling, and validation. It does not call it per microbatch.
- `Trainer.train_step` at lines 719-827 fetches all accumulation microbatches, runs each complete forward/backward, then clips gradients, waits for any checkpoint staging, steps the optimizer and LR scheduler, and finishes callbacks. Therefore an active interval includes all four accumulation microbatches and their communication, clipping and optimizer work. Prior attempt 004 rank-0 stdout line 182 confirms accumulation count 4, local batch 8, global batch 256, sequence 4096.
- Checkpoint and validation remain disabled as in the accepted diagnostic baseline; profiling does not replace training or require campaign trainer/timing observers.

## Existing settings for the selected interval

Retain accepted attempt 004 environment, all eight workers, descriptor-fix-001 dependency precedence, native entrypoint and exact preset. Replace only diagnostic stop/profiler/output controls:

```text
--training.steps 111
--lr_scheduler.total_steps 30000
--debug.seed 42
--checkpoint.enable false
--validator.enable false
--profiler.enable_profiling true
--profiler.enable_memory_snapshot false
--profiler.profile_freq 111
--profiler.profiler_warmup 1
--profiler.profiler_active 1
--profiler.profiler_repeat 1
--profiler.profiler_skip_first 0
--profiler.save_traces_folder profiling/traces
--dump_folder <fresh-attempt>/native
```

The launcher explicitly sets `--profiler.profiler_skip_first_wait 0`, preserving the initial wait interval. The schema types this control as an optional integer. Keep `TORCHINDUCTOR_CUDAGRAPHS=0`; it disables Inductor graph capture, not Kineto CUDA activity. Keep accepted `TORCHINDUCTOR_MIX_ORDER_REDUCTION=0` and `PYTORCH_ALLOC_CONF=expandable_segments:True`. No new profiler environment variable is required by the native source. Do not limit `CUDA_VISIBLE_DEVICES` to two GPUs: all eight ranks must train.

For a fresh native run, initial profiler step number is zero. Native code computes wait as `profile_freq - (active + warmup)`, hence 109. Expected schedule: updates 1-109 wait, update 110 profiler warmup, update 111 active and saved. After update 111, the existing final `profiler.step()` closes the active interval, calls the export callback with step number 111, and repeat=1 prevents a second recording cycle. The native loop exits at 111 and the profiler context closes. The coordinating agent reports that the actual `/opt/torchtitan-venv` PyTorch schedule implementation/signature has been inspected and matches this mapping.

Expected outputs after the missing controls are resolved:

```text
<fresh-attempt>/native/profiling/traces/iteration_111/rank0_trace.json
<fresh-attempt>/native/profiling/traces/iteration_111/rank1_trace.json
```

Profiler source builds CPU activity unconditionally and CUDA activity when CUDA is available. Output names use `torch.distributed.get_rank()` (global rank), not local rank. In the existing implementation all eight enabled ranks construct the profiler and export; the above two-file output is not achievable by these flags alone.

## Why update 111

Existing successful native200 attempts 003 and 004 show model/loss compilation announcements and quack in-process fallback compilation before the first logged update. No later compilation marker was found in rank-0 stdout. This supports excluding startup, but does not prove any unique compilation-completion threshold.

The displayed logs provide an additional conservative warmup criterion. In both runs updates 10-70 have roughly 32.4-34.1 seconds between ten-update log messages; update 50 reports further peak-memory growth. The windows ending at 90, 100, 110 and 120 have roughly 29.9-30.1 seconds per ten updates and flat reported peak memory. In attempt 004 these timestamps are 16:42:15.940, 16:42:45.864, 16:43:15.907, and 16:43:45.904; in attempt 003 they are 16:29:22.490, 16:29:52.407, 16:30:22.442, and 16:30:52.427.

Update 111 is a conservative selection inside this observed region, with profiler warmup on 110. Update 51 would be after the explicit initial compilation messages, but inside the earlier changing timing/memory region. The inference does not establish that 111 is uniquely required, that later dynamics cannot change, or that profiling reproduces unprofiled timing. These timestamps select a diagnostic interval; they are not a throughput benchmark or speedup claim.

Evidence is in `BASELINE-STABILITY/attempts/{003,004}/torchrun/*/attempt_0/0/stdout.log`, with the original logs and configuration retained. Update 111 is also the run's final update; checkpointing is disabled, so native final-step checkpoint handling returns without saving.

## Missing controls and minimal integration recommendation

Native `gift/torchtitan/config/schema/tools/profiler.py` exposes enable, folders, schedule and memory snapshot controls only. `gift/torchtitan/tools/profiler.py:199-266` has no rank selection and passes `record_shapes=True`. It does not pass stack or memory recording arguments to `torch.profiler.profile`; their actual pinned defaults should be verified by parent before launch. The separate memory-snapshot mechanism is explicitly disabled above.

The coordinating agent selected a launcher-local adapter outside `gift/`, using the existing native builder and PyTorch profile constructor. No generic native config extension, schema/default change, golden re-pin, copied profiler, or new trainer is required.

The adapter wraps native `Profiler.build_torch_profiler`: return `None` for global ranks 2-7; on global ranks 0 and 1 invoke the original builder while a scoped `unittest.mock.patch` redirects `torch.profiler.profile` to the captured real constructor with `record_shapes=False`, `with_stack=False`, and `profile_memory=False`. The patch exits immediately when the native builder returns, restoring the constructor before training. The returned object is the actual native PyTorch profiler; schedule, callback, output path, profiler step calls and teardown stay native. The coordinating agent reports that the pinned constructor defaults for stacks/memory are false; making all three explicit prevents reliance on those defaults.

Boundary review: capture the real constructor before patching to avoid recursion; guarantee restoration on normal return and exceptions with the patch context; forward all original constructor arguments except the three deliberate overrides; preserve disabled profiling behavior; gate using `torch.distributed.get_rank()` after distributed initialization, never local rank or the visible-device list. The patch is process-local and scoped to synchronous construction before the training loop. If native construction raises, propagate the error and restore the constructor rather than falling back to another profiler. Keep this diagnostic adapter outside training-source snapshots and visibly record its exact override identity because native resolved configuration cannot express rank/shape controls.

Expected runtime impact is one global-rank selection during profiler construction and three constructor option overrides; no extra per-step math, communication, transfer, checkpoint state, model changes or optimizer work. Shape instrumentation is disabled only for this diagnostic. Excluded ranks keep the native loop with no torch profiler handle. Profiling still perturbs CPU submission and rank timing; unprofiled peers can wait on the two profiled ranks. No timing magnitude or speedup claim follows.

Parent owns adapter assets and focused CPU behavior verification: excluded ranks must never call the original constructor, selected ranks must call the captured real constructor once with the required flags, and the patch must restore the original function after success and failure. The actual launch must show adapter activation outside resolved config and prove the same selected interval and two-file export.

## Existing test coverage and remaining gates

`gift/tests/unit_tests/test_profiler.py` checks defaults, disabled lifecycle, active argument storage and a basic enabled handle with a mocked rank. It does not verify requested rank gating, shape-off construction, exact schedule-to-update mapping, or one-export rank selection. Those controls are supplied by the selected launch-local adapter, not native schema changes. The integration feature list only enables profiling; it is not evidence for this exact request. No tests were executed by this read-only investigation.

Pinned PyTorch schedule/default inspection was closed by the coordinating agent; parent must close: adapter behavior checks, exact resolved launch settings, clean completion on all eight ranks, exactly one CPU+CUDA trace each for global ranks 0 and 1 at the same update, and no trace for ranks 2-7. Trace content and run activation proof remain GPU execution gates.

## Source pins

SHA-256 at inspection:

```text
bccac76138940b462d1b945b88a6e4de9f521319f0a91d87bd8383c65e37385d  gift/torchtitan/tools/profiler.py
82d888a2fb2d1d465f4cf89dc0502eaee84b722adaa8ae0998f3f0b63b64e0c4  gift/torchtitan/config/schema/tools/profiler.py
6c8be0d163d79b34f239b40087d0ec1a579c7ed4a01444501097f557522c1f8c  gift/torchtitan/trainer.py
```

Documentation-only output has no production runtime impact. This report has no Markdown links requiring local target checks; whitespace was checked after writing.
