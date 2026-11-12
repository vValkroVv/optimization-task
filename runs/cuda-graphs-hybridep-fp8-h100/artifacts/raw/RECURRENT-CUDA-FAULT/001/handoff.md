# Recurrent native CUDA fault: bounded diagnosis

## Result and acceptance

The corrected graph-off production path still fails with CUDA illegal instruction. The first material report is rank 6 in compiled backward at update 33, before the NCCL watchdog report. No originating kernel, lifetime defect, or additional version-matched source correction is established. The selected next diagnostic is one bounded completion checkpoint after each HybridEP `combine_bwd` return, to divide the native MoE backward region from the following attention/RoPE region.

Acceptance checklist: (1) identify first error and actual phase; (2) reconcile corrected source and relevant binary provenance; (3) inspect concrete trap/fix interpretations; (4) choose one native-path diagnostic with outcomes, costs, safety conditions and remaining gates. All four read-only items are complete. No diagnostic was implemented or executed.

Scope: frozen `SOURCE-SNAPSHOT/002`, repository HEAD `446431dfa191ef0300f116257ae918bc85e38c06`, manifest SHA-256 `3e2077942276bdcdb9f57be12d173aed21aab0d556b16f46e1f1a5d195c1a4b8`, preset SHA-256 `b341701bd1eef7774c110fbf7ed69cf6280ba8691100b05be457dee3e7d50a7b`. Source and preset postfault guards match. Later graph marker changes were absent during this run and are not fault evidence. All writes here are campaign artifacts; no CUDA initialization/query/execution, source/package edits, installs or resets occurred.

## First material evidence

- `SCAN-FIX-BASELINE-AA/attempts/001/native/structured_logs/` records 32 `optim_end` events on every rank. Update 33 has one `fwd_bwd_start` and fails during its first microbatch; this is not an 8-update run or microbatch number 33. `phase-summary.json` preserves the eight-rank summary.
- Rank 6, PID 493793: `fwd_bwd_error` at **22:49:52.506248 UTC**, then watchdog `CUDAEvent::query` at **22:49:52.681518 UTC**. The Python/driver error explicitly says `Sticky error detected`, `Returning 715 (CUDA_ERROR_ILLEGAL_INSTRUCTION) from cuLaunchKernel`.
- Elastic `attempt_0/6/error.json` provides the complete stack, copied to `first-error-stack.txt`: `loss.backward()` → AOTAutograd compiled backward → cached `chkdyyctwia2rt54e367cu7drsuuliiqgwpyl7lsu3gvmuuzsf2d.py:4911` → `triton_poi_fused__to_copy_cat_slice_transpose_view_34.run`. This is where an already sticky device fault was noticed, not proof that this pointwise kernel caused it. Its `raw_stream6` means device index 6; it does not establish stream ID 6.
- The exact cached callable, named in the stack, contains HybridEP `dispatch_bwd` at 4386, FP8 m/k-grouped GEMMs at 4425/4431/4466/4496/4529, HybridEP `combine_bwd` at 4537, FlexAttention backward at 4793, then the reported pointwise launch at 4911. SAC recomputation also occurs in this compiled path. Earlier asynchronously submitted work remains possible.
- Owner-produced `postfault-gpu-state.txt` reports all eight devices idle, no compute applications, ECC/remap counts zero and no pending/remap failure. Precise PID cleanup is recorded separately. These observations do not prove hardware or driver correctness.

The run preserved 3B MoE, HybridEP EP8/FSDP2, FP8, selective AC, model/loss compile, FlexAttention, local/global batch 8/256, S4096, accumulation 4 and scheduler horizon 30000; graphs/profiling were off. It exited before the required completed 50-update timing window, so no A/A variability or throughput result exists. Independent-run bitwise loss/norm equality is not a gate and is outside this investigation.

## Corrected provenance and trap interpretation

Rank 6's runtime manifest identifies the corrected four-expert bound, wrapper bytes unchanged, isolated per-rank runtime/cache, and PID 493793. Its E4 scan binary SHA-256 is `3684ba437c4b3163f393a73af2dc576540c3c67bab1e2130ed589ca9f340de8d`. CPU-only `cuobjdump` found `hybrid_ep::scan<256,108,128,64,8,1,4>`; its complete function SASS matches corrected smoke rank 2. No `TRAP`, `BPT`, `BRKPT` or `__assert` token appears in that function. See `rank6-scan.sass.txt`, `scan-comparison.json` and `receipts.json`. This verifies corrected scan code provenance more strongly than header presence; it does not clear dispatch, combine, DeepGEMM, or prior corruption.

The pinned HybridEP header has no explicit trap, timeout or device-printf path. Runtime `assert(status...)` sites are in NIXL/inter-node sections, not an evidenced explanation for this eight-rank/one-node specialization. There is no preceding device assertion/timeout text in the rank logs. Missing printf alone cannot rule out traps.

Installed DeepGEMM remains `2.3.0+f093a4c`, exact header hashes recorded. `common/utils.cuh` defines printf-plus-trap and silent `DG_TRAP_ONLY_DEVICE_ASSERT`. Concrete direct sites inspected: SM90 architecture fallback; M-grouped `shape_n % 8 == 0` (target outputs 1536/1024/2048 satisfy this); scheduler unreachable branch (MGroupedContiguous has its own return); shared-memory descriptor alignment assertion. None supplies evidence of a fired assertion. Inlined CUTLASS instructions and exact DeepGEMM JIT binaries were not reconstructed. An intentional trap remains one possible interpretation of error 715, not a diagnosis. [NVIDIA's PTX trap contract](https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions-trap) describes execution abort and host notification.

Prior primary research remains applicable: DeepEP `17cfb817` is the corrected scan bound, not proof of the original cause; DeepEP #756's standalone host-count path is not the enabled fused path; DeepGEMM `1f6f3f3` has not been shown to fit the installed fork's two-descriptor staging. CUDA compatibility loader evidence does not establish incompatibility. No blanket upgrade or additional transplant is supported.

## One selected next diagnostic

**Add an off-by-default, graph-off diagnostic completion checkpoint immediately after `_run_on_comm_stream(...)` returns in `hybridep._combine_bwd_impl`, before returning its output. Enable it on all eight ranks only for updates 21–40 in one native diagnostic attempt.** Record/flush the update and monotonically increasing custom-op ordinal before the wait, then record a no-timing CUDA event on the actual current compute stream and synchronize that event; flush completion or the exact exception. Keep outputs and handle live; do not introduce a rank barrier, device-wide wait, callback on a CUDA stream, or a wait inside the native communication launch.

Why this boundary: frozen `hybridep.py:102–116` orders comm after compute, launches the complete native call, and joins comm back onto compute. Thus the event includes that call's communication. The actual one-node native header launches the corresponding fused combine and device-sync kernels on its supplied stream (`:5817–5840`); its intra-node flag waits depend on peer work in that same corresponding call. There is no NVSHMEM invocation in this backend header; multi-node launch code is conditional on node count. If all ranks submit their matching call before waiting, this checkpoint does not withhold a later peer kernel needed by that call. This is source-supported progress reasoning, not runtime certification of the closed extension. Preserve and review that same-call ordering during implementation; do not extend this design to multi-node paths or add pre-call waits.

[NVIDIA's event contract](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html) states that event completion covers captured stream work and may report prior asynchronous errors. Successful completion narrows the executed prefix; it cannot rule out earlier silent memory corruption or an unrelated stream's fault.

Decisions from the one attempt:

1. Failure at the checkpoint: retain last completed ordinal; investigate the preceding native MoE/custom-op interval, including traps and buffer lifetimes. It still does not assign causality to combine itself.
2. The relevant checkpoint completes, then the later Flex/RoPE region reports the sticky fault: prioritize that later executed interval and its live/reused buffers, preserving earlier corruption as an unresolved alternative.
3. No fault through update 40: timing perturbation may have hidden a race. Keep causality open; this is not a repair, throughput result, or reason to disable strict diagnostics.

Costs: one event and host wait per instrumented MoE backward call plus flushed logs; overlap and allocation/lifetime timing change, and outputs stay live across the wait. This can mask races and is unsuitable for performance measurement. All production settings, native kernels, precision, AC, compilation, data and FlexAttention stay unchanged. Implementation requires separate source authorization/review and a new source snapshot; execution requires the owner's normal GPU release/preflight. Stop at a first fault or update 40. No blind `CUDA_LAUNCH_BLOCKING`, fixture replacement, package patch or graph-negative conclusion is justified.

## Evidence limits

This diagnosis establishes recurrence, phase and corrected scan provenance. Originating PC/kernel, exact fault mechanism, sustained native completion and valid A/A noise remain open. GPU evidence already recorded by the execution owner is reused; this investigation adds no GPU evidence. No new repair is claimed.

## Verification note

The initial artifact check incorrectly required empty cuobjdump stderr and failed. cuobjdump exited 0 but emitted a `Function listed in --function ... not found` warning, preserved verbatim in `rank6-scan.sass.stderr.txt`. Its stdout nevertheless contains exactly one function with the requested mangled E4 scan name and the full instruction text matches the earlier corrected-smoke scan. The warning's cause is not established; binary conclusions use that explicit matching function output, not empty stderr or exit code alone. JSON parsing, Markdown whitespace/link classification, eight-rank phase counts and the targeted function-content check passed after accounting for this preserved warning.
