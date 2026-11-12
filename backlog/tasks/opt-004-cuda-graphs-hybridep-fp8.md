# OPT-004 — Stabilize the native HybridEP FP8 baseline

**Status:** Complete for the bounded native baseline stability target. The isolated DeepGEMM descriptor-protocol repair passes the affected-kernel reproducer and both fresh graph-off native runs: each completes 200 optimizer updates and 800 microbatches on all eight ranks, with finite metrics, intact logs and clean teardown. CUDA Graph/performance research remains paused.

## Diagnostic trace follow-up — 12 November 2026

**Status: Complete — PASS.** One diagnostic run reused the completed two-run stability result and its accepted repaired launcher. All eight GPUs trained with the same graph-off preset and scheduler; the native profiler captured CPU and CUDA activity only on global ranks 0 and 1 for optimizer update **111**, then the run ended cleanly at 17:31:15 UTC. Updates 1–109 wait, update 110 warms the profiler, and update 111 is its sole active update. The learning-rate warmup remains the supplied 2,000 updates; this short run does not finish that schedule warmup.

The native profiler hardcodes shape recording and exposes no rank filter. A small launch-local constructor adapter, outside GIFT, selects ranks 0–1 and sets shape recording, stacks and memory profiling to false. Native scheduling, lifecycle and Chrome export remain in use; GIFT and shared dependencies stay unchanged. Profiling perturbs this diagnostic run; it is not a throughput benchmark or another stability campaign. [Schedule and boundary review](../../runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-TRACE/preparation/001/REPORT.md).

Finite acceptance:

- [x] Verify the current eight-H100 NV18 allocation, accepted source/dependency identities and frozen training settings.
- [x] Verify the native schedule advances once per complete optimizer update, with all four accumulation microbatches, communication, clipping and optimizer inside the selected interval.
- [x] Export exactly one readable CPU/CUDA Chrome/Kineto trace each for global ranks 0 and 1, both for update 111, with shape/stack/memory recording off and no traces on ranks 2–7.
- [x] Confirm all eight ranks finish the short run, finite native metrics, intact logs, no training error and clean process/GPU teardown.
- [x] Record the exact command, source/dependency identities, artifact paths and capture limits in the campaign report.

Result: all eight ranks completed **111 optimizer updates and 444 microbatches** with finite logged metrics and no training errors. Exactly two readable traces contain the full selected update; shapes, stacks and memory recording are off. [Independent trace and lifecycle review](../../runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-TRACE/review/003/REPORT.md), [root process/GPU/source checks](../../runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-TRACE/attempts/003/root-post-run-release.json), and [artifact paths and executed command](../../runs/cuda-graphs-hybridep-fp8-h100/report.md). No production source, shared package, checkpoint or benchmark was added.

## Governing scope

Establish stable native, CUDA-Graph-off execution on the exact supplied 3B MoE / HybridEP / FP8 preset. The accepted source inventory selects the native `train_scan_fix` entry with the pinned HybridEP scan correction **included**, the existing CudaCherry graph-safety custom-op wrapper **retained**, and `TORCHINDUCTOR_MIX_ORDER_REDUCTION=0`. The CudaCherry wrapper remains active with graphs off. The repaired verification launcher prepends the isolated `descriptor-fix-001` dependency copy before GIFT at process start; the package/header manifests pin its only source change. Use standard native logs; bypass the observer overlay and optional timing, phase-memory, step-marker, and all-18-block graph-exclusion overrides using existing controls, while preserving all source files.

The [accepted source inventory](../../runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY/source-inventory/001/REPORT.md) verifies the Source010 manifest for all 98 listed files and the exact preset hash. It identifies the baseline and a native 200-update launch payload; native diagnostic attempt 002 ran from 15:30:55 to 15:34:05 UTC with a fixed 200-update stop and all-rank exception capture. It completed 35 updates on all ranks, then faulted during update 36. Post-fix native runs003 and004 both pass the full 200-update gate; [final independent review](../../runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY/review/native200-001/REPORT.md) accepts the bounded result. The [existing-tool smoke result](../../runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY/tool-smoke/001/result.json) confirms CUDA exception capture and essential kernel/PC decoding on a one-thread synthetic trap. The initial full raw disassembly failed because required memory was omitted from the core; its [decode log](../../runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY/tool-smoke/001/decode.log) is preserved. The reduced essential decode passed. This validates tool capability only; it does not localize the workload's first fault.

Use existing fault-localization tools with verified relevant rank/kernel coverage to identify the first training fault at a kernel program counter or produce a narrow reproducer. Do not repeat the earlier attempt with patching disabled, an exhausted report limit, and unchanged sanitizer settings. Use narrowly scoped diagnostic source changes if needed; treat added synchronization as diagnostic unless it restores a demonstrated missing dependency. Once localized, make only the smallest causal fix in task-local `gift/` or an isolated copy of the task dependency. Shared packages remain read-only; do not update or install packages or apply a broad synchronization fix.

## Frozen workload

Use `gift/configs/experiments/gift_moe_3b_hybridep_fp8_compile.yaml` unchanged: MoE / HybridEP EP8; dense and sparse FP8; Flex attention; model and loss compilation; selective activation checkpointing; chunked cross-entropy; local/global batch 8/256; sequence length 4096; four accumulation microbatches; and the configured model, `c4_test` data, optimizer, and schedule (2,000 warmup steps and 30,000-step horizon). Preserve all supplied settings. Each acceptance run is limited to 200 optimizer updates while keeping the configured 30,000-step scheduler horizon; use the same frozen config and launch settings in both fresh processes.

The owner confirmed the server changed with the same topology. [Reconciliation 002](../../runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY/reconciliation/002/receipt.json) verifies the current eight idle H100s, NV18 links, unchanged runtime and source. The new GPU UUIDs are recorded; prior-server results are retained separately.

## Finite acceptance checklist

- [x] Inventory the exact graph-off active path against the native Trainer, choose standard logging, and identify which optional instrumentation is bypassed while source is preserved.
- [x] Freeze the graph-off baseline: include the pinned HybridEP scan correction, retain the CudaCherry graph-safety wrapper, and keep mixed-order reduction disabled.
- [x] Localize the captured device fault with all-rank capture enabled. Attempt 002 completes 35 updates on every rank and faults during update 36: rank 3 raises Warp Illegal Instruction Parameter at DeepGEMM `UTMALDG.2D.MULTICAST`, triggered PC `0x7fd81aff1cc0` (function +20416). [Triggered-PC decode](../../runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY/analysis/002/decode-trigger.log) is retained. Faulting-instruction attribution does not establish the cause.
- [x] Apply a narrowly justified fix in an isolated task dependency copy and rerun the affected reproducer: [descriptor repair](../../runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY/dependencies/descriptor-fix-001/REPORT.md), [independent review](../../runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY/review/descriptor-fix-001/REPORT.md), and [candidate fixture004](../../runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY/reproducer/004/root-result.json). All 2,112 outputs pass; the identical original cold-cache fixture fails at call2. Exact native corrupt descriptor bytes were not captured; both required native stability runs pass below.
- [x] Run **two fresh processes**, each native and graph-off for **200 optimizer updates on all eight ranks** (**800 microbatch updates per rank**), using the same frozen configuration and launch settings. [Run003](../../runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY/analysis/003/REPORT.md) and [run004](../../runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY/analysis/004/REPORT.md) pass, using identical launch bytes and resolved configs except output path.
- [x] For both runs, confirm finite native loss and norm values, intact per-rank logs, all expected updates with no skips, no CUDA fault, hang, OOM, silent fallback, or incomplete execution, and clean process teardown. [Final root release](../../runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY/attempts/004/root-post-pass-release.json) also verifies all eight GPUs idle and all 98 Source010, 844 original and 844 candidate file hashes unchanged. No task monitor remains.

Do not mark this task complete if fault localization, the causal fix, the affected-path rerun, or either 200-update run is missing or fails. Preserve failures and keep the status active.

## Verdict and limits

The repaired native path meets the requested two-run stability target. Premature descriptor reuse is demonstrated in the original exact-SASS fixture; the repair restores descriptor-read completion and publication ordering. The exact corrupt descriptor bytes in the native TMA-B exception were not captured. This is bounded stability evidence, with no claim of unlimited reliability, convergence, numerical parity or performance. All failed attempts and superseded findings remain preserved. No new checkpoint was created.

## Explicitly paused or stopped

There is no bitwise-equality gate. CUDA Graph work and performance comparisons are paused. Checkpoint continuation, RNG/full-state restoration, tensor-recording/numerical frameworks, benchmarks, and broader framework work are stopped. Do not extend those systems or claim a performance, convergence, or production-readiness result from this stability task.

The prior performance-only task and report were preserved byte-for-byte before this scope update. Their hashes and archive paths are in the [scope receipt](../../runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY/scope/001/receipt.json); see the [prior task snapshot](../../runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY/scope/001/prior-opt-004-task.md) and [prior report snapshot](../../runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY/scope/001/prior-report.md). The historical record below is superseded in full: its former acceptance criteria, status, launch plan, and authorization text are archival only.


---

## Historical performance-assessment record — superseded

The following task text is retained as the previous record. Nothing below changes or extends the governing stability scope above.

# OPT-004 — Assess the existing loss-only CUDA Graph candidate

**Status:** Complete as a scoped performance assessment: **measurement blocked**. The second graph-off control completed 33 optimizer updates on every rank, then failed during update 34 with a CUDA illegal instruction. Its fixed 21–40 window did not finish; graph-on runs were not released. A/A variability, graph throughput benefit and paired GPU-memory cost remain unmeasured. See the [campaign report](../../runs/cuda-graphs-hybridep-fp8-h100/report.md) and [reproducible short assessment](../../runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/PERFORMANCE-ASSESSMENT/analysis/001/REPORT.md).

## Current acceptance contract

This contract supersedes the earlier full research, numerical and lifecycle gates. Stop RNG continuation, checkpoint/resume, full-state restoration, tensor recording, detailed gradient/update comparison and long-run acceptance work. Preserve their completed evidence and stopped partial edits; do not finish or extend those systems. Independent-run bitwise equality is not required. No convergence or production-readiness claim is requested.

Use the existing A008 loss-only candidate. All eighteen model blocks remain compiled with selective activation checkpointing but are excluded from CUDA Graphs. Eligible capture covers chunked cross-entropy forward/backward and FlexAttention block-mask creation. Model/head GEMMs, MoE communication and optimizer execution remain outside graphs, inside the integrated training timer. Reuse the [all-rank A008 replay proof](../../runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/LANE-A/analysis/008/mechanism-review.md) and [copy-cost review](../../runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/LANE-A/analysis/008-copy-cost/report.md), with their actual tested coverage and limits.

Both graph-off and graph-on must use identical source, backend fixes, non-graph compiler settings, FlexAttention, data, batch sizes, precision, HybridEP, MoE, selective activation checkpointing, optimizer and schedule. Keep common `mix_order_reduction=false`, graph partitioning enabled and all eighteen model graph exclusions. Only compiler graph activation differs. No additional kernel fusion or backend replacement is required.

## Finite execution and acceptance checklist

These are dispositions of the reduced task. A blocked measurement is not a passing comparison.

- [x] Reuse verified upstream provenance, prior replay/copy evidence and every failed result. [Source010](../../runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/SOURCE-SNAPSHOT/010/manifest.json) preserves stopped numerical edits; [active-path equality](../../runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/PERFORMANCE-SCOPE/001/source-active-equality.json) reconciles them against A008.
- [x] Define the finite matrix and timer/memory contract before launch; [independent scoped review](../../runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/PERFORMANCE-ASSESSMENT/review/001/REVIEW.md) and [root integration review](../../runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/PERFORMANCE-ASSESSMENT/analysis/001/root-integration-review.json) are complete.
- [x] Inspect actual 006/007 resolved configurations, activation and all-rank compiler receipts: native Trainer, identical workload except output path, graphs off, model/loss compile, FlexAttention, FP8, HybridEP, selective AC and optimizer retained. No numerical/continuation callbacks, checkpoint saves, evaluation or profiling.
- [x] [Fresh preflight](../../runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/PERFORMANCE-ASSESSMENT/preflight/001/receipt.json) verifies the exact authorized eight idle H100/SM90 UUIDs, NV18 links, pinned runtime, Source010 bytes and NFS4/fsync. Recheck ownership before each launch; final release is verified.
- [x] Attempt serial off-AA1/off-AA2/on-B1/off-A2/on-B2, sharing AA2 between noise and alternating comparisons. Preserve initial pre-native harness failure 001 and unrun templates 002–005. Corrected 006 passes; 007 CUDA failure triggers the stop rule. 008–010 are **NOT RUN**.
- [x] Keep every window fixed at optimizer updates 21–40, twenty updates after predeclared warmup; native 41 closes the timer before fetch. 006 completes the full window. 007 completes 33 updates and fails during 34; no partial or selectively shortened result is substituted.
- [x] Check all-rank optimizer/microbatch completions, actual token counts, finite available native rank-zero metrics and intact logs. Reuse actual A008 replay proof at 21–23; new timed-window replay/recapture is unobserved, with no graph-on run after the fault.
- [x] Report single-control throughput, completed mean latency, setup and memory. **A/A variability, paired off/on throughput and graph GPU-memory cost are unavailable**, not zero or a negative result.
- [x] Stop at the actual CUDA fault, retain all failures and unrun variants, and finish without another fault-debugging framework, numerical system or long run.
- [x] Finish the scoped review, task/report updates and document checks. Verdict: **measurement blocked**, with no production-readiness, convergence or graph-incompatibility conclusion.

The timer includes fetching, accumulation, forward/loss/backward, clipping, optimizer, scheduler and recurring native metrics. Window tokens are 20 × 1,048,576 = 20,971,520 already-global useful positions; do not sum the duplicated global scalar across ranks. Throughput uses the maximum device-completed window wall time across ranks. Window wall time / 20 is a completed mean update latency. Host submission spans and current-stream CUDA-event spans are separate scopes and do not establish complete-update p50/p99.

Use identical accepted timing and phase-memory observers on both variants. Native allocator resets after updates 20, 30 and 40 define the 21–30 and 31–40 intervals; their maximum is the 21–40 peak envelope. Keep boundary-tail samples separate. Sampled NVML maxima are lower bounds on true device peaks, not exact capture-only memory. Record setup from process/native startup through first update and window opening with explicit clocks.

## Scope and frozen workload

The sole target is the task-local configuration [gift_moe_3b_hybridep_fp8_compile.yaml](../../gift/configs/experiments/gift_moe_3b_hybridep_fp8_compile.yaml). Its bytes match the supplied [sft-task/gift reference](../../../sft-task/gift/configs/experiments/gift_moe_3b_hybridep_fp8_compile.yaml), with SHA-256 `b341701bd1eef7774c110fbf7ed69cf6280ba8691100b05be457dee3e7d50a7b`. GIFT was refreshed exclusively from upstream `main` at `6fd9a3230ab6361f11bbbf0b6ec5a5f125b78fc2`, replacing `f840cf25e4294f93a5c91a20707c30b97e3c1dbf`; see the [verified refresh receipt](../../additional/baseline/gift-main-2026-11-11.md) and its recursive dependency pins. No feature-branch or original working-tree changes were imported. Keep the flattened source layout on the task repository's `main` branch. Before execution, reconcile this pin and any subsequent task-owned implementation diff without silently refreshing the baseline mid-comparison.

Keep the supplied model, data, optimizer, parallelism, precision, compile, activation-checkpointing, and schedule settings unchanged in every baseline and candidate comparison. The named target is a 3B-class MoE: model dimension 1536, vocabulary 102400, one dense layer and 17 MoE layers, 32 experts with top-4 softmax routing, expert hidden dimension 1024, shared hidden dimension 2048, and router auxiliary loss coefficient 0.0001. It uses the existing MLA Flex attention with YaRN, HybridEP with `pad_multiple=128` and `comm_stream=true`, FP8 for dense `w1/w2/w3/wo` and the shared quantizer plus sparse grouped FP8, selective activation checkpointing, model and loss compilation, chunked cross-entropy, local batch 8 / global batch 256, sequence length 4096, gradient clipping at 1, AdamW learning rate `4e-4`, betas `0.9/0.95`, epsilon `1e-20`, weight decay `4e-5`, 2,000-step warmup and 30,000-step horizon, and `hf_text` on `c4_test` with the configured local tokenizer. Preserve the recipe's mixed BF16/FP32 internal arithmetic; it is part of this FP8 target, not an alternate precision experiment.

The expected topology is EP=8 over the eight H100s, with dense FSDP=8, expert FSDP=1, DP batch 8, and accumulation 4; PP/TP/CP remain 1. Treat these as expectations to confirm in the rank-zero resolved configuration, process-group membership on every rank, and physical topology, not as proof that the runtime formed those groups. The target uses default FSDP2, not GFSDP2. Do not shrink the model, change batch or sequence length, substitute dense/BF16/non-EP configurations, disable activation checkpointing or compilation, or replace the loss, dispatcher, fusion, or FP8 recipe to make capture easier. Small diagnostic fixtures are permitted only when they exercise the actual target MoE/HybridEP/FP8 implementation and cannot replace full-workload forward/backward/update evidence.

## Preserved evidence and stopped work

The [pre-narrowing task and report](../../runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/PERFORMANCE-SCOPE/001/receipt.json) preserve the previous plan verbatim. The report retains original/corrected strict A/A diagnostics, faults at native updates 22, 33 and 48, A007 repeated recording, A008 persistent replay, the separate C1 ENOSPC/logging failure, bounded diagnostic/sanitizer outcomes and numerical-readiness failure. None is silently discarded or converted to a pass.

The prior 50-update loss-only control completed 47 updates and faulted during 48. It is not a valid 50-update result and does not replace the new matched twenty-update matrix. Prior A007 short timings tested dense+loss capture and have invalid/incomplete control logs; they cannot establish loss-only benefit. A008 traces establish mechanism at updates 21–23, not unprofiled throughput over 21–40 or long-run reliability.

Numerical readiness attempt 001 completed one native optimizer update and checkpoint save, then failed on complex64 inventory. Observe was not run. Preparation 002, the codec follow-up, full-state/tensor comparison, RNG and continuation work are stopped by the owner. Their old acceptance gates are superseded. The step-1 checkpoint has no remaining required consumer under this scope. It disappeared during the retention worker’s read-only hash pass before a durable complete inventory or metadata copy was written; root independently confirms absence at 09:40:34 UTC. Neither issued an unlink. The [changed-state receipt](../../runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/PERFORMANCE-SCOPE/001/checkpoint-retention/changed-state-receipt.md) preserves the prior audit, partial observed hashes and incomplete retention record; the cause is unknown and no successful deletion is claimed.

## Authorization boundary

Setup update, 2026-11-11: the owner additionally authorized committing and pushing all prepared changes to `optimization-task/main` and cloning that repository into the personal SR008 workspace. This publication and checkout setup does not start the research campaign or allocate its eight-H100 execution resources.

Campaign activation, 2026-11-11: the owner started the launch goal, authorized justified candidates and scoped fixes inside this repository's `gift/` tree, and confirmed the current local host and installed Python as the execution route. Preflight verified eight idle H100 80GB SM90 devices with NV18 connectivity and the pinned Python/PyTorch build. The source scope remains the task-local `optimization-task/gift/` snapshot only: CUDA Graph experiment plumbing and narrowly necessary fixes preserving the target features. Never edit the original workspace `gift/` checkout. This does not authorize package installation, a new allocation, additional remote/data access, or future campaign publication, merge, rebase, and cleanup of shared state. Record each exact launch and its environment in the campaign evidence; do not invent missing route details.

Retention update: the owner requested keeping disk use tight and deleting experiment checkpoints when no longer needed. Remove only task-owned campaign checkpoint payloads after every required comparison or diagnosis that needs them has finished; retain path/size/hash deletion inventories, configuration, commands, logs and failed results. Preserve needed continuation states and other contributors' files.

## Short protocol results and stop decision

| Attempt | Result |
| --- | --- |
| 001 off-AA1 | 09:53:36–09:53:48 UTC, before native training: required phase-memory identity omitted. Zero updates. Failure and original frozen assets preserved; all eleven recorded identities absent. |
| 002–005 | Original unrun templates, preserved and superseded. |
| 006 off-AA1 | Starts 09:57:00 UTC. All eight ranks complete 41 optimizer/164 microbatch updates and the full 21–40 window: **316,975.7 global useful tokens/s**, **3.30806 s completed mean/update**. |
| 007 off-AA2 | Starts 10:01:27 UTC. All eight ranks complete 33 optimizer updates; fails during 34. Rank 1 NCCL watchdog reports CUDA illegal instruction at 10:04:19 UTC; exit 1 at 10:04:39 UTC. Reporting may be asynchronous; origin unresolved. |
| 008–010 | On-B1 / off-A2 / on-B2 are not released or run after 007 failure. |

The [corrected launcher](../../runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/PERFORMANCE-ASSESSMENT/preparation/002/README.md) adds only the three required memory-observer identities; actual CPU installer regression reproduces the omission and passes the correction. No source or window change occurs between the two native controls.

006's all-rank allocator envelope is **28.090 GiB allocated / 33.625 GiB reserved** over native reset-aligned 21–30 and 31–40 intervals. Its sampled device peak is **36.032 GiB**, a lower bound, with a 545-ms maximum sample gap. Launch to window opening is **127.047 s**. These are one baseline's values, not graph memory cost or paired performance evidence. All available native metrics are finite, and all eight logs in both native runs are intact. At 10:05:23 UTC all recorded task identities are absent and the eight GPUs are idle.

The performance assessment ends here under the owner's stop rule. No further execution, fault diagnosis, checkpoint/numerical work or publication is scheduled. Missing A/A and graph-on measurements remain explicit; the result does not establish that graphs are incompatible, slower or unprofitable.
