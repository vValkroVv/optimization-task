# Conditional matched performance protocol

Status: design only, 2026-11-12. No launch, source change, or A007 polling performed.

## Acceptance checklist and next decision

1. Define one finite, unprofiled same-fix graph-off/on comparison and identical-control noise estimate.
2. Preserve the exact production workload and identify timer, token, memory, activation and stability limits.
3. State stop conditions and separate initial performance evidence from numerical, continuation and finalist acceptance.

Proceed only after the owner of A007 supplies successful full-update execution and representative repeated forward/backward `cudaGraphLaunch` replay proof on the pinned path. A flag, installed compiled callable, graph manager, or captured node is insufficient. A007's profiled/debug-log timings are not performance controls. This prerequisite does not require bitwise equality between independent runs.

## Proposed finite sequence

| Order | Variant | Native updates | Measured updates | Purpose |
| --- | --- | --- | --- | --- |
| 1 | C0a, same fixes, graphs off | 71 | 21–70, closure at 71 | First valid control and initial memory |
| 2 | C1a, same fixes, graphs on | 71 | 21–70, closure at 71 | First matched candidate |
| 3 | C0b, identical to C0a | 71 | 21–70, closure at 71 | A/A variation and temporal drift |
| 4 | C1b, identical to C1a | 71 | 21–70, closure at 71 | Second candidate and A/B/A/B evidence |

Use separate fresh processes serially on the same eight-GPU allocation, source, data order and initialization state. The two off runs supply the identical A/A pair; a preceding extra A/A pair would duplicate this purpose. C0a/C1a can yield an initial descriptive pair before the repeat pair finishes. Do not label it a demonstrated benefit until repeats/noise are available. Total bound: four processes, 284 updates, 200 measured updates, 209,715,200 nominal useful measured tokens. No automatic retries or fallback settings.

Stop the matrix on a faulted/incomplete control or candidate, missing all-rank window, invalid token accounting, absent required replay evidence, source/config contamination, OOM, or nonfinite update. Retain the failed attempt and first error; do not run the remaining variants to fill a table. A valid completed timing window followed by a teardown failure is retained as diagnostic evidence, not an accepted successful matrix run. Original off controls failed with CUDA 715 at updates 22 and 33; those failures remain unresolved evidence, not a measured regression or noise estimate. A fresh same-fix control must succeed before interpreting candidate throughput.

## Exact configuration proposal

Reuse the reviewed A007 `compiler_entry.py` bootstrap (including per-rank option assertions) and source-pin checks, with attempt-local output paths and fresh `GIFT_HYBRID_EP_SCAN_RUNTIME_DIR`. Do not reuse A007's release file or output directories. No runner is supplied here.

Common native CLI arguments, supported by the existing 71-update control:

```text
--training.steps 71
--lr_scheduler.total_steps 30000
--debug.seed 42
--checkpoint.enable false
--dump_folder <new-attempt>/native
```

The observer overlay imports `gift/configs/experiments/gift_moe_3b_hybridep_fp8_compile.yaml` and selects:

```yaml
callbacks:
  campaign_timing:
    type: campaign_timing_observer
    start_step: 21
    update_count: 50
    output_dir: timing_observer
  campaign_step_boundary:
    type: campaign_cudagraph_step_boundary
compile:
  cudagraph_exclude_blocks:
    - layers.1
    - layers.2
    - layers.3
    - layers.4
    - layers.5
    - layers.6
    - layers.7
    - layers.8
    - layers.9
    - layers.10
    - layers.11
    - layers.12
    - layers.13
    - layers.14
    - layers.15
    - layers.16
    - layers.17
profiler:
  enable_profiling: false
  enable_memory_snapshot: false
```

Common `TORCHINDUCTOR_MIX_ORDER_REDUCTION=0`, `graph_partition=True`, `PYTORCH_ALLOC_CONF=expandable_segments:True`, native seed-42 policy, offline data/cache settings and shared existing cache locations match A007. The only intended off/on variable is `TORCHINDUCTOR_CUDAGRAPHS=0/1`. Unset `TORCH_LOGS` in both; remove profiling flags in both; disable allocator-history/debug/profiler instrumentation in both. Keep native metric frequency 10 and preset logging destinations, checkpoint and evaluation defaults unchanged. “Logging off” means extra compiler diagnostics off, not removal of required native metrics. All 18 model blocks and loss remain compiled; the 17 exclusions affect graph capture only. The public full-update marker and dense GEMM unsafe adapter remain present in both variants.

Retain local batch 8, global batch 256, sequence 4096, four microbatches/update, EP8, dense FSDP8/FSDP2, singleton expert-FSDP groups, SAC18, FlexAttention, full FP8 recipe, genuine changing C4 inputs/routing and all optimizer settings. No balanced routing, precision reduction, batch reduction or shorter sequence. The scheduler's inspected `total_steps` override preserves horizon 30,000 and warmup 2,000 while `training.steps=71` bounds native teardown; do not shorten warmup.

Pin `SOURCE-SNAPSHOT/006/manifest.json` SHA-256 `3a606c81d6b13692e422da04c333252620e504df9f1c4ba59c4bcf0c1a51f603`, archive SHA-256 `324cad36a75404f235f5dd7de10b571a75ad486276a44c09fe425f3187ebd51b`, and preset SHA-256 `b341701bd1eef7774c110fbf7ed69cf6280ba8691100b05be457dee3e7d50a7b`. Freeze the source throughout. Record dependency/runtime versions and all-rank compiler options, resolved rank-zero config/diffs, imported source paths, GPU/topology/group membership, initial-state/seed/data identity and cache policy. Shared caches are not reset between runs: report startup/cache effects and exclude them only from the bounded training-window metric, never from lifecycle cost.

## Boundaries and validity

Updates 21–70 are a proposed observation interval, not established steady state. The existing controls chose the same interval, and the observer requires the following batch-start boundary, so native 71 is the smallest unchanged interface giving those 50 complete intervals and clean shutdown. Twenty updates are not empirical proof that compilation/capture has settled. Before launch, use A007's owner-provided capture chronology to reject this boundary if setup is known to persist past it. After each unprofiled run, inspect the retained series, native compile warnings and memory behavior before calling the window stable. If a one-time setup transition contaminates the window or a trend remains unexplained, report initial-window results as inconclusive for stable throughput and stop for one revised bounded design; do not silently pick the fastest suffix or extend runs until they look favorable.

Repeated candidate capture tied to changing inputs/routing is recurring workload, not startup. Leave its elapsed time and memory in 21–70; quantify capture/replay coverage from separate mechanism evidence. If it produces a repeatable recurring regime, report that regime including recapture cost. If the finite run cannot distinguish recurrence from startup or growth, stable performance and amortization remain open. Do not subtract recapture time or describe a replay-only interval as full training. Logging/profiling differences prevent A007 timings from establishing magnitude.

## Metrics and memory

The inspected `CampaignTimingObserver` synchronizes the device, samples allocator state and aligns ranks before opening its timer. It times `batch_start(21)` through `batch_start(71)` plus the closing device drain, then performs elapsed/validity MAX reduction and JSON persistence outside the timer. It includes four-microbatch data fetch, forward/backward, communication, optimizer, native post-update metrics and actual checkpoint/evaluation cadence within the boundaries. The closing boundary occurs before update 71's work. GC before that callback belongs to the closing boundary interval. Report the maximum completed elapsed over eight ranks.

Primary useful tokens/s = actual global valid tokens in successfully applied updates / slowest-rank completed window seconds. Every rank must report 50 complete token records of 1,048,576 each, total 52,428,800; never sum these already-global counts across ranks or multiply accumulation again. Invalid/missing records produce no accepted throughput. Retried work receives no additional useful tokens; this matrix does not retry.

Retain full per-update host-submission spans and asynchronous current-stream event spans. Report their medians/spread with those exact labels. They are not individually synchronized completed update latency, pure device compute, or independent-side-stream elapsed time. The supported completed average is slowest-rank window seconds / 50. A median completed full-update latency remains unavailable through this observer; do not relabel a submission median to satisfy that field. Host, waits/idle, copies, communication and device compute attribution require the separate representative trace and must not be inferred by subtraction or overlapping duration sums.

Use identical 500-ms external device-memory sampling in all four runs, mapped to rank/GPU and lifecycle timestamps. Report sampled per-device maxima and minimum sampled headroom, including separate startup/capture and window phases when their timestamps are known; these are lower bounds on instantaneous peaks. Preserve actual capacity and all-rank results, not just a mean.

Observer memory snapshots expose allocated/reserved current values and allocator high-water counters at the two boundaries. Native `MetricsProcessor.log()` resets peaks at step 1 and every tenth update. Consequently, boundary peak fields are neither lifetime nor guaranteed 50-update phase peaks. Native logged peak-active/peak-reserved percentages over intervals ending 30,40,50,60,70 can describe those matched intervals; distinguish active bytes from allocated bytes. Defaults do not guarantee structured all-rank allocator records, and exact capture-window peaks are not available from two boundary samples. Do not change logging cadence or add synchronous counters just to fill the table. List per-rank allocated/reserved phase peaks and exact device peaks as missing wherever the retained sources cannot establish them; resolve that observability gate separately before final acceptance. Retain allocator retry/OOM warnings and sampled growth; do not silently accept pressure as steady state.

## Interpretation, costs and remaining gates

Report all four count/time ratios, completed average latency, labeled submission/event distributions, matched memory, C0a/C0b variation and C1a/C1b variation, plus both adjacent C1/C0 ratios. Four process-level observations do not support a universal significance threshold or invented numerical tolerance. If benefit is within measured variability/drift, the answer is “not demonstrated”; repeat only one decision-relevant pair under an explicitly revised finite plan. No speculative speed estimate or runtime deadline follows from A007's slow profiled execution.

This comparison attributes the graph-policy effect within the complete same-fix path. It does not establish net gain over the original baseline: the backend scan correction, marker, dense adapter, exclusions and mixed-order setting can affect runtime independently. Both variants use mixed-order reduction off; graph partition/capture policy can still alter generated code, allocations and compiler boundaries. Without generated-code equivalence evidence, call this the integrated graph-policy effect, not pure launch-overhead elimination. Retain the original failed baseline; a valid original-baseline comparison or evidence-backed disposition of its required defect remains a separate attribution gate.

Expected costs: four cold-process startups and native compiles/cache accesses; possible capture setup/recapture and persistent graph-pool storage; extra host marker calls once/update; observer timestamps/events once/boundary and scalar bookkeeping; one opening drain/barrier and one closing drain/MAX reduction; external sampling and native metric synchronization at unchanged cadence. Integrated timing includes device/host/communication costs that occur in the window; no extra per-update barrier or tensor dump is proposed. This document changes no production runtime.

Detailed ten-update matched numerical/state comparisons, precision-calibrated production A/A criteria, empty/skewed experts and stale-buffer checks, fresh-process bidirectional full-state checkpoint continuation, and evaluation transition limitations remain separate required finalist gates. They do not block initial useful-token/memory research once capture is runnable, and unequal independent-run loss/norm bits are not a substitute failure criterion. No acceptance tolerance is invented here.

Only an eligible finalist proceeds to the requested baseline/best-candidate 500 consecutive-update gate, preserving the same horizon, workload and cadence. That gate checks sustained execution, recurring capture, memory growth and amortized setup; it cannot repair an invalid short control, replace A/B/A/B noise, establish convergence or satisfy checkpoint/numerical gates. Final production acceptance remains open until all applicable gates and independent integration/runtime-cost review findings are resolved.

## Inspected evidence

- [Task requirements](../../../../../../backlog/tasks/opt-004-cuda-graphs-hybridep-fp8.md), sections 4–6.
- [A007 launcher](../../LANE-A/attempts/007/launch.sh), [bootstrap](../../LANE-A/attempts/007/compiler_entry.py), [overlay](../../LANE-A/attempts/007/config/observer.yaml).
- [Existing 71-update launcher](../../SCAN-FIX-BASELINE-AA/attempts/002/launch.sh), [overlay](../../SCAN-FIX-BASELINE-AA/attempts/002/config/observer.yaml).
- [Observer implementation](../../../../../../gift/torchtitan/experiments/cuda_graph_campaign/observer.py), `batch_start`, `_finish_window`, `_memory_boundary`.
- [Native trainer](../../../../../../gift/torchtitan/trainer.py), `train_step`, `train`; [scheduler config](../../../../../../gift/torchtitan/config/schema/components/lr_scheduler.py), `build`; [metrics](../../../../../../gift/torchtitan/components/metrics.py), `log`; [memory utility](../../../../../../gift/torchtitan/tools/utils.py), `device_capacity_report`, `reset_peak_stats`.

Verification: source/interface inspection only; no GPU commands or new runtime evidence. Documentation links and whitespace checked after writing.

## Clarification after A007 capture-frequency evidence

New coordinator-supplied evidence: A007 completed 25 updates; the initial rank-zero profile contains 348 `cudaGraphLaunch` events versus 336 captures/instantiations over three updates, with recurring static-input-pointer guards in the final step. Full dense/loss and all-rank attribution remains under independent review. These counts do not establish persistent model replay, replay coverage on every rank, or a capture-time percentage: captures and launches are different events and their durations can overlap other work. No new runtime evidence was collected for this clarification.

This evidence changes the immediate next decision. Prefer one shorter matched diagnostic pair before committing to the full finalist matrix, provided the independent mechanism review establishes useful forward/backward replay on the supported path and that repeated capture remains the unresolved cost question. If that review establishes no useful replay, the activation prerequisite fails and the pair should not run. Resolve or disposition the capture boundary first. If attribution remains missing, hold the pair rather than infer replay from these counts.

For this pair only, use all source/environment/workload controls above, changing the common stop to `--training.steps 26` and the common observer to `start_step: 21`, `update_count: 5`; close at batch-start 26. Run C0-short off first and C1-short on second, separate processes on the same allocation. Keep horizon 30,000, warmup 2,000, seed 42 and all four microbatches. Bound: two processes, 52 total updates, ten measured updates, 5,242,880 nominal useful measured tokens per variant. Twenty excluded updates only align the diagnostic with the late capture behavior already observed in A007; they do not establish steady state. Retain every capture/recapture cost within updates 21–25. No retry, no fastest-subwindow selection and no third diagnostic run.

The question is whether the late-window integrated cost and memory of the runnable graph policy justify proceeding directly to finalist measurements or instead require resolving recurring capture first. Report actual count/time useful tokens/s, completed mean update latency, labeled submission/event series, per-rank boundary memory and sampled device memory with the limitations above. This pair measures integrated behavior with recapture included; it cannot isolate capture overhead, estimate A/A variation, or prove a general benefit/regression. No percent threshold or final negative verdict follows from it.

Finite outcomes:

- A control/candidate faults, token/window validation fails, or source/activation evidence is invalid: stop and retain the result; diagnose that material defect before another performance design.
- Both complete and the candidate exhibits recurring severe elapsed cost or continuing memory growth consistent with the reviewed mechanism: publish the initial descriptive comparison and stop performance scaling while the capture/reuse boundary is resolved or the lane receives an evidence-backed disposition. A single short pair alone is insufficient to close the entire campaign as “no benefit.”
- Both complete, useful replay is established, and the observed recurring regime makes the candidate worth evaluating further: proceed to the unchanged four-process, 50-update-window protocol above. Those runs supply the missing A/A/noise and alternating repeated magnitude evidence.
- Both complete but setup/drift/attribution prevents interpretation: retain an inconclusive diagnostic; do not automatically launch the long matrix. Name the single unresolved mechanism or boundary question and redesign only if its answer can change the decision.

The short pair is an additional bounded screening cost, not credit toward the required 50 stable updates per finalist run. Any runnable candidate advanced as a finalist still owes complete A/B/A/B 50-stable-update windows, original-baseline attribution, detailed ten-update numerical checks, bidirectional continuation, and the eligible baseline/candidate 500 consecutive-update gate. If no further candidate change can be justified and a runnable lane must receive a completed performance verdict, use the required repeated stable measurements; do not rename an incomplete finalist “diagnostic” to avoid them. This clarification supersedes only the immediate next-step recommendation, not those acceptance requirements.
