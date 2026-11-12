# OPT-004 — Native HybridEP FP8 baseline stability

**Final status (12 November 2026): PASS for the bounded native baseline stability target.** The isolated DeepGEMM descriptor-protocol repair passes the matched 2,112-call reproducer and two fresh native graph-off runs, each with 200 optimizer updates and 800 microbatches on all eight ranks. Finite metrics, required features, intact logs and clean teardown pass. CUDA Graph/performance research remains paused.

## Diagnostic trace follow-up — 12 November 2026

**PASS:** attempt003 ran **17:24:18–17:31:15 UTC**. All eight ranks completed **111 optimizer updates / 444 microbatches**, with finite native metrics, intact logs, zero training errors and clean teardown. The successful 2×200-update stability evidence was reused. CUDA Graphs and performance research remain paused.

The native profiler recorded the same complete optimizer update **111** on global ranks **0 and 1** only. Updates 1–109 waited, 110 warmed the profiler, and 111 recorded/exported once. Each trace contains 60,500 CPU operators and 15,781 CUDA kernels. Epoch-correlated native logs prove all four forward/backward microbatches, communication, gradient clipping and fused AdamW lie inside the captured interval. Neither active trace contains compilation/recompilation work. `ProfilerStep#110` is the zero-based profiler label for training update111.

| Global rank | Chrome/Kineto artifact, outside Git | Bytes |
| --- | --- | ---: |
| 0 | [rank0_trace.json](artifacts/raw/BASELINE-TRACE/attempts/003/native/profiling/traces/iteration_111/rank0_trace.json) | 66,720,244 |
| 1 | [rank1_trace.json](artifacts/raw/BASELINE-TRACE/attempts/003/native/profiling/traces/iteration_111/rank1_trace.json) | 66,754,593 |

Executed from the repository root; [launcher](artifacts/raw/BASELINE-TRACE/attempts/003/launch.sh) retains the full torchrun command and environment:

```bash
/opt/torchtitan-venv/bin/python3.12 runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-TRACE/preparation/001/controller.py 003
```

Identities: unchanged task HEAD `446431dfa191ef0300f116257ae918bc85e38c06`, [Source010](artifacts/raw/SOURCE-SNAPSHOT/010/manifest.json) SHA256 `b2e9144af50e8673e0b8646a4f03f629bc240e6fc3ed1aaccfe6814491fb7d20`, exact preset SHA256 `b341701bd1eef7774c110fbf7ed69cf6280ba8691100b05be457dee3e7d50a7b`. Isolated DeepGEMM `2.3.0+f093a4c`, descriptor-fix-001 library digest `a122322508f98be4c07c04c7237e53ab`, header SHA256 `ed8bfb6a182d6a8a225d36b359287f17dba73e95d5cc50150ca177ea979557d2`. [Release pins](artifacts/raw/BASELINE-TRACE/attempts/003/release.json) additionally identify native profiler, schema, Trainer and launch adapter. [Live activation](artifacts/raw/BASELINE-TRACE/attempts/003/live-activation.json) proves the candidate extension on every rank and only authorized resolved-config differences: profiler, stop and output. HybridEP correction, FP8, Flex, compile, selective AC, batches8/256, sequence4096, accumulation4, c4_test, seed42 and scheduler30000/warmup2000 remain intact.

[Independent review and trace checks](artifacts/raw/BASELINE-TRACE/review/003/REPORT.md), [machine evidence with trace hashes](artifacts/raw/BASELINE-TRACE/review/003/acceptance.json), [CPU preparation checks](artifacts/raw/BASELINE-TRACE/preparation/001/asset-checks.json) and [root final release](artifacts/raw/BASELINE-TRACE/attempts/003/root-post-run-release.json) close acceptance. All691 recorded processes, controller, owner and workers exited; all eight GPUs are idle. Source01098 and both844-file dependency manifests remain unchanged. No checkpoint was created.

Capture limits: shapes, stacks and memory profiling are off. A small [launch-local adapter](artifacts/raw/BASELINE-TRACE/attempts/003/profile_entry.py) supplies the native profiler's missing rank/shape controls; scheduling/export stay native and no GIFT/shared-package source changes were made. Profiling two ranks perturbs their timing and peers' waits; this is no throughput benchmark. The native metric cadence ends at110; no final111 loss display was added. The configured 2,000-update learning-rate warmup remains in progress. Historical attempts001/002 retain their logs/traces; a mistakenly overwritten historical001 preflight observation is explicitly disclosed in the [reconciliation note](artifacts/raw/BASELINE-TRACE/attempts/003/reconciliation.json).

## Active goal and frozen path

Establish stable native execution with CUDA Graphs off on the exact supplied 3B MoE / HybridEP / FP8 preset. The accepted [source inventory](artifacts/raw/BASELINE-STABILITY/source-inventory/001/REPORT.md) selects the native `train_scan_fix` entry, includes the pinned HybridEP scan correction, retains the CudaCherry graph-safety custom-op wrapper, and keeps `TORCHINDUCTOR_MIX_ORDER_REDUCTION=0`. The CudaCherry wrapper remains active with graphs off. Use standard native logs; bypass the observer overlay and optional timing, phase-memory, step-marker, and all-18-block graph-exclusion overrides through existing controls, preserving source. The inventory verifies all 98 listed Source010 files and the exact preset hash; its native 200-update diagnostic payload ran as attempt002 on the new server. The repaired baseline uses the stored verification launcher, which prepends the isolated dependency copy before GIFT at process start.

The preset `gift/configs/experiments/gift_moe_3b_hybridep_fp8_compile.yaml` remains unchanged: MoE / HybridEP EP8, dense and sparse FP8, Flex attention, model and loss compilation, selective activation checkpointing, chunked cross-entropy, local/global batch 8/256, sequence length 4096, four accumulation microbatches, and the configured model, data, optimizer, 2,000-step warmup, and 30,000-step schedule horizon. The affected-kernel reproducer and both fresh native graph-off verification runs have passed with identical frozen config and launch settings; each run completes 200 optimizer updates and 800 microbatches per rank.

At 15:25 UTC the owner confirmed that the server changed while topology stayed the same. [Fresh reconciliation](artifacts/raw/BASELINE-STABILITY/reconciliation/002/receipt.json) verifies eight idle H100/SM90 GPUs with all-pair NV18, unchanged runtime package identities, all 98 Source010 hashes, and 2,365.8 GiB free. GPU UUIDs differ from historical runs. The [current-server trap smoke](artifacts/raw/BASELINE-STABILITY/tool-smoke/002/checks.json) again passes exception capture and kernel/PC/instruction decoding. Interrupted preparation/attempt 001 contains no launch or training output; its empty directories are retained.

## Current result and acceptance

| Gate | Executed evidence / status |
| --- | --- |
| Frozen native graph-off path | Accepted: Source010, exact preset, scan bound correction included, CudaCherry wrapper and mixed-order workaround retained; experimental observers bypassed. |
| First captured device fault | Attempt 002 completes **35 optimizer updates on all eight ranks**, then faults during update 36. Rank 3 core identifies **Warp Illegal Instruction Parameter** at DeepGEMM `sm90_fp8_gemm_1d1d_impl`, PC `0x7fd81aff1cc0`, function offset `+20416` (`0x4fc0`): `UTMALDG.2D.MULTICAST [UR4][UR8],UR10,desc[UR14]`. This is instruction attribution, not the cause of the invalid parameter. |
| Cause and fix | The original exact-SASS fixture returns A(group3)×B(group1) where A(group1)×B(group1) is required. Its real scheduler reuses the old A descriptor before reads complete. An isolated one-header repair drains the issuing lane's descriptor reads before both publication branches and adds the missing corrective publication release. Shared packages are unchanged. Native CTA82 corrupt bytes remain unobserved. |
| Affected-path rerun | **Pass:** identical cold-cache candidate fixture004 checks all 2,112 outputs (1,024 positive-group, 1,024 trailing-empty-group, 64 production-scale-K calls), exit 0, actual patched library/include/cubin verified. The full native compiled/autograd/SAC path passes in both native runs below. |
| Native200 run003 | **Pass:** 16:23:14–16:34:57 UTC, exit 0; all eight ranks complete exactly 200 optimizer updates and 800 micros with intact logs, finite native metrics, all activation/teardown markers, no native errors or new fallback forms. [Independent result review](artifacts/raw/BASELINE-STABILITY/analysis/003/REPORT.md), [root acceptance and process/GPU release](artifacts/raw/BASELINE-STABILITY/attempts/003/root-post-pass-release.json). |
| Native200 run004 | **Pass:** 16:36:25–16:47:50 UTC, exit 0; all eight ranks complete exactly 200 optimizer updates and 800 micros, finite native metrics, intact logs, required activation and clean teardown. [Independent result review](artifacts/raw/BASELINE-STABILITY/analysis/004/REPORT.md), [root final acceptance](artifacts/raw/BASELINE-STABILITY/attempts/004/root-post-pass-release.json). The resolved config matches003 except output path; no new warning/error/fallback form appears. |

[Attempt 002](artifacts/raw/BASELINE-STABILITY/attempts/002/release.json) starts at 15:30:55 UTC and exits 1 naturally at 15:34:05 UTC. Its fixed stop was 200 updates; the failed attempt is preserved without shortening. All eight rank environments carry the reviewed exception-capture flags and graph-off policy. The [actual resolved config](artifacts/raw/BASELINE-STABILITY/attempts/002/native/config/resolved_config.yaml) preserves the intended workload. No timing/phase/numerical observer, graph marker, profiler or launch blocking is selected. Failure therefore reproduces after bypassing those experimental observers.

The completed rank-3 core is 343,873,935 bytes, SHA-256 `1e9b5a60a5ba7d550ed07c8b10205b07be9f7693a375d17bf0acfb4613cb6c71`. [Triggered-PC decoding](artifacts/raw/BASELINE-STABILITY/analysis/002/decode-trigger.log) and the exact relocated module are retained. The current focused lane's later `EXIT` is not the triggered instruction. Native watchdog/sticky-error stacks remain detection points. [Independent root release checks](artifacts/raw/BASELINE-STABILITY/attempts/002/root-post-failure-release.json) at 15:37:09 UTC confirm owner/controller identities absent, all 696 recorded descendants released, and all eight GPUs idle with no compute applications. [Completion/log and binary evidence](artifacts/raw/BASELINE-STABILITY/analysis/002/REPORT.md) is retained; no stability pass is claimed from attempt002.

[Causal assessment](artifacts/raw/BASELINE-STABILITY/cause/001/REPORT.md) maps the native instruction to expert w2 weight-gradient B loads. Static staging alignment and retained registers show no invalid operand; the snapshot is not trigger-time state. The [caller review](artifacts/raw/BASELINE-STABILITY/caller-review/001/REPORT.md) finds per-call scratch allocated and launched on the same current CUDA stream, with no specific caller lifetime defect. Native CTA82 with four positive groups does not exercise the earlier proposed slot-reuse sequence; that hypothesis is withdrawn. The narrow reproducer separately supplies evidence for a real descriptor-protocol defect.

The unmodified [exact-SASS fixture001](artifacts/raw/BASELINE-STABILITY/reproducer/001/REPORT.md) passed admission call1, then returned a wrong value at call2, stopping after33 submitted GEMMs. Coordinates were not retained. [Warm-cache002](artifacts/raw/BASELINE-STABILITY/reproducer/002/REPORT.md) passed all 2,112 calls. The predeclared [cold-cache003](artifacts/raw/BASELINE-STABILITY/reproducer/003/experiment-contract.json), with002's identical detailed fixture, reproduced call2 failure after33 submissions: [8,448 finite mismatches](artifacts/raw/BASELINE-STABILITY/reproducer/003/root-result.json), all in group1, maximum absolute error11. The actual original SASS is unchanged; the first harness cache-miss failure and the warm-cache nonreproduction remain preserved. Later cases in failing attempts were not run under the predeclared first-error rule.

The [executable scheduler and mismatch derivation](artifacts/raw/BASELINE-STABILITY/dependencies/descriptor-fix-001/mismatch-check.json) maps all 16 retained coordinates to persistent CTA43, group route0→1→3, whose corrective branch overwrites the descriptor slot previously used by group1. Every actual value uniquely equals A3×B1 among the16 possible group pairs. The printed fixture `tile_cta=111` row-major estimate and earlier informal M8/CTA19 mapping are invalid. The verified width16/effective12 scheduler mapping selects descriptor-read completion before overwrite as the concrete repair mechanism.

### Isolated repair and affected-path result

[Candidate descriptor-fix-001](artifacts/raw/BASELINE-STABILITY/dependencies/descriptor-fix-001/REPORT.md) copies the installed DeepGEMM `2.3.0+f093a4c` and changes only `include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh`. It adapts merged [DeepGEMM343](https://github.com/deepseek-ai/DeepGEMM/pull/343), commit `1f6f3f378920ccb5cc036ef43eb3f5972e921713`, to both publication branches in this fork: commit/read-wait and an elected-lane compiler scheduling barrier before GMEM descriptor replacement, plus release after corrective publication. Existing acquires remain. This restores missing descriptor dependencies without a device, stream, CTA-wide, host or distributed rendezvous. GEMM arithmetic, precision, shapes, scheduler, scratch and communication remain unchanged.

Patched header SHA256: `ed8bfb6a182d6a8a225d36b359287f17dba73e95d5cc50150ca177ea979557d2`; patch SHA256: `7613dbdcefebf11ef04d9516eac75a88b0b2f2b7207a1356c30290cf27bdb749`. All 844 original files remain unchanged. The copied extension stays byte-identical, SHA256 `99b7afad2e2fe7ebabd26c22afc99dc652e2e6935fcbf37324359eb5aeeefe26`; its actual compiler include root selects the copy, library digest `a122322508f98be4c07c04c7237e53ab`. [Independent integration and runtime-cost review](artifacts/raw/BASELINE-STABILITY/review/descriptor-fix-001/REPORT.md) found no blocking defect for bounded execution. Added producer descriptor-read waits and fences may cost overlap at group transitions; performance is unmeasured and outside this stability task.

[Matched candidate004](artifacts/raw/BASELINE-STABILITY/reproducer/004/root-result.json) ran 16:09:00–16:09:07 UTC, exit 0. It uses byte-identical003 inputs, cases, budgets and output checks, changing package activation and fresh task caches. All 1,024 four-positive-group,1,024 trailing-empty-group and 64 large-K calls pass the exact integer oracle. Actual runtime metadata verifies the candidate digest/include root and exact specialization; loaded patched cubin SHA256 `463c4ea6de42e38e1b873bc63c1bc10f0b460ee9d8869ae1693cdc65870339cd`. No fault, mismatch or remaining GPU process is observed. This establishes affected-kernel original-fail/candidate-pass evidence, not direct native descriptor-byte attribution or native stability.

Both fresh native graph-off 200-update runs passed through the real compiled/autograd/SAC path. Native fast warmup overrides the requested local cache path with `/temp/.cache/rank_<r>/deepgemm`; [003 activation](artifacts/raw/BASELINE-STABILITY/analysis/003/kernel-activation.json) and [004 activation](artifacts/raw/BASELINE-STABILITY/analysis/004/kernel-activation.json) verify the candidate digest/include root and both grouped-weight-gradient shapes in all eight actual rank caches. Run003 publishes 48 candidate metadata entries to its fresh shared cache;004 reuses 45 canonical warmup entries and publishes none. Neither run is claimed to have a fully cold local cache. The native compilation policy is identical and retains `DG_CACHE_FORCE_SKIP=1` after warmup; no new arithmetic/compiler option was introduced. Exception-core flags, launch blocking, profiling and all experimental observers are absent.

[Final independent review](artifacts/raw/BASELINE-STABILITY/review/native200-001/REPORT.md) accepts both native results. Each rank in each run has successful step/optimizer spans 1–200 and exactly four successful microbatches per step; error-associated ends cannot count as completions. All 21 native loss/norm observations and 42 token observations per rank are finite; expected local/global valid counts are 131,072/1,048,576. Full JSONL parsing, unique increasing sequence IDs, span ordering, required activation and process-group teardown pass. Runtime-message comparison preserves inherited Inductor complex-RoPE lowering, Triton tuning and quack disk-cache compilation warnings; no required feature was disabled or new fallback form observed.

At 16:50:33 UTC, [root final release](artifacts/raw/BASELINE-STABILITY/attempts/004/root-post-pass-release.json) confirms all task identities absent, all eight GPUs idle with no compute applications, and post-runtime integrity of Source010 98 plus original 844 and candidate 844 files. The two terminal watchers exited 0; no task monitor remains. No new checkpoint was created.

### Source, runtime and executed commands

| Identity | Frozen value |
| --- | --- |
| Task checkout | Dirty `main`, HEAD `446431dfa191ef0300f116257ae918bc85e38c06`; existing contributor edits preserved; Source010 manifest SHA256 `b2e9144af50e8673e0b8646a4f03f629bc240e6fc3ed1aaccfe6814491fb7d20` verifies98 selected GIFT files. |
| Preset | `gift_moe_3b_hybridep_fp8_compile.yaml`, SHA256 `b341701bd1eef7774c110fbf7ed69cf6280ba8691100b05be457dee3e7d50a7b`; seed42; `c4_test`; exact resolved configs retained per attempt. |
| Runtime | `/opt/torchtitan-venv/bin/python3.12` / Python3.12.3; Torch2.15.0.dev20260822+cu130 (`8b353065ce260336c5fdfa74d754e8bd9b91b062`); CUDA13.0; NCCL2.30.7; Triton3.8.0+git3f6e4113; DeepEP1.2.1+1b8f467; CudaCherry0.1.0; DeepGEMM2.3.0+f093a4c plus the pinned local header patch. Toolkit/NVCC13.2; driver570.133.20. |
| Hardware | One current eight-H10080GB SM90 server, EP8/FSDP2, all-pair NV18; UUID/order in per-run preflight. Earlier-server failures retained separately. |
| Verification launcher | [003](artifacts/raw/BASELINE-STABILITY/attempts/003/launch.sh) and [004](artifacts/raw/BASELINE-STABILITY/attempts/004/launch.sh) byte-identical, SHA256 `1b11a59590cd167129619204a361358d0b0c6f29b9954c1c1ff0da5e780fc91c`; [controller](artifacts/raw/BASELINE-STABILITY/preparation/003/controller.py) SHA256 `bab461d40e48d6bdef5f2163ce73a25935a3a0116bfd46bbc347fc20fd13aa50`; [prelaunch review](artifacts/raw/BASELINE-STABILITY/review/native200-001/REPORT.md). |

Commands executed from the repository root, with preflight/release receipts retained and the existing detached process owner:

```bash
/opt/torchtitan-venv/bin/python3.12 runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY/preparation/003/controller.py 003
/opt/torchtitan-venv/bin/python3.12 runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY/preparation/003/controller.py 004
```

The stored launch payload pins all command-line overrides and environment settings, checks Source010/preset plus both 844-file dependency manifests, and refuses existing outputs. These are native stability runs, with 200-update stops and the original 30,000-update scheduler horizon. No performance or convergence measurement is inferred.

### Verdict and limits

The repaired native path meets the requested two-run stability target. The original exact-SASS fixture demonstrates premature descriptor reuse, and the matched repaired fixture passes every output check. The native fault is localized to TMA-B, but its exact corrupt descriptor bytes were not captured; that attribution limit remains explicit. This is bounded stability evidence. It does not establish unlimited reliability, convergence, numerical parity, CUDA Graph benefit or runtime cost. There is no bitwise training-equality gate. Shared installed packages and the preserved GIFT source remain unchanged by this repair.

The verbatim pre-update documents and SHA-256 receipt are preserved under [BASELINE-STABILITY/scope/001](artifacts/raw/BASELINE-STABILITY/scope/001/receipt.json). Prior-server failures and performance results remain historical evidence.

## Paused and stopped work

There is no bitwise-equality gate. CUDA Graph and performance work is paused. Checkpoint continuation, RNG/full-state restoration, tensor-recording/numerical frameworks, benchmarks, and broader framework work are stopped. Do not claim performance, convergence, or production readiness from this task.


---

## Historical performance-assessment report — superseded

Everything below this heading is the prior report retained for history. Its former current-scope statements, verdict, acceptance gates, next actions, and authorization text are superseded by the stability scope above and do not authorize further work.

# CUDA Graphs on the 3B HybridEP FP8 H100 workload

## Current scope — performance assessment only

**Final scoped verdict: MEASUREMENT BLOCKED.** The owner narrowed OPT-004 on 12 November 2026 to the existing runnable loss-only candidate: does it improve end-to-end global useful tokens/s, and what GPU-memory cost does it introduce? This is a performance assessment of the tested capture scope, with no production-readiness or convergence acceptance.

The second graph-off control completed 33 optimizer updates on all ranks and failed during update 34 with a CUDA illegal instruction. The fixed window did not complete, so graph-on runs were withheld. A/A variability, graph benefit and paired GPU-memory cost are unavailable. This is not evidence that graphs are incompatible or unprofitable.

The current contract in [OPT-004](../../backlog/tasks/opt-004-cuda-graphs-hybridep-fp8.md) supersedes all numerical, RNG, full-state, checkpoint/resume, detailed gradient/update and long-run gates below. Those systems are stopped; their evidence and partial edits remain preserved. The historical text below records decisions as they stood and does not authorize further work on them. The [scope-change receipt](artifacts/raw/PERFORMANCE-SCOPE/001/receipt.json) preserves the complete prior task/report. No future numerical preparation or diagnostic launch is scheduled.

### Candidate and matched comparison

Use A008 loss-only capture: all eighteen model blocks stay compiled with selective AC but graph-excluded. Graph-eligible work is chunked cross-entropy forward/backward and FlexAttention mask creation. Model/head GEMMs, HybridEP communication and optimizer remain eager/compiled outside CUDA Graphs, while staying inside the end-to-end timer. No full-update graph is claimed.

[A008 mechanism evidence](artifacts/raw/LANE-A/analysis/008/mechanism-review.md) proves all-rank loss replay at updates 21–23: 32 CE-forward and 32 CE-backward launches/update/rank, eight reused IDs per phase, four mask launches, no late capture/instantiate. The [copy review](artifacts/raw/LANE-A/analysis/008-copy-cost/report.md) records 25 GiB logits staging plus 1 MiB labels/rank/update and 17.605–17.641 ms copy activity; this is not an exposed wall-time tax or peak residency claim. New timings must preserve exactly this active candidate path and report the limits of reused replay evidence.

Both variants retain identical backend fixes, FlexAttention, model/loss compilation, graph partitioning, mixed-order workaround, model graph exclusions, precision, MoE/HybridEP, selective AC, data, seed, local/global batch, accumulation, optimizer and 30,000-update schedule. Only graph activation differs. [Source010](artifacts/raw/SOURCE-SNAPSHOT/010/manifest.json), manifest SHA256 `b2e9144af50e8673e0b8646a4f03f629bc240e6fc3ed1aaccfe6814491fb7d20`, preserves Source009 plus three stopped numerical-codec files. Their unfinished checks are not extended; they must remain inactive in the native performance entry.

### Predeclared short protocol and validity

The [scoped design review](artifacts/raw/PERFORMANCE-ASSESSMENT/review/001/REVIEW.md) accepts serial **off-AA1, off-AA2, on-B1, off-A2, on-B2**. AA1/AA2 measures control variability; AA2/B1/A2/B2 provides alternating off/on/off/on. Sharing AA2 makes the summaries dependent. Target updates **21–40**, twenty stable optimizer updates per run; update 41's first fetch boundary closes the window. A008 supports this provisional warmup choice, but later warmup/recapture is not silently discarded. Preserve every attempt, complete step series and startup.

Each valid run needs the exact workload, 41 native optimizer and 164 microbatch completions per rank, correct measured token records, finite native loss/norm metrics and intact logs. Each measured update has **1,048,576 global useful positions**; the twenty-update window has **20,971,520**, without summing duplicated global counters across ranks. Independent-run bitwise equality is not a gate. Stop on a CUDA or logging/validity failure and report **measurement blocked** if valid matched windows cannot be obtained; no new fault framework or selective window shortening.

The primary timer is device-completed end-to-end window wall time, aggregated by maximum rank elapsed. Report tokens/time and elapsed/20 as completed mean-update latency. Existing per-update host spans and current-stream CUDA events retain their actual labels; neither establishes complete-update p50/p99. Report launch/native-start-to-first-update and to-window setup separately. Reuse the accepted phase-memory observer with expected final metrics step 40: native reset intervals 21–30 and 31–40 define the measured allocator envelope, while boundary-tail samples remain separate. NVML sampled device maxima are lower bounds; report gaps and rank/UUID alignment.

### Current acceptance ledger

| Item | Current state |
| --- | --- |
| Scope, replay/copy evidence, finite matrix and timer review | Reused and recorded; historical numerical/continuation/long-run gates superseded. |
| Frozen active path and actual resolved configurations | 006/007 native configs match except output path; all16 compiler receipts show graphs off. Native Trainer, preset, backend fixes and non-graph settings verified; no numerical hooks. Graph-on configurations remain prepared but unexecuted. |
| Fresh eight-H100 ownership/topology/software/storage preflight | [09:49:38 UTC preflight](artifacts/raw/PERFORMANCE-ASSESSMENT/preflight/001/receipt.json) passes: exact eight idle H100/SM90 UUIDs, NV18, matching installed runtime, all Source010 bytes, NFS4/fsync and 6,580.04 GiB free. Recheck ownership if launch is delayed. |
| A/A and two alternating twenty-update pairs | Attempted: 006 valid, 007 CUDA failure; 008–010 not run under the stop rule. A/A variability and graph pairs are unavailable. |
| Matched useful tokens/s, completed latency, setup and memory | 006 single-control values independently reviewed below; graph-on and paired costs unavailable. No zero cost, no-benefit or incompatibility inference. |
| Replay/recapture | A008 all-rank proof at 21–23; actual later interval to be labeled. |
| Scoped final review and verdict | [Integration review](artifacts/raw/PERFORMANCE-ASSESSMENT/analysis/001/root-integration-review.json) accepts the evidence and metric labels. **Measurement blocked**; reduced task complete, missing comparisons explicit. |

### Short-protocol attempt ledger

| Attempt | Variant | Result | Measured window |
| --- | --- | --- | --- |
| [001](artifacts/raw/PERFORMANCE-ASSESSMENT/attempts/001/attempt-result.json) | off-AA1 | 09:53:36–09:53:48 UTC, exit 1 before native training: required phase-memory run identity missing. Zero updates; all 11 recorded identities absent. Launcher-only omission, no CUDA fault. | None; no primary throughput/memory result. |
| 002–005 | original unrun templates | Superseded after launcher correction; not executed or discarded. | None. |
| 006 | off-AA1 | Started 09:57:00 UTC; all eight ranks complete 41 optimizer/164 microbatch updates, clean exit, complete memory records and valid timing window; finite native rank-zero metrics. Process/GPU release verified. | Full 21–40: **316,975.7 useful tokens/s**, **3.30806 s completed mean/update**; memory/setup below. |
| 007 | off-AA2 | Started 10:01:27 UTC; 33 optimizer updates/all8 ranks, failure during 34; 135 micro completions on rank1 and 134 on others. Rank1 watchdog reports CUDA illegal instruction at 10:04:19 UTC; exit1 at 10:04:39 UTC. Origin unresolved. Logs intact; owner724 and controller/owner identities absent. | Failed fixed window; no A/A result, no partial-window substitute. |
| 008–010 | on-B1 / off-A2 / on-B2 | **NOT RUN:** conditional release withheld after 007 CUDA failure; no retry or new diagnostic work. | No graph-on measurement or paired memory result. |

### Available measurements and limitations

| Metric | 006 off-AA1 | Paired graph cost / A/A variation |
| --- | --- | --- |
| Useful throughput | 316,975.7 global useful tokens/s | Unavailable |
| Completed twenty-update wall / mean update | 66.161293 s / 3.308065 s | Unavailable |
| Allocator peak allocated / reserved | 28.089726 / 33.625 GiB | Unavailable |
| Sampled device peak | 36.032227 GiB; lower bound, max sample gap 545 ms | Unavailable |
| Setup, launch to first completed update / window opening | 63.661568 / 127.047080 s | Unavailable |

Allocator peaks are maxima across all eight ranks' complete native reset-aligned 21–30 and 31–40 intervals; no allocator retry or OOM counter occurs there. Opening/closing memory samples are separate, with the closing reset tail excluded from that envelope. NVML samples are joined by actual UUID/rank and UTC brackets; non-allocator memory is included, sampling can miss peaks. Whole-process allocator bootstrap peak is unavailable. Setup labels include compilation/warmup without claiming separately measured subphase costs.

The [reviewed short report](artifacts/raw/PERFORMANCE-ASSESSMENT/analysis/001/REPORT.md), [machine summary](artifacts/raw/PERFORMANCE-ASSESSMENT/analysis/001/summary.json), [full/partial step series and exact commands](artifacts/raw/PERFORMANCE-ASSESSMENT/analysis/001/COMMANDS.md), and [root integration review](artifacts/raw/PERFORMANCE-ASSESSMENT/analysis/001/root-integration-review.json) preserve counts, native metrics, all-rank configuration/activation proof, timer definitions, memory joins and failure evidence. Root independently recomputed tokens/time, completed mean latency and allocator peaks from raw rank records, and compared actual 006/007 configs and graph receipts. All eight logs parse with unique increasing sequence IDs; native rank-zero loss/norm metrics are finite. These checks do not establish detailed numerical equality or convergence.

At [10:05:23 UTC](artifacts/raw/PERFORMANCE-ASSESSMENT/attempts/007/post-failure-availability.json), the exact eight GPUs have zero used memory/utilization and no compute applications; all recorded task processes plus controller/owner identities are absent. There are no live experiment monitors or later launches. The tested graph scope remains an experimental loss/mask candidate with earlier real replay proof; this assessment cannot recommend it for a measured speed or memory benefit. No new debugging framework, numerical system or production gate follows.

### Retention and stopped work

The old native numerical-readiness attempt saved step 1, then rejected complex64 inventory on all ranks; observe was not run. The owner stopped this subsystem and its follow-ups. Its 36.52-GiB checkpoint no longer has a required state consumer. During the retention worker’s read-only streaming hash pass, the target disappeared before a durable complete inventory/metadata copy; neither root nor worker issued an unlink. Root independently confirms absence at 09:40:34 UTC with analysis intact. The [changed-state receipt](artifacts/raw/PERFORMANCE-SCOPE/001/checkpoint-retention/changed-state-receipt.md) preserves the earlier sizes/metadata hash and partial observed hashes. Cause is unknown; no successful task deletion or complete hash inventory is claimed. No checkpoint creation belongs to the short performance matrix.

## Preserved historical campaign evidence

All sections below retain prior results and unresolved claims as history. Former gates, planned launches and next actions concerning numerical/continuation/500-update work are superseded by the current scope above. Prior CUDA faults remain unresolved; a successful short window would not prove their repair.

## Historical campaign state and former acceptance gates

Status: **IN PROGRESS**, activated by the owner's 11 November 2026 launch request.
The owner confirmed that the current host and installed Python are the authorized
execution route. No package installation, source refresh, or publication is planned.
The owner also authorized deleting campaign checkpoints after their experiment's
comparisons and diagnosis no longer need them. Retain required starting/continuation
states until their consumers finish; then remove only task-owned payloads and record
the deletion, original identity and completed consumers. Keep logs, metrics, source
and configuration manifests, failures and conclusions. Avoid unnecessary tensor dumps.

Completed: native smoke/timing integration and representative all-rank baseline
traces. Strict loss/norm A/A differences are retained diagnostics, not an acceptance
gate under the owner’s latest update. The first longer unprofiled control failed
with an illegal instruction during update 22. A version-matched upstream
HybridEP bound correction is implemented behind an experimental entry; both review
findings are resolved. Its corrected graph-off smoke completed 25 updates on all eight
ranks, including update 22, with valid diagnostic timing and expected activation markers.
This does not establish fault causality or sustained stability. Corrected strict A/A
also differs at updates 2–10. Baseline numerical variability must calibrate comparisons; eliminating all of its
nondeterminism is outside scope. No performance verdict is available.

Attempt 007 completed 25 native updates with the causal mixed-order compiler
workaround, exiting 0 at 01:46:11 UTC on 12 November. Its exact runner, eight
workers and sampler exited, and all eight H100s were idle afterward. This is
execution evidence. Independent trace review confirms real dense/loss forward
and backward graph launches on every rank, but persistent reuse fails: dense
backward re-records every profiled call. The profiled, verbose diagnostic run
is not a performance comparison.

The finite loss-only fallback A008 passes native execution and persistent loss
forward/backward replay on every rank, retaining all required features. Its first
50-update graph-off control then fails during update 48 with another asynchronous
CUDA illegal instruction, after 47 complete optimizer updates. No valid stable
window or noise pair exists. Later matrix attempts are withheld; this is a CUDA
fault investigation, separate from baseline nondeterminism.

The subsequent completion-prefix diagnostic fails at update 21 after 24 active
checkpoints complete on every rank; ordinal 1385 remains unmatched and rank 6
reports 715 at event wait. It narrows a joined stream prefix without identifying
the kernel. Its live binary receipt was missed because monitoring attached after
exit; the failure and missing evidence remain explicit. All task processes exited
and GPUs are idle. NFS later had only 12.04 GiB free, blocking native checkpoint
storage at that time. A fresh 05:49:51 UTC observation shows 6,701.46 GiB available;
the bounded disk-reference implementation contract can now advance, with actual
payload creation still subject to a fresh capacity check.

The finite attribution run subsequently reaches its 20-minute limit after 16
updates/all ranks, without reproducing a native fault. Same-run mapped-file
provenance is captured; kernel origin remains unknown. Its watcher cleanup
verification fails during exit, while root's independent check confirms all
638 recorded identities gone and the eight H100s idle. Source009 now includes
the reviewed memory observer and corrected numerical recorder. Applied checks
pass 34 tests; launcher corrections are independently accepted and fresh preflight
passes. The first native readiness phase completes update 1 and full step-1 save,
then the recorder rejects complex64 inventory on all eight ranks. Observe is not
run; code correction and readiness remain open. All 660 process identities are
absent and all eight GPUs are idle.

1. Reconcile source/config and verify the active eight-H100 workload and environment.
2. Retain native smoke, strict A/A diagnostics and all-rank traces; measure production
   baseline A/A numerical variability and 50-stable-update unprofiled noise.
3. Give lanes A (compiler regions), B (one trace-supported region), and conditional
   C (graph-safe counts) finite, evidence-backed dispositions and graph-off/on controls.
4. Fix numerical criteria from production A/A variability and precision before A/B
   interpretation. For each runnable finalist, pass ten-update state comparisons and
   fresh-process bidirectional checkpoint continuation, including RNG and loader state.
5. For eligible finalists, complete alternating 50-stable-update A/B/A/B and exact
   baseline/best-candidate 500-update runs, paired memory, setup and replay evidence.
6. Resolve the independent integration/runtime-cost review, reconcile every finding,
   and give a reproducible scoped verdict. Missing evidence remains an open gate.

## Historical numerical acceptance contract — superseded

Independent-run bitwise loss/grad-norm equality is **not an acceptance gate** for this
nondeterministic workload. Original and corrected strict results remain diagnostic
failures of exact equality, without blocking capture or performance research.
Eliminating all baseline nondeterminism is outside scope. Establish numerical
criteria from production-policy A/A variability and FP8/BF16/FP32 arithmetic before
interpreting A/B loss, elementwise gradients, parameter updates or floating state;
record metric definitions, near-zero treatment, precision rationale and calibration.
No arbitrary passing tolerance or automatic use of an A/A maximum is accepted.

Exact structural invariants, four-microbatch accumulation, valid-token/auxiliary
normalization, state ownership and full-state checkpoint/resume remain required.
CUDA faults and tensor-lifetime errors remain defects to investigate. FlexAttention
is frozen; no FlashAttention substitution is selected. Next priority is working
native forward/backward capture or a specific reproducible blocker, then matched
graph-off/on useful-token throughput and GPU memory with identical prerequisite
fixes, workload and non-graph compiler settings. Measurements may proceed while
full numerical/continuation acceptance is pending, with that limitation explicit.

## Frozen source and workload

- Latest complete source freeze: [snapshot 009](artifacts/raw/SOURCE-SNAPSHOT/009/manifest.json),
  98 paths; manifest SHA-256 `cf95f726518062714913b8155dad878b73442fc5a0d68d993b7abdf6a55cf3f8`,
  archive `c58638a869893758254ff486300d427455d8733bf6a1ef117efad9fe0275e443`.
  It adds the corrected opt-in numerical recorder to Source008 and includes its
  prior [mechanical Ruff corrections](artifacts/raw/PHASE-MEMORY-INTEGRATION/002-format/checks.json)
  plus a test-only RNG restoration fixture fix. Exact preset unchanged; all
  87 other Source008 paths unchanged. Applied recorder/continuation checks:
  34 tests, full Ruff, format and syntax pass. Source009 later runs the failed
  native readiness attempt below; Source008 was not run on GPU. The working
  tree now has a scoped codec correction underway; repin before another GPU run.
- Task repository: `main`, `446431dfa191ef0300f116257ae918bc85e38c06` at activation;
  clean working tree. The newer commit after the refresh changes tracking requirements,
  with no `gift/` source diff.
- GIFT upstream: `6fd9a3230ab6361f11bbbf0b6ec5a5f125b78fc2`; reuse the
  [verified recursive refresh receipt](../../additional/baseline/gift-main-2026-11-11.md).
  Its 22 component pins and documented credential sanitization remain authoritative.
- Exact [preset](../../gift/configs/experiments/gift_moe_3b_hybridep_fp8_compile.yaml)
  SHA-256: `b341701bd1eef7774c110fbf7ed69cf6280ba8691100b05be457dee3e7d50a7b`.
  Local config check passed with zero warnings.
- Preserve 3B MoE, EP8, default FSDP2, dense/sparse FP8, selective AC, compiled model
  and chunked loss, batch 8/global 256, sequence 4096, four accumulation microbatches,
  optimizer, data, and 2,000/30,000 schedule. Additional fusion is not a prerequisite.
- Native short-run overlay: `training.steps=N`, `lr_scheduler.total_steps=30000`.
  The inspected scheduler uses the explicit total without shortening warmup; this
  bounds native execution and clean teardown. Checkpoint validation must explicitly
  handle the native final-step save policy; performance retains disabled checkpointing.

## Authorized local preflight

Verified eight idle NVIDIA H100 80GB HBM3 devices, SM90, each 81,559 MiB,
driver 570.133.20, and `NV18` connectivity for every GPU pair; no compute processes.
The task checkout is on NFS. No remote connection or new allocation is needed.

| Component | Installed identity |
| --- | --- |
| Python | `/opt/torchtitan-venv/bin/python`, 3.12.3 |
| PyTorch | `2.15.0.dev20260822+cu130`, git `8b353065ce260336c5fdfa74d754e8bd9b91b062` |
| CUDA / NCCL | 13.0 / 2.30.7 |
| Triton | `3.8.0+git3f6e4113` |
| DeepEP | `1.2.1+1b8f467`; `deep_ep.HybridEPBuffer` import succeeds |
| DeepGEMM | `2.3.0`, commit `f093a4c` |
| cuDACHerry / NVSHMEM | 0.1.0 / 3.3.24 |
| datasets / TensorBoard | 4.7.0 / 2.20.0 |

The image has an editable GIFT elsewhere; each attempt must run from task-local
`gift/` and verify its import path. Local tokenizer JSONs and `c4_test/data.json`
are present and native loading passed in the recorded full-target smoke runs.

## Research and integration findings

Initial source/API investigation is complete. These are source facts and hypotheses,
not capture or performance results.

| Finding | Disposition / acceptance gate |
| --- | --- |
| Native valid-token all-reduce returns a Python scalar before forward | Keep count discovery outside capture; preserve per-update normalization. |
| Blocking HybridEP discovers host counts and dynamic receive shapes | Reproduce active behavior; do not freeze routing or counts. |
| Sparse FP8 rejects nonblocking capacity-factor mode | Lane C is not a simple config toggle; establish a legal Hopper implementation first. |
| DeepGEMM WGRAD uses CPU group counts; active row-to-column conversion is cuDACHerry | Inspect pinned installed backend, preserve precision and changing counts. |
| Native checkpoint does not save RNG state | Full-state continuation gate open; no exact-target numerical resume defect is proven yet. Use an opt-in rank-local observation/state extension if needed. |
| Native TPS uses physical positions and logging intervals | Resolve useful-token per-update slowest-rank observability without contaminating timings. |
| Chunked loss assembles FP32 hidden gradients and controls FSDP head hooks | Preserve loss and FP32 arithmetic; GraphTrainer replacement is unsuitable. |

### Version-matched established implementations and selection

The installed nightly reports `triton.cudagraphs=False`, graph trees enabled,
`graph_partition=True`, and graph-managed input mutation enabled. Both existing
model block compilation and loss compilation use the default mode. Flex has its
own explicit graph-off setting alongside autotuning. A graph-only compiler control
is the first candidate: preserve all other compiler options and Flex's existing
policy, and prove actual forward/backward replay. `max-autotune` would also change
kernel selection, so it cannot isolate graph replay. This selection follows the
[public compile options](https://docs.pytorch.org/docs/2.15/generated/torch.compile.html)
and inspected installed `torch/_inductor/config.py` / `list_mode_options()`.

[CUDA Graph Trees](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html)
can partition eligible work, but dynamic shapes can cause new recordings, input
mutations require supported ownership, and custom-op acceptance does not establish
that the operator's internal host work is replay safe. The nightly already enables
partitioning; do not change that setting between controls. Graph pools and input
copies may offset launch savings. Establish capture boundaries on every rank.

The [NCCL 2.30 family guide](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2303/user-guide/docs/usage/cudagraph.html)
requires consistent capture/replay participation and documents mixed-communicator
deadlock caveats. That support does not establish NVSHMEM/HybridEP compatibility.
[NVIDIA's graph troubleshooting](https://docs.nvidia.com/dl-cuda-graph/latest/troubleshooting/performance-issues.html)
identifies fragmentation, copies, hook deferral and stream serialization as possible
costs. Measure full-update effects and overlap rather than assuming launch savings.

The [TorchTitan wrapper at `45e33cd`](https://github.com/pytorch/torchtitan/blob/45e33cd413b64cb37a7840cfd3c9765d3c639ae1/torchtitan/distributed/cuda_graph.py)
provides useful input-metadata and buffer-lifetime checks. It retains capture-created
gradient references because resetting gradients to `None` otherwise hides replayed
gradients from the optimizer. Its separate forward/backward and optimizer capture
paths do not establish four-microbatch SAC/FP8/EP correctness here. Capture warmup
and immediate replay mutate state; this campaign must restore complete state before
matched comparisons and rebuild graphs after restart.

The [MoE report `2603.07685v2`](https://arxiv.org/html/2603.07685v2) motivates
partial static regions and device-driven counts. Its hardware stack uses HybridEP
on GB200/GB300 and DeepEP on H100; its DeepSeek-V3 H100 setup disables graphs.
These results do not validate this HybridEP H100 workload or supply its speedup.
The linked [ECHO draft](https://github.com/NVIDIA/Megatron-LM/pull/2368), inspected
at `a2b16b8733bb4ced46880d6adee6f19124732991`, requires Blackwell MXFP8 for its
device-initiated GEMM path. The merged [Paged Stashing implementation](https://github.com/NVIDIA/Megatron-LM/blob/c0c4fdc45e9f0f8047e29f2cb8669613169752c3/docs/user-guide/features/paged_stash.md)
uses an experimental TE expert path and explicit overflow recovery. Neither is a
maintainable drop-in replacement for the required pinned recipe.

DeepGEMM's maintained [API at `5081f851`](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/csrc/apis/gemm.hpp#L343)
still requires host group sizes for Hopper FP8 NT weight gradients; its device-count
TN/PSUM alternative targets SM100. Merely removing dispatch synchronization would
leave stale capture-time WGRAD launch metadata. Keep lane C conditional on a legal
complete count-consumer path and measured need. Do not relax the existing rejection
of fixed-capacity sparse FP8 tails that can propagate NaNs into gradients.

### Native integration and cost checklist

The active wrapper order is EP setup/reservation, selective AC, block compilation,
then FSDP2 (`models/gift_moe/_base/parallelize.py`). The native step discovers global
valid tokens before forward, executes four changing microbatches, runs chunked-head
and decoder backward, clips dense/expert gradients jointly, updates AdamW, then
advances the Python scheduler and native callbacks (`trainer.py:689,732,793`).
Keep host metrics, scheduler and checkpoint staging outside candidate capture.

Chunked loss uses eight chunks and a full FP32 hidden-gradient assembly buffer
(`components/loss.py:664`), about 192 MiB per local microbatch at this shape. It
borrows/restores FSDP head reshard and gradient-sync settings. SAC saves communication
and selected quantization operations and skips compiled forward Python side effects
during backward recomputation. New boundaries must preserve these exact contracts.
Expert and dense FP8 quantize current inputs/weights; do not freeze their scales.
HybridEP global buffer replacement invalidates graph-owned handles and addresses.

For worst-case routing, EP8, top-4 and four experts per receiver allow 1,048,576
received rows, versus 131,072 average balanced rows. One capacity-sized FP8 payload
with hidden 1536 is 1.5 GiB; one BF16 intermediate with hidden 2048 is 4 GiB.
These are shape arithmetic, not measured peaks, and exclude scales, gradients and
17-layer/SAC lifetimes. Safe capacity cannot assume balanced routing or drop tokens.

The active sparse WGRAD row-to-column path is installed cuDACHerry, not the local
Triton fallback. Its installed Python wrapper reads device group counts in the
kernel, leaves unused tail storage uninitialized, and requires 128-row alignment.
Its wrapper SHA-256 is `3839e99e3f3de2af14fbc0562d5d155f4d93fcc3b116a2a6077c488ef9cf72ae`.
Package metadata supplies no source commit; the Docker declaration's cuDACHerry pin
is not yet independent proof of installed source identity. No backend is replaced.

Native HF text emits shifted real-token labels with no ignored-label padding; verify
this through the actual batches before using the existing global valid-token count.
Its epoch shuffle uses an explicit seed and persisted cursor/buffers. No active
dropout or random routing is established, so omitted RNG serialization alone does
not prove numerical resume corruption. Required full-state continuation must still
account for per-rank CPU/CUDA RNG and iterator creation. Rank-qualified state keys
are necessary because DCP deduplicates common keys. Use identical synchronous,
full-state validation overlays: native final and initial saves otherwise default to
model-only behavior. Router counts are nonpersistent and must be proven reset at
the saved update boundary or explicitly restored. Preserve their baseline SAC behavior.

Before/after cost review for any candidate must cover graph setup/recordings, kernel
and copy frequency per microbatch, persistent inputs/gradients/pools, buffer lifetime,
host count waits, unchanged FP32 accumulation, all-rank collectives and stream overlap.
This report-only research adds no production runtime work. Candidate runtime costs
remain to be measured; all correctness/lifecycle gates remain open.

## Experiment ledger

Unavailable values are shown as `—`, never zero. Every attempt receives an ID
before launch. Raw logs, tensors and traces remain under ignored `artifacts/`.

| Experiment / attempt | Hypothesis, intended change and controls | Status | Stable window | Useful tokens/s | Median update ms | Peak memory | Correctness / continuation | Graph coverage | Evidence / next decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NATIVE-SMOKE / 001 | Exact native target can complete 10 updates; no source change; stop overlay only | PASS (exit 0, ten updates) | None (smoke) | — | — | Native logged 33.40 GiB; phase/per-rank peaks not yet established | Smoke only; strict A/A and continuation NOT RUN | Unverified native policy | [Manifest](artifacts/raw/NATIVE-SMOKE/001/manifest.json), [command](artifacts/raw/NATIVE-SMOKE/001/command.txt); review complete rank evidence, then deterministic controls and timing |
| NATIVE-SMOKE / 001-S01 | Prelaunch manifest generator used relative config path from task root instead of gift cwd | FAIL (setup only; no GPU launch) | None | — | — | — | INAPPLICABLE | INAPPLICABLE | [Original error](artifacts/raw/NATIVE-SMOKE/001/setup-subattempts/001-cwd-error.txt); fix generator working directory |
| NATIVE-SMOKE / 001-S02 | Prelaunch manifest generator reloaded config without eight-rank validator context | FAIL (setup only; no GPU launch) | None | — | — | — | INAPPLICABLE | INAPPLICABLE | [Original error](artifacts/raw/NATIVE-SMOKE/001/setup-subattempts/002-world-size-error.txt); reuse completed WORLD_SIZE=8 preflight |
| RNG-PROBE / 001a | Actual target CPU loader continuation: distinguish early and post-iterator RNG restore | PASS (component only) | Four continuation microbatches | — | — | — | Inputs and Python/NumPy/Torch CPU RNG match after correct restore; GPU continuation NOT RUN | INAPPLICABLE | [Handoff and commands](artifacts/raw/RNG-PROBE/001/handoff.md), [loader log](artifacts/raw/RNG-PROBE/001/loader-with-callback.log) |
| RNG-PROBE / 001b-S01 | Native CPU DCP shared-key diagnostic incorrectly expected a whole-rank winner | FAIL (diagnostic assertion) | None | — | — | — | Actual dedup mixes fields across ranks | INAPPLICABLE | [Retained failure](artifacts/raw/RNG-PROBE/001/dcp-first-failure.log); correct expected outcome to per-field loss of identity |
| RNG-PROBE / 001b-S02 | Retry reused a stale failed file-store | FAIL (diagnostic setup) | None | — | — | — | INAPPLICABLE | INAPPLICABLE | [Retained failure](artifacts/raw/RNG-PROBE/001/dcp-stale-rendezvous.log); reassess and use fresh rendezvous/output paths |
| RNG-PROBE / 001b | Real two-rank CPU/Gloo DCP with native CallbackHandler and rank-qualified state | PASS (component only) | None | — | — | — | Qualified full payloads match; shared keys fail both rank payloads | INAPPLICABLE | [DCP log](artifacts/raw/RNG-PROBE/001/dcp.log); use qualified immutable state and defer restore |
| CONFIG-CONTRACT / 001-S01 | Read-only resolved-overlay comparison omitted required logger keyword | FAIL (diagnostic setup; no GPU launch) | None | — | — | — | INAPPLICABLE | INAPPLICABLE | [Retained error](artifacts/raw/CONFIG-CONTRACT/001/setup-failure.txt); use inspected native loader signature |
| CONFIG-CONTRACT / 001 | Compare class-stripped base and continuation config with identical stop/horizon/seed overlays | PASS (decode only) | None | — | — | — | Model/loss/data/optimizer/parallelism/precision/AC/compile settings unchanged | INAPPLICABLE | [Exact differences](artifacts/raw/CONFIG-CONTRACT/001/result.json); only trainer type, RNG callback and full-state checkpoint overlay differ |
| PLUMBING-REGRESSIONS / 001 | Behavioral checks for post-window inertness, queued-device completion and RNG callback ownership | FAIL before fixes; PASS after fixes (CPU doubles only) | None | — | — | — | T1/T2 timing regressions reported by owner; R1 and intermediate failures retained in [continuation logs](artifacts/raw/RNG-CONTINUATION/001/) | INAPPLICABLE | All three material findings fixed; no GPU claim |
| REVIEW / 001 | Independent targeted verification of the three fixes and public type pin | PASS, five tests | None | — | — | — | CPU-only; CUDA hidden | INAPPLICABLE | [Commands](artifacts/raw/REVIEW/001/commands.json), [log](artifacts/raw/REVIEW/001/targeted-tests.log); native GPU verification may proceed |
| STATIC-CHECKS / 001 | Full scoped Ruff lint/format, whitespace and local document links | PASS after mechanical formatting of two RNG files | None | — | — | — | No production execution | INAPPLICABLE | Ruff check passed; 12 Python files format-clean; 31 local links across four documents exist; [link results](artifacts/raw/STATIC-CHECKS/001/document-links.json) |
| TIMING-OBSERVER / 001 | Native ten-update integration with window 3–7 and closure at 8, followed by two more updates | PASS, exit 0 on all eight ranks | Five diagnostic updates; not a stable performance comparison | 311,755.77 (diagnostic only) | Submission/event series retained; no performance conclusion | All-rank allocator high-water allocated 24.89–26.30 GiB, reserved 28.70–31.47 GiB; not device peaks | 5, 242,880 actual global valid tokens per rank; one activation + one complete record each; clean step-10 teardown | Native policy, replay unverified | [Attempt directory](artifacts/raw/TIMING-OBSERVER/001/); 16.8173-second slowest-rank completed window; metadata omissions corrected below |
| METADATA-REGRESSIONS / 001 | Include actual FSDP subclasses of CheckpointWrapper and singleton expert-FSDP group in activation inventory | Two tests FAIL before fix; affected eight-test module PASS after fix | None | — | — | — | Diagnostic metadata only; upcoming baseline runs verify actual enhanced inventory | INAPPLICABLE | [Before log](artifacts/raw/STATIC-CHECKS/metadata-fix/before.log), [after log](artifacts/raw/STATIC-CHECKS/metadata-fix/after-0.log); scoped lint/format/whitespace and targeted independent source review passed |
| GRAPH-OPTION / 001 | Compare fresh-process compiler configuration for TORCHINDUCTOR_CUDAGRAPHS=0/1 | PASS, configuration probe only | None | — | — | — | No CUDA initialized in either process | Not measured | [Probe](artifacts/raw/GRAPH-OPTION/001/probe.py), [exact diff](artifacts/raw/GRAPH-OPTION/001/result.json): only triton.cudagraphs changes; Lane A needs no extra config framework |
| DETERMINISTIC-AA / 001 | Strict native seed-42 ten-update control, original preset with per-step native TensorBoard | PASS execution, exit 0; diagnostic bitwise comparison differs | Correctness control, not performance | — | — | — | Strict deterministic=true, warn_only=false; ten raw loss/norm points retained | Native policy | [Attempt directory](artifacts/raw/DETERMINISTIC-AA/001/), [float32 metric bit patterns](artifacts/raw/DETERMINISTIC-AA/001/metrics.json) |
| DETERMINISTIC-AA / 002 | Independent strict control from the same seed, source, config and start | PASS execution, exit 0; diagnostic exact inequality | Correctness control, not performance | — | — | — | Both arrays match at update 1 and differ at updates 2–10; no tolerance waiver | Native policy | [Attempt directory](artifacts/raw/DETERMINISTIC-AA/002/); separate process, same allocation and frozen file hashes; scoped backend diagnosis continues |
| BASELINE-TRACE / 001 | Native 25-update run; built-in profiler skip 20, warmup 1, active 3, repeat 1 | FAIL before measurement: detached launcher disappeared before native outputs; empty supervisor log, no exit-status file or surviving task processes | None | — | — | — | No training result | No capture result | [Reviewed script](artifacts/raw/BASELINE-TRACE/attempts/001/launch.sh) retained; likely process-group supervision issue, not established workload failure; reuse the launch mechanism that completed the strict controls |
| BASELINE-TRACE / 002 | Same reviewed native profiler command, using the working start_new_session launch mechanism | PASS execution and all-rank mechanism analysis | Native updates 22–23 fully analyzed; update 24 recorded but full-step span includes export | — | — | — | Four microbatches and one native optimizer per update/rank; genuine HybridEP forward/backward and active compiled/FP8 kernels | Zero cudaGraphLaunch/cuGraphLaunch on every rank | [Analysis](artifacts/raw/BASELINE-TRACE/analysis/001/handoff.md), [original hashes and summaries](artifacts/raw/BASELINE-TRACE/analysis/001/summary.json); no speed estimate from profiling |
| BASELINE-AA / 001 | Unprofiled native 71-update run, observation window 21–70, closure at 71; 500-ms device memory sampling | FAIL, exit 1: rank 2 illegal instruction during update 22's fourth microbatch, before optimizer | Incomplete; no valid timing JSONL | — | — | 290 device samples/GPU, sampled maxima 32.4–37.4 GiB; not a valid paired memory result | Last successful optimizer update 21 on every rank; originating kernel unknown; earlier ten-update controls and profiled 25-update run passed | Graphs off per original policy; not traced in this attempt | [Attempt result](artifacts/raw/BASELINE-AA/attempts/001/attempt-result.md); all task processes/sampler exited; second control held for scoped diagnosis, no preset deviation |
| SCAN-FIX-IMPLEMENTATION / 001–002 | SHA-guarded rank-local wrapper/header staging, exact one-bound correction, native factory/argv delegation | Eighteen CPU tests PASS; both startup review findings resolved; independent targeted eight tests PASS | None | — | — | Small generated CPU source fixtures only | Actual GIFT get_buffer consumer and installed wrapper ABI pass with mocked native constructor and CUDA uninitialized; no CUDA correctness claim | INAPPLICABLE | [Fix verification](artifacts/raw/SCAN-FIX-IMPLEMENTATION/002/review-fixes.md), [independent log](artifacts/raw/SCAN-FIX-REVIEW/001/targeted-fixes.log); source frozen and archived before GPU use |
| SCAN-FIX-SMOKE / 001 | Corrected experimental entry, graph off, native 25 unprofiled updates; diagnostic observer 3–7 and later execution past original failure point | PASS execution (exit 0, 25 updates/all ranks) | Five early diagnostic updates only; not a stable performance window | — | — | Sampled device maxima 32,735–36,993 MiB; polling, not instantaneous peaks | All-rank optimizer completion and expected SAC/compile/FP8/groups; strict inequality diagnostic; calibrated comparisons pending | Explicit global graph policy off | [Result and raw evidence](artifacts/raw/SCAN-FIX-SMOKE/attempts/001/result.json), [handoff](artifacts/raw/SCAN-FIX-SMOKE/attempts/001/handoff.md), [frozen 19-file source](artifacts/raw/SOURCE-SNAPSHOT/002/manifest.json); original failed baseline preserved |
| SCAN-FIX-DETERMINISTIC-AA / 001–002 | Fresh-process strict seed-42 ten-update pair after the corrected prerequisite; graph off, unchanged target and 30,000-step horizon | Both PASS execution (exit 0); strict inequality diagnostic | Correctness control, not performance | — | — | — | Loss and global grad-norm bits match only update 1, differ 2–10; no tolerance waiver | Explicit global graph policy off | [Exact arrays and comparison](artifacts/raw/SCAN-FIX-DETERMINISTIC-AA/comparison.json); correction does not restore strict parity, actual cause unresolved |
| LANE-A / 001 | Original graph-only diagnostic plan, native entry without corrected prerequisite | NOT RUN; superseded before launch by 002 | None | — | — | — | No GPU execution | Unmeasured | Historical prepared command retained; baseline source defect changed the prerequisite |
| LANE-A / 002 | Corrected native entry, only global graph option on; 25 updates and all-rank profiler | FAIL (exit 1) during first forward/backward; clean teardown | No optimizer update completed, not performance | — | — | — | First decoder warmup rejects invalidated graph-owned mask output; all eight ranks fail, no optimizer update | No representative replay coverage established | [Reviewed command](artifacts/raw/LANE-A/attempts/002/launch.sh), [acceptance and scope](artifacts/raw/LANE-A/attempts/002/handoff.md); reviewed detached launch PID 488216; exact first error retained, no fallback or retry |
| SCAN-FIX-BASELINE-AA / 001–002 | Corrected graph-off 71-update pair, observer updates 21–70 and boundary 71 | 001 FAIL (exit 1); 002 NOT RUN after invalid first control | Intended updates 21–70 each; actual stability pending | — | — | — | No valid performance/noise pair; numerical criteria and finalist gates separate | Graph off | [Failure, memory and phases](artifacts/raw/SCAN-FIX-BASELINE-AA/attempts/001/result.json), [command](artifacts/raw/SCAN-FIX-BASELINE-AA/attempts/001/launch.sh); 001 launched 22:47:08 UTC, PID 493679; failed/incomplete control stops the pair |
| SCAN-FIX-GENERATED-COST / 001 | Offline original/corrected matched SM90 E=4 scan artifacts | PASS bounded static comparison | None | — | — | Same 69 registers/thread, 1,472 B shared/block, zero local/stack | Different machine instructions; no numerical/runtime claim | INAPPLICABLE | [Binary comparison and provenance limits](artifacts/raw/SCAN-FIX-GENERATED-COST/001/handoff.md); equal runtime cost and fault causality remain open |
| GRAPH-STEP-IMPLEMENTATION / 001 | Opt-in native batch_start Observer using one public generation mark per complete optimizer update | Implemented and independently reviewed; guarded 26 tests PASS | None | — | — | No tensor state retained; longer graph-pool lifetime unmeasured | Native four-microbatch event order verified on lightweight real trainer step; GPU semantics open | Not executed | [Implementation and commands](artifacts/raw/GRAPH-STEP-IMPLEMENTATION/001/handoff.md); preserve FlexAttention and all preset features |
| LANE-A / 003 | Reviewed full-update marker plus identical corrected backend and graph-on settings; native 25-update diagnostic | FAIL (exit 1) in first dense-block forward warmup | No optimizer update completed | — | — | Samples retained; not a completed paired window | All eight ranks report one untracked live allocation in the graph pool; no backward/optimizer completion | No representative replay proof | [Failure, all-rank phases and limits](artifacts/raw/LANE-A/attempts/003/failure-summary.json); original mask-input error not observed, allocation ownership remains under investigation |
| LANE-A / 004 | Same source 003 and marked graph-on path; built-in allocator history only | FAIL (exit 1), same first dense-block warmup pool guard; clean teardown | Diagnostic only | — | — | Samples retained; no paired window | All eight full error histories link the bad address to CudaCherry get_cached_workspace | No representative replay proof | [Launch record](artifacts/raw/LANE-A/attempts/004/launch-invocation.json), [command](artifacts/raw/LANE-A/attempts/004/launch.sh); single run 23:50:05–35 UTC, PID 515371; all eight idle after unwind |
| LANE-A / 005 | Reviewed dense-GEMM capture-safety adapter, source 004, same marked native graph-on workload | FAIL (exit 1) at 00:22:43 UTC; clean teardown | No optimizer update completed; not performance | — | — | Samples retained; no paired window | New storage-pointer/pool invariant error; cause under source/runtime diagnosis | No representative replay coverage established | [Launch record](artifacts/raw/LANE-A/attempts/005/launch-invocation.json), [command](artifacts/raw/LANE-A/attempts/005/launch.sh); one launch at 00:21:52 UTC, PID 524701; history off; all eight idle after unwind |
| LANE-A / 006 | Reviewed per-block graph exclusions, source 005; all 18 blocks/loss compiled, 17 MoE blocks outside graphs | FAIL (exit 1) at 00:56:25 UTC; clean teardown | No optimizer update completed; not performance | — | — | Samples retained; no completed timing window | First-update compiled backward uses undefined buf66 in generated partition arguments on all eight ranks | No representative replay proof | [Launch record](artifacts/raw/LANE-A/attempts/006/launch-invocation.json), [command](artifacts/raw/LANE-A/attempts/006/launch.sh); single run, all eight idle after unwind; mapped GEMM binary path/hash recorded |
| LANE-A / 007 | Same 17 MoE exclusions; causal common mixed-order-reduction setting off; source 006 | PASS execution, 25 updates/all eight ranks; single launch 2026-11-12 01:32:38–01:46:11 UTC, PID 547579, exit 0 | Profiled capture viability; not performance | — | — | Sampled device maxima 34,869–38,693 MiB; no paired window or instantaneous peak | Each rank: 25 optimizer completions, 100 forward/backward completions, clean teardown; 18 compiled/SAC blocks, target groups/FP8 active; diagnostic callback inactive | Real dense/loss forward/backward launches; no settled representative reuse | [Launch record](artifacts/raw/LANE-A/attempts/007/launch-invocation.json), [command](artifacts/raw/LANE-A/attempts/007/launch.sh), [independent mechanism review](artifacts/raw/LANE-A/analysis/007/mechanism-review.md); all task processes exited and eight GPUs idle after completion |
| MATCHED-RECAPTURE / 001–002 | Same source 006 and prerequisites; graph-off/on, no profiling or extra compiler logs; native 26 updates each | 001 PASS; 002 exit 0 but FAIL run-level event-completeness after ENOSPC | Valid descriptive observer windows 21–25; not stable performance/noise | Off 318,557.08; on 36,055.46 | On window 8.84× longer; descriptive only, no accepted repeated comparison | Whole-run sampled maxima: off 32,857–36,753 MiB, on 34,329–38,975 MiB; allocator boundary table in handoff; not phase/exact peaks | C0 complete; C1 missing/duplicated persisted events around 17–18, not proof of skipped optimizer; all finalist gates open | Prior A007 trace proves actual dense/loss launches with recurring recording; quiet run is not independently profiled | [Paired timing/memory and failure handoff](artifacts/raw/MATCHED-RECAPTURE/attempts/002/paired-handoff.md), [storage diagnosis](artifacts/raw/C1-STORAGE-LOGGING/001/diagnosis.md); no retry; exact GPU cleanup confirmed |
| LANE-A / 008 | One finite loss-only fallback: all 18 model blocks compiled/SAC but graph excluded; common prerequisites unchanged | PASS, 25 optimizer/100 forward-backward completions on every rank; exit 0 at 02:41:52 UTC | Profiled mechanism only; diagnostic observer 3–7 | — | No performance claim | Same sampling; phase peaks remain separate gate | Intact logs, no ENOSPC/fatal errors; all eight trace JSONs parse completely | Persistent actual CE forward/backward replay on every rank, zero late capture/instantiate | [Native result](artifacts/raw/LANE-A/attempts/008/monitor-handoff.md), [independent replay review](artifacts/raw/LANE-A/analysis/008/mechanism-review.md); GPUs released; proceed to matched controls |
| LOSS-ONLY-PERF / 001–004 | Exact loss-only off/on/off/on matrix, same source 006; native 71, observation 21–70 | 001 FAIL exit 1; 002–004 NOT RUN after failed control | No valid 50-update window; only 21–47 complete | — | No noise/comparison | Samples retained; no accepted paired result | All ranks completed 47 optimizer/188 microbatch backward events; first microbatch of 48 failed, before optimizer | Graph off in failed control; A008 remains independent graph-on mechanism proof | [Protocol](artifacts/raw/LOSS-ONLY-PERF/preparation/001/protocol.json), [launch](artifacts/raw/LOSS-ONLY-PERF/attempts/001/launch-invocation.json); CUDA illegal instruction, intact logging, clean all-eight release; no retry |
| COMBINE-BWD-PROBE / 001 | Source007 fixed window 21–60/native 60, post-combine host-completion and reference-retention diagnostic; graph off | FAIL exit 1 during update 21; no retry | Diagnostic only, not performance | — | No throughput inference | Raw 500 ms samples retained; no memory-performance comparison | All eight: 20 optimizer updates, 81 completed microbatches, 24 active completions, unmatched ordinal 1385; rank 6 event-wait 715 | A008 mechanism evidence remains separate | [Handoff](artifacts/raw/COMBINE-BWD-PROBE/attempts/001/handoff.md), [canonical extraction](artifacts/raw/COMBINE-BWD-PROBE/attempts/001/postrun-extraction-verified.json); live binary receipt missed, all exact PIDs/GPU released; parser/extraction/monitor failures preserved |
| SANITIZER-ORIGIN / 001 | Source007, graph off, native 60; one 20-minute memcheck attribution hypothesis with documented user-kernel patching disabled | INCONCLUSIVE; timeout 124 at 06:39:17 UTC after one launch | 16 complete updates, update 17 interrupted; diagnostic only | — | No throughput inference | Tool overhead/coverage unmeasured | No native 715; 50 initialization API reports fill print limit; same-run mapped binaries captured; watcher cleanup verification fails, separate root release passes | No new graph result | [Outcome and extraction](artifacts/raw/SANITIZER-ORIGIN/analysis/001/handoff.md), [complete release](artifacts/raw/SANITIZER-ORIGIN/attempts/001/gpu-release-check-complete.json); all 638 recorded identities gone, eight GPUs idle; earlier faults unresolved, no retry |
| PHASE-MEMORY-INTEGRATION / 001 | Apply accepted opt-in native-stat/timing-anchor adapter after Source007 release; freeze Source008 | PASS source application/11 CPU tests/lint/syntax; GPU NOT RUN | None | — | No performance claim | Native calls/reset cadence unchanged in CPU checks; phase GPU records pending | Exact applied bytes match independently accepted staging; other 91 Source007 files unchanged | No new graph result | [Checks](artifacts/raw/PHASE-MEMORY-INTEGRATION/001/checks.json), [Source008](artifacts/raw/SOURCE-SNAPSHOT/008/manifest.json); actual eight-rank integration and paired memory remain open |
| NUMERICAL-PROBE-INTEGRATION / 001 | Apply independently corrected direct-reference recorder; freeze Source009 | PASS source application/34 combined CPU tests/full lint/format/syntax; native GPU NOT RUN | None | — | Diagnostic tensor copying is not performance | Target storage/host peaks pending | Initial combined run: 33 tests passed, fixture teardown error reproduced and fixed; final 34 pass; original review/repros retained | No new graph result | [Applied checks](artifacts/raw/NUMERICAL-PROBE-INTEGRATION/001/checks.json), [targeted acceptance](artifacts/raw/NUMERICAL-PROBE-REVIEW/001/TARGETED-ACCEPTANCE.md); launcher corrections and fresh preflight required |
| NUMERICAL-NATIVE-PROBE / 001 | Source009 graph off; native full S0 step1, fresh full-load-only observed step2 | FAIL exit1 after native update1/full step1save; observe NOT RUN | Diagnostic readiness only | — | Not performance | Sampled RSS 176.78 GiB; device sampled maxima 28,729–33,043 MiB; not instantaneous/steady peaks | Invocation1 fails before training on missing parent; invocation2 reaches native save, then all8 inventory handlers reject complex64; all660 process identities absent, GPUs idle | No new graph result | [Preflight](artifacts/raw/NUMERICAL-NATIVE-PROBE/preparation/001/preflight/receipt-20261112T085508489561440.json), [launch002](artifacts/raw/NUMERICAL-NATIVE-PROBE/preparation/001/root-launch-record-002.json); no automatic retry/extension |

After two metadata-generator failures, preparation was reassessed: stop extending
that generator, retain the already-successful native config preflight and use the
reviewed direct native command with simple persistent supervision. Neither setup
failure ran training. The GPU attempt started once; no workload setting was relaxed.
The native attempt exited successfully at 2026-11-11 22:45:23 Europe/Moscow.
Step-10 rounded native loss was 10.55146 and grad norm 15.8008. These are smoke
observations, not bitwise comparison or throughput evidence; native TPS uses a
different denominator and its startup window is unsuitable for performance claims.
The [completed smoke summary](artifacts/raw/NATIVE-SMOKE/001/result.json) records
all eight stdout/stderr pairs, eight structured logs, source/runtime/data hashes,
resolved configuration and clean teardown. Native markers show compiled loss,
all 18 compiled blocks, selective AC, `fully_shard`, 72 dense/34 grouped FP8
projections, accumulation four, and an EP8 HybridEP buffer. All-rank group membership
and actual graph replay remain separate gates. Nonfatal quack disk-cache loader
warnings fell back to in-process compilation; preserve these startup costs and
verify stable-window exclusion rather than treating them as capture failures.

### CPU continuation evidence and retained limits

The [RNG probe handoff](artifacts/raw/RNG-PROBE/001/handoff.md) pins runtime,
commands, fixtures, source references and failures. The real native target loader
uses workers zero and pinning false, matching the resolved GPU smoke configuration.
Both fresh and restored iterator creation advance global Torch CPU RNG. Restoring
after iterator creation, or once after the first deterministic target fetch, passes
four `[8,4096]` continuation microbatches and remaining CPU RNG exactly. Earlier
`after_load`/`fit_start`/`batch_start` restore is insufficient because the iterator is
created lazily. An experiment-owned restore boundary should stage loaded state and
apply it once, without treating RNG mutation as metric observation.

Native two-rank DCP with qualified keys preserves independently seeded rank states.
Shared keys mix scalar/pickled fields from one rank with Torch RNG tensors from the
other; this is stronger evidence than a presumed whole-payload winner. The final
probe uses fresh file-store/checkpoint paths after the two retained diagnostic
failures. These CPU checks do not validate actual CUDA, SAC, FP8 or HybridEP resume.

Pinned Torch compiler code preserves CPU/current-device Torch RNG around selected
compilation, example-generation and graph setup paths, but does not restore Python,
NumPy, loader, model, optimizer or arbitrary backend state. Actual GPU generator
progression remains open. Native MoE optimizer pre-hooks update persistent expert
bias and clear nonpersistent expert counts; verify zero counts at postupdate saves.

Completed auxiliary DCP checkpoint payloads were deleted under the owner's retention
instruction after their comparisons finished (78,322 bytes across three directories).
The [deletion inventory](artifacts/raw/RNG-PROBE/001/checkpoint-deletion.json) retains
original paths, sizes and hashes; scripts, per-rank results and failed/passing logs
remain. At that point no training checkpoint had been created; the later failed
readiness run now retains a 36.52-GiB native S0 pending its named consumers.

### Planned observation extension: cost review before implementation

The native smoke prerequisite is now met. Add an opt-in experimental observer using
existing callback/config extension points. Preserve the native step, loss, optimizer,
checkpoint manager and all model/distributed wrappers. Required observations are
actual global valid-token counts, bounded complete-update windows, asynchronous CUDA
events, native lifecycle metrics and one-time per-rank runtime/group evidence.
Boundary synchronization and rank aggregation occur outside the measured window;
do not add per-update barriers, device scalar reads or tensor copies. The existing
Python valid-token value can be reused. Numerical snapshots and RNG continuation
remain separate, invasive validation work; they must not contaminate timings.

Expected performance-mode costs are callback dispatch, Python timestamps/integer
accounting, event recording and small deferred metadata. Persistent storage scales
with updates/ranks, not model size. No tensor arithmetic, communication ordering,
precision, optimizer updates or compile/fusion settings should change. Verify exact
callback boundaries and stream completion, default-off behavior, window/token
accounting, and runtime overhead before accepting timing evidence. The shared NFS
had about 993 GiB free at inspection; avoid full-model ten-update tensor dumps and
retain checkpoint payloads only while their required consumers remain unfinished.

Capture observability must also preserve the baseline: installed
`torch/_inductor/cudagraph_trees.py:get_manager` defaults to creating a manager,
whose constructor synchronizes and captures an empty pool-lifetime graph. Any
read-only diagnostic must pass `create_if_none_exists=False`; manager existence
alone cannot prove useful capture. The pinned source exposes recorded nodes/modes
and rerecord metadata but no direct total replay counter. Use representative native
CUDA traces with actual `cudaGraphLaunch` events to prove repeated forward/backward
replay; distinguish recorded nodes, live graphs, recaptures and replay counts.

### Planned continuation extension: cost review before implementation

The CPU probes justify an isolated, opt-in rank-local RNG state component and an
experiment trainer boundary for deferred restore. Preserve native checkpoint format,
model/optimizer/loader/scheduler saves and the full native training step. Add no
state capture, serialization, or restore to ordinary performance runs. Validation
checkpoint boundaries pay only small cloned CPU/Python/NumPy/current-device RNG
snapshots; restore occurs once after lazy iterator creation and deterministic fetch,
before resumed model execution. No per-update tensor arithmetic, collective, extra
GPU scalar read or precision change is needed. Model/optimizer checkpoint I/O remains
the dominant validation payload. The implementation must validate rank/world/schema,
keep snapshots immutable, avoid early RNG mutation during DCP load, and preserve
native input ordering/exhaustion. Full GPU 5+5 continuation remains an acceptance gate.

### Independent integration and runtime-cost review

The independent reviewer inspected inherited callback dispatch, native loop/checkpoint
ordering, source changes and focused tests. A completed worker test suite alone is
not acceptance. All three material findings are resolved for native verification:

| Finding | Affected contract | Required disposition |
| --- | --- | --- |
| T1: later callbacks revisit an already closed timing window | Native execution after the selected window must continue without more timing collectives/events or an exception | Closed guard added; regression fails before fix and verifies two later updates are inert after fix |
| T2: elapsed denominator stops before queued GPU tail completes | Full-update useful throughput must include device completion on every rank | Opening drain precedes timing; closing synchronization precedes completed stop; submission spans remain separate. Delayed-completion regression verifies 2 seconds of submission plus 3 seconds of tail gives a 5-second denominator |
| R1: RNG callback can be selected with native trainer that never applies pending restore | Unsupported callback/trainer combinations must fail explicitly | Experimental observer binds to the actual CampaignTrainer and rejects unbound native fit_start or restore. Decode intentionally accepts the callback alone; docs explain the runtime guard. Generic core validation additions were reverted |

The reviewer independently ran five targeted tests covering T1, T2, R1 and the public
callback type pin: all passed. [Commands](artifacts/raw/REVIEW/001/commands.json),
[raw output](artifacts/raw/REVIEW/001/targeted-tests.log) and
[result](artifacts/raw/REVIEW/001/result.json) are retained. CUDA was hidden; these
tests establish no GPU behavior. RNG tests restore incoming Python/NumPy/Torch CPU
state and seed a CPU Generator without changing CUDA seeds. The type-pin regression
now includes both experimental callbacks. The reviewer inspected the retained
19-test continuation run (actual target loader and two-rank Gloo DCP, CUDA mocked).
Timing-owner seven-test and fail-before-fix runs were reported in the worker session;
their stdout was not separately persisted, so only the independent targeted log is
claimed as retained raw verification.

Final runtime-cost review found no source blocker for native GPU verification.
Timing still adds Python callback/event work within the observation window; its real
overhead remains unmeasured. RNG state work is opt-in at validation checkpoint/load
boundaries. Ordinary performance runs keep native training/checkpoint behavior; the
validation overlay's resolved diff confirms all frozen workload fields unchanged.
Actual CUDA timing, detailed numerical comparisons and full-state continuation remain
open at that review boundary. The subsequent native observer run passed its timing,
token and post-window continuation checks. Its activation inventory initially
reported zero SAC wrappers because FSDP changes the class to a CheckpointWrapper
subclass, and omitted expert-FSDP size-one groups because the existing mesh inventory
filters them. Both metadata omissions now have fail-before/pass-after regressions.
The observer uses the actual wrapper base class and queries the existing `efsdp`
mesh explicitly. Independent targeted review confirmed no group creation, collective,
device read or per-update cost: both changes run once at fit_start. Native logs
already prove selective AC was applied; the original zero count was not evidence
that AC was disabled. Future runs verify the corrected per-rank inventory.

Detailed finalist numerical comparison remains conditional on a runnable candidate.
The [acceptance amendment](artifacts/raw/NUMERICAL-DESIGN/002/acceptance-amendment.md)
sets the production-variability/precision freeze procedure; numeric tolerances are
unset until actual calibration. The historical [bounded design](artifacts/raw/NUMERICAL-DESIGN/001/handoff.md) uses a native
full-state checkpoint at update 5 and five sequential tensor-family stripes for ten
matched updates. Its historical exact-default numerical rule is superseded by the
owner’s A/A-variability/precision contract above; exact structural invariants remain. Parameter states P0–P10 also
support FP64 update-delta comparison without storing a separate delta archive.
Its 170.3-GiB striped-reference estimate and provisional 192-GiB target do not
establish joint ten-update coverage under nondeterminism. The amendment keeps a
coherent full-trajectory storage decision open pending actual bytes, lossless
compression and free-space inventory; it does not silently replace that gate with
independent family stripes or short prefixes. This is a design,
not implemented or executed numerical evidence. Checkpoints will be removed once
their required comparisons finish, after
recording hashes and deletion inventories; logs and failed results remain.

## Timing and metric contract

Primary metric: sum of actual globally valid tokens in applied optimizer updates
divided by integrated window elapsed time at the slowest rank. Nominal physical
positions are 1,048,576 per update, matching actual globally valid-token counts in
the five-update observer diagnostic; a valid stable comparison remains unavailable.
Report count/time window throughput separately from median update latency/spread.
Traces will separate host submission, device kernels, waits, copies and communication;
these overlap and must not be summed into elapsed time. Startup, compilation,
autotuning, capture, stable updates, checkpoint/evaluation and teardown remain separate.

## Strict deterministic control result

Both seed-42 native processes completed ten updates with strict deterministic mode
enabled and `deterministic_warn_only=false`. Per-step TensorBoard scalars were read
as float32 bit patterns, without using rounded console text. Loss and gradient norm
each match at update 1 and differ at all nine subsequent updates. At update 2, loss
is 11.599272727966309 versus 11.599270820617676, and norm is 28.14470100402832 versus
28.144771575927734. The historical strict comparison **differs**; the owner removed bitwise equality
as an acceptance gate while retaining these results as diagnostics. Matching first-update aggregate scalars does not prove
elementwise gradient, optimizer or routing-state equality. An exact causal backend
has not yet been established. Strict Torch mode being accepted does not prove that
opaque custom kernels enforce it. Continue capture/performance research and calibrate numerical criteria from
production-policy A/A variability; baseline nondeterminism alone does not block it.

The [bounded backend diagnosis](artifacts/raw/DETERMINISTIC-AA/backend-determinism-diagnosis.json)
inspected the active dense cuDACHerry FP8, sparse cuDACHerry row2col/DeepGEMM and
DeepEP dispatch/combine wrapper surfaces. None exposes a deterministic argument at
the called entry point. The inspected extension sources do not establish Torch's
flag controls these kernels; they also do not establish that a particular kernel
caused this result. Generic CUTLASS reduction modes are not proof of the mode used
by the installed DeepGEMM entry point. A targeted follow-up traces actual ordering,
padding and routing-state ownership; no backend switch or precision workaround has
been applied. Both processes retained the original preset hash and identical current
source-file hashes. [Full scalar arrays and comparison](artifacts/raw/DETERMINISTIC-AA/comparison.json)
and the [prelaunch validator false alarm](artifacts/raw/DETERMINISTIC-AA/setup-subattempts/001-freeze-identity-check.json)
are preserved. That false alarm involved NUL-separated filename handling, ran no
GPU work, and was replaced with direct plain-path hashing rather than more harness
extensions.

Native strict mode also replaces `FlexAttention._compiled_flex_attn` with a fresh
`torch.compile(flex_attention)` call (`distributed/utils.py:145–151`), removing the
class-level explicit compiler options. Both strict controls share this supported
debug behavior; they do not exercise the production Flex compiler-policy dictionary
unchanged. Performance/profile runs keep deterministic debug disabled and retain
the original preset policy. Candidate correctness evidence must state this coverage
difference; the native debug override is not a campaign optimization.

## Baseline fault investigation and scoped prerequisite

The first native error is rank 2's fourth forward/backward microbatch at update 22;
the later NCCL watchdog event query propagates the asynchronous fault. The
originating kernel is unknown. Post-run devices reported zero ECC errors, no
pending/failed row remapping, low temperature and no surviving task processes.
These observations do not exclude a transient software or hardware fault. No Xid
log route was available. The passed profiled control and failed unprofiled run
remain separate results.

[Primary-source fault research](artifacts/raw/BASELINE-AA/fault-research/001/handoff.md)
found a concrete installed HybridEP defect: its fused scan allocates an integer
array of `NUM_OF_EXPERTS_PER_RANK - 1` but writes and reads through index
`NUM_OF_EXPERTS_PER_RANK - 1`. Four experts/rank accesses index 3 of an array of
length 3. The installed header exactly matches upstream
[`1b8f467`](https://github.com/deepseek-ai/DeepEP/blob/1b8f467/csrc/hybrid_ep/backend/hybrid_ep_backend.cuh#L5128).
The relevant single bound is corrected by
[`17cfb817`](https://github.com/deepseek-ai/DeepEP/commit/17cfb817bccec3a9c247013360cc550c2bac441e).
The same commit's standalone-permutation rewrite is outside this correction.
Issue #756's pinned-count lifetime race concerns standalone permutation; it is not
an established explanation for this fused target. DeepGEMM's related tensor-map-race
fix is not proven to fit the installed fork. No blanket upgrade is justified.

All eight failed-worker scan artifacts exist in their original per-process cache
directories; rank 2's SM90 specialization includes four experts/rank. The generated
source includes the installed header, and cache keys omit header hashes. This links
the affected specialization and installed source; emitted behavior and causality
for either observed failure remain unproven. A fresh CPU-only loader check resolves
CUDA compatibility library 595.45.04; NVIDIA supports that forward-compatibility
route over driver branch 570 on data-center GPUs. Driver 570 alone does not establish
an unsupported CUDA 13 stack.

### Selected correction and runtime-cost review before coding

Stage the unchanged installed Python wrapper and backend headers under a unique
task-owned per-rank runtime directory, apply only the upstream array-bound line,
and load that wrapper through standard `importlib` under the existing `deep_ep`
package. Its existing `__file__` supplies the native constructor's `base_path` and
JIT include root. An opt-in GIFT experiment entry selects this class at the GIFT
buffer factory before any buffer exists, then delegates arguments and training to
the native entry point. Original wrapper/header hashes must match inspected bytes;
fresh processes and distinct task-owned cache roots prevent stale kernel reuse.
Installed package files and the original native entry point remain preserved.

Expected work is startup-only source copy/hashing/import and fresh JIT compile.
There is no per-update Python branch, tensor arithmetic, synchronization, transfer,
collective, FP8 recipe, stream/fusion policy or checkpoint-loop change. The corrected
GPU local integer array has one additional element; register/local-memory allocation
and setup/steady-state cost need actual control measurements. This is a justified
undefined-behavior correction, not a proven repair of either runtime failure.
CPU staging/import/factory tests and independent review precede native GPU validation.
Actual fused-kernel correctness, calibrated numerical comparisons and longer
unprofiled stability remain required; source-text checks cannot pass those gates. Keep the original failed
baseline, then compare graph-off/on with this identical prerequisite if validated.
Additional kernel fusion is not required. The earlier observation/RNG source is
archived as a [15-file snapshot](artifacts/raw/SOURCE-SNAPSHOT/001/manifest.json)
with a 31-KB source archive and tracked patch; the new prerequisite gets a new pin.

### Corrected native integration result

The corrected entry completed all 25 native optimizer updates on all eight ranks,
including all four microbatches at the earlier failure's update 22. Each isolated
rank cache produced the E=4 scan specialization; runtime manifests record the exact
one-line header delta and unchanged wrapper. All ranks now report 18 SAC wrappers,
18 compiled model callables, 72 dense FP8 modules, 17 sparse grouped experts, 18 shared
quantizers, 17 HybridEP dispatchers and singleton expert-FSDP membership. Construction
metadata alone does not prove every custom-op invocation or loss lowering; preserve
the existing native logs and baseline trace evidence alongside this inventory.

The observer's early five-update diagnostic closed validly at boundary 8, followed
by 18 later native updates. It counted 5, 242,880 globally valid tokens over a
slowest-rank 16.892979-second window. This is a diagnostic interval, not stable-window
performance, noise, or a comparison with the original backend. Device polling maxima
were 32,735–36,993 MiB; allocator boundary high-water allocated was 24.89–26.29 GiB
and reserved was 28.87–31.22 GiB. Polling can miss transient peaks, and allocator
counters exclude backend allocations. Sixteen nonfatal Quack cache-lock timeouts are
retained as startup observations. Exact task processes exited and all GPUs were idle.
The [result](artifacts/raw/SCAN-FIX-SMOKE/attempts/001/result.json) retains per-rank
counts, hashes, sampled memory and cleanup. A successful bounded run does not prove
that the correction caused the earlier fault to disappear or resolve strict parity.

The independent [offline generated-cost check](artifacts/raw/SCAN-FIX-GENERATED-COST/001/handoff.md)
compared one matched E=4 SM90 scan artifact from the failed original baseline with
rank 2's corrected smoke artifact. Both report 69 registers/thread, 1,472 bytes of
shared memory/block, zero stack/local memory and no LDL/STL instructions. Text size
and instruction count match, but 99 of 1,984 instruction encodings and 48 printed
assembly slots differ. No increase in these static resources is observed for this
specialization. Instruction scheduling/dependencies differ, so equal runtime cost
is not established. Original source-to-binary provenance lacks a retained
preprocessed translation unit; the comparison does not assign every machine-code
difference solely to the header edit or establish the original fault's cause.

### Corrected strict A/A result

Both corrected ten-update controls exited 0 with strict deterministic mode and
`deterministic_warn_only=false`, the same resolved contract, graph off and the
corrected header on every rank. All eight ranks completed ten optimizer updates.
Loss and global grad-norm float32 bits again match only update 1 and differ at
updates 2–10. At update 2, loss is 11.599272727966309 versus 11.599274635314941,
and norm is 28.144922256469727 versus 28.14484405517578.
The [full comparison](artifacts/raw/SCAN-FIX-DETERMINISTIC-AA/comparison.json)
retains both arrays and all mismatches. The array-bound correction does not restore exact equality; exact equality is now
diagnostic only. The original failed controls remain separate evidence; neither the small
loss difference nor accepted deterministic mode establishes deterministic custom
kernels. No tolerance, backend or precision substitution has been adopted. The proposed
identical-operand nondeterminism diagnostic is not scheduled under the owner’s
updated scope; preserve its design without expanding into nondeterminism removal.

### Lane A first actual-target failure

The graph-on corrected-prerequisite attempt failed in the first native
forward/backward, before any optimizer update. All eight ranks report
`accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run`.
The first decoder FORWARD warmup fails while inspecting an input's storage pointer;
the stored allocation traceback identifies the earlier compiled Flex mask producer,
including `_dense_to_ordered(partial_blocks)`. This occurs before actual attention
computation or backward. Native finally-wrapper `fwd_bwd_end`/`step_end` records do
not mean successful execution: there is no optimizer start/end. The profiler had not
reached its update-20 skip boundary, so no CUDA API trace or representative replay
evidence exists. Exit 1 and clean teardown are retained; no fallback or feature
removal was applied. All-rank exceptions do not represent eight independent trials.

The [scoped lifetime diagnosis](artifacts/raw/LANE-A-LIFETIME-DESIGN/001/handoff.md)
traces the INFERENCE mask producer followed by a separately compiled FORWARD decoder.
Installed Trees can end an automatic generation between them before any backward
is pending, invalidating the still-needed mask. The selected public-API candidate
marks a generation once before each complete optimizer update, covering all four
microbatches, accumulated gradients, loss, clipping and AdamW. A per-microbatch mark
could invalidate accumulated gradients or retained losses; a mark after mask creation
would be misplaced. One CPU counter update adds no tensor work, synchronization or
communication, but longer capture paths/pool retention may cost memory and setup.
The opt-in marker callback is implemented and independently reviewed with no
material source blocker. A guarded independent batch passed 26 tests; the author’s
final callback/continuation batch passed 25. The earlier broad module selection
reported 123 passes but may invoke a pre-existing hardware query/init test; the
blanket no-CUDA claim was withdrawn. The reviewer’s broad selection separately
reported 142 passes and a continuation-fixture teardown error after that hardware
path, then a guard rejected import-time availability discovery before collection.
Both harness failures are retained. The final narrowed batch stubbed discovery
and guarded CUDA initialization, passed, and exited; no callback failure was
established. [Review, raw outputs and limits](artifacts/raw/GRAPH-STEP-REVIEW/001/handoff.md)
are retained. The subsequent GPU capture result is recorded below. The reviewed source is archived as [snapshot 003](artifacts/raw/SOURCE-SNAPSHOT/003/manifest.json),
22 changed source files with archive SHA-256
`ea215abaac14d98a21c36e8684dbd8ce88a1d48f47cec5fe4f85f61be5dd4e27`.
Keep the existing failure as its control; use the same marker prerequisite graph
off/on if it becomes runnable. No universal
graph incompatibility, runtime improvement or correctness is established.

The historical [strict-failure diagnostic contract](artifacts/raw/STRICT-FAILURE-DIAGNOSTIC-DESIGN/001/contract.md)
selects one genuine first-training-microbatch sparse-WGRAD sample per rank, rather
than synthetic startup warmup. It will replay the installed row2col and sparse GEMM
boundaries independently from immutable actual operands, preserving native outputs.
This design has no runtime evidence and is not scheduled: baseline nondeterminism
removal is outside the updated scope. It uses pinned runtime interception points,
including a private saved CustomOpDef implementation guarded against version changes;
CPU behavior, compile integration, memory and actual CUDA execution remain gates.

### Recurrent corrected graph-off fault

The corrected 71-update noise control exited 1 at 22:50:12 UTC. All eight ranks
completed optimizer update 32; rank 6 first reported a sticky CUDA illegal-instruction
error (715) during update 33. A later NCCL event-query/watchdog failure propagates
the asynchronous fault. The Python traceback detects it at a Triton launch in the
compiled backward; “sticky” explicitly prevents assigning that launch as the origin.
The preceding region includes communication, sparse FP8 GEMMs and attention backward.
The [scoped diagnosis](artifacts/raw/RECURRENT-CUDA-FAULT/001/handoff.md) verifies
that failed rank 6's E=4 scan SASS matches the corrected successful smoke and has
no explicit trap instruction; this does not clear other kernels or latent corruption.
No further version-matched source fix is established. A proposed bounded diagnostic
would wait on a compute-stream event immediately after native
`_run_on_comm_stream` has joined communication inside `combine_bwd`, before its
remaining Python fallback, separating that executed prefix from later Flex/RoPE work.
It is not implemented or scheduled ahead of the capture result, changes timing and
may hide a race; no device-wide or pre-communication wait is selected.
No valid 50-update window exists; control 002 was not launched. After teardown,
all exact task PIDs were absent and all eight GPUs were idle. This failure is not
numerical variability or evidence that graphs are unprofitable. The one-bound
correction remains a justified source-UB fix, without establishing fault repair.

The [exact diagnostic contract](artifacts/raw/RECURRENT-CUDA-FAULT-DIAGNOSTIC-CONTRACT/001/handoff.md)
now identifies a guarded experiment-only wrapper around the task module’s helper,
selected by the pinned native lambda’s module/path/line/qualified name. This
intercepts the actual registered body without replacing its framework kernel or
adding an ordinary-path hook. It uses native update/microbatch counters and
post-join no-timing events only in updates 21–40. The [implementation](artifacts/raw/COMBINE-BWD-COMPLETION-IMPLEMENTATION/001/handoff.md)
and [independent review](artifacts/raw/COMBINE-BWD-COMPLETION-REVIEW/001/handoff.md)
passed 38 guarded CPU tests plus scoped lint/whitespace checks. Review caught and
resolved the real module-path guard, zero-call active-microbatch evidence and
flush-error cleanup. Earlier failed test outputs are retained. Normal capture
has no diagnostic activation; actual GPU activation, closed-extension peer
progress and one graph-off attempt remain open behind capture priority. No
diagnostic execution or fault repair is claimed.

### Marked capture result and next ownership question

The reviewed-marker attempt 003 exited 1 after one failed first-microbatch forward,
with no optimizer start/end on any rank. All eight ranks report:
`Detected 1 tensor(s) in the cudagraph pool not tracked as outputs. All live allocations must be tracked for correctness.`
This is the post-model warmup pool check in the **first dense decoder block**,
before HybridEP. The prior invalid-mask-input exception was not observed; complete
mask lifetime through all microbatches/backwards is still unproven. Rank 5 has the earliest recorded structured exception and first supervisor
traceback, not rank 4 as initially reported from a tail excerpt; this does not
identify an originating GPU/kernel or assign fault causality.

A monitor-controller `json.dumps(..., flush=True)` typo occurred after Popen spawned
the single native run; it is retained separately from the genuine training error.
The monitor reattached to PID 510132; no second launch occurred. After exit,
all eight GPUs were idle and exact runner/sampler PIDs absent.

Source inspection identifies a specific ownership hypothesis: the dense GEMM library
retains a stream-keyed 32-MiB workspace allocated during private-pool warmup, while
its guard only rejects active stream capture. Before attempt 004, pointer ownership
was unproven. Trees uses a dedicated stream, so ordinary eager prewarm
on the default stream would not seed this cache. Do not bypass the pool correctness
guard. The [bounded pool diagnosis](artifacts/raw/LANE-A-POOL-DESIGN/001/handoff.md) selects
one built-in allocation-history diagnostic before a minimal workspace/actual-operator
capture-safety remedy. The pinned Trees history flag prints allocation frames on
the unchanged pool guard. Attempt 004 ran once at 23:50:05–35 UTC with that built-in
history option and source snapshot 003. It reproduced the same guard. An initial
supervisor excerpt showed only unsymbolized frames; that preliminary reading was
corrected by the full per-rank `error.json`. All eight ranks link the guard’s bad address
`597168226304` to `at::empty`, CudaCherry `get_cached_workspace` and `gemm` allocation
frames. This confirms the workspace producer. The existing observation route does
not expose the loaded extension’s runtime path/hash; binary identity remains a
separate provenance limitation. The [complete attempt record](artifacts/raw/LANE-A/attempts/004/handoff.md)
retains full rank errors and the corrected preliminary reading.
After unwind, the exact runner/sampler/worker PIDs were absent and all eight GPUs
were idle. There is no automatic second history run. Research now checks whether a
task-local public capture-unsafe operator boundary can isolate the known hidden
workspace behavior while preserving compiled surrounding work and the FP8 backend.
FlexAttention, SAC, FP8 and HybridEP remain unchanged.

### Selected dense-GEMM safety remedy

The [scoped design](artifacts/raw/DENSE-GEMM-GRAPH-SAFETY-DESIGN/001/handoff.md)
selects a task-owned opaque adapter with the public `torch.Tag.cudagraph_unsafe`
tag around the unchanged vendor GEMM. The installed partitioner recognizes that
outer operator; the intended effect is to keep its hidden workspace allocation
outside graph partitions. Native capture must verify that effect.
The functional route and a truthful mutable, None-return companion preserve
supplied-output and accumulate semantics; the existing FP8 autograd.Function
remains the gradient owner. Both old and new operator identities retain the
selected SAC policy's recompute treatment. No installed registration, backend,
precision, algorithm, workspace size or correctness guard changes.

This is a capture-safety prerequisite in both graph-off/on variants. It may add
operator dispatch and graph-partition boundaries, retain extra intermediates or
reduce capture coverage. It adds no intended tensor copy, synchronization or
collective on the preset's functional path. The production adapter is implemented
and frozen after review. The [implementation checks](artifacts/raw/DENSE-GEMM-GRAPH-SAFETY-IMPLEMENTATION/001/handoff.md)
passed all eight focused tests, required Ruff F821/F811/F841 and whitespace checks.
An independent probe confirms that actual FX tracing retains the tagged outer
operator and installed Inductor recognizes it as unsafe; QTensor fake metadata
and production CPU rejection also pass without CUDA initialization. Review
is [complete with no material source finding](artifacts/raw/DENSE-GEMM-GRAPH-SAFETY-REVIEW/001/handoff.md).
Three earlier test commands failed because of test-double/oracle issues and an
inherited unsupported compiled-mutation expectation; their commands, counts and
causes are retained, but full raw outputs were not saved. Compiled supplied-output
behavior remains unclaimed; the preset uses functional GEMM calls. Native capture
failed in attempt 005 on a different allocator-pool invariant before any optimizer
update: `These storage data ptrs are not allocated in pool ... should be: ...`.
The previous hidden-workspace guard was not the reported failure. Scope and cause
remain under diagnosis; this is not proof of complete mask/partition lifetime
correctness. Native unwind finished at 00:22:43 UTC; exact task processes were
absent and all eight GPUs idle at 00:23:19 UTC. The live `/proc` extension-mapping
read raced with process exit and failed with ENOENT, so that provenance gap remains.
No retry or new GPU run is scheduled before the scoped diagnosis. The [frozen snapshot 004](artifacts/raw/SOURCE-SNAPSHOT/004/manifest.json)
pins 25 changed source files; all 22 prior source-003 files are unchanged, plus the
adapter, alias integration and focused test. Archive SHA-256 is
`9ebf6ca337bd6a80b9d691df654d60dc458af3514e574c444e87745cc2674f74`.
Source snapshot 003 and every earlier failed attempt remain preserved. Numerical criteria and performance are not inferred from this design.

### Nested HybridEP tuple capture blocker

The [source/runtime diagnosis](artifacts/raw/LANE-A-PARTITION-POOL-DESIGN/001/handoff.md) narrows attempt 005 to an internally generated partition
whose top-level input is the full HybridEP dispatch tuple. The partition unpacks
and returns eight tensor leaves. Installed Graph Trees checks only top-level
Tensor inputs, overlooks that tuple, and misclassifies the returned aliases as
new private-pool allocations. Its pool check reports eight outside-pool pointers
on all ranks. The dense-GEMM unsafe operator is correctly outside captured
partitions; its earlier workspace guard is not this failure.

No established public internal-partition flattening hook has been identified.
The selected policy uses existing per-block compilation to set graph policy
off only for the 17 MoE-containing blocks, retaining compilation and unchanged
non-graph options. The dense block and compiled loss retain the global graph-off/
on policy. This reduces coverage and potential benefit; it preserves HybridEP,
FP8, FlexAttention, SAC, the objective and accumulation. The [implementation](artifacts/raw/MOE-GRAPH-REGION-IMPLEMENTATION/001/handoff.md)
and [independent review](artifacts/raw/MOE-GRAPH-REGION-REVIEW/001/handoff.md) are
complete at source level: 13 focused tests, the expected regression-sensitivity
failure followed by restoration, and 110 experiment/config checks passed. Review
caught and resolved test CUDA-function leakage and missing config-golden updates.
All 58 tracked experiment snapshots change only by the intended default empty
exclusion list; the existing continuation experiment receives its missing canonical
golden. Preset snapshots and production preset bytes remain unchanged. Lint,
whitespace and 28 documentation links passed. The actual A006 overlay also decoded
with all other workload settings equal to the preset plus identical short-run CLI.

[Source snapshot 005](artifacts/raw/SOURCE-SNAPSHOT/005/manifest.json) pins 88 files
including those golden changes; all 25 prior source-004 files are unchanged.
Archive SHA-256: `e0ee8a9880bff618b44ca05b83710f9a8cb00f4211a915f468e9652f68d49192`.
Native attempt 006 failed in the first backward before any optimizer update.
All eight ranks raise `UnboundLocalError` for `buf66` while generated Inductor
code constructs `partition5_args`. The [completed diagnosis](artifacts/raw/LANE-A-BACKWARD-CODEGEN-DESIGN/001/handoff.md)
finds a mixed-order RMSNorm reduction renaming its result to `buf67` without
updating producer/consumer partition signatures. The actual result stays local
to partition 4 while partition 5 receives undefined `buf66` and reads `buf67`. This is a generated-code
error, not a CUDA-kernel failure or numerical difference. All required 18 compiled
callables/SAC wrappers, 17 HybridEP dispatchers and target groups were active.
The incomplete timing window supplies no throughput result. All eight GPUs were
idle after unwind.

The live maps observation now identifies the loaded CudaCherry GEMM binary as
`cudacherry/gemm/_C_gemm.cpython-312-x86_64-linux-gnu.so`, 13,384,960 bytes,
SHA-256 `9548e0fdb0be6033a4bbd0e159ee9b04e20fcc12874b0d968e497ea18c53aaa5`.
This closes the previous loaded-path/hash observation gap; it does not provide a
source-to-binary build attestation. Capture, numerical, continuation and
performance acceptance remain open.

The next bounded viability attempt uses the installed
`TORCHINDUCTOR_MIX_ORDER_REDUCTION=0` before importing Torch. Source inspection
shows this prevents the specific faulty mixed-order generator path while
ordinary reductions retain fp32 accumulation. It changes reduction order,
fusion/scheduling, intermediates and potentially launches/memory traffic; it is
not an identity or performance claim. The same setting is required in graph-off/
on controls, with the original baseline retained separately. The earlier design
rejection was too strict and is explicitly superseded in its amendment.

Attempt 007 launched once at 01:32:38 UTC with the same 17 MoE exclusions and
dense/loss graph eligibility; no compiler flag sweep or package patch is selected. Its
bootstrap records the effective option on every rank. Its shared graph-off/on
CPU probes passed without CUDA initialization; among the three recorded options
only `triton.cudagraphs` differs. Full generated-kernel equality remains a runtime
attribution gate. The first graph-on-only bootstrap probe is retained historically;
the final shared bootstrap has SHA-256
`ffde3588d53012c7155fb5b6961a06be96095b44b8bc55c97a9808b2446ba065`.
[Integrated source snapshot 006](artifacts/raw/SOURCE-SNAPSHOT/006/manifest.json) pins
92 files; 86 prior source-005 files are unchanged, and the only prior changes are
the callback-type pin and maintained campaign README. The four new diagnostic
files are independently reviewed, inactive in this capture entrypoint. Archive
SHA-256: `324cad36a75404f235f5dd7de10b571a75ad486276a44c09fe425f3187ebd51b`. If this causal remedy cannot make the
surviving regions viable, loss-only capture remains one conditional finite
fallback; actual loss forward/backward replay is required, and mask-only capture
cannot satisfy it. Attempt 007 subsequently completed all 25 updates without the
prior backward code-generation error. Trace review must still establish actual
forward/backward replay and recurring capture costs; successful execution alone
does not establish capture, speed, parity or continuation.

### Successful capture with recurring recording

The [all-rank A007 mechanism review](artifacts/raw/LANE-A/analysis/007/mechanism-review.md)
correlates 348 actual graph launches with 1,560 GPU kernels per rank across three
profiled updates. Per rank these include 48 dense-forward, 96 dense-backward,
96 chunked-loss-forward and 96 loss-backward launches, plus 12 block-mask launches.
All ranks re-record dense backward on all 96 calls. Most other dense/loss launches
also follow new captures; intermittent reuse on three ranks does not settle.
Static-input pointer guards continue near shutdown. The reviewed public update
marker permits FlexAttention mask use through native updates, while the remaining
address/tree behavior prevents persistent representative reuse in this attempt.
Its precise address producer remains unproven; do not assign every mismatch to
FSDP without targeted evidence.

The run generated about 927 MB of verbose supervisor output. Its profile and
logging costs prevent a performance conclusion. A short same-fix, unprofiled
off/on diagnostic pair is being prepared: 26 native updates, measuring updates
21–25 and closing at 26. It includes recurring capture costs, with extra compiler
logging and profiling disabled in both variants. It answers whether that policy
has a material integrated cost without those diagnostics, or instead warrants
further reuse investigation. A fault stops the pair for diagnosis. It is not a
replacement for production A/A noise, 50-update A/B/A/B, numerical/continuation
checks or the 500-update finalist gate. See the [measurement protocol and limits](artifacts/raw/MATCHED-PERFORMANCE-DESIGN/001/handoff.md).

The independent review recommends the already bounded loss-only fallback if
A007 is discarded: exclude the first dense block as well, preserving all 18
compiled SAC blocks and actual FlexAttention/FP8/HybridEP execution. Removing the
unstable dense graph ancestor might permit reusable chunked-loss trees; this is
a hypothesis, not established fallback capture. Block-mask-only reuse is
insufficient, and no further region-removal sweep is selected.

### Loss-only reuse and the next failed control

A008 resolves that bounded mechanism question. Every rank's profiled updates
21–23 have 32 actual CE forward and 32 backward graph launches per update,
with the same eight graph IDs per phase reused and zero capture/instantiate
calls across all trace categories. The additional four launches per update
are block masks. No model, head GEMM, optimizer or communication graph capture
is claimed. All 18 model blocks remain compiled with SAC, FlexAttention, FP8
and HybridEP. Independent [trace attribution](artifacts/raw/LANE-A/analysis/008/mechanism-review.md)
and [native/log validity](artifacts/raw/LANE-A/attempts/008/monitor-handoff.md)
pass. The [copy-cost assessment](artifacts/raw/LANE-A/analysis/008-copy-cost/report.md)
resolves actual staging traffic across all 24 rank/update observations: each
rank/update copies 25 GiB BF16 logits and 1 MiB labels before its 32 CE-forward
replays. Those device copies total 17.605–17.641 ms; this is activity time, not
an integrated wall-time tax. Backward stages one FP32 scalar per replay, with
no large copy observed on that path; mask copies and inherited FSDP copies are
accounted separately. Host/device correlation and same-stream order establish
input-staging attribution. Cumulative bytes and graph-node counts do not imply
simultaneous residency. Paired pool/headroom and unprofiled integrated cost
remain open; these copies supply no qualifying new host gap for lane B.

The matched C0a/C1a/C0b/C1b plan uses 71 native updates and observes 21–70.
The two off controls supply A/A noise. Its first graph-off process instead
fails at update 48, rank 3, first microbatch: structured error at 03:03:53.706
UTC, followed by a CUDA/NCCL event-query illegal instruction. The observed
backward stack includes `cudacherry.quantize_blk128x128` during SAC recomputation;
asynchronous reporting does not identify the originating kernel. All ranks
completed exactly 47 optimizer updates and 188 backward microbatches. Logs
are contiguous and contain no ENOSPC or NUL corruption. The observer cannot
close at 71, so its directory is empty and no window/noise result exists.
All exact task processes and GPU allocations were released; attempts 002–004
remain unreleased. This extends the recurrent graph-off fault evidence at
updates 22 and 33; it is neither graph incompatibility nor numerical variability.

The next [single completion-prefix probe](artifacts/raw/RECURRENT-CUDA-FAULT/002/handoff.md)
requires a fixed diagnostic window through 60, since the former 21–40/stop-40
contract cannot cover update 48. Its [narrow extension](artifacts/raw/COMBINE-BWD-COMPLETION-WINDOW/001/handoff.md)
passes 39 focused CPU tests, including active checks at 48/60, inactive 61 and
fixed native-stop validation. The expected old-window regression fails and is
restored; scoped lint/whitespace pass. Root reviewed the exact source006 delta:
only the bound/stop validation, affected tests and diagnostic README change.
The helper, stream joins, tensor arithmetic, autograd and selector are unchanged.
[Source007](artifacts/raw/SOURCE-SNAPSHOT/007/manifest.json) freezes 92 files,
89 unchanged; manifest SHA-256 is
`2585e504350276cb4689ae9ea45a33706e9f898bb61969abe5d08f2650c55620`.
The native 60-update diagnostic launched once at 03:58:32 UTC on 12 November
(controller 591470, 20-minute deadline) and exited 1 at 04:00:54 UTC. Fresh device,
source and storage checks passed; two topology-parser failures are preserved,
and root directly confirmed all 56 NV18 links before launch. Each rank completes
20 updates/80 microbatches before the active window. At update 21 every rank
records 25 active checkpoint enters and 24 completes: ordinals 1361–1384 complete,
then 1385 remains unmatched in microbatch 1. Rank 6 reports CUDA 715 at event wait.
These boundaries narrow the executed prefix, without identifying a kernel or
establishing the same cause as earlier graph-off faults. Added waits/references
change overlap, peer progress and lifetimes; this is no repair or performance
control. The [completed handoff](artifacts/raw/COMBINE-BWD-PROBE/attempts/001/handoff.md)
records 81 completed microbatches per rank and clean process/GPU release.
Checkpoint records lack timestamps: exact ordinal gaps/wait duration are
unavailable. Rank 6's whole failing microbatch spans 575.82 ms, which is not an
event-wait measurement. The original malformed extraction and incomplete PID
check remain preserved beside their canonical corrected receipts. A single
read-only hardware query reports zero ECC/remapper counters, with retired-page
fields N/A; it does not clear transient hardware, driver or kernel causes.

The [external binary-receipt helper](artifacts/raw/LOADED-BINARY-RECEIPT/001/handoff.md)
passes ten CPU fixture groups, lint, syntax and document checks. Live provenance
was **not collected**: monitoring began at 04:01:02 UTC, after process exit, and
no helper was invoked on dead PIDs. This is a monitor-timing failure, not evidence
about which quantize binary executed. No provenance-only rerun is authorized by
this result, and no installed package is changed.

A [targeted quantizer source assessment](artifacts/raw/RECURRENT-CUDA-FAULT/003/handoff.md)
finds no explicit runtime trap or demonstrated dispatcher, shape, stream or
local output-lifetime violation for the reported `(2048,1536)` call. The kernel
uses vector loads with an alignment contract, but no violating address was
recorded. The launch check can report an earlier asynchronous error. Nineteen
source hashes are retained; installed source does not attest the loaded binary's
build origin. No quantizer repair or extra fault sweep is justified by this
static evidence.

### Persistence failure and remaining observation work

The earlier quiet dense/loss pair's C1 interval 21–25 remains descriptive:
5, 242,880 useful tokens over 145.411524 seconds, versus 16.458212 seconds off.
Its whole-run completion evidence fails after NFS logging flush errors around
update 17. [Byte/sequence evidence](artifacts/raw/C1-STORAGE-LOGGING/001/diagnosis.md)
shows missing formatted records and duplicate persistence, plus NULs in stderr
and supervisor output; native JSONL files themselves have no NULs. Missing
markers do not prove a skipped optimizer update. Current free space and inodes
recovered, but the historical backend cause is unresolved. A008 subsequently
wrote intact logs/traces. The new matrix preflight also verified current free
space and a task-owned write/fsync; its failure is CUDA, not storage. No shared
cleanup or checkpoint deletion was needed.

The earlier [bounded numerical design](artifacts/raw/NUMERICAL-DESIGN/003/handoff.md)
proposed replacing an infeasible full reference archive with a conditional lossless
encoding: nonzero native S0 plus ten actual preclip gradient sets, estimated
158.186 GiB. That proposal required native clipping/AdamW decoding to match
actual same-run hashes at every boundary, not bitwise equality between independent
A/A runs. No recorder or decoder probe was implemented. Its decoder-specific
gates are superseded by the direct-reference decision below; actual shard/state
inventory remains required. One reference serves calibration and A/B consumers,
then is deleted with an inventory; continuation payloads are generated serially
afterward. The diagnostic preflight
later observed 140,509,184,000 bytes free (about 130.86 GiB), below this proposed
176 GiB payload cap plus 32 GiB reserve. That fluctuating observation is not a
reservation; full numerical archival was not storage-cleared at that observation. No
checkpoint payload was created by the completion diagnostic.

The later [RAM feasibility check](artifacts/raw/NUMERICAL-DESIGN/004/handoff.md)
observes 1,294.71 GiB cgroup headroom at 04:20 UTC, enough capacity for a proposed
192 GiB process-resident reference/decoder budget before the unmeasured native
host peak. It is shared, overcommitted capacity rather than a reservation. A
standard Python buffer/socket store could keep gradient bytes in RAM, without
a second training model or `/dev/shm` dependence; no server, allocation or recorder
is implemented. At 04:19:10 UTC NFS free space fell to 12.04 GiB, below even the
estimated 36.50 GiB native S0 checkpoint, before its 32 GiB reserve. This blocks the
proposed RAM-gradient/native-S0-disk route at that time. The owner was asked about expected
space or an existing task-owned persistent location while independent work continued.
Task raw evidence occupies 11,476,320,256 bytes; no unrelated files were inspected
or removed to address the shared-space shortage.

At 05:49:51 UTC, [fresh statvfs evidence](artifacts/raw/STATIC-CHECKS/009-storage-recovery/statvfs.json)
records 7,195,643,084,800 available bytes (6,701.46 GiB) and 2,915,869,677 free inodes.
The earlier storage shortage is no longer the current planning blocker. A
disk reference is preferred over adding a RAM server. Its native observation
contract is resolved before source edits.
Shared free capacity is not reserved; recheck bytes, inodes and write/fsync before
creating the numerical reference or continuation payloads. At that storage
decision no native training checkpoint had been created or deleted; the later
readiness attempt is recorded below.

The [minimal probe contract](artifacts/raw/NUMERICAL-DESIGN/005/handoff.md)
resolves native accumulated-gradient, inner-optimizer and post-scheduler hooks,
and actual serialized/pending RNG ownership. Its arithmetic-decoder and temporary
S0-sidecar choice is superseded by the [storage decision](artifacts/raw/NUMERICAL-DESIGN/006/handoff.md)
after capacity recovered. Retain one direct actual preclip/postclip/P/m/v reference
for ten native updates plus full native S0: nominally 644.912 GiB, within a
700-GiB total live-payload ceiling plus 32-GiB free reserve. Count inputs, state,
metadata, logs and partial files in the ceiling; shared capacity is not reserved.
Reuse it for production A/A and then A/B, and delete eligible payloads only after
all named comparison, diagnosis and review consumers finish. Actual parameter
delta metrics retain a bounded previous-P CPU cache, nominally 1.521 GiB/rank;
measure host peak/headroom, including reference/FP64 scratch.

This removes the custom CUDA arithmetic decoder, singleton-group equivalence and
partial-DCP-reader gates. Same-execution lossless payload/readback checks replace
encoding reconstruction; they remain separate from independent-run numerical
acceptance. Native inventory/readiness, complete S0/RNG restore, one-update GPU
pipeline verification before scaling, calibrated ten-update comparisons and
bidirectional full-state continuation remain required. Staged implementation is
assigned outside `gift/`; no recorder result or numerical
A/B tolerance is established yet.

The staged direct-reference recorder subsequently passes nine CPU behavior tests,
full Ruff and formatting. Its [independent integration and cost review](artifacts/raw/NUMERICAL-PROBE-REVIEW/001/REVIEW.md)
returns **HOLD** before target GPU execution. A native CPU reproduction shows
that `aux_loss_scale` is reconstructed before each forward and is absent from the
checkpoint; comparing its post-update value against the fresh restored value
incorrectly rejects native restore. A second fault-injection reproduction shows
that a secondary failure-record write can bypass collective failure publication
when storage remains unavailable. The review also finds redundant post-update
P/m/v reads adding three full tensor families (nominally 36.50 GiB global D2H per
update). Corrections must preserve the inventory and native auxiliary weighting,
publish failure despite unavailable diagnostic storage, and reuse existing payload
hashes while accurately documenting remaining traffic. These are recorder defects,
not evidence of baseline numerical instability or a graph incompatibility. The
original failed reproductions and review remain evidence; no GPU probe or native
checkpoint has run. Launch admission must protect the native S0 save before the
observer's later storage check.

The [targeted correction review](artifacts/raw/NUMERICAL-PROBE-REVIEW/001/TARGETED-ACCEPTANCE.md)
subsequently resolves R1–R3: narrowly recognize native auxiliary-scale
reconstruction, preserve collective failure publication despite unavailable
diagnostic storage, and reuse existing endpoint hashes. Enabled observation
now has roughly eight large-family D2H reads versus five persisted families per
update, before small state/input reads; it remains invasive diagnostic work.
The corrected patch is [applied and checked](artifacts/raw/NUMERICAL-PROBE-INTEGRATION/001/checks.json).
The first combined check passes 33 tests but has a fixture teardown error when
the test replaces the CUDA RNG setter/owner. A new CPU regression fails the old
fixture and passes after the fixture captures its original restoration function.
The final combined recorder/continuation suite passes 34 tests with two known
deprecation warnings, full Ruff, format and syntax. The fix changes test code
only. Source009 freezes the applied result; post-test queries show the same
eight H100s at zero use and no compute apps. Those checks launch no training;
the subsequent native readiness attempt is recorded below.
The two completed CPU/gloo fixture checkpoints are [deleted with a hash/byte inventory](artifacts/raw/NUMERICAL-PROBE-INTEGRATION/001/cpu-fixture-checkpoint-deletion-verified.json)
(six files, 107,612 bytes); their logs and regression evidence remain. An initial
selector matched no paths and is preserved as a non-deletion receipt.

The separate [launcher review](artifacts/raw/NUMERICAL-NATIVE-PROBE/review/001/REVIEW.md)
holds execution: a missing config-check parent blocks startup, timeout/session
ownership misses separately sessioned native ranks and their RSS, and preflight
lacks current NV18/software identity checks. Correct these before fresh preflight
and root release. A 30-minute deadline per phase is the chosen bounded stop policy,
including startup and substantial checkpoint/readback I/O; expiry is not an
automatic extension or retry. This launch work does not reopen producer review
or change numerical criteria.

Launcher correction checks subsequently pass three CPU process tests and both
Source009 config resolutions with zero warnings, without creating attempt or
checkpoint directories. The CPU software/import check also passes. However,
[targeted review](artifacts/raw/NUMERICAL-NATIVE-PROBE/review/001/TARGETED-CORRECTION-REVIEW.md)
reproduces two remaining defects: a child adopted by the subreaper between scans
can survive a falsely successful owner exit, and valid topology reaches an
integer/string matrix-key mismatch. Execution remains **HOLD**. After these two
preparation review failures, the method is reassessed: retain the validated native
commands/configs, fix only owner-root ancestry/initial-registration release and
GPU-label indexing, and verify the exact failed behaviors. No broader supervisor
extension, GPU attempt or automatic retry is selected. Earlier passing fixtures
and both failed review reproductions remain evidence.

The [final targeted acceptance](artifacts/raw/NUMERICAL-NATIVE-PROBE/review/001/TARGETED-ACCEPTANCE.md)
resolves those two defects with the exact adopted-child/initial-registration
and topology behavior regressions. Fresh preflight at 08:55:11 UTC verifies the
same eight idle H100 SM90 UUIDs, all-pairs NV18, no compute apps, Source009/import
identity, NFS fsync/readback and 6,349.76 GiB available. Root separately confirms
the actual venv prefix, Torch git/CUDA build and NCCL/DeepEP/DeepGEMM/CudaCherry/
NVSHMEM versions against the established environment. The root launch release
pins all 12 preparation artifacts and exact runtime object.

The first controller invocation at 08:56:44 stops before training: the task-owned
`attempts/` parent is absent, so its unique-child `mkdir` fails. [Failure record](artifacts/raw/NUMERICAL-NATIVE-PROBE/preparation/001/root-launch-001-failure.json)
and log remain; no attempt output, native checkpoint or GPU rank is created.
Root creates only that missing parent, retains the reviewed scripts/config/source,
and manually launches [invocation 2](artifacts/raw/NUMERICAL-NATIVE-PROBE/preparation/001/root-launch-record-002.json)
at 08:58:01 UTC using the still-fresh preflight. This is a setup correction, with
no automatic retry. The two serial native readiness jobs remain bounded to
1,800 seconds each. The S0 phase then exits 1 after the native update/full step-1
save: all eight inventory handlers reject `torch.strided, torch.complex64` in
`after_checkpoint_save`. Observe is withheld by the serial failure gate. No
CUDA fault is reported in this bounded run; that does not clear earlier faults.
The [owner status](artifacts/raw/NUMERICAL-NATIVE-PROBE/attempts/001/create_s0/owner-status.json)
and independent [660-identity check](artifacts/raw/NUMERICAL-NATIVE-PROBE/attempts/001/root-release-check.json)
verify release, and [GPU queries](artifacts/raw/NUMERICAL-NATIVE-PROBE/attempts/001/root-postrun-gpu-query.json)
show the same eight devices idle with no compute apps. Source009 remained frozen
through execution. The completed native checkpoint is retained pending diagnosis
and fix review; its observer inventory/readback is not accepted. A scoped
lossless-codec correction and new isolated execution preparation are underway,
without changing training math or the preset. No numerical, graph or performance
pass follows from release, launch or a failed recorder.

The [reproducible attempt analysis](artifacts/raw/NUMERICAL-NATIVE-PROBE/analysis/001/REPORT.md)
confirms on every rank four fetch/backward completions, one optimizer completion
and one checkpoint-save completion, with 1,048,576 global useful tokens. The nine
native checkpoint files total 39,211,598,683 bytes (36.518647 GiB). Metadata covers
model, native optimizer, trainer/scheduler, eight loader ranks and eight owned
RNG objects; matching moment/model shapes do not establish value or restore
parity. No valid S0 rank report, actual serialization-time comparison record,
native-state sidecar or tensorpack exists. Existing observe correctly refuses
that incomplete S0 record, so a new valid S0 through the supported path is needed.
No repair mode or partial checkpoint reader is introduced.

Nine resource samples show a maximum sampled owned RSS sum of 176.777 GiB and
per-device sampled maxima of 28,729–33,043 MiB. RSS can double-count shared pages;
these are sampled readiness observations, not instantaneous or stable-window
peaks. Preserve the checkpoint until the codec owner, root and independent fix
review complete their named diagnosis; later authorized deletion must retain the
task's path/size/hash inventory. Analysis did not reread large payloads. Cleanup's
required payload-hash inventory supplies a concrete reason for that read once
eligibility is established. Earlier data-identity evidence remains reusable;
this attempt does not generate a new dataset digest or a numerical comparison.

The [selected HybridEP fault-site review](artifacts/raw/RECURRENT-CUDA-FAULT/006/handoff.md)
decodes the retained fused/blocking one-node specializations and finds that all
15 explicit backend device assertions are excluded by that JIT route. No explicit
trap or timeout is found in the selected backend/helper source. Reachable barrier,
chunk and mbarrier waits have concrete equality/parity contracts, but no captured
flag value demonstrates a violation. The retained scan correction remains exact.
These source findings identify no new repair or justified retry, and do not rule
out generated instructions, corruption or another kernel. Fault origin remains open.

Separate method research confirms a standard localization option:
[`CUDA_LAUNCH_BLOCKING=1` in PyTorch 2.15](https://docs.pytorch.org/docs/2.15/notes/cuda.html#asynchronous-execution)
and the [CUDA 13.0 programming guide](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#concurrent-execution-between-host-and-device)
make launches synchronous for debugging. The selected next fault-localization
hypothesis, after the bounded readiness work, is one graph-off native 60-update
run with that setting, no sanitizer/completion observer/timing/profiler/checkpoint,
and all required training features preserved. It needs its own reviewed release
and fresh preflight; it has not run. A 20-minute limit is an operational stop,
with no automatic extension/retry. A fault closer to its launch would direct the
next kernel/source investigation; completion without a fault would show only
non-reproduction under changed scheduling. A hang/unsupported blocking interaction
or timeout remains inconclusive. This is a diagnostic environment change, not a
production fix, performance control, or clearance of the original asynchronous
path. The native source and existing failures remain unchanged.

The [streaming comparison contract](artifacts/raw/NUMERICAL-COMPARISON/001/handoff.md)
keeps one reference for A/A calibration and later A/B. Consumers retain one
previous-parameter CPU family and emit compact FP64 elementwise-difference and
actual-delta metrics; they do not write another full reference. This is a design,
not an implemented comparison or numerical pass. Before generating ten updates,
the producer must retain actual small auxiliary totals and router counts/bias,
and classify necessary mutable state from the one-update inventory. Their
current hashes cannot quantify floating differences or count/reset behavior.
These additions do not block the bounded native inventory/readback probe. Metric
outputs are specified in advance; numerical limits remain unset until measured
production A/A behavior and precision justify a frozen criteria manifest.

The [phase-memory design](artifacts/raw/PHASE-MEMORY-DESIGN/001/design.md)
reuses native raw allocator statistics before their existing resets. It defines
an initial startup/capture-settling envelope through update 20 and reset-aligned
steady intervals afterward, without per-step synchronization or new resets.
The [staged implementation](artifacts/raw/PHASE-MEMORY-IMPLEMENTATION/001/REPORT.md)
initially passes nine CPU tests. Independent review reproduces R1: invalid
snapshots were promoted to complete, yielding misleading phase envelopes. The
corrected staging passes eleven tests, with both new regressions failing the old
guard. [Targeted independent verification](artifacts/raw/PHASE-MEMORY-REVIEW/001/REVIEW.md)
passes both affected regressions and accepts the correction; native call/reset
counts are unchanged. It adds an opt-in native-stat adapter/entry and UTC brackets
for existing timing boundaries. Lint, syntax, hashes and clean apply checks pass.
After the Source007 attribution attempt and independent GPU release, the exact
accepted patch is [applied and checked](artifacts/raw/PHASE-MEMORY-INTEGRATION/001/checks.json):
11 CPU tests, scoped Ruff and syntax pass. Source008 pins all 95 paths; other
91 Source007 files are unchanged. Actual GPU verification remains open.
The subsequent full Ruff/format gate initially fails on one test E731 and
formatting in the three new Python files. Those failures are retained; the
[mechanical correction](artifacts/raw/PHASE-MEMORY-INTEGRATION/002-format/checks.json)
preserves their production AST and passes all 11 tests, full Ruff and format.
No numerical or native metric behavior changed; Source009 includes this correction.
The pinned allocator-name query was verified
to make no CUDA call or device selection. Sampled device peaks are lower
bounds, not exact continuous device peaks. Source006 measurements are retained.

The [independent prefix review](artifacts/raw/RECURRENT-CUDA-FAULT/004/handoff.md)
finds no demonstrated observer-induced protocol/lifetime violation and selects
no source repair. It proposes one conditional native 60 Compute Sanitizer run
with documented user-kernel patching disabled, testing whether a hardware
exception can be attributed to a kernel/backtrace. The installed 2026.1 docs do
not guarantee exception reporting in that debugging setting or a precise PC;
this is a hypothesis, not a supported hardware-only-mode claim. The [prepared runner/watcher](artifacts/raw/SANITIZER-ORIGIN/attempts/001/handoff.md)
passes seven CPU watcher fixture groups, including real native log prefixes,
plus syntax/source-pin checks and ten reused helper fixtures. Its initial
missing sanitizer hash was corrected and the failed check preserved. The watcher
and runner were unstarted at that preparation stage. [Independent implementation review](artifacts/raw/SANITIZER-ORIGIN/review/001/REVIEW.md)
holds release for four findings: a real pin-schema mismatch hidden by synthetic
fixtures, non-atomic receipt publication without acknowledgement, incomplete
early/separate-session cleanup, and blocking helper execution suspending the
storage guard. [Targeted correction review](artifacts/raw/SANITIZER-ORIGIN/review/001/CORRECTION-REVIEW.md)
accepts F1, F2 and F4, with five focused fixture groups and seven current hashes
passing. Its remaining F3 unseen-orphan reproduction is preserved. The narrow
exact-entry/PID-start-time correction passes [final independent verification](artifacts/raw/SANITIZER-ORIGIN/review/001/FINAL-F3-REVIEW.md),
including false-match/PID-reuse rejection, parsed rank identity merge and final
rescan failure handling. Earlier failed pins/setup checks remain preserved.
Fresh [root preflight](artifacts/raw/SANITIZER-ORIGIN/attempts/001/preflight.json)
passes exact eight UUIDs, zero memory/use/apps, all 56 NV18 links, source007's
92 paths, native Python/import/version identity and NFS write/fsync/inodes.
Available NFS is 6,630.84 GiB at 06:18:39 UTC, not reserved. Root released one
attempt at 06:19:17 UTC; watcher 620912 was READY and its receipt-bound ACK
accepted runner 620913 before CUDA execution. Actual coverage, provenance,
attribution and release outcomes are recorded below. The 64 MiB allowance covers log/receipt outputs,
not unmeasured scan/compiler-cache growth. An active 4 GiB floor is a stop
condition, without reserving shared storage. Device ownership/idle and live
watcher READY/ACK guards passed before launch. No synchronization sweep or automatic full-instrumentation fallback
is selected.

### Attribution attempt outcome

[SANITIZER-ORIGIN/001 extraction](artifacts/raw/SANITIZER-ORIGIN/analysis/001/handoff.md)
records timeout 124 at 06:39:17.919938 UTC. All eight native logs have contiguous
seq0–1255, optimizer completions 1–16, and 68/67 forward-backward starts/ends.
Update 17's fourth microbatch is interrupted. No native CUDA error or 715 appears;
the supervisor traceback reports deadline SIGTERM. The run never reaches the
earlier fault updates 21/22/33/48 and cannot clear them.

The tool prints exactly 50 NCCL initialization API reports: 18 fatbinary,
17 relocatable-PTX and 15 ISA90 code-availability errors. Native work continues.
The count fills `print-limit 50`, with no footer or rank/PID binding; actual
eight-rank exception coverage and later-report visibility remain unverified.
There is no evidence proving a later error was suppressed, no kernel PC and
no demonstrated cause of the earlier fault. This is inconclusive attribution,
not memory correctness or an unsupported-stack verdict.

The one-shot helper successfully captures stable same-run mappings/hashes for
quantize, requantize, SwiGLU and GEMM shared objects on every rank. Operator
body, executed branch, build-source attestation and causality remain unproven.
The [bounded ELF/install metadata check](artifacts/raw/RECURRENT-CUDA-FAULT/005/handoff.md)
confirms current quantize/GEMM hashes match that receipt and package RECORD.
Build IDs, GCC metadata and host DWARF paths identify the local build artifacts,
but provide no source revision/content digest. This limits claims linking the
installed CUDA source bytes to the faulting binary; no cause is established.
The watcher's deadline cleanup records `verified_exit=false` after rank0 argv
changes during exit; retain that failure. Root's separate
[complete release receipt](artifacts/raw/SANITIZER-ORIGIN/attempts/001/gpu-release-check-complete.json)
at 06:40:46 records all 638 observed PID/start-time identities absent/no-live
and the same eight GPUs idle, with no root signals. No checkpoint was created,
and no retry or automatic instrumentation fallback is selected.

## Finite lane dispositions

The [bounded lane review](artifacts/raw/LANE-DISPOSITIONS/001/handoff.md)
retains A008 as a runnable persistent-loss mechanism with numerical,
continuation, repeated performance and memory acceptance open. A007 proves
dense capture churn, without a final incompatibility or profitability verdict.
Lane B remains unattempted: no material uncovered host-submission region with
a legal integration contract is demonstrated, and A008 already captures CE
forward/backward. Lane C remains unattempted: blocking counts exist, but a
material remaining replay opportunity and a complete legal Hopper count-consumer
path are unproven. Neither disposition establishes universal incompatibility.
The all-incompatible completion exception does not apply because A008 runs.

## Historical verdict and former remaining gates

**No verdict yet.** Native smoke and GPU timing-observer integration passed.
Strict deterministic A/A execution completed with exact inequality, retained as
diagnostics; bitwise parity is no longer required.
Representative all-rank baseline traces passed. The first longer unprofiled control
failed at update 22, so baseline noise is unavailable. The scoped corrected-backend
smoke subsequently completed 25 unprofiled updates; corrected strict A/A also differs diagnostically,
and sustained stability still needs separate checks. Persistent loss forward/backward
replay now passes, while its graph-off long control fails at update 48. Detailed
correctness, continuation, repeated throughput/memory and sustained-execution gates remain open.
No speedup, incompatibility, or production acceptance is claimed.
