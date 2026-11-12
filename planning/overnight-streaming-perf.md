# Overnight streaming and optimization campaign

Status: PLANNED — authorized for execution by the separate launch prompt; no run
has been started while drafting this plan. Starting state: 13 November 2026,
GIFT source as recorded at task commit `ee965d3c8e019f73c291104f345c40458a3f9f31`.
Goal: have reproducible measurements and per-candidate conclusions available at
the next morning handoff, prioritizing actual results over diagnostic scaffolding.
No exact clock deadline was supplied: record start time in Europe/Moscow, proceed
through the finite scope below, and leave durable partial results on interruption.

## Outcome and authorization

1. Establish the existing crop on GIFT `streaming_text` using tokenized local MDS.
2. Measure warmed, unprofiled baseline TPS and noise; then capture one complete
   optimizer update on global ranks 0 and 1 only.
3. Analyze both traces with Perfetto MCP and map observed bottlenecks to all
   mechanisms in `references/`.
4. Port and evaluate applicable mechanisms sequentially, with correctness checks,
   repeated TPS measurements and sparse trace evidence for each executed variant.
5. Deliver a table explaining each change, measured effect, mechanism, validation
   status and disposition; integrate only qualifying improvements.

The user explicitly authorizes unattended work on the existing eight-H100
`sr008-gpu` allocation and task checkout
`/workspace-SR008.nfs2/vkropotin/optimization-task`, including changes under its
`gift/`, runners, tests, MDS preparation, dependency/tool installation needed for
this campaign, scoped blocker repairs, owned-artifact cleanup, commit/push and
result publication. Do not wait for user answers to routine implementation choices.

Preserve other users' processes and files. Installation authority is for an
isolated personal/task environment or dependency copy, not modification of shared
`/opt/torchtitan-venv`, the driver, shared services or another user's environment.
Do not allocate another GPU job, expand beyond the existing eight GPUs, or use
unrelated private credentials. Preserve original workspace training checkouts.
Use existing authorized access; never guess replacement hosts. The eight GPUs run
one training experiment at a time. Resolve routine blockers within this scope;
an unavailable external capability is recorded, not bypassed by inventing evidence.

## Fixed workload and known starting constraints

Read [status](../docs/status.md), [baseline](../docs/baseline.md),
[candidates](../docs/candidates.md), [validation](validation.md) and `gift/AGENTS.md`.
Reuse the completed HF smoke, source tests and recorded failures as dated evidence.
Reconcile once with newer relevant commits/results and running processes on entry.

Keep D4096, eight trunk blocks, E128/H1536/top8/shared1, MTP1, microbatch 2,
sequence 4096, global 512, accumulation 32, EP8/GFSDP2, PP/TP/CP1, dense+sparse FP8,
CuTe, model/loss compile, AdamW, adaptive routing, production SAC, EMA and token
capacity 50,000. Seed2048 and scheduler horizon 65,000/warmup2000 stay fixed even
when training stops early. Performance measurement excludes runtime/compile
warmup, not optimizer LR warmup; record that the short runs remain in LR warmup.
Do not silently disable required features, shrink the model, change token capacity,
force routing, add gate/up z-loss or positive softcap, or turn off MTP.

Canonical monitoring interval is50. Preserve it in performance runs; interval 5
is permitted only for the short streaming smoke to exercise monitoring. Disable
validation, checkpoint saves and memory snapshots for this bounded campaign;
report that it excludes those production lifecycle costs and proves no resume
or convergence result. CUDA Graphs stay off; this is not a renewed graph campaign.

Known pitfalls requiring explicit handling:

- The bundled `tests/assets/c4_test_mds/pack_test_dataset.py` defaults to **2048**
  tokens and 384 rows. Do not use those shards unchanged for a 4096-token study:
  the current collator pads short rows and masks their labels.
- HF uses windows of 4097 tokens to produce 4096 input/target pairs; streaming
  truncates rows to 4096 and masks the final target. Its EOS-derived positions
  also need inspection. Same corpus/tokenizer does not imply identical batches
  or losses. Establish a new MDS baseline; do not compare its TPS/loss directly
  against the old HF smoke as a computation-preserving optimization.
- The native profiler currently lacks rank selection and hardcodes
  `record_shapes=True`. Do not invent `profile_ranks` YAML fields. Add a minimal,
  reviewed opt-in extension or task-local adapter using native scheduling/export,
  with profiling active only on ranks 0/1 and shapes/stacks/memory history off.
  Non-profiled ranks still execute every training step and collective.
- The environment was `/opt/torchtitan-venv/bin/python3.12`, torch
  `2.15.0.dev20260822+cu130`, with `/etc/profile.d/wm.sh` supplying CUDA compatibility.
  Verify current imports and versions; do not treat these notes as live preflight.
- Perfetto MCP availability on the executing agent is not verified. Check it early,
  including trace-path visibility. Installing/configuring a task-owned server or
  tools is authorized. Use existing access, no credential borrowing. Do not
  substitute a Python summary and mark the required MCP analysis complete.

## Progress ledger — update this file at milestones

Use NOT_STARTED, RUNNING, PASS, FAIL, BLOCKED or INAPPLICABLE with evidence.
PASS applies only to the stated gate. Keep one active-row next action and UTC/Moscow
timestamps; on resume check recorded processes before relaunching anything.

| Gate | Status | Evidence / next action |
| --- | --- | --- |
| G0 Environment, ownership, disk and Perfetto route | NOT_STARTED | Verify existing node, paths, interpreter, GPUs, MCP tools and source state |
| G1 Full-length MDS, data contract and streaming smoke | NOT_STARTED | Prepare 4096-token data; check sharding/labels/positions and all-rank native updates |
| G2 Warmed baseline A/A TPS and memory | NOT_STARTED | Fix measurement window after calibration; collect two controls |
| G3 Baseline trace pair and Perfetto bottleneck analysis | NOT_STARTED | Ranks0/1, same one complete ordinary update, two files total |
| G4 All reference mechanisms reconciled | NOT_STARTED | Fill coverage table below with source/trace evidence and experiment order |
| G5 Sequential applicable experiments | NOT_STARTED | Correctness, A/B/A/B, one trace pair per executed candidate |
| G6 Combined qualifying result and confirmation | NOT_STARTED | Integrate winners or retain MDS baseline; verify combined runtime and cost |
| G7 Report, state, Git publication and cleanup | NOT_STARTED | Reconcile every row, publish source/results and release owned resources |

Active execution: no PID/job yet. The executing agent records launcher/controller
PIDs, exact commands, source revision, environment, logs, current experiment,
completed gates, blockers and next action here. Keep numeric results in the run
matrix rather than repeating them in several documents.

## G0–G1: establish streaming without a long prerequisite campaign

Use `runs/overnight-streaming-perf/` for a single campaign report, `results.json`,
runner/parser, and per-experiment commands/configs/log pointers. Raw payloads and prepared data go under
`runs/overnight-streaming-perf/artifacts/raw/`, covered by the existing ignore rule;
verify `git check-ignore` before generating large files. Keep source changes inside task `gift/`; no nested
repositories. Preserve the old verified HF revision and receipt as historical
smoke evidence while establishing a separately identified MDS baseline.

1. Verify all eight GPUs are free for this task, no competing task-owned run is
   active, NFS is writable, and disk/inode capacity is sufficient. Inspect cleanup
   behavior before launch. Freeze source/environment for each measurement matrix.
2. Use the same local C4 JSONL and tokenizer as the existing smoke. Adapt the
   existing packer through its callable API into a new task-owned data directory,
   with full 4096-token rows. Inspect its `exist_ok` deletion semantics; never point
   it at existing shared data or the original bundled shards. Record corpus,
   tokenizer, row count/length, token dtype, packing policy and file hashes.
   Repeated local data are acceptable for this controlled perf fixture; label
   them clearly, never as a production mixture or quality evaluation.
3. Use the existing GIFT streaming preset as a starting point: FIM off, shuffle
   off, fixed canonical-node setting, infinite iteration. Start with its two
   workers; choose persistent workers/pinning only after supported-field checks
   and apply the same settings to every control/candidate. Verify partitioning
   and repeats over all 8 ranks, including epoch wrap. No tokenization in the
   timed training loop. Data decoding/collation/transfers remain timed.
4. Check real input/label shape `[2,4096]`, full row lengths, label shift, ignored
   targets, EOS/BOS/document positions, vocab range and sample fingerprints.
   For full rows the current collator yields 4095 valid main targets per sample;
   verify actual counts instead of assuming all4096 are supervised. Do not
   relabel padding as useful tokens. Any loader fix needs a focused regression.
5. Run the smallest useful native streaming integration early, then complete a
   ten-update smoke on all ranks with monitoring 5/10,32 microbatches/update,
   finite metrics and clean teardown. Reuse unchanged source tests; rerun those
   affected by data/profiler integration. This closes execution, not parity with
   differently packed HF data. Promote the validated streaming configuration as
   the campaign baseline, with its data-generation recipe recorded.

## G2: unprofiled TPS protocol

Use native training and one reviewed detached runner with persistent exit markers
and all-rank error logs. Do not edit a running script or running source matrix.
Preserve data order, initialization, seed, scheduler trajectory, optimizer updates,
monitoring cadence and backend environment across fresh-process controls.

Start with a calibration run. Plan at least 20 initial updates before considering
measurement, inspect compilation/warmup/allocator and step series, and extend in
10-update increments only while those observations justify it. Twenty is a
starting probe, not proof of warmup. Require no ongoing compilation or unexplained
monotonic settling in the chosen window. If no stable regime is reached, diagnose
that concrete failure rather than silently choosing fast steps.

After calibration, freeze discard count `W` and a **50-consecutive-update**
measurement window for this campaign. Fifty covers one monitoring event at the
normal cadence. Use the same absolute update range and monitoring position for
all comparisons; if a candidate needs more warmup, change the window for both
sides and rerun only affected controls. Record exact boundaries before measuring.
Collect two identical controls A1/A2, separate processes, identical data/cache
policy. A2 can later serve as the first A of an A/B/A/B comparison only if source,
environment, topology, window and intervening conditions remain comparable.

Primary metric: aggregate physical input tokens per second for the complete
optimizer-update window, including input pipeline and normal monitoring cost.
For full fixed batches the expected numerator is
`512 * 4096 = 2,097,152 input tokens/update` globally (262,144/GPU); verify it
against actual batch counts. Also report supervised-token TPS using actual valid
labels, median and spread of per-update times/TPS, global versus per-GPU units,
monitoring versus ordinary steps, and complete-window elapsed throughput.
Inspect the native timer/counter scopes first: supervised-token TPS must not be
misnamed input TPS; do not sum rank throughput or double-count MTP tokens.

Keep all 50 steps in the primary cadence-aware metric. A secondary ordinary-step
view may exclude monitoring only when labeled and applied symmetrically. Measure
allocated/reserved/native memory with definitions, retries/OOM and rank skew;
profiled timings, checkpoints and diagnostic tensor dumps are excluded from TPS
runs. Do not change log cadence between variants. State excluded validation/save
costs explicitly.

## G3: minimal traces and Perfetto MCP analysis

Run a **separate** baseline profiling process after the TPS controls. Capture one
complete non-monitoring update after the same settled warmup, including all32
microbatches, optimizer and relevant communication; global ranks 0 and 1 only.
Profiler schedule: one warmup step, active 1, repeat 1, wait/skip chosen from the
actual stable update. Verify native step numbering and end-of-update export in
the installed implementation. Keep all other ranks unprofiled, not just their
exports deleted. No shape/stack/memory-history capture. Confirm exactly two
valid traces and log their hashes, update, source and resolved config.

Use the actual available Perfetto MCP tools: `load_trace`, table/schema discovery
and `execute_sql`. Check accepted Chrome JSON/Perfetto input format rather than
renaming file extensions. If conversion or server-visible transfer is required,
verify the same events/timestamps and retain hashes before deleting duplicate
payloads. Preserve the queries and compact result tables with the run report.

For both ranks determine compute/kernel families, communication-only intervals,
copy traffic, device-idle interval unions, host input/submission time and exposed
waits, compile/recompile and SAC replay. Use CPU↔GPU correlation and dependencies
to separate queued work from late submission; overlapping kernel sums are not
elapsed time. Compare rank skew and relate the traced step to unprofiled ordinary
step duration. Report timing scopes unavailable in the trace. A trace supports
mechanism; repeated unprofiled runs support TPS magnitude.

## G4–G5: cover references, then run candidates in evidence-based order

Every row must receive a disposition; do not promise speedup or apply incompatible
features merely to fill the table. Source inspection can establish inapplicability
or that the mechanism is already present. A lack of a demonstrated hotspot ranks
priority lower but does not silently count an applicable candidate as evaluated.

| ID | Reference mechanism | Initial applicability question | Final status / evidence |
| --- | --- | --- | --- |
| K1 | Kindulov fused softcap/CE and chunk lifetimes | Cap-zero path with MTP, monitoring chunks and head hooks; identify already-present chunk release | NOT_STARTED |
| K2 | Kindulov dense/shared FFN recomputation | FP8 support and interaction with fixed per-layer SAC; avoid duplicate replay and auxiliary counters | NOT_STARTED |
| K3 | Kindulov routed-expert recomputation | Donor rejects routed FP8; retain required FP8, document source incompatibility rather than convert to BF16 | NOT_STARTED |
| Z1 | Fused SwiGLU/FP8 hidden z-loss | Active crop has final-norm-input z-loss only; gate/up regularizer is absent and external pinned kernels are missing | NOT_STARTED |
| V1 | Vision CompactCE | Reference rejects MTP; do not disable MTP to make it run | NOT_STARTED |
| V2 | Vision streaming prefetch | Now a relevant loader path; compare donor/current queue, state and exception behavior before porting | NOT_STARTED |
| V3 | Vision GFSDP buffer/prefetch work | Determine existing versus missing code and apply only supported crop-relevant changes | NOT_STARTED |
| V4 | Vision frozen BF16 storage | Current crop trains its parameters; no frozen-component benefit unless source shows an actual applicable path | NOT_STARTED |
| V5 | Vision metric accounting | Check duplication and exposed logging/synchronization; adding a metric alone is not a speed optimization | NOT_STARTED |

Allowed final candidate labels: TESTED_GAIN, TESTED_NO_GAIN, TESTED_REGRESSION,
INAPPLICABLE (source/contract reason), ALREADY_PRESENT, BLOCKED (failed required
path and remedies), or NOT_COMPLETED (work remaining). Never mark a real required
implementation complete because diagnostic time was spent. Preserve source pins
and helpful unsuccessful implementations in reachable commits, not active defaults.

For every applicable candidate:

1. Record the trace hypothesis and expected compute/memory/communication cost;
   inspect donor interfaces and current consumers before porting. Use one
   candidate difference. Keep optional changes independently selectable.
2. Run targeted behavior/regression tests and same-input forward/backward tests,
   preserving accumulation, masks, MTP normalization, SAC replay, routing state,
   precision and monitoring. Compare elementwise gradients and affected optimizer
   updates, not just norms. Use donor numerical criteria only if their precision
   and workload assumptions apply. Otherwise calibrate against a valid reference
   and A/A variation; no invented percentage tolerance or unjustified bitwise gate.
   Record uncalibrated differences as unresolved. Native execution can still be
   tested, but correctness failures cannot be promoted as wins.
3. Run a short native 8-GPU integration, including monitoring. For a valid candidate,
   measure A/B/A/B with the fixed 50-update windows; reuse a valid nearby A where
   justified. Fresh initialization/data order and equivalent cache policy on each
   side; no profiling in these runs. Report run-level paired effects and A/A noise,
   not a significance claim based on correlated steps as independent trials.
4. Capture at most one two-rank/one-update trace pair for that executed candidate
   using the baseline policy. Explain which kernels/copies/waits changed and
   whether that matches measured TPS. Reuse the common baseline trace when valid.
   Replace a bad trace only for a documented capture/representativeness failure.
5. Record source commit, config delta, correctness results, TPS series/aggregate,
   noise, memory, trace queries/mechanism and next disposition in `results.json`
   and the campaign report. A negative result is useful; do not tune indefinitely.

After isolated tests, combine qualifying winners sequentially. Recheck affected
interactions and obtain a final native, unprofiled comparison against the frozen
MDS baseline; component wins do not establish an additive combined gain. Retain
the baseline if no candidate meets correctness and measurable benefit. Mathematical
objective changes and new model-shape/token-capacity studies are outside this
campaign; no convergence claim is required or implied by an overnight perf study.

## Artifacts, disk ownership and mandatory cleanup

Check free bytes/inodes before preparation and each run; track task-owned size.
Keep checkpointing disabled. If a required state comparison genuinely needs a
checkpoint/tensor dump, create the minimum task-owned temporary artifact, record
what it proves, then remove it after the comparison and compact result verify.
Do not accumulate full-model snapshots or per-step tensor dumps.

After each experiment and before the morning handoff:

- Retain compact configs, commands, manifests, step series, exit status, first
  fatal errors, SQL queries/results and the per-experiment conclusion.
- Retain only the intended two-rank/one-update trace pair for the baseline and
  each executed candidate; prefer one losslessly compressed canonical copy after
  successful parsing/MCP analysis. Remove redundant JSON/decompressed/converted
  copies and superseded successful diagnostic traces after verified extraction.
- Remove task-owned temporary checkpoints, tensor dumps, cores and staging copies
  once their required comparison/diagnosis is complete. Keep the smallest failure
  evidence that still explains an unresolved error; record that exception and
  next action rather than deleting the sole reproducible artifact.
- Reuse validated data and compiler caches where compatible; remove obsolete
  campaign-owned cache copies only after their processes stop and no queued run
  references them. Do not clear shared `/temp`, `/tmp`, package caches, old
  unrelated evidence or other users' artifacts.
- Record cleaned paths/classes and reclaimed/remaining bytes in a compact cleanup
  receipt. Confirm owned processes have exited and the eight GPUs are released.

No arbitrary storage cap is introduced. Disk pressure is a concrete stop/cleanup
condition, not permission to delete useful unrelated data. Never use broad
`pkill`, `killall` or shared-memory cleanup; identify owned PIDs and resources.
Raw payloads/data/caches stay outside Git. Do not copy large traces to the local
Mac merely for convenience.

## Blockers, publication and completion

After two consecutive harness failures before a useful measurement, reassess the
method: repair the concrete interface or use the existing native path. Distinguish
runner/comparator errors, infrastructure failure and feature failure. Do not
restart broad source audits or build a general experiment framework. Continue
independent candidates when one is blocked; preserve failures and attempted fixes.

Run code review and checks required by the repository, including an independent
integration/runtime-cost review where its rules require one. Reconcile findings
once and test affected fixes; do not spawn redundant reviews. Keep valid source
changes and experiments in reachable commits. Integrate validated MDS/default
changes and qualifying candidates on task `main`, preserve nonselected variants
without enabling them, and push at useful milestones. No force push. Check remote
changes before publishing; preserve other contributors' work.

The campaign is complete when G0–G7 are evidenced, every reference row has a
supported final disposition, baseline TPS and both baseline MCP analyses exist,
every executed candidate has its change/correctness/TPS/trace explanation, combined
acceptance (or baseline retention) is explicit, docs/status and plan are current,
results/source are published, and owned resources/heavy temporaries are cleaned.
If a required outcome remains blocked or unfinished, publish a useful partial
handoff with exact open gates and next action; do not claim full completion.

The final morning report must lead with baseline TPS, best accepted TPS and delta
(or no accepted gain), memory tradeoff and data identity; then show one row per
candidate, observed mechanisms, validation limits and remaining blockers. Update
this ledger at stage boundaries, source integration, failures and completed
experiments, not after every tool call. Provider/client termination is not a
feature verdict; the prompt cannot guarantee that a stopped agent restarts itself.
