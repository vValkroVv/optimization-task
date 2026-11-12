# Independent staged phase-memory integration and cost review

## Verdict

Current disposition: R1 is resolved; staging is accepted after the focused fix
verification below. The following initial verdict and reproduction are retained
as historical evidence.

Changes required before accepting the staged adapter: one material validity
propagation bug. No staged file or Source007 file was changed or applied.
This is a focused source review and one CPU failure reproduction, not GPU
acceptance or performance evidence.

## Scope and finite acceptance checklist

1. Review the four staged paths against the approved design and actual native
   producers, reset bindings, timing consumers and entry cleanup.
2. Establish the install-time allocator query's initialization behavior from
   the pinned runtime implementation without calling CUDA.
3. Consolidate material findings, runtime costs and remaining execution gates.
4. Check this report's local links and whitespace; stop after this review.

Reviewed input: [approved design](../../PHASE-MEMORY-DESIGN/001/design.md),
[staging report](../../PHASE-MEMORY-IMPLEMENTATION/001/REPORT.md),
[staged adapter](../../PHASE-MEMORY-IMPLEMENTATION/001/staged/gift/torchtitan/experiments/cuda_graph_campaign/phase_memory.py)
and staged entry, test and README. Repository source revision recorded by staging:
`446431dfa191ef0300f116257ae918bc85e38c06`; staged adapter SHA-256:
`9b031d6d8132e9b3104254bb2b5079b871468a3efbc2bb13cec483beb404256e`.
Read code-rules and perf-experiments plus GIFT's applicable source manuals.

## Finding R1: invalid snapshots become complete

**Location:** staged `phase_memory.py`,
`PhaseMemoryObserver._complete_metrics_context`, lines 460–461; producer
`_capture_snapshot`, lines 256–321; consumer `_phase_envelopes`, lines 633–679.

The producer sets a snapshot to `invalid` for missing required counters, an
unknown reset origin or a duplicate generation, and stores its reasons. These
reasons are added to the observer but not necessarily to the active context.
Finalization then overwrites that snapshot's status with `complete` whenever a
snapshot and reset exist and the context has no reasons. `_phase_envelopes`
trusts the overwritten status, emits `complete`/`partial_values_only: false`,
and may compute a partial maximum or return null for a required peak. The
top-level document remains invalid, but its event and envelope claims contradict
the required invalid-coverage contract. A consumer of the envelope alone can
accept incomplete memory evidence.

**Reproduction:** [reproduce_status.py](reproduce_status.py), executed with
`PYTHONDONTWRITEBYTECODE=1 /opt/torchtitan-venv/bin/python3.12` and the script path.
It loads only the staged test's fake modules; no real torch import or CUDA query
occurs. One required allocated-peak key is removed before the actual installed
wrapper path runs. Observed result:

```json
{
  "snapshot_status": "complete",
  "snapshot_reasons": ["missing required allocator counters: allocated_bytes.all.peak"],
  "envelope_status": "complete",
  "envelope_partial_values_only": false,
  "envelope_allocated_peak": null,
  "document_status": "invalid",
  "memory_stats_calls": 1,
  "mem_get_info_calls": 1,
  "reset_calls": 2
}
```

**Small correction required:** only promote a snapshot that is still
`pending_reset` and has no snapshot reasons, in addition to the existing context
checks. Add a focused missing-required-counter regression asserting invalid
snapshot and envelope status and partial-value labeling. Cover unknown-origin
status through the same guard. Preserve original native calls and cadence.
The reproduction deliberately asserts the broken observation to document the
finding; it is not the proposed passing regression test.

## Integration conclusions

- Actual `MetricsProcessor.log` and `log_validation` invoke their imported
  `device_capacity_report`; that function resolves the utils module's
  `device_memory_usage` binding at runtime. The selected tap therefore sees the
  raw `memory_stats` dictionary without recreating native report formulas.
  Metrics' imported reset binding is correctly wrapped separately from utils.
  Logger rank filtering leaves capacity computation active on every rank.
- `Trainer.__init__` selects `cuda:LOCAL_RANK` before constructing metrics.
  Metrics' constructor reset follows DeepGEMM setup and meta construction and
  precedes parameter materialization. No pre-constructor snapshot is taken;
  whole-process bootstrap peak is explicitly unavailable.
- Native calls, arguments, results, exceptions and reset order are retained by
  the inspected wrappers. Native training and validation resets remain separate;
  reset origins and snapshot-to-reset gaps are recorded. Validation counters
  are not folded into training envelopes. The unchanged trainer runs validation
  after training and checkpoint work; resolved cadence must still be checked
  when interpreting actual interval memberships.
- `_capture_boundary` reads its opening timer before event recording;
  `_finish_window` synchronizes before its completed-stop timer, then reads
  closing memory and aggregates. The adapter anchors these existing timer reads
  and copies existing memory samples separately. It adds no timer/event/reset
  to force allocator intervals to coincide with the CPU timing window.
- Labels through 20 identify the post-constructor startup/settling envelope;
  subsequent labels preserve reset-aligned intervals. A008 updates 21–23 are
  replay, not capture-only peaks. A later actual recapture still needs campaign
  evidence and interval annotation.
- `train_phase_memory.main` preserves scan-fix installation and forwards the
  same argv object to native `train.main`. It catches BaseException, attempts
  partial flush and exact binding restoration, and preserves the primary
  exception if either cleanup operation fails. Construction failures unwind
  through the same path. Install rollback restores already borrowed bindings;
  restoration refuses to overwrite third-party replacements.
- The four-path patch does not change the normal train entry. No normal-entry
  import or installation path was introduced. No core logger, optimizer,
  checkpoint, RNG, model, precision or compilation state is modified.

## Pinned allocator metadata query

Read the installed files without importing torch:
`/opt/torchtitan-venv/lib/python3.12/site-packages/torch/version.py` reports
`2.15.0.dev20260822+cu130`, CUDA 13.0, git SHA
`8b353065ce260336c5fdfa74d754e8bd9b91b062`.
Installed `torch/cuda/memory.py:1289–1298` delegates directly to
`torch._C._cuda_getAllocatorBackend()`.

The exact upstream implementation was retrieved at that SHA:
[Module.cpp](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/torch/csrc/cuda/Module.cpp#L479)
calls `CUDACachingAllocator::name()` without lazy initialization;
[CUDACachingAllocator.h](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/c10/cuda/CUDACachingAllocator.h#L533)
uses an atomic allocator pointer load and virtual `name()`;
[native allocator](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/c10/cuda/CUDACachingAllocator.cpp#L5450)
returns a constant string. The async allocator name is also constant. Copies
are retained beside this report. An initial web-rendered search missed the
binding; direct retrieval of the exact raw file resolved it.

Thus this added backend metadata call does not initialize GPU0 or select a
device on the pinned native path. This conclusion is source evidence, not an
executed GPU probe. The earlier speculative initialization risk is resolved for
this call; no startup workaround is needed.

## Runtime cost review

- **Host:** installation imports runtime modules, validates signatures and
  identities, scans the output directory, and reads the allocator name once.
  Importing utils invokes its existing accelerator-availability discovery;
  this inherited initialization is moved earlier by explicit adapter import,
  not an added per-log allocator read. Native log/reset calls gain Python
  wrappers, context bookkeeping, eight integer copies and scalar event records.
- **Clocks:** each snapshot adds two clock brackets (four UTC reads and two
  monotonic reads); each reset adds one bracket (two UTC, one monotonic).
  Opening and completed-close anchors each add two UTC and two monotonic reads
  around the existing `perf_counter` read. Intermediate timing boundaries gain
  a Python branch, without new clock reads from the adapter.
- **Frequency/scale:** with log frequency 10, updates 1 and 10 through 500 yield
  51 training snapshots and 52 resets including construction, plus any actual
  validation calls. State is O(native log/reset count), roughly 103 native
  event dictionaries per rank, containing scalars and small metadata, with no
  model/token-sized storage. Phase aggregation and JSON persistence occur after
  native main returns/raises; no per-update output flush is introduced.
- **Device/communication/compilation:** no added per-log `memory_stats`,
  `mem_get_info`, tensor scalar reads, CUDA synchronization/events, collective,
  device copy/allocation, graph/compile region or autograd work. Existing native
  device reads, opening/closing syncs, events and aggregation remain inherited.
  Host wrapper/timer/bookkeeping costs are unmeasured; no zero-cost or speedup
  claim follows from these CPU checks.

## Evidence reused and remaining gates

Reuse the staging report's nine passing fake-module CPU tests, Ruff, syntax and
patch apply/whitespace evidence; those were not repeated. Added only the targeted
R1 reproduction because missing-counter validity was not covered and changes the
acceptance verdict. This script also confirms native helper/reset call counts
and restores borrowed fake bindings.

Required next steps: fix R1 in staging within its existing authorization and
run its focused regression, then recheck only affected behavior. Source007 stays
frozen and the patch stays unapplied. After the freeze and separately authorized
execution, the matched 500-update baseline/candidate still require actual native
integration, eight complete rank files, resolved allocator/logger settings,
native cadence, capture-settling attribution, sampler/device mapping and clock
joins. The 500 ms device maximum remains a lower bound and sampled free headroom
an upper bound on worst-case headroom. Exact whole-device and capture-only peaks
remain unavailable. No 500-update, GPU timing or production-memory claim is made.

This report's local links and whitespace were checked. No GPU queries,
initialization, training, installations, process operations, commits, source
edits or patch application occurred during this review.

## Focused R1 fix verification

**Verdict: R1 resolved; accept the corrected staging.** Application and GPU
acceptance remain separate open gates. This follow-up inspected only the guard,
the two R1 regressions and the updated staging evidence; it did not reopen the
integration review.

Corrected staged adapter SHA-256:
`5bceda8d30b3858b9c54efb5f5aba72b01bfe97b665c0828488d1f8e399ede2f`;
corrected patch SHA-256:
`d1fe7e88c084c295796c51e3bec301d76e7db27d0872c678e21d8b3afa4ae4c0`.
The completion guard now requires a snapshot still marked `pending_reset`, no
snapshot reasons, a paired reset and clean context. It therefore cannot turn
the producer's invalid snapshot into a complete event. The existing envelope
consumer consequently retains `invalid` and `partial_values_only: true` for
both a missing required peak and an unknown reset origin. Available partial
counters remain available without a false completeness claim.

Executed independently:

```text
PYTHONDONTWRITEBYTECODE=1 /opt/torchtitan-venv/bin/python3.12 -m pytest -q \
  -p no:cacheprovider \
  --basetemp=runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/PHASE-MEMORY-REVIEW/001/pytest-r1-fix \
  runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/PHASE-MEMORY-IMPLEMENTATION/001/staged/gift/tests/unit_tests/test_campaign_phase_memory.py \
  -k 'missing_required_peak_stays_invalid or unknown_reset_origin_stays_invalid'
2 passed, 9 deselected in 0.09s
```

The tests exercise the installed wrapper path and persisted snapshot/envelope
claims. They also verify unchanged `memory_stats`/`mem_get_info` counts (one
each), and native reset counts (two for missing key, three with the extra
unknown reset), plus restoration. Reuse the implementation report's old-guard
failures for these two regressions, full `11 passed` result, Ruff, syntax,
apply/whitespace and SHA checks; no broad check was repeated.

The guard adds only two host predicates at native metrics completion. It adds
no allocator/device query, reset, CUDA event, collective, transfer or
compilation work. The initial runtime-cost conclusions and pinned metadata
query conclusion remain applicable.

The original [reproduction](reproduce_status.py) remains unchanged as failure
evidence for the original hash; its assertions intentionally describe the
pre-fix bug and are not expected to pass against corrected staging. No R1 issue
remains open. Source007 remains frozen and the patch remains unapplied.
Authorized application, actual runtime integration, eight-rank completeness
and the matched 500-update memory/sampler gate remain open exactly as described
above. Follow-up report links and whitespace pass.
