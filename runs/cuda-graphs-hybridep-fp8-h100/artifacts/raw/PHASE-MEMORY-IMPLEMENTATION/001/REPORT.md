# Phase-memory adapter staging report

## Result

The opt-in adapter, entrypoint, CPU behavior test, and README section are staged
under `staged/gift/`. No file under `gift/` was written by this implementation.
The patch is checked against source 007 but remains unapplied while that source
is frozen.

The adapter wraps the module bindings used by native metric logging. It copies
the allocated/reserved current and peak counters from the raw stats dictionary
already passed to `device_memory_usage`, pairs snapshots with the imported
`MetricsProcessor` reset binding, and keeps validation snapshots separate.
Records label the constructor-origin interval through update 20, followed by
actual reset-aligned windows such as 21–30. Updates 21–23 remain replay; no
capture-only peak is inferred. The existing timing observer's opening and
completed-close timer reads receive UTC/monotonic brackets. Its existing
opening and closing memory samples are copied into a separate tail record.

## Source and patch evidence

The inspected source is task repository commit
`446431dfa191ef0300f116257ae918bc85e38c06`, source snapshot 007. The source 007
manifest records the changes since source 006 as the completion-window files;
the production path for this adapter remains the native metrics path described
in the approved phase-memory design. Before generating the patch, these source
007 hashes matched its manifest:

- `gift/torchtitan/experiments/cuda_graph_campaign/observer.py`:
  `de9a7dc612b3c2753bf8b8b11b062491ad47684d0f5d3ef340840ab2130beb4b`
- `gift/torchtitan/experiments/cuda_graph_campaign/README.md`:
  `04e23c6c92eb0d59d8ffa0c871fee4e319815200cabcbf48e0bcfdf802676751`
- `gift/tests/unit_tests/callbacks/test_campaign_timing_observer.py`:
  `2718e60b2a3c141b356dc691fb3655b9e164baa1b670060d81d49206fb42b162`

`git apply --check --whitespace=error-all` passed for
[`phase-memory.patch`](phase-memory.patch). The patch touches only two new
experiment files, one new CPU test file, and the campaign README. Per-file
SHA-256 values are in [`SHA256SUMS`](SHA256SUMS); clean apply instructions are
in [`APPLY.md`](APPLY.md).

## R1 correction delta

The independent staged review at
[`PHASE-MEMORY-REVIEW/001/REVIEW.md`](../../PHASE-MEMORY-REVIEW/001/REVIEW.md)
found that `_complete_metrics_context` promoted invalid snapshots to complete.
The original nine-test pin and the review's
[`reproduce_status.py`](../../PHASE-MEMORY-REVIEW/001/reproduce_status.py) remain
unchanged as historical failure evidence. That review records the pre-fix
adapter SHA-256 `9b031d6d8132e9b3104254bb2b5079b871468a3efbc2bb13cec483beb404256e`
and demonstrates a missing allocated peak incorrectly labeled complete. The
pre-fix nine-test staging report recorded `9 passed`; its patch SHA-256 was
`8755bcc19470a7e6f08e81d2728824acf0e941d7f88f0a467f36c16f22faef2f` and its
test-file SHA-256 was
`9f1f0cc8886bb589cc8d2e898e742b7cf701f92675269d4aeb581baf8af1dd5c`.

The guard now promotes only snapshots still marked `pending_reset` with no
snapshot reasons, while retaining the existing paired-reset and context checks.
Two regressions exercise the installed native wrappers: a missing required
allocated peak and a snapshot after an unrecognized reset origin. Both failed
against the prior guard (`2 failed, 9 deselected`), where the snapshot status
was incorrectly `complete`; both pass with the guard (`2 passed, 9 deselected`).
The missing-peak path still calls `memory_stats` once, `mem_get_info` once and
native reset twice; the unknown-origin path still calls them once, once and
three times respectively. No native calls or cadence changed.
The guard adds status/reason predicates to existing opt-in metrics completion;
their isolated host cost was not measured, and no device operation was added.

## Runtime cost review

The ordinary `train` entrypoint remains unchanged and never imports or installs
the adapter. The explicit `train_phase_memory` entrypoint retains the existing
scan-fix setup and forwards the original argument list to native `train.main`.

On the opt-in path, the adapter adds Python wrapper calls at native metric and
reset cadence, copies eight integer counters per native snapshot, stores scalar
event records, and reads UTC/monotonic clocks around stats copies and reset
completion. It performs one allocator-backend metadata query during install.
When the timing observer is enabled, its existing boundary method has one
additional wrapper branch per measured boundary; only the opening and
completed-close boundaries add UTC/monotonic bracket reads. The close memory
sample reuses the observer's existing values and is not merged with a native
interval peak.

There are no added `memory_stats`, `mem_get_info`, synchronization, CUDA event,
tensor-value, or collective calls. The adapter adds no device allocation,
transfer, model computation, compilation region, autograd/RNG state, or
checkpoint state. Host memory is O(native metric records), and the rank JSON is
written once after the native entry returns or raises. These costs are from
source inspection; their runtime magnitude was not measured on GPU.

## Verification and open gates

- Staged CPU behavior tests: `11 passed`; the fakes verify native call order and
  counts, exact helper results and exceptions, constructor/training/validation
  reset generations, incomplete/missing-capture handling, nested contexts,
  entry cleanup after a flush collision, timing anchors, exact restoration,
  and the two R1 invalid-status regressions.
- `ruff check --select F821,F811,F841 --no-cache` on the staged Python files:
  passed.
- Python AST syntax checks on the three staged Python files: passed.
- README local-link check for `backend_scan_fix.md`: passed.
- Patch whitespace and apply check against the current source 007 working tree:
  passed. The patch was not applied.

The CPU tests load the staged adapter with fake native modules; they do not
establish integration with the pinned runtime, CUDA allocator behavior, NCCL,
eight-rank file completeness, timing perturbation, or the 500-update GPU gate.
External sampler values remain null in the rank record until the campaign's
UTC sampler CSV is joined offline with verified device mapping and the recorded
clock brackets. The independent review found R1 and requested this focused
correction; follow-up review, source integration and the authorized GPU gate
remain open.
