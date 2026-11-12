# SANITIZER-ORIGIN / attempts/001

## Status

Corrected preparation is ready for targeted review. GPU release is **HOLD**.
No live watcher or runner was started; no GPU/NVML query, sanitizer target,
training process, or persistent service was started. No GIFT file was changed.
The attempt directory has no READY or launch-gate receipt.

## Preserved review and setup evidence

- review/001/REVIEW.md, check_integration.py, and check_integration.log remain
  unchanged. The original launcher and watcher hashes recorded there are
  05fd8704664f0cdef0db5b90aacadc8bfde8e17ade6e8883f18392ff1035fb03 and
  341e8dd4b8caf604ea17423931568dee804bb1374920cf204ddab813b9f0879d.
- The recorded failures remain part of the history: actual pins rejected the
  noncanonical launcher lookup; the old final-path receipt was visible while
  incomplete; early stop knew only the shell/rank PIDs. Review/001 also records
  the initial incomplete mock READY mistake and its correction.
- The earlier preparation's first verify_pins.py setup failed because the flat
  sanitizer hash pin was missing; it was corrected before review and is not
  counted as a pass. The pre-correction pins are preserved at
  review/correction-002/pins-before-correction.json, SHA-256
  58b307cf17bec3dc46dcdceb7942abe8e79374dd8ea17959d0ae3acc9b8f3c16.
- The prior handoff is preserved at
  review/correction-002/handoff-before-correction.md, SHA-256
  c27be4ec022a0be71954ffee160b5a1870045f400db33fe7636cbb6a6a408b2a.
- The first real CPU tree fixture failed during sibling command-line validation
  as the process exited. Exact fixture PID 612171 was checked afterward and
  was absent from /proc. The stop helper now checks the attempt marker before
  its first signal, then binds the same PID/start ticks through the exit
  transition. The rerun passed; evidence is in
  review/correction-002/cpu-process-tree-20261112T054813Z.json.

## Pinned run plan

- SOURCE-SNAPSHOT/007 remains 92 files, manifest SHA-256
  2585e504350276cb4689ae9ea45a33706e9f898bb61969abe5d08f2650c55620, archive
  SHA-256 0fc5ddad83c5f99b4036a47dd7f485ba2f42d1284c42f086d5f051cebad66082,
  base 446431dfa191ef0300f116257ae918bc85e38c06. Current GIFT tracked changes
  plus untracked files still match the manifest's 92 paths.
- Preset SHA-256 is b341701bd1eef7774c110fbf7ed69cf6280ba8691100b05be457dee3e7d50a7b.
  Scan-fix entry and implementation remain pinned in pins.json.
- Configuration remains the exact 3B FlexAttention/HybridEP FP8 preset:
  FSDP2/EP8, selective AC, model/loss compilation, batch 8/256, sequence 4096,
  accumulation 4, seed 42, 60 native updates and schedule 30000, CP false.
  All 18 model layers stay compiled and graph-excluded. The public
  campaign_cudagraph_step_boundary marker remains. Completion/timing observers
  and profiling are absent. Graphs are off, mix-order reduction is zero, graph
  partitioning remains true, and train_scan_fix is retained.
- Compute Sanitizer remains v2026.1.0.0 build 37182542, SHA-256
  5cba659e1cd603aa2370e103425f184368e6ed405da90ce85743c4399c324dbc.
  Previously verified local help/manual support is reused: memcheck,
  --target-processes all, error exit 86, backtrace yes, %p log path, print
  limit 50, launch timeout 180 seconds, and
  NV_COMPUTE_SANITIZER_BINARY_PATCHING=0. No historical binary-patching CLI,
  forced synchronization, padding, filters, or sanitizer coredumps are added.
  The runner sets core size to zero for this task process only.
- launch.sh retains one 20-minute timeout, torchrun restart count zero, and
  60 native updates. HOME and existing caches are preserved; scan-runtime uses
  a fresh attempt path.

## F1-F4 corrections

- F1 — Canonical pins: launcher SHA reads use
  pins["attempt_files"]["launch.sh"]. The fixture loads the actual pins.json
  and copied launcher, and asserts there is no top-level launcher_sha256 key.
  Refreshed pins.json SHA-256 after the scoped F3 refresh is
  acbd2b3f7ae1cac334d26efb1817b4feaacc7674f3757d8fef64e517133ef4ac. The prior
  pins and handoff are preserved under review/correction-003.
- F2 — Atomic receipt and ACK: watcher receipts are written to an exclusive
  same-directory temporary file, fsynced, then hard-linked to the final path
  without overwrite. The watcher validates the complete launch receipt,
  initializes task tracking and storage monitoring, then publishes an ACK bound
  to the receipt hash and both process identities. launch.sh waits up to
  15 seconds for the live ACK before starting the sanitizer. Active parser,
  identity, deadline, storage, and watcher errors run bounded task cleanup
  before the watcher exits.
- F3 — Bounded attempt process cleanup: the watcher retains receipt-bound
  ancestry tracking and also scans /proc for the exact NUL-delimited absolute
  pinned compiler_entry.py argument. A candidate must start no earlier than the
  receipt-bound runner and pass repeated PID/start-tick/argv checks before
  registration. Validated structured-log rank PIDs are merged through the same
  exact entrypoint check. Recovered roots record `ancestry: null` and an exact-
  argv proof; descendants use either receipt-bound ancestry or a separately
  labeled observed-parent proof. Signal-time checks revalidate PID/start ticks,
  receipt binding, and exact argv when that proof applies. Cleanup performs a
  final narrow recovery scan and reports verified exit only when no tracked
  identity remains live. Scan or identity errors fail closed. No process groups
  or UID-wide selection are signalled; the shell timeout trap remains bounded.
- F4 — Async one-shot helper: the unchanged helper starts once with Popen, and
  the watcher polls it while continuing storage/deadline checks. Output goes to
  a pending path; only successful parseable complete JSON is atomically
  published as the provenance receipt. The helper deadline is 120 seconds. A
  stall stops only that PID using its start ticks and helper path, records
  unavailable/partial output, and cannot be retried.

## CPU verification

- The previous F1/F2/F4 evidence remains accepted and reused from review/001;
  this correction did not rerun their full suite. The focused F3 suite passed
  seven checks: orphaned exact-entrypoint recovery and bounded stop; earlier
  start/path-lookalike rejection; PID/start-tick reuse and same-PID argv
  revalidation; final-scan failure cannot report verified exit; validated rank
  PID merge; storage stop targeting; and separate-session process-tree cleanup.
  Output is in review/correction-003/f3-focused-tests-final.log.
- The actual COMBINE-BWD-PROBE structured-log prefixes through optim_end step 1
  were parsed read-only. The resulting validated rank/PID states were merged
  into the task tracker with exact compiler-entry argv proof. Proc, helper,
  storage and signal actions were mocked for this parser/merge fixture; no old
  live PID was inspected and the helper was not invoked on old PIDs.
- A bounded CPU process-tree fixture used only new task-owned dummy scripts.
  It captured a TERM-resistant separate-session child before parent exit,
  stopped that exact child with TERM/KILL and verified exit, and proved its
  sibling sentinel survived task cleanup. Evidence with exact PID/start ticks
  is review/correction-003/cpu-process-tree-20261112T061408Z.json; a direct
  post-test PID/start-tick check confirmed all three fixture identities exited.
- The preserved old orphan reproduction remains in review/001/check_corrections.py
  and .log. The first new process-fixture setup failure and exact cleanup of
  task-owned PID 618460 are preserved in review/correction-003; the fixture was
  corrected to copy the pinned entrypoint and pins before the passing rerun.
- Python source compilation and bash -n launch.sh passed. After the focused
  suite, only the watcher and CPU fixture hashes were refreshed in the existing
  canonical attempt_files map; all seven attempt-file pins now match. The pins
  top-level schema is unchanged. GIFT remains unedited and SOURCE007 stays frozen.
- Existing LOADED-BINARY-RECEIPT evidence (10 CPU checks) and the unchanged
  helper SHA are reused; helper tests were not duplicated or rerun.

## Output estimate and runtime cost

The 64 MiB planning allowance covers **native and sanitizer outputs only**; it
is not a bound on total NFS writes. It uses the quiet MATCHED-RECAPTURE footprint
of 16,287,481 bytes over 26 updates, scaled to 60 updates (37,586,494 bytes),
rounded to 40 MiB native plus 24 MiB for up to 50 sanitizer reports.
--print-limit 50 caps report count, not report bytes. Small watcher/helper
control receipts are outside this allowance.

Whether fresh scan-runtime JIT writes and existing compiler-cache growth were
inside the prior measured footprint is not established; treat both as
unmeasured and outside this estimate. The active 4 GiB floor is a stop
condition, not a storage reservation or guarantee.

The watcher adds a 2 Hz /proc ancestry scan, a 2 Hz structured-log scan, a
three-second storage sample/fsync, and one bounded CPU hash helper. Their
contention and sanitizer time/memory overhead remain unmeasured. This is
exception attribution only; it makes no memory-correctness or performance
claim.

## Remaining gates

Targeted review of the corrected files remains open. Root must still provide a
fresh exact-eight GPU ownership/topology/free-memory check, timestamped NFS
free-space/inode/write/fsync preflight, and explicit compute release before
starting READY or launch. Actual sanitizer compatibility, eight-rank coverage,
same-run loaded-binary capture, fault attribution, full cleanup under the real
workload, memory sufficiency, and numerical/quality behavior remain unverified.
No automatic retry or fault clearance follows a clean, failed, uncovered, or
nonreproducing run.
