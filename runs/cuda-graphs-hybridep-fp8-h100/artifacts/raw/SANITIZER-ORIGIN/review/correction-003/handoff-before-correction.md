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
  Refreshed pins.json SHA-256 is
  23614eb3aa497aa0ce303f56c62a3df7fb6dd33f04c604b9b4f4db1be0d23d29.
- F2 — Atomic receipt and ACK: watcher receipts are written to an exclusive
  same-directory temporary file, fsynced, then hard-linked to the final path
  without overwrite. The watcher validates the complete launch receipt,
  initializes task tracking and storage monitoring, then publishes an ACK bound
  to the receipt hash and both process identities. launch.sh waits up to
  15 seconds for the live ACK before starting the sanitizer. Active parser,
  identity, deadline, storage, and watcher errors run bounded task cleanup
  before the watcher exits.
- F3 — Full task tree stop: from the receipt-bound runner, the watcher scans
  /proc every 0.5 seconds and records descendant PID, start ticks, parent/start
  identity, session, process group, command-line hash, attempt path, ancestry,
  and launch-receipt hash. Session and process-group IDs are recorded but never
  signalled. Cleanup sends TERM to exact tracked identities, rescans during the
  grace period, escalates remaining identities to KILL, and verifies exit. The
  shell timeout trap is also bounded and does not do an unbounded wait.
- F4 — Async one-shot helper: the unchanged helper starts once with Popen, and
  the watcher polls it while continuing storage/deadline checks. Output goes to
  a pending path; only successful parseable complete JSON is atomically
  published as the provenance receipt. The helper deadline is 120 seconds. A
  stall stops only that PID using its start ticks and helper path, records
  unavailable/partial output, and cannot be retried.

## CPU verification

- python verify_watcher_cpu.py passed all 11 fixture groups. They cover actual
  pins schema, complete/no-overwrite receipt publication, absent/dead watcher
  ACK rejection, all-eight-rank update-1 parsing, sequence gaps, missing ranks,
  pre-trigger errors, dead rank PIDs, pending helper with simulated storage
  floor crossing, one-shot helper timeout/escalation, and storage stop targeting.
- The real structured-log fixture reads prior COMBINE-BWD-PROBE rank prefixes
  only through optim_end step 1, with mocked /proc, helper, and storage actions.
  It does not inspect prior live PIDs or invoke the helper on them.
- The real CPU process-tree fixture started only task-owned dummy scripts under
  this attempt. It captured a separate-session TERM-resistant descendant before
  its parent exited, stopped that exact child with TERM/KILL and verified exit,
  and proved a sibling sentinel survived task cleanup before its own exact
  cleanup. Evidence records root PID 613268, child PID 613269, sentinel PID
  613267, start ticks, ancestry, and signal outcomes.
- Python syntax compilation and bash -n launch.sh passed. Focused verification
  confirmed the canonical launcher field and all seven current attempt-file
  hashes match pins.json. The GIFT changed-file set was compared with SOURCE007
  without rehashing the already verified 92 source files.
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
