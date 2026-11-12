# Preparation 002 mechanical transfer review

**ACCEPT for config resolution, fresh preflight and root release once the next
source correction is accepted and pinned.** This is scoped launcher/preparation
acceptance, not a successful native numerical probe result.

## Acceptance and inspected result

Finite checklist: preserve previously accepted executable behavior; move only
the intended preparation/attempt identities; create the attempts parent after
config/no-checkpoint checks; atomically claim a fresh attempt; preserve final
source/preflight/root-release gates.

Inspected the actual 001-to-002 launch/preflight/verifier diffs, release template
and transfer manifest. No material findings remain in this mechanical transfer.

- Launch and preflight target `preparation/002` and `attempts/002`. Verifier
  scope, receipt prefix, attempt absence checks and all task-local template
  artifact keys consistently target 002. The canonical loss-only adapter and
  offline cache retain their established external 001 paths.
- `ATTEMPT_PARENT` creation follows both config checks and the attempt/output
  no-checkpoint checks. The launcher creates only that task-owned parent, checks
  that attempt 002 remains absent, then claims the child using a single `mkdir`.
  Under `set -e`, a concurrent or duplicate claim fails before native execution.
  Existing attempt 001 contents are neither reused nor modified by this path.
- Eight assets are byte-identical to preparation 001: both phase configs, both
  compiler adapters, process owner, topology/software helpers and process tests.
  The accepted pidfd/subreaper ownership/release behavior, sampled RSS accounting,
  topology validation and 1,800-second operational deadline are preserved.
- All twelve release-asset hashes match the transfer manifest. Exact reviewed
  identities are recorded in [reviewed-sha256.json](reviewed-sha256.json).
  The template remains intentionally unapproved with the next source pin and
  real preflight receipt pending.

## Reused verification and runtime scope

Reused the reported syntax/AST/F821/F811/F841, template artifact-set, hash/link
and fresh-parent/duplicate-refusal fixture checks. Reused prior accepted config,
software, native checkpoint-path and six CPU process-fixture evidence; no repeat
of those unchanged paths was needed. This review directly compared the eight
unchanged assets and checked the twelve transfer hashes and new scope.

Only one startup parent-directory creation and an absence check are added;
there is no per-step compute, allocation, D2H, synchronization, communication or
compilation change. No performance or memory magnitude was measured here.

Attempt 001's native post-save complex64 failure belongs to the separate source
analysis/correction and does not invalidate the unchanged owner logic. This
review does not accept that source correction or reproduce its investigation.
Preparation 001 and its evidence remain preserved.

## Remaining gates

Root must accept and pin the corrected source, resolve both configs on that
pin without creating attempt/checkpoint folders, and run fresh real preflight
for preparation 002. Root then reviews its exact software, eight idle H100
UUIDs/no compute apps, SM90/NV18, NFS/fsync evidence and at least 732-GiB admission,
and fills the final hash-bound release. Only then may the exact serial full
step-1 S0 save and fresh-process full load-only step-2 observed update run.

Required results remain native full-state/readback correctness, complete eight-rank
manifests/compiler records, no observer checkpoint, actual storage/memory samples,
and verified owned-process release. Uninterruptible survivors remain explicit
release failures. Successful readiness does not establish A/A equivalence,
performance, graph-on behavior or prior CUDA fault clearance.

No GPU query/run, source/preparation edit, package installation, process cleanup
or subagent was used in this review.
