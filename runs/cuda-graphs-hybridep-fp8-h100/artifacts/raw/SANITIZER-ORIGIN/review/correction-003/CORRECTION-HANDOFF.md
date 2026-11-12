# SANITIZER-ORIGIN F3 correction handoff

Status: ready for affected-behavior re-review; GPU release remains HOLD.

## Fix

The tracker still follows receipt-bound ancestry. Every tracker refresh also
recovers processes whose NUL-delimited argv contains the exact absolute path to
the pinned `compiler_entry.py`, provided their start ticks are no earlier than
the receipt-bound runner. Registration and signal paths each revalidate PID,
start ticks and exact argv. Validated structured-log rank PID states join the
same tracker through that exact check. Recovered records keep `ancestry: null`
and name their actual argv proof; descendants of recovered records use a
separate observed-parent proof. Cleanup performs a final recovery refresh and
cannot return `verified_exit` if that scan fails or any tracked identity remains
live. It uses exact PID signals only.

## Affected checks

`f3-focused-tests-final.log` records seven passes: reparented orphan recovery,
exact path/age filtering, registration and signal-time identity checks, required
final scan, rank-log merge using the real prior structured-log prefixes,
storage-stop targeting, and separate-session TERM/KILL cleanup with a surviving
sibling sentinel. The CPU process-tree evidence is in
`cpu-process-tree-20261112T061408Z.json`; its root, child and sentinel identities
were checked absent after the test.

The earlier reviewer reproduction remains unchanged in `review/001` and still
documents the pre-fix false `verified_exit`. F1/F2/F4 accepted evidence is reused
without rerunning those suites. A first test-fixture setup error is preserved in
`f3-focused-tests.log`; its exact task-owned PID cleanup is recorded in
`fixture-setup-failure-cleanup.json`.

## Pins and scope

Only `attempt_files.watch_binary_provenance.py` and
`attempt_files.verify_watcher_cpu.py` changed. Their current SHA-256 values are
`2db7c5c30a08fba337aea407eff25234fd362ebada53b6852b62273d91c96645` and
`6cb530364f03b7f39fe5d1631fb9cbda0c076042d14b7f240e750b8831016d38`.
`pins.json` is `acbd2b3f7ae1cac334d26efb1817b4feaacc7674f3757d8fef64e517133ef4ac`;
its canonical top-level schema is unchanged. Prior pins and handoff are preserved
as `pins-before-correction.json` and `handoff-before-correction.md`.

No watcher, runner, sanitizer target, training, GPU query, source edit, install,
service or commit was started. Root still owns the final preflight and release.
