# Native execution record

**Final sequence:** all three authorized native runs passed. Raw attempt roots are
under [`runs/overnight-streaming-perf/artifacts/raw/`](../overnight-streaming-perf/artifacts/raw/).
Run metadata contains actual child argv; [`protocol.json`](protocol.json) preserves
those argv arrays and exact receipt paths/hashes.

## Completed runs

| Run | Measurement | Outcome |
| --- | --- | --- |
| `combined-four-baseline-003` | 140 updates; warmup 40; updates 41–140 timed | PASS on 8 ranks |
| `combined-four-candidate` | 140 updates; warmup 40; updates 41–140 timed | PASS on 8 ranks |
| `combined-four-diagnostic` | 21 updates; warmup 20; update 21 profiled on ranks 0/1 | PASS on 8 ranks |

For each run, [`completion.json`](../overnight-streaming-perf/artifacts/raw/combined-four-baseline-003/completion.json)
(or the run-specific sibling) reports PASS with source/config unchanged and task
scratch removed. The associated `process.json` has child exit 0, stopped process
group, no timeout, and no orphan. All rank records and exact actual argv are
linked from the run-specific sections of [protocol.json](protocol.json).

The baseline used a fresh compiler cache. Candidate and diagnostic reused the
baseline cache **in place** at
`runs/overnight-streaming-perf/artifacts/raw/combined-four-baseline-003/cache`,
including the eight stable per-rank Triton paths. Candidate and diagnostic logs
show 39 planned DeepGEMM startup kernels, all 39 cached, zero compiled at startup.
This does not establish that every compiler path was cache-only or that measured
updates had no compilation; cache observation/guard was disabled. Both timed runs
retained warmup 1–40. The exact runner/helper source hashes per run are in
[source-identities.json](source-identities.json); the old readiness hashes are
labeled historical there.

The diagnostic result has 32 microbatch calls and one optimizer call per rank at
update 21. Traces on global ranks 0 and 1 contain three captured microbatches and
are SHA-pinned in [protocol.json](protocol.json). The completed [trace analysis](trace-analysis.md) verifies all four mechanisms and
records import coverage plus event-specific wait and critical-path limits.
The diagnostic launcher emitted nonfatal multiprocessing finalizer `Errno 16`
messages while NFS `.nfs*` files were busy after training. Child exit remained 0,
the process group stopped, and task scratch was removed. This differs from the
historical rank-1 `Errno 116` lock failure below.

After all native groups exited, the task owner reverse-applied the exact
candidate core patch. `loss.py` and `hybridep.py` again match the frozen baseline
hashes; no default was promoted.

## Preserved failed attempts and repairs

The first `combined-four-baseline` detached launcher vanished before metadata or
TorchRun creation. No exit code or termination signal was captured. One matched
CPU `nohup` probe also disappeared before its marker; a managed foreground CPU
probe passed after 8.001 s. The owner then explicitly authorized the three-run
sequence using the managed foreground session. The first failure remains in
[launch-failure.md](launch-failure.md); its exact scope is historical and it did
not include a native child.

Comparable earlier successes ended by 60 updates at epoch size 51,200. The longer
`baseline-002` used 140 updates and epoch size 102,400; rank 1's background
`fcntl.flock` `Errno 116` (`ESTALE`) first appeared in logs after update 79 and
later stopped entry to update 140 after 139 complete updates. Its
`completed_steps: 140` counts entry to the failed update, so it has no valid
100-update window and is excluded. The greater exposure explains why the longer
run could encounter the issue while those earlier short runs did not; it does not
prove a deterministic threshold, package upgrade, or exact cause. Earlier runs
used the same task TMPDIR/runtime.

A bounded repository CPU probe recorded 3,200 lock acquisitions with 2,413
overlaps for the installed `filelock` 3.29.4 unlink/unlock mutex regression;
stable-inode locking had zero overlaps. This matches upstream
[filelock issue 574](https://github.com/tox-dev/filelock/issues/574) and its fix
in commit `ea594a5`; the installed task environment was not modified. The runner
uses the official stable-release behavior as a task-local subclass only for the
Streaming dataset and barrier bindings, without shared package edits or `/tmp`.
The direct raw probe receipt is [nfs-lock-probe/result.json](artifacts/raw/nfs-lock-probe/result.json); the independent NFS-lock review passed with 17 focused CPU cases.

The cache-reuse CPU suite first had 76 passes and 2 failures (missing backend
exception and TMP fixture); both were corrected, and the two affected cases then
passed. Exact logs are
[`pytest.log`](../overnight-streaming-perf/artifacts/raw/cache-reuse-tests/pytest.log)
and
[`pytest-corrections.log`](../overnight-streaming-perf/artifacts/raw/cache-reuse-tests/pytest-corrections.log).
Scoped Ruff and the original GPU composition checks also passed (3 passed,
4 deselected, 0 skipped). Earlier failure artifacts remain preserved; no failed
attempt is included in the valid baseline/candidate pair.
