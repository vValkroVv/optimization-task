# Targeted correction disposition

## Verdict: HOLD — F3 remains partly unresolved

F1, F2 and F4 are resolved on the inspected affected paths. F3 now cleans up
known separate-session descendants, but can miss a worker whose parent exits
between discovery scans. No GPU, live watcher, real process launch or signal,
source/preparation edit, or package change occurred during this verification.
The original review and its failures remain unchanged.

Reviewed corrected pins SHA-256:
`23614eb3aa497aa0ce303f56c62a3df7fb6dd33f04c604b9b4f4db1be0d23d29`.
Launcher: `d91cb45de270c0bf7b59cff7ad738e72ac96596492818b337785ba07274d2858`.
Watcher: `3175a319b45816814e33ea393bb012cc90762700cbb0628951a96756b2d62b7e`.

## Dispositions

| Original finding | Correction evidence | Disposition |
| --- | --- | --- |
| F1 actual launcher pin schema | Receipt writer consumes `attempt_files["launch.sh"]`; fixture copies the actual pins and launcher and rejects the former synthetic top-level field. Actual-artifact fixture passed. | Resolved |
| F2 incomplete receipt / missing ACK / active failure | Complete fsynced temporary JSON is hard-linked without replacement. Runner waits for ACK bound to receipt hash and live watcher/runner identities. `run_watcher` routes active failures through exact task cleanup. Atomic-publication, actual-schema ACK, dead-watcher rejection and active-failure fixtures passed or have preserved real evidence. | Resolved, subject to F3 cleanup coverage |
| F3 early and separate-session stop | Ancestry tracking starts before ACK, records PID/start ticks and survives later reparenting for already captured children. TERM/KILL is bounded; shell trap is bounded. Preserved real fixture proves a captured TERM-resistant separate-session child exits while a sibling sentinel survives. | Partly resolved; unseen orphan remains open |
| F4 helper suspends storage guard | `Popen`/poll replaces synchronous run; storage and process scans continue in the main loop; helper has a 120-second deadline and exact identity TERM/KILL. Pending-helper/storage-floor and helper timeout fixtures passed. Helper implementation hash remains unchanged. | Resolved |

## Remaining F3 failure

`TaskProcessTracker.refresh()` discovers only PPID edges reachable from already
known live identities. `run_watcher()` samples this at 0.5-second intervals.
If a launcher/agent creates a worker and exits before the next scan, the worker
can already have PPID 1 when the snapshot is read. Its exact attempt command
line is never examined, because no reachable ancestry edge selects it. Native
rank PID states are also not merged into the cleanup tracker. Cleanup then
returns `verified_exit=true` over its known subset, with that worker still live.

The actual supported route is shell → timeout → sanitizer → torchrun → workers
in separate sessions. An early launcher/agent failure before the first worker
discovery, or termination while the watcher is delayed, can break this ancestry
chain. This review does not claim such a failure occurred in a GPU run. The
existing real fixture explicitly captures the child *before* parent exit; it
establishes retained-identity cleanup, not discovery after early parent exit.

The new mock reproduction starts the tracker with only its live receipt-bound
runner, then presents an exact-attempt worker already reparented to PID 1. It
confirms the worker survives while cleanup reports verified exit. This is a
supported process-state transition, not a test of unrelated process managers.

**One scoped fix direction:** retain ancestry tracking, and additionally recover
this attempt's rank workers independently of current PPID using exact
NUL-delimited pinned `compiler_entry.py`/attempt argument matching, a start time
no older than the receipt-bound runner, and repeated PID/start-tick validation.
Merge validated native rank PID bindings into the same tracker. During cleanup,
rescan these narrowly identified task workers before declaring exit. This must
not use UID-wide selection, substring-only command matching, or broad process
group signals. Add a fixture where the worker was never captured before parent
exit and a same-user sibling remains untouched. If wider native compiler-child
cleanup is claimed, demonstrate its attempt-specific identity separately.

## Executed and reused evidence

[check_corrections.py](check_corrections.py) ran five targeted existing fixture
groups against the current modules and actual copied artifacts, plus the narrow
unseen-orphan reproduction. All assertions passed; [output](check_corrections.log)
records the five successful corrections checks and the remaining F3 condition.
Temporary fixtures were restricted to this review directory; all signals and
process accesses in the new checks were mocked. Reused the preserved real
process-tree evidence rather than launching another process fixture.

The restored handoff was inspected. Source007, configuration, unchanged helper
and sanitizer command retain the original review's accepted scope. Root's fresh
GPU/storage/pin preflight and actual `%p`/all-eight sanitizer tracking and
same-run provenance remain runtime gates; this correction review does not clear
them. No performance or memory-correctness claim follows these CPU checks.
