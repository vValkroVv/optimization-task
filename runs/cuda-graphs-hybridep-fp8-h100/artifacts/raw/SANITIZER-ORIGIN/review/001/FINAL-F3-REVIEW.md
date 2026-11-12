# Final F3 correction review

## Verdict: ACCEPT for coordinator preflight

F3 is resolved for the prepared native-worker launch contract. No material
implementation finding remains from F1–F4. This accepts the prepared runner
for root's fresh preflight and explicit release; it does not itself authorize
or establish a GPU run, live cleanup, sanitizer coverage or fault attribution.
The original review, correction review and false-verified-exit evidence remain
unchanged.

Final pins SHA-256:
`acbd2b3f7ae1cac334d26efb1817b4feaacc7674f3757d8fef64e517133ef4ac`.
Final watcher SHA-256:
`2db7c5c30a08fba337aea407eff25234fd362ebada53b6852b62273d91c96645`.
Final fixture SHA-256:
`6cb530364f03b7f39fe5d1631fb9cbda0c076042d14b7f240e750b8831016d38`.
Launcher and helper remain at the accepted correction's hashes. All seven
actual artifact hashes match the current pins. The restored handoff now names
the current pins and final F3 behavior/evidence.

## Verified correction

`TaskProcessTracker` verifies the on-disk compiler entrypoint pin and discovers
workers by its exact absolute NUL-delimited argv argument, independently of
current PPID. Start ticks must be at least the receipt-bound runner's. Repeated
identity/argv reads reject reuse during registration; signal-time checks repeat
the proof. Recovery records exact-entrypoint proof rather than inventing an
ancestry chain. `merge_rank_states()` incorporates validated native rank PIDs
using the same exact-entrypoint contract, and the active watcher calls it after
parsing. Previously proven descendants retain their bounded exact-PID cleanup.

`stop_task_processes()` performs narrow recovery during cleanup and again before
exit acceptance, sends bounded TERM/KILL to validated identities, and fails
instead of reporting verified exit if the final scan fails or a process remains
live. No UID-wide or process-group signals were added. The recovery is scoped
to this attempt's native entrypoint, not arbitrary unrelated orphan commands.

## Evidence

[check_final_f3.py](check_final_f3.py) ran the four affected prepared fixtures
plus a direct parsed-rank merge assertion. All passed:

- An unseen reparented exact-entrypoint worker is recovered, escalated and
  verified exited; lookalike paths and pre-runner identities are excluded.
- PID reuse during and after registration is rejected.
- Changed argv is rejected before signalling.
- A failed final recovery scan cannot yield verified exit.
- Parsed rank identity merges with the exact-entrypoint proof.

[check_final_f3.log](check_final_f3.log) records these results. The preparation's
[before-fix failure](../correction-003/f3-before-fix.log) shows the exact native
entrypoint regression failed on the prior implementation. Reused its
[real process-tree evidence](../correction-003/cpu-process-tree-20261112T061408Z.json):
captured separate-session TERM-resistant child exited after TERM/KILL, and a
sibling survived before its own exact cleanup. No real process fixture was
repeated in this review; new checks used only mocked process state/signals.

F1/F2/F4 acceptance is retained. Root owns fresh GPU ownership/topology,
Source007/config/pin consistency, storage byte/inode/write/fsync checks and
live READY/ACK release. Actual all-eight sanitizer tracking, same-run binary
provenance, bounded real-workload cleanup and exception attribution remain
runtime gates. The broader exact-argv recovery adds host `/proc` reads to the
diagnostic; its cost is unmeasured and supports no performance claim.
