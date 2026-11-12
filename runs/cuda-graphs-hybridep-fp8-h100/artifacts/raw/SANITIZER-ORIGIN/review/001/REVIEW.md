# Sanitizer runner integration review

## Verdict

**Do not release attempts/001.** Four material integration findings remain.
No source/preparation files were edited, no GPU queries or launches occurred,
and no processes were started in the background or signalled.

Finite acceptance: check the approved command/configuration and pins; trace
READY through launch and native log producers; check one-shot live capture;
trace deadline/storage cleanup across process lifetimes; review runtime cost;
consolidate findings and remaining gates. All review items are complete; the
experiment release gates remain open.

Reviewed base HEAD: `446431dfa191ef0300f116257ae918bc85e38c06`.
Reviewed attempt launcher hash:
`05fd8704664f0cdef0db5b90aacadc8bfde8e17ade6e8883f18392ff1035fb03`.
Reviewed watcher hash:
`341e8dd4b8caf604ea17423931568dee804bb1374920cf204ddab813b9f0879d`.
The code-rules, perf-experiments and titan-train-job skills informed this review.

## Material findings

### F1 — Actual pin schema prevents launch

[watch_binary_provenance.py](../../attempts/001/watch_binary_provenance.py)
lines 247 and 266 consume top-level `launcher_sha256`. Actual `pins.json`
only supplies `attempt_files["launch.sh"]`. The actual runner therefore fails
with `launch script SHA-256 mismatch`, even after valid READY and release gates.
The fixture at `verify_watcher_cpu.py:118` manufactures the absent field and
misses the integration failure. Use the canonical actual pin field consistently
and exercise the receipt writer using the actual pins schema.

### F2 — Receipt publication can lose the watcher before launch

Watcher lines 45–52 publish an empty final pathname before writing/fsyncing JSON.
Lines 542–544 treat pathname existence as publication completion. A reader in
that interval gets invalid JSON and exits; `runner` has not been assigned, so no
stop occurs. The launcher continues after its receipt-writing subprocess
finishes (`launch.sh:35–36`), without an acknowledgement that the watcher accepted
the receipt. READY checked before publication does not close this race.

Publish a complete receipt atomically without overwriting an existing attempt,
and require a live, receipt-bound watcher acknowledgement immediately before
launch. Also resolve active watcher failures: `WatchError` at lines 605–617 only
ends the watcher, leaving training/storage writes running without supervision.
Verify partial publication, watcher death before acknowledgement, and active
parser/identity failure through the launch/supervision path.

### F3 — Stop is not bounded across the full task process tree

Watcher lines 439–455 signal only the runner and ranks already discovered from
logs. Before rank logs exist, it signals only the shell. Runner cleanup
(`launch.sh:67–94`) sends TERM only to `TIMEOUT_PID`, then waits without its own
deadline. It neither records nor revalidates sanitizer/torchrun/worker identities
and cannot independently finish their cleanup.

The installed Torch `subprocess_handler.py:65–88` starts each worker with
`start_new_session=True`; `api.py:1023–1066` allows 30 seconds before worker KILL.
The runner's timeout KILL grace is only 10 seconds. If a worker does not respond
to TERM, killing the parent process group can terminate its cleanup agent before
that agent escalates against separate worker sessions. Unknown or surviving
workers can continue writing/holding GPUs after the watcher exits. This is a
concrete lifecycle gap, not evidence that all stops necessarily leak workers.

Track exact task descendants from launch, independently of native logs; bind
PID/start ticks and ancestry/attempt identity, then implement bounded TERM/KILL
and verified exit for only those processes. Cover stop before any rank logs,
partial rank discovery, separate sessions, and TERM-resistant workers. Never
signal the master, services, UID-wide sets, or broad command patterns.

### F4 — One-shot hashing suspends the active storage guard

`capture_helper_once()` blocks in unbounded `subprocess.run` at watcher line 416,
called synchronously at line 578. No storage sampling occurs until it returns.
The helper hashes mapped files using ordinary filesystem I/O; slow/stalled NFS
can therefore suspend the advertised three-second floor check while training
continues. The helper is also outside the runner's timeout process tree.
Keep storage/deadline supervision live while the single helper runs; bound and
clean up the exact helper process. Test a nonreturning helper with simulated
storage-floor crossing. No repeated GPU experiment is needed for this gate.

## Checks and accepted integration

The targeted CPU [reproduction](check_integration.py) confirmed F1 using actual
pins, F2 using a visible incomplete receipt, and F3's early-stop target set with
mock signals. The final invocation passed; [output](check_integration.log) records
the expected watcher rejection. An initial test setup supplied an incomplete
mock READY record and hit `KeyError: watcher_pid`; that harness mistake was fixed
before the asserted race result. F4 follows directly from the blocking call.

Reused the preparation's seven watcher fixture groups and ten helper groups;
did not rerun them. Their reported passes do not cover these integration gaps.
Preserve the preparation's earlier missing sanitizer hash failure as failed
setup evidence; this review does not replace it with a pass.

The config/entrypoint preserve graph-OFF/mix0, all 18 compiled/excluded layers,
Flex/FP8/HybridEP EP8/FSDP2, selective AC, model/loss compilation, seed42,
60 updates/30000 schedule, CPfalse, and no completion/timing observer/profiler.
The exact sanitizer CLI wraps torchrun with all-process tracking, patching=0,
bounded reports, no filters/forced blocking/coredumps, zero restart and one
outer deadline. Dump-folder and native entrypoint wiring match the inspected
producer. `train.py:40–45`, `jsonl_handler.py:81–87,140,177`, and
`trainer.py:793` support the parser's first event, rank/PID, sequence, filename
and optimizer event schema; the real-prefix fixture is relevant evidence.
All-eight update-1 checks, distinct live PID checks and an exclusive one-shot
marker precede helper invocation. Actual sanitizer tracking is still unproven.

## Runtime cost and remaining gates

No training arithmetic or source behavior changes in this preparation. The
sanitizer still adds interception, host/device tracking, memory use and possible
peer skew with patching disabled. The watcher adds per-second log/proc reads,
three-second storage sampling/fsync and once-only CPU mapped-file hashing.
These may contend for host/NFS resources. No device wait, CUDA API or barrier is
introduced by the helper. Actual time, peak memory and slowdown remain
unmeasured; this is exception attribution only, with no performance or memory
correctness claim.

After focused fixes and affected CPU checks, root must perform fresh exact-eight
authorized UUID/H100 idle ownership, NV18 topology, source/config hash and changed
set checks, and NFS byte/inode/write/fsync checks. A 4 GiB GPU-free field alone
does not establish idle ownership. The 64 MiB estimate excludes fresh scan and
compiler cache growth; the active 4 GiB floor is no reservation or guarantee.
Actual `%p` log naming, eight-rank sanitizer coverage, tool compatibility,
same-run loaded quantize/GEMM receipt, fault attribution and complete bounded
cleanup require evidence from the one released attempt. No automatic retry or
fault clearance follows a clean, failed, uncovered or nonreproducing run.
