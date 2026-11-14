# Baseline runner launch failure

Historical failed-launch diagnosis only: the owner later authorized and completed three native runs; see the [final report](report.md) and [execution record](native-execution.md).

14 November 2026. **The shell-background `nohup` route failed to retain its
child in this execution harness. The exact signal/cause of the original runner
termination is unknown.** No TorchRun child or optimizer update is evidenced;
this is not a training failure or performance result. No training retry follows.

## Finite diagnostic acceptance

- Inspect the frozen runner's launch and finalization ordering.
- Run one eight-second, standard-library-only CPU child using the same
  shell-background pattern and preserve the complete tool return.
- If that route fails, compare one runtime-managed foreground session.
- Record observed behavior, causal limits and the minimal future route. Stop.

All four items are complete. No training, CUDA import, allocation, package
installation, original-runner execution or source modification occurred.

## Original attempt: established facts and limits

The original command used `nohup /opt/torchtitan-venv/bin/python3.12
runs/overnight-streaming-perf/run.py --run-id combined-four-baseline --steps 140
--warmup 40 --window 100 --num-prefetch-batches 0 --epoch-size 102400`, with
stdout/stderr redirected to the empty
`artifacts/raw/combined-four-baseline-runner.stdout.log`, stdin `/dev/null`, and
shell `&`. Its observed PID was 17656, PPID/PGID 17654; it subsequently vanished.
The worker retained only the tool's `.output`, so the original outer exit code
and possible session ID are unavailable. A shell exit code would in any event
not establish the background child's successful completion.

The actual attempt directory is
`../overnight-streaming-perf/artifacts/raw/combined-four-baseline/`, because the
runner's `RAW_ROOT` is relative to its own directory. It contains only
`config/experiment.yaml` (361 bytes) and `preflight/task-tmpdir.json` (281 bytes).
No runtime, source-final, metadata, native-process, completion, failure or exit
receipt exists. The previously observed empty task temporary directory remains
evidence, not something this diagnosis removes.

In [`run.py`](../overnight-streaming-perf/run.py), `main` writes these two files
before `source_snapshot`, `_config_hashes`, `_runtime_preflight` and metadata.
The runtime probe's first persisted output follows a completed `nvidia-smi`
query. Thus the receipts place termination after initial setup and before
successful runtime preflight persistence; they do **not** identify the precise
instruction or prove that preflight never began. The `except BaseException`
and `finally` paths normally emit failure/exit/status and temporary cleanup
receipts. Their absence is consistent with abrupt external termination, but
does not prove a specific signal or actor. SIGTERM/SIGKILL, a harness group
cleanup, and other abrupt failures cannot be distinguished retrospectively.

## One matched lifetime probe

The tiny [`child.py`](artifacts/raw/launch-probe/child.py) imports only
`json`, `os`, `time` and `pathlib`; it writes its identity, sleeps eight seconds,
then writes an end marker. It ran once via the same `nohup ... &` pattern with
the same interpreter and redirected descriptors.

- Full initial `exec_command` receipt: chunk `07c1e5`, `exit_code: 0`,
  `wall_time_seconds: 0.000012693`, **no `session_id`**. Output PID was 17964;
  `ps` observed PPID/PGID/SID 17961. The anomalous elapsed-time rendering from
  that `ps` output is not used as timing evidence.
- After one ten-second runtime-managed wait, PID 17964 was absent. Neither
  `start.json` nor `end.json` existed; redirected stdout/stderr was zero bytes.
  The child did not reach its first persisted marker.
- `/proc/self/cgroup` was `0::/`; visible `memory.events` before and after had
  `low`, `high`, `max`, `oom`, `oom_kill` and `oom_group_kill` all zero. These
  counters provide no positive OOM evidence; they are not the original process's
  missing termination receipt and do not exclude every external resource event.

This independently reproduces the failed **launch method**, including disappearance
before useful child work. It supports execution-harness lifetime handling as
the leading explanation, without establishing the harness's internal cleanup
implementation or the original signal. `nohup` does not create a new session
or guarantee survival of an execution harness's descendant/group cleanup.

## One foreground session comparison and future route

One standard-library-only foreground Python child used the same interpreter,
wrote [`foreground-start.json`](artifacts/raw/launch-probe/foreground-start.json),
slept eight seconds and wrote
[`foreground-end.json`](artifacts/raw/launch-probe/foreground-end.json).
Initial tool receipt: chunk `b29440`, `session_id: 76135`, no exit code yet,
PID 18097, PPID 11024, PGID/SID 18097. Resuming that session with `write_stdin`
returned chunk `080cfa`, `exit_code: 0`, output `complete`. The markers span
8.001249776 seconds; the process was absent after completion.

For any **separately authorized** rerun, use foreground `exec_command` with a
short yield, retain its returned `session_id`, and resume that exact managed
session until its terminal exit receipt. Keep the existing runner unchanged
and preserve persistent stdout/stderr and its existing native artifact receipts.
Capture the entire tool result; a printed PID alone is insufficient. This route
passed the CPU lifetime check only: it does not validate a multi-hour session,
native training, GPU availability or performance. Do not infer that adding
`setsid` to an unmanaged background command has been validated.

Inspected repository revision: `6c6cd9c0e41a0a5d3d00e1e252b3c120052d3507`.
Runner SHA-256: `175483109917115f93e8b03c3de61f6941d3d3b1258bae349aca13c15ebf94a8`.
No runner/source bytes changed. Remaining original-signal uncertainty is
irrecoverable from retained receipts; there is no further diagnostic experiment
or training launch in this assignment.
