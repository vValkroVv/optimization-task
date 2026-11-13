# Four-fusion single-run driver

`evaluate.py` selects one independent variant and reuses the existing
`runs/overnight-streaming-perf/run.py`. It does not define another training
configuration or measurement path.

## Invocation

Use a fresh run ID for every invocation, including repeats:

```bash
/opt/torchtitan-venv/bin/python3.12 runs/four-fusion/evaluate.py control tps --run-id control-tps-001
/opt/torchtitan-venv/bin/python3.12 runs/four-fusion/evaluate.py input trace --run-id input-trace-001 --dry-run
```

Variants are `control`, `input`, `routing`, `linear-ce`, and `ce`; modes are
`tps` and `trace`. Candidate selectors map to one patch each:

| Variant | Patch |
| --- | --- |
| `input` | `input-prepare.patch` |
| `routing` | `routing-metadata.patch` |
| `linear-ce` | `linear-ce.patch` |
| `ce` | `ce.patch` |

The control applies no patch. The first two patch files must exist and pass
validation before those selectors can run. Candidate patches may touch only
paths under `gift/`; the existing untracked helper modules remain available.

## Reproducing the final source

Keep the supplied final helper modules and tests, not just the activation
patches. [The final manifest](runtime/post-wrapper-implementation-pins.json)
pins their bytes. Corrected captures and repeat TPS runs also share the
setup-only canonical-head metadata in `gift/torchtitan/protocols/module.py`.
It is already present in this workspace. In a fresh `33b336a5` copy with the
final helper files, apply only that file's correction:

```bash
git apply --include=gift/torchtitan/protocols/module.py runs/four-fusion/linear-ce-parallel-wrapper-fix.patch
```

The remaining hunks in that correction file record changes to earlier helper
and test versions; do not apply them over the supplied final files. Keep this
common metadata on both sides of a corrected comparison. It adds setup Python
references, with no tensor or communication work; only the `linear-ce` selector
uses it. The [wrapper evidence](loss-parallel-wrapper-fix.md) explains the
failed initial activation and verified correction. Each invocation still
applies only its selected activation patch, so alternatives remain independent.

## Fixed workload

The driver uses the fixed accepted workload: native epoch size 51,200 and
prefetch depth 32. TPS mode runs 20 optimizer updates, with 10 warmup and 10
measured updates. Trace mode runs 11 updates, with 10 warmup, update 11
profiled, and at most 3 captured microbatches per profiled rank. It leaves the
native runner's accumulation setting at 32, so the diagnostic capture does not
shorten training updates. The runner retains its existing rank selection,
resolved-config checks, raw-artifact path guard, process-group cleanup and
result parser.

## Patch and process safety

Before mutation, the driver holds an exclusive advisory lock at
`.tmp/four-fusion-evaluate.lock`, checks that the task repository is the Git
root, verifies the core source targets are clean against `HEAD`, and
checks the selected patch with `git apply --check`. An already-applied selected
patch or dirty integration target fails closed. It then applies exactly the
selected patch and keeps that patch active while the existing runner executes.
The runner compares its task GIFT source snapshots before and after training.
No reset or stash is used.

The native runner writes detailed artifacts under its required
`runs/overnight-streaming-perf/artifacts/raw/<runner-run-id>` root. The wrapper
stores a compact JSON record under `runs/four-fusion/driver-records/` with the
patch SHA, exact driver and runner argv, runner receipt paths and status fields,
and whether the selected patch was restored. Detailed source/config/runtime and
result evidence stays in the native runner attempt. Runner IDs are namespaced as
`four-fusion-<variant>-<run-id>`; reusing a driver run ID is rejected.

The wrapper starts the runner in its own session. On SIGINT, SIGTERM, or SIGHUP,
it forwards SIGINT to the runner process; the existing runner then performs its
own exact process-group cleanup. The wrapper does not signal torchrun, mpirun,
sshd, or any other process group. It reverses only the selected patch after the
runner's `process.json` proves that the owned group stopped. Its interrupted-run
fallback requires an `INTERRUPTED` failure receipt, a final source receipt, and
an OS check that the recorded runner group no longer exists. If exit is
uncertain, the selected patch stays active, the record says so, and the driver
fails for manual recovery. A nonzero child result with confirmed group exit is
recorded as a failure after safely reversing only the selected patch.

`--dry-run` performs the same read-only baseline and patch checks, then prints
the patch decision and exact runner command. It does not apply a patch, create
a run record, or launch the native runner.

## Runtime-cost review

The wrapper adds no training operations, configuration overrides beyond the
fixed measurement workload, device synchronization, transfers, collectives,
compilation, or GPU memory allocations. Its Git checks and compact record writes
occur outside the training timer; while training runs, it only waits for the
existing runner. The child is the same pinned runner and owns all preflight,
training, tracing, result parsing, and task temporary-directory cleanup. Thus
the wrapper has no intended training-runtime cost; candidate kernel costs are
measured by the existing runner and are outside this driver review.

## Focused verification

The task-local tests use a temporary Git fixture and mocked runner receipts.
They cover fixed TPS/trace arguments, dry-run non-mutation, one-patch
application/reversal, acceptance of untracked helper modules, clean-source
rejection, nonzero child cleanup, and preserving a patch when process-group
exit is uncertain. They do not establish GPU, NCCL, kernel correctness,
training performance, or candidate activation.

The actual `linear-ce.patch` and `ce.patch` use unified `---` / `+++` headers;
both parsed and passed read-only `git apply --check` in a temporary repository
made from the task's `HEAD` loss file. The receipt reader also returned
`completion_status=PASS` and `group_exit_confirmed=true` for the completed V2
attempt `v2-long-epoch-candidate-perf-001`. Neither check applied a patch to the
task source or launched training.
