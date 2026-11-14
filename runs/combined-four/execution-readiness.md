# Combined-four execution readiness

Historical pre-launch inventory only: all three native runs later passed; see the [final report](report.md) and [execution record](native-execution.md).

Read-only execution inventory completed 14 November 2026. No training, new
allocation, install, source or runner edit, process termination, or publication
was performed. At inventory start the task was at `6c6cd9c0e41a0a5d3d00e1e252b3c120052d3507`.
The shared tree received concurrent integration edits during this inventory;
the source owner reports the baseline has since been restored. This inventory
did not make those edits. The execution worker must use the reviewed source
identities and freeze each intended state before launch.

Acceptance for this note: verify the current authorized route and GPU state,
inspect the native runner and result contract, parse the requested command
protocols, and identify saved baseline traces with their limits. These checks
are complete. The [integration assessment](integration-assessment.md) owns
candidate source composition; [protocol.json](protocol.json) and
[source-identities.json](source-identities.json) own the exact source setup.

## Current route and availability

The authorized route is direct local execution in existing one-node job
`lm-mpi-job-23027817-4bfe-4730-926f-5c2de590b32b`, on master
`lm-mpi-job-23027817-4bfe-4730-926f-5c2de590b32b-mpimaster-0`. Its current
`/etc/volcano/mpimaster.host` lists that master. Use the existing shell; no SSH
fallback or additional host is needed. The earlier
[BF16 readiness note](../bf16-expert-cache/execution-readiness.md) described a
previous job and is historical.

The task path `/workspace-SR008.nfs2/vkropotin/optimization-task` is on NFS4
from `pd62-nfs-002.sr008.aicloud.sbercloud.tech:/vol_fg_002/namespace_ai0001071-01058`.
This mapping was verified from the current host. `nvidia-smi` reported no
compute applications; all eight NVIDIA H100 80GB HBM3 devices showed 0 MiB used
and 0% utilization. The only allocation processes observed were its PID 1
`mpirun ... sleep infinity` wrapper and SSH service; no task or Perfetto process
was present. Preserve the allocation wrapper and SSH service. No current GPU
ownership conflict was observed.

`/opt/torchtitan-venv/bin/python3.12` is executable and `/etc/profile.d/wm.sh`
is present. Runtime package versions were not re-probed in this read-only
inventory. The runner checks them, the CUDA/NCCL stack, eight visible H100/SM90
devices, installed package/header hashes, task-local imports, and free GPU
compute contexts immediately before each launch.

## Runner and planned commands

Use [run.py](../overnight-streaming-perf/run.py) directly. The current runner,
measurement adapter, and parser hashes are respectively
`175483109917115f93e8b03c3de61f6941d3d3b1258bae349aca13c15ebf94a8`,
`2558db65e8b9591b1786fb54fc5fbfc31f92ec7dfa00adf9f5848832c78672bd`, and
`15c63032c850d0f03bae80ac90fee680152237c4d03107ae98a739ff809efe29`.
The runner creates one eight-rank, one-node TorchRun process group with
`--max_restarts=0`; it calls [measure.py](../overnight-streaming-perf/measure.py)
and native `torchtitan.train` directly. It does not use `gift/run_train.sh`.
All rank output is retained under the per-run raw artifact directory.

The requested update schedules and child arguments passed a pure
`run._parse_args` → `run._command` → `measure._parse_args` check using the
current runner. This imported the parsers only; it did not call `main`, create
run directories, probe CUDA, or launch training. Use unique run IDs and the
existing ignored raw root:

```bash
/opt/torchtitan-venv/bin/python3.12 runs/overnight-streaming-perf/run.py \
  --run-id combined-four-baseline --steps 140 --warmup 40 --window 100 \
  --num-prefetch-batches 0 --epoch-size 102400
```

```bash
/opt/torchtitan-venv/bin/python3.12 runs/overnight-streaming-perf/run.py \
  --run-id combined-four-candidate --steps 140 --warmup 40 --window 100 \
  --num-prefetch-batches 32 --epoch-size 102400 \
  --gfsdp-reuse-bf16-expert-weights
```

```bash
/opt/torchtitan-venv/bin/python3.12 runs/overnight-streaming-perf/run.py \
  --run-id combined-four-diagnostic --steps 21 --warmup 20 --window 0 \
  --profile-update 21 --profile-microbatches 3 \
  --num-prefetch-batches 32 --epoch-size 102400 \
  --gfsdp-reuse-bf16-expert-weights --bf16-cache-counters --capture-microbatches
```

The baseline omits the reuse flag, which resolves false, and explicitly uses
prefetch depth 0. Both candidate commands use depth 32 and reuse true. The
counter and microbatch-capture flags are diagnostic-only; keep them off in both
timed runs. `--bf16-cache-counters` is rejected unless reuse is true and a
profile-update is selected. The measured window is updates 41–140. At the
canonical 32 microbatches per update, it covers 209,715,200 global physical
input tokens. The runner retains seed 2048, scheduler total/warmup 65,000/2,000,
logging each update, the existing crop config, and the 32-microbatch update.

For the timing runs, generated config disables the profiler, memory snapshots,
validation, checkpointing, and detailed monitoring callbacks; ordinary
two-chunk loss logging remains. The adapter uses two CUDA synchronizations at
the timing-window boundaries, not per update. It reports the eight-rank
synchronized elapsed envelope, global physical-input and valid-target TPS,
per-rank cadence, per-rank allocator peak allocated/reserved memory, and
per-rank allocator retry/OOM deltas. Startup, warmup, and teardown are outside
the measured window. The diagnostic run has no TPS or matched-window memory
result.

The helper parser accepted all three schedules and emitted the expected native
overrides: baseline prefetch 0/reuse false; candidate and diagnostic prefetch
32/reuse true; epoch size 102400; seed 2048; scheduler 65000/2000; profiling
and snapshots disabled for timing; profile ranks `[0, 1]`, skip-first 19, one
warmup/active profile cycle, shapes off, and snapshots off for the diagnostic.
The native resolved YAML is stored at each attempt's
`native/config/resolved_config.yaml`; `run.py` verifies it and the runner returns
PASS only after that check and all-rank result parsing. This config check runs
after the child exits, so inspect the saved YAML before interpreting results.
The resolved-config check covers the requested training, seed, scheduler,
prefetch, epoch, reuse, batch/sequence, parallelism, FP8, compile, SAC, and
callback settings.

`four-fusion/evaluate.py` cannot express this plan unchanged: it hardcodes
20/10/10 and 11/10 profiles and applies only one patch. No native-runner change
is needed for the schedules or measurements. Run only after the source owner
has prepared and frozen the intended source state; keep it unchanged throughout
each run. The native receipt stores the full `gift/` source snapshot and config
hashes, but does not name an externally applied patch stack.

## Trace and saved baseline evidence

The native runner writes plain `rank0_trace.json` and `rank1_trace.json` files;
neither the runner nor parser compresses them. The parser requires exactly
those selected-rank traces and validates the three complete microbatch markers,
GPU projections, backward coverage, and token-count collective. It also checks
all eight ranks completed 21 updates, one train step and optimizer step, and
32 microbatch calls in update 21. Profiling stops collection after three
microbatches on ranks 0/1 while full accumulation and the optimizer update
continue. Preserve raw JSON for validation; the current parser does not accept
gzip input.

The closest saved update-21 depth-0 baseline traces are:

- `../v2-prefetch-baseline/artifacts/raw/v2-long-epoch-baseline-trace-001/native/profiling/traces/iteration_21/rank0_trace.json.gz`
- `../v2-prefetch-baseline/artifacts/raw/v2-long-epoch-baseline-trace-001/native/profiling/traces/iteration_21/rank1_trace.json.gz`

They are present and covered by
[trace-captures-long-epoch-001.json](../v2-prefetch-baseline/trace-captures-long-epoch-001.json),
which records lossless gzip/uncompressed hashes and PASS coverage. They use
prefetch 0, reuse false, epoch size 51200, update 21 after 20 warmup updates,
and three traced microbatches. The paired saved depth-32 traces are in
`v2-long-epoch-candidate-trace-001` and use reuse false, also at epoch size
51200. Both are historical source/config evidence, not exact traces for the
current source and epoch size 102400. The four-fusion control traces target
update 11 and are not substitutes for the requested update-21 baseline.

## Completion and remaining gate

The default runner timeout is four hours. It starts one process group in a new
session, uses no automatic TorchRun restart, waits for normal group exit, and
limits TERM/KILL cleanup to that owned group if needed. It removes only its
validated task-local `.tmp/run-<random>` directory after the group stops; it
preserves scratch if children survive. Per-run receipts include all eight rank
sidecars, launcher and TorchRun logs, `summary.json`, resolved config, source and
config hashes, process/group status, exit code, and completion/failure status.

Route and GPU availability are ready for the authorized existing allocation.
The baseline is restored and the [independent review](review.md) passes.
The exact commands and source identities are frozen in [protocol.json](protocol.json)
and [source-identities.json](source-identities.json). Native execution remains open. Recheck the GPU preflight
before each run; do not use the historical job or modify source while a run is
active.
