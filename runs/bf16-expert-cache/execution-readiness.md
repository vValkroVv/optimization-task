# BF16 expert cache execution readiness

Read-only reconciliation completed 13 November 2026 at starting task HEAD
`657e77a4f8ddb116b75925e5671b00642a0e4d32` (`main`). No launch, allocation,
installation, source edit, process termination or publication occurred here.

Acceptance for this bounded reconciliation: verify the existing execution route,
inspect native command/measurement parsing and cleanup, identify minimal plumbing,
and select saved compatible V2 evidence with explicit limitations. All four items
are complete; feature implementation and GPU acceptance remain separate gates.

## Verified execution route

The current shell already runs on
`lm-mpi-job-8dff1e19-674b-42ed-96a9-bcd042dfefe5-mpimaster-0`, in existing job
`lm-mpi-job-8dff1e19-674b-42ed-96a9-bcd042dfefe5`. Use direct local execution.
The established record explains why the historical `sr008-gpu` alias is not used
([runtime route](../v2-prefetch-baseline/report.md#runtime-and-integration-preflight)).
No SSH fallback, notebook host or new allocation is needed.

`/etc/volcano/mpimaster.host` contains this one master; there is no worker host
file. This is the complete one-node allocation. All eight NVIDIA H100 80GB HBM3
GPUs currently report 0 MiB used, 0% utilization and no compute applications.
The only pre-existing task analysis processes observed are Perfetto servers;
preserve them and PID 1's allocation `mpirun ... sleep infinity` wrapper.

Physical task path is `/workspace-SR008.nfs2/vkropotin/optimization-task`, mounted
NFS4 from `pd62-nfs-002.sr008.aicloud.sbercloud.tech:/vol_fg_002/namespace_ai0001071-01058`.
`/opt/torchtitan-venv/bin/python3.12` and `/etc/profile.d/wm.sh` exist. The existing
runner (`run.py:512 _runtime_preflight`, `run.py:50 WM_SCRIPT/WM_BASH`) sources
that profile and verifies Python 3.12.3, Torch
2.15.0.dev20260822+cu130, CUDA 13.0, NCCL, package/header hashes, eight SM90 GPUs,
NVCC V13.2.51 and task-local torchtitan imports before each actual launch.
The saved environment evidence remains applicable; the full runtime probe was
not repeated here because it belongs to the native run-local preflight.

All-node lightweight preflight on this verified one-node allocation:

```bash
hostname
cat /etc/volcano/mpimaster.host
findmnt -T /workspace-SR008.nfs2/vkropotin/optimization-task
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
ps -eo pid,ppid,args | rg 'optimization-task'
```

Recheck immediately before launch. A changed host list requires renewed all-node
verification; this report grants no permission to probe guessed hosts.

## Native runner and parsed protocol

Use [run.py](../overnight-streaming-perf/run.py), bypassing
[four-fusion/evaluate.py](../four-fusion/evaluate.py), whose `_build_command`
hardcodes 20/10/10 timing and update-11 diagnostics. The native runner already
accepts the required explicit protocol (`run.py:234 validate_args`,
`run.py:266 _parse_args`, `run.py:635 _training_overrides`,
`run.py:731 _command`, `measure.py:1037 _parse_args`). Its `_command` constructs one
`python3.12 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8
--max_restarts=0` child running [measure.py](../overnight-streaming-perf/measure.py).
The helper receives `--steps`, `--warmup`, `--window`, profiling arguments,
then `--`, the generated experiment YAML, and native training overrides.

These base commands were **not executed**. Add the implemented feature opt-in
and diagnostic-only counters through the minimal plumbing below before use:

```bash
/opt/torchtitan-venv/bin/python3.12 runs/overnight-streaming-perf/run.py \
  --run-id bf16-cache-tps-001 --steps 60 --warmup 20 --window 40 \
  --num-prefetch-batches 32 --epoch-size 51200

/opt/torchtitan-venv/bin/python3.12 runs/overnight-streaming-perf/run.py \
  --run-id bf16-cache-trace-001 --steps 21 --warmup 20 --window 0 \
  --profile-update 21 --profile-microbatches 3 \
  --num-prefetch-batches 32 --epoch-size 51200 --capture-microbatches
```

Preserve the existing ignored raw root and shared offline cache:
`runs/overnight-streaming-perf/artifacts/raw/<run-id>/`. Compact new results belong
under this experiment. No alternative framework or four-fusion patch application
is needed. The four previous variants are retained as unapplied integration
patches; their helper modules have no active control imports, as recorded in
[the report](../four-fusion/report.md#evidence-and-decisions).

A pure import with `PYTHONDONTWRITEBYTECODE=1` exercised `run._parse_args`,
`run._command`, and `measure._parse_args` for both requested protocols; no
`main`, run directory creation or runtime probe was invoked. Both child parsers
receive exactly 60/20/40 and 21/20/0 respectively. The diagnostic profile ranks
are `[0, 1]`; native skip-first is 19, profiler warmup/active are 1/1 and repeat
is 1. The bounded helper stops capture after three complete microbatches while
normal 32-microbatch accumulation continues. No timing default leaks through.

Native overrides preserve LR total/warmup 65000/2000, seed 2048, logging every
update, no checkpoint, validation or memory snapshot, local tokenizer, V2
prefetch 32 and epoch 51200. The canonical crop determines accumulation 32,
EP8, expert FSDP1, dense FSDP8, FP8, compile, SAC, EMA and MTP. Generated YAML
removes detailed monitoring callbacks. Timing is updates 21–60, with
83,886,080 global physical input tokens. Feature activation, final child argv,
rank-0 resolved config and full accumulation must be verified again after the
feature-specific plumbing exists.

### Minimal pending plumbing

The native runner has no arbitrary passthrough CLI. Add narrowly scoped,
default-off candidate and diagnostic-counter arguments, translate them to the
actual implemented config/environment interface in `_training_overrides` or
`_run_environment`, record them in metadata and verify resolved/activation
receipts. Counter reporting must observe every group's reuse across all 32
microbatches and successive optimizer boundaries, outside the three-microbatch
trace cap. Keep counters off in timing. Reuse the existing helper's update
boundaries and persistent rank sidecars; no new timing framework is needed.
Feature names are intentionally unspecified until the implementation owns them.

Runner cleanup (`run.py:770 _terminate_owned_group`, `run.py:807 _launch`,
`run.py:395 _cleanup_task_tmpdir`) is bounded to the session created by
`Popen(start_new_session=True)`.
It signals that exact child PID/process group, waits, and uses SIGKILL only for
surviving members of that group. No broad `pkill`, MPI cleanup or shared-service
termination occurs. Temporary cleanup is restricted to validated task-owned
`.tmp/run-<8 random characters>` after the group stops; surviving children retain
scratch. The runner freezes source/config before and after training and fails
if either changes. Do not edit source while a run is active.

## Saved V2 control and diagnostic evidence

Use the depth-32 entry **`v2-long-epoch-candidate-perf-001`** in
[pair-long-epoch-001.json](../v2-prefetch-baseline/pair-long-epoch-001.json).
It is the uncached V2 control for this new candidate. Its raw directory is
`runs/v2-prefetch-baseline/artifacts/raw/v2-long-epoch-candidate-perf-001`.
The saved source HEAD was `3b6b8928ec4b15a529d07ee99e672f2fc75ef73a` with
tracked-plus-untracked source digest
`ac22aae7ced85f3970b89e420010d87076b1ed0fa58b162b3aa245215aae35b7`.
Current native run.py and measure.py exactly match the saved hashes
`2e55a19740f9244c7ad515f31c53c1be2a8f621aa36e7a71c42c51b82d99992f` and
`8288698d1495cfdfed66953e22b0e84fc8b93f4720702f979309d38e9be99194`.

| Saved uncached V2 observation | Value |
| --- | ---: |
| Total / warmup / measured optimizer updates | 60 / 20 / 40 |
| Exact measured updates | 21–60 inclusive |
| Full eight-rank synchronized envelope | 595.219380370 s |
| Global physical input TPS | 140,933.045473 |
| Per-GPU physical input TPS | 17,616.630684 |
| Global valid-target TPS | 140,898.637991 |
| Rank-0 cadence median | 14.971230144 s |
| Rank-0 cadence min / max | 13.804412458 / 15.560705662 s |
| Maximum allocated / reserved peak | 70.839288 / 71.998047 GiB |
| Allocator retry / OOM deltas | 0 / 0 |

Full update series for every rank is already owned by
`runs[1].cadence.per_rank` in the pair receipt; do not duplicate it. It contains
all 40 entries, including update 50. Cadence is host intervals at native profiler
callbacks, not independently CUDA-synchronized update times. The integrated TPS
envelope includes loading, transfers, compute, communication, optimizer and
lightweight logging; startup, first 20 updates and teardown are excluded.
Memory peaks are allocator observations at matched lifecycle phases, not total
device memory. Both old variants used `--capture-microbatches`, adding one host
integer increment per microbatch; enabling it for the new timing candidate is
optional but aligns this tiny inherited instrumentation cost.

Use existing depth-32 update-21 rank-0/1 gzip traces, whose presence was checked:

- `runs/v2-prefetch-baseline/artifacts/raw/v2-long-epoch-candidate-trace-001/native/profiling/traces/iteration_21/rank0_trace.json.gz`
- `runs/v2-prefetch-baseline/artifacts/raw/v2-long-epoch-candidate-trace-001/native/profiling/traces/iteration_21/rank1_trace.json.gz`

[Capture coverage and hashes](../v2-prefetch-baseline/trace-captures-long-epoch-001.json)
and [existing analysis](../v2-prefetch-baseline/trace-analysis-long-epoch-001.json)
confirm three complete microbatches after 20 warmup updates and all-rank
accumulation 32. These retain original expert casts/copies. No fresh V2 trace
or control run is needed. The saved four-fusion controls use updates 11–20;
they are additional historical context and cannot be labeled updates 21–60.

## Remaining evidence limits

The saved V2 control has a compatible window/workload, but is historical, with a
different source identity and elapsed runtime conditions. Compare final resolved
configuration and relevant source changes; document the difference without
claiming a contemporary paired A/B experiment. No repeat/noise estimate,
confidence interval or statistical significance follows from one new candidate.
Counter activation, focused correctness, independent review, candidate memory,
TPS and trace mechanism are still open. No claim of speedup or readiness for
production PP4/EP32 is made. Stop after the two authorized candidate launches;
ambiguity does not authorize repeats or another V2 run.
