# Native BF16 expert-cache execution

Both authorized native launches completed serially on 14 November 2026. Each
runner returned exit 0, `status.txt` is `PASS`, all ranks completed their
requested updates, and each completion receipt reports unchanged source/config
and removed task scratch. Neither run had a launch or training failure. Timing
run allocator retry/OOM deltas were zero on every rank; the diagnostic run has
no matched memory window or allocator-counter deltas. The timing run is the
only run with a TPS/memory window; no statistical significance or general
speedup claim follows from this single candidate and its historical control.

## Frozen source, configuration, and runtime

- Job `lm-mpi-job-8dff1e19-674b-42ed-96a9-bcd042dfefe5`, one node, eight
  NVIDIA H100 80GB HBM3 GPUs. Execution used the direct task path on the
  verified NFS4 mount `/workspace-SR008.nfs2/vkropotin/optimization-task`.
- Runtime: `/opt/torchtitan-venv/bin/python3.12` (Python 3.12.3), Torch
  `2.15.0.dev20260822+cu130`, CUDA 13.0, NCCL 2.30.7, NVCC 13.2.51.
- Source HEAD: `657e77a4f8ddb116b75925e5671b00642a0e4d32`; frozen source patch
  SHA256: `4634d6ac225cc0993567de491605cb31b44121965ab633ea292179fa7436cb26`
  ([source identity](source-identity.json)). Both run completion receipts
  verify `source_unchanged=true`.
- Canonical experiment SHA256:
  `121a5f11a4b431cb632badc08f9ccf718fc4fd5640c68c9279be57921c012479`;
  generated experiment SHA256:
  `e6a89b9ad8a9b68dbf4f6bc49fff352f9a1386358c23f06c6074aa0ebac08b1a`.
  Resolved-config SHA256 is `9a176351592815abe3e0b60b361cb9f1e44ee44a83ffaccb5c79b78bc2e09c1e`
  for `bf16-cache-tps-001` and
  `a71150cc5997eafd3d6d6cedc6bfab13c5c1472c7a06e0befdc1051f4cc5777e` for
  `bf16-cache-trace-001`; both report `config_unchanged=true`.
- Resolved behavior retains the accepted eight-layer crop, PP1/EP8/GFSDP2,
  local/global batch 2/512 (32 microbatches per optimizer update), sequence
  length 4096, FP32 training/master and reduction arithmetic, BF16 parameters,
  FP8, compile, selective SAC, EMA, and MTP. The run overrides retain LR
  horizon/warmup 65,000/2,000, prefetch depth 32, and logical epoch 51,200.
  The four prior fusion candidates remain disabled. Validation, checkpointing,
  detailed monitoring, and memory snapshots are off in both runs. Profiling
  and cache counters are off in the timing run.

## Commands and child-protocol checks

Before each launch, the runner `_parse_args` / `_command` child command was
passed through `measure._parse_args`. The paths below are canonical receipts
for the generated child argv and resolved configuration.

```bash
/opt/torchtitan-venv/bin/python3.12 runs/overnight-streaming-perf/run.py \
  --run-id bf16-cache-tps-001 --steps 60 --warmup 20 --window 40 \
  --num-prefetch-batches 32 --epoch-size 51200 \
  --gfsdp-reuse-bf16-expert-weights
```

The parsed timing schedule was exactly 60/20/40 optimizer updates. Profiling,
cache counters, and microbatch capture were off. The resolved cache flag was
true once; the LR, prefetch, epoch, FP8/compile/SAC/EMA/MTP, and monitoring
settings matched the accepted configuration.

```bash
/opt/torchtitan-venv/bin/python3.12 runs/overnight-streaming-perf/run.py \
  --run-id bf16-cache-trace-001 --steps 21 --warmup 20 --window 0 \
  --profile-update 21 --profile-microbatches 3 \
  --num-prefetch-batches 32 --epoch-size 51200 \
  --gfsdp-reuse-bf16-expert-weights --bf16-cache-counters \
  --capture-microbatches
```

The parsed diagnostic schedule was exactly 21/20/0, with update 21 profiled on
ranks 0/1, skip-first 19, and three complete microbatches captured per trace.
All ranks completed the full 32-microbatch accumulation and optimizer update.

## Timing candidate

Canonical run root:
`runs/overnight-streaming-perf/artifacts/raw/bf16-cache-tps-001/`.
The canonical receipts are `metadata.json`, `native/config/resolved_config.yaml`,
`summary.json`, `completion.json`, `process.json`, `exit-code.txt`, and
`measurement/rank-00.json` through `rank-07.json`.

| Metric | Result |
| --- | ---: |
| Updates | 60 total; 20 warmup; measured updates 21–60 (40 updates) |
| Global physical input tokens | 83,886,080 |
| Global physical input TPS | 143,736.69049696592 |
| Per-GPU physical input TPS | 17,967.08631212074 |
| Global supervised-target TPS | 143,701.59853151257 |
| Full eight-rank elapsed envelope | 583.609374266 s |
| Rank-0 host update cadence, median / min / max | 14.6254128915 / 13.536819126 / 15.438273734 s |
| Maximum allocated / active peak across ranks | 70.81349802017212 GiB |
| Maximum reserved peak across ranks | 71.939453125 GiB |
| Allocator retries / OOMs, every rank | 0 / 0 |

All eight rank receipts report `complete`, 60/60 updates, and no training error.
The summary has `errors=[]` and finite recorded numeric values. Its timing
window includes normal loading, transfers, compute, communication, optimizer,
and lightweight logging; it excludes startup, the first 20 updates, and
teardown. Rank-0 cadence is host callback timing, not a device-only step time.
No detailed profiler or cache counters perturbed this window.

The post-training activation snapshot reports all seven eligible expert groups
(`layers.1[efsdp]` through `layers.7[efsdp]`) enabled on each rank, generation
60. Counters were off, so this run does not provide refresh/hit/direct-use
counts. The compatible uncached V2 reference is the saved historical
updates-21–60 control in
[`pair-long-epoch-001.json`](../v2-prefetch-baseline/pair-long-epoch-001.json),
at 140,933.045473 global physical input tokens/s. It is not a contemporary
paired control; no repeat, uncertainty estimate, or significance claim is
made.

## Diagnostic candidate and counter coverage

Canonical run root:
`runs/overnight-streaming-perf/artifacts/raw/bf16-cache-trace-001/`.
The summary and per-rank receipts are `summary.json` and
`measurement/rank-00.json` through `rank-07.json`.

All eight ranks report 21/21 complete updates, one train-step call, one
optimizer-step call, and `profile_capture.microbatch_calls=32` at update 21.
Ranks 0/1 each collected three complete microbatches and one train-step marker;
ranks 2–7 collected no trace, as configured, while still recording 32 calls.
The generic `microbatch_calls_by_step` map is empty in profile mode; the
profile-specific call-count receipt records the complete 32-call update on
every rank.

Each rank has 22 cache snapshots: `before_train` at update 0 and an
`after_optimizer_update` snapshot for every update 1–21. All seven eligible
groups were active on all eight ranks. For every group on every rank, the
update-20-to-21 counter delta is:

| Generation | Refreshes | Hits | Direct uses | Invalidations |
| ---: | ---: | ---: | ---: | ---: |
| +1 | +1 | +63 | +64 | +1 |

After update 21, each eligible group reports generation 21, refreshes 21,
hits 1,323, direct uses 1,344, and invalidations 21. The diagnostic run has no
TPS window or matched-window memory peak; `measurement_status=disabled` is
expected for this protocol.

Canonical, unmodified profiler traces (three complete microbatches each):

- Rank 0:
  `runs/overnight-streaming-perf/artifacts/raw/bf16-cache-trace-001/native/profiling/traces/iteration_21/rank0_trace.json`
  — 27,555,783 bytes, SHA256
  `fdcf72769a9ddbad51db80b0bf7663784f7bbf4c72971e3ee8d7925cfc800203`.
- Rank 1:
  `runs/overnight-streaming-perf/artifacts/raw/bf16-cache-trace-001/native/profiling/traces/iteration_21/rank1_trace.json`
  — 27,566,679 bytes, SHA256
  `3594219f6ea8282cd85e963ea7d5db2b51a2dfc52732e91de08f59fa3b419902`.

## Exit and cleanup

Both runs have `status.txt=PASS`, `exit-code.txt=0`, `completion.json` status
PASS, `source_unchanged=true`, and `config_unchanged=true`. Their exact runner
process groups stopped without orphans, and task-local scratch was removed.
After the second run, the authorized host had no GPU compute applications;
all eight H100s reported 0 MiB used and 0% utilization. The allocation's PID 1
and existing Perfetto analysis servers were preserved. No additional launch,
control, repeat, or training-source edit was performed here; original plain
traces were left unchanged for mechanism analysis.
