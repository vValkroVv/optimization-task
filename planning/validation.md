# Validation protocol

This protocol is for an authorized, selected production workload. It does not authorize compute, source edits, data access, or dependency installation.

## Before an experiment

1. Verify the source revision and recursive dependency pins using [baseline provenance](../references/provenance.json). Record the working diff.
2. Freeze the workload: model/preset, data and split, tokenizer, sequence/packing rules, optimizer, global batch, schedule, seed, evaluation and checkpoint cadence, precision, AC/compile settings, and agreed quality thresholds.
3. Record hardware class, rank placement, mesh dimensions, exact process-group membership, network topology, software/build versions, launcher, and timer scopes. Check all allocated nodes before an authorized run.
4. Inspect the rank-zero resolved configuration and confirm active markers for the features under study. Do not infer activation from a requested setting.
5. Record the expected correctness and runtime costs before implementation.

## Correctness gates

For a computation-preserving change, use matched data and states. Compare loss, elementwise preclip gradients, norm and clip scale, postclip gradients, optimizer moments, and parameter updates. Check boundary cases for masks, packed sequences, short/long lengths, unequal valid-target counts, and stateful paths affected by the change. Compare eager and compiled execution when both are supported.

Use exact checks for structural invariants such as forbidden cross-document attention or causal zero gradients. Use tolerances only for explicitly identified floating-point outputs, with dtype, comparison method, and nondeterminism stated. A norm-only or weight-only comparison is insufficient.

Test checkpoint continuation from the same saved point, including optimizer, schedule, loader cursor, relevant RNG, and persistent model state. Report weights-only loading separately. If the candidate changes the mathematical function, use the accepted quality/convergence plan rather than claiming parity.

## Performance gates

- Run an identical A/A control pair to estimate noise.
- Alternate baseline and candidate in unprofiled runs with identical workload and topology.
- Separate startup, compilation, autotuning, warmup, stable steps, validation, checkpointing, and shutdown. Compare stable-step series and report medians and spread.
- Use traces to explain a bottleneck or overlap mechanism; estimate magnitude from repeated unprofiled runs.
- Define elapsed scopes for host submission, device work, synchronization/waits, transfers, collectives, and integrated training. Mark overlapping intervals; never add them as if they were disjoint.
- Include data preparation, metric reads, evaluation, checkpoint I/O, and other lifecycle costs at their production cadence.
- Pair memory results with the same phases and windows; identify allocator and device-level counters and their limits.
- Report failed/skipped checks, exclusions, noise, uncertainty, job and source identities, and remaining hardware or quality gates.

One trace, one run, an isolated kernel, or an unrepresentative fixture does not establish an end-to-end speedup.

## Reproducing the baseline smoke

This is the bounded startup/training check already completed, not a benchmark.
The maintained [launcher](../runs/100b-crop8-ep8-smoke/smoke.sh) is unchanged.
On the authorized GPU host, use the verified task checkout and existing runtime:

```bash
bash /workspace-SR008.nfs2/vkropotin/optimization-task/runs/100b-crop8-ep8-smoke/smoke.sh \
  /workspace-SR008.nfs2/vkropotin/optimization-task \
  /opt/torchtitan-venv/bin/python3.12 \
  /opt/torchtitan-venv/lib/python3.12/site-packages \
  YYYYMMDD-baseline-001
```

Choose a new attempt ID. This command launches eight GPU processes; execute only
with authorized compute and verified ownership. For a long remote run, use a
reviewed detached invocation with persistent logs, not a fragile SSH foreground
session. The launcher checks a clean GIFT tree, eight available H100/SM90 devices,
CUDA compatibility, NVCC and dependency identities. It refuses existing attempt
directories and uses a one-hour timeout with bounded termination of its own job.
Before launch, confirm current node access and NFS mapping; do not infer them from
this historical example. Do not terminate unrelated cluster processes.

The launcher runs `torchtitan.train` through eight-rank standalone torchrun,
with ten updates, scheduler horizon 65,000 and monitoring interval 5. Validation,
checkpoints, profiler and memory snapshots are disabled. The canonical YAML
otherwise remains unchanged. It uses offline cached smoke data; its existing
HF-cache reuse path is retained. If unavailable in a fresh environment, establish
the approved data cache before launching; do not silently change data or download.

Evidence is written under `runs/100b-crop8-ep8-smoke/artifacts/raw/<attempt>/`:
`metadata.txt`, copied source configs, `native/config/resolved_config.yaml`,
`logs/torchrun/`, `native/structured_logs/`, TensorBoard events and `exit-code.txt`.
The [existing receipt](../runs/100b-crop8-ep8-smoke/result.json) pins the successful
config hashes and executed source revision.

Acceptance requires all of the following, not just exit code 0:

- Verify the resolved config and required FP8, CuTe, compile, GFSDP2/EP8 and SAC
  activation. Account for the documented smoke overrides only.
- Each rank 0–7 completes ten optimizer updates and 320 forward/backward
  microbatches, exactly 32 per update. Check all-rank structured logs; an aggregate
  line from rank 0 alone does not prove this.
- All logged loss and gradient norms are finite. Monitoring emits at steps 5/10;
  retain ordinary/monitoring loss chunk settings 2/32. Source/hook correlation
  is not a direct chunk-count measurement.
- Inspect output inventory: no checkpoint, profiler trace or memory snapshot.
- Confirm clean distributed teardown, absence of owned run processes and released
  GPU compute allocations. Record exit status and log completeness.

The historical final loss 14.57517 and gradient norm 7.6661 are observations,
not bitwise acceptance thresholds for another run. The 72.61 GiB native metric
combines reserved peak and report-time non-PyTorch memory; do not add its printed
non-Torch component again or call it a device-wide instantaneous peak.

## Interpreting and recording a verdict

| Evidence | Supported conclusion | Not established |
| --- | --- | --- |
| Config decode / source checks | Configuration structure and inspected contracts | Active GPU kernels or runtime behavior |
| Finite ten-update smoke | Basic training execution for this workload/runtime | Convergence, rare-fault absence, numerical parity, speedup or resume |
| Loss/elementwise-gradient and state comparison | Agreement within predeclared scope/tolerances | General quality if the computation changes |
| Trace / graph replay counts | Observed mechanism in the captured window | Steady-state throughput magnitude |
| Repeated unprofiled controlled comparison | Measured effect and variability in the tested setup | Production-scale transfer without matching topology/workload |

Choose tolerances and a material performance threshold before viewing candidate
results; do not invent universal numeric tolerances in this document. Record the
noise estimate, median/spread, sample windows and memory tradeoff. An effect
indistinguishable from noise is inconclusive, not a win. Exclude startup and
profiling perturbation explicitly; do not select favorable steps after the fact.

When changing stateful/checkpoint behavior, the continuation gate above is
required. If an experiment deliberately omits checkpoints, state that its scope
does not establish continuation. Mathematical changes require an agreed quality
study rather than a relaxed parity threshold. Preserve failed controls and skipped
checks; stop and resolve a material correctness failure before promotion.
