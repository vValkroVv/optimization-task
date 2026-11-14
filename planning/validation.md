# Validation protocol

This protocol is for an authorized, selected production workload. It does not authorize compute, source edits, data access, or dependency installation.

The [completed BF16 expert-weight reuse assignment](README.md#active-assignment-bf16-expert-weight-reuse)
overrode the generic future-run defaults below: one candidate used 60 updates,
20 warmup and measurement of updates 21–60; one diagnostic candidate used 21
updates, 20 warmup and at most three complete microbatches on update 21 for ranks
0/1. All updates retained 32 microbatches and the original LR schedule. Saved
controls and focused GFSDP numerical/lifecycle checks completed the finite
checklist. Ambiguous performance does not authorize repeats or another control.

## Before an experiment

1. Verify the source revision and recursive dependency pins using [baseline provenance](../references/provenance.json). Record the working diff.
2. Freeze the workload: model/preset, data and split, tokenizer, sequence/packing rules, optimizer, global batch, schedule, seed, evaluation and checkpoint cadence, precision, AC/compile settings, and agreed quality thresholds.
3. Record hardware class, rank placement, mesh dimensions, exact process-group membership, network topology, software/build versions, launcher, and timer scopes. Check all allocated nodes before an authorized run.
4. Inspect the rank-zero resolved configuration and confirm active markers for the features under study. Do not infer activation from a requested setting.
5. Record the expected correctness and runtime costs before implementation.

## Future candidate order: performance first, then local numerical assessment

The [active assignment](README.md#active-assignment-four-independent-fusion-experiments)
authorizes four independent comparisons against accepted V2, including matched
bounded traces for every candidate. Historical cleanup-only restrictions below
describe completed work. This protocol replaces the former full-model numerical
prerequisites.

Start with a matched short baseline/candidate pair: 10 warmup optimizer updates
and 10 measured optimizer updates, retaining 32 microbatches per update. Preserve
LR horizon 65,000 and LR warmup 2,000; early stop does not rescale the scheduler.
Disable detailed monitoring callbacks/hooks, use ordinary two-chunk loss, and keep
lightweight native loss/time/TPS/error logging. EMA and training losses remain.
Validation, checkpoint saves, memory snapshots, numerical capture, profiling and
CUDA Graphs are disabled in unprofiled performance runs. Preserve all other
[baseline settings](../docs/baseline.md).

Extend warmup only when actual compilation or continuing runtime settling affects
the measured window. A cache entry alone is not sufficient evidence. No mandatory
110-update warmup, 50-update window, exhaustive compiler-cache validation or large
A/A or A/B/A/B matrix precedes an initial comparison. A single pair is provisional;
repeat promising or ambiguous effects when needed to distinguish them from noise.
A failed control cannot establish a candidate gain. Historical monitoring-inclusive
TPS is not directly comparable with the new monitoring-off configuration.

Record physical input and valid-target TPS separately, global/per-GPU units,
step series, median/spread, timer boundaries and excluded lifecycle costs. Pair
memory scopes by phase. Separate device compute, host work, synchronization/idle,
copies/communication and integrated elapsed time when evidence permits; mark
unavailable scopes and never sum overlapping durations. Reuse historical traces
for mechanism. The active assignment explicitly requires matched new candidate
traces, reusing a control capture only when source, workload, runtime and capture
conditions remain applicable.

For a candidate worth retaining after performance measurement:

1. Identify the smallest meaningful affected block. Capture representative inputs
   at its boundary and only weights/state needed to replay it.
2. Run original and changed blocks on identical inputs; compare outputs through
   the actual supported path, including eager/compiled behavior when affected.
3. If backward changes, compare that affected block with matched activations and
   incoming gradients. Do not collect whole-model gradients, parameters or AdamW state.
4. Interpret differences using accumulation precision, casts, rounding, reduction
   order and fusion behavior. Baseline output is not assumed exact; use a
   higher-precision local reference when needed to assess relative accuracy.
5. Justify comparison methods and tolerances for the block/dtypes. There is no
   universal tolerance or default bitwise requirement. Keep unexplained
   differences explicit; do not claim candidate acceptance while they are material.

Check structural invariants and affected stateful loader behavior with scoped
checks. A local replay does not establish full-model continuation or convergence.
Deliberate mathematical changes still need an agreed quality criterion. Do not
implement another numerical framework or run these local checks during cleanup.

## Trace capture window

For future authorized trace captures, record at most **3 complete microbatches
per rank trace** after warmup. Limit the profiler window; keep training microbatch
size 2 and gradient accumulation at 32 microbatches per optimizer update.

TPS estimation keeps the original unprofiled procedure: 10 warmup optimizer
updates followed by 10 measured full optimizer updates, each with 32 microbatches.
Keep the original workload, LR schedule and timing/token scopes. Do not estimate
TPS from the shortened diagnostic trace. Existing full-update traces remain
historical evidence; this policy does not change their recorded coverage.

## Historical baseline smoke evidence

The [HF smoke report](../runs/100b-crop8-ep8-smoke/report.md) and
[receipt](../runs/100b-crop8-ep8-smoke/result.json) preserve the executed source
revision, configuration hashes, commands and all-rank evidence. Its ten updates
and 320 microbatches per rank exercised monitoring at steps 5/10 with ordinary/
monitoring chunk settings 2/32. The [historical launcher](../runs/100b-crop8-ep8-smoke/smoke.sh)
is retained as evidence, not a command to run against the current canonical YAML.
Reproduction would require the pinned executed revision/configuration and separate
compute authorization. The current canonical performance configuration disables
detailed monitoring and its monitoring-specific chunk setting; the old launcher's
monitoring override must not be assumed compatible with those defaults.

The historical final loss 14.57517 and gradient norm 7.6661 are observations,
not bitwise acceptance thresholds. Its 72.61 GiB native memory metric combines
reserved peak and report-time non-PyTorch memory, rather than a simultaneous
device-wide maximum. This smoke proves neither current monitoring-off execution
nor performance, convergence or continuation. No smoke is run during cleanup.

## Interpreting and recording a verdict

| Evidence | Supported conclusion | Not established |
| --- | --- | --- |
| CPU/configuration checks | Imports, parsing and inspected configuration contracts | GPU/NCCL behavior or speed |
| Historical finite smoke | Basic execution in its recorded workload/runtime | Convergence, numerical candidate acceptance or continuation |
| One short matched performance pair | Provisional effect in its stated window | General speedup or resolved noise |
| Repeated matched comparison when needed | Effect/variation in the tested setup | Transfer to another topology/workload |
| Matched local outputs/affected backward | Numerical behavior of the affected block | Whole-model parameter/Adam agreement or convergence |
| Historical trace analysis | Observed mechanism in its captured window | New monitoring-off throughput magnitude |

Preserve failed controls, skipped checks, exact source/config/runtime identities
and remaining gates. An effect indistinguishable from noise is inconclusive.
Record why more evidence changes the decision before adding checks. Stop after
the finite authorized question is answered.
