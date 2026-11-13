# V2 CPU prefetch above the monitoring-free baseline

Status: the long-epoch follow-up passed its matched TPS pair and four bounded
trace gates. V2 measured 2.12% higher input TPS in this provisional pair; the
original short-epoch pair measured 2.08% lower TPS and remains separate. No
general gain or production adoption is established. The default remains zero.

Publication: the subsequent user request authorizes committing these recorded
results with the supporting V2 source, runners and tests, then pushing to
`origin/main`. Run receipts retain the prepublication execution state. Large raw
logs, traces, datasets and caches remain outside Git.

The [compact results](results.json) index all gate receipts and final checks.
The [plan](../../planning/README.md)
owns acceptance and next actions; [baseline](../../docs/baseline.md) owns workload
defaults. All new outputs, scratch, caches and small loader checkpoints remain
inside this task directory. The existing MDS fixture is preserved.

## Long logical epoch comparison — complete

The follow-up comparison uses native Mosaic `dataloader.dataset.epoch_size:
51200` identically for baseline and V2. The same 896 physical MDS rows are
repeated logically: this creates neither new samples nor a more diverse corpus.
The epoch is 6,400 samples / 3,200 B2 microbatches per rank, or 100 optimizer
updates at accumulation 32. Both timing runs stop after update 60; updates 1–20
are warmup and exactly updates 21–60 form the measurement window, containing
83,886,080 global physical input tokens.

The [literal run plan](artifacts/raw/long-epoch-001-plan.json) gives distinct
`v2-long-epoch-*-001` IDs and serial commands. The ordinary workload and native
timer boundaries remain fixed. Separate diagnostics capture the input prefix,
global valid-token reduction and first three complete microbatches of update
21 on ranks 0/1; all eight ranks retain accumulation 32. Diagnostic TPS is not
used. Prior short-epoch results below remain a separate experiment.

The [independent review](long-epoch-review.md) found no material launch blocker.
The [runner verification](epoch-runner-verification.json) records
29 passing targeted checks, corrected failures and the absence of saved raw
stdout. The broader native measurement check is retained as **49 passed,
5 failed** in its [log](artifacts/raw/epoch-receipts-pytest-final-20261113.log):
all new receipts and timer tests passed; five unchanged cache-route tests use an
unsupported nested fixture path. The [measurement verification](epoch-receipts-verification.json) retains initial
wrong-runtime/path attempts; they do not establish source regressions. No production cache guard was changed.

New rank receipts record effective epoch length and shared epoch state before
training, before the existing timing boundary and after training, plus actual
V2 queue capacity/readiness/live producer before timing. The existing optional
completed-call counter is enabled identically in A/B to verify 32 microbatches
per update. It adds a host integer increment per microbatch; scalar receipt
reads and the queue snapshot occur outside the timer. V2 source, data files and
model arithmetic are unchanged. The TPS pair, trace verification and owned
process/GPU-release gates passed.

### Matched TPS results

The [new pair receipt](pair-long-epoch-001.json) owns exact commands, native argv,
resolved-config/data/runtime/source identities, complete update 21–60 series for
all eight ranks, native loss/gradient scalars and paired memory observations.
Both variants completed 60 optimizer updates on every rank, with exactly 32
completed forward/backward calls per update. Every rank's native logs contain
60 optimizer-end and step-end events. All retained loss and gradient values are
finite. The measured physical input count is exactly 83,886,080; the separate
valid-target count is 83,865,600.

| Observation | A: depth 0 | B: depth 32 |
| --- | ---: | ---: |
| Complete 40-update window, s | 607.826737325 | 595.219380370 |
| Global physical input tokens/s | 138,009.855192 | 140,933.045473 |
| Per-GPU physical input tokens/s | 17,251.231899 | 17,616.630684 |
| Global valid-target tokens/s | 137,976.161380 | 140,898.637991 |
| Per-GPU valid-target tokens/s | 17,247.020172 | 17,612.329749 |
| Rank-0 cadence median, s | 15.2742472445 | 14.9712301440 |
| Rank-0 cadence min–max, s | 14.208478927–15.973962590 | 13.804412458–15.560705662 |
| Rank-0 cadence IQR, s | 0.5923127603 | 0.6140789955 |
| Maximum allocated/active peak across ranks, GiB | 70.818533 | 70.839288 |
| Maximum reserved peak across ranks, GiB | 71.958984 | 71.998047 |
| Allocator retry / OOM deltas, every rank | 0 / 0 | 0 / 0 |

V2 has **2.1181% higher physical input TPS** in this pair, with **2.0742% less
window elapsed time**, both relative to A. Allocated/active peak increases by
21.2529 MiB (0.0293%); reserved peak increases by 40 MiB (0.0543%). These are
sampled allocator peaks at matched lifecycle phases, not simultaneous whole-device
memory peaks. The complete series retains every point, including update 50.

Integrated TPS uses `max(rank end) - min(rank start)` for the full eight-rank
window. The initial CUDA drain precedes the start timestamp; the final drain is
inside the end boundary. Normal loading, transfers, compute, communication,
optimizer and lightweight native logging remain included. Startup, updates 1–20
and teardown are excluded. Per-update intervals are host cadence at native
`Profiler.step`, without a CUDA synchronization per update; the last interval
includes the final drain. Their medians are descriptive cadence observations,
not independently synchronized per-update device throughput.

There is within-window variation: rank-0 cadence medians for updates 21–40 /
41–60 are 15.477071 / 15.116051 s for A and 15.197157 / 14.797599 s for B. The
corresponding cadence-derived global input rates are 135,500.57 / 138,736.76 and
137,996.34 / 141,722.45 tokens/s. These halves do not replace the complete-window
TPS and are not repeatability evidence. No window was trimmed or extended.

Host memory is observed immediately before timing and after timing:

| Boundary observation, GiB | A: before / after | B: before / after |
| --- | ---: | ---: |
| Maximum one-rank self + live-descendant RSS | 6.927498 / 6.955513 | 6.927765 / 6.956703 |
| Sum of eight rank-tree RSS observations | 54.307644 / 54.535244 | 54.285221 / 54.523357 |
| Maximum rank self process-lifetime HWM | 3.464645 / 3.493168 | 3.464828 / 3.495033 |

RSS sums can double-count shared pages; descendants include loader/compiler
workers. These snapshots do not measure host-memory peaks. No total-memory
saving is established.

All 16 TPS rank receipts report effective epoch size 51,200, length 6,400 and
shared `next_epoch` 0 before training, 1 before timing and 1 after training.
Startup sample-order records all identify epoch 0. Native epoch advancement is
monotonic, and the consumer alone can start another epoch; these populated
receipts establish no rollover during updates 1–60. Before timing, every V2
rank has an actual capacity-32 iterator, 32 ready batches and a live
`streaming-batch-prefetch` producer. A has no prefetch iterator. The complete
resolved A/B configuration diff is exactly `dataloader.num_prefetch_batches`
and operational `dump_folder`; source/config hashes are unchanged across runs.

This **single pair remains provisional**. There is no identical-control repeat,
run-to-run noise estimate, confidence interval or established causal explanation
for the full TPS difference. It supports neither general speedup nor production
adoption. The default remains zero. The earlier 2.08% TPS regression below is
preserved as a separate short-epoch experiment. The cause of that earlier
candidate update-16 outlier remains unassigned.

### Bounded traces and mechanism

The [new capture receipt](trace-captures-long-epoch-001.json) verifies all four
native update-21 traces after 20 warmup updates. Only ranks 0/1 constructed
profilers; shapes, stacks and profiler memory were off. Each trace contains
input preparation, global valid-token reduction and exactly three complete
CPU/GPU forward/backward microbatches. No fourth microbatch or optimizer is
captured. All eight ranks completed 21 full updates, with accumulation 32 and
one optimizer step in the target update; the remaining 29 microbatches ran
normally outside capture. No TPS is calculated from these diagnostic runs.

| Variant / rank | Canonical gzip trace | SHA-256 |
| --- | --- | --- |
| A / 0 | [rank0](artifacts/raw/v2-long-epoch-baseline-trace-001/native/profiling/traces/iteration_21/rank0_trace.json.gz) | `5b2a23c893e59d7317dca46468b9d8416435245ebbcb865dfa3f81344c760576` |
| A / 1 | [rank1](artifacts/raw/v2-long-epoch-baseline-trace-001/native/profiling/traces/iteration_21/rank1_trace.json.gz) | `1a09edab1e9bbdbd55deea3ad8e5280c3e523e0fa60b3474ed02f88ef9783ae5` |
| B / 0 | [rank0](artifacts/raw/v2-long-epoch-candidate-trace-001/native/profiling/traces/iteration_21/rank0_trace.json.gz) | `196806cee1fe857e97e4d80f3b7aa5961e323860f5fa1274d3e4b59a6f522c94` |
| B / 1 | [rank1](artifacts/raw/v2-long-epoch-candidate-trace-001/native/profiling/traces/iteration_21/rank1_trace.json.gz) | `6a8bcd4624f06459656db535618ddc2ae0c21f6167e2d74f5276c044de08cfc7` |

Deterministic gzip round trips matched all original parser hashes. Capture and
analysis receipts were persisted before redundant plain files were removed,
after every runner/parser consumer stopped. Both compressed and uncompressed
hashes, sidecar hashes, trace identities and full coverage counts are retained.

The [native JSON analysis](trace-analysis-long-epoch-001.json) reads every
canonical event: each trace has 8,795 kernels, 274 copies and 861 memsets. No
Perfetto import or analysis service was used for this follow-up; native JSON
avoids the importer event loss seen in the earlier experiment.

| Observed input prefix, ms | A rank 0 | A rank 1 | B rank 0 | B rank 1 |
| --- | ---: | ---: | ---: | ---: |
| Input start to first F/B | 675.782 | 677.895 | 8.446 | 8.994 |
| Host loader-next total, 32 calls | 645.431 | 637.817 | — | — |
| Host queue-pop total, 32 calls | — | — | 0.564 | 0.678 |
| Global token-reduction host span | 15.765 | 28.024 | 0.396 | 0.603 |
| Recorded local-device busy union during input | 15.342 | 27.630 | 0.026 | 0.433 |
| Local-GPU event-free time during input | 660.440 | 650.265 | 8.420 | 8.561 |

Both V2 traces begin with 32 ready batches. Readiness remains positive throughout
all 32 successful pops (minimum 5 / 4); the producer is live in every observation.
No trace contains a consumer epoch-entry span. The baseline's observed loader
calls and V2's observed queue-pop spans have zero overlap with recorded local
device activity. The ready queue supports input being prepared earlier, but
previous-update producer/worker activity is outside capture, so its CPU cost and
direct CPU/GPU overlap cannot be quantified here.

For the analyzed region from input start through the third CPU F/B return,
before the final CUDA drain, integrated host spans are
1,943.212 / 1,942.405 ms for A and 1,272.129 / 1,272.802 ms for B. Local-device
busy unions are 1,237.022 / 1,250.517 versus 1,220.919 / 1,223.282 ms. Remaining
kernel unions after explicitly named NCCL/EP families are 987.359 / 948.509
versus 990.895 / 945.546 ms; these may retain unclassified wait kernels and are
not a pure useful-compute measure. Explicit NCCL unions are 234.250 / 263.052
versus 209.975 / 245.400 ms; EP unions are 157.963 / 190.135 versus 157.627 /
189.657 ms. Copy/memset unions are 24.696 / 24.207 versus 24.436 / 24.448 ms.
Communication includes peer waits and overlaps other categories.

Host launch API unions are 57.836 / 48.288 versus 47.917 / 49.189 ms. Host
synchronization API unions are 676.643 / 752.822 versus 717.926 / 728.165 ms;
these can overlap device execution and are not device-idle budgets. Host spans
include waiting/scheduling, not measured CPU utilization. All device values use
interval unions clipped to the named region; overlapping scopes must not be
added. The full capture also contains the third microbatch's device tail through
the final drain; that tail is outside these CPU-return-clipped metrics. Worker
stacks and preceding producer work are unavailable.

These captures support ready input and substantially less exposed input wait
without rollover. They do not causally assign the separate 2.12% TPS difference,
establish steady-state variance, or provide full-step/device-compute speedups.
Only two ranks and the first three microbatches are visible; profiling and peer
waiting can perturb the prefix. Convergence, numerical candidate acceptance and
production-scale behavior remain outside this assignment.

### Failures, release and stop decision

The affected test/runner receipts retain corrected import/default-serialization
failures, the wrong-runtime/path attempts and the five remaining unchanged
cache-fixture failures. The new pair receipt also records an oversized inline
receipt-write rejection, corrected by the file-based extractor. Three execution
validator assumptions were corrected: multiple startup logs can all identify
epoch 0, profile-only runs have no timed-window boundary snapshot, and final
trace retention uses verified gzip files instead of plain paths. None of these
is a failed native measurement. All four native runs exited successfully;
there was no failed/incomplete TPS control or trace capture.

The [execution receipt](artifacts/raw/long-epoch-001-execution.json) retains
literal commands, process identities, per-run receipts and final release.
Owned runner and torchrun groups exited; all eight GPUs had zero reported used
memory and no compute applications. Allocation infrastructure and old analysis
services were preserved. Task-local logs, caches, MDS and compact evidence remain;
redundant plain traces and exact run-local temporary directories were removed.

The finite follow-up is complete. No repeat was needed to report this provisional
pair; resolving repeatability or making an adoption decision would require a
separate comparison. No installation, download or new allocation was performed.
Default depth remains zero and no further execution follows. Publication is now
authorized by the subsequent user request noted above.

The following sections retain the **original short-epoch experiment**. Their
20-update TPS commands, update-11 traces, execution hashes and process-release
IDs refer to that assignment; the long-epoch receipts above are separate.

## Question and fixed contract

Does a bounded queue of 32 materialized CPU microbatches reduce integrated input
wait and improve TPS above the current monitoring-free baseline while preserving
FIFO contents, epoch/RNG behavior, committed scalar state and loader lifecycle?
Default depth is zero. Consumer epoch entry, 32-microbatch collection, callbacks,
valid-token normalization and device transfers remain unchanged. GPU prefetch,
normalization changes and other optimization candidates are out of scope.

Starting source is `3b6b8928`; previous execution baseline is
`16b13c2548317945712ded2e66f136b169130d95`; preserved V2 source is
`21c1891dda531a1222e6c350802f2765758794f9`. The
[prior source/CPU review](../overnight-streaming-perf/v2-prefetch-review.md) and
[finite V2 contract](../overnight-streaming-perf/reference-coverage.md#v2-lifetime-and-state-findings)
are reused and reconciled with the integrated source and new executed evidence.

Before any training-source edit, the requested `git pull --ff-only origin main`
completed with `Already up to date` at
`3b6b8928ec4b15a529d07ee99e672f2fc75ef73a`. Repository rules are current with that
baseline. Installed personal skill directories have no Git repository to pull.
The production diff is restricted to V2 prefetch and its required lifecycle,
configuration and tests; trace-window support belongs in the task runner.
Each new rank trace is limited to three complete microbatches during collection.

The [source manifest](source-manifest.json) records all 18 changed/new training
source, test and golden paths and their hashes. The complete scoped patch is
`artifacts/raw/source/v2-source.patch`, SHA256
`f00931dce406cad4071e30987c212797d61792801ae012e5471ca2acbfc34e2a`.
The canonical experiment/model/distributed files are unchanged; `train_step`
is unchanged. The loader's public default is `num_prefetch_batches: 0`; only
candidate run overrides select 32.
The implementation is in the [CPU FIFO](../../gift/torchtitan/datasets/streaming/core/prefetch.py)
and [streaming loader](../../gift/torchtitan/datasets/streaming/core/dataloader.py),
with the required wrapper/trainer lifecycle changes listed in the manifest.

## Evidence ledger

| Gate | State | Evidence / remaining question |
| --- | --- | --- |
| Current source and runtime reconciliation | Passed | Latest origin/main is `3b6b8928`; all existing V2-touched runtime/test files matched the donor parent. Later baseline runner repairs are retained. Direct master execution and all four run-local preflights verified the authorized runtime and eight idle H100s |
| Required source port and CPU/config checks | Passed | 245 affected CPU/config checks in 149.67 s, no deselections; controlled worker replay passed; scoped Ruff and whitespace checks passed. Original interrupted control remains recorded |
| Real eight-owner Gloo/DCP loader checks | Passed | [Gate receipt](loader-gate.json): both persistence modes, actual skewed epoch entry, 58-batch OFF/ON hashes, 896-row coverage and retained decoded scalar DCP proofs |
| Independent final integration/cost review | Passed | [Review](integration-review.md): no demonstrated implementation blocker; CPU timeout disposition and probe-count wording resolved. Gates that were open at review time are reconciled with the executed evidence here |
| Native forward/backward/optimizer | Passed | Both GIFT variants completed 20 full updates on all eight ranks with finite logging and clean exit |
| Fresh matched unprofiled TPS/memory | Passed as a provisional pair | 10 warmup + 10 measured updates each; depth32 input TPS was 2.08% lower; no general performance gain is established |
| Bounded baseline/candidate traces | Passed | Four native rank traces, actual three-microbatch collection, all-rank full updates, lossless gzip and reconciled Perfetto MCP imports |
| Artifact checks and process/GPU release | Passed | Four source/config completion receipts, verified canonical gzip hashes, stopped training groups and eight idle GPUs; final maintained-file checks are recorded in compact results |

Failed and skipped checks remain recorded here or in linked compact results.
Historical TPS and trace-derived overlap budgets do not establish a V2 gain.

## Runtime and integration preflight

The current process already runs on the authorized one-node master
`lm-mpi-job-8dff1e19-674b-42ed-96a9-bcd042dfefe5-mpimaster-0`. Direct execution is
the established route; the historical `sr008-gpu` SSH alias does not resolve here.
No new host or allocation is needed. The master host file lists one node and no
worker node. Eight H100 80GB devices had zero reported used memory/utilization and
no compute applications at preflight. The allocation's PID 1 `mpirun` sleep
wrapper remains untouched. Existing task Perfetto services are analysis tools.
The checkout is on the mounted NFS4 volume.

The established interpreter is `/opt/torchtitan-venv/bin/python3.12`, Python
3.12.3. Installed metadata matches Torch `2.15.0.dev20260822+cu130`, DeepGEMM
`2.3.0+f093a4c`, DeepEP `1.2.1+1b8f467`, Transformers `4.57.1` and Tokenizers
`0.22.2`; native receipts report NCCL `2.30.7` and CUTLASS DSL `4.4.1`.
The existing launcher sources `/etc/profile.d/wm.sh`, uses task-local
source/scratch/cache paths and owns one bounded torchrun process group; its
cleanup targets that group. Actual import/source hashes and resolved settings
remain run-local checks. No dependencies or shared environment are modified.

The fixed input is the existing 896-row MDS fixture at
`runs/overnight-streaming-perf/artifacts/raw/mds4096`; the
[data receipt](../overnight-streaming-perf/mds-data.json) owns its packing,
tokenizer and shard hashes. New launches reuse those bytes. Native source
snapshots before/after A and B match at
`ac22aae7ced85f3970b89e420010d87076b1ed0fa58b162b3aa245215aae35b7`
(tracked and untracked GIFT bytes). Config and runner hashes also remain fixed.
These tree-snapshot hashes differ by definition from the scoped patch hash.

The initial source assessment found all required V2 runtime/test files identical
to `21c1891d`'s parent. Therefore only its scoped delta is ported; current
monitoring-free YAML semantics, runner fixes and configuration goldens are
preserved. The old registry deselection is obsolete on the latest baseline and
will not be carried forward. Historical helper defects (stale source pin,
process-PID comparisons, local/global edge counts and missing decoded DCP scalar
proof) are repaired in a new task-local helper, preserving historical bytes.

## Measurement scopes fixed before launch

Each full optimizer update has 32 microbatches per rank at B2/S4096, across eight
ranks: 2,097,152 physical input tokens and 2,096,640 valid main targets. The ten
measured updates therefore contain 20,971,520 input tokens and 20,966,400 valid
targets. Valid-target TPS uses the native supervised-token counter and remains
separate from physical input TPS. Per-GPU TPS is the complete-window global value
divided by eight, rather than an average of rounded console TPS values.

Retain the established timer: boundary CUDA drains and the complete eight-rank
window from earliest rank start to latest rank end on this verified single host.
Normal input loading, transfers, compute, communication, optimizer and lightweight
logging are included; initialization, cold compilation, warmup and teardown are
excluded. Per-update host cadence is a separate observation. Sampled allocator
peaks/native composite memory are labeled by lifecycle and never presented as
simultaneous device-wide memory maxima. Diagnostic trace runs establish mechanism
and capture coverage, not TPS magnitude.

Update 11 begins at local microbatch offset `10*32 mod 56 = 40`, so its 32-batch
fetch crosses the epoch after 16 old-epoch deliveries. Enabled producer queues
cannot cross that epoch entry ahead of the consumer. Epoch-entry spans must be
reported separately from ordinary queue-pop/input work.

## Executed runner verification

The [runner receipt](runner-verification.json) records exact commands, code hashes,
logs and retained harness failures. The normal eight-rank Linear/NCCL diagnostic
ran two updates per rank (64 forward/backward calls and two optimizer calls each).
On the target update all ranks completed 32 microbatches and one optimizer call;
only ranks 0/1 constructed profilers, each collecting the prefix/token reduction
and exactly three complete CPU/GPU microbatch scopes. A diagnostic failure after
the fourth target forward/backward preserved the two already-stopped exports,
without duplicate export; all ranks destroyed NCCL and GPUs returned idle. These
are adapter lifecycle checks, not crop TPS or numerical evidence.

The collector closes its prefix annotation, drains CUDA after the third complete
forward/backward, exits the native profiler context, and clears its handle so
the remaining training and native cleanup cannot resume or double-stop it.
Shapes, stacks and profiler memory are off. This uses the established native
context lifecycle described in [PyTorch profiler documentation](https://docs.pytorch.org/docs/2.15/profiler.html),
checked against the installed implementation. No export trimming substitutes for
bounded collection. Diagnostic queue/epoch/token spans are enabled only in the
bounded profile mode. Timed runs retain the original two boundary drains and
have no new diagnostic per-microbatch wrappers or synchronization. Host process
tree RSS/HWM reads occur before the start timestamp and after the end timestamp;
RSS sums can double count shared pages, and HWM is process-lifetime memory.

## Preserved harness failures and disposition

The initial source CPU suite was interrupted after 250.58 s and six passes.
A targeted 45 s faulthandler capture showed the main thread waiting for worker
data in the **depth-0 control**, before the candidate iterator/producer started.
That shell had OMP/MKL unset, Torch 96 intra-op threads, and a 104-byte TMPDIR.
The queue wait and fork warning do not establish which condition caused the
stall. Both logs remain in `artifacts/raw/cpu-checks/`.

The exact worker case then passed in 9.93 s with OMP/MKL/OpenBLAS set to one,
Torch intra-op one and inter-op 96, and a short 70-byte task-local TMPDIR. The
same test proceeds through creating/consuming the expected default loader while
the V2 producer is live, so that validation-like worker-creation interaction is
covered in the intended harness. No source, multiprocessing start method, or
test-order workaround was introduced. The complete affected CPU/config suite
then passed **245 tests in 149.67 s**, with 90 warnings and no deselections. It
covered the V2 helper/text/checkpoint/replication/trainer suites, DataWorld,
torch-free configuration, crop configuration and golden snapshots. Scoped Ruff
`F821,F811,F841` and `git diff --check` passed. No source change was needed after
the selective port, and owned test processes exited. Exact commands and logs
remain under `artifacts/raw/cpu-checks/` and the
[CPU verification receipt](artifacts/raw/cpu-checks/verification.md).

The first real-loader helper attempt at
`artifacts/raw/loader-check-20261113T130245Z-4063b6ed/` failed its preliminary
world-one probe with an observed `OSError: AF_UNIX path too long` in tensor IPC.
It used an excessively deep temporary directory. Its failed logs are retained;
the helper reuses short task-local `.tmp/run-*` paths before rerunning. This
observed helper defect is not retroactively assigned as the CPU pytest cause.

The first successful distributed receipt at
`artifacts/raw/loader-check-20261113T130946Z-ae570e7c/` establishes content/DCP
behavior within its recorded checks. Parent review found its explicit boundary
barrier delayed peers before actual epoch entry, so that receipt is superseded
for the skewed-rollover claim. The final targeted two-case extension records
peer epoch-entry calls while rank 0 holds a queued old-epoch item paused, and
persists complete paired OFF/ON hashes through rollover. No training-source
change is involved.

The corrected gate is `loader-check-20261113T132448Z-435f3793`; its
[compact receipt](loader-gate.json) links the full receipt (SHA256
`d279a102e9f44ac47e079cd5fb5af6ca476cb4646dd6299063b6b96f4ff3db93`),
commands, logs, per-rank hashes and checkpoint artifacts. Both eight-owner Gloo
cases use two workers and pass with persistent workers true and false. All 896
first-epoch rows occur exactly once globally, and expected row-edge counts match.
All rank-specific OFF/ON input, label, position and sequence hashes match through
58 delivered batches, including rollover.

At delivery 55, rank 0 holds the final queued old-epoch batch and committed cursor
55 inside a 1.030 s / 1.001 s pause. All seven peers start their actual epoch-1
`next()` call during that pause; those calls finish after release, without the
superseded pre-rollover barrier. All owners complete under the existing group
and launcher deadlines. The parent inspected the actual monotonic proof and
matched wrap hashes.

Installed DCP saves/loads the real Stateful loaders while source cursors are
ahead. CPU decoding verifies the actual complete scalar tree under
`dataloader.dp_rank_0`, including epoch, sample cursor, canonical/physical node
counts and unchanged Mosaic shuffle seed 9176. Delivery 12 is epoch 0 / sample 192;
chained delivery 58 is epoch 1 / sample 32. Same-instance older-state replay, exact-end delivery 56,
fresh chained restore and stale-iterator invalidation pass. Six tiny checkpoints,
48 rank shards and six decoded files remain under this run's checkpoint directory
with hashes. No model/optimizer/EMA checkpoint was created. Both owned launcher
groups exited and their short temporary directories were removed.

## Fresh matched native TPS pair

Both unprofiled runs passed all 20 optimizer updates on all eight ranks, with
monitoring and profiling off. The source, canonical workload and runner stayed
fixed. Only the typed prefetch depth and run-local output/cache paths differ.
The [pair receipt](pair-001.json) owns exact values, all-rank step series,
resolved-config comparison, host-memory boundaries, finite logging and commands.

| Observation | A: depth 0 | B: depth 32 |
| --- | ---: | ---: |
| Complete ten-update window, s | 142.273518046 | 145.299132130 |
| Global physical input tokens/s | 147,402.835665 | 144,333.415435 |
| Per-GPU physical input tokens/s | 18,425.354458 | 18,041.676929 |
| Global valid-target tokens/s | 147,366.848644 | 144,298.177784 |
| Per-GPU valid-target tokens/s | 18,420.856081 | 18,037.272223 |
| Rank-0 update cadence median, s | 14.360208416 | 14.615665911 |
| Rank-0 cadence range, s | 13.538051577–14.982518644 | 13.304187755–17.625949414 |
| Maximum sampled allocated/active peak, GiB | 69.674641 | 69.664616 |
| Maximum sampled reserved peak, GiB | 70.162109 | 70.181641 |
| Allocator retry / OOM deltas, every rank | 0 / 0 | 0 / 0 |

Depth 32 delivered **2.08% lower input TPS in this pair**. This is a provisional
single A/B observation: there is no repeated-pair or identical-control noise
estimate. Candidate update 16 took 17.626 s and remains in the complete window;
no outlier is removed. Step cadence is a host-boundary observation, not a new
per-update synchronized timer. No general slowdown or speedup is established,
and the public default remains zero.

Both variants cross the same finite-epoch boundary during update 16. The
targeted existing-log review found no compilation marker after startup, but
compiler observation and the DeepGEMM guard were disabled in these timing runs.
Unlogged JIT work cannot be ruled out, and the outlier's cause is unassigned;
there is no observed continuing compilation/settling that justifies changing the
fixed warmup window. The complete outlier and log disposition are in the pair
receipt.

Host RSS was sampled just before the start timestamp and just after the end
timestamp, with no sample inside the measured loop:

| Host observation, GiB | A: before / after | B: before / after |
| --- | ---: | ---: |
| Maximum one-rank self + live-descendant RSS | 6.913467 / 6.924465 | 6.917622 / 6.928638 |
| Sum of eight rank-tree RSS observations | 54.165245 / 54.253403 | 54.219475 / 54.307858 |
| Maximum rank self process-lifetime HWM | 3.452332 / 3.463322 | 3.454018 / 3.465027 |

The trees include loader/compiler descendants. Shared pages can be counted more
than once; these are boundary observations, not unique physical memory or
window peaks. GPU allocator peaks in the table are sampled lifecycle peaks,
not simultaneous device-wide maxima. No total-memory saving is established.

Both runs completed real forward/backward and optimizer work. Rank-0 first-step
loss/gradient norm were 15.319454 / 9.535783 for A and 15.319469 / 9.538402 for B;
these finite observations do not claim numerical parity. Cold first-step times
of about 265 s / 318 s are excluded by the ten-update warmup. All diagnostic Gloo
groups stopped about 52 seconds before A's measured update 11; no Gloo diagnostic
ran in either timed window. The owning runner groups exited before trace capture,
and all eight GPUs were idle at that handoff.

All 20 entries of each retained rank-zero total-loss, global-max-loss and global
gradient-norm series are finite. The full decoded rank-zero configuration diff
contains only `dataloader.num_prefetch_batches` (0 to 32) and `dump_folder`.
Resolved-config SHA256 values are
`9ccf9287b930f4a8c65258ce05df5e367a740bf47196d97e4c18032d2084381d` (A) and
`b3a1487c0ff25c792d38d7cbd566fd74ccd4b131b1c6ff6fbdad0a4dbcbc278b` (B).

An earlier outer shell-background launch disappeared before creating an attempt
or starting GPU work. Its empty log/process outcome is preserved in the pair
receipt; it is not a failed native training run. The surviving task-owned
controllers completed without replacement launches.

## Bounded workload traces

Both native captures passed. All eight ranks trained eleven full updates with
32 microbatches per update; only global ranks 0/1 constructed profilers for
update 11 after ten warmup updates. Each trace contains the input prefix, global
token-count reduction and exactly three complete forward/backward microbatches,
with three CPU and GPU projections. There is no fourth microbatch or optimizer
marker in the capture. All eight target-update sidecars record 32 microbatch
calls and one optimizer call, including the remaining 29 outside capture.
Shapes, stacks and profiler memory are off.

The [capture receipt](trace-captures.json) owns literal commands, parsed coverage,
per-rank training/profiler counts, all four canonical gzip and uncompressed
SHA256 values, resolved config identities, data manifest and runtime. The source
snapshot matches the TPS pair. Both trace runs retain the frozen canonical
workload, differing only in prefetch depth and run-local paths.

| Variant / rank | Canonical trace | Gzip SHA256 |
| --- | --- | --- |
| Depth 0 / 0 | [rank0](artifacts/raw/v2-baseline-trace-001/native/profiling/traces/iteration_11/rank0_trace.json.gz) | `b4e4ff33b2829b7e43de7a6194434bcffe7e0856f610e3cb301040d72dbfb3e6` |
| Depth 0 / 1 | [rank1](artifacts/raw/v2-baseline-trace-001/native/profiling/traces/iteration_11/rank1_trace.json.gz) | `bea3907f583f7f5dd53a3e6c8b1e922315da41ab9f69bcf426784994bc2c1bb0` |
| Depth 32 / 0 | [rank0](artifacts/raw/v2-candidate-trace-001/native/profiling/traces/iteration_11/rank0_trace.json.gz) | `8e4399a6397a0b1c9b7761e4737b878605b05494eb4cbfa5b22137cfaae2f109` |
| Depth 32 / 1 | [rank1](artifacts/raw/v2-candidate-trace-001/native/profiling/traces/iteration_11/rank1_trace.json.gz) | `e65491fb54b5a707fab354b5cb394f4b66b1e71c67f749c44b7d96dfea929881` |

Deterministic gzip round trips matched the original parser hashes before plain
JSON removal. Trace runner groups 561276 and 584250 stopped; all eight GPUs had
zero reported used memory/utilization and no compute applications at release.
Actual Perfetto MCP inspection is complete; no TPS is derived from these
diagnostic captures.

## Perfetto mechanism findings

The [analysis receipt](trace-analysis.json) retains actual MCP trace handles,
SQL/results, canonical/derived hashes, importer census, launch-flow checks and
observability limits. Raw imports lost 50/45/49/46 overlapping kernel events.
The existing display-lane normalization helper recovered all **8,795 kernels,
274 copies and 861 memsets per trace**, with complete host-launch linkage and
zero final importer errors. It preserves retained timestamps, durations, names,
arguments, actual stream IDs, CPU events and metadata; canonical gzip bytes are
unchanged. Derived display lanes are an import repair, not a scheduling change.

| Observed prefix, ms | A rank 0 | A rank 1 | B rank 0 | B rank 1 |
| --- | ---: | ---: | ---: | ---: |
| Input start to first forward/backward | 1310.574 | 1311.677 | 1138.004 | 1137.747 |
| Largest local-GPU event-free input gap | 1225.698 | 1286.669 | 1026.021 | 1109.393 |
| First token-count NCCL kernel | 83.538 | 24.028 | 110.960 | 27.355 |
| Candidate synchronous consumer epoch entry | — | — | 953.096 | 886.272 |

Candidate sidecars prove activation beyond the requested depth: the first
16 old-epoch ready counts decrease from 16 to 1 at capacity 32. Their pops total
only 0.169/0.180 ms on ranks 0/1. The finished old-epoch producer is valid while
those ready items remain. The new epoch records a live
`streaming-batch-prefetch` producer; its 16 successful pops take
64.059/214.818 ms. There are 33 queue-pop spans because one old-epoch exhaustion
attempt is included, while successful deliveries remain 32.

The candidate's raw first-new-epoch DataLoader call takes 952.390/885.158 ms,
inside the 953.096/886.272 ms consumer epoch spans. All observed loader,
queue-pop and epoch spans have zero overlap with local recorded device work.
The baseline has 33 DataLoader calls totaling 1212.514/1275.038 ms. Ordinal 17
is old-epoch exhaustion (0.044/0.050 ms), and ordinal 18 is the first new-epoch
read and the largest call on both ranks (1033.365/1165.211 ms). The remaining
ordinary-call sums, including exhaustion, are 179.148/109.826 ms. Unannotated
`__iter__` and barrier costs cannot be isolated. Queue capacity cannot hide the
synchronous entry into the next finite epoch.

These two profiled prefixes reach first forward/backward about 173 ms sooner
with V2. This does not establish an end-to-end gain or explain the separate TPS
regression. NCCL durations include waiting for peers; device busy is the union
of recorded kernel/copy/memset intervals on one rank, including wait kernels.
Overlapping category sums are not elapsed time or useful-compute budgets.
Only ranks 0/1 and the first three microbatches are observed. No stacks, worker
traces or complete producer activity are available, so internal barrier, I/O,
worker, scheduling and GIL attribution remains unresolved. No convergence or
whole-model numerical-parity claim follows from these captures.

## Commands and retained evidence

Run from `/workspace-SR008.nfs2/vkropotin/optimization-task` with
`/opt/torchtitan-venv/bin/python3.12 runs/overnight-streaming-perf/run.py`.
All four runs use the explicit output root
`/workspace-SR008.nfs2/vkropotin/optimization-task/runs/v2-prefetch-baseline/artifacts/raw`.

| `--run-id` | `--num-prefetch-batches` | Remaining flags |
| --- | ---: | --- |
| `v2-baseline-perf-001` | 0 | `--steps 20 --warmup 10 --window 10` |
| `v2-candidate-perf-001` | 32 | `--steps 20 --warmup 10 --window 10` |
| `v2-baseline-trace-001` | 0 | `--steps 11 --warmup 10 --window 0 --profile-update 11 --profile-microbatches 3` |
| `v2-candidate-trace-001` | 32 | `--steps 11 --warmup 10 --window 0 --profile-update 11 --profile-microbatches 3` |

The pair receipt retains literal runner commands and actual native child argv;
each attempt also retains `metadata.json`, `source-final.json`, `completion.json`,
`process.json`, `summary.json`, resolved YAML, native logs and per-rank sidecars.
Controller logs are under `artifacts/raw/controller-logs/`. CPU and Gloo commands
and logs are linked by their gate receipts. All generated artifacts remain
task-local and outside the training-source tree; large raw outputs are ignored
by Git. At the measurement handoff, commit/push had not been authorized;
the subsequent user request above now covers publication.

## Completion and limits

Required implementation, source/CPU review, real loader/DCP execution, native
training, TPS/memory comparison and bounded trace inspection are complete.
Training groups 513691, 538757, 561276 and 584250 exited; the Gloo owners also
exited. At final training release all eight GPUs had no compute applications and
zero reported used memory/utilization. Allocation infrastructure and existing
analysis services were preserved. Short run-local temporary directories were
removed; the existing MDS data, tiny loader checkpoints, logs and verified gzip
artifacts are retained.

V2 is integrated default-off and reviewable without a commit. This single pair
does not support adopting depth 32 for a performance gain. Repeatability, other
data mixtures/topologies, convergence and long-run stability are not established
by this finite assignment. No further experiment follows; publication is
covered by the subsequent user request above.
