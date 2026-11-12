# CUDA Graphs: findings and decision

Keep CUDA Graphs off in the default [100B crop](baseline.md). Historical work
shows integration complexity and no accepted speedup. It does not show universal
incompatibility: persistent loss-only replay worked.

## What was tested

The historical workload was a different 3B MoE on eight H100s: HybridEP EP8,
dense FSDP8, dense+sparse FP8, model/loss compile, selective AC, local/global
batch 8/256, sequence 4096 and four microbatches per update. These results are
not runtime or performance validation of the new crop.

| Path / observation | Established result | Interpretation |
| --- | --- | --- |
| Broad compiler-region capture | Failures involving graph-owned output lifetime, external workspace ownership, pool invariants and generated backward code | A global switch was not a reliable integration route on that tested stack |
| Dense/loss A007 | 25 updates completed; dense backward recorded on 96/96 calls per rank in the reviewed window | Actual graph launches existed, but stable dense reuse was not established |
| Loss-only A008 | 25 updates completed on all ranks; persistent CE forward/backward replay in updates 21–23, with no new capture/instantiate | Working narrow mechanism; not full head/model/optimizer capture and not a speed result |
| Loss input staging | 25 GiB BF16 logits plus 1 MiB labels copied per rank/update; about 17.605–17.641 ms of copy activity in the reviewed observations | Replay adds input-staging traffic; cumulative bytes are not simultaneous memory residency, and activity time is not an integrated slowdown |
| Matched performance attempts | No accepted repeated performance/noise comparison; a quiet graph-on run had incomplete persisted events, and a later graph-off control failed before its window completed | Do not interpret these as a measured graph speedup, reliable slowdown, or proof that graphs caused the control failure |
| Later graph-off trace | Approximately 99% of main-stream launches were already queued before previous GPU work ended | Little evidence of a large exposed CPU launch bottleneck in that captured workload |

The full campaign subsequently completed bounded graph-off stability checks;
that outcome does not close the earlier graph performance or numerical gates.
The [archive locator](../references/README.md) preserves the exact original
runtime conditions, source snapshots, failures and verification history.

## What the trace supports

In the later update-111 trace, rank 0's device activity span was 3,106.797 ms.
13,268 of 13,412 main-stream kernels (98.9%) were submitted before preceding
GPU work ended. Total device idle was 135.094 ms; a 76.912 ms initial interval
coincided with input loading. The remaining idle was 58.182 ms (1.87%).
Communication-only activity accounted for 728.715 ms; recoverable overlap is
unknown. Kernel-duration sums and host synchronization durations overlap and
must not be added to that elapsed partition.

These are observations from one profiled update on two of eight ranks. The
1.87% is neither predicted speedup nor a strict upper bound. Graphs can affect
device scheduling as well as launch overhead; profile perturbation and unseen
peers limit inference. Read the preserved [trace assessment](../runs/cuda-graphs-evidence/trace-assessment.md)
for method, source hashes and exact definitions.

## Decision and reopening criteria

Prioritize a representative crop performance control and critical-path analysis.
Do not invest in broader graph capture merely because many kernels are small.
Reopen only if current evidence identifies material exposed host-submission cost
or another plausible graph benefit, with a supported capture boundary.

A future test must establish actual repeated replay, finite loss/gradients,
required numerical/state checks, and repeated unprofiled A/A plus alternating
A/B windows with paired memory. Count fresh captures, input copies and graph-pool
retention. Preserve baseline FP8, SAC, MTP, monitoring and topology; do not hide
capture failures by removing required workload features. Use
[the validation protocol](../planning/validation.md).

[Compact evidence](../runs/cuda-graphs-evidence/README.md) is retained locally;
full historical records are recoverable through Git. No new graph experiment
was run during the repository simplification.
