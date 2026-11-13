# Input preparation: matched native trace analysis

The explicit input operation reduces the measured local preparation region on
both ranks and all three sampled microbatches. It does not establish an overall
training benefit. The full three-microbatch region is slightly longer in the
candidate, and the [unprofiled pair](input-pair-001.json) is provisionally neutral
(+0.0046% input TPS). The approximately 6.2 ms difference across ten measured
updates cannot be causally assigned to this trace mechanism.

## Capture and import gates

Acceptance was bounded to matched receipts, lossless import, actual activation,
equivalent input boundaries, downstream embedding, and the three complete F/B
microbatches. All gates passed; event dependency identities remain unavailable.
No source, runner, configuration or instrumentation was changed for analysis.

Control `four-fusion-control-matched-baseline-trace-001` and candidate
`four-fusion-input-input-trace-001` both passed, stopped their owned process
groups, completed ten warmup updates and captured update 11. All eight ranks
executed 32 microbatches and one optimizer call. Only ranks 0/1 recorded exactly
three complete microbatches. Resolved configurations differ only in `dump_folder`:
B2, S4096, EP8, dense FSDP8, expert FSDP1, FP8 CuTe, compile, SAC, MTP1,
accumulation 32, prefetch 32, epoch 51200, seed 2048 and schedule 65000/2000 remain
matched. Monitoring, validation, checkpoint saving and CUDA Graphs stay disabled.

Both captures use source HEAD `33b336a592746340d327d86e34149a23dc8688ee`.
The candidate driver records the Trainer-only [activation patch](input-prepare.patch)
SHA256 `e0eea03e83e7ea88bdccd90b9c4a3927727753057df89a4843d072245602f7a0`
and its restoration. Source snapshots differ as expected and were unchanged
during each run. Runner, adapter, generated configuration and canonical
configuration hashes match between captures. Full source/config/runtime pins,
canonical and prepared trace hashes, exact SQL, results, slice IDs, bounds and
CUDA streams are in [machine-readable evidence](input-trace-analysis.json).

The existing `monitorfree-perfetto-prepare.py` helper, SHA256
`4402b99ea328542009304f7d87723b5ff46ad3cbb9c22536fda3079fe5abe597`,
removed GPU annotation projections and repaired overlapping display lanes.
Original JSON bytes remain untouched; canonical gzip copies are lossless.
Every device event retains its timestamp, duration, name and arguments,
including CUDA stream. Synthetic display tids are not CUDA streams.

Actual Perfetto MCP imports reconcile exactly with preparation receipts:

| Variant | Each rank: kernels / copies / memsets | Linked device events | Import errors |
| --- | ---: | ---: | ---: |
| Control | 8795 / 274 / 861 | 9930 / 9930 | 0 |
| Input | 8768 / 274 / 861 | 9903 / 9903 | 0 |

Category duration sums also match at source timestamp precision. Every event
has exactly one runtime/driver launch link. Ranks 0/1 control moved 36/53 device
events and flow endpoints; candidate moved 43/48. All four remove 354 projected
ranges. The reduction of 27 kernels per rank is nine kernels per microbatch.
Exact device dictionaries match before/after preparation except display tid.

Reusable prepared files are under `artifacts/raw/prepared-traces/` with names
`control-rank{0,1}.perfetto.json.gz` and `input-rank{0,1}.perfetto.json.gz`,
with adjacent preparation receipts. Original trace paths are pinned in JSON.
Trace handles are control 0 `trace_3de58236775b5631`, control 1
`trace_e072b049b15a928d`, input 0 `trace_415bc25a29955e37`, input 1
`trace_b30070e298dc1c5b`. Tool cache eviction can remove derived SQL tables;
recreate them from saved queries after reloading. One such failed baseline
follow-up query is retained; it does not invalidate already completed results.

## Equivalent local input region

The input boundary starts at the CPU microbatch marker and ends at the first
`GFSDP::root_pre_forward`. It includes position/mask preparation, local token
counting, aux share update and target packing. The enclosing downstream check
extends through the first `aten::embedding` launch and its actual GPU gather
completion, including embedding weight allgather and layer 0 forward prefetch.
CPU/device launch ownership, rather than temporal GPU overlap, assigns work.

Each candidate microbatch contains exactly one
`four_fusion::prepare_input_tensors_scalar` and one `_prepare_kernel`, linked
through `cuLaunchKernelEx`. The kernel runs 6.496–6.720 us on CUDA stream 7.
The retained aux-buffer DtoD copy also runs on stream 7. Baseline launches cover
arange, repeat copy, two label comparisons, cast, reduction, two divisions,
the aux DtoD copy, position cast and target concatenation. The fused operation
performs the masks, position generation, packing, count and share calculation;
it does not remove the aux-buffer copy or move prefetch/token reduction into
the operation. The retained copy is included in every preparation timing.

All values below are per microbatch; `C → I` means control → input candidate.
Preparation device sum equals interval union because its events do not overlap
on stream 7. Device span includes launch gaps and is distinct from that union.

| Rank / MB | Prep host elapsed ms C → I | Prep device sum/union us C → I | Prep device span us C → I | Entry→embedding GPU end ms C → I |
| --- | --- | --- | --- | --- |
| 0 / 1 | 0.629 → 0.362 | 19.360 → 7.808 | 368.801 → 39.104 | 3.731 → 3.417 |
| 0 / 2 | 0.748 → 0.517 | 19.968 → 8.224 | 432.257 → 55.904 | 3.913 → 3.704 |
| 0 / 3 | 0.858 → 0.545 | 20.736 → 7.840 | 478.913 → 54.272 | 4.001 → 3.821 |
| 1 / 1 | 0.701 → 0.375 | 18.945 → 7.840 | 405.793 → 42.400 | 3.967 → 3.809 |
| 1 / 2 | 0.736 → 0.534 | 20.673 → 7.744 | 437.761 → 62.016 | 3.901 → 3.742 |
| 1 / 3 | 0.850 → 0.589 | 19.647 → 7.936 | 491.361 → 65.440 | 4.001 → 3.802 |

Preparation uses 11 device launches in control and 2 in candidate. Summed host
launch API time is 76–106 us versus 20–52 us; this sum excludes Python and
allocation time. Host preparation elapsed falls 201–326 us. Entry→embedding
GPU completion falls 158–314 us, so the local benefit persists through a
common visible consumer rather than appearing only in a kernel histogram.

First microbatch follows the separate input prefix: prefetch 32 queue-pop host
sums are 0.748/0.706 ms in control ranks 0/1 and 0.687/0.610 ms in candidate.
Global-valid-token reduction host elapsed is 0.534/1.949 ms versus 0.682/0.908 ms.
These costs are outside the preparation boundary and are not fusion savings.

Microbatches 2/3 start 206/220 us after the previous control device tail on rank 0,
and 221/257 us on rank 1; candidate gaps are 216/230 us and 218/273 us. Thus their
preparation is not hidden behind the prior captured backward work. Within the
entry→model-root interval, all-device idle falls from 609–837 us to 354–581 us.
These idle values are complements of clipped interval unions over every device
event, not sums of per-stream gaps.

## Downstream waits and enclosing F/B guard

The first embedding weight allgather stays on stream 24; the first embedding
gather compute stays on stream 7. Allgather durations per microbatch are
rank 0 control[2.646,2.653,2.680] ms, candidate[2.647,2.715,2.822] ms;
rank 1 control[2.860,2.705,2.716] ms, candidate[3.024,2.667,2.672] ms.
Embedding launch-start→GPU-start delays remain 2.22–2.58 ms. The preparation
tail→embedding-start gap is 3.07–3.40 ms and mostly contains stream 24 activity;
its all-device idle portion is 0.242–0.380 ms. It must not be described as a
multi-millisecond exposed input stall.

Both variants retain eight `cudaStreamWaitEvent`, six `cudaEventRecord` and
five `cudaEventRecordWithFlags` calls before the first embedding finishes on
the host. Launch flow links and FSDP ordering support the allgather/consumer
relationship, but the capture lacks usable CUDA event identities and stacks
for a complete device-dependency reconstruction. Launch delay includes queued
work and dependencies; it is not a measured host wait or pure kernel overhead.

Full F/B guards include every device launch whose host timestamp falls within
each complete CPU microbatch marker, including autograd worker threads. Worker
launches are temporally assigned, not claimed to nest on the main CPU thread.
Microbatch elapsed below includes the final associated GPU tail:

| Rank / MB | CPU marker ms C → I | Enclosing with GPU tail ms C → I | Device union ms C → I |
| --- | --- | --- | --- |
| 0 / 1 | 386.861 → 387.600 | 411.988 → 412.623 | 398.268 → 399.345 |
| 0 / 2 | 384.921 → 384.915 | 414.338 → 414.717 | 400.693 → 401.393 |
| 0 / 3 | 385.899 → 387.727 | 415.617 → 417.229 | 401.774 → 403.786 |
| 1 / 1 | 387.418 → 390.555 | 412.234 → 413.024 | 398.038 → 397.399 |
| 1 / 2 | 384.782 → 385.795 | 414.342 → 414.738 | 400.202 → 400.250 |
| 1 / 3 | 386.466 → 388.396 | 415.578 → 417.169 | 401.445 → 402.749 |

| Rank | Entire 3 F/B CPU span ms C → I | Enclosing GPU-tail span ms C → I | Device sum ms C → I | Device union ms C → I |
| --- | --- | --- | --- | --- |
| 0 | 1212.651 → 1215.513 | 1242.370 → 1245.015 | 1358.096 → 1389.908 | 1200.735 → 1204.524 |
| 1 | 1213.519 → 1216.649 | 1242.632 → 1245.422 | 1381.308 → 1382.597 | 1199.684 → 1200.398 |

Device sums exceed unions because streams overlap. The full-region guard is
2.645/2.790 ms longer, despite the smaller input region. A single diagnostic
capture cannot establish whether this variation is repeatable or attributable
to a specific unchanged downstream operation. No full-update speedup is
inferred; the remaining 29 microbatches and optimizer are outside the trace.
Numerical correctness, convergence, repeatability and final retention belong
to their separate gates and owner decision.

## Verification and remaining limits

Actual MCP SQL verified import counts/durations/link multiplicity, coverage,
activation, enclosing bounds, launch ownership, streams, clipped unions and
previous-microbatch tails. Canonical→prepared device event dictionaries were
compared exactly except display tid. Matched resolved-config comparison,
JSON parse and scoped whitespace checks passed. Runtime source/config/runner
bytes were read-only. No additional GPU work or benchmark was performed.

This supports a local input preparation benefit for the fixed workload and
these six samples. It supports neither a general training speedup nor an
explanation of the neutral unprofiled TPS pair. Exact CUDA dependency attribution
and run-to-run magnitude remain unknown.
