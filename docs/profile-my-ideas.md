# Profile research: my ideas

Research origin: the repository owner's own investigation of the baseline profile.
Recorded: 13 November 2026.
Status: historical baseline-profile findings, followed by an owner-selected
MTP experiment and capacity-selection research ideas from later BF16-cache
profiling. A row-owner input-embedding design was selected on 14 November 2026;
it is a research proposal only, with no source implementation or benchmark.
The later
[fusion experiments](../runs/four-fusion/report.md) and
[BF16 reuse experiment](../runs/bf16-expert-cache/report.md) own implementation,
measurements and current conclusions. Earlier observations retain their
original trace scope; the MTP section records a separate, unimplemented and
unbenchmarked experiment, and the capacity ideas below remain unimplemented
research with no measured speedup.

## Finding and evidence scope

The first microbatch has a short GPU preparation prefix before the embedding
weight all-gather. Its eager PyTorch operations generate ten small kernels with
gaps between launches. CPU/GPU launch correlations show that host submission
limits this prefix. CPU core saturation is not established by this observation.

Evidence comes from Perfetto inspection of `monitorfree-profile-update11-001`,
optimizer update 11, rank 0, first CPU microbatch marker `slice[707]`. The execution
revision is `16b13c2548317945712ded2e66f136b169130d95`. The
[baseline report](../runs/overnight-streaming-perf/report.md#fresh-traces-and-cleanup)
owns configuration/runtime identities and canonical trace paths/hashes. The
inspected artifact is `rank0_trace.json.gz` under that run's
`native/profiling/traces/iteration_11/` directory. Times below use the trace UI's
relative clock. SQL IDs refer to this trace import.

The [baseline](baseline.md) has microbatch size 2, sequence length 4096 and 32
microbatches per optimizer update. Model and loss compilation are enabled.
Shapes below follow those settings and inspected source; input-shape and stack
recording were disabled in this trace.

## What the small kernels do

Attribution uses each kernel's CUDA correlation ID, its CPU launch, the enclosing
CPU operator and the corresponding source operation.

| GPU slice(s) | Operation and purpose | Source |
| --- | --- | --- |
| 714 | `arange`: generate int32 positions `[0, ..., 4095]` for the active causal-mask path | [Position generation](../gift/torchtitan/trainer.py#L584) |
| 728 | `repeat` / `copy_`: materialize positions for both samples as `[2, 4096]` | [Position replication](../gift/torchtitan/trainer.py#L586) |
| 751 | `ne`: create `token_mask = labels != IGNORE_INDEX` | [Token mask](../gift/torchtitan/trainer.py#L594) |
| 754, 763, 765 | Repeat the validity comparison, cast its Boolean result to int64 and sum valid targets for this microbatch | [Auxiliary-loss token count](../gift/torchtitan/trainer.py#L630) |
| 768, 771 | Two divisions: `valid_tokens / global_valid_tokens / self._splits` | [Auxiliary-loss scaling](../gift/torchtitan/components/aux_loss.py#L93) |
| 780 | Cast positions from int32 to the labels' int64 dtype | [MTP target packing](../gift/torchtitan/components/loss.py#L932) |
| 785 | `stack`, lowered to `cat` and a view: pack labels and positions into `[B, 2, S]` | [MTP target packing](../gift/torchtitan/components/loss.py#L932) |

The validity comparison is computed twice: once for the model's token mask and
once for the auxiliary-loss count. Auxiliary setup also copies the resulting
scale into its state through `state.scale.copy_(share)`; that copy is outside the
ten-kernel duration sum below.

## Why this prefix is limited by host submission

The first listed kernel starts at **1.010759288 s**; the last ends at
**1.011164346 s**. This spans **405.058 microseconds**, while the ten listed kernel
durations sum to **18.401 microseconds**. This sum is kernel execution time, not
the elapsed cost of the entire preparation path or a device-wide utilization
measurement.

A concrete launch correlation supports the host-submission diagnosis:

| Event | Relative time |
| --- | --- |
| GPU kernel 751 finishes | 1.010911769 s |
| CPU begins launching kernel 754 | 1.010999422 s |
| GPU kernel 754 starts | 1.011008953 s |
| GPU kernel 754 duration | 1.344 microseconds |

There are **87.653 microseconds** after kernel 751 finishes before the CPU even
submits kernel 754. The next kernel then starts 9.531 microseconds after launch
entry. This is evidence of delayed host submission, rather than a backlog of
these tiny GPU kernels. The trace does not isolate Python, PyTorch dispatch,
profiling and host scheduling costs from one another.

The conclusion applies to this prefix. The subsequent embedding and layer weight
all-gathers have their own dependencies; this observation does not establish that
the whole forward or optimizer update is CPU-bound.

## Existing compilation does not cover this region

- Model compilation wraps repeated blocks through
  [`segment.module.compile(...)`](../gift/torchtitan/distributed/compile.py#L107).
- Loss compilation wraps
  [`self.fn`, the cross-entropy function](../gift/torchtitan/components/loss.py#L162).
  It does not compile every method of the loss object.
- The trainer calls
  [input postprocessing, auxiliary scaling and `pack_targets`](../gift/torchtitan/trainer.py#L624)
  before the model forward. These calls remain eager, including MTP packing even
  though that method belongs to the loss object.

Enabling the existing `model` and `loss` compile components therefore does not
fuse the operations listed above.

## Fusion hypothesis

Fusion could reduce the separate kernel launches and associated host-submission
overhead in this preparation region. The current model/loss compile settings do
not cover it. No fused variant or speedup has been measured.

## Reuse BF16 expert weights across microbatches

### Observed baseline behavior

The same trace contains preparation of layer 2's routed expert weights on stream
24 while layer 1 is computing:

| GPU slice | Operation | Duration |
| --- | --- | --- |
| 1572 | Cast the local expert weights from FP32 into a BF16 input buffer | 588.002 microseconds |
| 1575 | Copy 603,979,776 bytes (576 MiB) from that BF16 input into the BF16 output buffer | 398.274 microseconds |

The baseline has `efsdp=1`: these operations prepare weights already resident on
the same GPU. Slice 1575 is a local device-to-device copy under the
`nccl:_all_gather_base` annotation.

The [BF16 input buffer is allocated per parameter group](../gift/torchtitan/distributed/gfsdp/_param_group.py#L499)
and remains allocated between microbatches. For each MoE layer, it contains all
experts owned by this rank. The
[BF16 output uses the shared working-buffer pool](../gift/torchtitan/distributed/gfsdp/_param_group.py#L418).
The current path is therefore:

```text
FP32 expert weights -> per-layer BF16 input -> shared BF16 output -> compute
```

After backward, [GFSDP clears the input-validity flag](../gift/torchtitan/distributed/gfsdp/_param_group.py#L880),
without releasing the BF16 input storage. The next microbatch repeats the
[FP32-to-BF16 cast](../gift/torchtitan/distributed/gfsdp/_collectives.py#L54).
Perfetto inspection found 32 such casts per MoE layer in this update. The
[optimizer runs after all 32 microbatches](../gift/torchtitan/trainer.py#L807),
so the expert weights do not change between them.

### Idea

Prepare the BF16 expert-weight representation once per optimizer update and reuse
it across the 32 microbatches, including backward and activation recomputation.
Refresh it when the underlying weights change, while keeping the FP32 master
weights.

For `efsdp=1`, compute could consume the persistent per-layer BF16 buffer directly
to also avoid the extra input-to-output copy. Simply skipping later copies with
the current arrangement would not work: other layers overwrite the shared output
buffer. This idea reuses BF16 storage already allocated by the current baseline;
repeated casting currently does not free that storage.

Follow-up, 14 November 2026: the opt-in implementation and focused GFSDP checks
pass. Rank-0/1 traces confirm one cast per expert group per update and removal
of the singleton copies. One timing candidate is 1.99% above historical V2;
causal speedup remains inconclusive without a contemporary control/noise estimate.
The [completed report](../runs/bf16-expert-cache/report.md) owns the evidence and
limits. The feature stays disabled by default. These crop results do not
establish behavior or performance of production PP4/EP32.

## Fuse MoE routing metadata preparation

### Observed baseline behavior

The owner's next finding concerns the small kernels after router top-k and before
FP8 token quantization, in the same first microbatch and rank-0 trace identified
above. GPU slice 2276 (`triton_poi_fused_scatter_zeros_like_8`) starts at
1.035268302 s. The capacity check then launches all-reduce slice 2285, followed
by small kernels and long gaps before slice 2370
(`_rowwise_1d_quantize_kernel`) starts at 1.035891728 s.

### What the all-reduce does

Slice 2285 sums the token-assignment counts used by the token-drop capacity
check. Its NCCL metadata records 128 `Long` (int64) elements, eight ranks
`[0, 1, 2, 3, 4, 5, 6, 7]` and a duration of 32.416 microseconds. The input
contains 1,024 bytes of counters.

The [baseline enables `drop_by_group_capacity`](../gift/configs/experiments/gift_100b_crop8_ep8_fp8_compile_cute.yaml#L19).
The [capacity-selection code](../gift/torchtitan/models/common/moe/dispatch/capacity.py#L124)
builds an expert-count histogram, all-reduces it, sums the global counts for each
destination group, and combines the local expert-overload and group-overload
conditions into `needs_drop`.

| GPU slice | Operation |
| --- | --- |
| 2285 | All-reduce the 128 expert counts across the eight EP ranks |
| 2286, `triton_per_fused_slice_sum_view_15` | Sum the global counts into eight destination-group loads, with 16 experts per group |
| 2287, `triton_per_fused_any_bitwise_and_gt_16` | Compute the final Boolean capacity predicate |

### The capacity predicate stops CPU submission

The [conditional capacity custom operator](../gift/torchtitan/models/common/moe/dispatch/capacity.py#L63)
contains `if needs_drop`. Evaluating this GPU Boolean on the CPU produces the
following trace events:

| Slice | Recorded event | Evidence |
| --- | --- | --- |
| CPU 2233 | `torchtitan::conditional_capacity_mask` | Encloses the predicate read and the subsequent clone |
| CPU 2234–2236 | `aten::is_nonzero`, `aten::item`, `aten::_local_scalar_dense` | Convert the device predicate into a host scalar |
| GPU 2288 | `Memcpy DtoH (Device -> Pinned)` | One-byte transfer; 2.208 microseconds |
| CPU 2239 | `cudaStreamSynchronize` | 1.032911660–1.035354507 s; 2.442847 milliseconds |
| CPU 2289 / GPU 2293 | `aten::clone` / device-to-device copy | The no-drop branch returns `valid.clone()` in this occurrence |

Most of the CPU synchronization overlaps the preceding GDN computation. Its full
2.442847 milliseconds must not be counted as additional GPU idle time. It stops
the CPU from submitting later work until the capacity predicate is available,
allowing the GPU queue to drain.

### Eager routing operations then leave the GPU waiting for CPU launches

CPU slice 2297 is `hybridep::dispatch`. Its
[custom-operator implementation](../gift/torchtitan/distributed/deepep/hybridep.py#L243)
calls [the routing-metadata helper](../gift/torchtitan/distributed/deepep/hybridep.py#L288)
before token quantization. The helper,
[`indices_to_map`](../gift/torchtitan/distributed/deepep/routing.py#L6), constructs
dense routing and probability matrices from the top-k expert IDs and scores:

```text
compare IDs -> arange -> add offsets -> mask invalid IDs
-> zero routing map -> scatter -> zero probabilities -> mask scores -> scatter
```

Although the surrounding model is compiled, these operations execute eagerly
inside the opaque custom operator. CPU events include individual `aten::ge`,
`aten::arange`, `aten::add`, `aten::masked_fill_`, `aten::zeros` and
`aten::scatter_` calls with separate CUDA launches. GPU slices 2300–2360
correlate with these launches; their CPU launches are enclosed by dispatch
slice 2297.

A concrete launch correlation establishes delayed host submission:

| Event | Relative time |
| --- | --- |
| GPU scatter 2360 finishes | 1.035803408 s |
| CPU launch 2369 begins submitting quantization kernel 2370 | 1.035883653 s |
| GPU quantization 2370 starts | 1.035891728 s |

The GPU finishes the scatter 80.245 microseconds before the CPU starts submitting
the quantization kernel. The trace does not separate Python, dispatcher,
profiling and host-scheduling overhead, or establish CPU core saturation.

From the start of slice 2286 to the start of slice 2370, Perfetto records:

| Measurement | Duration |
| --- | --- |
| Elapsed interval, 1.035335406–1.035891728 s | 556.322 microseconds |
| Execution of 14 GPU kernels | 27.711 microseconds |
| Execution of three GPU copies | 4.735 microseconds |
| Time without a recorded kernel, copy or memset | 523.876 microseconds (94.2%) |

All recorded GPU work in this interval runs on stream 7 without overlap;
no kernel, copy or memset on another recorded rank-0 CUDA stream overlaps this
interval. The table excludes kernel 2370 itself, which quantizes token
activations for FP8 dispatch and takes 38.176 microseconds.

### Fusion idea and limits of the evidence

Fuse the routing-metadata operations into one GPU kernel that constructs the
routing mask and probabilities directly from the expert IDs and scores. The
observed CPU launch gaps make this a concrete opportunity to reduce overhead in
this region.

The capacity-check all-reduce and the CPU read of `needs_drop` are separate
costs. Fusing metadata construction alone does not remove either. Avoiding the
host predicate read would also require keeping the capacity decision on the GPU
while preserving token-drop behavior.

No fused implementation or speedup has been measured. The 523.876 microseconds
of observed gaps are not established as fully recoverable, and this one interval
does not establish an optimizer-update or whole-training speedup.

## Overlap MTP FFN gradient reduction with the remaining MTP backward

### Decision and evidence scope

The owner selected the next experiment: put MTP FFN `w1`, `w2` and `w3` in a
separate compile and GFSDP group, then launch their reduce-scatter while the
remaining MTP backward executes. This is a selected experiment, not an
implementation or benchmark result. The initial target is FFN
gradient-reduction overlap.

The evidence is from `bf16-cache-trace-001`: V2 with the BF16 expert cache,
optimizer update 21, rank 0, three complete microbatches, including the MTP
reduce-scatter at Perfetto slice 9233. It matches
the events analyzed in the
[native readiness evidence](../runs/bf16-expert-cache/mtp-gradient-readiness-native.json)
and [Perfetto readiness evidence](../runs/bf16-expert-cache/mtp-gradient-readiness-perfetto.json).
The source revision is `6c6cd9c0e41a0a5d3d00e1e252b3c120052d3507`; the native
trace SHA-256 is
`fdcf72769a9ddbad51db80b0bf7663784f7bbf4c72971e3ee8d7925cfc800203`.

### Why a separate FFN boundary is a candidate

The current MTP is one GFSDP unit inside a fullgraph-compiled block. The GFSDP
post-backward hook waits for gradients of the wrapped module's inputs. With the
whole MTP under one parent boundary, that hook withholds the reduce-scatter
opportunity until the MTP input gradients return, even though local FFN weight
gradients are ready earlier. The backward dependency runs through FFN, MLA,
MTP input projection, then input norms and the end of MTP. A separate FFN unit
could expose its input-gradient readiness while those later operations continue.

The relevant integration points are the
[MTP wrapping path](../gift/torchtitan/models/gift_moe/_base/parallelize.py#L241),
[GFSDP post-backward hooks](../gift/torchtitan/distributed/gfsdp/_param_group.py#L1186)
and [gradient readiness handling](../gift/torchtitan/distributed/gfsdp/_param_group.py#L1347),
[block compilation](../gift/torchtitan/distributed/compile.py#L95), and the
[FP8 GEMM path](../gift/torchtitan/ops/float8/cuda_gemm.py#L201).

### Payload and producer timing

Each FFN weight contributes 171 MiB of FP32 reduce-scatter input: 513 MiB for
`w1`, `w2` and `w3`, or about 57.4% of the 893.674 MiB MTP total. These sizes
are the FP32 conversion and packing payload; the producer gradients are BF16.
For comparison, the MLA projections contribute 249 MiB, the input projection
128 MiB, and eight norms 3.674 MiB.

First-microbatch producer times below are trace-relative milliseconds within the
MTP backward interval. “Remaining” columns are elapsed time and recorded kernel
union from that point to MTP backward end. Rows for individual and cumulative
producers overlap; they are subsets and must not be added together.

| Producer point | Time (ms) | Remaining elapsed (ms) | Remaining kernel union (ms) |
| --- | ---: | ---: | ---: |
| `w2`, including unpad/crop clone | 192.305304 | 12.128140 | 11.931374 |
| One of `w1` or `w3` (exact order not recorded) | 194.188287 | 10.245157 | 10.072103 |
| All three FFN weights | 195.800677 | 8.632767 | 8.478657 |
| MLA `wo` | 197.000681 | 7.432763 | 7.316990 |
| Candidate completion of all MLA projection GEMMs | 202.002555 | 2.430889 | 2.388588 |
| Candidate completion of input projection | 203.854210 | 0.579234 | 0.558948 |

Producer-gradient completion is not itself the module-input hook. A conservative
candidate FFN boundary is after fused add/norm-prep kernel ordinal 97. On the
first microbatch that point is 195.864773 ms, leaving 8.568671 ms elapsed and
8.417153 ms of recorded kernel union. The exact add operands are unavailable,
so this is a candidate boundary inferred from the trace, not a measurement of a
separate-module hook.

| Microbatch | Remaining elapsed (ms) | Remaining kernel union (ms) |
| --- | ---: | ---: |
| 1 | 8.568671 | 8.417153 |
| 2 | 8.848800 | 8.700988 |
| 3 | 8.760927 | 8.613922 |

### Existing communication and possible overlap

On the first microbatch, MTP backward ends at 204.433444 ms. Whole-group packing
runs from 204.435620 to 204.938597 ms. The existing MTP reduce-scatter,
Perfetto slice 9233, runs from 204.946757 to 212.433632 ms, a duration of
7.486875 ms. The preceding communication completes at 192.540089 ms, and stream
24 is free when the whole FFN is ready; the same stream-availability pattern
holds across all three microbatches.

This shows a candidate interval in which an FFN group's reduce-scatter could
overlap later MTP backward work. It does not establish that the candidate
8.6 ms window equals 8.6 ms of step reduction: partial-group packing, collective
duration, contention and dependencies have not been measured.

### Trace limits and implementation requirements

All three MTP graphs have 163 native kernels with identical names and order.
The native trace retains the full graph intervals because the Perfetto import
omits some kernels; selected timestamps and ownership were cross-checked between
the native and Perfetto evidence. Mapping parameter groups to source modules and
the backward dependency is supported by the source and trace. Tensor shapes and
generated FX outputs are unavailable, so BF16 per-projection labels are less
certain than the aggregate FFN payload.

The experiment must expose the FFN as its own callable, compile boundary and
GFSDP unit with input-gradient readiness. Splitting only the parameter list
under the opaque parent compile would not establish the required boundary.
Preserve the existing FP8/BF16/FP32 numerical path, selective activation
checkpointing (SAC), gradient accumulation, optimizer behavior and checkpoint
semantics. Changes to compile or SAC boundaries can change fusion, recomputation,
memory use and launch overhead.

Before drawing a performance conclusion, verify per-rank readiness and
consistent collective ordering, FP32 packing/casting and buffer lifetimes,
partial reduce-scatter duration and contention, and loss and gradient agreement
for affected blocks. Then compare repeated, matched unprofiled throughput and
peak memory. A finer split for `w2` may be considered later; it is outside this
initial experiment.

## Investigate lower-volume capacity counters and SAC reuse

### Status and evidence scope

These are research ideas worth investigating, not implementation decisions or
measured speedups. Preserve the current conditional capacity-drop conditions,
quotas, score-based selection, and tie ordering. Do not change the capacity
policy or bypass its current path.

The evidence is `bf16-cache-trace-001`: V2 with the BF16 expert cache, optimizer
update 21, rank 0, and three microbatches. It records 39 capacity all-reduces,
each with 128 int64 counters (1,024 bytes). Their summed intervals are 1.206983
ms across 21 forward calls and 13.840274 ms across 18 backward recomputations,
for 15.047257 ms total in a 1,269.710 ms window (about 1.19%). The trace records
no other local GPU work overlapping the collectives. The
[capacity-window budget](../runs/bf16-expert-cache/capacity-window-budget.json)
and [screening summary](../runs/bf16-expert-cache/capacity-optimization-screening.json)
contain the measurements and scope limits.

These are exposed collective intervals, not predicted step savings. The
collectives retain the same EP participants and synchronization point, and
their lifetimes can include peer waits. Removing or shortening this work may
also move a wait to the following EP dispatch.

### Idea 1: reduce group counters before the EP all-reduce

The current code builds 128 local expert counts, all-reduces all 128, then
reshapes the global counts into `(EP, E / EP)` and sums each group. Keep the
local expert counts for the existing local expert-overload condition, but sum
each destination group locally before the all-reduce. Reducing those `EP`
int64 counters yields the same global group sums and threshold result, without
dividing counts, quotas, or capacity by `EP`. Preserve the current gate exactly:
local expert overflow and global group overload must both hold. The aggregation
is localized in [`select_capacity_routes`](../gift/torchtitan/models/common/moe/dispatch/capacity.py#L134).

For `E=128`, the per-rank collective input becomes 64 bytes at EP=8 (16x
smaller), 128 bytes at EP=16 (8x), and 256 bytes at EP=32 (4x); generally the
reduction factor is `E / EP`. This reduces payload only. It keeps the same
participants and barrier, so it does not predict a 16x, 8x, or 4x reduction in
the measured 15.047 ms.

### Idea 2: reuse the exact capacity result during SAC replay

Run the existing capacity selection in the original forward and retain its
Boolean keep mask for recomputation. Recompute differentiable routing scores
and gradients as usual, and apply the saved mask to the same expert assignments.
The existing mask has shape `[8192, 8]` and costs 64 KiB per layer,
microbatch, and rank. Reuse must preserve mask/assignment alignment, lifetimes,
the current no-drop behavior, capacity quotas, and the existing score selection
and tie ordering.

This needs a granular SAC policy. Saving only
`conditional_capacity_mask` does not ensure that its upstream collective inputs
are reused. Either save the targeted collective and existing mask operation, or
place the current capacity selection behind a functional saved boundary; then
verify that the compiled graph actually reuses both. The existing
[`comm` SAC category](../gift/torchtitan/distributed/activation_checkpoint.py#L85)
also saves large EP dispatch/combine outputs, so enabling it broadly would not
be a metadata-only change.

### Cross-rank limits and next checks

On both recorded ranks, the selected long backward collectives end within 0.5
microseconds: rank 1 starts 115.866 microseconds before rank 0 for Perfetto
slice 58892, and 167.205 microseconds before rank 0 for slice 60411. Ranks 2–7
are absent, so peer arrival, protocol progress, and resource effects remain
unresolved. See the [cross-rank evidence](../runs/bf16-expert-cache/capacity-crossrank-wait.json).
This makes the collective intervals poor proxies for saved step time; any
remaining wait may appear at the following EP dispatch.

If investigated, first check counter and mask identity for no-drop, overload,
and tied-score cases; then confirm compiled SAC graph reuse and all-rank
collective order, measure retained memory, and compare matched unprofiled
performance. This note authorizes research only: no training-source or
configuration edits and no new compute runs are included.

## Owner-selected row-owner input embedding

### Selection, scope, and status

On 14 November 2026, the owner selected full row-owner routing for the input token
embedding as the next research design. The objective is to remove both the full-table
weight all-gathers and the dense full-table gradient reduce-scatter from the embedding
path. Every row remains owned by one rank in the embedding's data-parallel process
group. Ranks route token IDs to those owners for lookup, return the selected embedding
rows, and route coalesced row gradients back to their owners.

The first target is the eight-GPU crop only. It is a proposal, not an implementation,
runtime result, or performance claim. The production topology and its owner group remain
a later compatibility gate. No source, configuration, runner, or training change is
authorized by this design note.

Keep the existing model and training contract: FP32 persistent parameters and gradients,
BF16 embedding compute, `acc32`, one MTP branch, the selected per-layer selective
activation checkpointing (SAC) policy, enabled model and loss compilation, dense fused
AdamW, EMA, and checkpoint/resume. The design is acceptable only if it integrates with
those enabled settings. A result that requires disabling SAC, compilation, EMA, or
another production setting does not meet the selected scope.

### Current embedding and rank contract

The input parameter is a plain, untied `nn.Embedding` table of shape `[128256, 4096]`,
with 525,336,576 elements. `lm_head` is separate. The embedding is one GFSDP parameter
group. The crop's eight-way contiguous row partition contains 16,032 rows per owner, or
65,667,072 elements. This gives 250.5 MiB per owner for the FP32 parameter/master shard
and its dense FP32 gradient, 125.25 MiB for a BF16 local row shard, 1,002 MiB for a
gathered BF16 compute table, and 2,004 MiB for the full FP32 table or gradient.

For this divisible crop only, the row mapping is `owner = token_id // 16032` and
`local_row = token_id % 16032`. The owner is the token embedding's dense FSDP
process-group rank. It is not an EP-local-expert rank and must not be derived from a
global-rank assumption. Resolve the actual mesh, group order, rank membership, and shard
metadata at integration time. The general GFSDP layout can include padding and
alignment, so the crop's simple division is not a universal mapping.

The decoder performs one token lookup before the repeated block loop. That output feeds
the trunk and is also carried into MTP, which shifts the same embedding tensor and
zero-fills its tail; MTP does not perform another lookup or introduce token IDs. Trunk
and MTP derivatives must therefore be combined at the shared embedding output before one
row-gradient exchange. Sending a first-branch gradient early would omit the later MTP
contribution or require a separate, numerically validated accumulation path.

Relevant local contracts are recorded in the [embedding source and trace
contract](../runs/bf16-expert-cache/embedding-local-contract.json). The lookup and MTP
dataflow are visible in [the
decoder](../gift/torchtitan/models/common/decoder_blocks/decoder.py#L135), [the
embedding module](../gift/torchtitan/models/common/embeddings/token_embeddings.py#L12),
and [the MTP block](../gift/torchtitan/models/common/decoder_blocks/mtp.py#L100).

### Baseline communication and numerical boundaries

The baseline keeps original/master shards in FP32, gathers BF16 compute weights, and
uses dense embedding backward. The native dense BF16 weight gradient is copied into a
full FP32 buffer and summed with reduce-scatter into the FP32 local shard. The
configured gradient divide factor is 1.0: this is a SUM with factor one, not an average
by eight or an extra division by the 32 microbatches. Global valid-token normalization
is owned by the loss path.

The crop runs 32 microbatches per optimizer update. After each synchronized microbatch,
the local FP32 shard gradient is initialized from its first contribution and accumulated
with later contributions. Preserve the same normalization, 32-microbatch cadence, FP32
accumulation, dense gradient materialization, optimizer timing, and update order.

Native trace evidence identifies `embedding_dense_backward`,
`compute_grad_weight<c10::BFloat16,long>`, and `sum_and_scatter<c10::BFloat16,long>`
with a BF16 output pointer. This confirms the BF16 output boundary but does not
establish every internal reduction detail of the installed PyTorch build. The
source-rank local duplicate coalescing and BF16 rounding that occur before cross-rank
FP32 communication are part of the numerical contract. FP32 owner accumulation alone
does not prove parity if either that local coalescing or its rounding boundary changes.

In particular, autograd through a deduplicated lookup, inverse map, or `index_select`
expansion is not evidence that duplicate token gradients are coalesced in the same
order, precision, or rounding point as native dense embedding backward. The
implementation must compare that path against the baseline or explicitly provide an
equivalent local accumulation and BF16 output boundary. It must not silently replace
source-local BF16 coalescing with per-occurrence FP32 accumulation at the owner.

One no-extra-rounding seam is available to a narrower backward-only adapter: after
native dense BF16 `unsharded_param.grad` is complete and before GFSDP copies and clears
it, select touched rows and route those already-coalesced BF16 values. This seam is not
the lifecycle of the selected full-owner design: that design excludes the embedding
parameter from GFSDP and must not depend on an unsharded GFSDP gradient buffer.

For full owner routing, produce source-local duplicate-coalesced BF16 row contributions
from the completed shared embedding-output gradient, matching the baseline's local
accumulation and BF16 output boundary. Native dense embedding backward can serve as the
numerical reference or an initial staging path if it can be invoked without the
embedding weight gather. That staging path materializes a full 1,002 MiB BF16 gradient
table per rank before rows are selected, so account for its allocation and lifetime. A
compact local coalescing path may replace it only after affected-path numerical parity
is demonstrated. Widen each source-rank BF16 contribution to FP32 before the owner SUM
and before adding it to the FP32 local gradient. Do not use BF16 collective SUM or BF16
owner accumulation. If a source contribution has already accumulated in FP32, including
a deferred or no-sync path, it cannot be cast to BF16 under this rule; that path needs
separate handling or an explicit support guard.

The owner reduction factor remains exactly one. Do not add `/8`, `/32`, a second loss
normalization, or another averaging step. The order of owner-side FP32 additions can
differ from the current NCCL reduce-scatter order, so assess the resulting elementwise
numerical difference under the scoped deterministic comparison. Do not promise arbitrary
bitwise identity without evidence.

### Proposed forward route

Keep the FP32 row shard as the registered owner parameter. For each actual lookup,
validate signed int64 token IDs against `[0, 128256)` before computing owners or
offsets. Empty-request markers and padding sentinels must not be treated as valid
vocabulary rows. Partition the valid IDs by the embedding owner process group, retaining
the original sequence position and the inverse mapping needed to restore output order.

Within each requester-to-owner segment, deduplicate repeated IDs for the forward lookup.
Send each unique row ID to its owner, let that owner gather the corresponding local
rows, and return values to the requester. Use the retained inverse map to expand the
returned values to the exact original token order. Forward values must preserve the
baseline BF16 compute dtype and shape. The owner can cast selected FP32 rows to BF16 for
lookup. A cached BF16 owner view or selected-row representation is optional; begin
without a cache unless invalidation is proven for optimizer updates, checkpoint loads,
and EMA swaps.

The route metadata is per lookup invocation. Keep immutable IDs, counts, permutation,
unique-row indices, and inverse mapping alive until that invocation's backward has
completed. Do not store a single mutable route slot that a later microbatch can
overwrite. Masked or loss-zero tokens may produce zero derivatives, so the input IDs
define a conservative candidate touched set, not guaranteed nonzero gradient support;
this is sufficient for correct zero-filled dense local gradients.

### Proposed backward route and local accumulation

Wait for the gradient of the shared embedding output after both trunk and MTP branches
have contributed. Sum occurrence gradients for each source-rank, owner, and row using
the baseline-compatible local coalescing and BF16 rounding boundary described above.
Keep an inverse relation from every occurrence to its source-local unique row; the
values sent to an owner must represent the sum of all occurrences of that row on that
source rank.

Route one coalesced row contribution per source-rank/owner/row to the row owner. Widen
every received source contribution to FP32 before summing across the owner group. For
each microbatch, write the result into the owner's dense FP32 row shard: zero rows not
present in the contribution set on the first microbatch, and add the current
contribution to the existing FP32 shard gradient on later microbatches. Preserve a
materialized dense `.grad` even when every contribution is zero. A `None` gradient can
change optimizer behavior and is not a valid substitute for a zero tensor.

The owner exchange must participate in a rank-consistent order, including empty
requesters, empty owners, all-zero gradients, and ranks with no unique rows for a peer.
If an implementation omits empty payload collectives, it must still make that decision
consistently for the entire process group. Error and no-gradient paths must not leave
peers waiting in unmatched collectives.

### Preserve dense optimizer, clipping, EMA, and checkpoint state

Keep the registered owner parameter and a dense FP32 gradient with its current FQN and
full logical shape. Continue using the configured dense fused AdamW path with its
existing moments, weight decay, clipping, and once-per-update step. Do not switch to
`SparseAdam`, LazyAdam, row-only optimizer updates, or optimizer-in-backward. Untouched
rows still take part in dense AdamW semantics, including momentum and weight decay;
zeroing rows in the gradient is not the same as skipping those parameter updates.

The owner shard still needs to participate in global norm clipping. A standard
`Shard(0)` DTensor is a possible ownership representation, but its interaction with
GFSDP `RaggedShard` DTensors, fused AdamW, foreach operations, and clipping has not been
established. Validate the mixed representation before choosing it. Optimizer
construction enumerates registered parameters, so this proposal does not by itself
require a new optimizer.

EMA is a required integration item. Current EMA setup, update, and swap code tracks
GFSDP parameter groups. A parameter excluded from GFSDP will not be covered
automatically. The owner shard needs explicit EMA registration, initialization,
per-update maintenance, swap behavior, checkpoint save/load, and invalidation of any
cached BF16 row representation after a swap. Treating EMA as a final verification-only
check would leave the selected design incomplete.

The main DCP path starts with ordinary module state and substitutes GFSDP-managed
entries; an ordinary owner parameter is not automatically dropped. Preserve the model
FQN, global shape, owner placements and group metadata. Verify a full save/load
continuation across main weights, dense Adam state, and EMA state through the next
optimizer update. A state-dict roundtrip without optimizer/EMA continuation is
insufficient.

### GFSDP integration boundary

GFSDP exposes `ignored_params`, but the current wrapping path includes the token
embedding and root wrapper and builds a prefetch chain. The selected parameter must be
excluded from every enclosing GFSDP wrapper that could shard or reduce it, including the
token-embedding and root paths. Update the prefetch relationship so it does not continue
to assume the owner is a normal GFSDP parameter. Do not let GFSDP and the owner path
both reduce or update the same gradient.

Exclusion also bypasses GFSDP's precision conversion, FP32 accumulation, post-backward
reduction, completion events, and gradient-buffer lifecycle for that parameter. The new
path must own source-row collection, dense FP32 local gradient storage,
first-versus-later microbatch accumulation, asynchronous buffer lifetime, collective
ordering, and readiness before clipping and the optimizer. It must handle no-grad/eval,
frozen parameters, absent backward, exceptions, and zero contributions. Guard
unsupported layouts or execution modes explicitly until they are integrated, including
tied weights, multiple parameter groups, incompatible mesh/placement/padding,
PP/TP/CP/HSDP, offload, and deferred/no-sync accumulation.

Source references for this seam are [`fully_shard` ignored-parameter
support](../gift/torchtitan/distributed/gfsdp/fully_shard.py#L76), [token embedding and
root wrapping](../gift/torchtitan/models/gift_moe/_base/parallelize.py#L215), [root
wrapper construction](../gift/torchtitan/models/gift_moe/_base/parallelize.py#L250),
[optimizer parameter enumeration](../gift/torchtitan/components/optimizer.py#L108), [EMA
setup and update](../gift/torchtitan/distributed/gfsdp/fully_shard.py#L533), and
[ordinary DCP state handling](../gift/torchtitan/distributed/gfsdp/_dcp.py#L383). The
[local owner integration
contract](../runs/bf16-expert-cache/embedding-local-contract.json) records the remaining
source-level limits.

### Compilation and SAC are required behavior

The current decoder performs the embedding lookup before the repeated block loop. Model
compilation wraps eligible repeated blocks, and the SAC policy selects blocks; neither
requirement implies that the whole model or the existing embedding stem is compiled as
one full graph. Keep the embedding at its current stem boundary if the integrated path
works with the enabled block compilation and the exact selected per-layer SAC policy. Do
not add a blanket requirement for a new whole-model fullgraph compile.

At the same time, compilation and SAC are not optional compatibility checks. Test the
owner path with the current enabled model and loss compilation, the same SAC selection,
MTP enabled, FP8 enabled, and `acc32` unchanged. Do not silence a compile or replay
failure by disabling a component or changing the SAC policy. Preserve the graph boundary
the baseline actually uses and record which part of the embedding route remains eager or
captured.

Any mutable routing state, owner count, permutation, or communication tensor must belong
to one forward invocation and remain valid until its backward. SAC replay must use the
same route and values, and must not replay or mismatch mutable collectives. Activation
recomputation must not issue duplicate owner communication or alter collective order.
Because MTP reuses the original embedding output, replay must not invent another token
lookup or a second set of embedding IDs.

If a custom operator is needed, provide correct schema, autograd, fake/meta behavior,
and eager/compiled tests for that operator. Do not rely on a static dummy zero-gradient
tensor, stale global route context, or unverified graph capture behavior. A custom
operator is not required if a module boundary and the supported eager stem already meet
the integration contract.

The source integration points are the [decoder
lookup](../gift/torchtitan/models/common/decoder_blocks/decoder.py#L135), [block
compilation](../gift/torchtitan/distributed/compile.py#L95), [SAC block
selection](../gift/torchtitan/distributed/activation_checkpoint.py#L362), and [MTP input
reuse](../gift/torchtitan/models/common/decoder_blocks/mtp.py#L100).

### Routing counts, padding, skew, and memory bounds

The crop has 8,192 input tokens per rank (`B=2`, `S=4096`), with signed int64 IDs. The
absolute unique-row upper bound per requester is therefore 8,192. At hidden width 4,096,
an unpadded compact payload at this upper bound is 64 MiB in BF16 or 128 MiB in FP32;
the ID array itself is at most 64 KiB. These are compact useful-row bounds before owner
padding and protocol overhead.

The inspected NeMo exact-count route gathers a `P x P` count matrix, uses equal-split
padding to the global maximum count, and reads count metadata to the host. At `P=8`, the
count matrix is 512 bytes. Its strict validation adds more device-host synchronizations.
With one padded segment for each of eight owners, the BF16 row-value buffer can reach 8
x 8,192 x 4,096 x 2 bytes, or 512 MiB per rank for one route phase. Returned forward
values and backward gradient values are separate phases. FP32 staging would double the
corresponding value-buffer bound. Actual allocation and peak overlap depend on
implementation lifetimes and must be measured.

Raw token IDs are not a uniform hashed workload. A frequent token or skewed owner can
make one requester-to-owner segment set the maximum padding for every peer. Count
metadata, useful row bytes, padded bytes, local duplicates, and owner skew must be
measured separately. The 64 MiB compact BF16 bound does not bound NeMo's padded network
buffer and cannot be read as a speedup prediction.

Capacity mode is not a safe general answer by itself. The inspected NeMo mode calibrates
from the maximum count of its first lookup, multiplies by 1.25, aligns to 64 with a
minimum of 64, then uses a fixed capacity with an overflow assert. It does not establish
that a first batch bounds later batches. Never drop rows silently. A chosen count
strategy must be rank-consistent, handle arbitrary supported batches, and report
overflow before any contribution is lost. Dynamic exact counts also need an explicit
treatment of host reads and synchronization rather than an assumption that moving
`unique` earlier removes the synchronization.

### Historical trace evidence and its limits

The timing reference is `bf16-cache-trace-001`, optimizer update 21, rank 0, three
microbatches, with source revision `657e77a4f8ddb116b75925e5671b00642a0e4d32`. The
current inspected task source revision is `6c6cd9c0e41a0a5d3d00e1e252b3c120052d3507`; do
not claim the historical trace directly executes every current source byte. The raw
resolved config is not present locally. Its retained receipt records SHA-256
`a71150cc5997eafd3d6d6cedc6bfab13c5c1472c7a06e0befdc1051f4cc5777e`.

The rank-0 Perfetto embedding backward gradient reduce-scatter is slice 43466, starting
at 853.516399 ms and lasting 5.182387 ms in the trace-relative clock. The layer-0
gradient reduce-scatter is a distinct slice (43435); do not attribute that layer event
to the embedding. The three recorded forward embedding all-gathers last 2.649900,
2.825164, and 2.809420 ms. The three backward embedding all-gathers last 4.214321,
3.941328, and 4.047825 ms. The dense FP32 gradient reduce-scatters last 5.198802,
5.182387, and 5.217619 ms.

These are collective intervals, not exposed loss, additive step time, or measured
recoverable speedup. The payload sizes are not wire-byte counts, and the trace is rank 0
only. The trace and local source/trace contract are linked in the [local embedding
contract](../runs/bf16-expert-cache/embedding-local-contract.json) and [native tail
record](../runs/bf16-expert-cache/tail-optimization-native.json). Treat these timings as
motivation to test mechanism removal, not a performance estimate.

There is a smaller, distinct existing control:
`model.tok_embeddings.set_unshard_in_backward(False)` after wrapping skips the GFSDP
backward weight unshard for the plain embedding. Native embedding backward needs IDs,
incoming gradients, and saved shape, not weight values. The forward weight all-gather
and dense gradient reduce-scatter remain. The BF16 shared-buffer allocation may be
reused after forward; reshard releases ownership but does not promise that storage was
freed or that its contents remain intact. This narrow setting is source-level feasible
but untested with the required compiled/SAC path. It is not a substitute for the
selected row-owner design and must be evaluated separately if pursued.

### Reusable implementation research and alternatives

The strongest donor inspected is NVIDIA NeMo Automodel's generic
`Qwen3_8_FlashNextOwnerShardedEmbedding`, pinned at
[`3ea1de8d0aa902c3329b585d194c2e5274a85f3f`](https://github.com/NVIDIA-NeMo/Automodel/blob/3ea1de8d0aa902c3329b585d194c2e5274a85f3f/nemo_automodel/components/models/qwen3_8_flash_next/engram.py).
Its generic owner table accepts ordinary global IDs; it does not require n-gram hashing.
It sorts and routes IDs to owners, runs owner-local `F.embedding`, returns values
through a reverse all-to-all and inverse permutation, then reverses the value exchange
in autograd. Its backward routes occurrence gradients and lets the owner-local embedding
backward coalesce duplicates. It sends every occurrence and has no requester-side
deduplication.

Reuse its owner-routing and inverse-route concepts, not its numerical or optimizer path
unchanged. BF16 owner parameters demote the baseline FP32 persistent storage and
accumulation. FP32 owner `F.embedding` changes compute and local duplicate-coalescing
boundaries; casting only the returned values to BF16 does not demonstrate backward
parity. Owner sharding also bypasses GFSDP's accumulation and reduction. Adaptation is
required for this workload.

NeMo's exact-count route uses count all-gather and host count reads, while its capacity
mode fixes padding after first-batch calibration and asserts on later overflow. Its
[pinned official Qwen
guide](https://github.com/NVIDIA-NeMo/Automodel/blob/3ea1de8d0aa902c3329b585d194c2e5274a85f3f/docs/model-coverage/llm/qwen/qwen3-8-flash-next.mdx)
documents an eight-node/64-H100, 100-step owner-sharding run, but the owner block is
eager and outside activation checkpoint recomputation and MTP is unsupported there. A
separate pinned DeepSeek guide reports 100-update H100 validation under its own
Torch/NCCL, owner-count, BF16-gradient/moment, accumulation, network-plugin, and PXN
conditions. This is real external owner-path evidence, not validation of our `[128256,
4096]`, DP8, 32-microbatch `acc32`, GFSDP, EMA, MTP, compile, or SAC contract. The
[alternatives research
record](../runs/bf16-expert-cache/embedding-alternatives-research.json) contains the
pinned paths and exact conditions.

Other inspected candidates do not provide a drop-in implementation:

| Candidate | Reusable part | Gap for this design |
| --- | --- | --- |
| [TorchRec row-wise sequence sharding](https://github.com/meta-pytorch/torchrec/blob/16c16e982bde3eb48a8d8505305c70019cb362ec/torchrec/distributed/sharding/rw_sequence_sharding.py#L121) and [pinned FBGEMM backward](https://github.com/pytorch/FBGEMM/blob/81de8c84ee278ca16a65d3f8ae0618573a0280d8/fbgemm_gpu/codegen/training/backward/embedding_backward_split_template.cu#L811) | Owner routing and inverse autograd all-to-all | Inspected DENSE backward is limited to embedding dimensions up to 2,048; our width is 4,096. The inspected weight dispatcher lacks BF16 weight support, and dense AdamW/EMA/GFSDP integration remains. |
| [NVIDIA cuEmbed](https://github.com/NVIDIA/cuEmbed/tree/792db47311f4556bda205da2e9c7a00df5bbcf5b) | Unique-row compressed backward and inverse mapping concepts | Inspected support is FP32/FP16 and single-GPU; BF16 and multi-GPU are future work. H100 examples are forward embedding-bag workloads at widths 32–128, not this D4096 distributed backward. |
| [NVIDIA DynamicEmb](https://github.com/NVIDIA/recsys-examples/tree/85e628db7a6ac4d5f5f070d1e2d15d51eead0534/corelib/dynamicemb) | Maintained BF16 row routing and H100/NVSwitch examples | Backward/update separation is on its roadmap; non-rowwise optimizer states follow embedding dtype in the inspected API. D4096 and the external FP32 dense AdamW/32-microbatch contract were not established. |
| [DeepSpeed sparse embedding path](https://github.com/deepspeedai/DeepSpeed/blob/892ed83e417df01d9d4f21f559a0b88183cf5e4c/deepspeed/runtime/engine.py) | Sparse gradient all-gather support | This is not a GFSDP owner-directed reduce-scatter replacement; ZeRO routing and optimizer handling differ, and a dense AdamW seam remains. |
| [PyTorch sparse NCCLX path](https://github.com/pytorch/pytorch/blob/e63b330cbcbcc136d3aab1895a183579dbb80dde/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp) | Sparse tensor API surface | The inspected NCCLX operation unpacks into dense output and runs dense all-reduce; it is not sparse-wire transport. |
| [DeepEP Engram buffer](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/buffers/engram.py) | Remote BF16/FP8 fetch/write and wait concepts | The inspected interface has no backward gradient push/reduce API; its current feature also requires a newer CUDA baseline than the recorded run. |

The inspected NCCLX [sparse block all-reduce
implementation](https://github.com/meta-pytorch/torchcomms/blob/8ecc3488fb132278b406f2abb755f1e8ba8a16e8/comms/ncclx/meta/collectives/AllReduceSparseBlock.cc#L91)
clears a dense receive tensor, unpacks sparse blocks, then performs dense all-reduce.
This is not sparse-wire communication.

The [kernel and transport research
record](../runs/bf16-expert-cache/embedding-kernel-research.json) and [alternative
survey](../runs/bf16-expert-cache/embedding-alternatives-research.json) preserve the
pinned source revisions and limitations. The choice is therefore to adapt the NeMo
owner-routing pattern around our local precision, optimizer, SAC, compile, EMA, and
checkpoint contracts. No donor is a drop-in, and no performance advantage has been
established.

### Finite implementation and acceptance gates

If source work is separately authorized, implement and review against these
affected-path gates before any production claim:

1. **Lookup and routing.** Compare owner-routed outputs with baseline BF16
   embedding outputs for valid int64 IDs, repeated IDs within one requester,
   IDs repeated across ranks, frequent-token skew, empty requesters, empty
   owners, and all-zero/ignored-loss cases. Check original token order, row
   owner, local row, inverse mapping, range validation, and empty sentinels.
2. **Precision and gradients.** Compare the source-local duplicate-coalesced
   BF16 rows at the native output boundary, owner-side FP32 SUM with factor 1,
   and dense FP32 shard gradients. Include combined trunk plus MTP gradients,
   per-microbatch accumulation over 32 microbatches, and the configured
   `acc32` behavior. Record observed numerical differences and their cause;
   do not impose a universal tolerance without the affected-path evidence.
3. **Dense update semantics.** Verify materialized zero gradients and dense
   fused AdamW behavior for untouched rows, including existing momentum and
   weight decay. Check global norm clipping when owner shards coexist with
   GFSDP ragged shards. Confirm one optimizer update after the same 32
   microbatches.
4. **EMA and persistence.** Exercise EMA initialization, every update, swap,
   swap-back, and cached-row invalidation. Save and reload main weights,
   Adam state, and EMA state, then compare the next update with uninterrupted
   execution. Preserve FQN, global shape, and owner group metadata.
5. **Compilation, SAC, and ordering.** Run the enabled compiled model/loss
   path with the exact current per-layer SAC selection, MTP1, FP8, and `acc32`.
   Confirm backward/recompute uses the original immutable route once, all
   ranks issue collectives in matching order for empty and nonempty traffic,
   and buffers are not reused before the last reader completes. Do not disable
   compilation or SAC to make this gate pass.
6. **Performance mechanism and cost.** Verify by trace that both weight
   all-gathers and dense gradient reduce-scatter are absent for the target
   embedding, that owner payloads match useful/padded row counts, and that no
   new host synchronization or staging allocation dominates the path. Measure
   peak memory and unprofiled throughput, including sorting, deduplication,
   padding, casts, pack/unpack, collective startup, and lookup work.

Use the affected embedding block and deterministic repository settings for numerical
comparisons. Compare with an identical control pair to measure A/A noise. For
performance, use the agreed production baseline and candidate, paired within the same
job, with 10 warmup and 10 measured steps after any startup or compilation. Record
software, configuration, seeds, topology, owner group, job identity, measured step
series, peak memory, and useful and padded traffic. Use traces to establish the
mechanism and unprofiled runs to measure magnitude. Neither the 5.2 ms historical
collective interval nor the 64 MiB compact payload is an expected step-time saving.

CPU/Gloo checks can cover route semantics and selected state behavior but do not close
H100/NCCL collective ordering, kernel, compilation, memory-lifetime, or performance
gates. Those checks require separate authorization and the actual target hardware. This
proposal itself authorizes no GPU job.

### Evidence and open status

The local evidence is read-only and dated 14 November 2026:

- [Embedding source, precision, trace, and integration contract](../runs/bf16-expert-cache/embedding-local-contract.json)
- [Pinned owner-sharding alternatives survey](../runs/bf16-expert-cache/embedding-alternatives-research.json)
- [Kernel and transport survey](../runs/bf16-expert-cache/embedding-kernel-research.json)
- [Native update-21 tail record](../runs/bf16-expert-cache/tail-optimization-native.json)

No implementation, test execution, GPU run, or benchmark is reported here. This document
records a proposal only; implementation and GPU validation require their own
authorization. The unresolved gates are actual runtime configuration and mesh order,
numerical equivalence of the source-local BF16 coalescing seam, dynamic count/padding
cost, integrated compiled/SAC behavior, dense optimizer/clipping compatibility, EMA and
DCP continuation, and H100/NCCL performance.
