# Activated FLCE: matched native loss trace

**Activation passes. The entire head/loss forward and both gradients are
slower in this capture.** The loss device union increases from 88.63–90.81 ms
to 105.53–106.15 ms per sampled microbatch, a 16.57–19.09% increase relative to
its matched control sample. This is a local profiled comparison, not a general
training-throughput estimate. The coordinator owns unprofiled TPS, numerical
acceptance and retention.

## Runs and import gate

The fresh control is `four-fusion-control-post-wrapper-baseline-trace-002`;
the candidate is `four-fusion-linear-ce-linear-ce-active-trace-002`, both under
`runs/overnight-streaming-perf/artifacts/raw/`. Both completed with unchanged
source/config receipts, exit code zero and stopped process groups. Both use
the canonical-wrapper fix recorded in
[post-wrapper implementation pins](runtime/post-wrapper-implementation-pins.json)
at source HEAD `33b336a592746340d327d86e34149a23dc8688ee`.
Their resolved configuration differs only in `dump_folder`.

The production setup remains accepted V2: FP8 trunk, BF16 head, model/loss
compilation, CuTe, SAC, one MTP head, B2/S4096, EP8, dense FSDP8/expert FSDP1,
accumulation 32, prefetch 32, epoch 51200, seed 2048 and the original LR schedule.
Update 11 captures three complete F/B microbatches on ranks 0/1 after ten warmup
updates. All eight ranks complete all 32 microbatches and optimizer updates;
ranks 2–7 have no direct per-kernel activation capture.

Actual `mcp__perfetto__load_trace` and `execute_sql` imported all four prepared
traces. Each control rank has 8,795 kernels + 274 copies + 861 memsets = 9,930
device events; each candidate rank has 8,987 + 274 + 939 = 10,200. Every event
has exactly one host-launch flow, with zero importer errors. Category counts
and duration sums match preparation receipts. Original plain JSON equals its
gzip copy; canonical/prepared device fingerprints match after excluding only
display `tid`. Synthetic lanes have no collisions or ambiguous original
device endpoints. Moved events/endpoints are 39/39 and 46/46 for control,
35/35 and 49/49 for candidate. CUDA streams remain original `args.stream`.

The [JSON](active-linear-ce-trace-analysis.json) retains exact handles,
canonical/prepared hashes, source/config/runtime pins, roots, bounds, device
and host-flow IDs, query ledger and results. Large complete SQL results stay
in the ignored artifact `artifacts/raw/active-linear-ce-perfetto-queries.json`,
with its SHA-256 in the JSON. Old baseline IDs were not reused.

## Actual activation and work

| Per captured rank, three microbatches | Fresh control | Activated FLCE |
| --- | ---: | ---: |
| Outer head/loss chunks | 6 | 6 |
| `four_fusion::native_linear_ce` CPU roots | 0 | 6 |
| Centered CE forward/backward device tiles | 0 / 0 | 48 / 48 |
| Projection / dH / dW GEMMs | 6 / 6 / 6 | 48 / 48 / 48 |
| Original compiled CE device kernels | 36 | 0 |
| Native-op FP32 dW fill / BF16 dW cast kernels | Not this op | 6 / 6 |

Every centered tile has device grid `[1024,1,1]`. The active source and resolved
dimensions explain eight 1,024-row tiles per outer chunk: main and MTP are
folded together, giving 8,192 rows, hidden 4,096 and vocabulary 128,256.
Tensor-shape recording is off, so those tensor dimensions are source/config
derivations; the tile grids and all launch counts are observed.

Each baseline chunk has three `aten::mm` roots. Each candidate chunk has
16 `aten::mm` plus eight `aten::addmm` roots, with 24 actual GEMM kernels.
The candidate CPU op owns the centered kernels, all GEMMs, FP32 zero fill,
FP32 dW updates, BF16 output cast, reductions and term construction by
same-thread launch containment followed by device flows. It computes **dH
and dW during custom-op forward**. Its later autograd backward returns those
saved gradients; that does not eliminate gradient computation.

## Equivalent bounds and latency

The common enclosing loss guard starts at the end of
`GFSDP::post_forward (mtp.layers.0)` and ends at
`_DecoderOutputGradientBackProp`. It includes folded setup, two head calls,
both immediate inner backwards, loss statistics/SUM/scaling, FP32 hidden
gradient accumulation, callbacks, gradient accumulation/sync/reshard and final
hidden-gradient output cast. Chunk boundaries are the two head
`GFSDP::root_pre_forward` roots through the next root or bridge.

CPU loss submission and GPU completion are different scopes. Both main-thread
and autograd-worker launches are included; control backward launches occur on
a different `utid` from head forward. Omitting that worker would lose dH/dW.
Compiled CE graph hashes differ across ranks; exact mapped roots and linked
kernel names establish identity, not hash equality.

All values below are ms per rank/microbatch, control → candidate. Device union
uses interval merging across owned streams, not duration addition. Device span
is first owned device start → last owned device end. CPU→tail is loss CPU start
→ `max(CPU end, last owned device end)` and includes asynchronous queue backlog.

| Rank / micro | CPU submission | Device union | Device span | Kernel-duration sum | CPU→tail |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 / 1 | 8.639 → 7.905 | 88.633 → 105.530 | 88.797 → 105.901 | 103.278 → 108.107 | 94.278 → 111.613 |
| 0 / 2 | 7.678 → 8.614 | 90.614 → 105.660 | 90.760 → 106.450 | 94.235 → 108.232 | 97.756 → 113.320 |
| 0 / 3 | 7.192 → 8.176 | 90.809 → 106.131 | 90.960 → 106.777 | 93.955 → 109.149 | 97.864 → 113.560 |
| 1 / 1 | 8.138 → 9.877 | 88.650 → 105.576 | 88.779 → 105.944 | 91.334 → 109.435 | 94.597 → 112.498 |
| 1 / 2 | 7.418 → 8.836 | 90.644 → 105.663 | 91.361 → 106.039 | 105.637 → 109.833 | 98.417 → 112.987 |
| 1 / 3 | 7.300 → 8.950 | 90.735 → 106.146 | 90.883 → 107.064 | 93.921 → 110.915 | 97.867 → 114.046 |

Kernel sums can exceed the union because NCCL on stream 24 overlaps stream 7.
The JSON records all twelve chunk measurements separately. Chunk one has no
owned NCCL and still rises from 37.38–38.05 ms union to 45.96–47.84 ms;
the regression is visible before the second chunk's communication/callback
contamination. Spans of overlapping chunks must not be added.

## Source costs and observed mechanism

| Device kernel sum per entire two-chunk microbatch | Control range, ms | Candidate range, ms |
| --- | ---: | ---: |
| Projection | 21.92–24.62 | 22.87–23.26 |
| dH GEMMs | 21.99–22.70 | 20.56–22.52 |
| dW GEMMs / FP32 addmm updates | 21.32–22.85 | 38.17–42.37 |
| CE forward + backward kernels | 5.91–6.39 | 4.55–4.70 |

The dominant added device cost is the tiled dW operation. Source uses
`torch.addmm(..., out_dtype=torch.float32)` into one accumulator across eight
tiles; its actual kernels are
`nvjet_sm90_tss_320x128_64x3_1x2_h_badd_coopB_NTT`. Baseline uses a whole-chunk
BF16 dW GEMM. The trace establishes different kernels, launch granularity and
measured cost; it does not isolate accumulation precision from tile shape or
library algorithm as independent causes. The CE saving does not offset the
dW increase. The two variants retain the same three dense arithmetic passes;
there is no fewer-GEMM or fewer-kernel explanation.

Across three sampled microbatches, custom-op-owned FP32 dW fill costs
4.25/4.13 ms on ranks 0/1 and final BF16 dW cast costs 6.11/6.12 ms.
These are included in the enclosing measurements. The selected path adds no
device-value guard read or host synchronization by design. Recorded loss APIs
contain no `cudaEventSynchronize` or `cudaStreamSynchronize`; candidate event
queries, capture-mode calls, allocations, submission and event waits remain
included. `cudaStreamWaitEvent` counts stay 43 per micro; memset launches rise
6 → 32. An event query or stream wait is not evidence of a blocking host wait.
API sums, launch durations and launch→device delays are separate in the JSON.

Source memory arithmetic predicts the original full outer-chunk BF16 logits
at 1.957 GiB versus 0.245 GiB per tile. The candidate also allocates a 1.957 GiB
FP32 dW accumulator and returns a 0.979 GiB BF16 dW tensor, while GFSDP's existing
gradient buffers and saved dH/dW lifetimes still matter. The wrapper fix itself
adds setup-only Python references and placement checks. No snapshot or memory
timeline was captured; buffer arithmetic is not a measured peak saving.
The prior inactive pair's −1.97 GiB is not FLCE memory evidence.

## Communication, gaps and downstream work

The loss's owned streams are 7 and 24. Stream-7 interval unions rise from
79.48–82.38 ms to 96.31–98.68 ms. Owned stream-24 unions range
11.10–26.21 ms in control and 9.63–14.56 ms in candidate; communication-kernel
duration includes possible device waiting. There is no separately identified
EP kernel owned by this loss guard. Loss-owned memcpy sums are 0.047–0.048 ms
in both variants; memset sums rise approximately 0.005–0.007 → 0.028–0.030 ms.
Other kernels remain explicitly classified as other, without assuming all are
useful computation.

Head all-gather is prefetched during MTP forward and is reported separately:
one conversion kernel plus one NCCL all-gather per micro, 3.10–3.22 ms NCCL
duration. Gather→bridge is contaminated by remaining MTP work. The second
inner backward prefetches MTP parameters and callbacks traverse module states;
their enclosing communication is retained and not called exclusive head cost.
Concurrent devices in the loss span are separately clipped and merged in JSON.
Communication overlaps other device work by 10.83–25.61 ms in control and
9.17–14.04 ms in candidate. These overlaps are not additive elapsed costs.

Owned-device gaps total 0.129–0.718 ms in control and 0.368–0.918 ms in
candidate. After accounting for **all** concurrent recorded device work,
all-device idle inside those spans is 0.129–0.165 ms and 0.368–0.388 ms.
The slowdown is mostly extra busy time, not an observed long all-device idle
interval. The first subsequent stream-7 launch is scalar fill, followed by
compiled conversion/add and CuTe normalization on the decoder-backward path.
Its gap after the owned stream-7 tail is 2.37–3.94 microseconds; other streams
cover those gaps, so their all-device idle intersection is zero. Exact host
launches, owners, device IDs and bounds are retained. Flow links alone do not
identify CUDA event dependencies; no causal downstream wait claim is made.

Global token reduction completes before the first microbatch in both variants.
Whole-micro F/B CPU/device summaries are retained as wider guards; input-prefix
work, remaining 29 microbatches, optimizer, evaluation and checkpoint costs are
outside this local head comparison. The bounded profiler region is not a full
optimizer-update measurement.

## Verification and remaining gates

Preparation preservation, import/flow gates, three microbatches, two chunks,
actual activation, GEMM/tile counts and mapped whole-loss costs pass for both
captured ranks. JSON parses and local links/whitespace pass. One artifact patch
was rejected because it attempted delete/add for one path in one operation;
it was retried separately. One SQL query hit derived-table eviction after
loading four traces; schema discovery confirmed absence and own tables were
recreated. Both failures remain in the evidence; no measurements were dropped.
No source/config/runner write, installation, test run or GPU command was made.

This trace establishes an active path and a local regression. Repeated matched
unprofiled TPS/memory, the affected numerical block and quality criteria,
unprofiled-rank activation, long-run stability and continuation remain separate
gates. The old inactive FLCE timing/memory pair remains invalid. Its update-11
microbatch-18 stall still has unknown cause; this analysis supplies no compile
proof and no reason to extend warmup.
