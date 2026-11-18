# Four fresh update-21 Perfetto comparisons

## Result and scope

All eight fresh rank-0/1 captures pass original/imported event, annotation,
launch-flow/correlation and asynchronous-tail reconciliation. Each diagnostic
completed all 21 updates on eight ranks, including accumulation 32 and optimizer
work; the separate native proofs own that fact. **The requested layers.7 compiled
compute overlap with MTP-parent reduce-scatter is absent in every captured
microbatch.** This is a measured negative mechanism result, not a capture failure.
No source fix or additional GPU run was performed.

The four fresh runs are `schedule-embedding-{d0,s,se,se-fused}-diagnostic-140-004`.
They capture the first three complete forward/backward microbatches of update 21,
after 20 warmups, on ranks 0 and 1. Historical D0 traces were not substituted.
These profiles establish neither native throughput nor full-update critical-path
improvement, convergence, peak memory or behavior on the other six ranks.
[Primary native timings](../timing-results-140.md) remain separate.

## Evidence and method

- [Original/derivative identities](identities-004.json): original hashes, exact
  derivative hashes, event counts and permitted-change verification for all eight.
- [Final capture acceptance](capture-acceptance-final-005.json): immutable receipt
  joining each original hash to its actual native parser proof SHA, actual MCP
  response hashes, SQL, methods and tail evidence. Native/raw receipts remain
  unchanged; this closes their capture-tail gate separately.
- [Compact comparisons](comparison-004.json): all ranks and microbatches, exact
  collective payload/dtype/group metadata, interval unions, host counts and every
  owner kernel/copy/host-operation count difference. Its pinned ignored detail
  retains exact operation names, launch ancestry, first layers.7 kernels and all
  individual intervals.
- [Rank 0 timeline](four-way-004-rank0.html) and
  [rank 1 timeline](four-way-004-rank1.html): all three microbatches and all four
  variants, with head, parent/layers.7 and owner-embedding panels. Hover gives exact
  kernel names, times and original CUDA streams. Panels have independent zeroes.
- Actual MCP `load_trace` loaded all eight derivatives. Actual `execute_sql`
  ran [existing queries](queries.json) and [additional queries](queries-extra-004.json).
  Device and work-launch queries used SQL pages of 4,000 rows at offsets
  0/4000/8000, below the 5,000-row cap. Every response was checked for errors and
  truncation; final partial pages and reconciled counts establish exhaustion.
  Responses and load receipts remain under ignored `artifacts/raw/trace-analysis/`.

`prepare.prepare` preserves every event and field except overlapping GPU display
`tids`, corresponding finish-flow `tids`, and missing original-stream arguments.
All CPU ops, GFSDP annotations, timestamps, durations and launch flows remain.
Synthetic display lanes are **not CUDA streams**; analysis uses `args.stream`.
`capture004.py` extends the unchanged semantic analyzer with inverse launch
coverage, device tails, exact host differences, compact-fill launch geometry,
semantic checks and later-backward gap summaries. No training runtime changes.

## Capture integrity and tails

| Variant | Rank | Kernels | Copies | Memsets | GPU annotations | Matched launch/device flows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D0 | 0 / 1 | 8,230 / 8,230 | 202 / 202 | 885 / 885 | 330 / 330 | 9,317 / 9,317 |
| S | 0 / 1 | 8,230 / 8,230 | 202 / 202 | 885 / 885 | 330 / 330 | 9,317 / 9,317 |
| SE | 0 / 1 | 8,197 / 8,194 | 202 / 202 | 882 / 879 | 330 / 330 | 9,281 / 9,275 |
| SE-fused | 0 / 1 | 8,173 / 8,170 | 199 / 199 | 882 / 879 | 330 / 330 | 9,254 / 9,248 |

All original complete-event categories, including CPU ops and user annotations,
match their imported counts exactly. Every device kernel/copy/memset has one
launch flow with equal correlation ID, and every captured work-producing CUDA
runtime/driver launch/copy/memset has exactly one device target. There are no
nonzero importer error/drop statistics or truncated SQL results.

Device ownership follows host launch membership in each CPU microbatch, including
GPU tails beyond that CPU scope. Every microbatch has captured asynchronous work.
The third microbatch retains 187–206 such events, extending **17.999651–19.708998
ms** beyond its CPU end. The last device event ends 4.314–6.078 microseconds before
the recorded final `cudaDeviceSynchronize` returns; capture ends another
30.325157–31.288521 ms later. No work launch lacks its final device event.
This flow/correlation and drain evidence, together with the original-trace/native
parser hashes, closes the tail gate; projection counts alone were not used.

## GFSDP scheduling

Matching uses exact module/operation ownership, launch ancestry, NCCL family,
element counts, dtype, group and microbatch. Slice IDs are local joins only.
The original compiled-block association remains unchanged: each module's
`GFSDP::pre_backward` is followed by its local `CompiledFunctionBackward`, and
actual GPU kernels belong through launch ancestry. First matched layers.7 kernel
names/ancestors remain in the result for inspection.

| Observed property | D0 | S / SE / SE-fused |
| --- | --- | --- |
| Both MTP AGs complete before head RS | No: FFN AG follows head RS | Yes, all 18 candidate rank/microbatch cases |
| FFN parameters wait behind head gradient communication | FFN AG is queued behind it | Parameters are already gathered before head RS |
| Early FFN RS overlaps local compute | 1.492–1.956 ms | 1.492–4.823 ms |
| FFN RS end → layers.7 dense AG start | 13.498–13.670 ms | S 0.030–0.208; SE 0.075–0.701; fused 0.003–2.008 ms |
| layers.7 dense AG ∩ remaining local kernels | 0.084–0.105 ms | S 0.789–2.685; SE 0.559–2.370; fused 0.662–1.440 ms |
| layers.7 dense AG ∩ positively owned remaining MTP-prefix kernels | UNESTABLISHED | UNESTABLISHED |
| Early FFN RS ∩ positively owned remaining MTP-prefix kernels | UNESTABLISHED | UNESTABLISHED |
| Norm gradient packing starts before parent RS ends | No | Yes, every captured case |
| Parent RS ∩ actual layers.7 compiled kernels | **0 ms** | **0 ms** |

The last row is an unmet original overlap objective. In S/SE/fused, actual
layers.7 compiled kernels begin **2.336–56.065 microseconds after** parent RS
ends, despite much earlier CPU compiled submission. For example S rank 0 MB0:
parent RS 200.022091–201.843694 ms; layers.7 kernels begin 201.850254 ms.
The local kernels overlapping parent RS are norm/auxiliary backward operations,
not layers.7 compiled compute. Annotation occupancy and other stream activity
cannot establish the requested overlap. The precise remaining wait producer and
consumer are not established by the captured event data; no causal attribution
is inferred from adjacent endpoints alone.

Norm packing, earlier weight gathering and observed FFN RS overlap with generic
local kernels are separate findings. Neither generic overlap row establishes
overlap with positively owned remaining MTP-prefix compute. The bounded saved-MCP
inspection finds the overlapping norm/attention-recompute kernels under a later
anonymous `CompiledFunctionBackward`, distinct from the first compiled scope
following the parent's `pre_backward`. That evidence does not uniquely establish
the later block's module ownership. Both prefix-specific metrics therefore remain
**UNESTABLISHED**; the generic unions retain their measured values under explicit
local-kernel field names. They do not imply that the unmet layers.7 overlap occurred.
The accepted seven-storage, 6,973,565,952-byte-per-rank pool inventory and two
region-remap/lifetime checks are reused; profiling did not introduce an inventory
or new buffers. Source mapping and prior lifetime tests establish the intended
released FFN-region reuse; trace timing alone does not establish alias safety.

All captured GFSDP collective counts, exact kernel families, input/output counts,
dtypes and groups match D0 across variants and matched microbatches. Examples:
head RS 525,336,576 FP32 input elements; MTP-parent RS 99,791,360; FFN RS
134,479,872; norm RS 135,168. Parent/FFN AG outputs remain 99,791,360 /
134,479,872 BF16 elements; final dense AG output remains 85,230,080 BF16.
The result retains layers.7 dense/expert and layers.6/5 collectives to avoid
mistaking a nearby reduction for the parent operation.

### Later backward and next boundaries

The retained `head_to_ffn_gap_ms` values span D0 **5.214–11.012 ms**, S
**0.874–11.863 ms**, SE **0.845–12.343 ms**, and SE-fused **0.888–12.466 ms**.
Its exact inherited definition is the gap between (a) the latest ending device
event on original CUDA stream 7 whose end is at or before the backward FFN AG's
end and (b) the earliest starting device event on stream 7 whose start is at or
after that AG's end, within the same host-launch-owned microbatch. This is an
**AG-adjacent stream-7 inactivity proxy**, despite the historical field name;
its endpoints are not necessarily head-RS completion or actual compiled-FFN
consumer start. Large candidate gaps remain in these samples. Having weights
ready before head RS does not prove elimination of all FFN compute waits, and
these captures establish neither a causal regression nor its absence.

Layer-6 compiled computation begins before layers.7 dense RS ends in every
capture: D0 −4.043 to −2.022 ms; S −3.266 to −1.487; SE −3.832 to −1.485;
fused −4.267 to −1.501. Captured later-backward gaps without local compute sum
to D0 11.375–17.367 ms, S 10.657–16.316, SE 12.421–15.883 and fused
12.116–18.430 ms. Their largest individual gaps range 0.753–2.241 ms across
all captures. The result retains the five largest gaps per microbatch, NCCL
activity unions and the portion before the next host launch; these classify
activity, not proven dependency stalls.

Device-tail→next-microbatch-first gaps are D0 0.388–0.421 ms,
S 0.382–0.430, SE 0.371–0.427 and fused 0.381–0.473 ms. No large new displaced
stall is apparent in these two captured boundaries. **Only two next-microbatch
boundaries per rank exist; the third boundary and remaining 29 microbatches are
not captured.** Single profiled samples cannot prove absence of a later or
full-update stall. The fused later-gap increase is retained rather than promoted
to a causal regression or silently excluded.

## Embedding comparisons

All owner operations use `_OwnerLookup` / `_OwnerLookupBackward` launch ancestry.
The following sums cover the three owner forwards plus three backwards per rank.
Kernel sums are separate from elapsed span; concurrent categories are not added
to estimate elapsed time. Host scope spans include ordinary Python/native work
and synchronization; nested host operator durations are not additive.

| Variant | Rank | Local kernels ms | Copy+memset ms | NCCL ms | Gap union ms | Sum of owner device spans ms | Host scope spans ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D0 | 0 | 3.786511 | 0.071741 | 0.891968 | 3.000870 | 7.751090 | 9.011932 |
| D0 | 1 | 2.787800 | 0.070847 | 1.501989 | 2.985192 | 7.345828 | 9.549570 |
| S | 0 | 3.756809 | 0.072158 | 0.992609 | 3.233289 | 8.054865 | 11.655749 |
| S | 1 | 2.760456 | 0.070531 | 2.399339 | 2.969326 | 8.199652 | 12.189907 |
| SE | 0 | 1.712130 | 0.069696 | 1.908644 | 2.995686 | 6.686156 | 7.717919 |
| SE | 1 | 1.039624 | 0.061821 | 1.571592 | 3.010733 | 5.683770 | 7.273967 |
| SE-fused | 0 | 1.676771 | 0.065120 | 1.515171 | 2.321125 | 5.578187 | 6.915031 |
| SE-fused | 1 | 1.011525 | 0.058751 | 0.985829 | 2.932748 | 4.988853 | 7.185027 |

NCCL durations can contain peer waiting. Differences in those spans or host gaps
are not isolated transfer or fusion costs.

### D0 versus S

Owner device operation identities/counts are unchanged: each forward has 31
local kernels, three NCCL operations, three copies and ten memsets; backward has
34 local kernels, one NCCL operation and five memsets. Host synchronization/copy
counts remain unchanged. The changed schedule interacts with peer/host timing:
rank-1 owner NCCL sum rises from 1.501989 to 2.399339 ms in this capture, while
local kernel sums are nearly unchanged. This does not isolate a native slowdown.

### S versus SE: three non-routing optimizations

Both SE variants execute `_owner_lookup_bf16_kernel` once per forward and
`_owner_add_bf16_kernel` eight times per backward. The former directly loads FP32
owner rows and stores BF16; the latter retains source-rank-ordered FP32
load/add/store. See [route and backward source](../../../gift/torchtitan/distributed/gfsdp/row_owner_embedding.py)
and [kernel source](../../../gift/torchtitan/distributed/gfsdp/_row_owner_embedding_kernels.py).
The native fallback conversion/index-add pairs disappear. Forward local-kernel
count falls 31→30, and backward falls 34→24 on rank 0 / 23 on rank 1.

Compact coalescing replaces the full-vocabulary BF16 table/zero-fill with a
unique-row table; it does **not** remove every BF16 fill. The exact same native
BF16 fill kernel's grid shrinks from 513,024 blocks to rank-0
9,276 / 5,932 / 9,636 and rank-1 860 / 828 / 788 blocks. Both SE variants have
the same compact grids. The FP32 dense owner-gradient zero remains required.
Raw launch geometry, source selection and matched coalescing ancestry distinguish
this smaller fill from a missing full-vocabulary allocation.

### SE versus SE-fused: complete path difference inventory

SE has **zero** `_routing_metadata_kernel` and `_validate_clamp_ids_kernel`
executions; SE-fused has three of each per rank, one per 8,192-token forward.
The supported `flat.numel() <= 8192` route and all other stages remain identical.
Every owner backward device identity/count is unchanged. Non-owner device
identity/count multisets are also identical on each rank.

Per forward, fusion adds one validation/clamp and one routing-metadata kernel;
it removes ten native kernels: two signed comparisons, boolean AND, boolean
all-reduction, long validity conversion, clamp, owner floor division, two long
fills and scatter-add. Net local count is 30→22. It removes the single metadata
DtoD copy, leaving two DtoH copies and the same two host stream synchronizations.
The unique/sort path, owner lookup, request exchange and response exchange remain.

The compact JSON lists **every exact kernel/copy identity and every changed host
operator/API count for every rank/microbatch**, not just these categories.
Host changes include removal of `ge`, `lt`, `__and__`, `bitwise_and`, `clamp`,
`div`, `ones_like`, `scatter_add_`, `zeros` and `zero_`; fewer `all`, `to`,
`_to_copy`, `copy_`, `empty_strided`, `fill_`, `as_strided` and `slice`; one extra
`select`; CUDA runtime kernel launches 29→19, driver launches 4→6, and async
copies 3→2. Backward has no count/path difference. Exact unchanged communication
identities/payloads are retained beside the differences.

Across three forwards, observed owner local-kernel sums decrease only
0.035359 ms on rank 0 and 0.028099 ms on rank 1 from SE to fused; much larger
owner elapsed-span changes also contain peer waits and host gaps. These local
savings do not establish a native-training benefit. Use the separate 40+100
unprofiled matrix for the requested SE-fused/SE throughput comparison.

## Verification and limits

Executed all eight semantic summaries, original/category censuses, full
launch/device correlation reconciliation, inverse work-launch checks,
third-microbatch drain bounds, and cross-variant GFSDP signature equality.
Saved SQL pages are untruncated and contain no importer errors. Static HTML/SVG,
JSON/hash and scoped Python checks are recorded in final acceptance. No browser
binary was available for a screenshot; timelines are static SVG with native
hover text and no JavaScript dependency. No GPU, installation, training source,
configuration, frozen harness, cache or historical artifact was modified.

The first immutable capture receipt was retained when later-backward gap evidence
was added. The immutable final-004 receipt is also retained after correcting the
ownership labels; its affected prior derived bytes are archived by
[the correction archive manifest](ownership-correction-005-archive.json).
Final-005 acceptance explicitly supersedes those earlier derived-file references.
Original traces, native proofs and accepted census/flow/tail evidence are unchanged.
The frozen native proofs continue to say `TAIL_PENDING`; only the separate final
receipt closes that gate. Schedule overlap remains a negative outcome and its
remaining dependency cause is unknown.
