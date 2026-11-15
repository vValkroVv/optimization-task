# Production-data A/B/C/D bounded trace analysis

All eight completed original traces were analyzed through actual MCP Perfetto. Original imports lose overlapping kernels; the labeled, annotation-preserving analysis derivatives reconcile every event category and device correlation flow with zero parser errors. Original files are unchanged.

This is supporting diagnostic evidence for the [campaign report](report.md). The [machine-readable result](trace-analysis.json) owns exact values, original/derivative hashes, source hashes, queries, capture hashes, per-microbatch scopes, payloads and comparisons.

## Coverage and import reconciliation

Each diagnostic follows 20 complete warmup updates and captures the first three complete forward/backward microbatches of update 21 on ranks 0/1. The closed [native execution receipt](artifacts/raw/native-abcd-orchestration/native-campaign-results.json) verifies all eight ranks still complete 32 microbatches and the optimizer; that work is outside this bounded trace. No optimizer is captured. The train-step marker closes at collection stop and is a prefix marker, not a complete update.

| Variant/rank | Original dropped kernels | Retained kernels | Retained GPU annotations | Retained GPU post_backward annotations | Device events / matched flows |
| --- | ---: | ---: | ---: | ---: | ---: |
| A/0 | 41 | 8795 | 354 | 98 | 9930 |
| A/1 | 52 | 8795 | 354 | 98 | 9930 |
| B/0 | 35 | 8391 | 298 | 98 | 9445 |
| B/1 | 42 | 8391 | 298 | 98 | 9445 |
| C/0 | 32 | 8103 | 298 | 98 | 9157 |
| C/1 | 45 | 8103 | 298 | 98 | 9157 |
| D/0 | 40 | 8230 | 330 | 103 | 9317 |
| D/1 | 39 | 8230 | 330 | 103 | 9317 |

Original `slice_drop_overlapping_complete_event` errors are retained in the result. All eight derivatives have zero errors and retain all raw X-event counts by category. Every kernel/memcpy/memset has exactly one imported launch flow with matching correlation. Raw flow starts/finishes are not expected to equal device counts: the originals also contain fwdbwd flows and GPU annotation projections. Raw phase counts and imported flow counts are recorded separately.

The [preparation script](artifacts/raw/trace-analysis/prepare.py) only colors overlapping device/annotation X events into nonoverlapping display tids and moves matching device flow-finish tids. Every event remains in its original array position; names, categories, timestamps, durations, args, metadata and all other fields are checked equal. CUDA `args.stream` stays unchanged. Synthetic lanes do not imply changed streams or execution. Unlike the historical normalizer, no GPU ranges are removed. CPU microbatch markers are exactly three per trace; GPU microbatch projections are three in A/B/C and six in D. The extra D projections are range structure, not extra microbatches.

## Comparable host and device scopes

Device events are selected by flows whose host launch starts inside one of the three CPU microbatch scopes. Device work may complete after its CPU scope closes. Busy/compute/communication times below are interval unions over those selected events. Device idle is the first-to-last selected device envelope minus its busy union. Copies and memset are included in busy and excluded from compute/communication. Communication means observed NCCL kernel families; its duration can include peer waiting.

All times are ms per rank for the three microbatches. These diagnostic spans do not estimate throughput or a full-update critical path.

| Variant/rank | CPU F+B scopes, sum | Device envelope | Device busy union | Compute union | Comm union | Comm ∩ compute | Device idle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A/0 | 1153.613 | 1235.253 | 1196.189 | 1128.323 | 207.059 | 146.533 | 39.064 |
| A/1 | 1154.646 | 1235.114 | 1194.341 | 1123.662 | 207.259 | 147.056 | 40.773 |
| B/0 | 1128.148 | 1211.396 | 1176.663 | 1104.986 | 225.481 | 156.462 | 34.733 |
| B/1 | 1126.028 | 1211.250 | 1172.509 | 1109.006 | 209.080 | 147.428 | 38.741 |
| C/0 | 1096.570 | 1179.029 | 1145.180 | 1078.014 | 216.392 | 154.262 | 33.849 |
| C/1 | 1100.314 | 1179.193 | 1144.554 | 1080.175 | 201.242 | 141.913 | 34.639 |
| D/0 | 1107.738 | 1164.618 | 1124.420 | 1065.795 | 170.454 | 116.979 | 40.199 |
| D/1 | 1107.000 | 1164.290 | 1125.014 | 1064.456 | 150.867 | 94.635 | 39.276 |

Per-family sums, unions, exposed intervals, stream IDs and host launch/wait API totals are in JSON. Overlapping family totals are never added as elapsed time. Exposed communication and nearby idle intervals do not prove dependency or causality.

## Incremental and cumulative mechanisms

### B/A: the combined four changes

The trace verifies ready-queue consumption in B and synchronous DataLoader consumption in A. Across 32 prefetched microbatches, rank-0/1 A DataLoader annotation totals are shown below against B queue-pop waits; their scopes differ and do not isolate a causal speedup. The three-F+B DtoD volume falls from 34,037,467,596 to 8,660,093,388 bytes per rank; DtoD copies fall from 181 to 100. Overall captured kernels fall from 8795 to 8391 per rank, while selected F+B host kernel launches fall from 8792 to 8388. The scan/gate kernel family remains native FLA. B changes four controls together; none of these differences establishes the individual throughput contribution of prefetch, reuse, routing or CE.

| Rank | A DataLoader host annotations (32), ms | B queue-pop host annotations (32), ms | A DtoD kernel sum, ms | B DtoD kernel sum, ms |
| --- | ---: | ---: | ---: | ---: |
| 0 | 246.657 | 0.383 | 23.460 | 6.392 |
| 1 | 195.903 | 0.415 | 23.282 | 6.497 |

### C/B: FlashQLA external_gate

B has FLA scan/gate/recompute/WY kernels; C has `tilelang_fused_chunk_gdr_fwd_kernel_kernel`, `tilelang_fused_chunk_gdr_bwd_kernel_kernel`, local cumsum and explicit `gdn_gate_fwd_kernel`/`gdn_gate_bwd_kernel`. This is consistent with the requested external-gate path; the native activation receipts establish the actual backend binding. The selected scan/gate family drops from 396 to 216 kernels per rank. Its summed kernel work is below; shared output gated norm and fused projections are excluded by the exact prefix definition in JSON.

| Variant | Rank 0 scan/gate kernel sum, ms | Rank 1 scan/gate kernel sum, ms |
| --- | ---: | ---: |
| A | 45.976 | 45.414 |
| B | 45.361 | 44.873 |
| C | 23.464 | 23.292 |
| D | 23.245 | 23.183 |

### D/C: FFN early reductions and row-owner embedding

D contains six FFN split reduce-scatters per rank: two groups × three microbatches, twelve combined across both ranks. Each carries 134,479,872 FP32 input elements (537,919,488 bytes, 513 MiB), group size eight, and starts before its parent group reduce-scatter. Their overlap with parent backward compute is 96.094–98.246% of the FFN reduce-scatter kernel duration. The numerator is the interval union of intersections between that RS kernel and non-NCCL, non-copy, non-memset kernels linked to the next parent `CompiledFunctionBackward` scope after the identified FFN scope (the latter has three TNT weight-gradient kernels). This is compute overlap; it does not prove which event each wait consumes. Exact scope/device IDs and payloads are retained in JSON.

Each C trace contains nine `tok_embeddings` collectives, while D contains zero; the six `lm_head` collectives per trace remain. D has three `_OwnerLookup` and three `_OwnerLookupBackward` scopes. Each forward routes Long count/ID metadata and exchanges BF16 values, while backward exchanges BF16 source-coalesced gradients before local owner accumulation. The captured rank-0 ID requests are 2319/1483/2409, versus rank-1 215/207/197. These are different real local batches, and owner receive counts differ; this is not balanced synthetic traffic. All owner payloads and local native coalescing kernels are saved.

D also adds six host `cudaStreamSynchronize` calls per rank inside the three owner forwards, about 0.034/0.033 ms summed on ranks 0/1, plus pageable/pinned DtoH metadata copies. The bounded three-F+B DtoH metadata volume is 1794 bytes/rank in D versus 42 bytes/rank in C; API/copy counts and directions are recorded. Host wait API duration is not exposed GPU delay. Both GPU idle envelopes and CPU F+B scope totals can increase despite shorter device busy unions, so no causal training gain is assigned to the controls individually.

### D/A: cumulative recipe

The cumulative traces retain the input-ready queue, external-gate scan, split FFN reductions and row-owner lookup. Device busy/compute/communication unions and copy volume differ as shown above; the JSON records B/A, C/B, D/C and D/A side by side for both ranks. These mechanisms support activation and execution interpretation. The separately unprofiled A/B/C/D runs own throughput and memory magnitudes, including fixed-order/cadence limitations.

## Limits and reproducibility

- Only ranks 0/1 and three of 32 microbatches are traced. No complete optimizer-update critical path, optimizer tail, cross-rank straggler cause, long-run behavior or numerical equivalence is inferred. Kernel launch flows cover the selected device events, but complete inter-stream CUDA event dependencies are unavailable; exact wait causality remains unproved.
- The real mixture has four domains globally. DP8, no shuffle and row-round-robin placement give rank 0 English Wikipedia and rank 1 FineMath local rows in this captured prefix. Exact rows/hashes and domains are taken from loader receipts at microbatches 640–642 in JSON; EP traffic can mix all eight ranks. These traces do not locally cover Russian-web/Python inputs on ranks 0/1.
- Profiled host synchronization and kernel sums are distinct from unprofiled elapsed performance. No isolated speedup, bandwidth or convergence claim follows from these captures. The scan/gate family definition excludes common output norm/projections; compute classification is kernel execution excluding current NCCL/copy/memset families, not a proof that all kernels do useful arithmetic throughout their span.
- One initial preparation invocation used the wrong parent directory and exited before writing derivatives; its path was corrected. One query-index patch was rejected before writing; the complete query catalog is `all-queries.json`. These tooling failures did not alter originals or measurements.

Actual MCP `load_trace` was called on every completed original before its derivative. `list_tables` and `list_table_structure(slice, flow)` established the schema. [All SQL](artifacts/raw/trace-analysis/all-queries.json), eight original/derivative import receipts and query results (`*.mcp.json`), host supplements (`*.supplement.json`), copy queries (`*.copies.json`), and [verification log](artifacts/raw/trace-analysis/checks.log) are retained under the derivative directory. The [analysis script](artifacts/raw/trace-analysis/analyze.py) computes all unions from saved MCP rows and verifies identities, preservation, census, flows and FFN cases. Source revisions/hashes, exact replay commands and capture hashes are in JSON. No new capture, training, package installation or training-source change was performed.
