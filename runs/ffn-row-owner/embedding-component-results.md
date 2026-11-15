# Final standard-autograd embedding component results

The completed standard-autograd owner measures **2.554 ms versus 13.040 ms** for fixture component forward + backward. Absolute allocated peak is **5,781,722,112 versus 8,950,373,888 bytes**. These are component measurements; the resident untied head is never computed. No whole-training speedup is established.

The superseded manual-gradient results remain byte-preserved in [historical Markdown](embedding-component-results-manual-history.md) and [historical JSON](embedding-component-results-manual-history.json). Their owner runs are not final acceptance evidence.

## Setup and scope

Eight H100 80GB ranks, one dense NCCL shard group; V=128256, D=4096, B=2, S=4096 (8192 tokens/rank), FP32 owner masters/returned shard gradient, native BF16 lookup/source coalescing and BF16 payloads. The actual untied FP32 lm_head and 2,101,346,304-byte private gradient bucket remain resident in both variants. No separate owner scratch pool or allocating fallback was observed; every owner rank borrows that bucket.

Each case uses two warmups and three unprofiled samples. A table median is the median of the three sample-wise slowest-rank spans. The displayed range is max minus min, not an uncertainty estimate. Baseline runs are reused unchanged: `baseline-fixture-001`, `baseline-integrated-001`, `baseline-phases-001`. Final owner runs are `owner-standard-{integrated,phases,stages}-001`; all exited 0 with eight rank receipts. Every per-rank signed-int64 input hash, route census, shape, head residency, fixture index and unchanged active GFSDP file hash matches its baseline.

## Integrated component forward + backward

| Case | Baseline wall ms (range) | Final owner wall ms (range) | Median wall change | Allocated peak GiB, base → owner | Reserved peak GiB, base → owner |
|---|---:|---:|---:|---:|---:|
| fixture | 13.040 (range 0.042) | 2.554 (range 0.878) | -80.41% | 8.336 → 5.385 | 15.172 → 16.846 |
| duplicates | 12.811 (range 0.285) | 2.437 (range 0.194) | -80.98% | 8.336 → 5.385 | 15.172 → 16.846 |
| skew | 13.130 (range 0.049) | 8.126 (range 0.117) | -38.11% | 8.336 → 5.385 | 15.172 → 16.846 |
| empty-requester | 12.783 (range 0.153) | 2.140 (range 0.038) | -83.26% | 8.336 → 5.385 | 15.172 → 16.846 |
| empty-all | 12.539 (range 0.021) | 1.799 (range 0.012) | -85.65% | 8.073 → 5.260 | 15.172 → 16.846 |
| boundaries | 12.777 (range 0.067) | 2.662 (range 0.055) | -79.16% | 8.336 → 5.385 | 15.172 → 16.846 |

Wall spans end after explicit device synchronization and include internal host waits. CUDA events are elapsed stream spans, including queue gaps and communication dependencies, not kernel-duration sums. Peaks are the maximum over every rank/sample; exact bytes, rank peaks and receipt hashes are in [the JSON](embedding-component-results.json). Allocator cache is retained between cases. Zeroing before measurement is excluded.

Allocated peak is lower for every case, while reserved high-water is higher in this process/cache history. This does not close the native total-memory gate. The independent equal-cache-warm [full-shape numerical run](embedding-numeric-results.md) measured allocated 5,781,722,112 / 8,950,373,888 and reserved 7,648,313,344 / 9,003,073,536 bytes (owner/baseline), demonstrating why reserved-cache histories must be stated.

### Host and device endpoint scopes

| Case | Variant | CUDA elapsed span ms | Host call ms | End synchronization ms |
|---|---|---:|---:|---:|
| fixture | baseline | 13.021 | 1.263 | 11.989 |
| fixture | owner | 2.534 | 2.458 | 0.438 |
| duplicates | baseline | 12.795 | 1.096 | 11.742 |
| duplicates | owner | 2.419 | 2.332 | 0.505 |
| skew | baseline | 13.109 | 1.017 | 12.170 |
| skew | owner | 8.106 | 2.635 | 5.441 |
| empty-requester | baseline | 12.767 | 1.003 | 11.965 |
| empty-requester | owner | 2.119 | 1.784 | 0.322 |
| empty-all | baseline | 12.521 | 0.991 | 11.710 |
| empty-all | owner | 1.768 | 1.464 | 0.341 |
| boundaries | baseline | 12.761 | 0.977 | 11.854 |
| boundaries | owner | 2.645 | 2.553 | 0.516 |

Each column independently takes the sample-wise slowest rank. Host call includes Python/dispatcher/autograd/submission and internal waits; columns must not be subtracted to infer CPU overhead or added to reconstruct a path.

## Diagnostic phases

Forward and backward synchronize separately; do not add their medians to estimate integrated latency.

| Case | Variant | Forward wall ms (range) | Backward wall ms (range) | Forward allocated / reserved GiB | Backward allocated / reserved GiB |
|---|---|---:|---:|---:|---:|
| fixture | baseline | 2.981 (range 0.172) | 10.018 (range 0.143) | 7.219 / 15.172 | 8.336 / 15.172 |
| fixture | owner | 1.596 (range 0.609) | 1.618 (range 0.230) | 4.162 / 16.846 | 5.385 / 16.846 |
| skew | baseline | 3.052 (range 0.104) | 10.337 (range 0.111) | 7.219 / 15.172 | 8.336 / 15.172 |
| skew | owner | 3.309 (range 0.184) | 4.705 (range 0.028) | 4.162 / 16.846 | 5.385 / 16.846 |

## Owner stage diagnostics

Nine named stages have ten contexts because `source_coalescing` occurs twice: native dense BF16 backward, then gather of unique rows into borrowed scratch. Event/host context spans are instrumented diagnostics without per-stage barriers. They include submission gaps and waits.

| Stage | Fixture event / host ms (event range) | Skew event / host ms (event range) |
|---|---:|---:|
| partition_ids#1 | 0.379 / 0.395 (0.295) | 0.609 / 0.643 (0.102) |
| counts#1 | 0.178 / 0.196 (0.264) | 0.654 / 0.675 (0.003) |
| request_exchange#1 | 0.099 / 0.099 (0.034) | 0.205 / 0.185 (0.016) |
| owner_lookup#1 | 0.053 / 0.070 (0.012) | 1.237 / 0.139 (0.026) |
| value_exchange#1 | 0.111 / 0.084 (0.033) | 2.444 / 0.174 (0.008) |
| reconstruct#1 | 0.049 / 0.054 (0.006) | 0.053 / 0.104 (0.000) |
| source_coalescing#1 | 0.640 / 0.178 (0.032) | 0.876 / 0.259 (0.001) |
| source_coalescing#2 | 0.007 / 0.062 (0.001) | 0.060 / 0.080 (0.001) |
| gradient_exchange#1 | 0.041 / 0.126 (0.017) | 1.115 / 0.140 (0.006) |
| owner_reduction#1 | 0.328 / 0.699 (0.116) | 2.118 / 1.065 (0.010) |

## Logical traffic and host reads

| Case | Unique requests/rank | Requests by owner | ID bytes | BF16 values / gradients bytes each | Padding bytes |
|---|---|---|---:|---:|---:|
| fixture | [80, 70, 77, 65, 70, 73, 79, 75] | [589, 0, 0, 0, 0, 0, 0, 0] | 4,712 | 4,825,088 | 0 |
| duplicates | [16, 16, 16, 16, 16, 16, 16, 16] | [128, 0, 0, 0, 0, 0, 0, 0] | 1,024 | 1,048,576 | 0 |
| skew | [8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192] | [63744, 256, 256, 256, 256, 256, 256, 256] | 524,288 | 536,870,912 | 0 |
| empty-requester | [1, 0, 0, 0, 0, 0, 0, 0] | [1, 0, 0, 0, 0, 0, 0, 0] | 8 | 8,192 | 0 |
| empty-all | [0, 0, 0, 0, 0, 0, 0, 0] | [0, 0, 0, 0, 0, 0, 0, 0] | 0 | 0 | 0 |
| boundaries | [4, 4, 4, 4, 4, 4, 4, 4] | [16, 8, 0, 0, 0, 0, 0, 8] | 256 | 262,144 | 0 |

These are application bytes summed over all eight ranks, not physical fabric traffic. Each owner contributes a 72-byte count vector; each rank explicitly reads its 576-byte gathered count matrix once/forward (4608 logical bytes across ranks). This is not the only inherited synchronization: MCP also observes a pinned DtoH and `cudaStreamSynchronize` in `partition_ids`, alongside the count-stage pageable DtoH. Native unique/sorting dynamic-output work remains on the affected path.

The unchanged baseline logically gathers a 1,050,673,152-byte BF16 full table in forward and again in backward, then reduce-scatters a 2,101,346,304-byte FP32 full gradient. Owner traffic replaces those payloads with routed rows plus count metadata.

## Bounded MCP Perfetto mechanism evidence

All **16** new canonical plain-JSON traces were loaded with MCP `load_trace`, then queried with `execute_sql`. Exact raw/imported X-event category counts match on every trace, including all kernels, GPU copies/memsets and overlapping GPU annotations; no normalization or timestamp rewriting was necessary. Total raw size is 4,177,957 bytes. One extra complete F+B/case/rank was profiled outside timing samples (one microbatch/rank, below the three-microbatch bound). [SQL, compact results and trace handles](embedding-component-perfetto.json) retain the queries; [primary JSON](embedding-component-results.json) pins all trace paths/hashes.

Source → CPU range → CUDA runtime/driver launch → device flow establishes the following for fixture and skew on all eight ranks:

- `partition_ids`: signed-int64 native sort/unique/reduction kernels and DtoD/pinned DtoH work; `counts`: one metadata NCCL AllGather and one pageable DtoH. The one AllGather is the 72-byte local count vector / 576-byte output, not an embedding table gather.
- `request_exchange`, `value_exchange`, `gradient_exchange`: exactly one `ncclDevKernel_SendRecv` each, tied by launch flow to the corresponding `all_to_all_single` context; three routed transports per rank/case.
- `owner_lookup` and `reconstruct`: native gather/copy kernels; native BF16 `compute_grad_weight<c10::BFloat16, long>` and `sum_and_scatter<c10::BFloat16, long>` occur in source coalescing, followed by gathered unique rows.
- `owner_reduction`: per-source BF16-to-FP32 copies and FP32 indexed accumulation; standard autograd subsequently accumulates the owned returned gradient. There is no full-table owner AllGather or ReduceScatter device kernel in any trace.

Profile durations are mechanism evidence and include rank-arrival/profiler perturbation. For example fixture rank 0 spends 14.496 ms in count-stage `cudaMemcpyAsync`, paired with a 14.540 ms metadata NCCL kernel; this is a wait-containing profiled span, not the unprofiled count-stage cost. Do not use it as a general communication bandwidth or CPU cost estimate.

| Profile, rank 0 | Non-NCCL kernel sum ms | NCCL kernel sum ms | Copy sum ms | Memset sum ms | Device interval union ms | Device first-to-last span ms |
|---|---:|---:|---:|---:|---:|---:|
| fixture | 1.085 | 16.418 | 0.0065 | 0.0179 | 17.527 | 19.678 |
| skew | 4.738 | 2.516 | 0.0056 | 0.0176 | 7.277 | 8.185 |

Kernel/copy/communication sums may overlap; the interval union is elapsed covered time across device tracks. Host runtime waits, per-stage launch attribution and every rank are retained separately in the MCP result. No baseline component profile was collected here, so before/after kernel-only magnitude is unavailable.

## Storage, numerical evidence and remaining gates

- The component baseline retains two distinct GFSDP gradient storages totaling 4,202,692,608 bytes; owner retains one head bucket of 2,101,346,304 bytes and returns a separately owned 262,668,288-byte FP32 shard gradient. Ordinary autograd owns parameter-gradient accumulation; no manual `.grad` write or guard path is timed.
- Native source backward still allocates the 1,050,673,152-byte BF16 dense gradient and 147,636,224-byte FP32 partial scratch at this crop, plus sort/CUB scratch. Measured allocator peaks include them. Retained route metadata is bounded by 655,360 bytes excluding input IDs.
- [Owner storage](owner-storage.md) owns production buffer/lifetime accounting: ordinary embedding budget 2004 MiB becomes computed padded MTP 893.673828125 MiB, private head remains 2004 MiB, backward borrows 704 MiB and returns 250.5 MiB owned output. This component harness retains the unchanged baseline embedding bucket and does not measure the future full-model pool.
- [Full-shape numerical results](embedding-numeric-results.md) and `owner-standard-autograd-eight-rank-nccl.log` already passed standard-autograd outputs/gradients, duplicate/order/normalization, shared consumers, accumulation/reset/hooks, `autograd.grad` without `.grad` mutation, pending reuse, clipping/Adam, EMA and flat DCP continuation. This evidence task did not rerun them. The numerical launcher Markdown-writer failure followed complete JSON save; its log and offline report repair remain recorded.
- Full native SAC/compile integration, actual pool/peak/lifetime accounting, and matched native training performance remain separate gates. The head is resident but never computed here, so these traces cannot establish native pending-head/FFN overlap. Existing pending-borrow lifecycle tests remain valid for their tested scope. No additional MTP/layer0 A/A investigation is required by this report; the owner accepted that variability.

## Reproduction and checks

Final owner source SHA256 `f3a01c6cf99b981db7b4ef66866fe46259af84e9d2c21e2044e9ae18ff1696fd`; harness SHA256 `5a1a129b3c1d4fe239f66d93f664436aefade87e2fc09e98df70f5904f43b07b`. The final and historical variants have distinct source/diff maps in their receipts. Fixture index SHA256 `7cc8d53152c801450fdae44ba84c182eb8dcb9106d8bba9e3c3d69ebdaa8a8c2`. Actual runtime is PyTorch `2.15.0.dev20260822+cu130`, CUDA 13.0, NCCL 2.30.7, Python 3.12.3. Exact argv, source/runtime maps, hashed stdout/all-rank receipts, per-rank sample series and absolute peaks are linked by the [primary JSON](embedding-component-results.json).

Offline verification passed: six-case per-rank input/census/config comparison, three-sample slowest-rank aggregation, all-rank peak aggregation, unchanged baseline-path hashes, all16 raw/Perfetto event-count parity, routed transport/native BF16 mechanism checks. No GPU launch, numeric tolerance/test, source edit or additional campaign was made by this evidence task. Historical CPU/Gloo PREMUL_SUM limitations remain preserved without retry. The coordinator recorded all eight GPUs empty after these embedding runs before transferring the lease to FFN work.

Relevant supporting documents: [harness scopes](component-harness.md), [owner storage](owner-storage.md), [full-shape numerical evidence](embedding-numeric-results.md).
