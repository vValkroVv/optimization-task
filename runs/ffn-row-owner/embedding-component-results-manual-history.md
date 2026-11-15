# Embedding component GPU measurements

Historical measurements of the superseded manual-gradient owner implementation.
The subsequent standard-autograd revision requires a new complete forward/backward
and peak-memory check. Unchanged stage evidence may be reused after source review.

The required component GPU scopes completed on the frozen source. The integrated table compares the actual GFSDP baseline with the actual row-owner path at V=128256, D=4096, B=2, S=4096 on eight H100 ranks. Each case used two warmups and three unprofiled samples; the displayed median is over the three sample-wise slowest-rank latencies. The sample spread is max minus min, not a run-to-run noise estimate.

## Integrated forward + backward

| Case | Baseline wall ms | Owner wall ms | Owner / baseline median change | Peak allocated GiB, base → owner | Peak reserved GiB, base → owner |
|---|---:|---:|---:|---:|---:|
| fixture | 13.040 ms (spread 0.042) | 2.449 ms (spread 0.753) | -81.2% | 8.34 → 5.28 | 15.17 → 17.33 |
| duplicates | 12.811 ms (spread 0.285) | 2.295 ms (spread 0.726) | -82.1% | 8.34 → 5.28 | 15.17 → 17.33 |
| skew | 13.130 ms (spread 0.049) | 7.814 ms (spread 0.191) | -40.5% | 8.34 → 5.28 | 15.17 → 17.33 |
| empty-requester | 12.783 ms (spread 0.153) | 1.969 ms (spread 0.009) | -84.6% | 8.34 → 5.28 | 15.17 → 17.33 |
| empty-all | 12.539 ms (spread 0.021) | 1.488 ms (spread 0.033) | -88.1% | 8.07 → 5.01 | 15.17 → 17.33 |
| boundaries | 12.777 ms (spread 0.067) | 2.173 ms (spread 0.573) | -83.0% | 8.34 → 5.28 | 15.17 → 17.33 |

The spread and complete per-rank sample series are in each linked all-ranks receipt. Allocated peaks are absolute measured peaks, including actual GFSDP buffers, native embedding backward output/scratch, and owner gradient. Reserved peaks retain allocator cache high-water marks. The owner path is lower in allocated peak but higher in reserved peak; this measurement alone does not close total-memory acceptance.

## Logical routing traffic and count reads

| Case | Unique rows per requester rank | Unique requester rows by owner | Request IDs B | BF16 values returned B | BF16 gradients sent B | Padded B |
|---|---|---|---:|---:|---:|---:|
| fixture | [80, 70, 77, 65, 70, 73, 79, 75] | [589, 0, 0, 0, 0, 0, 0, 0] | 4,712 | 4,825,088 | 4,825,088 | 0 |
| duplicates | 16 × 8 | [128, 0, 0, 0, 0, 0, 0, 0] | 1,024 | 1,048,576 | 1,048,576 | 0 |
| skew | 8192 × 8 | [63744, 256, 256, 256, 256, 256, 256, 256] | 524,288 | 536,870,912 | 536,870,912 | 0 |
| empty-requester | [1, 0, 0, 0, 0, 0, 0, 0] | [1, 0, 0, 0, 0, 0, 0, 0] | 8 | 8,192 | 8,192 | 0 |
| empty-all | 0 × 8 | [0, 0, 0, 0, 0, 0, 0, 0] | 0 | 0 | 0 | 0 |
| boundaries | 4 × 8 | [16, 8, 0, 0, 0, 0, 0, 8] | 256 | 262,144 | 262,144 | 0 |

Each owner rank also produces a 72-byte count vector; each requester reads a 576-byte count-matrix output once per forward (eight explicit host reads total per invocation, 4,608 logical bytes across ranks). Padding is zero for every measured case. These are logical application bytes, not physical fabric traffic. The baseline path performs full-table BF16 gathers in forward and backward and a full-table FP32 reduce-scatter in backward; byte counts for those tensor sizes are in the JSON receipt.

## Diagnostic phase timings and peaks

Forward and backward were synchronized separately here. Do not add these spans to estimate the integrated path.

| Case | Variant | Forward wall ms | Backward wall ms | Forward peak alloc / reserved GiB | Backward peak alloc / reserved GiB |
|---|---|---:|---:|---:|---:|
| fixture | baseline | 2.981 ms (spread 0.172) | 10.018 ms (spread 0.143) | 7.22 / 15.17 | 8.34 / 15.17 |
| fixture | owner | 0.986 ms (spread 0.531) | 1.371 ms (spread 0.167) | 4.16 / 17.33 | 5.28 / 17.33 |
| skew | baseline | 3.052 ms (spread 0.104) | 10.337 ms (spread 0.111) | 7.22 / 15.17 | 8.34 / 15.17 |
| skew | owner | 3.303 ms (spread 0.168) | 4.695 ms (spread 0.017) | 4.16 / 17.33 | 5.28 / 17.33 |

## Owner stage diagnostics

Slowest-rank medians of stage CUDA-event spans and host context-call spans follow. These are instrumented stage spans, not kernel-only costs; counts include the CPU count-matrix read. Per-rank series and traces are linked in the JSON and raw output directory.

| Stage | Fixture event / host ms | Skew event / host ms |
|---|---:|---:|
| partition_ids#1 | 0.378 / 0.398 (event spread 0.221) | 0.440 / 0.460 (event spread 0.111) |
| counts#1 | 0.183 / 0.200 (event spread 0.180) | 0.212 / 0.233 (event spread 0.055) |
| request_exchange#1 | 0.107 / 0.116 (event spread 0.020) | 0.110 / 0.115 (event spread 0.013) |
| owner_lookup#1 | 0.051 / 0.067 (event spread 0.009) | 1.264 / 0.090 (event spread 0.001) |
| value_exchange#1 | 0.097 / 0.093 (event spread 0.014) | 2.415 / 0.099 (event spread 0.013) |
| reconstruct#1 | 0.050 / 0.054 (event spread 0.006) | 0.053 / 0.061 (event spread 0.000) |
| source_coalescing#1 | 0.634 / 0.178 (event spread 0.029) | 0.880 / 0.239 (event spread 0.002) |
| source_coalescing#2 | 0.006 / 0.065 (event spread 0.000) | 0.056 / 0.074 (event spread 0.005) |
| gradient_exchange#1 | 0.084 / 0.121 (event spread 0.027) | 1.115 / 0.139 (event spread 0.024) |
| owner_reduction#1 | 0.597 / 0.709 (event spread 0.127) | 2.465 / 0.749 (event spread 0.006) |

Repeated source-coalescing contexts are numbered in execution order: #1 is native dense BF16 embedding backward; #2 selects coalesced rows into borrowed scratch. Stage contexts contain host submission and stream gaps/waits. MCP Perfetto analysis is needed for device, copy, communication, and dependency attribution.

## Buffer inventory and temporary memory

- Both variants retain the untied FP32 lm_head and its 2,101,346,304-byte private gradient buffer.
- The baseline has 2 deduplicated GFSDP gradient storages totaling 4,202,692,608 bytes; the owner variant has 1 totaling 2,101,346,304 bytes.
- In the owner path, scratch aliases the lm_head private buffer on all ranks; no separate scratch pool or allocating fallback was observed.
- Native BF16 backward still allocates a 1,050,673,152-byte dense gradient and a 147,636,224-byte FP32 partial buffer at 8,192 occurrences, plus native sorting/CUB scratch; allocator peaks include this work.
- Retained routing IDs/indices are bounded by 655,360 bytes (640 KiB) at this crop, excluding input IDs; they remain independent of reusable scratch.

## Protocol, identity, and remaining gates

Source revision 2ada87125e433e1f76c4cb2bfad244b641a78a93; tracked GIFT diff SHA256 3db7d1e26d435ca2e54400e113481a5cbf6dc79d044b2bdc1a9a1b11e37ae55b; row-owner source SHA256 ef2f0ef26dc9e0107b1c0d533bdfe225f47e448c93bdf56a76294ec24140fba1; harness SHA256 5a1a129b3c1d4fe239f66d93f664436aefade87e2fc09e98df70f5904f43b07b. All run receipts match this identity and the fixture index hash 7cc8d53152c801450fdae44ba84c182eb8dcb9106d8bba9e3c3d69ebdaa8a8c2.

The fixture integrated pair was reused without rerun. The remaining integrated cases, phase cases, and owner stage/profile cases each ran serially with a 600-second timeout and exited 0. Every run produced eight rank records with the expected mesh, shape, and active implementation. Baseline/owner signed-int64 input hashes and route censuses match rank by rank.

The stage profile contains one extra complete F+B per case on each rank, outside timed samples, with 16 raw traces total. The profiling traces and stdout/rank receipts are retained under [the raw artifact directory](artifacts/raw/embedding-components/owner-stages-001/). Detailed run commands, hashes, all-rank series, exact memory bytes, traffic, and trace hashes are in [the JSON result](embedding-component-results.json).

The assigned component measurements are complete. Full-shape numerical replay, MCP trace interpretation, resolution of the higher reserved-memory peak, and the native training comparison remain coordinator gates. These results do not establish whole-training TPS, convergence, or long-run stability. The documented CPU/Gloo backward attempt was not retried.

After the final torchrun exited, GPUs 0–7 each reported 0 MiB and 0% utilization and the compute-app list was empty; the GPU lease was released before report generation.

## Existing task evidence

- [Embedding component harness and timing scopes](component-harness.md)
- [Owner storage accounting and NCCL checks](owner-storage.md)
- [Row-owner implementation](../../gift/torchtitan/distributed/gfsdp/row_owner_embedding.py)
