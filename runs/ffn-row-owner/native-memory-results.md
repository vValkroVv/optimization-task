# Native GFSDP memory and integration results

**Result:** all-three accumulation-32 native integration and the common-baseline total-memory cap pass. Candidate GFSDP storage is lower by 1,746,392,064 B/rank. Candidate global allocated/reserved maxima are lower. Paired rank 2 reserved is 62,914,560 B (60 MiB) above baseline rank 2 but remains below the baseline worst-rank cap; allocator cause is unproven.

## Native integration

A002 and B001 completed one optimizer update on every rank. Resolved batch values (local 2, global 512, DP shard 8, sequence length 4096) imply 32 microbatches and 262,144 physical input tokens/rank/update; per-rank receipts confirm the token count. B001 resolved all three new flags on and logged all three activation markers on all eight ranks. FP8 grouped GEMM/DeepGEMM warmup, SAC, model/loss compile, MTP loss, BF16 expert reuse, optimizer/clipping/EMA and default data order stayed enabled. These are memory diagnostics; they do not establish steady-state TPS.

## Deduplicated storage

Every rank reported identical unique storage. `lm_head` data aliases an existing data slot and is counted once; its private FP32 gradient pool remains 2,101,346,304 B.

| Per-rank pool | A002 | B001 |
|---|---:|---:|
| Unique GFSDP storage | 8,719,958,016 B | 6,973,565,952 B |
| Ordinary dense BF16 data | 2 × 1,050,673,152 B | 1,050,673,152 B + 468,542,464 B |
| Ordinary dense FP32 grad | 2,101,346,304 B | 937,084,928 B |
| Expert BF16 data | 2 × 603,979,776 B | unchanged |
| Expert FP32 grad | 1,207,959,552 B | unchanged |
| Private `lm_head` FP32 grad | 2,101,346,304 B | unchanged |

Dense padded requirement falls from 525,336,576 elements to 234,271,232 (MTP remainder 99,791,360 + FFN 134,479,872); expert maximum remains 301,989,888 elements. On all ranks, layer-0 data intervals `[0,135,889,280)` and `[135,889,280,404,849,024)` and gradient intervals `[0,271,778,560)` and `[271,778,560,809,698,048)` are contiguous/disjoint in existing slots. MTP intervals exactly cover existing data `[0,199,582,720)` + `[199,582,720,468,542,464)` and gradients `[0,399,165,440)` + `[399,165,440,937,084,928)`. Regular layer data slots alternate by parity; no pool growth.

Owner output is an owned 262,668,288 B/rank tensor. Its scratch borrows the private `lm_head` pool without a separate pool. Native sort/CUB scratch and metadata are included in allocator peaks but excluded from this GFSDP-only inventory; see [embedding component accounting](embedding-component-results.json).

## Allocator peaks by rank

Cells show `allocated / reserved` bytes. `first update` spans initialization through update 1; the reset occurs in `MetricsProcessor.__init__` before model-parallel/model initialization and the window starts before `Trainer.train`, so constructor/model-init and compile/lazy-init high-water may contribute. Warm step 2 measures with persistent Adam state live; combined peaks retain both phases.

| Rank | A first | B first | A warm 2 | B warm 2 | A combined | B combined | B−A combined reserved |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 61,164,528,640 / 61,331,210,240 | 59,286,802,432 / 61,480,108,032 | 71,949,230,080 / 72,427,241,472 | 70,578,472,448 / 72,303,509,504 | 71,949,230,080 / 72,427,241,472 | 70,578,472,448 / 72,303,509,504 | -123,731,968 |
| 1 | 61,164,528,640 / 61,310,238,720 | 59,286,802,432 / 61,501,079,552 | 72,060,826,112 / 72,406,269,952 | 70,708,547,072 / 72,364,326,912 | 72,060,826,112 / 72,406,269,952 | 70,708,547,072 / 72,364,326,912 | -41,943,040 |
| 2 | 61,164,530,688 / 61,205,381,120 | 59,286,802,432 / 61,459,136,512 | 71,877,903,360 / 72,200,749,056 | 70,479,796,736 / 72,263,663,616 | 71,877,903,360 / 72,200,749,056 | 70,479,796,736 / 72,263,663,616 | +62,914,560 |
| 3 | 61,164,528,640 / 61,329,113,088 | 59,286,802,432 / 61,457,039,360 | 72,124,427,776 / 72,764,882,944 | 70,550,977,024 / 72,198,651,904 | 72,124,427,776 / 72,764,882,944 | 70,550,977,024 / 72,198,651,904 | -566,231,040 |
| 4 | 61,164,528,640 / 61,247,324,160 | 59,286,802,432 / 61,438,164,992 | 72,195,010,048 / 72,645,345,280 | 70,477,277,184 / 72,207,040,512 | 72,195,010,048 / 72,645,345,280 | 70,477,277,184 / 72,207,040,512 | -438,304,768 |
| 5 | 61,164,528,640 / 61,352,181,760 | 59,286,803,456 / 61,480,108,032 | 72,071,354,368 / 72,509,030,400 | 70,633,229,824 / 72,282,537,984 | 72,071,354,368 / 72,509,030,400 | 70,633,229,824 / 72,282,537,984 | -226,492,416 |
| 6 | 61,164,533,760 / 61,190,701,056 | 59,286,807,552 / 61,423,484,928 | 72,170,534,912 / 72,517,419,008 | 70,448,528,384 / 72,037,171,200 | 72,170,534,912 / 72,517,419,008 | 70,448,528,384 / 72,037,171,200 | -480,247,808 |
| 7 | 61,164,528,640 / 61,331,210,240 | 59,286,803,456 / 61,459,136,512 | 72,173,265,408 / 72,676,802,560 | 70,386,661,888 / 71,997,325,312 | 72,173,265,408 / 72,676,802,560 | 70,386,661,888 / 71,997,325,312 | -679,477,248 |
| Global max | 61,164,533,760 / 61,352,181,760 | 59,286,807,552 / 61,501,079,552 | 72,195,010,048 / 72,764,882,944 | 70,708,547,072 / 72,364,326,912 | 72,195,010,048 / 72,764,882,944 | 70,708,547,072 / 72,364,326,912 | — |

Common worst-rank cap is 72,195,010,048 B allocated / 72,764,882,944 B reserved; **all eight B ranks pass both**. Candidate global combined maximum falls 1,486,462,976 B allocated and 400,556,032 B reserved. First-update reserved premium persists on all ranks: 127,926,272–253,755,392 B. At the combined paired-rank comparison, rank 2 is +62,914,560 B; other seven ranks are lower.

## Run evidence and caveats

| Run | Outcome | Raw record |
|---|---|---|
| A002 | Child exit 0; old outer parser rejected warmup=0. Recovered offline without rewriting the attempt. | [recovered summary](../ffn-row-owner/artifacts/raw/ffn-row-owner-memory-a-002-summary-recovered.json), SHA256 `934b1abdb316c43099dc5ba7f7bf6bdc7f1896c22ab2d5a6b2b1a24638d62744`; [raw run](../ffn-row-owner/artifacts/raw/ffn-row-owner-memory-a-002/) |
| B001 | PASS, 8 ranks, no OOM/retry. | [summary](../ffn-row-owner/artifacts/raw/ffn-row-owner-memory-b-001/summary.json), SHA256 `2313a4ed6682dae81aabbfbfa4a14e30a09eb446813352c55fc7b0df4511df8e`; [raw run](../ffn-row-owner/artifacts/raw/ffn-row-owner-memory-b-001/) |
| Warm A | PASS, 8 ranks; update 2 measured. | [summary](../ffn-row-owner/artifacts/raw/ffn-row-owner-memory-warm-a-001/summary.json), SHA256 `d24e8a8f007fa45772060b31479adcb46a3d5d2691583bad7fa314653d4499c0`; [raw run](../ffn-row-owner/artifacts/raw/ffn-row-owner-memory-warm-a-001/) |
| Warm B | PASS, 8 ranks; update 2 measured. | [summary](../ffn-row-owner/artifacts/raw/ffn-row-owner-memory-warm-b-001/summary.json), SHA256 `ef6fd8ec108e6827331bd3186faf1bb0433eee2f58c64bfd7516f04a5ec1e3ef`; [raw run](../ffn-row-owner/artifacts/raw/ffn-row-owner-memory-warm-b-001/) |

A001 failed before training because the MDS fragment used the obsolete `opt-fq` path. Its one-line repair points to the task-local fixture. The index hash remains `7cc8d53152c801450fdae44ba84c182eb8dcb9106d8bba9e3c3d69ebdaa8a8c2`; a read-only audit found 30 files / 14,702,684 B and byte-identical old/new fixture contents. The A002 original failed summary and parser failure remain preserved. A scoped parent parser fix produced the separate recovered summary.

A002 shutdown logs retain multiprocessing `_remove_temp_dir` `OSError: [Errno 16] Device or resource busy` on an NFS `.nfs*` file. Runner cleanup says the task temp directory was removed; its process group stopped with no orphan or leftover temp files.

Source head `2ada87125e433e1f76c4cb2bfad244b641a78a93`, tracked tree SHA256 `5ce91928f221b15be1fe594c2c3b55c6c430d4877354039f3bfd3be4b5e1fabd`. Exact commands and their hashes, resolved config hashes, raw rank-manifest hashes, and all per-rank phase values are in [native-memory-results.json](native-memory-results.json). Cache: `runs/overnight-streaming-perf/artifacts/raw/combined-four-baseline-003/cache`. Our processes stopped and the lease was returned after warm B; root has reassigned it. No TPS claim or additional GPU work is part of this result.
