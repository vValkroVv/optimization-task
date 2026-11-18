# Historical native D0/S/SE results

This record reconciles the completed original 10-warmup + 10-measured matrix.
Its historical raw `SE` variant is the four-stage embedding variant. Raw run
IDs and receipts remain `SE`; this is not the later `SE-fused` recipe and does
not complete the revised 40+100 campaign.

The machine-readable record contains the complete measured cadence and
`step_end` series, per-rank peak memory and allocator deltas, exact run
arguments, configuration and source identities, and SHA256 references for
the raw logs and receipts: [native-results.json](native-results.json).

## Timing results

Each row is one fixed-order run on eight H100 80GB GPUs. Physical input was
2,097,152 tokens per complete optimizer update. The physical TPS is the
driver's global window metric: 20,971,520 physical tokens divided by the
measured-window elapsed envelope (`max rank end - min rank start`). The cadence
median is a separate host interval between native `Profiler.step` returns;
the adapter does not synchronize each update, and its last interval includes
the end-boundary drain.

| Raw variant | Updates (warmup / measured) | Window elapsed (s) | Global physical tokens/s | Host cadence median (s) | Max rank active/allocated (GiB) | Max rank reserved (GiB) | Retry / OOM deltas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D0 | 10 / 10 | 123.422303 | 169,916.777 | 12.342949 | 65.884018 | 67.580078 | 0 / 0 |
| S | 10 / 10 | 122.424281 | 171,301.966 | 12.237741 | 65.889649 | 67.560547 | 0 / 0 |
| SE, historical four-stage | 10 / 10 | 124.570606 | 168,350.469 | 12.329494 | 65.893264 | 67.474609 | 0 / 0 |

The separate fixed-order comparisons are D0→S **+0.815% TPS**, S→historical
SE **−1.723%**, and D0→historical SE **−0.922%**. Host cadence median changes
were −0.852%, +0.750%, and −0.109%, respectively. This is one matrix without
an A/A control or repeat; these are provisional observations, not a general or
causal speedup claim. All 160 rank-step loss records per run were finite.

## Cache and allocator findings

All variants reused the saved D cache at
`runs/production-data-abcd/artifacts/raw/proddata-d-timing-001/cache`. D0 and S
reported no changed compiler-observer keys across ranks. Historical SE reported
Triton key changes for
`torchtitan.distributed.gfsdp._row_owner_embedding_kernels:_routing_metadata_kernel`
on ranks 1, 2, 3, 5, and 7. The changed constexpr shapes are preserved by rank
in the JSON and exact keys remain in each run's `summary.json`. The observer
cannot tell whether a key was loaded from disk or compiled, or when it appeared
within the run. Coverage is partial; asynchronous compiler-worker completion
and some native DeepGEMM JIT paths remain unverified. The run record does not
establish a JIT event.

Peak active/allocated memory varied by less than 0.010 GiB across the three
runs; all ranks reported zero allocator retries and OOMs. Per-rank byte and
GiB peaks are in the JSON and native summaries.

The one-update D0 and historical SE inventory smokes also passed on all ranks.
Each had seven backing storages and 6,973,565,952 unique communication-pool
bytes per rank. The two intended view remaps were `layers.7` dense data and the
norm gradient input; all untargeted views and contexts matched. Smoke losses
(15.32930 D0, 15.32932 SE) are finite observations, not parity evidence.

## Terminal state

Both original campaign sessions exited 0, all three timing run receipts are
PASS, all child process groups are stopped with no orphans, and the timing
campaign receipt is `campaign_complete PASS`. The failed initial smoke wrapper
attempt has an empty log and created no native run; the later foreground smoke
session completed both runs successfully. Root's latest process reconciliation
found no training/campaign/torchrun processes and all eight H100s at 0 MiB with
no compute applications.

This closes only the historical smoke and short-window matrix. No later
40-warmup + 100-measured matrix or update-21 profile was launched or resumed by
this execution owner.
