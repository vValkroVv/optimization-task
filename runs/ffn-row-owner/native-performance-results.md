# Native performance results

These are historical test-tokenizer/low-diversity fixture measurements. The
[global input requirement](../../docs/baseline.md#global-requirement-real-data-and-production-tokenizer)
requires remeasurement on real, diverse data with the production tokenizer.
The [completed production-data rerun](../production-data-abcd/report.md#native-abcd-timing)
supplies the current cumulative comparison; retain this table as historical.

**Result:** The authorized eight-rank A/B pair and bounded diagnostic passed. In this single fixed-order pair, candidate B recorded 166,825.08 global physical tokens/s versus 164,737.68 for A (+1.27%); valid/supervised throughput differed by +1.27%. This is a provisional observation, not a general or causal speedup. Both windows show late cadence drift, and there is no A/A or repeat estimate.

## Protocol

A used the accepted baseline; B added all three controls together. Runs were serial, A first, with 10 warmup and 10 measured optimizer updates and 32 microbatches per update. Each run used fresh seed 2048 initialization, logical epoch 102,400, and the unchanged 65,000/2,000 LR horizon/warmup. FP8/BF16/FP32, model/loss compile, selective activation checkpointing, MTP, fused AdamW and EMA stayed enabled. Profiling, detailed monitoring, validation, checkpoint saves, memory snapshots and numerical capture were off for timing.

The separate diagnostic ran 21 updates, warming up 20 and profiling update 21 on ranks 0/1 for at most three microbatches. Every rank completed all 32 microbatches and the optimizer update. Exact argv arrays are in [native-plan.json](native-plan.json); per-run resolved configs, metadata and summaries are linked from the [JSON results](native-performance-results.json).

## A/B results

| Variant | Global physical input tokens/s | Global valid/supervised tokens/s | 10-update window | Max allocated/active | Max reserved | Median host cadence |
|---|---:|---:|---:|---:|---:|---:|
| A | 164,737.68 | 164,697.46 | 127.303 s | 67.662 GiB | 68.141 GiB | 12.708020 s |
| B | 166,825.08 | 166,784.35 | 125.710 s | 66.008 GiB | 67.689 GiB | 12.497278 s |

Each window processed 20,971,520 physical tokens and 20,966,400 valid/supervised tokens. The synchronized global window envelope was 127.303 s for A and 125.710 s for B. B/A differences are ratios of these two observed window rates, not paired statistics. Full measured host cadence for every rank and update is in each raw [A summary](artifacts/raw/ffn-row-owner-ab-a-001/summary.json) and [B summary](artifacts/raw/ffn-row-owner-ab-b-001/summary.json); full warmup cadence is in the linked per-rank measurement records. The cadence scope is host intervals between native Profiler.step returns and is separate from window elapsed.

Rank-0 measured intervals:

| Update | A cadence (s) | B cadence (s) |
|---:|---:|---:|
| 11 | 12.606650 | 12.448897 |
| 12 | 12.601711 | 12.477361 |
| 13 | 12.609263 | 12.452206 |
| 14 | 12.719108 | 12.454920 |
| 15 | 12.647065 | 12.480054 |
| 16 | 12.695762 | 12.514450 |
| 17 | 12.735674 | 12.585807 |
| 18 | 12.797581 | 12.739850 |
| 19 | 12.872310 | 12.734335 |
| 20 | 13.016546 | 12.820990 |

Rank 0 rises from 12.606650 to 13.016546 s for A, and from 12.448897 to 12.820990 s for B. Compilation observation was disabled by the accepted low-instrumentation recipe, so continued compilation within the measured window is unverified. No warmup extension or repeat was added.

All eight ranks completed every requested update, 32 forward/backward calls and one optimizer step per update. Summaries report no errors; source/config stayed unchanged, process groups exited cleanly, and retry/OOM deltas were zero. Per-rank allocated/reserved peaks are in the JSON. These are allocator-window measurements, not device-wide or process-lifetime peaks.

## Activation and trace

A resolved all three new fields false; B and the diagnostic resolved them true. Runtime setup INFO text was absent, so activation uses the same-source all-eight-rank materialized inventory and actual profile execution. The inventory is in [native-memory-results.json](native-memory-results.json); the combined MCP mechanism result is in [native-trace-results.md](native-trace-results.md).

The update-21 parser passed bounded trace coverage on ranks 0/1. Canonical gzip traces are lossless copies of the raw trace files:

- Rank 0 raw SHA256: 27415d3cdc02550fabc87438c8a580a75068d695db12adb49448547c94004704; gzip SHA256: 213a6abc63b9d6afcec8902fa1f1264db81de3f6cdf08d61c51ea248382b130b.
- Rank 1 raw SHA256: 55722f98b0d97ccbfcb7b831674c54b23c9ed953a003b4832fa64f1340bb355b; gzip SHA256: d15dbcbf0b9a7980f4e36ca28f049e240291198e6925d4f4322187cf3881c779.

MCP confirms all 12 FFN rank/microbatch/component cases had the 513 MiB reduction before parent reduction and 96.20–98.32% overlap with remaining parent compute. Owner lookup and backward run once per microbatch with routed exchanges; embedding-table GFSDP collectives are absent, while the untied lm_head full-table collectives remain. Four natural cross-microbatch head-buffer reuse boundaries have positive gaps in both directions. Exact event-wait causality and the full-update critical path remain unavailable; see the trace report for scope.

## Provenance and limits

Source HEAD is 2ada87125e433e1f76c4cb2bfad244b641a78a93; the unchanged tracked-plus-untracked source manifest is cd9102008dd19819b39c2ee9ce5820565b0d2a0e14783dfb33122691073c41b1. The canonical experiment hash is 121a5f11a4b431cb632badc08f9ccf718fc4fd5640c68c9279be57921c012479, MDS index hash is 7cc8d53152c801450fdae44ba84c182eb8dcb9106d8bba9e3c3d69ebdaa8a8c2, and runtime preflight hash is 756b18c7bbee8133dc5b27579ce426b774ac4ba62173ce091a9554bdc1664d6f. A/B/diagnostic all selected the same completed cache and producer; the producer metadata/completion/config hashes are recorded in the JSON. Runtime was eight H100 80GB GPUs with Python 3.12.3, Torch 2.15.0.dev20260822+cu130, CUDA 13.0 and NCCL 2.30.7.

The separately accepted two-update memory gate recorded max allocated A/B values of 72,195,010,048 / 70,708,547,072 bytes and reserved values of 72,764,882,944 / 72,364,326,912 bytes; the candidate pool shrank by 1,746,392,064 bytes per rank. First-update reserved premiums against A002 were 127,926,272–253,755,392 bytes (122–242 MiB); the separate warm comparison had rank 2 +60 MiB versus its own control. These are phase observations, not new caps or tolerances, and allocator cause is unproven; see [native-memory-results.json](native-memory-results.json).

This result is one fixed-order pair only. It does not establish a general speedup, convergence, long-run stability or production-scale behavior. Earlier runner-harness attempts remain retained in [native memory results](native-memory-results.md); no timing A/B or update-21 diagnostic run failed. The raw launch logs and per-rank records remain under [artifacts/raw](artifacts/raw/).
