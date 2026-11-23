# Expert GFSDP on original A

`run.py` uses the original A eight-H100 streaming recipe with native FLA,
original HybridEP routing and cross-entropy, prefetch depth 0, and the verified
`data-002` dataset and production tokenizer. It retains FP8, compilation,
selective activation checkpointing, EMA, MTP, the LR schedule and batch settings.
Each arm runs 10 warmup updates followed by 10 measured updates.

Run with the verified image runtime and an existing completed original A
attempt (`--reference`) that contains metadata, completion, and the resolved
configuration. Use short physical checkout paths so the original runner's
task-owned temporary directory fits the local Unix socket path limit.

```bash
python -B runs/efsdp-optimizations/run.py run a-ep8 --ep 8 \
  --reference "$reference" --original-a-source "$clean_a_checkout"
python -B runs/efsdp-optimizations/run.py run all-ep8 --ep 8 \
  --reference "$reference" --direct-dw-build "$verified_build" \
  --optimization bf16-cache --optimization fp8-cache \
  --optimization direct-dw --optimization tail-retention
```

Use `--ep 4` and fresh run IDs for EP4/eFSDP2. Omit `--original-a-source`
for an all-off control on the candidate source. Supply individual
`--optimization` options for standalone arms. When both cache options are
enabled, FP8 replaces BF16 in eligible routed groups. See the
[direct dW build and component check](direct-dw/README.md). The launcher uses
the original reference's resolved DeepGEMM SM budget. `plan` checks the bridge
and native measurement CLI without starting distributed execution.

Each attempt writes its run artifacts under
`runs/overnight-streaming-perf/artifacts/raw/<run-id>` in the selected source
checkout. Use the same compiler-cache policy for matched arms. The default is
a fresh per-arm cache. `expert-gfsdp-summary.json` reports global throughput
and the maximum native `memory: … GiB` entry across ranks 0–7 and updates 11–20.
It retains per-rank activation and console records, allocator diagnostics,
source and configuration hashes, and compiler-window observations. For an
existing attempt, `summarize <run-id>` reprocesses the native logs. Use the
same `--original-a-source` argument for a clean-source attempt.
