# V2 CPU prefetch group review

The 22 November 2026 unprofiled 10+10 comparison measured **+2.39% TPS**
and **0.00 GiB** change in peak total GPU memory on EP8/eFSDP1. On
EP4/eFSDP2 it measured **+1.39% TPS** and **+0.04 GiB**. The comparison
used the `prefetch-fix` branch. [Final 10+10 report][report]

## Source branch and tested variants

The local branch `prefetch-fix` is checked out at `.worktrees/prefetch-fix`.
The corresponding worktree on `sr008-gpu1` is
`/workspace-SR008.nfs2/vkropotin/pf-fix`. Both used compiled Original A
at `e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f`, with the same clean
GIFT tree as `linear-fixes` (`0f44b971069af55d6d573628129900483ca3e1dc`).
Original A already contains V2. Baseline A used
`dataloader.num_prefetch_batches: 0`, the loader default. The candidate
set it to `32` through the launcher. [Loader default][default] · [Launch adapter][runner]

## Change under review

| Change and switch | Intended effect |
| --- | --- |
| V2 CPU batch prefetch · `dataloader.num_prefetch_batches` | Move CPU batch materialization ahead of consumption. A producer thread on each rank fills a queue of up to 32 ready batches while the trainer works through previous batches, reducing the host wait for the next batch. The trainer consumes batches in order. [Streaming loader][loader] · [Producer thread][producer] |

After warmup, all eight ranks in each candidate arm had a live producer
and 32 ready batches. The baseline receipts showed no prefetch queue.
An earlier eight-rank CPU/Gloo replay on the same `data-002` input
matched sample order, inputs, labels and loader positions at depths 0
and 32. [All-rank run receipts][receipts] · [CPU/Gloo loader replay][loader-replay]

## Final 10+10 on compiled Original A

Four unprofiled runs used the same eight H100 80 GB GPUs on `sr008-gpu1`.
They used the eight-layer 100B crop, production-tokenizer `data-002` with
a 102,400-row epoch, seed 2048, local batch 2, sequence length 4096 and
global batch 512. All runs preserved FP8, compile, selective activation
checkpointing, MTP, EMA, native FLA, original HybridEP routing and cross-entropy,
with dense FSDP8 and PP/TP/CP1. Each arm started with a fresh compiler cache.
EP8 ran depth 0 first and EP4 ran depth 32 first. [Campaign log][progress]

Global TPS uses the 20,971,520 physical input tokens divided by the drained
all-rank elapsed time for updates 11–20. Median update TPS uses the median
of summed per-rank console TPS over the same ten updates. Peak total GPU
memory is the maximum native GIFT reading across those updates and all eight
ranks. It combines peak PyTorch reserved memory since the prior report and
non-PyTorch use at report time, rounded to 0.01 GiB.
[Measurement protocol][protocol] · [GIFT memory calculation][memory]

| Topology | Arm | Depth | Global TPS | Median update TPS | TPS Δ | Peak total GPU memory, GiB | Memory Δ, GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EP8 / eFSDP1 | `pf001-e8-off` | 0 | 153,865.28 | 154,280.5 | — | 72.38 | — |
| EP8 / eFSDP1 | `pf001-e8-on` | 32 | 157,542.37 | 158,000.0 | **+2.39%** | 72.38 | **0.00** |
| EP4 / eFSDP2 | `pf001-e4-off` | 0 | 144,670.20 | 144,768.0 | — | 76.00 | — |
| EP4 / eFSDP2 | `pf001-e4-on` | 32 | 146,687.23 | 146,646.5 | **+1.39%** | 76.04 | **+0.04** |

All four arms completed 20 updates on every rank with no epoch rollover.
The measured windows had zero CUDA allocator retries and OOMs. The compiler
observer reported stable observed state in all four arms, with partial
coverage of asynchronous compilation. One pair per topology leaves
run-to-run variation unmeasured. Raw logs and resolved configs are under
`/workspace-SR008.nfs2/vkropotin/pf-fix/runs/prefetch-fix/artifacts/raw`.
[Final report and limitations][report] · [All-rank run receipts][receipts]

## Earlier experiments

| Experiment | Observation |
| --- | --- |
| Long-epoch unprofiled 20+40 | Depth 32 measured **+2.12% TPS**. The historical allocator peaks rose by 0.021 GiB allocated and 0.039 GiB reserved. [20+40 pair][long] |
| Short-epoch unprofiled 10+10 | Depth 32 measured **−2.08% TPS**. Both runs crossed an epoch boundary in update 16. The candidate had a long update at that boundary, with the cause unresolved. [Short-epoch pair][short] |
| Long-epoch Perfetto diagnostic | Rank-0 time from input entry to first forward/backward fell from 675.8 to 8.4 ms. The sampled three-microbatch device span tracked closely at 1,294.5 and 1,293.7 ms. These measurements describe the diagnostic window. [Perfetto comparison][trace] |

[report]: ../../.worktrees/prefetch-fix/runs/prefetch-fix/report.md
[default]: ../../.worktrees/prefetch-fix/gift/torchtitan/config/schema/datasets/streaming/core/dataloader.py#L21
[runner]: ../../.worktrees/prefetch-fix/runs/prefetch-fix/run.py
[loader]: ../../.worktrees/prefetch-fix/gift/torchtitan/datasets/streaming/core/dataloader.py#L181
[producer]: ../../.worktrees/prefetch-fix/gift/torchtitan/datasets/streaming/core/prefetch.py#L17
[receipts]: ../../.worktrees/prefetch-fix/runs/prefetch-fix/results/pf001.json.gz
[loader-replay]: ../../runs/production-data-abcd/data-002-loader-summary.json
[progress]: ../../.worktrees/prefetch-fix/runs/prefetch-fix/results/pf001-progress.tsv
[protocol]: ../../planning/validation.md#memory-and-final-approval
[memory]: ../../.worktrees/prefetch-fix/gift/torchtitan/tools/utils.py#L278
[long]: ../../runs/v2-prefetch-baseline/report.md#matched-tps-results
[short]: ../../runs/v2-prefetch-baseline/report.md#fresh-matched-native-tps-pair
[trace]: ../../runs/v2-prefetch-baseline/report.md#mcp-perfetto-comparison-and-acceptance
