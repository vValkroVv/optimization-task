# V2 CPU batch prefetch on Original A

## 10+10 results

Depth 32 raised measured global TPS by 2.39% on EP8 and 1.39% on EP4.
Peak total GPU memory matched to console precision on EP8 and rose by
0.04 GiB on EP4.

| Expert topology | TPS, depth 0 | TPS, depth 32 | TPS change | GPU GiB, depth 0 | GPU GiB, depth 32 | GPU GiB change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EP8 / eFSDP1 | 153,865 | 157,542 | +2.39% | 72.38 | 72.38 | 0.00 |
| EP4 / eFSDP2 | 144,670 | 146,687 | +1.39% | 76.00 | 76.04 | +0.04 |

| Expert topology | Timed seconds, depth 0 | Timed seconds, depth 32 | Change in median per-update TPS | Peak rank, depth 0 / 32 |
| --- | ---: | ---: | ---: | ---: |
| EP8 / eFSDP1 | 136.298 | 133.117 | +2.41% | 0 / 0 |
| EP4 / eFSDP2 | 144.961 | 142.968 | +1.30% | 0 / 3 |

All four arms completed on November 22, 2026. The GPU UUIDs matched at
every start. All ranks logged each optimizer update. At the measured
boundary, all eight depth-32 queues held 32 ready batches and had a live
producer. Depth-0 receipts recorded the default iterator. Each arm
accounted for 20,971,520 input tokens and 20,966,400 valid targets in the
timed window, with the same epoch throughout. The allocator reported
zero allocation retries and OOMs in the measured windows. The compiler
observer reported stable observed state in all four arms, with partial
coverage of asynchronous compilation. [Campaign log](results/pf001-progress.tsv)
· [All-rank receipts](results/pf001.json.gz)

The result represents one pair per topology. These four runs estimate
the effect on this crop and leave run-to-run variability unmeasured.
Rounded console losses differed by up to 0.00028 on EP8 and 0.00032 on
EP4. The corresponding maximum gradient-norm differences were 0.0201
and 0.0123. The earlier long-epoch 20+40 run recorded +2.12% TPS, while
the earlier short-epoch 10+10 result included an epoch rollover and
recorded −2.08%. [Historical V2 report](../../../../runs/v2-prefetch-baseline/report.md)

## Configuration and measurement

The `prefetch-fix` worktree uses Original A, revision
`e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f`, with the same clean GIFT tree as
`linear-fixes` (`0f44b971069af55d6d573628129900483ca3e1dc`). At depth 32, a
producer thread on each rank prepares up to 32 CPU batches ahead of the
trainer, which can then consume a ready batch while the producer continues
working. Depth 0 uses the default iterator. [Streaming dataloader](../../gift/torchtitan/datasets/streaming/core/dataloader.py#L181)
· [Producer thread](../../gift/torchtitan/datasets/streaming/core/prefetch.py#L17)

The four unprofiled arms use the same eight H100 80 GB GPUs on
`sr008-gpu1`, with dense FSDP8, DP replication 1, PP/TP/CP1, and the
Original A production batch of 512 sequences at length 4,096. The pin
includes the same 102,400-row `data-002` epoch, seed, and original FLA,
routing, and cross-entropy paths. The native resolved-config comparison
allows only expert-parallel degree, prefetch depth, and output or dataset
paths to vary from the verified Original A reference. Each arm starts
from a fresh compiler cache. Execution order is EP8 depth 0 then 32,
followed by EP4 depth 32 then 0. The runtime used Python 3.12.3,
PyTorch `2.15.0.dev20260822+cu130`, CUDA 13.0, and NCCL 2.30.7.
[Launch adapter](run.py) · [Matrix driver](matrix.sh)

Global input TPS is 20,971,520 tokens divided by the drained elapsed time
across all ranks for updates 11–20. Peak total GPU memory is the largest
native GIFT estimate over those ten updates and all eight ranks. GIFT
adds PyTorch's peak reserved memory since the previous log to non-PyTorch
usage at log time. The console rounds to 0.01 GiB, and brief non-PyTorch
spikes between logs can fall outside this estimate. [Collector](collect.py)
· [Native memory calculation](../../gift/torchtitan/tools/utils.py#L278)

Raw native logs, resolved configurations, environment receipts, and GPU
preflights are on `sr008-gpu1` under
`/workspace-SR008.nfs2/vkropotin/pf-fix/runs/prefetch-fix/artifacts/raw`.
The four run IDs are `pf001-e8-off`, `pf001-e8-on`, `pf001-e4-on`,
and `pf001-e4-off`. The collector verifies source and config identity,
rank completion, measured windows, and native memory records before
computing the deltas.
