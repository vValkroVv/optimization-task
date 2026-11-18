# Row-owner embedding component results

All five authorized eight-rank component stages completed successfully. The
[JSON result](component-results.json) preserves compact rank-max sample
distributions, per-rank medians and peaks, allocator deltas, route distributions,
source/cache/input/runtime identities, profile summaries, trace hashes, launch
arguments, logs and PIDs. It points to each exact all-ranks report by path and
SHA256; those raw reports retain every rank's full warmup and sample series. The
raw traces remain under
`artifacts/raw/row-owner-components/`.

## Measurement conditions

- One fresh process group per stage, run serially on eight H100 80 GB GPUs.
- Inputs are the eight distinct signed-int64 B2×4096 tensors captured from the
  corrected production-data native run. Each rank processes 8,192 tokens with
  the same fixture in every stage.
- Python 3.12, PyTorch 2.15.0.dev20260822+cu130, CUDA 13.0,
  NCCL 2.30.7. All stages use the
  validated D compiler cache at `runs/production-data-abcd/artifacts/raw/proddata-d-timing-001/cache`.
- Ten warmups and ten unprofiled samples for forward, backward and integrated
  forward/backward. The separate profiler pass follows timing.
- CUDA-event spans join compute, all-gather and reduce-scatter streams and include
  queue gaps/waits. Host call, endpoint synchronization and synchronized wall
  are reported as separate, overlapping scopes. They must not be added together.
  Rank-max medians below take each sample's maximum across ranks, then the median
  of those ten maxima. Rank-local medians and rank-max ten-sample series are in
  the JSON; the referenced raw reports retain the full per-rank sample arrays.
- Gradient clearing/barriers are outside the timed calls; absolute allocator
  peaks include resident GFSDP storage. Forward/backward component calls include
  the embedding communication but not full `lm_head` compute. There is no TPS
  result.

## Results

Times are milliseconds. Kernel counts are **local + NCCL per rank** from the
separate profiler pass. Copies and memsets are shown forward/backward. Memory is
the integrated call's maximum across ranks in GiB (1 GiB = 2³⁰ bytes).

| Prefix | CUDA-event F / B / integrated | Integrated host call / end sync / wall | Profile kernels F / B | Integrated peak allocated / reserved | Copies F/B | Memsets F/B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| d0 | 1.395 / 1.703 / 2.722 | 2.309 / 0.509 / 2.739 | 31+3 / 35+1 | 5.52 / 17.64 | 6/0 | 20/10 |
| d1-compact | 1.255 / 1.541 / 2.479 | 2.423 / 0.089 / 2.496 | 31+3 / 33+1 | 4.49 / 15.88 | 6/0 | 20/8 |
| d2-compact-owner_add | 1.212 / 1.051 / 2.100 | 1.766 / 0.361 / 2.118 | 31+3 / 25+1 | 4.49 / 15.88 | 6/0 | 20/8 |
| d3-compact-owner_add-owner_lookup | 1.147 / 1.070 / 2.048 | 1.828 / 0.250 / 2.070 | 30+3 / 25+1 | 4.49 / 15.88 | 6/0 | 20/8 |
| d4-compact-owner_add-owner_lookup-routing | 1.032 / 1.051 / 1.915 | 1.691 / 0.304 / 1.935 | 22+3 / 25+1 | 4.49 / 15.88 | 4/0 | 20/8 |

The compact stage replaced the 1,050,673,152-byte full-vocabulary BF16 table
with compact native outputs sized to each rank's unique input rows; the largest
observed output was 26,279,936 bytes. For the representative rank-0 capture
(N=8,192, U=2,141, D=4,096), the pinned source-derived FP32 partial tensor sizes
are 48,496,640 bytes for compact dispatch and 147,636,224 bytes for full-vocabulary
dispatch ([source-derived sizing](embedding-implementation.md#2a-compact-source-local-coalescing)).
These are tensor-size calculations, not measured allocations or peak memory.
The exact internal allocation and lifetime remain uninstrumented; the harness
retains `native_fp32_partial_scratch_bytes: null`, and allocator peaks are the
primary measured memory result. D0 integrated peak allocated/reserved was
5,929,826,816/18,941,476,864 bytes (5.52/17.64 GiB); D1–D4 were
4,824,576,512/17,049,845,760 bytes (4.49/15.88 GiB). The raw JSON retains every rank's
forward, backward and integrated peak; referenced all-rank reports retain the
per-sample allocation-retry/OOM deltas, all of which were zero. The owner scratch view remained aliased
to the private `lm_head` GFSDP storage, with one unique gradient storage per
rank in every stage.

Profiled device counts changed as the prefixes were added: forward kernels per
rank were 34, 34, 34, 33 and 25 (NCCL 3 in every stage); backward kernels were
36, 34, 26, 26 and 26 (NCCL 1 in every stage). D4 also reduced forward device
copies from 6 to 4. Per-rank profile traces, interval unions, first-to-last
spans and stage observer host calls are retained. The harness does not emit an
independent host-gap attribution.

## Staged tradeoffs

- `d0 → d1-compact`: rank-max median CUDA spans F/B/integrated 1.395→1.255 / 1.703→1.541 / 2.722→2.479 ms; integrated peak allocated/reserved changed by -1,105,250,304/-1,891,631,104 bytes.
- `d1-compact → d2-compact-owner_add`: rank-max median CUDA spans F/B/integrated 1.255→1.212 / 1.541→1.051 / 2.479→2.100 ms; integrated peak allocated/reserved changed by +0/+0 bytes.
- `d2-compact-owner_add → d3-compact-owner_add-owner_lookup`: rank-max median CUDA spans F/B/integrated 1.212→1.147 / 1.051→1.070 / 2.100→2.048 ms; integrated peak allocated/reserved changed by +0/+0 bytes.
- `d3-compact-owner_add-owner_lookup → d4-compact-owner_add-owner_lookup-routing`: rank-max median CUDA spans F/B/integrated 1.147→1.032 / 1.070→1.051 / 2.048→1.915 ms; integrated peak allocated/reserved changed by +0/+0 bytes.

The measured component tradeoffs provisionally support carrying all four
optimization stages into the integrated native smoke: compact coalescing lowers
the measured backward/integrated allocation peak; owner-add reduces the backward
kernel count; owner lookup and routing reduce forward kernel counts, with the
last stage also removing two copies. D2–D4 did not increase the measured peak
relative to D1. D3's backward span was slightly above D2's; the staged median is
not enough to establish a regression or gain. These are single fixed-order
observations without A/A noise, repeated variants or reversed order. They do
not establish a causal component speedup, full-update throughput or general TPS.

## Warmup, warnings and limits

The first instrumented warmup was a multi-second outlier in every stage; later
warmups were near the measured millisecond range. The JSON retains each rank's
first, later-warmup median and last warmup; referenced raw reports retain the
complete per-rank warmup and sample series. Launcher logs contain
Quack `.o`-loader/cache fallback messages in `d0`: 8 messages, `d1-compact`: 8 messages; the fallback selected
in-process compilation without disk cache. The same validated D cache paths were
used for all stages. Logs also retain the nonfatal c10d address-family warnings;
all ranks completed successfully.

D0 source revision was `e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f`. Source hashes were unchanged within
and across all five stages. No component source or harness edits were made during
execution. The GPU query after D4 reported zero memory use on every H100 and no
compute-app PIDs; the allocation is released to the parent.
