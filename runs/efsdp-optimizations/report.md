# Expert GFSDP on original A

The `expert-gfsdp-fix` worktrees start from `e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f`.
The starting GIFT tree is `0f44b971069af55d6d573628129900483ca3e1dc`, identical to
`linear-fixes`. The acceptance controls run clean GIFT at that revision. The
[launcher](README.md) selects original A's FLA, routing metadata, cross-entropy,
and input-prefetch settings for the controls and candidates. Both topologies
use eight H100s, the production-tokenizer `data-002` stream, the eight-layer
crop, dense FSDP8, PP/TP/CP1, global batch 512, microbatch 2, sequence length
4,096 and accumulation 32. The four acceptance runs are unprofiled and measure
updates 11–20. The memory metric is the maximum native total GPU memory across
all eight ranks in that window.

## Historical isolated evidence

The EP8/eFSDP1 series at `c401b122d676cd8cfe24df930aa2563ed71ec791` used the
later fixed recipe with BF16 reuse already enabled in the control. The compact
receipts cover 32 rank receipts and four resolved configs. The archived SR008
console logs contain exactly one sample for each rank and update in 11–20.
The native memory maximum is rank 4, update 20 in all four runs.

| Arm | Global physical tokens/s | TPS change | Peak native total GPU memory | Allocated / reserved changes |
| --- | ---: | ---: | ---: | ---: |
| BF16 control | 171,205.53 | | 73.20 GiB | |
| FP8 cache | 173,472.33 | +1.324% | 69.72 GiB | −3.359 / −3.477 GiB |
| Direct expert dW | 174,185.79 | +1.741% | 73.20 GiB | +0.008 / −0.004 GiB |
| Tail retention | 172,809.33 | +0.937% | 73.20 GiB | +0.011 / 0.000 GiB |

The four runs are one fixed-order series. The BF16 baseline, direct dW and tail
arms had seven active BF16 groups per rank. The FP8 arm used seven FP8 cache
groups per rank. The separate historical BF16 comparison measured 143,736.69
tokens/s and 74.49 GiB against saved V2 at 140,933.05 tokens/s and 74.55 GiB.
Both have all 320 native samples for updates 21–60. They have an older tokenizer
and different source and capture settings, so the timing difference is provisional.

| Fix | Observed historical mechanism |
| --- | --- |
| BF16 reuse | FP32-to-BF16 casts fell from 21 to 7 and singleton weight copies from 42 to 0 over three microbatches per rank. FP8 weight quantization stayed at 78 launches. |
| FP8 cache | Forward and recompute weight quantization fell from 26 launches per microbatch to 14 on the first and zero on the next two. The EP8 path aliases the local E4M3 payload and FP32 scale planes. |
| Direct expert dW | The trace recorded 14 rounded FP32 GEMMs per microbatch and removed all 21 expert gradient pack operations. The native component compared FP32 slot bits against zeroed BF16 TMA reduce-add followed by BF16-to-FP32 copy. Earlier whole-run pairs were sensitive to the choice of control. |
| Tail retention | Logical expert unshards fell from 14 to 12 per microbatch at eFSDP1. The trace also removed both inter-callback dense MTP gathers. The dense result is specific to that later fixed recipe. |

Archived raw run roots are on `sr008-gpu1` under the original task repository.
For the four later runs, use
`runs/kernel-fusion/artifacts/raw/efsdp-four-{baseline,fp8,direct,tail}-ep8-001-timing`.
For BF16 use `runs/overnight-streaming-perf/artifacts/raw/bf16-cache-tps-001` and
`runs/v2-prefetch-baseline/artifacts/raw/v2-long-epoch-candidate-perf-001`.
The console reader uses each run's `logs/torchrun/*/attempt_0/{0..7}/stdout.log`.
The timing and allocator sources are
`runs/efsdp-optimizations/four-standalone-receipts.json` on the task's main branch
and this branch's [BF16 receipts](../bf16-expert-cache/results.json).

## Multi-rank design

The previous code below refers to the original A revision recorded above. A
already included the disabled singleton BF16 implementation.

The original BF16 cache required one rank in the all-gather group. The
direct dW donor `c68cc9805` required EP8 and one rank in the reduce-scatter
group. The FP8 donor `5b9ae4f15` already had typed multi-rank gathers
for aligned shard boundaries. Tail retention in `a0602a887` used the
backward state and the GFSDP backend, with no shard-count check. The saved
individual performance series exercised EP8/eFSDP1 on all four paths.

### BF16 expert weight reuse

Original A, `_collectives.py:47–53`

```python
direct_cache_use = (
    param_group._bf16_cache_enabled and all_gather_dtype is None
)
if direct_cache_use and all_gather_output is not all_gather_input:
    raise AssertionError(
        "An enabled BF16 expert cache must use its input as output"
    )
```

The ordinary path ran `all_gather_input.copy_(padded_sharded_param_data)`
and cleared `_all_gather_input_set_dtypes` on reshard. The old cache coupled
reuse to input-output aliasing, which requires one rank to own the full group.
On multiple ranks, reuse the local cast until the next weight generation and
let the native collective populate the shared full-weight buffer on each
unshard. The singleton avoids both the collective and the slot copy.
Optimizer steps, DCP, EMA, and master replacement invalidate the cache.
CUDA events order refreshes, collective and compute readers, and writers.
[Gather](../../gift/torchtitan/distributed/gfsdp/_collectives.py) ·
[Lifetime and invalidation](../../gift/torchtitan/distributed/gfsdp/_param_group.py)

### FP8 expert weight cache

Original A, `GroupedExperts._quantize_local_expert_weights()` in `gmm.py:168`

```python
return [quantize_expert_weight(w) for w in self._local_expert_weights()]
```

This ran after BF16 materialization. Repeated consumers prepared the same
weight values and read the large BF16 representation. The cache prepares the
E4M3 bytes and FP32 scales once per generation. Multiple ranks gather the
prepared planes, while a singleton aliases them. Quantization still sees the value of the
original FP32 master rounded to BF16. The grouped GEMM resolves the shared
view while its slot is live, including backward through SAC recomputation.
All rank boundaries must preserve the rank-major 128-by-128 tile layout.
Incompatible groups use the existing BF16 route.
[Consumer](../../gift/torchtitan/models/common/moe/experts/gmm.py) ·
[Layout and gather](../../gift/torchtitan/distributed/gfsdp/_fp8_expert_weight_cache.py)

### Direct expert dW

Original A, `_GroupedGemm.backward()` in `grouped_gemm.py:222–224`

```python
wgrad = torch.zeros(
    ctx.weight_shape, dtype=torch.bfloat16, device=ctx.device
)
```

The old path passed `d=wgrad` to grouped DeepGEMM. GFSDP packed that
gradient with `torch._foreach_copy_` into the FP32 reduce-scatter input. The
direct epilogue applies the same BF16 rounding and zero-add semantics while
writing each projection into its range of that FP32 slot. Multiple ranks use
the existing FP32 reduction and retain the ordinary gradient accumulators.
The direct writer requires supported Hopper shapes and an unpadded native
slot. The current production integration targets PP/TP/CP1.
[Backward and empty routing](../../gift/torchtitan/ops/float8/grouped_gemm.py) ·
[Producer lease](../../gift/torchtitan/ops/float8/direct_expert_dw.py)

### Tail weight retention

Original A, root callback in `_state.py:699–703`

```python
if fsdp_param_group._training_state != TrainingState.POST_BACKWARD:
    # Run post-backward in case forward inputs did not require
    # gradient so the autograd backward did not run
    fsdp_param_group.post_backward()
fsdp_param_group._training_state = TrainingState.IDLE
```

On an intermediate split-head backward, that resharded groups which
had only prefetched for a later backward. The new callback defers nonfinal
cleanup of those groups. It still handles missing post-backward hooks for
participating groups and final cleanup. The same retention applies to a
singleton alias or a multi-rank gathered view with a pending-work fence.
[Callback](../../gift/torchtitan/distributed/gfsdp/_state.py) ·
[Shared slot release](../../gift/torchtitan/distributed/gfsdp/_param_group.py)

PyTorch FSDP2 at the deployed Torch revision
[`8b353065`](https://github.com/pytorch/pytorch/tree/8b353065ce260336c5fdfa74d754e8bd9b91b062/torch/distributed/fsdp/_fully_shard)
provides the stream, input lifetime and final-callback references. The task's
pinned LLM Foundry reference uses revision `eca85af5e254520722efdc3de1a7c9bb834d1a2a`
and GigaFSDP gitlink `27a1825817bec332a5e327ddcb23cabb122bb32e` for
reduced-precision preparation and weight views. The direct dW component uses
the [production DeepGEMM pin](direct-dw/README.md).

## Component verification

Local Torch `2.14.1` ran the two-rank Gloo BF16 test through real GFSDP.
Two updates with two microbatches each matched BF16 outputs and FP32 grad
shards exactly. The cache refreshed once per update, kept its local cast
separate from the shared receive buffer, and used no singleton aliases.
The initial focused suite passed 75 tests and skipped 11 CUDA or
Inductor cases. The expanded focused suite passed 78 tests and skipped 30
CUDA or Inductor cases in the local environment.

The four-rank BF16 probe used the same GFSDP Linear fixture with real Gloo
all-gather and reduce-scatter. On every rank, four BF16 forwards, four FP32
gradient shards and two master-shard updates matched bitwise. Each rank
refreshed once per update and gathered into a separate receive buffer. The
fixture used explicit write hooks. [Probe helper](../../../../artifacts/expert-bf16-gloo4.gD0Wqs/run.py) ·
[Rank receipts](../../../../artifacts/expert-bf16-gloo4.gD0Wqs/run.log)

The four-rank fixture also passed with the preset's fused AdamW and the
actual optimizer hooks on `RaggedShardDTensor`. The hooks advanced the
weight generation once per step. BF16 forwards, FP32 gradient shards,
master shards and AdamW moments matched bitwise. The CPU run verified Python
hook calls. CUDA event ordering remains a separate check. The initial
for-loop probe reached the pre-hook and failed at unsupported `aten.lerp_`
dispatch. The production recipe uses fused AdamW.
[Fused AdamW receipt](../../../../artifacts/expert-bf16-gloo4-adamw.QioRvE/fused-cpu-observed/run.log) ·
[For-loop failure](../../../../artifacts/expert-bf16-gloo4-adamw.QioRvE/for-loop.log)

A four-rank Gloo FP8 probe checked both receive planes bitwise across two
generations with synchronous and asynchronous gathers. It ran the
production cache and real typed collectives with a simulated quantizer and
CUDA events. Rank 2's local shard crossed the W13/W2 boundary.
[Four-rank FP8 probe](artifacts/raw/gloo-four-rank/fp8-typed-gather.md)

The two-rank H100 FP8 component passed eight cases on the production Torch
revision. It compared E4M3 bytes and FP32 scale bits with the legacy
FP32-to-BF16-to-FP8 path over two weight generations, including synchronous
and asynchronous typed all-gathers and a shard boundary inside W13. Cache
hits reused the prepared local representation. The native receipt is
`optimization-task/artifacts/expert-gfsdp-nov22-01/fp8-component.log` on
`sr008-gpu1`, with exit code zero in `fp8-component.exit`.

The same NCCL check passed on four H100 ranks using the crop's EP32/eFSDP4
expert geometry, with W13 `(4, 3072, 4096)` and W2 `(4, 4096, 1536)`.
Every rank matched the legacy BF16-to-FP8 data and scale bits through two
generations and both collective submission modes. This exercised the four-rank
expert weight component on one node.
[Four-rank native test](artifacts/raw/fp8-native-four-01/probe.py) ·
[Result](artifacts/raw/fp8-native-four-01/pytest.log)

The H100 direct dW component used Torch
`2.15.0.dev20260822+cu130` at `8b353065ce260336c5fdfa74d754e8bd9b91b062`
and the baseline's 116-SM DeepGEMM setting. Both projections passed bitwise
for 16 and 32 local experts with full, mixed, and all-empty routing, including
canaries and the 64-lane special-value probe. The preallocated-slot component
allocated 59,392 incremental bytes for full and mixed cases, compared with
604,039,168 legacy bytes at 16 experts and 1,208,018,944 at 32. This excludes the
existing FP32 slot, training activations, and collective buffers.
The archived build and component receipts are under
`optimization-task/artifacts/dw-Zi0UA8` on `sr008-gpu1`. The first build in
`dw-la3whB` failed during CUDA initialization in a raw SSH shell. The
training image's established `/etc/profile.d/wm.sh` initialization restored
CUDA and the pinned retry completed.

## Native acceptance

The EP8 acceptance pair uses `efg-n22-03`. The matched EP4 rerun uses
`efg-n22-04`. Both arms for each topology use fresh per-arm compiler caches.
The table reports physical global tokens/s over the measured window, the
median of the summed rank throughput per update, and full GPU peak memory
across ranks and updates.

| Topology and arm | Window TPS | Median update TPS | Peak total GPU memory | Peak rank / update |
| --- | ---: | ---: | ---: | --- |
| EP8/eFSDP1 original A | 156,284.18 | 156,336 | 72.30 GiB | 3 / 20 |
| EP8/eFSDP1 expert fixes | 162,750.45 | 162,664 | 68.25 GiB | 1 / 20 |
| EP4/eFSDP2 original A | 144,474.59 | 144,124 | 76.02 GiB | 0 / 20 |
| EP4/eFSDP2 expert fixes | 153,641.40 | 154,248 | 71.04 GiB | 6 / 20 |

The EP8 pair measured **+4.14% TPS** and **−4.05 GiB**. All eight ranks
confirmed active FP8 cache, direct dW and tail retention in all seven expert
groups. The FP8 path took precedence over BF16 reuse. The largest displayed
differences between the arms across updates 1–20 were 0.00024 for loss and
0.0158 for gradient norm. These are rounded training-log values, with
deterministic numerical parity still to be established at the model level.

The original A build repeats showed up to 0.00022 in displayed loss and
0.0071 in gradient norm across 20 updates on two fresh-cache runs. The same
clean A compared with the first 20 updates of a reused-cache 140-update A
run showed maxima of 0.00027 and 0.0112. These runs use seed 2048 and
`debug.deterministic=false`, with matching source and dataset bytes. The
optimized loss difference is within that observed variation. The optimized
gradient-norm maximum of 0.0158 occurred at warmup update 8. Its maximum
over measured updates 11–20 was 0.0019, compared with 0.0020 and 0.0025
for the two A/A comparisons over the same window.

The fresh EP4 pair measured **+6.34% TPS** and **−4.98 GiB**. Both arms ran
on the same eight device UUIDs after the allocation recovered. All 80
measured rank updates per arm were present. Each candidate rank confirmed
FP8 cache, direct dW and tail retention in all seven eFSDP2 groups. The FP8
cache took precedence over BF16 reuse. Maximum allocator allocated and
reserved memory fell by 4.78 and 4.98 GiB. The compiler observer showed
stability over its observed channels, with partial overall coverage.
The maximum displayed rank-0 loss difference was 0.00033. The maximum
gradient-norm difference was 0.0118 over all 20 updates and 0.0013 over the
measured 10 updates.
[Matched EP4 receipts](artifacts/raw/efg-n22-04-export/completed-ep4-receipts.json)

The unprofiled summaries are on `sr008-gpu1` under
`egfa/runs/overnight-streaming-perf/artifacts/raw/efg-n22-03-ep8-a/` and
`egf/runs/overnight-streaming-perf/artifacts/raw/efg-n22-03-ep8-expert/`.
Each contains `expert-gfsdp-summary.json` with rank metrics and activation.
The interrupted campaign's complete EP4 A run is `efg-n22-03-ep4-a` under
the same clean tree. The matched rerun's result directories are
`efg-n22-04-ep4-a` in `egfa` and `efg-n22-04-ep4-expert` in `egf`.
[First campaign receipts](artifacts/raw/efg-n22-03-export/completed-acceptance-receipts.json)

The candidate log reached update 12 in `efg-n22-03-ep4-expert` at 12:28 UTC.
The GPU SSH gateway then returned `target host connection failed`. The
documented `sr008` development host can read the matching shared NFS paths.
Its readback confirmed that the eight rank logs stopped updating and that
`attempt-03/acceptance.exit` is absent. ML Space reported the configured
allocation as `Pending`. This attempt had an incomplete optimized EP4 window.
Each of the eight rank logs records seven FP8 expert weight caches and
seven direct expert dW activations during native EP4 initialization.
At updates 11 and 12, the eight rank-reported throughputs summed to 154,736
and 154,528 tokens/s for the candidate, and 146,856 and 146,752 for A.
The largest reported total memory in those two updates was 71.09 GiB for
the candidate and 75.96 GiB for A. These are two-step diagnostics from the
interrupted campaign. The recovered allocation was used to rerun both EP4
arms with matched hardware.

## Follow-on experiments

### Native EP4 trace

The first profile attempt stopped before training because the adapter passed
`profile_ranks` into `WindowMeasurement`. The corrected adapter kept that
field for rank selection and completed all 32 microbatches of update 21.
It recorded the first three microbatches on ranks 0 and 4.
[Trace extraction](artifacts/raw/efg-n22-04-ep4-expert-trace2/extracted.json)

The six expert-gradient slot waits per traced microbatch delayed work on the
compute stream. Rank 0 showed about 26.95, 32.20 and 32.14 ms of those gaps.
Rank 4 showed 25.34, 32.79 and 32.10 ms. The two ranks observed the same
communication sequence. Each gap was counted once per rank. Previous expert
reduce-scatter completed just before the next compute kernel on the first
microbatch. Subsequent microbatches also waited for FP32 shard accumulation.
CUPTI lacked exact record-to-wait timestamps. Host launch order, same-stream
kernel order and native release placement constrain the exposed intervals.

On rank 0 in microbatch 2, the layer 6 reduce-scatter at sequence 19889
(correlation 1731209) was followed by the FP32 add (1731239). The layer 5
release wait (1731406) was followed by recompute (1731587), prepared-weight
transpose, dgrad and row-to-column conversion before the first direct dW
write (1733497). These kernels supply about 8–10 ms of work before the
slot's next write on the observed path. The first follow-up will move the
existing release dependency to that write or the empty-path zero, keeping
the same slot and FP32 reduction. The residual waits will determine the
priority of a separate owner-transport trial.

The trace also verified the first two FP8 forward gathers in microbatches
2 and 3 reread the last backward writers of the two shared slots. The four
logical repeats moved 1.125 GiB per rank in each direction. The gather planes
had already finished at least 2.775 ms before the preceding compute kernel
ended at those consumers. The trace supports removable traffic. It showed
no directly exposed gather-kernel time for those four repeats. Defer that
reuse experiment unless contention or another topology exposes the cost.

### Deferred expert-gradient slot wait

The overlay keeps the original slot owner claim and captures the prior
release events. It waits on those events on the lease stream immediately
before the first DeepGEMM or empty-path zero write. Earlier work can execute
while the previous reduction reads the slot. The buffer count, dtype and
reduction path stay the same.

The first native fixture could not demonstrate independent work arriving
before its delayed NCCL reader. The corrected fixture used warmed explicit
streams and a no-lease control alongside each writer order. On both H100
ranks, independent compute completed in 0.17–0.34 ms while the old reader
completed in 50.67–50.69 ms. All new writes completed after that reader.
The old reduce-scatter result, the new FP32 dW and the zeroed view matched
the reference bits in both writer orders. The artificial reader delay
establishes ordering in this component. The unprofiled EP4 A/B/B/A campaign
measures training throughput and peak memory.
[Overlay and method](experiments/deferred-dw/README.md) ·
[Rank 0 receipt](artifacts/raw/deferred-dw-component-04/rank0.json) ·
[Rank 1 receipt](artifacts/raw/deferred-dw-component-04/rank1.json)

All four 10+10 arms passed with the same eight H100s, matched resolved
settings, fresh per-arm compiler caches and all seven groups active on every
rank. The compiler observer showed stability in its observed channels.
Peak GPU memory uses the maximum over ranks 0–7 and updates 11–20.

| Order and source | Window tokens/s | Median step tokens/s | Peak GPU memory |
| --- | ---: | ---: | ---: |
| A1, original combined fixes | 152,140 | 152,376 | 70.91 GiB |
| B1, deferred wait | 156,439 | 156,728 | 71.08 GiB |
| B2, deferred wait | 154,686 | 154,612 | 70.95 GiB |
| A2, original combined fixes | 151,689 | 151,428 | 70.97 GiB |

B1 improved on A1 by 2.83% and B2 improved on A2 by 1.98%. Their
geometric mean improvement was 2.40%. A2 and B2 ran 0.30% and 1.12%
slower than their same-source repeats. Peak GPU memory changed by +0.17
and −0.02 GiB in the two matched pairs. Maximum allocated memory changed
by at most 0.001 GiB, and maximum reserved memory changed by +0.160 and
−0.020 GiB. Maximum rank-0 loss differences were 0.00020 and 0.00024.
Maximum measured-window rank-0 gradient-norm differences were 0.0009 and
0.0017, compared with 0.0010 between controls and 0.0016 between deferred
repeats. These are rounded training logs from one allocation.
[Four-arm comparison](artifacts/raw/deferred-dw-abba-01/comparison.json)

The repeat trace covered the same first three microbatches of update 21 on
ranks 0 and 4. Each rank moved all 18 gradient-slot waits from pre-backward
to its first writer. The 15 prepared-weight waits remained in pre-backward.
The old exposed gap was 91.28 and 90.22 ms per rank across the three
microbatches. New first-writer gaps totaled 0.049 and 0.051 ms. They began
after their launch calls returned and overlapped no predecessor
reduce-scatter or FP32 accumulation.

The new schedule overlapped 72.41 and 72.25 ms of pre-write compute with the
prior expert reductions. Most overlapped work was recompute. It also
overlapped 1.02 and 4.20 ms with FP32 shard add. CUPTI omitted exact
producer event records on both captures. Host scopes, device order and
native release placement constrain these timings. Gathers still queued
behind shared-stream expert reduce-scatter. The matched weight consumers
showed no idle time overlapping those gathers or preceding reductions.
[Control trace extraction](artifacts/raw/efg-n22-04-ep4-expert-trace2/direct-dw-extracted.json) ·
[Deferred trace extraction](artifacts/raw/efg-n22-defer01-ep4-trace/extracted.json)

### Planned comparisons

The isolation campaign uses the candidate source in every arm and fresh
per-arm compiler caches. EP8 compares all features disabled with
`bf16-cache` alone, including its singleton copy savings. EP4 runs each
feature on its own between two all-off controls on the same devices.
The EP4 BF16 arm includes the BF16 collective, local cast reuse and its
fences. The FP8 cache stays off in both BF16 arms so the BF16 branch is active.
The campaign is `expert-isolation-01` on SR008.

The EP8 10+10 pair completed at 152,968 tokens/s with all features off and
157,185 tokens/s with BF16 reuse (+2.76%). Median step throughput rose
2.52%. Both peaks were 72.30 GiB, at rank 3 and update 20. Allocated
memory rose by 0.004 GiB and reserved memory was unchanged. Resolved
configuration matched after the single feature toggle and run path.
All eight ranks activated BF16 reuse in all seven expert groups. The
compiler observer showed stability for all monitored ranks. This is
one matched pair in the original A recipe.
[Fresh EP8 BF16 isolation](artifacts/raw/expert-isolation-01/ep8-comparison.json)

Consider a separate expert reduce-scatter stream and communicator if shared
FIFO ordering still delays all-gathers after the deferred-wait trial.
Compare with unchanged precision, bytes, and buffer count.
PyTorch's newer implementation shows the independent-communicator pattern.
[FSDP2 v2.15](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/distributed/fsdp/_fully_shard/_fully_shard.py#L962)

Test typed FP8 data/scale collective coalescing if host or launch overhead is
exposed. Keep both receive views contiguous and avoid an unpack copy. Try
additional buffer depth only when acquisition stalls and the native peak has
headroom. One additional full EP4 expert FP32 reduce-scatter slot costs
2.25 GiB. Use component checks to select candidates for a new matched native
10+10 pair.

The staged two-H100 components compare typed gather coalescing, persistent
gathered scales across weight-generation hits, and symmetric FP8 planes on
the NCCL copy-engine path. The separate gradient experiment exchanges only
the peer-owned BF16-rounded slice and keeps the owner reduction in FP32.
Its one-chunk and eight-chunk arms account for the extra staging buffers.
[FP8 gather experiments](experiments/README.md) ·
[BF16 owner transport](experiments/bf16-owner-transport.md)

The EP4 layout alternates seven expert groups across two FP8 receive slots.
The native prefetch chain runs in forward order and reverses for backward.
Check whether a slot's last backward writer is also its first forward reader
in the next microbatch. If both ranks show this order, test reusing the
completed receive planes with normal slot ownership and no extra weight buffer.
A valid hit skips the E4M3 and FP32 scale gathers, saving 302,063,616 bytes in
each direction per rank. Guard hits with the writer, weight generation,
completion dependency, and a process-group-consistent decision. Test a
competing write, async completion, and one-rank invalidation before estimating
any exposed time savings.

A dedicated receive pair could also keep one group resident for a complete
weight generation. It adds 604,127,232 bytes (0.563 GiB) per rank because both
shared slots still serve other groups. Check the completed EP4 memory peak and
the exposed duration of the remaining gathers before trying it. The pinned
FSDP2 API documents the same memory versus communication tradeoff across
accumulation microbatches.
[FSDP2 reshard after backward](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/torch/distributed/fsdp/_fully_shard/_fully_shard.py#L453-L473)

### Two-H100 communication results

The component campaign used the recovered allocation and the production Torch
and NCCL pins. The table reports medians across repeated timing blocks, each
based on the maximum duration across the two ranks for each sample.

| Component | Control device span | Candidate device span | Observed cost |
| --- | ---: | ---: | --- |
| Typed FP8 gather coalescing | 1,205.06 µs | 1,211.05 µs | Host submission fell from 48.61 to 28.79 µs. Device span increased in every cycle. |
| FP32 scale cache hit | 1,191.46 µs | 1,179.92 µs | Host submission fell from 43.95 to 24.26 µs. Estimated production storage is 720 KiB per rank. |
| FP32 scale cache refresh | 1,194.63 µs | 1,194.66 µs | Both routes gathered the scale plane. |
| Copy-engine gather with concurrent GEMM | 1,220.09 µs | 3,377.46 µs | The proxy GEMM itself improved from 151.88 to 87.12 µs. The joint span increased. |
| BF16 owner transport, one chunk | 4,293.78 µs | 3,289.71 µs | −23.38% device span, +1.125 GiB staging per rank, +42.68 µs host submission. |
| BF16 owner transport, eight chunks | 4,268.21 µs | 3,945.09 µs | −7.57% device span, +144 MiB staging per rank, +403.02 µs host submission. |

The copy-engine probe confirmed the paths for both the 301,989,888-byte data
plane and the 73,728-byte scale plane. The default registered-memory route
also increased the joint span to 1,367.03 µs. Keep the ordinary gather for
this fabric and payload. Scale-cache hits saved 9.84–11.71 µs across the three
ABBA cycles, while refresh timing stayed within control variation. An
illustrative 448 fully exposed hits at the overall median saving would save
about 5.2 ms in a 13.65 s update. A native scale-cache trial needs evidence
of a larger scheduling effect or an accompanying material change.

The owner transport matched FP32 output bits for every tested BF16-producer
case, including when one rank was empty and when the output was aliased.
The deliberate non-BF16 peer operand differed. A native trial needs the
two-rank direct-dW producer guard and must preserve downstream FP32
accumulation. The one-chunk arm has the larger timing margin. The trace
must first establish how much expert reduce-scatter is exposed.
[Rank samples, activation and parity receipts](artifacts/raw/ep4-components-01/component-receipts.json)

### Reusing dead peer gradient storage

The follow-up used the same H100 pair and restored the FP32 slot before
every timing sample. It packed the peer contribution into a disjoint BF16
send buffer, then received into the dead peer FP32 region. The owner region
and the FP32 accumulation destination retained their original types.

| Chunks | Native device span | Disjoint owner span | Receive-alias span | Aliased staging per rank |
| --- | ---: | ---: | ---: | ---: |
| 1 | 4,281.22 µs | 3,283.61 µs | 3,286.29 µs | 576 MiB |
| 2 | 4,279.45 µs | 3,323.26 µs | 3,318.80 µs | 288 MiB |

Receive aliasing halved measured staging and retained the device-time gain.
The two-chunk alias arm saved 22.45% of the native device span and added
107.78 µs of host submission. Both ranks passed bitwise peer overwrite,
owner result, canary, next collective and two-update FP32 accumulation
checks. The measured device span includes packing, communication, and owner
addition. Native throughput still depends on how those costs overlap in
training.
[Receive-alias summary](artifacts/raw/bf16-owner-alias-01/summary.json)

The production GFSDP reduction entry point then passed on the same H100
pair with the two-chunk overlay. All 301,989,888 FP32 elements matched per
rank after the first out-of-place reduction and after 14 reductions into
the same persistent accumulation buffer. The second group cycle included
an empty contributor. Each slot overwrite joined the preceding reduction
event. The compiled direct dW route and total training memory still need
native validation if the deferred-wait trace shows material residual
communication.
[Native GFSDP receipts](artifacts/raw/owner-transport-native-01/receipts.json)

### BF16 dW slot and pack-free owner exchange

The two-H100 component passed 14 transport checks per rank. It sends the
peer slice directly from the BF16 producer slot and uses one disjoint BF16
receive buffer. Two ordered FP32 additions preserve the native reduction and
accumulation roundings. Both ranks matched all checked FP32 output bits
from native reduce-scatter. The stock BF16 DeepGEMM writer and current
rounded FP32 writer also matched bitwise from real FP8 operands in the
full, mixed, and empty routing cases.

| Transport scope | Native FP32 reduce-scatter | Packed FP32 owner slot | Pack-free BF16 slot |
| --- | ---: | ---: | ---: |
| First owner output | 4.309 ms | 3.322 ms | 2.690 ms |
| Subsequent FP32 accumulation | 5.424 ms | 4.498 ms | 3.060 ms |
| Input slot plus transport scratch per rank | 2,304 MiB | 2,592 MiB | 1,440 MiB |

The reported device spans are medians of the repeated block medians of
samplewise rank maxima. The BF16 slot saved 864 MiB of measured staging
per rank against native and increased host submission from 39 to 113 µs
in the subsequent accumulation scope. Full-routing stock BF16 zeroing
plus GEMM took 6.330 ms versus 6.044 ms for the current direct writer.
Mixed routing took 4.394 versus 4.297 ms with overlapping repeated
measurements. Empty routing took 0.375 versus 0.747 ms. The producer and
transport have separate timings. The deferred trace showed the targeted
reduction waits already hidden. A guarded training integration can test
the staging savings and any remaining effect on training throughput.
[Component](experiments/bf16_owner_slot.py) ·
[Native timing and memory receipts](artifacts/raw/bf16-owner-slot-01/evidence/summary.json)

If the first weight refresh delays an otherwise-ready gather, test a separate
producer stream for local quantization. It can reuse the existing persistent
planes and make the communication stream wait on their generation-ready event.
The deployed FSDP2 source uses a separate copy-in stream, and TorchAO uses
post-optimizer FP8 scale preparation for its tensorwise recipe. Here the
blockwise quantizer and its BF16 rounding stay the same.
[FSDP2 copy-in scheduling](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py#L309-L345) ·
[TorchAO post-optimizer preparation](https://github.com/pytorch/ao/blob/6eb4c0010ad82db7b4a876a9c4d0ce0d1d34f792/torchao/float8/fsdp_utils.py)

The update-21 trace placed all seven local refreshes on the expert
communication stream in the first microbatch on both ranks. They occupied
3.83 ms on rank 0 and 3.72 ms on rank 4. The next two microbatches had
no refresh launches. Fully hiding that measured work would save about 0.03%
of the ordinary 13.65 s update at this crop. The scheduling trial would
need evidence of a larger downstream delay or a different workload.

In the same trace, prepared-weight FP8 transpose occupied 3.080–3.087 ms
per microbatch per rank across 28 same-stream kernels. Fully hiding that
duration for 32 microbatches would save about 99 ms, or 0.72% of the
unprofiled EP4 update. The pinned DeepGEMM Hopper grouped FP8 entry point
asserts K-major weights. Its NN wrapper only creates a strided transpose
view. A physical transpose is required for this input layout. A separate
stream could perform the existing exact transpose after backward prefetch,
at the cost of dedicated scratch and additional dependency tracking.
FlashAttention-3 demonstrates a different kernel design that transposes
FP8 tiles in shared memory. It would require changes to the grouped GEMM
pipeline in this setting.
[NVIDIA Hopper FP8 layout](https://docs.nvidia.com/cutlass/4.8.0/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_warpgroup.html) ·
[FlashAttention-3, Appendix B.8](https://proceedings.neurips.cc/paper_files/paper/2024/file/7ede97c3e082c6df10a8d6103a2eebd2-Paper-Conference.pdf) ·
[Transformer Engine blockwise layout](https://nvidia.github.io/TransformerEngine/features/low_precision_training/fp8_blockwise_scaling/fp8_blockwise_scaling.html)

TIDE provides mixed-precision reduction and optional vacant-gather-slot
staging on NCCL 2.31+. The current runtime is NCCL 2.30.7. The owner-transport
component exercises the available two-rank typed all-to-all and compares its
full pack, transfer and FP32 addition against native reduce-scatter.
[TIDE study](https://arxiv.org/abs/2610.38228)

### Native EP8 numerical control

Two original-A runs completed the first full optimizer update on the same
eight H100s with compile, SAC, and accumulation32. The offline validator now
checks the six 128-element FP32 router arrays separately from the
4,096-element sampled expert vectors. All 16 native rank receipts passed
source, initial state, input, configuration, counter, and cache checks.
The capture helper was unchanged.

All 256 rank-local microbatch losses and eight update losses matched bitwise.
Sampled pre-reduction expert gradients differed in both projections of
layers 1 and 4 from the first microbatch on every rank. Layer 7 matched in
all 32 microbatches on every rank. In total, 1,024 of 1,536 pre-reduction
vectors differed. The largest absolute element difference was
2.05e−8 and the largest vector relative L2 difference was 3.17%.
The global gradient norms were 1.08216941 and 1.08228683. The earliest
observed differences preceded native eFSDP reduction and the optimizer.
The pair measures repeatability under this recipe. It leaves the cause of
the variation and an acceptance tolerance open.
[Numerical method](numerics/README.md) ·
[A/A comparison](artifacts/raw/ep8num-01/ep8-control.json)

The combined expert candidate completed the same full update. Input
digests, exact counters, all raw losses and sampled router state matched
both controls bitwise. The candidate norm was 1.08220863. Both candidate
comparisons first differed before reduction in microbatch 1 in the same
projections, layers and ranks as the A/A control. Layer 7 stayed exact.

| Comparison | Global grad-norm delta | Largest sampled pre-reduce abs delta | p90 pre-reduce vector max abs delta |
| --- | ---: | ---: | ---: |
| A1 vs A2 | 0.000117421 | 2.049e−8 | 8.149e−9 |
| A1 vs expert | 0.000039220 | 1.490e−8 | 9.779e−9 |
| A2 vs expert | 0.000078201 | 2.421e−8 | 1.164e−8 |

Some sampled candidate gradient differences exceed the observed A/A
differences. The exact losses and narrower global norm deltas bound those
observed outputs. One A/A pair gives no acceptance tolerance for gradients
or optimizer state.
[Candidate versus both controls](artifacts/raw/ep8num-01/ep8-parity-both-controls.json)
