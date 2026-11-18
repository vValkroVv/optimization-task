# Current-source timing-drift research

15 November 2026. Bounded read-only research before the seven new 140-update
series exist. This is supporting evidence for the existing
[drift record](../flash-qla/timing-drift.md), not a new timing result or a causal
attribution. Training source, configuration, caches and GC behavior were not changed.

## Acceptance and evidence boundary

- Re-establish the requested routing, padding, communication, input, compiler,
  GC/host/allocator and clock mechanisms in current source.
- Identify existing evidence usable for the new series without extra per-step
  synchronization, tensor reads, collectives or GC changes.
- State what the planned series can decide and the smallest missing discriminator.
- Check this document's local links and preserve reproducible source identities.

Inspected task HEAD is `e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f`.
Working-tree bytes matter because another contributor owns a capacity experiment
and another owner is integrating stage selection. The campaign's
[capacity overlay](measure.py) loads the original module from
[capacity.original.py](../capacity-group-allreduce/capacity.original.py), validates
it against Git HEAD and SHA256
`2f034b072fdaf02ac844936366134fafcd5e815588ce750b0dc7dfcc50467120`, and records the
loaded identity. Claims below use those original capacity bytes, not the edited
working-tree capacity module.

Environment metadata was read with `/opt/torchtitan-venv/bin/python3.12` and
`importlib.metadata`, without importing CUDA packages: torch
`2.15.0.dev20260822+cu130`, Triton `3.8.0+git3f6e4113`, TileLang `0.1.12`,
Mosaic Streaming `0.13.0`, DeepEP `1.2.1+1b8f467`, DeepGEMM `2.3.0+f093a4c`.
Installed `torch/version.py` gives Git
`8b353065ce260336c5fdfa74d754e8bd9b91b062`. Existing
[component runtime receipt](component-results.json) corroborates CUDA 13.0,
NCCL 2.30.7 and eight H100 80GB hardware. An initial SSH metadata read failed
because `sr008-gpu` was not resolved in this shell; the coordinator established
that this shell already runs on the authorized master, so subsequent reads were local.

The [historical JSON](../flash-qla/historical-drift-comparison.json) records
roughly 9.9–10.4% lower host cadence in updates 121–140 than 41–60 across all
eight ranks, with a preceding hump around 31–40. Those runs used the old repeated
fixture. The current [production-data report](../production-data-abcd/report.md)
and [short-run summary](../production-data-abcd/performance-summary.json) establish
the corrected input and 10+10 results, not recurrence of the historical trend.
The previous all-four-stage SE cache finding remains separate; its extra
`_routing_metadata_kernel` specializations do not explain the entire slowdown.

## Source-supported mechanisms

| Mechanism | Current active path and implication | Missing discriminator |
| --- | --- | --- |
| Adaptive balance bias | [MoE.forward](../../gift/torchtitan/models/common/moe/__init__.py) accumulates selected-expert counts over forwards, masked for auxiliary accounting; SAC recomputation can also contribute. [expert_bias_delta/register_moe_load_balancing_hook](../../gift/torchtitan/models/common/moe/load_balance.py) reduces counts, updates bias in the optimizer pre-hook, then clears counts. Coefficient is 0.001; adaptive magnitude is `clamp(1+max_vio,1,5)` times the sign rule, recentered to zero mean. It runs every optimizer update, independently of LR warmup 2000 and measurement warmup 40. | No count/max-violation/bias series across early and late windows. Uniform doubling of counts preserves the sign and approximately preserves max violation; nonuniform recomputation must not be assumed equivalent. |
| Top-k and destination work | [TokenChoiceTopKRouter.forward](../../gift/torchtitan/models/common/moe/router.py) adds bias to FP32 sigmoid scores before top-8 choice, then gathers unbiased weights, normalizes and scales by 2.5. Expert IDs determine owner/destination work, so equal total routes do not imply equal rank load. Changing learned weights, bias and actual text can all change routing. | Per-layer source/expert counts and destination load, linked to actual receive/compute readiness. |
| Grouped GEMM and padding | [GroupedExperts._fp8_experts_forward](../../gift/torchtitan/models/common/moe/experts/gmm.py) consumes actual dispatcher payload and padded group sizes for both FP8 grouped GEMMs, builds row-to-expert indices, and uses unpadded counts for auxiliary accounting. Padding is 128; DeepGEMM SM budget is 116. A changing distribution affects local M, kernel efficiency, memory traffic and straggler work. | Actual padded/unpadded M distributions and per-rank kernel spans; global padded volume alone cannot establish time. |
| HybridEP readiness and waiting | [blocking dispatch](../../gift/torchtitan/distributed/deepep/hybridep.py), `_launch_on_comm_stream` and `_join_dispatch` overlap weight quantization with dispatch, then join. `non_blocking_capacity_factor=None` is required for the configured FP8 path. Blocking metadata makes counts available in pinned host memory, computes exact output rows, narrows counts on CPU and copies metadata back asynchronously. Combine can overlap the shared expert before `sync_combine`. | Long communication kernels can include peer waiting; ranks 0/1 update-21 traces cannot identify all-rank changing stragglers at 41–140. Need all-rank receive readiness and expert compute completion at sampled windows. |
| Capacity/drop | Original `select_capacity_routes` counts active routes in int64, applies source capacity and the global destination-group gate. For 8192 tokens/source, top-8, EP8 and 128 experts: 65536 local routes; 524288 global routes/microbatch; source quota `50000//8=6250`; group gate requires load greater than `6250*128=800000`. No group can exceed all 524288 routes. Thus dropping is impossible under the unchanged resolved recipe, regardless of text, though capacity predicates/reductions still execute. | Verify resolved config and original overlay receipt in every run; this argument does not remove the control or measure its runtime cost. |
| Loader/prefetch/I/O | [StreamingDataLoader](../../gift/torchtitan/datasets/streaming/core/dataloader.py) primes each finite epoch on the consumer. Depth 32 uses a producer and bounded CPU FIFO; [PrefetchIterator.__next__](../../gift/torchtitan/datasets/streaming/core/prefetch.py) waits for ready data. A uses depth 0; the other requested variants use 32. Two persistent workers, no pinning, shuffle off and local MDS on NFS remain configured. | Boundary readiness does not show readiness within the full window. No per-update queue wait, I/O wait, page fault or NFS-latency series is already recorded. |
| Data/domain ordering | Corrected data has 102400 distinct packed rows, 25600/domain, 26 shards. Deterministic round-robin plus DP8 assigns ranks 0/4 Wikipedia, 1/5 mathematics, 2/6 Python, 3/7 Russian web. Each rank's domain stays fixed while its text evolves; within-domain source-file/length/token changes can still alter routing and unique embedding IDs. 140 updates consume 71680 global rows, below the logical epoch size. | New receipts must confirm no rollover; old 14.7 MB fixture-cache arguments cannot transfer to these distinct real rows. Ranks 0/1 profiles sample only two local domains. |
| First-use compilation/cache | Model/loss compilation, DeepGEMM JIT and custom Triton/TileLang caches are distinct paths. [build_routing_metadata](../../gift/torchtitan/distributed/gfsdp/_row_owner_embedding_kernels.py) buckets unique-ID count with `next_power_of_2`; scalar divisibility adds signatures. A new unique count does not necessarily cause a new binary. Newly populated process JIT keys can mean disk load or compilation. | Existing before/after snapshots bound changes but do not timestamp first appearance within 41–140 or separate load from compile duration. SE must have no execution of the routing kernel; SE-fused may add signatures. |
| GC/host/allocator | [GarbageCollection](../../gift/torchtitan/tools/utils.py) disables automatic Python GC and runs normal generation-1 periodic collection; configured default is 50. [Trainer.train](../../gift/torchtitan/trainer.py) invokes it before the update, so updates 50/100 can include collection in cadence. Host scheduling, GIL/producer/compiler contention and allocation retries remain possible. EMA is configured to start at 10000, outside these 140 updates. | GC logs can explain aligned spikes, not by themselves a smooth recovery. Boundary RSS/allocator counters cannot localize host contention or fragmentation to particular updates. |

Padding bound re-established: each of 128 global experts adds at most 127 rows,
so overhead is at most 16256/524288 = 3.100586% per microbatch. This is a
route-volume bound, not a bound on runtime or a proof against distribution-driven
stragglers. Balanced useful M is 4096/expert, 65536 rows/destination rank; skew can
change the rank critical path substantially while total useful routes stay fixed.

## External primary-source cross-checks

These explain mechanisms, not the cause of our drift. Current documentation is
context; installed pinned source and receipts govern this campaign.

- [Loss-Free Balancing, arXiv:2408.15664](https://arxiv.org/abs/2408.15664)
  supports bias-adjusted selection using recent expert load. Our inspected
  implementation explicitly adds adaptive scaling beyond its baseline sign rule;
  the paper does not predict our 31–40-update hump or settling time.
- Pinned [DeepEP Python wrapper at 1b8f467](https://github.com/deepseek-ai/DeepEP/blob/1b8f467/deep_ep/hybrid_ep_buffer.py)
  documents blocking pinned-memory counts. Its
  [Executor::metadata_preprocess_core](https://github.com/deepseek-ai/DeepEP/blob/1b8f467/csrc/hybrid_ep/executor/executor.cu)
  pads expert counts, synchronizes the active stream in blocking mode and sums
  padded counts for exact output size. This supports exposed readiness waits,
  without showing which peer caused a wait in our run.
- [PyTorch recompilation documentation](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.recompilation.html)
  explains guard-triggered recompilation. The installed
  `torch/_inductor/async_compile.py` uses asynchronous futures/process workers;
  [pinned upstream source](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/torch/_inductor/async_compile.py)
  corroborates that architecture. Stable parent counters do not prove all
  asynchronous work has settled. No compiler settings should change here.
- [Pinned PyTorch memory API](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/torch/cuda/memory.py)
  defines allocation retry and OOM counters. The campaign records these at
  boundaries; zero retries rules out that counted retry path, not all allocation
  overhead or all stream waiting.
- [Mosaic Streaming shuffling documentation](https://docs.mosaicml.com/projects/streaming/en/stable/dataset_configuration/shuffling.html)
  describes ordering controls. Task loader acceptance and its recorded row
  mapping establish actual rank domains; documentation alone does not.
- [NVIDIA SMI documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
  supports read-only clock, power, temperature and clock-event queries. Reported
  event support depends on driver/device; `N/A` must be preserved.

## Existing evidence and joins for the new series

The inherited [NativeWindowRecorder](../overnight-streaming-perf/measure.py)
already supplies the following without adding per-step device synchronization:

| Existing artifact/field | Usable decision and limit |
| --- | --- |
| `cadence[].step`, `monotonic_ns`, `interval_s`, `monitoring_step`, `window_boundary` on all eight ranks | Compare fixed windows 41–60 versus 121–140, ten-update bins across 1–140, within-run spread, rank maxima and rank identity. The callback is after native `Profiler.step`; cadence includes host work and does not identify the responsible device/rank. Synchronization at measurement boundaries makes those intervals special. |
| `window.start/end_monotonic_ns`, `start/end_wall_time_ns`, physical and supervised token totals, `cuda_synchronize_calls` | Preserve full 41–140 envelope as primary; align same-host external monotonic samples and timestamped logs. Derive wall/monotonic offsets only with stated boundary timing uncertainty. |
| `compilation_observer.before/after`, `source_metadata`, `comparison` | Compare Dynamo metric lengths/identities, Inductor async cache counters, CuTe map keys, Triton JIT keys/objects across all ranks. Report snapshot errors/uncovered paths. No per-update first-appearance timing is implied by a boundary delta. |
| DeepGEMM guard receipt and startup cached/compiled messages | Check guarded tested APIs and cache startup. Untested native JIT paths and asynchronous activity remain explicit. Reuse matching timing cache in diagnostics. |
| Training-boundary epoch receipt, window-boundary epoch/prefetch receipt | Confirm epoch, ready count, capacity and producer liveness at captured boundaries; no continuous input-wait attribution. |
| Native logs: update/loss/LR and `[GC]` collection duration | Match explicit collection events to update 50/100, retain spikes in the primary result, and optionally report supporting summaries excluding identified spikes without replacing the declared window. Loss/LR are context, not routing-load measurements. |
| Window memory peaks, allocator start/end/deltas (`num_alloc_retries`, `num_ooms`), host RSS/HWM before/after | Detect retries/OOM and lifecycle differences. Current per-update allocator reads feed peaks, but a detailed per-step allocation series is not persisted. RSS sums double-count shared pages; HWM is process lifetime. |
| Config/bootstrap/source and stage activation receipts; separate update-21 traces | Establish comparability and active operations. Profiles can identify implementation differences, not temporal cause of 41–140 drift. |

The current observer is explicitly boundary-only; see
[compiler_stats.py](../overnight-streaming-perf/compiler_stats.py). Do not infer
first-seen update numbers by interpolating two snapshots. Cache-file mtimes after
the run can be supporting evidence only: build/load timestamps and clock domains
need interpretation, and copying/reusing caches alters their meaning. No cache
scan or watcher was started for this research.

Precisely, `_start_window` and `_end_window` call `_capture_compiler_state`;
`on_profiler_step` records cadence but does not sample compiler maps. Triton's
`kernel_entry_ids` preserves encoded function/device/specialization keys and
object IDs without a per-key update/timestamp. A key missing after update 40
and present after update 140 was first observed in **(40,140]**; the existing
schema cannot locate an exact update or prove compilation rather than a load.
Already-created disk entries can first enter a new process cache during that
interval. Preserve actual before/after key strings in the final receipts and
state this interval instead of inventing a first-use update.

A single lightweight read-only `nvidia-smi` query succeeded on the authorized
master at approximately 14:14–14:15 UTC on 15 November: all eight GPUs were idle
(0% utilization), SM clock 345 MHz, memory clock 2619 MHz, power 68.24–71.90 W,
limit 700 W, temperature 29–33 C, active clock-event mask `0x1`. This proves
query availability only. It supplies no training-time clock evidence. The exact
working query was:

```sh
nvidia-smi --query-gpu=index,uuid,pstate,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,clocks_event_reasons.active --format=csv,noheader,nounits
```

At 14:18:15 UTC a second targeted query also verified the `timestamp` field:
prepend `timestamp,` to the field list above. It returns host-local
`YYYY/MM/DD HH:MM:SS.mmm` (UTC in this environment); clocks are MHz, power W,
temperature C, utilization percent, and reasons a hexadecimal mask. All GPUs
were idle; `0x1` identifies idle clock limiting. The coordinator can append
`--loop=60` to this verified query for one owned logger and retain its exact PID
for cleanup. This research did not start that logger.

If the coordinator uses existing external telemetry, join timestamp/UUID to
cadence without inserting CUDA reads or collectives into training. A proposed
low-rate observer could add host UTC and monotonic timestamps to the query output
every 30–60 seconds in one task-owned process. No such monitor was started here.
Do not lock clocks, change power limits or alter normal GC. Thermal/power flags
and changing clocks would support a system explanation; stable samples would
only weaken it at the sampling resolution.

## Bounded analysis decisions and unresolved attribution

1. Once all seven receipts exist, determine whether corrected data reproduces
   the hump, whether it is common to all variants, and whether SE-fused adds
   outliers/settling relative to SE. Fixed run order remains confounded with
   cross-run system/cache state; do not launch another matrix automatically.
   Reuse the same dataset/tokenizer hashes, seed, initial-state procedure and
   loaded-input identities across variants; do not regenerate inputs to study
   drift. The 71680-row consumption remains below the 102400-row epoch.
2. Compare all-rank curves and timestamp skew. Shared cadence direction is
   consistent with a distributed critical path, including one changing
   straggler; it does not prove all eight ranks have equal local compute cost.
3. Align explicit GC/compile/input/system evidence. Stable boundary compiler
   state plus no logged compile event weakens only covered compiler explanations.
   No event alignment is proof that the remaining hypothesis is MoE balance.
4. If routing attribution remains material, the missing discriminator is
   **joint time-resolved routing volume and readiness**, not another aggregate
   step statistic. The smallest separately proposed diagnostic retains the exact
   recipe and records bounded all-rank/per-layer samples around updates 31–40,
   41–60 and 121–140: unpadded expert counts, padded M, destination route totals,
   max/mean violation, and dispatch-ready/expert-done/combine-ready timing; also
   exposed loader-pop wait. Use existing blocking CPU count metadata where
   available and deferred/batched device counter export outside capture, with
   no per-step synchronizing reads or extra training collectives. This needs a
   reviewed diagnostic implementation and separate compute decision; it was
   not executed or inserted into timings.

If rank-critical work/readiness falls together with imbalance while input,
compiler and clocks remain stable, evidence supports routing/communication
settling. If expert/destination work stays stable while exposed input wait or
system events fall, reject routing-volume attribution and follow the aligned
path. If only padding falls by its small volume bound without matching critical
path changes, it cannot establish the whole-step cause. A new update-21 profile
alone cannot decide among those outcomes.

## Inspected byte identities

SHA256, captured without bulk scans. Paths are relative to task root unless
prefixed by installed package location.

```text
4b848faf7490a3a9343a74d7cb82ed8822ea2efca71346f4fb57ea94e91b87ec gift/torchtitan/models/common/moe/load_balance.py
a63679c01dd0f7fa532780b09d0dc225b1ced4fad4e32ab81cb1a4c5d302dfa1 gift/torchtitan/models/common/moe/router.py
1e31fd2cea3b06fe5ed4d1387a29c060f484e58ad058545d875d62674e8a08dd gift/torchtitan/models/common/moe/__init__.py
39ec44c7faabff090642988b2cb6665bde90fe3bbed9373a949ada70135d1206 gift/torchtitan/models/common/moe/experts/gmm.py
d98d1ad2859864447006798013921700a28ecab6627dd4983ae1645980e1d628 gift/torchtitan/distributed/deepep/hybridep.py
3f5d9cc980ccaace1ebd254ee6d8480bce44812b2ac86dfb2584e8647f4deb88 gift/torchtitan/config/moe.py
8145103e756f972e4f075c3ddd93b901a4fab832cb17114d96efe03a1c5ba471 gift/torchtitan/tools/utils.py
2d563d918e4c818ab67efbd6718818d769aa7ad391bbfa1845a6994fc7d9c248 installed torch/_inductor/async_compile.py
38b8151fcceec1a0210a696775b7c64d7fac5e9b69eee2dda7e93c1e2a377509 installed triton/runtime/jit.py
63e425916b3c6b0c881ad402af35a2b00c4c84203fcde611a6715744b89210cb installed deep_ep/hybrid_ep_buffer.py
```

Installed packages are under
`/opt/torchtitan-venv/lib/python3.12/site-packages`. Dependency metadata versions
and upstream commits do not imply byte equality with installed downstream files;
the local hashes and campaign observer source metadata retain that distinction.
