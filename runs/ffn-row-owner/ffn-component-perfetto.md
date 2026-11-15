# FFN component device overlap

Both native component diagnostics show early FFN reduce-scatter overlapping remaining parent backward on every rank and microbatch. This establishes the component mechanism; native head/combined scheduling and throughput remain separate gates.

## Scope and import verification

The existing layer0 and MTP captures each contain three microbatches on all eight H100 ranks. The receipts preserve FP8 linears, BF16 compute, FP32 gradient reduction, model compilation and selective activation checkpointing. No GPU run, profiler rerun or numerical check was added.

Raw Perfetto imports dropped overlapping kernels: layer0 ranks 0–7 lost **8/3/4/5/8/8/5/6**; MTP lost **9/7/14/7/6/11/11/9**. The pinned [display-lane normalizer](../overnight-streaming-perf/monitorfree-perfetto-prepare.py) was applied only to ignored gzip derivatives. Canonical plain trace bytes remain unchanged. Exact device names, timestamps, durations and arguments match before/after; only display tids and matching flow endpoint tids move, and GPU user ranges are removed.

Normalized MCP imports recover **570 kernels + 75 memsets/rank** for layer0 and **711 kernels + 99 memsets + 6 DtoD copies/rank** for MTP. Every device event has exactly one actual launch flow: **645/645** and **816/816** distinct endpoints, respectively; all correlations match, with zero importer errors/data loss. The [JSON](ffn-component-perfetto.json) retains trace IDs, raw/derivative hashes, SQL, per-event slice IDs, source identities and parity receipts.

## Actual device overlap

Values below are min–max across the three captured microbatches on each rank, in ms. Overlap is the **interval union** of FFN NCCL execution intersecting remaining-parent compute kernels; memsets and DtoD copies are excluded.

| Component | Rank | FFN RS span | RS ∩ parent compute | Compute coverage of RS |
|---|---:|---:|---:|---:|
| layer0 | 0 | 1.719–4.304 | 1.680–4.207 | 97.51–97.75% |
| layer0 | 1 | 1.590–4.279 | 1.551–4.181 | 97.58–97.72% |
| layer0 | 2 | 1.556–4.712 | 1.509–4.609 | 96.99–97.83% |
| layer0 | 3 | 1.773–4.689 | 1.736–4.589 | 97.70–97.90% |
| layer0 | 4 | 1.651–4.329 | 1.607–4.227 | 97.27–97.63% |
| layer0 | 5 | 2.030–4.809 | 1.978–4.704 | 97.43–97.86% |
| layer0 | 6 | 1.882–4.803 | 1.832–4.700 | 97.37–97.84% |
| layer0 | 7 | 1.948–4.922 | 1.897–4.823 | 97.39–97.98% |
| mtp | 0 | 1.615–2.033 | 1.581–2.004 | 97.88–98.61% |
| mtp | 1 | 1.594–2.610 | 1.560–2.556 | 97.89–98.12% |
| mtp | 2 | 1.697–2.808 | 1.663–2.737 | 97.48–98.39% |
| mtp | 3 | 1.723–2.859 | 1.692–2.788 | 97.52–98.57% |
| mtp | 4 | 1.887–3.007 | 1.852–2.939 | 97.73–98.29% |
| mtp | 5 | 1.878–2.983 | 1.846–2.921 | 97.93–98.34% |
| mtp | 6 | 1.862–2.644 | 1.825–2.584 | 97.74–98.19% |
| mtp | 7 | 1.768–2.850 | 1.734–2.781 | 97.55–98.41% |

On all **48 rank/microbatch cases**, FFN dInput and all three weight-gradient kernels finish before the FP32 foreach-copy packing kernel, which finishes before the actual FFN NCCL kernel. Exactly one FFN RS precedes exactly one parent RS. Device compute/packing uses stream **7**; actual NCCL kernels use stream **21** on every rank. CPU scope → descendant CUDA runtime/driver launch → actual flow → device linkage establishes attribution; host readiness markers are supporting evidence only.

The cached compiled FFN backward identifies the three TNT GEMMs as w2, w3 and w1 gradients, with a clone unpadding w2 and `triton_poi_fused_add_3` finishing dInput. Generated code and hash are in the JSON; this is source/trace identification, not a new numerical check.

For layer0 rank0/MB0, relative to its host microbatch start: dInput ends at **15.862 ms**; w2/w3/w1 GEMMs end at **14.599/16.564/17.268 ms**; packing runs **17.270–17.552 ms**; FFN NCCL runs **17.554–20.508 ms**, overlapping **2.881 ms** of parent compute; parent compute ends **25.500 ms**, then parent NCCL starts **25.649 ms**. All-rank per-MB bounds and actual slice/launch IDs are retained in JSON.

FFN input is **134,479,872 Float elements = 537,919,488 B = 513 MiB**; each rank receives 67,239,936 B. Parent inputs are **271,778,560 B = 259.188232 MiB** for layer0 and **399,165,440 B = 380.673828 MiB** for MTP. Group size is eight and dtype is Float throughout. These are collective input/output payloads, not wire traffic or bandwidth. The [existing control trace record](control-trace.json) owns whole-group baseline evidence; no second control capture was made.

## Region release and reuse

All eight receipts per component show two physical storages, each split into adjacent, disjoint parent/child views with unchanged total budget:

| Component | Unique pool bytes | Data parent / child byte intervals | Gradient parent / child byte intervals |
|---|---:|---|---|
| layer0 | 1,214,547,072 | [0,135889280) / [135889280,404849024) | [0,271778560) / [271778560,809698048) |
| MTP | 1,405,627,392 | [0,199582720) / [199582720,468542464) | [0,399165440) / [399165440,937084928) |

Next-microbatch parent/child all-gathers reuse these regions; first gathers start **7.223–7.435 ms** after prior FFN RS completion in layer0 and **12.037–12.605 ms** later in MTP. However, this diagnostic explicitly synchronizes after each microbatch. It observes completed-reader reuse and cannot stress overlapping pending-reader reuse. Backward prefetch markers do not imply a fresh device gather: the component retains those weights until post-backward reshard.

The inspected, receipt-hash-matched source makes the intended dependency explicit: [`GFSDPBufferContext.wait_for_release`](../../gift/torchtitan/distributed/gfsdp/_param_group.py) waits all overlapping contexts; `post_backward` acquires its region before packing, publishes `_post_reduce_event` and clears ownership; [`reduce_scatter`](../../gift/torchtitan/distributed/gfsdp/_collectives.py) waits the current stream before reading gradients and records completion after reduction/output handling. `unshard` waits overlapping data-region release events before all-gather preparation. The [existing buffer-region record](buffer-regions.md) owns delayed-NCCL pending-reader/borrow tests; none was rerun.

CUDA event record/query and stream-wait API scopes exist, but event/stream **handles are absent**. Therefore the trace does not prove which release event each wait consumes. Source plus existing delayed-NCCL tests complement this limit; adjacent times or an empty main stream are not treated as causal waits.

## Verdict and remaining gate

Early FFN reduction, actual parent-backward overlap, all-rank ordering, stream attribution and unchanged disjoint storage pass for these components. No ordering or missing-overlap counterexample was found. NCCL spans can include peer waiting and contention, so the trace does not quantify causal speedup or establish contention-free execution.

The separate native bounded trace must cover head/combined scheduling, natural accumulation, prefetch and overlapping-reader reuse; unprofiled timing and memory lifecycle acceptance remain owned by the native/component result records. No additional numerical investigation is requested or implied.

## Reproduction and retained evidence

Use MCP `load_trace` on the paths in JSON, then its retained SQL against each `trace_id`. Canonical and normalized paths, SHA256 values, importer censuses, correlation checks and source identities are included. The ignored `perfetto-normalized/analyze.py` reconstructs each row from the MCP kernel export; `build_report.py` produces these two maintained files. Assertions verified exact device-payload parity, all 16 source identities, all 48 readiness/packing/RS orders, metadata and disjoint buffer intervals. Source/config/runner bytes were not edited.
