# Routing metadata: matched native trace analysis

## Result

Native activation and a local metadata reduction are verified on ranks 0 and 1, for all three sampled microbatches and both original forward and SAC recompute. Each control call launches **11 kernels and one DtoD copy**; the candidate launches **one `_routing_metadata_kernel`**. The matched helper host interval falls from roughly 248/270 µs to 111/107 µs per call on ranks 0/1. Metadata kernel duration falls from 24.29/23.23 µs to 6.34/6.35 µs. These traces support retaining the candidate for the parent-owned performance and numerical acceptance gates; they do not establish full-update throughput or numerical equivalence.

## Captures, identity and import

The control is `four-fusion-control-matched-baseline-trace-001`; the candidate is `four-fusion-routing-routing-trace-001`, both under `runs/overnight-streaming-perf/artifacts/raw/`. Both native summaries and driver completion receipts passed. Each completed 11 updates: ten warmup updates followed by update 11, with exactly three complete CPU F/B markers captured on ranks 0/1. All eight ranks executed accumulation 32 and one optimizer call in the profiled update. The other 29 microbatches and optimizer are outside this trace crop.

The source pin is `33b336a592746340d327d86e34149a23dc8688ee` plus the independently applied [routing patch](routing-metadata.patch). Actual resolved YAMLs differ only in `dump_folder`; the [exact diff](artifacts/raw/prepared-traces/routing-config-differences.json) is retained. B2, S4096, FP8, EP8, dense FSDP8, expert FSDP1, CuTe norms, model/loss compile, SAC, dense MTP1, accumulation32, prefetch32 and logical epoch51200 are preserved. In particular `comm_stream:false`, `fuse_permute:true`, `drop_by_group_capacity:true` and `branchless_capacity_drop:false` remain active.

Actual MCP `load_trace` imported the prepared files. Control: 8,795 kernels, 274 copies, 861 memsets, 9,930 device events per rank. Candidate: 8,405 kernels, 235 copies, 861 memsets, 9,501 device events per rank. Every device event has exactly one host launch flow; all four imports have zero importer errors and zero ambiguous links. The complete 390-kernel and 39-copy reduction per rank equals 39 metadata calls times the local difference. No events were inferred from a missing or truncated import.

Canonical bytes were retained; preparation removed GPU annotation projections and moved only colliding display lanes and their flow endpoints. Device fingerprints, actual timestamps, durations, names and arguments were preserved; no synthetic lane collided with an original track. Original and prepared SHA256s, helper pin, category duration sums and move counts are in [control integrity](artifacts/raw/prepared-traces/control-integrity.json), [routing integrity](artifacts/raw/prepared-traces/routing-integrity.json) and the [analysis JSON](routing-trace-analysis.json). Actual CUDA stream **7**, from `args.stream`, owns the metadata and listed HybridEP consumers; synthetic display tids were never interpreted as streams.

## Module, phase and equivalent boundary

The [crop preset](../../gift/configs/presets/models/gift_moe/gift_3_5/100b_crop8.yaml) has one dense trunk block followed by MoE layers1–7; its MTP block is dense. [Decoder.forward](../../gift/torchtitan/models/common/decoder_blocks/decoder.py) iterates trunk layers before MTP. Original forward therefore has seven dispatches per microbatch, in layer order1–7. The resolved SAC modes save communication only for layer5; [the comm category](../../gift/torchtitan/distributed/activation_checkpoint.py) includes HybridEP dispatch/combine. Backward contains six dispatch recomputes per microbatch in order **7,6,4,3,2,1**. Layer5 dispatch is saved, and MTP supplies no routing call. This mapping uses the actual configured modules and dispatch ordinals, rather than assuming every MoE communication op recomputes.

Each candidate `four_fusion::routing_metadata_cuda` is a CPU descendant inside the corresponding `hybridep::dispatch`; its launch flow owns exactly one `_routing_metadata_kernel`. Each original call is inside `CompiledFunction`/`Torch-Compiled Region`; each recompute is inside `CompiledFunctionBackward` and its autograd engine ancestor. All 39 dispatch IDs, source-based layer mappings, phase/ordinal, native op IDs and enclosing compiled graph names are retained. Fresh-cache compiled graph hashes differ between captures and are not used as cross-run module identities.

For control, the equivalent visible CPU interval starts at the helper's first `aten::ge` and ends after its final returned `aten::view`, including the slice/view construction after the final scatter. The next row-quantization launch provides an independent upper bound. The candidate interval is the complete custom op, including output allocation and returned views. Neither interval exposes the preceding Python prelude; in particular the candidate's shape/dtype/stride eligibility checks precede its custom op. The enclosing dispatch comparison includes this overhead. Ownership follows same-thread host launches through CUDA flows, not GPU overlap with CPU intervals. GPU span is first owned device start to last owned end; kernel sum and interval union are separate.

The [original helper](../../gift/torchtitan/distributed/deepep/routing.py) constructs valid bits, arange offsets, added/masked indices, bool and FP32 zero storage, two scatters, two inverse masks and a score clone/masked-fill. Their 11 kernels and one DtoD clone copy disappear from this region. The dense bool/FP32 outputs and required dispatch work remain.

## Per-rank and microbatch results

Rows sum equivalent calls within each phase; values are milliseconds. F is seven original calls; R is six SAC recomputes. Kernel sum excludes copies. GPU span includes launch gaps; busy union includes metadata copies. The full per-call metrics, every stream and all three microbatches are in the JSON.

| Rank/micro/phase | Host metadata C→R | GPU span C→R | Kernel sum C→R | Busy union C→R | Dispatch host C→R |
|---|---:|---:|---:|---:|---:|
| 0/1/F | 1.613→0.691 | 1.376→0.044 | 0.168→0.044 | 0.176→0.044 | 7.792→6.732 |
| 0/1/R | 1.633→0.734 | 1.408→0.038 | 0.149→0.038 | 0.157→0.038 | 6.914→6.112 |
| 0/2/F | 1.572→0.669 | 1.341→0.044 | 0.168→0.044 | 0.177→0.044 | 7.311→5.951 |
| 0/2/R | 1.589→0.830 | 1.351→0.038 | 0.149→0.038 | 0.156→0.038 | 6.931→5.688 |
| 0/3/F | 1.690→0.678 | 1.458→0.044 | 0.168→0.044 | 0.177→0.044 | 7.237→5.985 |
| 0/3/R | 1.579→0.737 | 1.330→0.038 | 0.145→0.038 | 0.153→0.038 | 6.668→6.068 |
| 1/1/F | 1.765→0.675 | 1.510→0.044 | 0.161→0.044 | 0.169→0.044 | 8.320→6.686 |
| 1/1/R | 1.755→0.702 | 1.483→0.038 | 0.141→0.038 | 0.149→0.038 | 6.721→5.835 |
| 1/2/F | 1.755→0.650 | 1.472→0.044 | 0.161→0.044 | 0.170→0.044 | 7.255→5.973 |
| 1/2/R | 1.747→0.775 | 1.472→0.038 | 0.141→0.038 | 0.148→0.038 | 6.787→5.822 |
| 1/3/F | 1.797→0.664 | 1.532→0.044 | 0.160→0.044 | 0.169→0.044 | 7.151→6.006 |
| 1/3/R | 1.711→0.703 | 1.435→0.038 | 0.141→0.038 | 0.148→0.038 | 6.578→6.153 |

## Dispatch, retained waits and downstream activity

Across all39 calls, mean dispatch CPU duration decreases **1,098.8→936.8 µs** on rank0 and **1,097.8→935.3 µs** on rank1. Its broader elapsed interval, from dispatch CPU entry through the later of CPU return and its final owned device completion, decreases **1,631.8→1,460.8 µs** and **1,613.2→1,471.2 µs**. These are matched local regions with FP8 row/weight quantization, HybridEP all-gather/scan/dispatch, synchronization and output copies still included; they are not isolated communication timings.

The first `ag_nvl_kernel` and routing-map `hybrid_ep::scan` remain after quantization. Scan duration is essentially unchanged: mean35.75→35.74 µs on rank0 and37.03→35.85 µs on rank1. Native dispatch kernel mean is659.04→652.61 µs and652.21→651.73 µs. There is no demonstrated general downstream kernel acceleration. Metadata-end to scan-start is590.16→560.57 µs on rank0 but547.01→575.23 µs on rank1; the gap does not vanish. It contains unchanged quantization, host setup and EP work, and is not entirely idle.

All-device interval unions were clipped to each exact metadata span, metadata-end-to-scan window and enclosing dispatch interval. Metadata span idle averages186.36/203.89 µs for control ranks0/1; a one-kernel candidate span has zero interior idle by definition. Across the wider dispatch interval, all-device idle averages430.11→312.27 µs on rank0 and465.75→303.63 µs on rank1. In the downstream gap alone idle is127.65→126.28 µs and134.51→123.61 µs. These measurements describe the selected windows, not an independently attributed critical path. All concurrent copy/memset/EP/NCCL unions and sums are retained; no NCCL work intersects these dispatch windows, whereas EP and quantization remain active. Copy changes inside the metadata block are exactly the removed DtoD clone; enclosing dispatch copies remain.

Every dispatch still has `cudaStreamSynchronize`: mean291.32→258.80 µs on rank0 and227.90→281.77 µs on rank1. Rank1 synchronization grows while metadata shrinks, so launch reduction does not justify crediting synchronization removal. The host API duration, launch-to-device delay and actual stream for each first EP kernel are in the JSON; neither flow arrows nor timestamp adjacency supplies a CUDA event dependency. Captured CUDA API arguments contain correlation/external IDs but no concrete event handle or stream handle, so event-specific cross-stream wait attribution is unavailable. The resolved `comm_stream:false` and actual stream7 ordering avoid assuming an unobserved side-stream handoff.

The preceding capacity guard remains substantial: the nearest same-thread capacity op averages14.30→14.12 ms per dispatch on rank0 and14.11→14.28 ms on rank1, dominated by its host predicate synchronization. These CPU waits often expose queued earlier work, especially during backward, and are not standalone capacity GPU compute time. Both variants retain42 `torchtitan::conditional_capacity_mask` / `aten::is_nonzero` calls per rank,39 in-place functional capacity all-reduces, and40 native uint64 NCCL all-reduces including the input/token reduction. Source [capacity selection](../../gift/torchtitan/models/common/moe/dispatch/capacity.py) still reduces group loads and evaluates the host predicate before conditional dropping. No saving is assigned to removal of this policy, collective or predicate.

[HybridEP backward](../../gift/torchtitan/distributed/deepep/hybridep.py) still calls `combine_bwd` then `gather_route_gradients`. The unchanged compiled clamp_min/gather/lt/masked_fill backward kernel families remain with matched counts; their fused costs cannot be isolated to the gather alone. There is no standalone metadata-backward marker. The21 visible eager `aten::gather` calls belong to capacity selection, not this gradient gather, and are not presented as metadata-backward timings. No backward elimination or numerical conclusion is claimed.

## Reproduction and limits

[Analysis JSON](routing-trace-analysis.json) retains all156 matched call records, phase aggregates, selected first-layer forward/recompute launches, integer nanosecond bounds and source hashes. [Exact SQL and complete results](artifacts/raw/prepared-traces/routing-trace-sql.json) retain table setup, full native kernel names, selected IDs, CUDA launch flows, region ancestry, wait APIs, guard queries and per-window family unions without truncated results. Recreate the `rt_*` tables and finish dependent queries before switching trace handles: the MCP cache may evict a trace processor and its derived tables.

Two candidate preparation diagnostics initially failed: one compared string host tids numerically; a second mistook existing large host tids for synthetic device lanes. The final check uses newly introduced device tracks against all original tracks and passed, with canonical bytes unchanged. Missing derived SQL tables after cache eviction were recreated. These were analysis-harness failures, not failed training runs.

This deliverable changes only task evidence. Source/config/runner bytes were not edited and no GPU/test/install operation was run. The observed reduction is local host overhead, metadata GPU work and selected dispatch idle, with retained waits and variable communication effects. Repeated unprofiled throughput, affected-block numerical replay and overall candidate disposition remain parent-owned. No full-update extrapolation, memory saving, loss parity or convergence conclusion follows from the three-microbatch trace.
