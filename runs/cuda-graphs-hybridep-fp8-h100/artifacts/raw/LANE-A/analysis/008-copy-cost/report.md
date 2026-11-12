# A008 CE replay input copy cost

## Result

All eight A008 traces directly expose BF16 logits and int64 labels copied before every CE-forward replay in native updates 21–23. Each rank/update copies **25 GiB logits plus 1 MiB labels**. The corresponding device memcpy duration sum is **17.605–17.641 ms per rank/update**, across the 24 rank/update observations. This resolves the previously unmeasured byte-volume question. It does not measure a causal wall-time tax, a speedup/regression, or additional peak residency.

The finite acceptance checklist is complete: inventory complete updates on all eight ranks; correlate device copies to actual host producers and replay; distinguish forward, backward, mask and other copies; report recorded bytes, duration sums and unions; preserve unavailable fields and the existing finalist gates. No training source changes, GPU commands, new runs, package installation, baseline reanalysis or network research were performed. This evidence-only change has no training runtime impact.

## Observed scope

Ranges below cover all eight ranks separately in updates 21, 22 and 23. They are **per rank per update**. Bytes are one-way payload, not read-plus-write memory traffic. Device duration sums are activity totals; interval unions are copy-device busy time over the selected events, not exposed critical-path time. All graph-input copies use stream 7.

| Input activity | Events/update/rank | Bytes/event | Bytes/update/rank | GPU duration sum, ms |
| --- | ---: | ---: | ---: | ---: |
| CE forward BF16 logits `[8,512,102400]` | 32 memcpy | 838,860,800 recorded | 26,843,545,600 (25 GiB) | 17.554–17.589 |
| CE forward int64 labels `[8,512]` | 32 memcpy | 32,768 recorded | 1,048,576 | 0.0496–0.0551 |
| CE backward FP32 scalar `[]` | 32 copy kernels | unavailable in kernel event; 4 logical bytes inferred | 128 logical bytes inferred | 0.0490–0.0533 |
| Mask int64 input `[8,4096]` | 4 copy kernels | unavailable in kernel event; 262,144 logical bytes inferred | 1,048,576 logical bytes inferred | 0.0433–0.0469 |

Each rank has 96 logits memcpy events, 96 label memcpy events, 96 backward scalar-copy kernels and 12 mask-copy kernels across the window. Across eight ranks this is 768 logits events carrying 600 GiB of summed payload; this is traffic across devices/time, not memory capacity. There are no additional large CE-backward input copies observed on the correlated `_foreach_copy_` path. The backward tensor list contains one scalar; its copy kernel is explicitly `Copy<float,float>`. Mask kernels are `Copy<long,long>`. Their logical byte counts follow actual profiler shapes and kernel types, not recorded transfer-byte counters. This does not claim all saved-state pointers or lifetimes are exposed.

Combined logits/labels GPU interval unions are 17.605–17.641 ms per rank/update, matching sums to trace timestamp rounding. Backward and mask union ranges likewise match their sums. Host CUDA API duration sums are 1.387–1.709 ms for forward memcpy submissions, 0.330–0.392 ms for backward copy-kernel launches, and 0.057–0.076 ms for mask copy-kernel launches, each summed over one update/rank. Enclosing `_foreach_copy_` CPU operator duration sums are respectively 1.941–2.281 ms, 0.597–0.699 ms, and 0.137–0.168 ms. These nested scopes must not be added. They exclude surrounding Python work and graph launches; they are not synchronization or host-gap measurements. [summary.csv](summary.csv) gives all 24 observations, including separate sums/unions and host API totals.

## Attribution

The forward chain is observed, not inferred from shape alone:

`Call CompiledFxGraph (CE forward)` → `_foreach_copy_` → `aten::copy_` with BF16 logits or int64 labels → `cudaMemcpyAsync` → device memcpy with matching `args.correlation` → `cudaGraphLaunch` in the same compiled call → graph-correlated CE kernels.

All **1,536** forward memcpy submissions precede their corresponding graph launch on the host. Every correlated device memcpy finishes before the associated CE graph kernels on the same stream. The **864** backward/mask input-copy kernels satisfy the same host and stream-order checks. These are graph-input staging copies outside the recorded graph (`graph id=0`), followed by persistent graph replay. Complete representative host/device events, shapes, correlations and compiled hashes for every rank are in [copy-cost.json](copy-cost.json) and [copy-scopes.json](copy-scopes.json).

For example, rank 0's first logits copy has correlation `396890`, 838,860,800 bytes and 548.289 us device duration. A 32,768-byte label copy follows. CE graph launch correlation `396903` then runs graph `8` on stream 7. This is distinct from the fused CE kernel's `_to_copy` name: that name is not evidence of a second standalone logits memcpy.

This matches the already inspected pinned CUDAGraphTrees path: `CUDAGraphNode.run` copies non-static inputs through `_copy_inputs_and_remove_from_src` / `torch._foreach_copy_` before `run_graph`. The observed producer contract is the separately executed head's BF16 logits passed into compiled CE. Current `loss.py` produces logits using `lm_head(h_chunk)`, calls `self.fn(logits, labels)`, normalizes outside CE and then calls backward. The source casts logits to FP32 inside CE; the observed pre-replay logits transfer remains BF16. The copy evidence warrants including this boundary in finalist cost accounting; it does not authorize a custom buffer change or extending capture scope.

## Other copies and limits

Each rank/update also contains 929 other host-correlated memcpy events carrying **10,333,840,336 bytes**, with GPU sums/unions of **8.732–9.244 ms** and host API sums of **10.947–12.260 ms**, on streams 7/32. These events are not labelled CE replay input copies. Actual ancestry identifies inherited FSDP work: 68 copies of 75,497,472 bytes occur under `FSDP::post_backward_reduce`; 136 copies each of 25,165,824 and 12,582,912 bytes occur under `FSDP::pre_forward`. The remaining small copies include other scalar/data paths. Full size/direction/shape buckets and representative ancestors are in [copy-cost.csv](copy-cost.csv). NCCL/HybridEP kernel communication and kernel-based copies elsewhere are not included in this **memcpy-only** residual; it must not be presented as total transfer/communication cost or all inherited copy work.

Rank 0 has 36 memcpy events without host attribution inside the selected complete steps; each other rank has 42. They are excluded rather than charged to updates or replay. Host-launch step assignment is deliberate: correlated GPU work may cross a CPU annotation boundary; GPU `ProfilerStep` annotations are not used as extra native updates. Trace export is excluded.

Source pins and checks are recorded in [summary.json](summary.json). A008 uses `SOURCE-SNAPSHOT/006`, base `446431dfa191ef0300f116257ae918bc85e38c06`, manifest SHA256 `3a606c81d6b13692e422da04c333252620e504df9f1c4ba59c4bcf0c1a51f603`. Runtime Torch is `2.15.0.dev20260822+cu130`, git `8b353065ce260336c5fdfa74d754e8bd9b91b062`; CUDAGraphTrees SHA256 `7cab9ba4338305b10c792ffc7389898b46623a8e87052928158581b33e413e76`. Reused [mechanism review](../008/mechanism-review.md) establishes configuration/activation and the zero-late-capture result; [runtime excerpts](../008/runtime-source-excerpts.txt) pin the input-copy path. The current unchanged local loss file was inspected as supporting producer-path context, not substituted for snapshot provenance.

Both scripts process one rank at a time; the second pass completes kernel-copy and aggregate-union scopes absent from the first memcpy table. Checks assert all eight outputs, all 24 forward shape/count/byte combinations, backward/mask tensor-list shapes and counts, and host/device ordering. No full critical-path or capture-count audit was repeated.

**Open gates:** matched unprofiled integrated cost/noise; paired allocated/reserved memory and actual headroom; pointer-level saved-state/storage ownership; causal synchronization/critical-path attribution; existing numerical/checkpoint/finalist gates. One copied 800 MiB tensor implies a real staging operation, but neither graph-ID counts nor cumulative copied bytes establish simultaneous buffer residency. CUDAGraphTrees pool sharing/reuse prevents summing graph-node shapes into a peak-memory estimate. Existing sampled `nvidia-smi memory.used` cannot resolve an isolated staging/pool increment. Copies alone do not demonstrate a qualifying uncovered host region: finite B/C dispositions remain unchanged. Stop after this bounded cost assessment.
