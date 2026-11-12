# A008 actual loss graph replay review

## Verdict and acceptance

A008 passes the bounded graph mechanism gate on all eight ranks: actual chunked cross-entropy forward and backward graphs replay persistently during native updates 21–23, with no fresh capture or instantiate calls in that complete active window. It merits the matched unprofiled off/on matrix, conditional on the execution owner's separate native-update/configuration/log-integrity checks. This is eligibility for measurement, not a speedup or production acceptance claim.

Finite acceptance: parse every rank trace completely; correlate real graph launches with compiled regions and GPU kernels; count loss forward/backward launches, captures, instantiations and reused IDs per profiled native update; exclude model graphs; identify the finite next gate. All mechanism items pass. No source changes, GPU work or further candidate changes were performed.

## Reproducible evidence

`replay_review.py` and `phase_counts.py` adapt A007's stdlib summarizers to A008 input/output paths. Traces are `../../attempts/008/native/profiling/traces/iteration_24/rank{0..7}_trace.json`. They were processed sequentially, one rank in memory. Python `json.load` completed on every entire trace. `replay-counts.json` preserves complete kernel names, compiled hashes, API totals and graph IDs; `phase-counts.json` resolves profiled updates by timestamps across autograd threads. Backward's first-pass `by_step=unknown` is superseded by that timestamp attribution.

Attribution uses `cudaGraphLaunch.args.correlation` to GPU kernels and their nonzero `graph id`, with enclosing same-thread `Call CompiledFxGraph` and compiled-region/backward annotations. Every launch maps to one of the three groups below. Runtime and driver API categories are present. The second pass additionally inventories capture/instantiate/graph-launch names across **all** categories: only `cudaGraphLaunch`, `cudaStreamGetCaptureInfo`, and `cudaThreadExchangeStreamCaptureMode` appear. Capture-info queries and capture-mode exchanges are not graph recording. There are no begin-capture, end-capture or graph-instantiation calls hidden by a category filter.

## Captured scope and counts

The same counts hold separately on ranks 0, 1, 2, 3, 4, 5, 6 and 7.

| Phase | Per native update | Three-update total per rank | Actual GPU kernels per rank | Reused graph IDs per rank |
| --- | ---: | ---: | ---: | --- |
| Blockmask region `0/0` | 4 launches | 12 | 348 | `5` |
| Loss forward region `2/0` | 32 launches | 96 | 192 | `8,14,20,26,32,38,44,50` |
| Loss backward | 32 launches | 96 | 96 | `11,17,23,29,35,41,47,53` |
| Model/dense regions | 0 graph launches | 0 | 0 graph-correlated | none |

Each of updates 21, 22 and 23 has four natural microbatches and eight loss chunks per microbatch: 32 loss-forward and 32 loss-backward launches. Every rank reuses the same eight forward and eight backward graph IDs in each update; the eight IDs collectively execute 96 times per loss phase across the active window. Fresh captures and instantiations are **zero for each step and rank**, as well as globally. Per rank: 204 graph launches, 636 graph-correlated GPU kernels; all ranks: 1,632 launches, including 768 actual loss forward and 768 actual loss backward launches. Only 96 launches are blockmask.

Rank-0 forward compiled hash is `f3vxowyb7jxtqcto25rxvawpojzhtfcdarer3wkhmhsk2fmwsaqs`; backward is `fjpss73bfqx454ufhqkvrzcmgazic2yletlxeierhc42zyd3tuwl`. Other hashes are recorded in JSON. Forward launches each run two Triton fused log-softmax/NLL kernels (`prepare_softmax_online_view_0`, `nll_loss_forward_view_1`); backward launches each run the fused log-softmax/NLL backward kernel. This proves actual loss work beyond mask-only capture. No dense, attention, MoE, head GEMM, optimizer, communication or full-update graph capture is established or claimed. Model computation remains outside these launched graphs.

## Runtime cost and limits

The natural chunk loop in `gift/torchtitan/components/loss.py` calls `lm_head(h_chunk)`, then `_chunk_loss`, then backward for each chunk (lines 727–760). Compiled CE calls `cross_entropy_loss`, whose local path casts logits to fp32 before summed CE (lines 34–44); normalization remains outside `self.fn` at lines 789–791. Replay coverage is specifically CE forward/backward, not the whole chunk loop or full loss/head path.

A material remaining cost question is graph input storage and copying. The installed runtime is torch `2.15.0.dev20260822+cu130`; inspected `torch/_inductor/cudagraph_trees.py` SHA256 is `7cab9ba4338305b10c792ffc7389898b46623a8e87052928158581b33e413e76`. `CUDAGraphNode.run` copies non-static inputs before replay (lines 1264–1271); `_copy_inputs_and_remove_from_src` uses `torch._foreach_copy_` (1206–1226), and `_allocate_and_copy_recording_inputs` allocates non-static, non-managed inputs in the graph memory pool (1911–1946). The same source explains shared-pool lifetime/reuse at lines 1–25. Selected excerpts are preserved in `runtime-source-excerpts.txt`.

These facts justify measuring paired peak allocated/reserved memory and integrated cost; they do not establish A008's exact copied bytes, persistent-pool overhead, or a regression. Eight reused loss graph pairs are compatible with tree paths through sequential chunks; ID count alone does not mean eight fully independent live allocations. No new custom copy or pool fix is justified by this successful mechanism evidence.

This review makes no timing-magnitude claim. Host graph-launch API sums, kernel sums, integrated step spans, synchronization, copies and communication are distinct overlapping scopes; none is an unprofiled end-to-end comparison here. Profiled timings and diagnostic observer updates 3–7 cannot establish performance. Native completion, full-update markers, resolved configuration, memory telemetry and logging-integrity checks belong to the execution owner's companion result; successful trace parsing does not replace those checks.

## Single next decision

Proceed with the exact loss-only candidate and a matched unprofiled baseline/candidate matrix once companion validity checks pass. Keep the frozen SOURCE-SNAPSHOT/006 pin and A008 workload/precision/parallelism/SAC/model-and-loss compilation, and retain common `mix_order_reduction=false` and `graph_partition=true` on both sides. Exclude all 18 model blocks from graphs on both sides; toggle the agreed global graph control for the comparison and report the mask capture difference explicitly. Measure identical-control noise and alternate off/on runs on the same eight-H100 topology, using settled native full updates, paired peak memory, no profiling or verbose compiler logs, and preserved startup/lifecycle costs. Apply the existing numerical, native-update, checkpoint and other finalist gates without inventing tolerances. Stop candidate changes: the mechanism question is resolved, and measured benefit/cost plus existing correctness gates decide promotion.
