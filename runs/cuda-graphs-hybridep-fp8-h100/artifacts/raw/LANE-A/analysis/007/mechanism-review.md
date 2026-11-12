# A007 native CUDA Graph mechanism review

## Acceptance and verdict

Finite checklist: attribute actual graph launches to dense/loss forward/backward on every rank; distinguish reused graphs from fresh capture; identify affected operation families and pointer guards; decide whether A007 or the conditional loss-only fallback merits further research. No performance estimate, source edit, or GPU launch is part of this review.

A007 executes real graphs in the first dense block and chunked loss, in forward and backward, on all eight ranks. It does **not** establish persistent representative replay: most graph launches immediately follow fresh recording, even in native profiler steps 21–23. The only consistently reused graph is the blockmask graph. A007 should not automatically advance to a full performance matrix as a settled replay candidate.

## Evidence and method

The two scripts beside this document read the native `attempts/007/native/profiling/traces/iteration_24/rank{0..7}_trace.json` files one rank at a time using Python stdlib. They correlate `cudaGraphLaunch` runtime API `args.correlation` with GPU kernels and nonzero `graph id`, and locate enclosing `Call CompiledFxGraph` CPU annotations. Both runtime and driver event categories are inspected; graph launches here use the runtime API. Forward and backward phase identity comes from parent compiled calls and kernel families, not a capture flag.

`replay-counts.json` contains full kernel names, graph IDs, launch/capture/instantiate totals and log line references; `phase-counts.json` resolves CPU step spans across the autograd worker thread. The first script's backward `by_step=unknown` entries are superseded by the second script's timestamp-based step attribution. Per-step counts refer to trace labels `ProfilerStep#21`, `#22`, `#23`; each has four native microbatches. These are traced training updates, not a synthetic replay loop.

The first summarizer was deliberately stopped after rank 0 to replace a quadratic capture-API scan with a filtered scan. Rank 0's completed result was reused; remaining ranks completed with the same attribution rules. No native evidence was changed.

## Compiled boundaries and captured work

The hash prefixes in this table are rank 0; other ranks use different compiled hashes, with the same phase and kernel-family mapping preserved in the JSON evidence.

| Compiled hash prefix | Boundary | Launches per rank over three steps | GPU kernels per rank |
| --- | --- | ---: | ---: |
| `fo6d6ztf` | Region `0/0`, blockmask inference | 12 | 348 |
| `flhceb5h` | Region `1/0`, first dense block forward | 48 | 264 |
| `f3mrkhp6` | First dense block backward, including SAC recomputation | 96 | 660 |
| `f3vxowyb` | Region `2/0`, chunked loss forward | 96 | 192 |
| `fjpss73b` | Chunked loss backward | 96 | 96 |

Each microbatch therefore has one blockmask launch, four dense-forward partition launches, eight dense-backward partition launches, eight loss-forward chunk launches, and eight loss-backward chunk launches. Total: 348 actual graph launches per rank, with 1,560 correlated GPU kernels per rank.

Rank-0 `stderr.log` lines 24–128 show seven dense-forward scheduler partitions: safe scheduler partitions 0/2/4/6 map to tree functions `partition_0..3`; three interleaved partitions remain outside graphs because of `torchtitan.fp8_dense_gemm.default`. Lines 4319 onward show 16 dense-backward scheduler partitions, eight safe and eight unsafe, mapping to tree functions `partition_0..7`. Region `1/1` MoE compilations are marked entirely non-cudagraphable, consistent with the supplied exclusion. Loss region `2/0` has one safe forward and one safe backward partition; log lines 4712 onward show chunk shapes `[8,512,102400]` logits and `[8,512]` int64 labels.

Captured dense kernels include RMSNorm, rotary/complex and gather operations, BF16 GEMMs, FP8 quantization/transposes, SwiGLU forward/backward, and fused attention backward surrounding operations. Captured loss forward has two Triton log-softmax/NLL kernels per launch; backward has one fused log-softmax/NLL gradient kernel. This coverage exceeds blockmask-only capture, but excludes the unsafe CudaCherry dense GEMM calls. It is not whole-block, whole-model, optimizer, communication, or full-update capture.

## Capture/reuse counts

All ranks launch 348 graphs. Ranks 0/1/2/3/5 begin and instantiate 336 fresh graphs during the active trace; ranks 4/6/7 begin and instantiate 296. The remaining 12 launches per rank are consistently the already recorded blockmask graph. Among dense/loss launches, ranks 4/6/7 each have 40 launches without a new capture; this intermittent reuse does not become a settled phase. Dense backward captures on every call, on every rank.

The generated table below gives fresh captures as `dense forward / dense backward / loss forward / loss backward`. Launch denominators per step are always `16 / 32 / 32 / 32`.

| Rank | Step 21 | Step 22 | Step 23 |
| --- | --- | --- | --- |
| 0 | 16 / 32 / 32 / 32 | 16 / 32 / 32 / 32 | 16 / 32 / 32 / 32 |
| 1 | 16 / 32 / 32 / 32 | 16 / 32 / 32 / 32 | 16 / 32 / 32 / 32 |
| 2 | 16 / 32 / 32 / 32 | 16 / 32 / 32 / 32 | 16 / 32 / 32 / 32 |
| 3 | 16 / 32 / 32 / 32 | 16 / 32 / 32 / 32 | 16 / 32 / 32 / 32 |
| 4 | 12 / 32 / 24 / 24 | 16 / 32 / 32 / 32 | 12 / 32 / 24 / 24 |
| 5 | 16 / 32 / 32 / 32 | 16 / 32 / 32 / 32 | 16 / 32 / 32 / 32 |
| 6 | 12 / 32 / 24 / 24 | 12 / 32 / 24 / 24 | 16 / 32 / 32 / 32 |
| 7 | 12 / 32 / 24 / 24 | 12 / 32 / 24 / 24 | 16 / 32 / 32 / 32 |

Rank 6 repeats some dense/loss GPU graph IDs across profiled launches; ranks 4 and 7 sometimes replay distinct trees captured before the active trace. Graph ID uniqueness alone is therefore insufficient to count capture or reuse: the API events provide that distinction. Last profiled step 23 still has 336 fresh captures on ranks 0/1/2/3/5/6/7 and 316 on rank 4.

## Blocker and runtime concern

Logs repeatedly state `Re-recording function=partition_0, mode=FORWARD, reason=static input data pointer changed.` Rank 0 has 4,483 such forward guard messages plus one backward guard message; all-rank total guard-message counts range from 4,002 to 4,560. These are comparisons against previous trees, not counts of newly recorded graphs. Rank-0 lines 4736–4785 tie changing static inputs to attention norm, KV norm and `wkv_b` through the SAC wrapper; additional diagnostics identify other inputs. The trace proves capture churn; assigning every unstable address specifically to FSDP rematerialization versus allocator/tree lifetime would require further targeted evidence.

Recording continues through the end of training: rank-0 forward mismatch messages occur at 01:45:55 UTC and backward partition recording at 01:46:02 UTC. Twenty-five completed updates do not mean graph startup settled. Changing microbatch inputs and parameter/output storage addresses are encountered through the natural training path, without a synthetic input mutation fixture.

For the complete three-step active trace, per-rank summed host API durations are: graph launch 14.650–18.263 ms; begin capture 4.406–5.538 ms; end capture 0.517–0.761 ms; instantiate 48.585–59.729 ms. Driver kernel submission sums are 525.354–719.134 ms (`cuLaunchKernel`) and 147.447–171.563 ms (`cuLaunchKernelEx`); runtime `cudaLaunchKernel` sums are 609.078–4362.256 ms. These are host API duration sums, not device work or full-step latency. The three CPU step spans total 89.069–89.239 seconds per rank. Thus the measured graph APIs do not dominate integrated elapsed time. Python guard comparisons, verbose stack logging, tree management, capture execution and allocation costs are broader than these API intervals and remain a material cost concern.

Correlated graph-kernel duration sums are device-work totals, not busy-time unions or speed estimates. This review does not quantify whole-device compute, synchronization, communication, copies, or their exposed critical path; it makes no attribution or performance claim for those scopes. The profiler and exceptionally verbose guard logs perturb execution. Generated-code and kernel-selection differences from disabling mixed ordering/partitioning or graph controls remain an attribution caveat for any later off/on comparison.

## Finite next decision

Use the already approved conditional loss-only fallback as one bounded mechanism check: exclude the first dense block as well as the 17 MoE blocks from CUDA Graph capture, retaining all 18 compiled SAC blocks, FP8 and compiled graph-enabled loss. The dense block's unstable static inputs currently force new tree paths, with repeated downstream loss recordings. Removing that ancestor is a credible hypothesis for permitting reusable loss trees; **A007 does not prove that the fallback will persistently reuse loss graphs**. The blockmask graph may still replay separately and must not substitute for loss proof.

Proceed to matched unprofiled loss-only off/on research only if the fallback shows real loss forward/backward launches and stable reuse across natural four-microbatch updates on every rank, with no late recurring capture/instantiate pattern. Otherwise stop this candidate and report the remaining loss/tree input-lifetime blocker. Do not treat A007's recurring captures as startup or promote runnable status to a performance finalist. No fallback implementation or launch was performed by this review.
