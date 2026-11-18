# Timing drift during training

Updated 15 November 2026. **The hump and recovery recur with corrected real data
in all seven configurations. Their cause remains unresolved.** Every rank's
mean host cadence is lower in updates 121–140 than 41–60; decreases range from
6.77% to 8.08% across configurations. Neither the four fixes nor routing fusion
uniquely explains the common trend. The recorded clocks, GC and compiler
boundaries constrain attribution but do not identify the changing critical path.

## Evidence and acceptance boundary

The accepted [seven-run analysis](../schedule-embedding/timing-results-140.json)
(SHA256 `a50c7bca5149d5f27c099b6f25c1a9944955fd713d75e475766261fff4dc3a05`)
contains full 140-update cadence records for eight ranks, configurations,
identity receipts, GC events and compiler observations. All seven final-004
adjudications pass the same method. The separate
[envelope reconstruction](../schedule-embedding/artifacts/raw/analysis-execution-004/primary-envelope-check.json)
passes. [Compact drift evidence](current-data-drift-evidence.json) records the
calculations, exact added specialization keys, timestamp joins, telemetry hashes
and sampling windows. It is supporting evidence, not another benchmark report.

Finite acceptance for this investigation: answer the six drift questions below,
preserve the declared measurement window, reconcile current mechanisms with the
observations, and specify a bounded discriminator if attribution remains open.
No source/configuration/runner bytes, compiler settings, GC behavior or GPU state
were changed for this analysis. Documentation/evidence changes have no training
runtime effect. No additional training or instrumentation was run.

The primary result remains the CUDA-drained global envelope from after update
40 to after update 140, divided by 100. Each measured run contains 209,715,200
physical and 209,664,000 supervised global tokens. All intervals, GC events and
outliers remain included. Cadence is the host interval between callbacks after
native `Profiler.step`; it includes host work and distributed dependencies, and
is **not device compute time**. Separate device compute, host work, synchronization,
copies and communication durations are unavailable from these records.

### Historical evidence

The [historical comparison](historical-drift-comparison.json) retains the older
9.9–10.4% median-cadence decrease and old repeated-fixture observations. Its
14.7 MB/896-row cache discussion describes that historical fixture only. The
formerly called “current control” `fqla-native-fla-001` is historical here.
Those measurements and historical GC durations are not new campaign evidence.
Current results below use distinct real packed rows and the production tokenizer;
B003 is excluded because concurrent source scans contaminated its timing.

## 1. Corrected-data curves and feature dependence

First/last figures below are arithmetic means over 20 intervals on each rank,
then averaged across all eight ranks (160 intervals per window). Percent change
is `(last / first - 1) * 100`. These differ from the earlier A/B **median**
figures of approximately −7.45%/−7.76%. Primary full-window means also differ
because they use the drained envelope and all 100 measured updates.

| Variant | Primary mean update, s | Mean 41–60, s | Mean 121–140, s | Cadence change |
| --- | ---: | ---: | ---: | ---: |
| A | 14.01499 | 14.721160 | 13.588196 | -7.6962% |
| B | 13.37970 | 14.102222 | 12.971122 | -8.0207% |
| C | 13.09044 | 13.783556 | 12.669938 | -8.0793% |
| D0 | 12.92371 | 13.393954 | 12.486591 | -6.7744% |
| S | 12.84851 | 13.364457 | 12.396938 | -7.2395% |
| SE | 12.84793 | 13.330877 | 12.400428 | -6.9797% |
| SE-fused | 12.85377 | 13.380718 | 12.419461 | -7.1839% |

Ten-update **mean** cadence, pooled over eight ranks, in seconds:

| Updates | A | B | C | D0 | S | SE | SE-fused |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11–20 | 13.4541 | 12.8529 | 12.5661 | 12.3241 | 12.2581 | 12.2449 | 12.2595 |
| 21–30 | 13.7047 | 13.1192 | 12.8289 | 12.6630 | 12.6205 | 12.6018 | 12.7862 |
| 31–40 | 14.3055 | 13.7156 | 13.4250 | 13.2011 | 13.1549 | 13.1325 | 13.1429 |
| 41–50 | 14.7439 | 14.1227 | 13.8119 | 13.4805 | 13.4840 | 13.4322 | 13.5060 |
| 51–60 | 14.6984 | 14.0817 | 13.7552 | 13.3074 | 13.2449 | 13.2296 | 13.2554 |
| 61–70 | 14.3122 | 13.6647 | 13.3871 | 13.0869 | 13.0081 | 12.9448 | 12.9772 |
| 71–80 | 14.0143 | 13.3622 | 13.0813 | 13.1566 | 13.0677 | 13.0312 | 13.0460 |
| 81–90 | 13.9528 | 13.3346 | 13.0389 | 13.0613 | 12.9766 | 12.9615 | 12.9609 |
| 91–100 | 13.9101 | 13.2381 | 12.9940 | 12.9187 | 12.8283 | 12.8403 | 12.9147 |
| 101–110 | 13.7218 | 13.0653 | 12.7873 | 12.6633 | 12.5739 | 12.7524 | 12.5465 |
| 111–120 | 13.6198 | 12.9853 | 12.7087 | 12.5890 | 12.5077 | 12.4855 | 12.4919 |
| 121–130 | 13.5945 | 12.9783 | 12.6778 | 12.5184 | 12.4350 | 12.4258 | 12.4722 |
| 131–140 | 13.5819 | 12.9639 | 12.6621 | 12.4547 | 12.3588 | 12.3751 | 12.3667 |

All seven rise from 11–20 through 31–40 into 41–50, then broadly recover.
D0/S/SE/fused also show a smaller 71–80 increase; recovery is not monotonic.
The fixed 40-update warmup does not establish stationary cadence. The last
131–140 bin remains below 121–130 by 0.41% for SE and 0.85% for fused, so an
exact settling update cannot be established. These tails do not replace the
primary window or justify discarding additional updates.

A already shows the effect without the four fixes. B adds those fixes, including
BF16 expert-weight reuse and prefetch depth 32; C changes QLA backend; D0/S add
the split schedule/row-owner path; SE enables compact, owner_lookup and owner_add;
fused additionally enables routing. Actual stage receipts distinguish SE from
fused. The trend survives every configuration. This weakens an explanation
requiring one later feature, or the old repeated fixture alone. One fixed
A→B→C→D0→S→SE→fused sequence cannot separate cross-run system/cache drift from
feature effects or establish a general speedup.

## 2. Rank synchrony and the straggler question

Every one of the 56 rank/configuration first/last mean comparisons decreases.
Within each run the rank changes agree closely; the widest range is SE,
−6.9804% to −6.9767%. Pairwise Pearson correlations of 100 cadence intervals
are at least 0.9999788. Median within-update max-minus-min rank interval spreads
are 0.323–0.641 ms; their largest value is 19.082 ms in C.

The drift is shared at host callback boundaries. This does **not** show equal
local GPU work or rule out a changing straggler: one rank's receive/compute delay
can propagate through distributed synchronization. Callback interval maxima
cannot identify the responsible GPU, expert or communication peer. All-rank
readiness and local compute spans are missing.

## 3. Compiler, GC, input and system alignment

### Compiler boundaries

A through SE have stable observed Dynamo, Inductor, CuTe and Triton state on
all ranks between updates 40 and 140. Coverage is partial: asynchronous compiler
worker completion and native DeepGEMM JIT outside the guarded tested grouped APIs
are not fully observed. Stable boundary state weakens a covered first-use
specialization explanation for their common trend; it does not prove no compiler
or cache activity occurred.

SE-fused adds exactly 13 Triton process kernel entries. Inspection of the actual
encoded keys shows **all 13 are `_routing_metadata_kernel`**, on ranks 0, 3, 4,
5, 6 and 7; ranks 1/2 add none. They retain constexprs 16032 and 8, and include
both nondivisible (`i32,''`) and divisible (`i32,'D'`) scalar signatures:

| Rank | Added block-size/scalar signatures | Count |
| --- | --- | ---: |
| 0 | 1024/nondivisible, 512/nondivisible, 1024/divisible | 3 |
| 3 | 32/nondivisible | 1 |
| 4 | 512/nondivisible, 1024/divisible | 2 |
| 5 | 128/divisible, 512/divisible | 2 |
| 6 | 4096/nondivisible, 128/divisible, 512/divisible | 3 |
| 7 | 128/nondivisible, 128/divisible | 2 |

The 4096 entry uses eight warps; the others use four. Exact encoded keys remain
in the evidence JSON. Their first-observation interval is **(40,140]**; snapshots
do not give an exact update, duration or compilation-versus-disk-load distinction.
They cannot be assigned to a cadence spike. Their presence is concrete additional
fused cache evolution, not evidence that compilation causes all seven curves.
[PyTorch recompilation documentation](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.recompilation.html)
and the [pinned asynchronous compiler](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/torch/_inductor/async_compile.py)
explain why these distinctions matter.

### GC, host and allocator

Each run records two periodic collections per rank, lasting 0.55–0.99 seconds
across the campaign. The raw GC lines contain **no explicit update number**.
Joining their UTC timestamps to same-rank monotonic callback intervals places
all 112 periodic events in intervals 50 or 100. This is inferred temporal
alignment, supported by the current source's before-update period-50 call.
Both intervals show positive local excursions in each run. Their events remain
in the primary result; isolated collections cannot alone establish the smooth
multi-bin recovery. EMA starts at update 10000, outside this campaign.

All ranks have zero measured allocation retries and OOM counters. This rules out
those counted paths, not allocation cost or memory/host scheduling effects.
There are no time-resolved GIL, CPU scheduling, page-fault or NFS-wait records.
The [pinned PyTorch allocator API](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/torch/cuda/memory.py)
defines the recorded counters.

### Input

The current input has 102400 distinct packed rows, 25600 per domain, across 26
shards. Deterministic round-robin/DP8 gives fixed rank domains: ranks 0/4
Wikipedia, 1/5 mathematics, 2/6 Python and 3/7 Russian web. Text evolves within
each rank's domain. Consumption is 140×512=71680 rows, below the logical epoch
size; an epoch rollover is not an explanation. Evolving text, unique embedding
IDs and routing remain possible workload changes. Boundary prefetch readiness
does not measure per-update loader-pop waits. A uses depth 0; later variants
use depth 32, yet both have the trend. No observed input stall series establishes
or excludes a shared input/cache contribution. The historical tiny-fixture cache
argument cannot be transferred to these rows.

### GPU telemetry

Only the clean A segment of `gpu-140-003.csv` and B-through-fused segment of
`gpu-140-004.csv` were joined to current measurement windows. Timestamps are UTC;
same-rank wall-minus-monotonic offsets differ by only 1.6–2.8 microseconds between
window boundaries. Samples are approximately 60 seconds apart, about four or
five updates, and cannot resolve individual spikes. The following are unweighted
instantaneous sample means across eight GPUs inside the early/late 20-update
windows, not time-weighted energy or clock integrals:

| Run | Full-window samples | SM MHz, early → late | Power W, early → late | Temperature C, early → late |
| --- | ---: | ---: | ---: | ---: |
| A | 192 | 1849.9 → 1834.1 | 638.3 → 662.7 | 62.5 → 63.9 |
| B | 176 | 1941.4 → 1616.7 | 627.5 → 679.8 | 61.2 → 64.5 |
| C | 168 | 1877.3 → 1771.4 | 635.2 → 671.8 | 62.9 → 64.7 |
| D0 | 168 | 1852.5 → 1867.0 | 642.2 → 674.9 | 61.9 → 64.3 |
| S | 176 | 1783.9 → 1749.8 | 648.1 → 680.1 | 63.0 → 64.9 |
| SE | 168 | 1824.8 → 1809.4 | 652.8 → 675.6 | 63.2 → 64.2 |
| SE-fused | 168 | 1760.2 → 1936.9 | 649.7 → 671.6 | 63.2 → 64.3 |

Memory clocks remain 2619 MHz in all selected samples. SM clocks span
1425–1965 MHz campaign-wide; power and temperature are not constant. Every
selected clock-event mask is `0x0000000000000004`; this observed mask must not be
reported as an absence of clock limiting. Query semantics are described by
[NVIDIA SMI](https://docs.nvidia.com/deploy/nvidia-smi/index.html).
There is no common rise in sampled SM clocks as cadence recovers: most variants
have lower late averages; fused has a rise. This weakens a universal explanation
based only on clocks ramping up. It does not isolate power, thermal, phase or
system effects: coarse samples alias compute/communication phases and cannot
establish causality or equal device work.

## 4. MoE mechanisms: support, contradictions and unknowns

The [current-source research](../schedule-embedding/drift-research.md) pins the
inspected source/dependencies and external primary citations. Campaign identity
receipts freeze source under SHA256
`3b3244ae1e759111b8240af929715074b751c946881202a9ff95e9f3b9918ffd` and retain the
original capacity overlay, excluding the contributor's capacity edits.

Source-supported mechanisms shared by all seven are:

- `register_moe_load_balancing_hook` updates bias every optimizer update:
  `0.001 * sign(mean-load) * clamp(1+max_vio,1,5)`, recentered to zero mean.
  It is independent of LR warmup 2000 and measurement warmup 40. SAC
  recomputation may contribute counts; nonuniform effects cannot be assumed away.
- `TokenChoiceTopKRouter.forward` adds bias to FP32 sigmoid scores before top-8
  selection, then gathers unbiased, normalized weights scaled by 2.5. Learned
  weights, bias and evolving text can all change expert/destination choices.
- FP8 grouped experts consume actual payload and 128-row padded expert groups;
  local group sizes can change GEMM efficiency and the distributed critical path.
- HybridEP blocking metadata/dispatch and combine readiness can expose peer
  waits. [Pinned DeepEP metadata processing](https://github.com/deepseek-ai/DeepEP/blob/1b8f467/csrc/hybrid_ep/executor/executor.cu)
  supports this mechanism; a long communication span can contain waiting.
  [Loss-Free Balancing](https://arxiv.org/abs/2408.15664) supports load-adjusted
  bias as context, not a prediction of this hump or its settling time.

The current capacity bound rules out **actual dropping** under this unchanged
recipe: 8192 local tokens×top-8×EP8 = 524288 global routes per microbatch;
`50000//8×128 = 800000` is the destination-group threshold. No group can exceed
all global routes. Capacity predicates and reductions still execute. Across 128
experts padding adds at most 128×127=16256 rows, or 3.100586% of useful routes.
That bounds added route volume, not runtime: it cannot by itself explain or rule
out an approximately 7–8% whole-update timing change, because skew and peer
readiness can change the critical path at fixed total volume.

Recurrence with corrected input and all configurations is consistent with a
shared routing/balance mechanism. It is not direct evidence that imbalance
actually decreases. No early/late per-layer expert counts, padded M,
max-violation, destination volumes or receive/compute readiness series was
recorded. Bias updates alone do not predict a hump; evolving text and system
conditions remain alternatives. Dropping, logical-epoch rollover, the old tiny
fixture alone and a fused-only specialization are contradicted as universal
explanations under the established bounds. **MoE imbalance remains a plausible,
unproven hypothesis; it is not an established leading cause from timing alone.**

## 5. SE versus SE-fused settling and excursions

The primary full-window mean is 12.84793 s for SE and 12.85377 s for fused:
fused physical TPS is 0.04544% lower in this one fixed sequence. This does not
establish a general difference, and no window/outlier is removed to favor either.
Both have the common hump and continuing tail decrease. Fused has additional
process cache entries, but no timestamped key events establish additional
settling duration.

SE itself has the largest isolated non-GC excursion: update 106 averages
14.699588 s across ranks, versus 12.533633 s for the median of neighboring
updates 104/105/107/108, an excess of 2.165955 s. It has no timestamp-aligned GC
event and its observed compiler boundaries are stable under partial coverage;
its cause is unknown. Fused's largest comparable positive excursion away from
GC neighborhoods is update 95: 13.239780 s versus 12.831077 s neighboring median,
+0.408704 s; further excursions occur at 82 and 130. The supporting local-event
search uses offsets ±1/±2, excludes candidates within two updates of 50/100,
and leaves **all** primary intervals intact. No cause or performance correction
is inferred from this event search. Fused does not uniquely introduce timing
outliers, and these records cannot tie its excursions to specific new keys.

## 6. Precise missing discriminator and bounded next test

Attribution is unresolved because routing/load and exposed readiness are not
recorded jointly over the changing interval. An update-21 profile can identify
active operations but cannot decide what changes between updates 41 and 140.
Another aggregate timing matrix would not supply that missing information.

If separately authorized, use one 140-update diagnostic of SE with the same
frozen source/overlay, initial-state procedure, seed 2048, dataset/tokenizer
hashes, ordering, production shape, FP8/BF16 arithmetic, SAC/compile, accumulation,
optimizer and LR schedule. Preserve normal GC and clocks. Sample only updates
31–40, 41–60 and 121–140 on all eight ranks, per MoE layer: unpadded expert counts,
padded M, destination totals, max/mean violation and bias summary; dispatch-ready,
expert-done and combine-ready events; and exposed loader-pop wait. Record exact
command, resolved identities and this sampling plan beside the diagnostic.
Reuse blocking CPU metadata where valid, defer/batch device exports, and add no
per-step synchronizing reads or training collectives. Review instrumentation
cost before execution; this would be mechanism evidence, not a replacement TPS
measurement. Timestamp compiler first-use/load observations and GC alongside
these windows if a reviewed low-cost path is available.

A reduction in rank-critical routed work/imbalance together with shorter receive
or expert-ready spans, while input wait and system conditions stay comparable,
would support routing/communication settling. Stable routing work with declining
loader wait would instead support an input contribution. Stable routing/input
with aligned clock/host events would motivate the corresponding system path.
Padding changes alone, without matching rank-critical spans, would remain
insufficient. If several vary together, report that confounding rather than
claiming a cause. This proposal does not authorize source instrumentation or
additional GPU work. The present six-question investigation is complete with
causal attribution explicitly open.
