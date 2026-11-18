# Current state

Updated: 19 November 2026. This file owns integrated results, observed costs,
evidence-based priorities and unknowns. [SOTA ideas](sota-ideas.md) owns
hypotheses/proposals; [the plan](../planning/README.md) owns future execution.
Update this file when new evidence changes a conclusion.

## Production baseline and measured throughput

The owner accepted **S + GPU capacity selector + five Hopper fusions** on
19 November 2026 as the current production baseline for the eight-layer
crop on eight H100s. [Recipe and launcher](baseline.md#current-final-task-baseline)

TPS counts global physical input tokens per second. Memory is the maximum
PyTorch allocator peak across eight ranks within the measurement window.
The 40+100 rows cover updates 41–140 and 209,715,200 physical tokens each.

| Variant | Warmup + measured | TPS | Mean update, s | Peak allocated, GiB | Peak reserved, GiB | TPS vs A |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A · original baseline | 40+100 | 149,636 | 14.015 | 70.387 | 71.670 | — |
| B · four fixes | 40+100 | 156,741 | 13.380 | 68.520 | 69.807 | +4.75% |
| C · FlashQLA `external_gate` | 40+100 | 160,205 | 13.090 | 68.554 | 69.746 | +7.06% |
| D0 · MTP/layer-0 FFN splits and row-owner embeddings | 40+100 | 162,272 | 12.924 | 66.963 | 69.201 | +8.44% |
| S · GFSDP schedule repairs | 40+100 | 163,221 | 12.849 | 66.954 | 69.201 | +9.08% |
| SE · S with three embedding optimizations | 40+100 | 163,229 | 12.848 | 66.924 | 68.334 | +9.08% |
| SE-fused · SE with embedding routing fusion | 40+100 | 163,155 | 12.854 | 66.962 | 68.352 | +9.03% |
| S · selector control | 10+10 | 170,321 | 12.313 | 65.887 | 67.580 | — |
| S + GPU selector · selector candidate | 10+10 | 171,475 | 12.230 | 65.889 | 67.561 | — |
| S + GPU selector · fusion control | 40+100 | 164,537 | 12.746 | 66.998 | 69.260 | +9.96% |
| **S + GPU selector + five fusions · production** | **40+100** | **166,519** | **12.594** | **66.986** | **69.162** | **+11.28%** |

The five fusions gained **1.20% TPS** in the matched 40+100 pair and lowered
peak allocated by **12.46 MiB** and reserved by **100 MiB**. The selector gained
**0.68%** in its separate 10+10 pair. Chaining the measured A→S, selector
and fusion gains gives **+11.14%**. Comparing the final and original 40+100
runs across allocations gives **+11.28% TPS**, **−3.40 GiB allocated** and
**−2.51 GiB reserved**. The historical S and fusion control used the same
data, tokenizer, seed, runtime versions and pre-fusion GIFT bytes. Their
training configs differ only in enabling the GPU selector.
[A–SE-fused results](../runs/schedule-embedding/timing-results-140.md) ·
[Selector pair and trace](../runs/capacity-selector/report.md) ·
[Fusion pair and audit](../runs/kernel-fusion/report.md#combined-five-at-40100)

### Fixed-recipe control for expert GFSDP candidates

On 19 November 2026, the fixed production recipe at `1f02311a5` passed the
unprofiled 10+10 control on the eight-H100 `sr008-gpu1` allocation. The
resolved configuration retained EP8/eFSDP1, dense FSDP8, the GPU selector
and the five fusions. The ten measured updates processed 20,971,520 physical
input tokens in 120.603 seconds, or **173,889 tokens/s**. Maximum-rank peak
allocated and reserved memory were **65.958 GiB** and **67.854 GiB**.
The compiler-cache observer reported stable metadata across its observed
window. This control uses the 10+10 protocol, with startup outside the
timing window.

The server receipt is under
`runs/kernel-fusion/artifacts/raw/efsdp-control-ep8-001-timing/`.
Its `summary.json` retains all-rank timings and allocator peaks, and
`completion.json` records PASS and the resolved config hash. The control
profile is captured separately under `efsdp-control-ep8-001-trace`.
[Active campaign and acceptance](../planning/README.md#active-assignment-expert-gfsdp-optimizations)

## Consolidated rank-0 analysis: observed costs and priorities

This section consolidates the studied S trace, source/grid arithmetic and prior
component results. It ranks measured areas for investigation; it makes no
implementation choice or predicted speedup. The hypotheses and proposals remain
in [SOTA ideas](sota-ideas.md); the [plan](../planning/README.md) owns execution.
This documentation update changes no training source, configuration or runtime.

### Evidence identity and timing scope

The local `artifacts/4fix_qla_embed.json` is the S `diagnostic-140-004` rank-0
capture of optimizer update 21: three of 32 microbatches, world size eight,
NVIDIA H100 80GB HBM3, 132 SMs. Its SHA256 is
`c921f19d52934fe7b597a79b81f4a179ec1c0dde2bf4616f9a9e4973163e6b95`.
Supporting local files are `artifacts/microbatch_costs.json` and its same-stem
analysis script, `artifacts/hybrid_ep_rank0_quant.json`, and
`artifacts/gemm_roofline.json`; they are ignored artifacts, so essential results
are reproduced here. The source reviewed for this cost ledger was
`e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f`. The follow-up selector campaign
recorded matched S runtime and GIFT fingerprints and resolved training
controls. S has a separate run history from the corrected-input A/B/C/D
campaign below. [S comparison receipts](../runs/capacity-selector/report.md#native-1010-result).

Attribution covers all 9,317 physical kernel/memcpy/memset events exactly once,
with zero missing CPU/runtime links and zero category/phase sum residuals.
Fifteen prefix events total 0.057184 ms; 9,302 events belong to the three
microbatches. Nested GPU annotations are excluded. Summed durations count work
on every stream; interval unions count occupied elapsed time. Communication
kernels include device waits/spin time. Occupancy is not SM utilization.
Host waits overlap GPU activity and must not be added to device totals.
Neither one rank nor this three-microbatch prefix establishes the all-rank,
full-update critical path or steady-state throughput.

### Microbatch elapsed time, work, overlap and host waits

All entries are milliseconds. GPU span runs from the first to last correlated
physical event for that microbatch, including its GPU tail after the CPU
annotation. Idle is span minus the all-stream event union. Compute includes
noncommunication kernels; copies are separate. Overlap is the interval
intersection of compute and communication, not a removable-time estimate.
Event-sum minus union credits all simultaneous physical events; it is a
different quantity from compute/communication overlap.

| Scope | MB1 | MB2 | MB3 | Mean |
| --- | ---: | ---: | ---: | ---: |
| GPU span | 388.875 | 381.506 | 385.623 | 385.335 |
| Physical-event sum | 420.577 | 425.983 | 440.385 | 428.982 |
| All-stream GPU union | 372.172 | 369.116 | 373.528 | 371.605 |
| Event-sum minus union (overlap credit) | 48.405 | 56.868 | 66.857 | 57.377 |
| All-stream idle | 16.703 | 12.390 | 12.096 | 13.730 |
| Compute union | 313.571 | 314.384 | 314.023 | 313.993 |
| Communication union | 99.411 | 106.079 | 121.957 | 109.149 |
| Compute/communication overlap | 41.982 | 51.738 | 62.837 | 52.186 |
| Communication without compute | 57.429 | 54.341 | 59.120 | 56.963 |
| Memcpy/memset union | 6.667 | 0.469 | 0.482 | 2.539 |
| CPU F/B annotation | 371.596 | 364.867 | 367.346 | 367.936 |
| GPU tail after CPU annotation | 17.547 | 16.888 | 18.566 | 17.667 |
| CUDA API duration sum | 176.476 | 204.515 | 205.059 | 195.350 |
| Host synchronization sum | 156.715 | 186.048 | 188.346 | 177.037 |

The whole-trace physical-event sum is 1,287.003 ms and the all-stream union is
1,114.873 ms. Whole-trace first-to-last physical-event elapsed time is
1,165.651 ms, including the prefix and inter-microbatch gaps. These scopes differ
from mean microbatch span (385.335 ms). The 87 host synchronization calls across the
microbatches total 531.110 ms; this is host waiting, not additional GPU work.

### Complete exclusive device-cost ledger

Each row is one exclusive category. Times are physical-event sums in
milliseconds for each microbatch and their mean; the 15 prefix events are
excluded and remain recorded separately above. The 26 rows total 420.577 /
425.983 / 440.385 ms, or 428.982 ms per microbatch. Category sums exceed elapsed
time because streams overlap. Ambiguous owners remain explicit.

| Category | MB1 summed ms | MB2 summed ms | MB3 summed ms | Mean summed ms/MB |
| --- | ---: | ---: | ---: | ---: |
| LM-head GEMMs | 68.515 | 69.606 | 65.628 | 67.916 |
| Routed expert GEMMs | 49.996 | 50.440 | 50.521 | 50.319 |
| FSDP all-gather | 22.358 | 34.238 | 36.536 | 31.044 |
| HybridEP combine | 30.352 | 29.716 | 30.122 | 30.063 |
| FSDP reduce-scatter | 26.959 | 23.754 | 36.269 | 28.994 |
| Norm, router and activation kernels | 28.408 | 28.612 | 28.787 | 28.603 |
| FP8 scales, quantization and layout | 21.978 | 22.019 | 22.065 | 22.021 |
| Other elementwise, reduction and layout kernels | 13.323 | 22.884 | 22.655 | 19.621 |
| Combined-block dense GEMMs: attention/shared-expert owner ambiguous | 19.255 | 19.333 | 19.455 | 19.347 |
| GDN attention core and local operations | 19.177 | 19.311 | 19.125 | 19.204 |
| Copy/cast/fill kernels | 22.024 | 16.553 | 15.613 | 18.063 |
| Other attention projection GEMMs | 15.921 | 15.706 | 15.960 | 15.862 |
| Dense FFN GEMMs | 15.032 | 14.666 | 15.952 | 15.217 |
| HybridEP dispatch | 13.979 | 13.390 | 13.500 | 13.623 |
| MLA attention core | 11.253 | 10.971 | 11.739 | 11.321 |
| MTP attention projection GEMMs | 9.656 | 9.370 | 10.419 | 9.815 |
| FP32 GEMMs: attention/router owner ambiguous | 5.723 | 5.716 | 5.745 | 5.728 |
| Loss kernels | 4.706 | 4.731 | 4.618 | 4.685 |
| Other NCCL communication | 3.607 | 3.931 | 4.647 | 4.062 |
| Layer-0 attention projection GEMMs | 3.928 | 3.943 | 3.972 | 3.948 |
| Identified shared-expert GEMMs | 3.746 | 3.719 | 3.745 | 3.737 |
| Memcpy events | 6.333 | 0.138 | 0.140 | 2.204 |
| HybridEP helper kernels | 2.727 | 1.662 | 1.521 | 1.970 |
| Embedding | 1.258 | 1.217 | 1.282 | 1.252 |
| Memset events | 0.333 | 0.331 | 0.342 | 0.335 |
| EP padding/layout preparation | 0.028 | 0.028 | 0.028 | 0.028 |

The 19.347 ms combined-block and 5.728 ms FP32 ambiguous GEMMs are not silently
assigned to routed experts or attention. The identified shared-expert row is
therefore not a complete shared-expert backward cost.

### Forward, SAC replay and actual backward

These phases partition the same three-microbatch events, excluding the prefix;
times are summed milliseconds, not extra costs to add to the ledger.

| Phase | MB1 summed ms | MB2 summed ms | MB3 summed ms | Mean summed ms/MB |
| --- | ---: | ---: | ---: | ---: |
| Original forward | 110.087 | 108.095 | 106.477 | 108.220 |
| Outer SAC replay | 59.421 | 59.701 | 61.679 | 60.267 |
| Backward after outer replay | 247.130 | 254.218 | 268.337 | 256.561 |
| Intrinsic FLA VJP reconstruction | 2.729 | 2.735 | 2.723 | 2.729 |
| Mixed backward preparation | 1.018 | 1.039 | 0.975 | 1.011 |
| Other local backward replay | 0.192 | 0.195 | 0.193 | 0.194 |

Outer SAC replay is 60.267 ms/MB. FLA's intrinsic VJP reconstruction is a separate
2.729 ms/MB; removing outer replay would not establish removal of intrinsic VJP
work. The SAC label uses compiled-backward replay boundaries plus replayed
HybridEP/grouped-GEMM operations. Fused mixed helpers are attributed to an
execution phase, not proven avoidable SAC overhead.

SAC composition is a regrouping of the 60.267 ms, with no additional work:

| SAC component | Mean summed ms/MB |
| --- | ---: |
| Attention projection GEMMs | 12.360 |
| EP main dispatch + combine | 11.220 |
| Norm/router/activation | 7.499 |
| Routed expert GEMMs | 5.409 |
| FP8 scale/quant/layout | 4.602 |
| GDN core/local + MLA core | 6.171 |
| Other NCCL communication | 3.343 |
| Dense FFN + identified shared-expert GEMMs | 4.718 |
| FP32 GEMMs, owner ambiguous | 1.353 |
| Other elementwise/reduction/layout | 2.744 |
| EP helpers/preparation + copies/fills | 0.847 |

### Routed GEMM work and arithmetic rates

The 147 routed GEMMs include 42 original-forward calls, 21 SAC replay calls,
42 dgrad calls and 42 wgrad calls. Counts below cover all three microbatches;
FLOPs and milliseconds are per-microbatch means. Arithmetic is source/grid
reconstruction: two FLOPs per FMA, including padded rows, divided by summed
kernel durations. It excludes quantization, layout, communication, copies,
submission and gaps. It is not actual hardware-counter utilization or full-step
MFU.

The [NVIDIA H100 specification](https://www.nvidia.com/en-us/data-center/h100/)
quotes 1,979 BF16 / 3,958 FP8 TFLOP/s **with sparsity**. The dense references used
here are 989.5 BF16 / 1,979 FP8 TFLOP/s. Routed kernel templates use 116 SMs;
linear scaling of the FP8 reference by 116/132 gives 1,739.121 TFLOP/s. This is
an optimistic SM-budget reference. Actual clocks, power limits and tensor/SM
counters were not measured.

| Routed role | Calls | TFLOPs/MB | Kernel ms/MB | Achieved TFLOP/s | % of 116-SM FP8 reference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Up/gate forward | 21 | 10.852 | 8.363 | 1297.7 | 74.6% |
| Down forward | 21 | 5.426 | 4.701 | 1154.3 | 66.4% |
| Up/gate SAC replay | 3 | 1.693 | 1.305 | 1297.4 | 74.6% |
| Down SAC replay | 18 | 4.667 | 4.104 | 1137.2 | 65.4% |
| Down dgrad | 21 | 5.426 | 4.262 | 1273.1 | 73.2% |
| Up/gate dgrad | 21 | 10.852 | 8.793 | 1234.2 | 71.0% |
| Down wgrad | 21 | 5.426 | 6.527 | 831.3 | 47.8% |
| Up/gate wgrad | 21 | 10.852 | 12.264 | 884.9 | 50.9% |
| All routed GEMMs | 147 | 55.196 | 50.319 | 1096.9 | 63.1% |
| Wgrad subtotal | 42 | 16.278 | 18.791 | 866.3 | 49.8% |

For the head, MTP folds two heads × batch two into four rows of sequences;
two loss chunks give M=8192, N=128256, K=4096 per matrix call. Six BF16 GEMMs
per microbatch (forward, dX and dW for both chunks) perform 51.643 TFLOPs in
67.916 ms: 760.385 TFLOP/s, 76.85% of the dense BF16 reference. These shapes
come from the workload and local [MTPLoss/chunked-loss](../gift/torchtitan/components/loss.py)
and [linear projection](../gift/torchtitan/models/common/embeddings/token_debeddings.py)
source, not captured input dimensions.

For experts, [SwiGLU quantization](../gift/torchtitan/ops/float8/triton_kernels/fused_swiglu_quantization.py)
uses a 64-row launch block. Its grid and the [grouped-GEMM payload path](../gift/torchtitan/models/common/moe/experts/gmm.py)
reconstruct the following summed padded
receiver-local rows M per layer; D=4096 and expert width=1536 remain fixed.
Rows vary by layer and microbatch. These are not exact true-token counts or
per-expert load-balance measurements.

| MoE layer | MB1 padded M | MB2 padded M | MB3 padded M |
| --- | ---: | ---: | ---: |
| L1.GDN_MoE | 67,712 | 66,944 | 67,200 |
| L2.GDN_MoE | 66,944 | 67,200 | 67,456 |
| L3.MLA_MoE | 65,536 | 65,792 | 65,536 |
| L4.GDN_MoE | 63,744 | 64,384 | 64,384 |
| L5.GDN_MoE | 59,776 | 60,800 | 60,416 |
| L6.GDN_MoE | 59,136 | 59,648 | 60,416 |
| L7.MLA_MoE | 46,720 | 47,872 | 46,080 |

The head's rated-peak arithmetic floor is 52.191 ms, leaving a 15.726 ms
kernel-only ceiling relative to this trace. Expert wgrad's 116-SM reference
floor is 9.360 ms, leaving a 9.431 ms ceiling. Neither is an attainable-gain
forecast, and neither converts directly to step latency or TPS. FP8 operands
and FP32 scales are source-verified; BF16 output/storage does not turn those
expert tensor multiplies into BF16 work. Installed DeepGEMM accumulation
behavior was not independently reverified.

Backward FP8 preparation across the whole model includes grouped requantization
4.257 ms/MB (81 launches over the capture), scaling-aware FP8 transpose
2.291 ms/MB (231), and last-two-dimension transpose 1.395 ms/MB (78). These are
subsets of the ledger's FP8 row, not additions or exclusively MoE costs.

### HybridEP: main kernels, preparation, host scopes and overlap

All EP kernels are on stream 7. The 120 main dispatch/combine calls total
131.059 ms over the capture, or 43.686 ms/MB. Including 318 helper kernels gives
136.969 ms total (45.656 ms/MB). Correlating all device events launched inside
EP CPU scopes, including preparation, gives 849 events / 152.305 ms total
(50.768 ms/MB). CPU scope durations are submission/wait spans and overlap GPU
execution. The table separates these scopes; counts cover three microbatches
and times are per-microbatch duration sums.

| EP operation | Calls | Main GPU ms/MB | All correlated GPU ms/MB | CPU scope ms/MB |
| --- | ---: | ---: | ---: | ---: |
| forward dispatch | 21 | 4.493 | 8.017 | 6.899 |
| forward combine | 21 | 11.078 | 11.195 | 0.525 |
| recompute dispatch | 18 | 3.856 | 6.690 | 5.556 |
| recompute combine | 18 | 7.364 | 7.469 | 0.528 |
| backward dispatch_bwd | 21 | 5.274 | 5.659 | 1.270 |
| backward combine_bwd | 21 | 11.621 | 11.737 | 0.600 |

EP helper totals over the whole capture are: NVLink gather 2.387 ms (39 calls),
scan 1.419 ms (39), device-sync kernel 1.940 ms (120), and expected-value update
0.164 ms (120). Correlated preparation includes rowwise quantization 2.305 ms
(60), block quantization 12.471 ms (78), routing metadata 0.248 ms (39), padding
0.083 ms (39), and BF16 fill 0.040 ms (39). These totals are already included in
the correlated scope or exclusive ledger; do not add them again.

Only 1.818 ms of the main EP kernels overlaps FSDP all-gather across all three
microbatches; compute overlap is 0.025 ms and FSDP reduce-scatter overlap is zero.
Helper expected-value updates add 0.003 ms of all-gather overlap. Main EP work
is therefore largely exposed against the measured rank-0 compute timeline;
this does not prove a collective dependency or a removable all-rank wait.

Inside the 39 original/recompute dispatch scopes, `cudaStreamSynchronize` totals
8.871 ms (4.907 forward + 3.963 recompute). Separately, the 39 preceding
`aten::is_nonzero` scopes total 475.860 ms, with 474.829 ms inside their runtime
synchronizations. The preceding scalar scopes end shortly before dispatch but
can wait for earlier queued work; neither host total is an estimate of saved
time, nor can it be added to the device communication budget.

The inspected [capacity implementation](../gift/torchtitan/models/common/moe/dispatch/capacity.py)
and [MoE configuration](../gift/torchtitan/config/moe.py) give EP8/B2/S4096/top-8/E128
at most 524,288 global routes versus a `group_gate` budget of 6,250 × 128 = 800,000. This makes
the gate capacity inactive for this crop's source-level geometry; it does not
remove valid-mask semantics or establish a production EP32 result. The existing
nonblocking path's nominal 524,288-row capacity is below the conservative
526,208-row padded bound. Source explicitly rejects fused nonblocking plus
token drop because of a count/NaN bug, and the grouped-wgrad `.tolist()` fallback
synchronizes. These are existing integration constraints, not an approved flag
change or a speedup estimate.

### Memory evidence is scoped separately

The [reported SE comparison](sota-ideas.md#measured-headroom-and-quack-component-evidence) records
66.924 / 68.334 GiB allocated/reserved peaks versus A's 70.387 / 71.670 GiB:
reductions of 3.463 / 3.336 GiB. This is a separately reported 140-update
comparison, not a local memory receipt for the S trace or guaranteed reusable
headroom at every phase.

The isolated [Quack LM-head report](../runs/quack-lmhead/report.md) measures
native component peaks of 9.370 / 9.943 GiB versus patched Q4096's
6.251 / 7.109 GiB: 3.119 GiB less allocated, without robust direct latency gain.
Q8192's paired mean elapsed decrease is 0.94–1.00%, close to the 0.623% A/A
mean difference; its component allocated peak is 7.297 GiB. Those processes
include a materialized untied head and FP32 dW collector, not native GFSDP state.
Both Quack variants have larger local gradient error than native against FP32;
full-training memory, convergence and continuation are not established. Their
memory reductions cannot be added to the SE reduction without lifecycle-matched
native measurements. The historical 1,024-token FLCE port below reduced
allocated peak by 2.6384 GiB but regressed TPS by 3.2898%.

### Evidence-based priorities and lower-priority areas

This ordering identifies measured costs and unresolved evidence, not proposed
mechanisms, implementation authorization or projected gains.

| Priority | Area | Evidence and limit |
| --- | --- | --- |
| First | Memory lifetimes and outer SAC replay | 60.267 ms/MB of observed replay spans communication, attention, GEMMs and preparation. Separate component/run memory reductions exist; which saved tensors affect the native peak is unmeasured. No replay budget is declared wholly removable. |
| First | Exposed MoE communication | Main EP kernels cost 43.686 ms/MB with only 1.818 ms all-gather and 0.025 ms compute overlap across the capture. Host waits and source constraints require separate interpretation. |
| Next | Expert wgrad and adjacent FP8/layout work | Wgrad is 18.791 ms/MB at 866.3 TFLOP/s, 49.8% of the 116-SM reference, versus 66.4–74.6% for original forward roles. Backward requantization/transposes have separately measured budgets. The limiting hardware mechanism is unknown. |
| Lower for generic kernel tuning | LM head and already efficient routed forward GEMMs | The head is large at 67.916 ms/MB but reaches 76.85% of rated dense BF16; up/gate forward reaches 74.6% and down forward 66.4% of the 116-SM FP8 reference. These rates bound arithmetic-only room and rank generic tuning below the areas above; they do not reject targeted GEMM or head work. |
| Lower for isolated latency work | CE and GDN intrinsic replay | All loss kernels sum to 4.685 ms/MB; intrinsic VJP replay is 2.729 ms/MB. Neither explains the 60.267 ms outer SAC budget. The Quack component has no robust Q4096 latency gain. |
| Lower without exposed-wait evidence | Raw FSDP collective duration | AG + RS sum to 60.038 ms/MB outside the main compute stream and already overlap other work. This raw sum is not a 60 ms latency opportunity; only an affected exposed dependency can support that claim. |

Remaining unknowns include all-rank and full-update critical paths, actual
clocks/counters, per-expert token counts, native tensor lifetimes and global
peak effects, and repeated unprofiled throughput results.

## Seven-configuration campaign: measurements complete; layers.7 overlap unmet

The revised 15 November goal selects A/B/C/D0/S/SE/SE-fused timing runs, each
with 40 warmup and 100 measured complete optimizer updates, then four separate
update-21 diagnostics. The [campaign report](../runs/schedule-embedding/report.md#revised-measurement-assignment)
owns finite acceptance and evidence. New SE includes compact coalescing, ordered
owner accumulation and direct lookup; SE-fused also enables embedding routing
preparation. Historical SE receipts contain all four stages and retain their
original labels. Existing capacity edits remain preserved and excluded.
Both implementations are integrated and focused GPU checks pass: 71 GFSDP
CUDA/NCCL cases, local kernel bit parity, and optimized eight-rank embedding
autograd/state checks. These checks establish implemented and tested code; they
do not establish the original layers.7 overlap objective. The independent
review finding is resolved.
The five-stage eight-rank component ladder passed with lower measured peak
memory and zero retry/OOM deltas. Both native accumulation-32 smokes pass;
D0/SE have identical 6,973,565,952-byte communication pools per rank and exactly
the two intended remaps. The historical short-window timings completed; they
do not replace the new fixed-window campaign. On resumption, no training
children remained and all eight H100s were idle. Minimal native stage selection,
targeted CPU/CUDA checks and the independent launch review pass. All seven fixed
40+100 native timings now have final `ADJUDICATED_PASS` proofs under one reviewed
offline method. The five raw nullable-selector reporting failures remain
preserved. Primary envelopes, token totals and memory were independently
reconstructed; the report links full cadence, plots and every comparison.
SE-fused/SE is −0.0454% physical throughput in this one sequence, without an
established repeatable effect. Corrected data retains the shared hump/recovery;
causal attribution remains unresolved. Four separate update-21 captures completed
on all eight ranks with the same frozen source/configuration/harness and saved-D
cache. All eight actual MCP imports, event counts, annotations, matched/inverse
launch flows and asynchronous tails pass under the separate
[final-005 receipt](../runs/schedule-embedding/trace-analysis/capture-acceptance-final-005.json).
The native proof table, [comparison report](../runs/schedule-embedding/trace-analysis/comparison-004.md),
[comparison JSON](../runs/schedule-embedding/trace-analysis/comparison-004.json),
and rank-[0](../runs/schedule-embedding/trace-analysis/four-way-004-rank0.html)/
[1](../runs/schedule-embedding/trace-analysis/four-way-004-rank1.html) timelines
are summarized in the [campaign report](../runs/schedule-embedding/report.md#revised-perfetto-profile-findings).
The independent [final integration review](../runs/schedule-embedding/final-campaign-review.md)
passes with no unresolved material finding.

Across 24 semantic cases, actual layers.7 compiled-kernel overlap with
MTP-parent reduce-scatter is zero; candidate kernels start 2.336–56.065
microseconds after it ends, with the remaining wait cause unknown. Both MTP
parameter all-gathers precede head reduce-scatter; norm packing starts before
parent reduce-scatter ends in all 18 S/SE/SE-fused cases. Early FFN reduce-scatter
overlaps generic local compute, but positive ownership of the overlapping
MTP-prefix compiled compute is **UNESTABLISHED** for both that reduction and
layers.7 dense all-gather. The AG-adjacent stream-7 inactivity proxy is not a
head-RS-to-compiled-FFN interval; its ranges do not support a causal regression
claim. The campaign's measurements and review are complete, while the original
layers.7 overlap objective remains unmet.

The unprofiled SE-fused/SE observation remains provisional at −0.04544% physical
TPS, +5.841 ms/update, +38.49 MiB maximum allocated and +18 MiB maximum reserved.
All seven timing variants retain a 6.77–8.08% first-to-last-20 mean cadence
decrease; drift cause and the 13 fused specializations' settling behavior remain
unresolved. Raw wrapper failures and protocol exclusions remain preserved. Task
training processes and the telemetry logger are stopped, and all eight GPUs are
idle; no further GPU work is scheduled.

## Capacity destination-group all-reduce: candidate timing and profile pending

The 15 November goal authorizes a narrow arithmetic reorder in this task's
capacity path and measurements on the existing owner allocation. The candidate
is integrated and independently reviewed: local int64 expert counts are grouped
before functional EP SUM, preserving the local/global gate and stable selection.
Focused CPU/Gloo and CUDA compile/SAC checks pass; the payload regression fails
on the original implementation and passes on the candidate. The
[single report](../runs/capacity-group-allreduce/report.md) owns exact checks,
the completed null eager component result and the remaining native pair/trace
gates. Reduced tensor payload is established; a training-speed benefit is not.
The synchronized development tree retains the SR008 implementation. It groups
the counters whenever `drop_by_group_capacity` is enabled. The selected
[128-counter production recipe](baseline.md#current-final-task-baseline) is
pinned to the earlier selector revision.

## Global data correction: current comparisons use the corrected inputs

The [global input requirement](baseline.md#global-requirement-real-data-and-production-tokenizer)
applies to every campaign. Historical test-tokenizer runs remain evidence for
those inputs only. The completed production-data A/B/C/D rerun below supplies
fresh matched controls, component measurements and diagnostics for the four
requested cumulative variants; it does not isolate each of the four bundled
fixes. Accepted implementation and MTP/FP8 A/A decisions are preserved.

## Production-data cumulative A/B/C/D rerun: complete

The 15 November goal authorizes the existing eight-H100 allocation for a fresh
matched comparison on real diverse data encoded with the production 100B
tokenizer. The [baseline](baseline.md#selected-production-tokenizer-and-real-data-inputs)
owns current input defaults; the [single campaign report](../runs/production-data-abcd/report.md)
owns the finite checklist, exact variants, commands, evidence and remaining gates.
Historical fixture throughput is excluded as a control.

The named production tokenizer and native wrapper are verified, and the completed
`data-002` mixture contains 102400 distinct full rows, with 25600 from each of
four GigaSets domains. The [preparation receipt](../runs/production-data-abcd/data-002.json)
owns source files, hashes, tokenizer identities and packing details. The actual
native CPU/Gloo loader check passes at prefetch depths 0 and 32 on all eight
ranks; it verifies loaded rows, exact input/label/position order, bounds,
uniqueness and owner skew. The base runner's actual
`_verify_ready_inputs(TASK_ROOT)` check passes as recorded in
[input-readiness.json](../runs/production-data-abcd/input-readiness.json). See
the [loader receipt](../runs/production-data-abcd/data-002-loader-summary.json)
for per-rank totals and distributions. `data-001` remains a failed partial and
is excluded. These are CPU/input checks and do not establish training behavior.

Independent source/runtime-cost review and both native component meta tests
pass. The one-update native acquisition and all six component timing processes
passed; [component evidence](../runs/production-data-abcd/component-results.json)
records their scope and limitations. All four serial A/B/C/D timing runs
completed, with +4.708781% B/A, +2.211793% C/B, +1.938239% D/C and +9.099118%
D/A physical-TPS ratios in this single fixed-order pair. These are provisional
observations, not repeatable or causal speedup claims. The [campaign report](../runs/production-data-abcd/report.md)
and [performance summary](../runs/production-data-abcd/performance-summary.json)
own the timer, full cadence series, memory, activation, component and uncertainty
details. All eight timing/diagnostic rows are terminal with no failures; the
source/config guard passed, child groups are gone and all eight GPUs are idle
with the lease closed. The final campaign JSON is hashed in the report, and MCP
Perfetto reconciliation is complete; its import limits and scoped findings are
recorded there. The single fixed-order throughput differences remain
provisional, with no baseline/default promotion. The completed implementation
and accepted MTP/FP8 A/A decision are preserved. The owner subsequently
authorized local/server synchronization, downloading the four original rank-0
traces into ignored `artifacts/`, and committing/pushing the consolidated task.
No additional training or installation follows from that publication request.

## Current final task baseline: owner-selected recipe

The 14 November selection of the 100B-dev crop, combined-four (including V2
prefetch), and FlashQLA `external_gate` is the foundation for S. The 18 November
selector selection and the 19 November fusion selection extend that recipe.
[Exact settings](baseline.md#current-final-task-baseline)

The combined-four result remains a provisional +4.49% fixed-window difference
from one sequential pair with settling cadence. FlashQLA's fixed-order native
comparison remains confounded by about 10.4% within-window cadence drift. These
measurements do not establish a general or causal throughput gain; the baseline
selection is the owner's decision. Long-run convergence, checkpoint/resume, EMA
updates, CP warmup, and production PP4/EP32 behavior remain unverified.

## FFN overlap and row-owner embeddings: bounded campaign complete; A/B result provisional

The [campaign report](../runs/ffn-row-owner/report.md) owns the integrated
source identity, checks and evidence limits. The final standard-autograd owner
source SHA256 is
`f3a01c6cf99b981db7b4ef66866fe46259af84e9d2c21e2044e9ae18ff1696fd`.
Backward returns an owned 262,668,288-byte FP32 `RaggedShardDTensor` local
gradient to ordinary autograd; up to 704 MiB of scratch borrows the resident
2004 MiB private `lm_head` bucket. Manual `.grad` mutation and engine/hook
restrictions are removed, while native BF16 coalescing, FP32 owner SUM and
normalization are preserved. Eight-rank NCCL checks cover routing/duplicates,
accumulation/reset, hooks and `autograd.grad`, clipping, dense AdamW, EMA and
flat-DCP next-update behavior. Full-shape lookup outputs and first/later/reset gradients match
exactly. In the equal-cache full-shape check,
the warmed accumulated owner peak was 5,781,722,112 allocated / 7,648,313,344
reserved bytes, versus 8,950,373,888 / 9,003,073,536 for baseline. The six-case
[owner component table](../runs/ffn-row-owner/embedding-component-results.md)
owns timing scopes and allocator-history distinctions; its fixture F+B median
was 2.554 ms versus 13.040 ms. This is component evidence, not native training
throughput.

The [FFN component table](../runs/ffn-row-owner/ffn-component-results.md)
reports one isolated control/split pair per component: layer0 latency was
23.58189→21.56324 ms (8.56% lower), MTP 31.41067→29.49934 ms (6.08% lower),
with unchanged unique data/gradient pool bytes. The
[MCP analysis](../runs/ffn-row-owner/ffn-component-perfetto.md) links device
flows on all eight ranks and observes FFN reduce-scatter overlapping
parent-backward compute in all 48 captured cases. It does not quantify a
whole-training gain or prove which event handles each wait consumes. The owner
accepted the measured MTP/FP8 A/A variability; it remains recorded but no
further MTP/layer0 numerical investigation is a campaign gate.

The [native memory receipt](../runs/ffn-row-owner/native-memory-results.md)
closes the initialized-pool, common-baseline memory-cap and all-three
32-microbatch native integration gates. Candidate B's unique per-rank GFSDP
pool is 1,746,392,064 B smaller; all eight B ranks remain within the baseline
worst-rank allocated/reserved cap, and the candidate's global combined peak is
lower. The receipt retains the first-update reserved premium and the warm rank-2
reserved increase of 60 MiB; neither changes the common-cap result. Its phase
windows do not prove a constructor-inclusive full-lifecycle or long-run peak.

The bounded [native trace](../runs/ffn-row-owner/native-trace-results.md)
confirms all-three activation and the combined mechanisms: in 12 FFN
rank/microbatch cases, the 513 MiB reduce-scatter precedes the parent reduction
and overlaps 96.20–98.32% with remaining parent compute. It also confirms that
embedding-table collectives are removed while the untied `lm_head` collectives
remain. The trace covers three of 32 microbatches on two ranks and cannot prove
exact event-wait causality or a full-update critical path. The completed [native
performance report](../runs/ffn-row-owner/native-performance-results.md)
records the all-eight-rank 10+10 A/B pair: global physical throughput was
164,737.68→166,825.08 tokens/s (+1.27%), supervised throughput was
164,697.46→166,784.35 tokens/s, and elapsed time was 127.303→125.710 s. The
single serial pair has about 3% within-window cadence drift and no A/A noise
estimate, so this provisional fixed-window difference does not establish a
general or causal whole-training speedup. Production convergence and PP4/EP32
behavior remain unverified.

## Timing drift in fixed-window runs

Across the three eight-rank runs, rank-zero median host cadence from updates 41–60 to
121–140 decreases by about 9.9–10.4%, with the same direction on every rank and a
nonmonotonic early hump. The cause of the drift is unresolved. The [timing-drift record](../runs/flash-qla/timing-drift.md)
owns the evidence and limits. The [comparison JSON](../runs/flash-qla/historical-drift-comparison.json)
gives the exact all-rank values and hashes; these cadence intervals do not establish
device-only time or stationary/general throughput.

## FlashQLA gate variants: bounded experiment complete

The task-copy block comparison and three native eight-rank runs completed; the two
new update-21 rank-0/1 trace captures and MCP analyses also passed. Each capture
covers three of 32 microbatches. Whole-GDN 3F+B work fell from 59.2–59.7 ms to
30.9–31.4 ms (1.90–1.92×), removing 2.25–2.28% of the matched three-microbatch
device span. This is not full-update critical-path or TPS evidence. Fixed-order
native windows measured 156,452.36 / 160,090.72 / 159,735.21 global physical
tokens/s for FLA / external / fused; about 10.4% within-window cadence drift
precludes a general or causal TPS conclusion. Both variants remain task-copy
implementations. The owner subsequently selected `external_gate` for the current
final task recipe, using an explicit backend flag; no generic backend default or
production source default was promoted. CP warmup, PP4/EP32, convergence,
checkpoint/resume and EMA updates remain unverified. The [integrated
report](../runs/flash-qla/report.md), [block results](../runs/flash-qla/block-results.md),
[native results](../runs/flash-qla/native-results.md), [trace analysis](../runs/flash-qla/trace-results.md),
and [capture record](../runs/flash-qla/trace-capture-results.md) own their detailed
protocols, receipts and limits.

## Combined four improvements: complete with provisional performance evidence

The authorized baseline `combined-four-baseline-003`, combined candidate, and
21-update diagnostic passed on eight H100 ranks. Both timing runs completed
140 updates with 40 warmup and a 100-update measurement window. The candidate
measured **156,094.27 global physical input tokens/s** versus **149,386.56**,
a provisional **+4.49% fixed-window difference**; elapsed fell 4.30%, allocated
peak by 1.80 GiB, and reserved peak by 2.18 GiB. Cadence continued settling
through the measurement window; one sequential pair has no A/A noise estimate
or repeat, so steady-state/general speedup is not established.

The diagnostic traces verify all four mechanisms: seven BF16 casts and zero
singleton copies per traced rank, 39 routing kernels/rank, 12 stable-CE forward
and backward kernels/rank, and unchanged FP8 quantization. Full combined trace
imports have 9,445/9,445 events per rank with no errors or ambiguous links.
Exact device waits and a full-update critical path remain unavailable because
the profiler omits event/stream handles; the capture covers only three of 32
microbatches. The historical control differs in source, job, and epoch, so trace
comparisons establish mechanism only.

The [report](../runs/combined-four/report.md), [compact results](../runs/combined-four/results.json),
[protocol](../runs/combined-four/protocol.json), [full cadence series](../runs/combined-four/update-series.json),
and [trace analysis](../runs/combined-four/trace-analysis.md) own values, identities,
receipts, mechanism findings, and limits. The first detached launch and later
`baseline-002` MDS lock failure are preserved and excluded. After all groups
exited, candidate core patches were reversed and canonical hashes matched the
baseline; no source default was promoted at experiment close. The owner
subsequently selected this combined foundation for the final task baseline; its
exact recipe is in [baseline](baseline.md#current-final-task-baseline). All task
processes stopped and scratch was removed. Performance evidence is provisional
and does not establish production validation, convergence, continuation, or
general speedup.

## BF16 expert-weight reuse: integrated, verified, default off

The [completed assignment](../planning/README.md#active-assignment-bf16-expert-weight-reuse)
adds opt-in reuse of existing persistent BF16 expert inputs per FP32 generation
and bypasses the singleton input-to-output copy. Explicit optimizer, load, EMA
and backing-replacement boundaries preserve freshness, identity and stream
ordering. FP32 masters/gradients and FP8 quantization are preserved. Canonical
configuration remains unchanged; the four fusion candidates stay disabled.

Actual-GFSDP deterministic accumulation, recomputation, shared-pool ownership,
load/DCP, EMA, fallback and nondefault-writer checks pass, including exact local
loss/gradient/master comparisons. Scoped lint, 76 unique affected runner/config
checks and the consolidated independent review pass with no material finding
open. Retained failures and fixes are linked from the
[report](../runs/bf16-expert-cache/report.md).

Exactly two native candidate launches passed on all eight H100 ranks. Updates
21–60 measured **143,736.69 global physical input tokens/s**, versus saved V2's
**140,933.05**, a provisional **+1.99% historical difference**. Peak allocated /
reserved memory was **70.813498 / 71.939453 GiB**, with zero retry/OOM deltas.
There is no contemporary control or noise estimate; causal speedup remains
inconclusive and no material memory saving is established. Rank-0/1 traces show
21→7 expert casts and 42→0 local copies over three microbatches, with unchanged
FP8 quantization frequency/grids. Diagnostic counters cover all 32 microbatches
and 21 update boundaries; exact device wait attribution is unavailable.

The [compact results](../runs/bf16-expert-cache/results.json) and report own
metrics, source/config identities, commands, artifacts and comparison limits.
The finite experiment is closed; no further run or default promotion follows.
Production PP4/EP32, convergence and exact native continuation are unvalidated.
Owned groups exited and all eight GPUs were idle at experiment close. The owner
subsequently authorized publication, local checkout synchronization and trace download.

## V2 accepted: depth 32 for the long-epoch crop workload

The completed [four-fusion assignment](../planning/README.md#active-assignment-four-independent-fusion-experiments)
started from this accepted V2. All four explicit-kernel variants passed their
scoped native/regression checks and were active in matched rank-0/1 traces.
Input preparation, routing metadata and cap-zero CE improve their local regions,
but their small positive TPS changes vary between initial and repeated pairs
(input +0.0046%/+0.1630%, routing +0.2851%/+0.0897%, CE +0.3034%/+0.0560%).
They do not establish whole-training gains against the approximately 0.247%
descriptive range across the four initial controls; no material memory saving
was measured for them. Routing and CE repeats reused the completed input-repeat
V2 control as detailed in the [report](../runs/four-fusion/report.md).

The activated 1,024-token-tile linear-CE port is rejected: throughput fell 3.2898%
and the complete head/loss trace region slowed despite a 2.6384 GiB lower peak
allocated memory. The cap-zero CE precision-aware synthetic local replay passes
within its stated scope; it does not establish convergence or continuation. No
four-fusion candidate is accepted or enabled. The owner prohibits further V2
runs; all promised finite comparison gates are complete, so no additional run
is outstanding. Historical stop statements below describe earlier assignments
and neither reopen them nor the completed four-fusion work.

On 13 November 2026 the owner accepted V2 CPU prefetch at depth 32 based on the
long-epoch TPS pair and the subsequent
[MCP Perfetto comparison](../runs/v2-prefetch-baseline/report.md#mcp-perfetto-comparison-and-acceptance).
Rank-0 input-to-first-F/B falls from 675.782 to 8.446 ms. The complete three-F/B
device span is 1294.519 / 1293.713 ms; no material execution regression is
observed in that sampled prefix. The accepted variant uses the explicit
`num_prefetch_batches: 32` override; the generic loader and canonical control
default remain zero. No source/configuration change accompanies this decision.

The separately authorized follow-up removes frequent fixture epoch transitions
with a native logical-epoch override. The fresh matched pair over updates 21–60
passed: **138,009.86 / 140,933.05 global physical input tokens/s** at depths 0/32,
a provisional **2.12% increase** for V2, with no epoch rollover. The
[completed checklist](../planning/README.md#completed-assignment-v2-long-logical-epoch-comparison)
and [report](../runs/v2-prefetch-baseline/report.md#long-logical-epoch-comparison--complete)
own the protocol, exact results, full update series, memory and retained failures.
All four bounded update-21 traces passed coverage and lossless compression.
Input-to-first-F/B spans were about 676–678 ms for A and 8–9 ms for B; the V2
queue stayed ready. Prior-producer cost/overlap is outside capture. All owned
training groups exited and all eight GPUs are idle. Original short-epoch results
below are preserved separately. This single new pair has no repeat/noise estimate
and establishes no general speedup, convergence or production-scale behavior.
Acceptance is scoped to the current long-epoch workload.

## V2 CPU prefetch: integrated and measured, default off

The separately authorized V2 assignment ported only CPU prefetch and required
loader lifecycle/config/tests above the latest baseline, pulled at `3b6b8928`.
The default is zero; candidate runs use 32. Canonical model/experiment/distributed
source, `train_step`, accumulation, normalization and transfers are unchanged.
The [experiment report](../runs/v2-prefetch-baseline/report.md) and
[compact results](../runs/v2-prefetch-baseline/results.json) own identities,
commands, exact values, retained failures and verification.

The affected CPU/config suite passed 245 tests with no deselections; scoped Ruff
and whitespace checks passed. Real eight-owner Gloo/DCP passed both worker
persistence modes, content hashes through rollover, skewed consumer epoch entry
and decoded scalar checkpoint restores. Independent integration/runtime-cost
review found no unresolved material defect. Both native variants completed
20 full updates on all eight ranks with finite logging.

The matched unprofiled pair measured **147,402.84 global input tokens/s at depth
0** and **144,333.42 at depth 32** (18,425.35 / 18,041.68 per GPU), a provisional
**2.08% decrease**. Each ran ten warmup and ten measured updates. GPU allocator
peaks were effectively unchanged; the report distinguishes GPU peaks and host
boundary memory. A single pair has no run-to-run noise estimate, and the retained
candidate update-16 outlier has no established cause. No general performance
gain was accepted at that assignment's completion. The subsequent long-epoch
acceptance above supersedes that decision for the current workload.

Four canonical gzip traces passed coverage and lossless verification: ranks 0/1
for each variant, input prefix/token reduction and exactly three complete
microbatches per trace, with all ranks still training 32 plus optimizer. Actual
Perfetto MCP recovered all 8,795 kernels per trace after display-lane import
repair. Queue use is verified; input-to-first-F/B was about 173 ms shorter in
these captures, but consumer epoch entry still cost 886–953 ms. Those observations
do not explain the separate TPS result. Owned training groups exited, all eight
GPUs were idle at release, and task-local artifacts are retained. No commit or
push was performed for this assignment.

## Update-11 trace analysis: complete; candidate gains unmeasured

The separately requested read-only analysis used actual Perfetto MCP on the fresh
monitoring-off trace, with rank-1 CPU/CE cross-checks. Streaming prefetch has the
clearest exposed opportunity: a 972.483 ms input-phase GPU gap in a 14,337.954 ms
rank-0 train step. Head GEMMs plus CE account for 2,389.069 ms of kernel cost;
CE alone is 195.220 ms. The 448 capacity checks include 6,611.223 ms of host
synchronization, but only 92.149 ms coincides with all-device idle; the host sum
is not a recoverable step-time estimate.

The [analysis section](../runs/overnight-streaming-perf/report.md#update-11-candidate-analysis-with-perfetto-mcp)
owns the full candidate dispositions, timing scopes, source-derived no-drop
bound, import repair, limitations and next comparisons. Its
[SQL/results](../runs/overnight-streaming-perf/monitorfree-update11-perfetto.json)
retain failed and superseded queries. The original import dropped 417 overlapping
CuTe events; display-lane normalization recovered all 94,436 kernels with complete
launch-flow coverage and no importer errors. Canonical gzip bytes are unchanged.
No source/config/runner change, GPU experiment or candidate acceptance followed.

## Monitoring-free baseline: TPS and fresh traces verified

The new unprofiled baseline passed at execution revision
`16b13c2548317945712ded2e66f136b169130d95`: **145,163.84 global input tokens/s**,
**18,145.48 per GPU**, 10 warmup updates followed by updates 11–20.
The complete window was 144.467936499 s including normal loading/transfers.
Rank-0 host update cadence: median 14.562491 s, range 13.542409–15.803554 s.
Maximum sampled allocator peaks across ranks were 69.667068 GiB allocated/active
and 70.181641 GiB reserved; retry/OOM deltas were zero. The rounded all-rank
native composite maximum was 72.73 GiB. The [report](../runs/overnight-streaming-perf/report.md)
and [results](../runs/overnight-streaming-perf/results.json) define these scopes,
retain exact values and own source/config/data/runtime identities.

Warmup stayed at 10; no compilation-scale stall appeared in the measured window.
This is one provisional run with observed timing variation, not a repeatability
or production-scale claim. Historical monitoring-inclusive TPS is explicitly
historical and not directly comparable; no candidate optimization gain is claimed.

The separate trace run passed: exactly two canonical gzip traces cover optimizer
update 11 on global ranks 0/1 after warmup 10. All eight ranks trained; each trace
contains 32 microbatches and the optimizer. Parse/coverage and lossless round trips
passed; shapes/stacks/profiler memory/snapshots were off. Trace paths and hashes
are in the report/results. Owned training processes exited and all GPUs are free.
Owned obsolete caches, redundant plain traces and temporary files are removed;
MDS data and compact evidence are preserved. Results/documentation publication
uses `origin/main` and the existing repository identity. This assignment stops
after publication; no candidate or numerical comparison follows.

GIFT/model/data settings are unchanged from `eadcc6ac`. Necessary runner repairs
serialize existing wall timestamps with observation off and keep scratch/per-rank
Triton caches inside this task directory. Their scoped checks and costs are in
the results. First attempt `001` failed on a stale Triton handle before a complete
update; its compact receipt is retained, and its process group stopped. Candidate
experiments and full-model numerical validation remain cancelled. The
[execution plan](../planning/overnight-streaming-perf.md) records completed acceptance.

## Baseline: selected; campaign cancelled

The [canonical baseline](baseline.md) is a GIFT 3.5 100B-dev crop: eight trunk
layers, D4096, one dense FFN and seven MoE layers, 128 routed experts, expert
width 1536, top-8, one shared expert and MTP1. It preserves production layer
widths rather than substituting the small L8 architecture.

EP8, dense FSDP8, expert FSDP1, PP/TP/CP1; GFSDP2, dense+sparse FP8, model/loss
compile, fused AdamW, CuTe norms and EMA. Microbatch 2, sequence 4096, global
batch 512, accumulation 32. Per-layer SAC and token capacity 50,000 are fixed.
The [YAML](../gift/configs/experiments/gift_100b_crop8_ep8_fp8_compile_cute.yaml)
and [model preset](../gift/configs/presets/models/gift_moe/gift_3_5/100b_crop8.yaml)
are the machine-readable defaults. The baseline document owns the full settings.

Task GIFT is based on upstream `6fd9a3230ab6361f11bbbf0b6ec5a5f125b78fc2` with
selected development mechanisms from `cute_norm_prodrun@ec47fb8e40a9a1e55d3c18dfa704ca104ae773af`
and its 100B ancestor. Implemented mechanisms include per-layer SAC, adaptive
expert-bias balancing, typed production initialization, monitoring loss chunks
and a functional grouped-FP8 output compatible with SAC result caching.
[Provenance](../references/provenance.json) pins the source and dependency history.
The runtime uses native `torchtitan.train`, installed DeepGEMM `2.3.0+f093a4c`
and DeepEP `1.2.1+1b8f467`.

## Historical HF smoke verification

| Item | Outcome | Evidence / limit |
| --- | --- | --- |
| Eight-H100 smoke | PASS: 10 optimizer updates, 320 microbatches per rank, exit 0 | [Report](../runs/100b-crop8-ep8-smoke/report.md), [receipt](../runs/100b-crop8-ep8-smoke/result.json) |
| Monitoring | Steps 5 and 10 completed | Smoke interval 5; then-default interval 50. Ordinary/monitoring loss chunks 2/32 supported by resolved config and active source/hook path |
| Observed metrics | Final rank-0 loss 14.57517, gradient norm 7.6661; finite logged metrics | These do not establish numerical equivalence or convergence |
| Memory | Maximum native console metric 72.61 GiB | Reserved peak plus report-time non-PyTorch usage; not a simultaneous device-wide peak |
| Lifecycle | All ranks tore down, GPUs released; no checkpoint, profiler trace or memory snapshot | No checkpoint continuation was tested in this smoke |
| Source checks | Focused initialization, routing, SAC, loss, FP8 regressions and configuration goldens passed | Detailed counts and CPU test-double scope are in the report; 112 golden checks passed |
| Known local test limitation | Combined local CPU harness: 12 compiled-router failures after fake-Triton preload, plus 55 passes, one skip and one deselection | Isolated compiled regressions passed. The combined suite is not declared passing |

Smoke data are bundled `c4_test` and the test tokenizer. Production data,
long-run stability, convergence, stable throughput, numerical candidate parity,
and production-scale PP/inter-node behavior are not established by this result.
The eight-GPU crop does not reproduce EP32 expert count per rank, expert-weight
collectives or production PP overlap. Token-drop pressure remains a future study.

## Other candidate inventory: preserved, not integrated

[Candidates](candidates.md) maps mechanisms to 109 byte-preserved upstream files,
licenses, tests and hashes under `references/`.

- Kindulov: softcap/CE fusion, chunk lifetime reduction and FFN recomputation.
  The crop has no positive softcap; routed recomputation in the reference rejects FP8.
- Hidden z-loss: SwiGLU/FP8/gate-up integration is preserved. The crop's existing
  z-loss is on final norm input, so this is not a direct replacement. External
  CudaCherry kernel sources at the recorded pin are still missing.
- Vision-related work: CompactCE, streaming prefetch, GFSDP, frozen storage and
  metrics. This collection is not confirmed as the promised vision team list.
  CompactCE rejects MTP; the baseline has MTP1.
- Earlier source-review questions are retained in [candidates](candidates.md)
  and [SOTA ideas](sota-ideas.md). Former 700B-specific settings are not crop
  defaults.

The four explicit-kernel experiments above remain task-local; this preserved
donor inventory was not benchmarked on the current crop. The owner-selected
combined-four and FlashQLA recipe is documented in the current final task
baseline above; V2 CPU prefetch remains accepted as described above.
Production 100B traces and the exact remaining production MR list have not been
established by the preserved research. The small L8 config remains a donor
example, not a confirmed future test setup or our baseline.

## CUDA Graphs: low priority, not globally incompatible

The historical 3B campaign demonstrated persistent loss-only forward/backward
replay. Broader capture encountered tensor/workspace lifetime issues and repeated
dense backward recording. A valid repeated speed comparison was not completed.
A later graph-off trace showed about 99% of main-stream kernels already queued
before prior work finished, reducing evidence for a large CPU-submission benefit.
See [the conclusions and limits](cuda-graphs.md).

Decision: keep CUDA Graphs off in the default. This is not proof that all graph
boundaries fail, nor that the 3B trace predicts the current crop's performance.
Revisit only with a concrete crop bottleneck and a finite comparison plan.

## Completed cleanup and preserved campaign results

The original [overnight campaign](../planning/overnight-streaming-perf.md) is
cancelled and incomplete. The completed cleanup removed full-model numerical
machinery, disabled detailed performance monitoring and recorded the revised
short-run/local-block protocol. Its finite checklist is retained in the execution
plan; the new baseline-only authorization above does not reopen candidate work. Numerical machinery is removed. Canonical/generated
configuration and short-run behavior passed 101 scoped CPU/config checks and an
independent integration/runtime-cost review. The [cleanup receipt](../runs/overnight-streaming-perf/cancellation-cleanup.json)
confirms no owned numerical processes or reference tensor blobs remain; 40,650
files totaling 381,915,112,855 logical bytes were deleted, without attributing NFS
free-space changes. Compact history, MDS, candidate source and Perfetto evidence
remain. No training, benchmark, new trace or local numerical block run occurred
during that completed cleanup assignment.

Preserved completed evidence:

- The 896-row, full-length MDS preparation and real eight-rank loader checks,
  plus the ten-update/320-microbatch native smoke, remain valid within their
  recorded scope. The [MDS receipt](../runs/overnight-streaming-perf/g1-smoke-002-receipt.json)
  also records zero frozen parameters on every rank. Canonical data selection
  and packing/partitioning remain unchanged.
- Historical monitoring-inclusive G2 controls measured median input throughput
  **124454.81175967702 global tokens/s (15556.851469959627 per GPU)** over
  updates 111–160 including monitor 150. These are historical results, **not
  directly comparable to future monitoring-off measurements**. Their W110/cache
  acceptance method is archived, not a future prerequisite.
- G3 update 111 trace acceptance was recovered post hoc by category-aware parsing;
  the original parser failure stays recorded. Both ranks were analyzed using
  actual Perfetto MCP. [Canonical results](../runs/overnight-streaming-perf/g3-baseline-perfetto.json),
  SQL, compact source evidence and compressed trace pair are retained. The input
  wait was an unresolved finding at that historical stage, not a measured gain.
- C1 B2 passed; compact history is retained. C2 A reached terminal PASS export
  at 08:34:48 UTC; child 418189 and its process group stopped, and its controller
  was absent at 08:45 UTC. C2 B and C50 were cancelled without launch. The
  full-model validation programme is abandoned, not completed; there is no
  replacement whole-model parameter/gradient/Adam acceptance gate.
- V2 source variant `21c1891dda531a1222e6c350802f2765758794f9` on
  `vkropotin-v2-prefetch` is preserved with default 0 and its
  [source/CPU review](../runs/overnight-streaming-perf/v2-prefetch-review.md).
  Actual eight-owner Gloo/DCP and native/performance gates were not executed
  or completed. No candidate has been accepted.

Future separately authorized comparisons start with **10 warmup + 10 measured
optimizer updates**, accumulation 32, with LR horizon 65,000/LR warmup 2,000
unchanged. Disable detailed monitoring callbacks/hooks rather than increasing an
interval; ordinary loss uses two chunks. Retain lightweight native loss/time/TPS/
errors, EMA and training losses. Validation, checkpoint saves, snapshots,
numerical capture and CUDA Graphs remain off. Repeat promising or ambiguous
short-pair results when needed; one pair is provisional. Extend warmup only for
actual compilation/runtime settling that affects the window, not a cache entry
alone. No mandatory 110/50, exhaustive-cache gate or large control matrix remains.

After performance identifies a candidate worth retaining, compare the smallest
meaningful affected block on matched boundary inputs and needed weights/state;
compare affected backward with matched incoming gradients if it changes. Assess
precision, casts, rounding, reduction order and fusion, with a higher-precision
local reference when needed. Do not assume baseline exactness, universal tolerance
or bitwise equality. See [validation](../planning/validation.md). No such checks
are run during cleanup.

The source execution pin before cleanup is `e9048f65`. The
[report](../runs/overnight-streaming-perf/report.md) and
[results](../runs/overnight-streaming-perf/results.json) retain the original
measurements, failures and cancellation status. Original G4–G7 incompleteness is
not an instruction to continue the campaign. Source/config/protocol and cleanup
acceptance are verified; cleanup commit `aa45e3cd8d0c8856d383e72a9ad5f5d60c0530ae`
was pushed to `origin/main`. Stop after recording completion; no experiments
are authorized by this completed assignment.

## Working layout and historical consolidation

Current documents, editable planning, active flattened `gift/`, inert candidate
references and compact run evidence remain at their existing paths. Historical
consolidation records are recoverable from `9d56011df976017972d8f2993b910bde03487e2a`
using [recovery instructions](../references/README.md). That earlier consolidation
had no source/runtime effect and passed its documented link/JSON/hash checks;
those historical checks do not substitute for this cleanup's affected checks.
