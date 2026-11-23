# Expert GFSDP group review

Updated 22 November 2026. Against compiled Original A, the final Expert GFSDP
package raised TPS by **3.55%** and cut peak total GPU memory by **4.03 GiB**
on EP8/eFSDP1. On EP4/eFSDP2 it raised TPS by **9.81%** and cut peak memory
by **5.71 GiB**. Each topology has one final unprofiled A/B pair.
[Final 10+10 matrix][matrix]

## Source branch and tested variants

The port of the individual fixes onto compiled Original A is in the
**`expert-gfsdp-fix`** worktree at `.worktrees/expert-gfsdp-fix`.
Its branch HEAD is the Original A commit
`e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f`, with GIFT tree
`0f44b971069af55d6d573628129900483ca3e1dc`. The port is preserved as
uncommitted worktree changes, experiment patches and run receipts. A fresh
checkout of the branch alone yields the baseline. The older
`codex/efsdp-training-optimizations` branch at `f972714fc` records the fixed
EP8 recipe and is the historical donor, including `5b9ae4f15` for FP8 weights,
`c68cc9805` for direct `dW`, and `a0602a887` for tail retention.
[Original A port and runner][runner] · [Historical donor and source notes][report]

| Variant | Tested source | Digest prefix |
| --- | --- | --- |
| Original A, both topologies | Clean GIFT from compiled Original A, `egfa` | `8450bbe7e8f2` |
| EP8 B | Original A port and deferred `dW` wait, `egfd` | `81ebc3700344` |
| EP4 B | Same port and deferred wait plus the compact BF16 slot overlay, `egfc` | `7a5977c0bf9e` |

The digests fingerprint the staged GIFT file contents. The local base port and
the frozen variants have different GIFT file inventories. The launch scripts
verify the frozen digests, and the retry checks the native `dW` build against
the archived patch and binding. The matrix directory preserves the resolved
configs, GPU UUIDs and per-rank activation. The base overlay manifest records
the SR008 archive and checksum. [Final launch and staging][launch] · [Source digests][digests]
· [Base overlay manifest][overlay-manifest]

## Changes under review

Both B arms activated FP8 weight caching, direct `dW` and tail retention in
all seven routed-expert groups on all eight ranks. The BF16 weight cache was
requested as a fallback. Every routed group used FP8, so the BF16 weight
cache was inactive in both B arms. BF16 gradient staging was active on all
eight ranks of EP4 B only. [Activation and run receipts][matrix] · [Compact markers][compact-markers]

| Change | Original behavior and intent | EP8 / eFSDP1 | EP4 / eFSDP2 |
| --- | --- | --- | --- |
| **BF16 weight-cache fallback** | The old cache required the local BF16 input to alias the all-gather output, which restricted it to a one-rank expert group. Split local-cast lifetime from receive-buffer ownership. | An eligible fallback reuses the local cast as the weight view. Inactive in B. | An eligible fallback reuses the local cast and gathers into a separate receive buffer. Inactive in B. [Cache selection][cache-select] |
| **FP8 weight cache** | The expert path quantized materialized BF16 weights for forward and recompute. Prepare the local FP32 shard into E4M3 data and FP32 scales once per weight generation, preserving the BF16 rounding before blockwise quantization. | DeepGEMM reads the local cache planes directly. | Gather the two prepared planes. A rank contributes 288 MiB of E4M3 plus 72 KiB of scales per logical gather, down from 576 MiB of BF16 weight data. [Weight cache][fp8-cache] |
| **Tail weight retention** | Intermediate chunked LM-head backwards triggered a root GFSDP callback that could reshard groups whose backward had yet to run. Preserve their weight slots until their actual consumers or final cleanup. | The callback preserves tail expert and dense weights. | The same policy also avoids the corresponding physical expert gathers. [Callback][callback] |
| **Direct expert `dW`** | Grouped DeepGEMM wrote a temporary BF16 tensor. GFSDP then copied and widened it into the shared gradient slot. Give the writer the actual slot and offsets for `W13` and `W2`. | The Hopper epilogue preserves BF16 rounding and zero-add semantics, then writes the rounded result directly into the existing FP32 slot. | The stock DeepGEMM writer writes directly into its zeroed BF16 range in the compact slot. [Grouped backward][grouped-backward] · [Rounded writer][rounded-writer] |
| **Deferred `dW` slot wait** | The first direct writer waited in `pre_backward` for the previous layer's reduction to finish reading the shared slot. Capture that release event when claiming the slot and wait just before the first write. | Recompute, `dX` and operand preparation can run before the wait. | The same schedule waits before zeroing, the first write to the BF16 slot. [Deferred wait patch][deferred-patch] |
| **Compact BF16 gradient exchange** | Native reduce-scatter read a full FP32 slot. The underlying contributions already had BF16 precision. Send peer-owned ranges directly from a BF16 slot and sum on the shard owner. | One rank per expert group. Local FP32 accumulation uses the existing slot path. | Two BF16 exchanges per group reuse one receive tensor. The owner widens both contributions and reduces and accumulates in FP32, with separate rounding for each add. [Compact BF16 overlay][compact-patch] |

The main and MTP loss share one LM-head projection. The head already held its
own weights across chunks. Each detached chunk backward triggers the shared
GFSDP callback before the full decoder backward. Tail retention keeps groups
such as `mtp.layers.0` and the last expert groups resident across that gap.
Pending prefetches keep slot ownership until their GPU writes finish.
[Chunked loss][loss] · [GFSDP slot release][slot-release]

```text
Original A   DeepGEMM → temporary BF16 dW → copy and widen → shared FP32 slot
EP8 B        DeepGEMM → BF16 rounding in epilogue → shared FP32 slot
EP4 B        DeepGEMM → shared BF16 slot → BF16 peer exchange → FP32 owner gradient
```

Each EP4 pair `[0,4]`, `[1,5]`, `[2,6]` and `[3,7]` contributes to both halves
of its logical expert gradient. Each rank sends the half owned by its peer
and adds the received contribution to the half it owns. Splitting the send
in two lets the receiver reuse a 288 MiB buffer. The shared slot and receive
storage use `1,152 + 288 = 1,440 MiB` per rank, down from `2,304 MiB` for the
FP32 slot. The **864 MiB** reduction measures these temporary buffers. The
optimizer-facing gradient remains FP32. [Compact transport design][compact-design]

## Final 10+10 on compiled Original A

Four unprofiled runs used the same eight H100 80 GB GPUs on `sr008-gpu1`
in job `28bb9cd2-096c-4f7f-b154-096065b323c9`. They used the eight-layer
100B crop, real `data-002`, the production tokenizer, seed 2048, local batch 2,
sequence length 4096, global batch 512 and 32 accumulation microbatches.
Both topologies used dense FSDP8 and TP/PP/CP1. All arms kept
FP8, Inductor model and loss, selective activation checkpointing, MTP, EMA and
the LR schedule. They used original A's native FLA, original HybridEP routing
and cross-entropy, zero batch prefetch and fresh per-arm compile caches.
[Runner recipe][runner] · [Runtime][runtime] · [Run records][matrix]

Global TPS divides the 20,971,520 physical input tokens by elapsed time over
updates 11–20. Median update TPS is the median of the summed per-rank TPS.
Memory is the maximum native GIFT `memory: … GiB` across all eight ranks and
the same updates. The compiler observer reported stable observed channels
with partial coverage. The measured windows had zero allocator retries and
OOMs. Deltas compare B with A on the same topology. [Measurement protocol][validation]

| Topology | Arm | Window TPS | Median update TPS | TPS Δ | Peak total GPU memory, GiB | Memory Δ, GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| EP8 / eFSDP1 | `efg-n22-07-final-ep8-a` | 153,756.88 | 153,852 | — | 72.30 | — |
| EP8 / eFSDP1 | `efg-n22-07-final-ep8-b-retry` | 159,211.93 | 159,408 | **+3.55%** | 68.27 | **−4.03** |
| EP4 / eFSDP2 | `efg-n22-07-final-ep4-a` | 143,448.48 | 143,764 | — | 76.19 | — |
| EP4 / eFSDP2 | `efg-n22-07-final-ep4-b` | 157,518.28 | 157,220 | **+9.81%** | 70.48 | **−5.71** |

The EP8 B retry is the completed measurement. The first attempt lacked the
staged DeepGEMM patch required by the native build manifest.
[Full result data][matrix] · [Retry and source validation][retry]

## Isolated and earlier training experiments

| Experiment | Comparison | Measured result |
| --- | --- | --- |
| BF16 weight cache only | EP8 10+10, Original A recipe, same port with cache off and on | 152,967.52 → 157,185.48 TPS (**+2.76%**), 72.30 → 72.30 GiB. The multi-rank fallback passed two- and four-rank Gloo component checks. No completed isolated EP4 training result is recorded. [BF16 pair][bf16-pair] · [Multi-rank checks][bf16-component] |
| FP8 weight cache only | Historical EP8 fixed recipe, control already using BF16 weight reuse | 171,205.53 → 173,472.33 TPS (**+1.32%**), 73.20 → 69.72 GiB. [Historical series][historical] |
| Direct expert `dW` only | Same historical EP8 control | 171,205.53 → 174,185.79 TPS (**+1.74%**), peak 73.20 GiB in both arms. [Historical series][historical] |
| Tail retention only | Same historical EP8 control | 171,205.53 → 172,809.33 TPS (**+0.94%**), peak 73.20 GiB in both arms. [Historical series][historical] |
| Earlier combined port on Original A | EP8 and EP4, before deferred wait and the compact slot | EP8 **+4.14% TPS**, **−4.05 GiB**. EP4 **+6.34% TPS**, **−4.98 GiB**. [Earlier 10+10 results][earlier] |
| Deferred `dW` wait | EP4 10+10 A/B/B/A over the earlier package with FP32 reduce-scatter | **+2.40% TPS** geometric mean. Pair deltas were +2.83% and +1.98%, with peak changes +0.17 and −0.02 GiB. Same-source TPS varied by 0.30% for A and 1.12% for B. [ABBA comparison][deferred-abba] |
| Compact BF16 slot and exchange | Two H100s, component benchmark, exchange plus subsequent FP32 accumulation | **5.42 → 3.06 ms** device span, **−864 MiB** of temporary storage per rank. Host submission increased from 39 to 113 µs. The final EP4 training pair measures the combined effect. [Component results][compact-component] |

The historical isolated EP8 arms used a fixed-order series and a different
recipe from the final Original A matrix. Each standalone number describes
its measured comparison. The final A/B pairs measure the combined package.

The earlier packed BF16 owner experiment improved exchange time at the cost
of extra staging. Directly producing into the BF16 slot removed the pack.
Typed FP8 gather coalescing reduced host submission while slightly increasing
the device span. FP32 scale reuse saved about 12 µs on a component cache hit.
The tested copy-engine gather increased the joint gather-plus-GEMM span.
[Two-H100 follow-up experiments][comm-components]

## What the four 10+1 profiles showed

All runs completed update 11 after ten warmup updates. Captures cover the
first three forward/backward microbatches on ranks 0 and 4. The included
window spans CPU entry of microbatch 1 through completion of the last linked
GPU work in microbatch 3. Operation rows are rank-local GPU busy interval
unions by family, rounded across the two sampled ranks. Different families
can overlap. [Profile extraction, run IDs and capture receipts][profiles]

| Observation, per rank over three microbatches | EP8 A → B | EP4 A → B |
| --- | --- | --- |
| Forward/backward prefix span | 1,259 → 1,204 ms | 1,307 → 1,230 ms |
| Expert weight gathering | No peer gathers. 42 singleton weight copies removed. | 42 BF16 gathers → 36 E4M3 gathers plus 36 small FP32 scale gathers. GPU busy time approximately 143 → 66 ms. |
| Dense MTP gathers | 6 → 3 | 6 → 3 |
| Expert gradient packing | 13.5–13.6 → 0 ms | 27.0–27.2 → 0 ms |
| Expert gradient communication | Singleton. The local FP32 grad path remains active. | 21 FP32 reduce-scatters → 42 BF16 exchanges. Exchange busy time outside main compute intervals fell from 19–26 to 2–8 ms. |

The profiles show why EP4 gained more. Both arms removed local weight
preparation and gradient packing. EP4 also reduced peer traffic during weight
gather and gradient reduction. The unprofiled step savings were **0.47 s on
EP8** and **1.31 s on EP4**. Two sampled ranks and three microbatches leave the
exact attribution of the additional 0.84 s per full update unresolved. Exact
CUDA producer-event matches were unavailable in these captures. The separate
EP4 deferred-wait experiment showed the earlier portion of backward
overlapping with the prior reduction after the wait moved.
[Final timing][matrix] · [Deferred-wait trace comparison][deferred-trace]

On EP8 B, the largest sampled compute gap coincided with dense
`mtp.layers.0` reduce-scatter on rank 4. Both sampled ranks also showed a
recurring embedding all-gather gap. These measurements identify places to
trace rank arrival and buffer dependencies for an EP8 follow-up.
[EP8 B profile receipt][ep8-profile]

## Numerical evidence

The native FP8 cache component matched legacy E4M3 data and FP32 scale bits
on two and four H100 ranks over two weight generations. The rounded `dW`
writer matched the BF16 producer followed by widening for tested 16- and
32-expert cases, including empty routing. The compact two-H100 component
matched native FP32 reduction and persistent accumulation in the checked
producer cases. The earlier one-update EP8 A/A/B diagnostic had bitwise
loss agreement and sampled gradient differences in both A/A and A/B.
The full compiled EP4 compact numerical comparison has a diagnostic harness
with no completed result recorded. The final performance matrix establishes
short-run execution and performance on this crop. [Native components][bf16-component]
· [EP8 A/A/B][ep8-numerics] · [EP4 diagnostic][ep4-numerics]

[matrix]: ../../.worktrees/expert-gfsdp-fix/runs/efsdp-optimizations/artifacts/raw/final-matrix-01/comparison.json
[runner]: ../../.worktrees/expert-gfsdp-fix/runs/efsdp-optimizations/README.md
[report]: ../../.worktrees/expert-gfsdp-fix/runs/efsdp-optimizations/report.md#multi-rank-design
[launch]: ../../.worktrees/expert-gfsdp-fix/runs/efsdp-optimizations/artifacts/raw/final-matrix-01/launch.sh
[runtime]: ../../.worktrees/expert-gfsdp-fix/runs/efsdp-optimizations/artifacts/raw/final-matrix-01/runtime.json
[digests]: ../../.worktrees/expert-gfsdp-fix/runs/efsdp-optimizations/artifacts/raw/final-matrix-01/source-digests.json
[overlay-manifest]: ../../.worktrees/expert-gfsdp-fix/runs/efsdp-optimizations/artifacts/raw/final-matrix-01/base-overlay.sha256
[compact-markers]: ../../.worktrees/expert-gfsdp-fix/runs/efsdp-optimizations/artifacts/raw/final-matrix-01/efg-n22-07-final-ep4-b/compact-markers.tsv
[cache-select]: ../../.worktrees/expert-gfsdp-fix/gift/torchtitan/distributed/gfsdp/_param_group.py#L661
[fp8-cache]: ../../.worktrees/expert-gfsdp-fix/gift/torchtitan/distributed/gfsdp/_fp8_expert_weight_cache.py#L186
[callback]: ../../.worktrees/expert-gfsdp-fix/gift/torchtitan/distributed/gfsdp/_state.py#L721
[grouped-backward]: ../../.worktrees/expert-gfsdp-fix/gift/torchtitan/ops/float8/grouped_gemm.py#L242
[rounded-writer]: ../../.worktrees/expert-gfsdp-fix/runs/efsdp-optimizations/direct-dw/deepgemm-rounded-f32.patch#L9
[deferred-patch]: ../../.worktrees/expert-gfsdp-fix/runs/efsdp-optimizations/experiments/deferred-dw/deferred-dw.patch
[compact-patch]: ../../.worktrees/expert-gfsdp-fix/runs/efsdp-optimizations/experiments/owner-bf16-slot/owner-bf16-slot.patch
[loss]: ../../.worktrees/expert-gfsdp-fix/gift/torchtitan/components/loss.py#L709
[slot-release]: ../../.worktrees/expert-gfsdp-fix/gift/torchtitan/distributed/gfsdp/_param_group.py#L1208
[compact-design]: ../../.worktrees/expert-gfsdp-fix/runs/efsdp-optimizations/experiments/owner-bf16-slot/README.md
[validation]: ../../planning/validation.md#memory-and-final-approval
[retry]: ../../.worktrees/expert-gfsdp-fix/runs/efsdp-optimizations/artifacts/raw/final-matrix-01/retry.sh
[bf16-pair]: ../../.worktrees/expert-gfsdp-fix/runs/efsdp-optimizations/artifacts/raw/expert-isolation-01/ep8-comparison.json
[bf16-component]: ../../.worktrees/expert-gfsdp-fix/runs/efsdp-optimizations/report.md#component-verification
[historical]: ../../.worktrees/expert-gfsdp-fix/runs/efsdp-optimizations/report.md#historical-isolated-evidence
[earlier]: ../../.worktrees/expert-gfsdp-fix/runs/efsdp-optimizations/report.md#native-acceptance
[deferred-abba]: ../../.worktrees/expert-gfsdp-fix/runs/efsdp-optimizations/artifacts/raw/deferred-dw-abba-01/comparison.json
[compact-component]: ../../.worktrees/expert-gfsdp-fix/runs/efsdp-optimizations/report.md#bf16-dw-slot-and-pack-free-owner-exchange
[comm-components]: ../../.worktrees/expert-gfsdp-fix/runs/efsdp-optimizations/report.md#two-h100-communication-results
[profiles]: ../../.worktrees/expert-gfsdp-fix/runs/efsdp-optimizations/artifacts/raw/final-profiles-01/analysis.json
[deferred-trace]: ../../.worktrees/expert-gfsdp-fix/runs/efsdp-optimizations/report.md#deferred-expert-gradient-slot-wait
[ep8-profile]: ../../.worktrees/expert-gfsdp-fix/runs/efsdp-optimizations/artifacts/raw/final-profiles-01/efg-n22-08-profile-ep8-b/summary.json
[ep8-numerics]: ../../.worktrees/expert-gfsdp-fix/runs/efsdp-optimizations/artifacts/raw/ep8num-01/ep8-parity-both-controls.json
[ep4-numerics]: ../../.worktrees/expert-gfsdp-fix/runs/efsdp-optimizations/numerics/ep4-owner/README.md
