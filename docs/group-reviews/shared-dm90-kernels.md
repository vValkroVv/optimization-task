# Shared Hopper kernels group review

The [final 10+10 comparison](#final-1010-acceptance) measured a TPS gain
of 0.883% on EP8/eFSDP1 and 1.263% on EP4/eFSDP2 for the six accepted
fixes against original A. Development and acceptance testing used the
`shk-fix` branch, based on original A at
`e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f`.

## Changes under review

| Change and switch | Intended effect |
| --- | --- |
| CuTe norm + residual fusion · `GIFT_CUTE_NORM_FUSE_RESIDUAL` | Fuse the post-norm residual add into the CuTe forward kernel to remove a separate activation read and write. Preserve BF16 rounding and reuse the original norm backward. [Implementation](../../.worktrees/shk-fix/gift/torchtitan/models/common/norms/cute_gated_norm/op.py) |
| Dense FP8 weight tail · `GIFT_FP8_QUANTIZE_WEIGHT_TAIL` | Let CudaCherry zero the partial weight tile during quantization in forward and dgrad. This removes the explicit BF16 padding for the eligible `10944 × 4096` and `4096 × 10944` weights. [Implementation](../../.worktrees/shk-fix/gift/torchtitan/ops/float8/cuda_gemm.py) |
| Statistics-only SAC · `GIFT_CUTE_NORM_SAC_STATS_ONLY` | Skip the activation write when compiled SAC replay only needs the saved values for backward. Keep the six backward inputs and statistics. [Compiler pass](../../.worktrees/shk-fix/gift/torchtitan/models/common/norms/cute_gated_norm/compiler.py) |

## 10+10 acceptance against original A

Four unprofiled runs used the same eight H100 80 GB GPUs on `sr008-gpu1`,
the eight-layer 100B crop, production-tokenizer `data-002`, seed 2048, local
batch 2, sequence length 4096 and global batch 512. All runs used native FLA,
FP8, compile, selective AC, dense FSDP8 and TP/PP/CP1. The baseline set all
three switches to `0` and the combined arm set all three to `1`. Each topology
ran baseline first and combined second. [Acceptance launcher](../../.worktrees/shk-fix/runs/shk-fix/acceptance.py)

Each arm warmed up for 10 optimizer updates and measured updates 11–20.
Global TPS uses all 20,971,520 physical input tokens in the measured window.
Peak total GPU memory is the maximum native GIFT `memory: … GiB` across all
eight ranks and the same updates. That metric includes the peak PyTorch
reserved memory since the last report and non-PyTorch usage at report time.
Deltas use the baseline on the same topology.
[Measurement protocol](../../planning/validation.md#memory-and-final-approval) · [GIFT memory calculation](../../.worktrees/shk-fix/gift/torchtitan/tools/utils.py#L278)

| Topology | Arm | Global TPS | TPS Δ | Peak total GPU memory, GiB | Memory Δ, GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| EP8 / eFSDP1 | Baseline A | 156,495.69 | — | 70.34 | — |
| EP8 / eFSDP1 | All three fixes | 157,187.26 | +0.44% | 70.42 | +0.08 |
| EP4 / eFSDP2 | Baseline A | 147,162.73 | — | 73.40 | — |
| EP4 / eFSDP2 | All three fixes | 148,134.96 | +0.66% | 73.34 | −0.06 |

All four runs completed with finite loss and gradient norms on all eight ranks.
The compiler observer reported `observed_stable` in every measured window.
Each topology has one pair of runs. The table measures the combined effect.
[Result matrix and per-arm log paths](../../.worktrees/shk-fix/runs/shk-fix/artifacts/raw/shk-10x10-gpu1-002/matrix-results.json)

The raw campaign is `shk-10x10-gpu1-002` under
`/workspace-SR008.nfs2/vkropotin/shk-fix/runs/shk-fix/artifacts/raw/`.
The first attempt stopped before measurement because its Unix socket path was
too long. The completed campaign used the same worktree at the shorter path.

## Additional 10+10 candidates

Each individual candidate ran in its own 10+10 comparison with the original three
fixes enabled in both arms. The first pairs ran control first. The CuTe
skip-gradient and routed BF16 overwrite repeats ran candidate first on
both topologies.

| Candidate | EP8 TPS Δ | EP8 peak total GPU memory, GiB | EP4 TPS Δ | EP4 peak total GPU memory, GiB |
| --- | ---: | ---: | ---: | ---: |
| CuTe backward skip-gradient add, control first | +0.325% | 70.41 → 70.41 | +0.748% | 73.32 → 73.34 |
| CuTe backward skip-gradient add, candidate first | +0.172% | 70.41 → 70.43 | +0.651% | 73.32 → 73.34 |
| Routed BF16 weight-gradient overwrite, control first | +0.145% | 70.41 → 70.41 | +0.362% | 73.48 → 73.32 |
| Routed BF16 weight-gradient overwrite, candidate first | −0.001% | 70.41 → 70.40 | +0.258% | 73.34 → 73.34 |
| Dense logical-width w2 gradient | −0.435% | 70.40 → 70.41 | +0.352% | 73.34 → 73.34 |
| Residual F2 and next F1 tile reuse, candidate first | +0.291% | 70.41 → 70.28 | +0.382% | 73.32 → 73.56 |

The skip-gradient candidate rounds the CuTe result to BF16 before adding
the skip gradient inside the same kernel. The routed candidate overwrites
nonempty expert gradients through TMA and zeroes only empty expert slices.
Both preserve the existing forward-to-backward save set.
The skip-gradient candidate gained TPS in both measured windows on both
topologies. EP4 also gained in both five-update halves of each pair.
Routed overwrite gained TPS in both EP4 pairs. The EP8 repeat was flat.
[Follow-up benchmarks and logs](../../.worktrees/shk-fix/runs/shk-fix/followup/results.md)

Two additional CuTe experiments reached the component stage. F1 vector loads
reduced CUDA Graph replay time for a compiled norm pair by 0.67%, about 4.6 µs,
with bitwise-identical output and gradients. Both synchronized compiled
comparisons had a latency interval that included zero. Reusing a residual
norm's BF16 tile in the following norm cut resident component latency by
10.81%, with bitwise parity and the same 0.815 GiB device peak in that scope.
The guarded compiled forward and backward replay with SAC cut latency by
12.02% [95% CI 11.88%, 12.16%] with bitwise-identical output and gradients
and the same 1.005 GiB device peak. The first native EP8 pair gained TPS
in both five-update halves, as did the first EP4 pair. Peak total GIFT
memory fell by 0.13 GiB on EP8 and rose by 0.24 GiB on EP4.
[Component measurements](../../.worktrees/shk-fix/runs/shk-fix/followup/results.md#component-measurements) · [Native matrix](../../.worktrees/shk-fix/runs/shk-fix/artifacts/raw/shk-norm-pair-10x10-001/matrix-results.json)

### Combined follow-ups against the three-fix control

The candidate enabled CuTe backward skip-gradient fusion, residual F2 to next
F1 tile reuse, and routed BF16 weight-gradient overwrite together. Both arms
kept the three original fixes enabled. The four unprofiled runs on `sr008-gpu1`
used the same eight-H100 crop and measured updates 11–20 after ten warmup
updates. The control ran first on each topology.

| Topology | Global TPS, control → combined | TPS Δ | Peak total GIFT memory, GiB | Memory Δ, GiB |
| --- | ---: | ---: | ---: | ---: |
| EP8 / eFSDP1 | 157,023.50 → 157,902.79 | +0.560% | 70.41 → 70.30 | −0.11 |
| EP4 / eFSDP2 | 148,155.76 → 149,474.65 | +0.890% | 73.34 → 73.56 | +0.22 |

All four runs completed with finite loss and gradient norms, stable observed
compilation, and zero allocator retries or OOMs. Each follow-up activated on
all eight candidate ranks. The candidate gained in both five-update halves on
both topologies. Each topology has one A/B pair.
[Run evidence](../../.worktrees/shk-fix/runs/shk-fix/followup/results.md#combined-native-1010) · [Result matrix](../../.worktrees/shk-fix/runs/shk-fix/artifacts/raw/shk-stack-10x10-001/matrix-results.json)

### Further component benchmarks

| Candidate and timed scope | Latency change, paired 95% CI | Peak total GIFT memory, GiB |
| --- | ---: | ---: |
| Dense GDN QKV fused dY quantization and transpose through dgrad and wgrad, late transpose control | −1.71% [−2.56%, −0.87%] | 1.1083 → 1.1083 |
| Dense GDN QKV fused dY quantization and transpose through dgrad and wgrad, early transpose control | −1.54% [−2.39%, −0.69%] | 1.1083 → 1.1083 |
| Shared FFN W2 fused dY quantization and transpose through dgrad and wgrad | +1.75% [+1.51%, +1.99%] | 0.8446 → 0.8446 |
| Routed SwiGLU forward feature-first CTA order, resident producer replay | −16.60% [−16.61%, −16.59%] | 1.1493 → 1.1493 |
| Routed SwiGLU backward feature-first CTA order through stock dgrad | −2.10% [−3.30%, −0.91%] | 2.0751 → 2.0751 |
| Routed row2col hardware warp max through captured EP8 W2 wgrad | −0.40% [−0.91%, +0.09%] | 2.3934 → 2.3934 |
| Routed row2col hardware warp max through captured EP8 W13 wgrad | −0.09% [−0.37%, +0.20%] | 3.1434 → 3.1434 |

The QKV and W2 replays kept the existing GEMMs and matched their gradient
outputs bitwise. The QKV data came from a native projection capture and its
stock short-convolution VJP. The shared FFN W2 replay used seeded inputs at
the native shape. The shared W2 fusion stays disabled. The routed SwiGLU
producer used native captured gate/up and routing
probabilities, with bitwise parity for FP8 values and FP32 scales. The
hardware warp max also matched bitwise and slowed every isolated row2col
conversion by 0.86–2.92%. It stays disabled.
[Dense dY details](../../.worktrees/shk-fix/runs/shk-fix/followup/results.md#dense-dy-through-dgrad-and-wgrad) · [SwiGLU forward details](../../.worktrees/shk-fix/runs/shk-fix/followup/results.md#routed-swiglu-forward-cta-mapping) · [SwiGLU backward details](../../.worktrees/shk-fix/runs/shk-fix/followup/results.md#routed-swiglu-backward-cta-order) · [Row2col details](../../.worktrees/shk-fix/runs/shk-fix/followup/results.md#routed-row2col-hardware-warp-maximum)

### Dense GDN QKV dY fusion against the three-fix control

Each topology ran one unprofiled control-first 10+10 pair with the original
three fixes in both arms. The QKV fusion ran during warmup on every
candidate rank. All four runs had finite loss and gradient norms, stable
compilation, and zero allocator retries or OOMs.

| Topology | Global TPS, control → candidate | TPS Δ | Peak total GIFT memory, GiB |
| --- | ---: | ---: | ---: |
| EP8 / eFSDP1 | 156,742.73 → 156,716.92 | −0.016% | 70.40 → 70.41 |
| EP4 / eFSDP2 | 148,194.37 → 147,875.42 | −0.215% | 73.34 → 73.34 |

Keep this switch disabled. The EP8 halves had opposite signs and both EP4
halves slowed. [Run evidence](../../.worktrees/shk-fix/runs/shk-fix/followup/results.md#dense-gdn-qkv-dy-native-1010) · [Native matrix](../../.worktrees/shk-fix/runs/shk-fix/artifacts/raw/shk-dense-dy-qkv-10x10-002/matrix-results.json)

### Routed SwiGLU forward CTA order against the three-fix control

The candidate changes the order in which the routed SwiGLU forward kernel
visits token and feature tiles. Both arms kept the same three original
fixes. Each topology ran one control-first 10+10 pair.

| Topology | Global TPS, control → candidate | TPS Δ | Peak total GIFT memory, GiB |
| --- | ---: | ---: | ---: |
| EP8 / eFSDP1 | 156,734.30 → 157,415.40 | +0.435% | 70.40 → 70.40 |
| EP4 / eFSDP2 | 147,624.61 → 148,484.85 | +0.583% | 73.34 → 73.32 |

Both five-update halves gained TPS on both topologies. All ranks verified
the expected launch grid before the measured window. Compilation was
stable and allocator retries and OOMs were zero in all four windows.
[Run evidence](../../.worktrees/shk-fix/runs/shk-fix/followup/results.md#routed-swiglu-forward-cta-order-native-1010) · [Native matrix](../../.worktrees/shk-fix/runs/shk-fix/artifacts/raw/shk-swiglu-fwd-10x10-002/matrix-results.json)

### Routed SwiGLU forward after the six-fix stack

Both arms enable the three original fixes and the additional three CuTe
and routed dW switches. Each topology ran one candidate-first 10+10 pair.
Only the new SwiGLU launch order differs.

| Topology | Global TPS, six → seven | TPS Δ | Peak total GIFT memory, GiB |
| --- | ---: | ---: | ---: |
| EP8 / eFSDP1 | 157,824.15 → 157,601.69 | −0.141% | 70.28 → 70.30 |
| EP4 / eFSDP2 | 149,450.63 → 149,784.10 | +0.223% | 73.60 → 73.59 |

EP8's five-update halves differed in sign. Both EP4 halves gained about
0.22%. These pairs leave the seventh switch's benefit across topologies
unresolved. [Run evidence](../../.worktrees/shk-fix/runs/shk-fix/followup/results.md#routed-swiglu-forward-with-the-six-fix-stack) · [Native matrix](../../.worktrees/shk-fix/runs/shk-fix/artifacts/raw/shk-stack-swiglu-fwd-10x10-001/matrix-results.json)

### Routed SwiGLU backward CTA order against the three-fix control

Each topology ran one control-first 10+10 pair. Only the backward CTA
order changes. The active kernel kept the native scale layout and saved
tensors. All eight ranks verified their launch geometry during warmup.

| Topology | Global TPS, control → candidate | TPS Δ | Peak total GIFT memory, GiB |
| --- | ---: | ---: | ---: |
| EP8 / eFSDP1 | 157,165.53 → 156,662.71 | −0.320% | 70.41 → 70.41 |
| EP4 / eFSDP2 | 148,086.93 → 148,468.27 | +0.258% | 73.32 → 73.34 |

Both EP8 halves slowed and both EP4 halves gained. The backward CTA
switch remains disabled in the common accepted package.
[Run evidence](../../.worktrees/shk-fix/runs/shk-fix/followup/results.md#routed-swiglu-backward-cta-order-native-1010) · [Native matrix](../../.worktrees/shk-fix/runs/shk-fix/artifacts/raw/shk-swiglu-bwd-10x10-001/matrix-results.json)

## Final 10+10 acceptance

On 22 November 2026, `shk-final-10x10-001` ran one unprofiled 10+10
per arm and topology on the same eight H100 80 GB GPUs on `sr008-gpu1`.
Each topology ran original A, the original three fixes, then all six
accepted fixes. The six-fix arm adds CuTe backward skip-gradient fusion,
CuTe residual F2 → next F1 tile reuse and routed BF16 dW overwrite.
The recipe kept its existing GDN path and the default routed SwiGLU CTA order.

| Topology | Arm | Global TPS | TPS Δ vs A | Peak total GIFT memory, GiB | Memory Δ vs A, GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| EP8 / eFSDP1 | Original A | 156,083.35 | — | 70.34 | — |
| EP8 / eFSDP1 | Original three | 156,723.21 | +0.410% | 70.41 | +0.07 |
| EP8 / eFSDP1 | All six accepted | 157,461.08 | +0.883% | 70.30 | −0.04 |
| EP4 / eFSDP2 | Original A | 147,435.71 | — | 73.38 | — |
| EP4 / eFSDP2 | Original three | 148,061.97 | +0.425% | 73.34 | −0.04 |
| EP4 / eFSDP2 | All six accepted | 149,297.43 | +1.263% | 73.65 | +0.27 |

Against the three-fix arm in the same matrix, the six-fix arm gained
0.471% TPS with 0.11 GiB lower peak on EP8 and 0.834% TPS with 0.31 GiB
higher peak on EP4. Both five-update halves gained TPS on both topologies.
Peak memory uses native GIFT logs across all eight ranks and updates
11–20. All six runs had stable compilation, finite loss and gradient
norms, and zero allocator retries and OOMs.
[Run evidence](../../.worktrees/shk-fix/runs/shk-fix/followup/results.md#final-native-1010-acceptance) · [Frozen inputs](../../.worktrees/shk-fix/runs/shk-fix/artifacts/raw/shk-final-10x10-001/preflight.json) · [Native matrix](../../.worktrees/shk-fix/runs/shk-fix/artifacts/raw/shk-final-10x10-001/matrix-results.json)

## Overlap with the other optimization groups

| Candidate or result | Existing optimization | Scope decision |
| --- | --- | --- |
| Q/K-only short-conv replay | The FLA short-conv backward replaces the same replay and gradient-packing path. | Dropped before GPU execution. The earlier QKV dY projection results above remain historical and the candidate stays disabled. [FLA group review](fla.md#changes-under-review) |
| Routed expert weight quantizer traversal | FP8 expert-weight caching removes per-microbatch quantization for the measured EP8 recipe. Its update-21 trace shows 14 refresh launches followed by zero quantizers in the next two microbatches. | Dropped before GPU execution. The older uncached launch budget does not describe the promoted EP8 path. [Expert cache traces](../status.md#expert-gfsdp-standalone-results) |
| Routed BF16 weight-gradient overwrite | Direct expert dW writes to the FP32 reduction slot and bypasses the returned BF16 gradient branch on eligible EP8/eFSDP1 groups. | The 10+10 result applies to the three-fix control with the BF16 return branch. Direct FP32 dW and BF16 overwrite offer no additive saving on an eligible call. [Direct-slot guard](../../gift/torchtitan/distributed/gfsdp/_param_group.py#L899) · [Branch selection](../../gift/torchtitan/ops/float8/grouped_gemm.py#L281) |
| Routed SwiGLU CTA order | The historical row16 experiment changes the backward tile height. Dual-layout SwiGLU adds a second FP8 layout. Both experiments remain disabled in the production recipe. | The CTA experiment retains the existing tile sizes, FP8 layouts, routing math and autograd saves. HybridEP routing metadata, capacity selection, attention and LM-head work stay outside the experiment. [Historical SwiGLU experiments](../../runs/fusion-followup/report.md) · [HybridEP group review](hybridep.md) · [LM-head group review](lm-head-mtp.md) |
