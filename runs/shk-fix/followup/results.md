# Shared Hopper kernel follow-up

The comparison baseline is original A `e9b7cdbb4` with residual fusion, the FP8 weight-tail fix, and statistics-only SAC enabled. All GPU measurements use `sr008-gpu1`. The candidates below retain the existing math and add no tensors to the forward-to-backward save set.

## Component measurements

All timing deltas below refer to the named component. The norm and FFN replays use native `data-002` activations and tangents where available. None of these measurements establishes a change in full-training TPS or peak memory.

| Candidate | Compiled comparison | Numerical result | Decision |
| --- | --- | --- | --- |
| CuTe `GnBwdDx` skip-gradient add | Backward CUDA-event median 0.5024 → 0.4547 ms. Synchronized forward + backward 0.8828 → 0.8333 ms, paired mean −5.79% with 95% t interval [−6.01%, −5.57%]. Eight AB/BA blocks, 100 warmups, 256 calls per block. | Output and all gradients matched bytewise in the native-shape SAC replay. The skip tangent is a one-column roll of the captured native tangent. Empty tensors, a 65-row tail, both add orders, and unaligned fallback also matched. Peak allocated 485,573,632 bytes and peak reserved 1,889,533,952 bytes in both arms. | Advance to 10+10 on EP8 and EP4. |
| Logical-width dense w2 gradient | Isolated wgrad CUDA-event median 0.9576 → 0.9057 ms in 100 paired calls. Compiled FFN with SAC synchronized median 9.3263 → 9.2626 ms, one 100-warmup/256-call process per arm. | The same cuBLASLt kernel produced a bitwise gradient from compacted scales. All compiled FFN outputs and gradients matched, and the trace verified width 11008 → 10944. Device-wide component peak was 1.9325 GiB in both arms. | Advance to 10+10 with the other follow-up switches off. |
| Routed SwiGLU MN-major scales | Producer graph replay median about 331.5 → 295.1 µs. The two full producer/consumer blocks per arm gave 1847.7 and 1902.4 µs for control, 1913.6 and 1836.6 µs for candidate. | Eager and compiled FP8 bytes, FP32 scales, routing gradients and consumer outputs matched. Both scale copies disappeared. Peak allocated dropped 5.82 MiB with unchanged reserved memory. Compile + SAC integration and EP4 are still unverified. | Keep disabled. The chain speedup remains unresolved. |
| CuTe B2 gate-down tile reuse | Backward CUDA-event median 0.5123 → 0.5122 ms. Synchronized forward + backward 0.8977 → 0.8971 ms, paired mean −0.206% with 95% t interval [−0.605%, +0.193%]. Eight AB/BA blocks, 100 warmups and 256 calls per block, skip-gradient fusion off in both arms. | All outputs and gradients matched bitwise at native shape and in 65-row tails. The B2 specialization used 2 KiB more shared memory per CTA with the same register count and grid. Allocated and reserved peaks matched. | Keep the original B2 kernel. [Result](../artifacts/raw/norm-gate-down-001/result.json) · [Experiment patch](gate-down-tile-reuse.patch) |
| CudaCherry weight scale reuse | Two-quantizer CUDA-event medians at the captured tail shapes were 148.376 → 153.144 µs and 173.064 → 177.952 µs. Paired mean increases were 4.722 µs [4.576, 4.868] and 4.844 µs [4.431, 5.257]. The two derived aligned cases also slowed. Eight AB/BA blocks, 100 warmups and 256 calls per block per shape. | FP8 data, FP32 scales and BF16 GEMM outputs matched bitwise at all four shapes. Both expected quantizer kernels appeared in each arm. Allocated and reserved peaks matched within each shape. | Keep scale recomputation. [Result](../artifacts/raw/dense-weight-scale-002/result.json) · [Harness](dense_weight_scale.py) |
| Routed BF16 dW overwrite | Isolated CUDA Graph replay improved 4.40% [3.96%, 4.84%] on captured EP8 and 11.53% [11.16%, 11.90%] on the derived EP4 shape across 24 pairs. Through the compiled GIFT operator, synchronized medians were 3.1512 → 3.0922 ms on EP8 and 3.3725 → 3.1021 ms on derived EP4. The paired latency changes were −1.99% [−2.30%, −1.69%] and −8.54% [−9.41%, −7.67%]. Eight AB/BA process pairs used 100 warmups and 256 measured calls per arm. | All BF16 outputs matched bitwise, including capacity tails, empty groups, special values, and dirty destinations across the two probes. Both compiled zero fills disappeared in every profiled pair and GEMM launch geometry matched. Peak allocated bytes matched. One baseline process had 50 MiB more reserved at both shapes. | Completed native 10+10 on both topologies. [CUDA Graph result](../artifacts/raw/routed-dw-bf16-001/component.json) · [Compiled result](../artifacts/raw/routed-dw-bf16-compiled-002/compiled.json) · [Build and probe](routed_dw/README.md) |
| Routed row-major scale loads | Omitting the scale-layout copy slowed the standalone converter by 2.74–3.55% across the two captured EP8 roles and their derived EP4 versions. All four converter intervals excluded zero. Through the BF16 GEMM, all four intervals included zero. Eight AB/BA blocks used 100 warmups and 256 calls per arm per scope. | Consumed FP8 bytes, FP32 scales and all BF16 weight-gradient bytes matched. The removed copy reduced converter-only peak allocated memory by 7.80 MiB. Allocated and reserved peaks matched through wgrad. | Keep the existing CudaCherry layout conversion. [Component result](../artifacts/raw/routed-row2col-strided-002/component.json) |
| Routed FP8 weight transpose generated by Inductor | CUDA-event medians for w13 rose from 192.840 to 201.592 µs on EP8 and from 350.000 to 375.808 µs on EP4. EP8 w2 fell from 127.720 to 120.368 µs. The EP4 w2 interval included zero. Eight AB/BA blocks used 100 warmups and 256 calls per arm per shape. | Seeded FP8 bytes and FP32 scales transposed bitwise to the expected contiguous layouts. Peak allocated and reserved bytes matched within each shape. The generated weight-byte kernel took longer in every single-call profile, including EP8 w2. | Keep the native weight transpose before dgrad. The component times include submission gaps. [Component result](../artifacts/raw/routed-weight-transpose-001/component.json) · [Harness](routed_weight_transpose_component.py) |
| CuTe three-input merge with indexed shared stores | The separate Triton merge and compiled CuTe backward took 0.2877 ms. The fused version took 0.5486 ms, a paired increase of 90.62% [90.43%, 90.82%]. Eight AB/BA blocks used 100 warmups and 256 calls per arm. | The merged BF16 data and all four gradients matched at native shape and in 65-row tails. The 88.0 µs merge launch disappeared. CuTe B1 rose from 80.7 to 383.8 µs with the same registers and shared memory per CTA. Allocated and reserved peaks matched in the replay. | Keep the separate merge. [Component result](../artifacts/raw/norm-dy-merge-002/component.json) |
| CuTe three-input merge with a swizzled vector store | Replacing the indexed writes with a 128-bit copy through the original swizzled view gave 0.2878 → 0.5291 ms. Paired latency increased 83.99% [83.59%, 84.39%]. Eight AB/BA blocks used 100 warmups and 256 calls per arm. | Bitwise parity held for the merged data and every gradient at both shapes. The 88.1 µs separate launch disappeared and B1 rose from 80.4 to 364.6 µs. Kernel geometry, registers, shared memory, and allocator peaks matched. | Keep the separate merge. [Component result](../artifacts/raw/norm-dy-merge-003/component.json) · [Experiment patch](norm-dy-merge-experiment.patch) |
| CuTe F1 vectorized sum-square loads | The isolated compiled kernel took 36.736 → 31.344 µs by CUDA events, a paired change of −14.69% [−14.93%, −14.45%]. Through compiled residual and plain norms plus SAC, synchronized latency changed by −0.05% [−0.46%, +0.37%] and −0.42% [−1.56%, +0.71%]. Capturing the same compiled pair in CUDA Graphs gave −0.670% [−0.716%, −0.624%]. | FP32 partials matched bitwise at 8,192 and 65 rows. The compiled output and eight gradients matched bitwise in both timed series and both graph capture orders. Both arms reached 0.72156 GiB in isolation and 2.25671 GiB through the synchronized compiled pair by the GIFT memory formula. | Preserve the experiment. At 4.62 µs per four-launch pair in graph replay, it does not justify a standalone 10+10. [Isolated result](../artifacts/raw/norm-f1-vector-001/component.json) · [First compiled series](../artifacts/raw/norm-f1-compiled-001/component.json) · [Compiled repeat](../artifacts/raw/norm-f1-compiled-002/component.json) · [Graph replay](../artifacts/raw/norm-f1-compiled-graph-002/component.json) |
| CuTe B1 shared gate-weight layout | CUDA-event medians were 71.752 → 71.904 µs, a paired change of +0.13% [−0.23%, +0.50%]. Synchronized wall time was 78.349 → 78.511 µs, a paired change of +0.13% [−0.23%, +0.48%]. Eight AB/BA blocks used 100 warmups and 256 calls per arm. | FP32 dS, c1 and gate-weight gradient partials matched bitwise at 8,192 and 65 rows. Profiled kernel medians were 66.272 → 66.896 µs. Both arms reached 0.87390 GiB device peak by the GIFT formula with identical allocated and reserved peaks in the resident component. | Keep the existing B1 layout. [Component result](../artifacts/raw/norm-b1-gate-layout-001/component.json) · [Harness](norm_b1_component.py) |
| Residual F2 followed by the next F1 | Resident synchronized wall time was 120.317 → 107.381 µs, a paired change of −10.81% [−10.92%, −10.70%]. The guarded compiled forward and backward replay with SAC took 1,393.265 → 1,224.371 µs, a paired change of −12.02% [−12.16%, −11.88%]. | Five resident outputs matched bitwise at 8,192 and 65 rows. The compiled output and eight gradients matched bitwise. Both arms reached 0.81531 GiB in the resident test and 1.00476 GiB in the compiled test by the GIFT formula. | Completed native 10+10 on EP8 and EP4. [Resident result](../artifacts/raw/norm-pair-tile-reuse-002/component.json) · [Compiled result](../artifacts/raw/norm-pair-tile-compiled-002/component.json) · [Kernel patch](norm_pair_tile_reuse.patch) |
| Dense dY row quantization and FTZ transpose | Synchronized wall times were 117.827 → 105.064 µs at width 4,096 and 176.062 → 153.802 µs at width 8,192. Paired changes were −10.87% [−11.03%, −10.70%] and −12.65% [−12.75%, −12.55%]. Each shape used 16 AB/BA pairs with 256 timed calls per arm and 100 initial warmups. | Both FP8 layouts and both FP32 scale planes matched bitwise for deterministic inputs at both native shapes, all BF16 encodings, and the mixed scale and special-value case. The trace confirms two launches became one. Device peak stayed at 0.73328 and 0.87585 GiB respectively by the GIFT formula. | Check the entire dgrad and wgrad neighborhood, including early column-layout allocation. [Width 4,096](../artifacts/raw/dense-dy-quant-4096-001/component.json) · [Width 8,192](../artifacts/raw/dense-dy-quant-8192-001/component.json) · [Probe](dense_dy_quant_transpose.py) |
| Dense dY fusion through QKV dgrad and wgrad | Synchronized median 1,292.384 → 1,266.959 µs for late transpose and 1,287.769 → 1,271.201 µs for early transpose. Paired mean changes −1.71% [−2.56%, −0.87%] and −1.54% [−2.39%, −0.69%] across 16 AB/BA pairs each. | Four FP8/FP32 planes and both BF16 gradients matched bitwise. The GIFT device peak stayed at 1.10828 GiB in both schedules. Peak allocated memory rose by 32 MiB with late transpose and matched with early transpose. | Advance only the QKV shape to a native trial. [Full cut and evidence](#dense-dy-through-dgrad-and-wgrad) |
| Routed SwiGLU forward CTA mapping | Resident CUDA Graph median wall time 199.661 → 166.512 µs. Paired change −16.60% [−16.61%, −16.59%]. The identical control differed by +0.003% [−0.005%, +0.011%]. | Native captured FP8 values and FP32 scales matched bitwise. GIFT device peak was 1.14929 GiB in both arms. The profiled kernel used 160 → 114 registers per thread. | Advance to native 10+10 with the three original fixes. [Producer and timing scope](#routed-swiglu-forward-cta-mapping) |
| CuTe F2 with an FP8 output epilogue | Resident CUDA Graph wall time rose from 105.774 to 133.139 µs, a paired change of +26.45% [+25.77%, +27.14%]. Sixteen AB/BA pairs used 100 warmups and 256 timed calls per arm. The median sum of GPU kernels rose from 94.560 to 122.095 µs in a separate profile. | BF16 output, `rho`, `P`, FP8 bytes and FP32 scales matched bitwise on captured rows, graph replay and the 68-row edge case. Both arms retained the BF16 output and peaked at 1.51062 GiB in the resident graph process. | Keep the separate quantizer for the dual-consumer path. [Component result](../artifacts/raw/norm-out-quant-component-001/component.json) · [Kernel patch](norm_out_quant_experiment.patch) |
| Same-layer plain CuTe F1 and F2 in one CTA cluster | Resident synchronized latency rose from 101.276 to 140.985 µs, a paired change of +39.40% [+38.75%, +40.06%]. CUDA-event time rose by 42.08% [+41.39%, +42.78%]. Sixteen AB/BA pairs used 100 warmups and 256 timed calls per arm. | BF16 output and both FP32 statistics matched at 65 and 8,192 rows. The driver reported a bound of 15 active clusters for the exact launch requesting 14. Both arms reached 1.47351 GiB with the control workspaces resident. | Keep the separate F1 and F2 kernels. [Tail result](../artifacts/raw/norm-same-layer-cluster-tail-001/component.json) · [Native-shaped result](../artifacts/raw/norm-same-layer-cluster-native-001/component.json) · [Occupancy](../artifacts/raw/norm-same-layer-cluster-native-001/occupancy.json) |
| Residual TMA ring depth | Four and six stages had mixed results against nine stages. Both arms shared a 0.9130 GiB component peak. | Output and all gradients matched bytewise. | Keep the nine-stage baseline. |
| Late residual load | Synchronized live SAC replay about 1.20 ms in both baseline processes, 1.44 and 1.30 ms in the late-load processes. Both arms shared a 0.9130 GiB component peak. | The variant used the original BF16 rounding. The [experiment patch](late-residual-experiment.patch) preserves its kernel source. | Keep the TMA-staged residual path. |

Raw artifact root on `sr008-gpu1` is `/workspace-SR008.nfs2/vkropotin/shk-fix/runs/shk-fix/artifacts/raw`. Component runs are `norm-bwd-skip-002`, `dense-wgrad-width-001`, `dense-wgrad-ffn-001-{a,b}`, `moe-scale-layout-003`, `residual-f2-depth-001`, and `residual-f2-late-001`.

The first row-major scale attempt passed CPU routing indices into the
CUDA converter and failed before timing. The rerun moved the indices
to CUDA up front, checked input devices, and synchronized each arm's
parity call before measuring.

The routed FP8 transpose probe measures both planes of each projection
at the configured 16-expert and 32-expert shapes. Paired CUDA-event changes
were +4.55% [4.44%, 4.67%] and +7.38% [7.25%, 7.52%] for EP8 and EP4 w13,
and −5.80% [−5.94%, −5.66%] and −0.02% [−0.25%, +0.21%] for EP8 and EP4 w2.
For EP8 w2 the profiled GPU work alone rose from 75.74 to 86.37 µs. The
probe uses standalone compiled calls with CUDA Graphs off and measures
no GEMM, autograd, or collectives.

In the skip-gradient component trace, `GnBwdDx` rose from 81.25 to
95.91 µs while the separate 62.37 µs add disappeared. Shared memory
per CTA rose from 184 to 212 KiB. Both kernels used 168 registers per
thread. These are profiled launches from `norm-bwd-skip-002/result.{A,B}.trace.json`.

In `norm-dy-merge-001`, the rebased vector store wrote the correct global
BF16 merge and failed gradient parity. Directly indexing the existing
swizzled shared tile restored bitwise gradients in `norm-dy-merge-002`.
The probe uses compiled CuTe and Triton kernels with a resident norm
backward input. The other two branches are rolls of the captured tangent.

The F1 compiled repeat used 200 warmups per arm and 16 paired AB/BA blocks
with 512 calls per arm, for 16,384 timed calls. Synchronized median block
times were 1.271478 → 1.271473 ms. The four F1 launches accounted for
131.648 → 123.105 µs of summed kernel durations in a separate profile.
The remaining 16 kernels accounted for 570.526 → 570.305 µs. These are
medians over seven profiled calls after the first call. Launch geometry,
registers and shared memory per CTA matched. Both allocator peaks matched
throughout timing. [Baseline trace](../artifacts/raw/norm-f1-compiled-002/component.A.trace.json) · [Candidate trace](../artifacts/raw/norm-f1-compiled-002/component.B.trace.json)

Across ranks 0 and 1 in the historical five-fusion training capture, 473
of 474 H4096 F1 launches had their same-stream successor `GnFwdOut` already
submitted when F1 ended. The median GPU gaps to that successor were
2.46 and 2.27 µs. This measures submission slack at the F1 boundary in
the historical recipe. Exposed savings over a current training update
remain unmeasured. [Rank 0](../../../../../artifacts/4fix_qla_embed_selector_gpu_5fusions.json) · [Rank 1](../../../../../artifacts/4fix_qla_embed_selector_gpu_5fusions_rank1.json)

The first CUDA Graph attempt failed before measurement. Initial autograd
parity on the default stream left the leaf accumulation nodes bound to
that stream. Running the initial parity calls and warmup on the capture
stream fixed the error. The completed diagnostic used two independent
capture orders, 200 graph warmups per capture and 16 AB/BA timing blocks.
Each arm used 256 CUDA-event samples of eight replays per block.
Median block times were 689.116 → 684.382 µs per replay. The paired
mean change was −4.617 µs [−4.932, −4.302]. The two sets of captures
had distinct allocator pools. Memory is reported above from the
ungraphed pair. [Harness](norm_compiled_stage_component.py) · [Baseline graph trace](../artifacts/raw/norm-f1-compiled-graph-002/component.A.trace.json) · [Candidate graph trace](../artifacts/raw/norm-f1-compiled-graph-002/component.B.trace.json)

The resident norm-pair candidate replaced the 75.680 µs F2 and 33.088 µs
F1 kernel medians with a single 95.888 µs kernel. The grid remained 116
blocks of 384 threads, with 168 registers per thread. Residual F2 shared
memory grew from 202 to 224 KiB. The first attempt's profiler-order check
failed after parity. The rerun checked each synchronized call's kernel
sequence by GPU timestamps. The resident replay measures equal workspaces
and excludes co-allocation through the full replay. [Baseline trace](../artifacts/raw/norm-pair-tile-reuse-002/component.A.trace.json) · [Candidate trace](../artifacts/raw/norm-pair-tile-reuse-002/component.B.trace.json)

Both compiled series used 200 warmup calls per arm before timing, then
16 matched blocks with eight in each order. Each arm started each block
with four extra warmup calls and recorded 512 timed calls. The first series
measured 1,259.163 → 1,097.404 µs, a paired change of −12.91% [−12.97%, −12.86%].
The guarded repeat required contiguous CUDA BF16 tensors of shape
`8192 × 4096`, the expected tuple consumers, and only local preparations
between the two norm calls. Its forward event interval fell by 13.03%
[12.86%, 13.20%]. The backward interval, which includes SAC replay, fell by
11.80% [11.67%, 11.93%]. Each full call fused one pair in live forward and
one pair in replay after the statistics-only pass. The profiled CuTe
launch count fell from 20 to 18, with identical backward kernel counts.
Both arms allocated a peak of 419,710,976 bytes and reserved 427,819,008
bytes. The synchronized timer covers replay and autograd. The intervals
use paired block medians. In the guarded profile, summed GPU kernel
time per call fell from 700.0 to 678.1 µs. The unprofiled host submission
median fell from 1,322.6 to 1,151.7 µs. These timings measure the component's
launch and replay schedule. The separate native comparison measures
the effect inside training. [First series](../artifacts/raw/norm-pair-tile-compiled-001/component.json) · [Guarded series](../artifacts/raw/norm-pair-tile-compiled-002/component.json) · [Baseline trace](../artifacts/raw/norm-pair-tile-compiled-002/component.A.trace.json) · [Candidate trace](../artifacts/raw/norm-pair-tile-compiled-002/component.B.trace.json) · [Compiled harness](norm_pair_tile_compiled.py)

### Dense dY through dgrad and wgrad

The QKV replay uses BF16 projection input and weight from the historical
real-data capture at `0f0bf5604`. The current stock FLA short-convolution
backward regenerates BF16 dY from the captured dQ/dK/dV outside timing. The
cut starts with columnwise X resident and includes transpose weight
quantization and both GEMMs. The profiled control creates columnwise dY
after dgrad. Fusion brings that 66 MiB column data and scale plane forward.
[Fixture derivation](dense_dy_qkv_fixture.py) · [Fixture receipt](../artifacts/raw/dense-dy-qkv-fixture-001/run.log)

| QKV metric | Separate | Fused |
| --- | ---: | ---: |
| Synchronized call median, µs | 1,292.384 | 1,266.959 |
| CUDA-event median around host calls, µs | 1,272.712 | 1,247.592 |
| Host submission median, µs | 333.679 | 340.200 |
| Host drain median, µs | 955.580 | 925.341 |
| Kernel sum in a separate single-call profile, µs | 1,129.272 | 1,104.454 |
| Peak allocated / reserved, MiB | 453.0 / 504.0 | 485.0 / 504.0 |
| GIFT device peak, GiB | 1.10828 | 1.10828 |

The paired wall-time change was −1.71% [95% CI −2.56%, −0.87%]. The paired
CUDA-event change was −1.75% [−2.62%, −0.88%]. Both cuBLASLt kernels kept
their symbols and launch grids. Each arm issued the same two workspace
memsets and the cut has no collectives. The caller holds BF16 dY and weight
throughout, so the isolated peaks describe this replay. The profiler and
the unprofiled paired timers cover separate invocations.
[QKV result](../artifacts/raw/dense-dy-backward-qkv-001/component.json) · [Control trace](../artifacts/raw/dense-dy-backward-qkv-001/component.A.trace.json) · [Candidate trace](../artifacts/raw/dense-dy-backward-qkv-001/component.B.trace.json)

The initial early-transpose control scheduled dgrad before the transpose and
was rejected before timing. Disabling locality and peak-memory reordering in
both arms reproduced the other native order, with columnwise dY created
before dgrad in the control. On the same captured operands, synchronized
medians were 1,287.769 → 1,271.201 µs, a paired change of −1.54% [95% CI
−2.39%, −0.69%]. CUDA-event medians were 1,268.664 → 1,252.504 µs, host
submission 324.102 → 337.277 µs, and host drain 962.847 → 933.152 µs. The
single-call kernel sum decreased from 1,172.061 to 1,142.878 µs, with matched
GEMM names and grids and the same two memsets. All six planes matched
bitwise. Both arms peaked at 485.0 MiB allocated, 504.0 MiB reserved and
1.10828 GiB by the GIFT formula.
[Rejected schedule](../artifacts/raw/dense-dy-backward-qkv-early-001/component.json) · [Early transpose result](../artifacts/raw/dense-dy-backward-qkv-early-002/component.json) · [Early transpose patch](dense-dy-backward-qkv-early.patch)

The first synthetic shared-W2 replay matched both gradients and all four
FP8/FP32 planes bitwise. Inductor scheduled wgrad ahead of dgrad in both
arms and moved the candidate X transpose ahead of fused quantization. The
schedule check rejected the cut before timing. [First W2 receipt](../artifacts/raw/dense-dy-backward-w2-001/component.json)

A fresh W2 replay disabled Inductor locality and peak-memory reordering in
both arms to reproduce the emitted training order. The trace confirmed
identical GEMM symbols and grids. Synchronized medians were 394.670 →
402.295 µs, a paired increase of 1.75% [95% CI 1.51%, 1.99%]. CUDA-event
medians were 377.912 → 385.344 µs, and host submission medians were 324.269
→ 340.857 µs. The separate profile reduced kernel sum from 266.400 to
257.857 µs and retained two workspace memsets. Both BF16 gradients and all
four FP8/FP32 planes matched bitwise. Both arms peaked at 0.84460 GiB by
the GIFT formula. This diagnostic uses seeded native-shaped operands and
omits the intervening SwiGLU backward and unrelated FP8 preparation.
Keep the original shared-W2 path. [Schedule-controlled W2 receipt](../artifacts/raw/dense-dy-backward-w2-002/component.json) · [Compiler options patch](dense-dy-backward-fixed-order.patch)

The native QKV bridge passed a separate Hopper smoke on the captured projection.
The stock FLA backward returned contiguous dY with stride
`(33554432, 8192, 1)`. The compiled candidate launched exactly once with
grid `(64, 64, 1)` and its two GEMM consumers matched the compiled audit.
Projection output, dX and dW matched bitwise. Both verified CudaCherry
extension hashes match the component receipts.
The initial native preflight compared the Triton module version recorded in
the component to the distribution version with its build suffix. The
corrected preflight compares module versions and records both values.
[Smoke harness](dense_dy_qkv_native_smoke.py) · [GPU smoke](../artifacts/raw/shk-dense-dy-qkv-10x10-001-launcher/smoke/run.log) · [Initial preflight log](../artifacts/raw/shk-dense-dy-qkv-10x10-001-launcher/run.log)

### Routed SwiGLU forward CTA mapping

The routed forward producer loads gate and up activations, multiplies by
the routing probability and applies the native rowwise FP8 quantizer.
The experiment exchanges token and feature CTA axes. Each program still
computes the same 64 × 128 tile, with the same clamp and FP32 arithmetic.
On captured EP8 rank-zero operands, the grid changes from `(994, 12, 1)` to
`(12, 994, 1)`. The FP8 data and FP32 scales matched the stock wrapper
bitwise. The surgical rewrite follows the program-ID remapping technique
described in the Triton tutorial. [Producer probe](routed_swiglu_fwd_order_component.py) · [Triton program ordering](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#l2-cache-optimizations)

The resident graph benchmark used 100 warmup launches, 256 launches per
graph, 16 alternating A/B pairs and 16 repeated baseline pairs. Median
CUDA-event time was 199.514 → 166.365 µs. The two single-launch profiles
show 200.800 → 169.024 µs, 160 → 114 registers per thread and a placement
limit of 3 → 4 CTAs per SM. The shared-memory sizes and 128-thread block
size matched. No device copies or memsets appeared in either profile.
Allocated and reserved peaks matched. The timed graph has preallocated
operands and outputs. It excludes downstream routed w2, collectives and
the recompiled training graph's host overhead.
[Timing, identical control and launch evidence](../artifacts/raw/routed-swiglu-fwd-order-001/component.json)

The first native EP8 control completed at 157,192.57 TPS and 70.40 GiB.
The candidate stopped before update 1 because its Python grid-range
condition tried to evaluate the unbacked HybridEP token count during
compilation. The static-shape GPU smoke could not exercise this condition.
This attempt has no candidate timing.
[Run order](../artifacts/raw/shk-swiglu-fwd-10x10-001/matrix.tsv) · [Failure log](../artifacts/raw/shk-swiglu-fwd-10x10-001-launcher/run.log)

The corrected bridge expresses the CUDA grid limit as a symbolic host shape
check during compilation. A Hopper smoke replayed two unbacked token counts
through the same compiled graph, with the expected launch geometry and
bitwise equality for both output planes.
[Dynamic smoke harness](routed_swiglu_fwd_native_smoke.py) · [GPU smoke](../artifacts/raw/shk-swiglu-fwd-10x10-002-launcher/smoke/run.log)

## Native 10+10

`shk-norm-pair-10x10-001` compares the three-fix control with the additional
`GIFT_CUTE_NORM_PAIR=1` rewrite. Both topologies ran candidate first.
Peak total GIFT memory is the maximum over all eight ranks and updates
11–20. The half-window checks use the equivalent TPS change from the
maximum rank-local elapsed time in each five-update interval.

| Topology | Global TPS, control → candidate | TPS Δ | Peak total GIFT memory, GiB | Half-window TPS Δ |
| --- | ---: | ---: | ---: | ---: |
| EP8 / eFSDP1 | 156,728.67 → 157,184.84 | +0.291% | 70.41 → 70.28 | +0.453%, +0.130% |
| EP4 / eFSDP2 | 147,867.20 → 148,432.44 | +0.382% | 73.32 → 73.56 | +0.387%, +0.377% |

All four arms completed with stable compilation and each candidate
logged activation on all eight ranks. The generated wrappers contain 104
live and eight stats-only fused calls for EP8, and 64 live fused calls
for EP4, across rank and graph variants. These are one native pair per
topology. The bitwise result covers the isolated SAC replay.
The GIFT peak deltas track allocator reservation. Maximum PyTorch
reserved fell by 0.133 GiB on EP8 and rose by 0.244 GiB on EP4. Maximum
allocated changed by +1.5 MiB and −3.2 MiB respectively. Each arm recorded
zero allocator retries and OOMs in the measured window.
[Native matrix](../artifacts/raw/shk-norm-pair-10x10-001/matrix-results.json) · [Native measurement bridge](native_norm_pair_measure.py)

`shk-norm-skip-10x10-001` compares the three-fix control to `GIFT_CUTE_NORM_BWD_SKIP=1`. The [acceptance runner](../acceptance.py) freezes the source and resolved settings, runs both EP8 and EP4, and reports physical global TPS plus the maximum native total GPU memory across eight ranks over updates 11–20. The launcher and PID are in `shk-norm-skip-10x10-001-launcher` under the raw root.

`shk-norm-skip-10x10-002` repeated the same active switches with the
candidate first on both topologies. All four arms used one frozen
`shk-fix` snapshot. That snapshot also included the later default-off
follow-up code. [Frozen inputs](../artifacts/raw/shk-norm-skip-10x10-002/preflight.json) · [Executed order](../artifacts/raw/shk-norm-skip-10x10-002/matrix.tsv)

`shk-dense-width-10x10-001` compares the same control to `GIFT_FP8_WGRAD_LOGICAL_WIDTH=1`. Its launcher waits for the skip campaign to exit successfully and checks all eight GPUs before starting. The other follow-up switches are disabled for each campaign.

`shk-routed-bf16-10x10-001` compares the same control to `GIFT_ROUTED_DW_BF16_OVERWRITE=1`. All four arms passed on `sr008-gpu1` with one source snapshot and the pinned private DeepGEMM build. Every candidate rank logged the actual overwrite activation once. [Frozen inputs](../artifacts/raw/shk-routed-bf16-10x10-001/preflight.json)

`shk-routed-bf16-10x10-002` repeated the same comparison candidate first
on both topologies. Both routed campaigns used the same frozen source,
switches, and private DeepGEMM build. The four repeat windows were
compile-stable and every candidate rank logged one overwrite activation.
[Frozen inputs](../artifacts/raw/shk-routed-bf16-10x10-002/preflight.json) · [Executed order](../artifacts/raw/shk-routed-bf16-10x10-002/matrix.tsv)

| Candidate | Topology | Global TPS control → candidate | TPS change | Peak total GIFT memory control → candidate |
| --- | --- | ---: | ---: | ---: |
| CuTe skip-gradient, control first | EP8 / eFSDP1 | 157,035.72 → 157,545.86 | +0.325% | 70.41 → 70.41 GiB |
| CuTe skip-gradient, control first | EP4 / eFSDP2 | 147,585.33 → 148,689.29 | +0.748% | 73.32 → 73.34 GiB |
| CuTe skip-gradient, candidate first | EP8 / eFSDP1 | 156,898.74 → 157,168.32 | +0.172% | 70.41 → 70.43 GiB |
| CuTe skip-gradient, candidate first | EP4 / eFSDP2 | 147,841.92 → 148,803.72 | +0.651% | 73.32 → 73.34 GiB |
| Logical-width dense w2 gradient | EP8 / eFSDP1 | 157,265.29 → 156,581.24 | −0.435% | 70.40 → 70.41 GiB |
| Logical-width dense w2 gradient | EP4 / eFSDP2 | 147,843.72 → 148,364.67 | +0.352% | 73.34 → 73.34 GiB |
| Routed BF16 dW overwrite, control first | EP8 / eFSDP1 | 157,198.78 → 157,427.29 | +0.145% | 70.41 → 70.41 GiB |
| Routed BF16 dW overwrite, control first | EP4 / eFSDP2 | 148,358.59 → 148,896.21 | +0.362% | 73.48 → 73.32 GiB |
| Routed BF16 dW overwrite, candidate first | EP8 / eFSDP1 | 157,234.39 → 157,233.12 | −0.001% | 70.41 → 70.40 GiB |
| Routed BF16 dW overwrite, candidate first | EP4 / eFSDP2 | 148,239.38 → 148,621.45 | +0.258% | 73.34 → 73.34 GiB |

All measured skip-gradient windows were compile-stable. In the first pair, every EP8 candidate rank logged six compiled graphs with two rewritten sites each. Every EP4 candidate rank logged four such graphs. In the candidate-first repeat all eight ranks on both topologies logged four graphs with two rewritten sites each. Both controls in the repeat logged zero. [Control-first matrix](../artifacts/raw/shk-norm-skip-10x10-001/matrix-results.json) · [Candidate-first matrix](../artifacts/raw/shk-norm-skip-10x10-002/matrix-results.json)

In the control-first skip-gradient pair, rank-0 median step TPS on EP8 was 19,643 → 19,731 for updates 11–15 and 19,642 → 19,625 for updates 16–20. EP4 medians were 18,483 → 18,585 and 18,458 → 18,579 over the same halves. In the candidate-first pair, the respective medians were 19,633 → 19,686 and 19,603 → 19,580 on EP8, and 18,538 → 18,601 and 18,453 → 18,587 on EP4. Both topologies gained over the full window in both orders. EP8 varied between halves. In the repeat all ranks reported finite loss and gradient norms. Rank-0 printed loss gaps reached 0.00030 on EP8 and 0.00043 on EP4. Printed gradient-norm gaps reached 0.0027 and 0.0079 during warmup.

The dense matrix preserved one source snapshot in all four arms and verified the requested switch values, 20 completed updates per rank, and stable compilation. The isolated compiled FFN replay verified the narrower GEMM operand and bitwise parity. Sixteen native Inductor wrappers emitted during the EP8 candidate launch contain the 10,944-wide wgrad GEMM operand. Rank-0 median step TPS was 19,695 → 19,645 and 19,619 → 19,515 in the two EP8 halves. EP4 medians were 18,488 → 18,561 and 18,487 → 18,531. Keep the change disabled by default. A repeated EP4 pair would be needed before treating the gain as repeatable. [Dense matrix](../artifacts/raw/shk-dense-width-10x10-001/matrix-results.json)

In the control-first routed pair, rank-0 median step TPS rose in both halves on EP8 (19,674 → 19,709 and 19,618 → 19,643) and EP4 (18,528 → 18,625 and 18,567 → 18,592). In the candidate-first pair, EP8 was 19,689 → 19,638 and 19,604 → 19,654. EP4 was 18,502 → 18,575 and 18,547 → 18,616. EP4 gained in both full windows. The EP8 repeat was flat. Native loss and gradient norms were finite. In the first pair, the largest printed loss gaps were 0.00036 on EP8 and 0.00028 on EP4. In the repeat they were 0.00035 and 0.00034. The largest printed gradient-norm gaps in the repeat were 0.0412 and 0.0454, both at warmup update 8. The bitwise check applies to the isolated grouped weight-gradient outputs. The native switch remains opt-in. [Control-first matrix](../artifacts/raw/shk-routed-bf16-10x10-001/matrix-results.json) · [Candidate-first matrix](../artifacts/raw/shk-routed-bf16-10x10-002/matrix-results.json)

### Combined native 10+10

`shk-stack-10x10-001` compared the three-fix control with all three additional
switches enabled together (`GIFT_CUTE_NORM_BWD_SKIP`, `GIFT_CUTE_NORM_PAIR`, and
`GIFT_ROUTED_DW_BF16_OVERWRITE`). The control ran first on each topology. Native
FLA, FP8, selective AC, compile, the real `data-002` input, the original LR
schedule and the eight-GPU crop were unchanged. Within each topology the
resolved configuration differs only in the output directory. The preflight
pins one GIFT snapshot at `e9b7cdbb4`, the norm-pair source and the private
DeepGEMM binary. [Frozen inputs](../artifacts/raw/shk-stack-10x10-001/preflight.json) · [Execution order](../artifacts/raw/shk-stack-10x10-001/matrix.tsv)

| Topology | Physical global TPS, control → combined | TPS Δ | Median maximum rank-local update, s | Peak total GIFT memory, GiB | Half-window equivalent TPS Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| EP8 / eFSDP1 | 157,023.50 → 157,902.79 | +0.560% | 13.3623 → 13.2962 | 70.41 → 70.30 | +0.536%, +0.585% |
| EP4 / eFSDP2 | 148,155.76 → 149,474.65 | +0.890% | 14.1599 → 14.0376 | 73.34 → 73.56 | +0.871%, +0.912% |

The global timer covers all 20,971,520 physical input tokens in updates 11–20.
Supervised throughput was 156,985.17 → 157,864.24 tokens/s on EP8 and
148,119.59 → 149,438.15 on EP4. The half-window check sums the maximum rank-local
elapsed interval per update in each half. There is one paired comparison per
topology. The measurement is end to end and unprofiled, with no new breakdown
of device compute, host waits or collectives.

Both candidates logged the norm-pair and routed overwrite activations once
per rank. Skip-gradient fusion rewrote two sites in each of six graphs per EP8
rank and four graphs per EP4 rank. All four arms finished with finite loss and
gradient norms and `observed_stable` compiler status. Across ranks and
measured updates, the maximum absolute printed loss and gradient-norm gaps
were 0.00043 and 0.0022 on EP8 and 0.00024 and 0.0011 on EP4. The GIFT memory peak is
the maximum over all eight ranks and updates 11–20. Its change tracks PyTorch
reservation in this run (−0.113 GiB on EP8 and +0.225 GiB on EP4). Allocator
retries and OOMs were zero in every measured window. [Matrix and activation receipts](../artifacts/raw/shk-stack-10x10-001/matrix-results.json) · [Launcher log](../artifacts/raw/shk-stack-10x10-001-launcher/run.log)

### Dense GDN QKV dY native 10+10

`shk-dense-dy-qkv-10x10-002` enabled only the fused dY quantization and
transpose in the candidate. Both arms had the original three fixes. On
each topology the control ran first, and the resolved configurations
matched after normalizing the output path. The unprofiled window covered
20,971,520 global physical input tokens across updates 11–20, after ten
warmup updates. [Preflight inputs and component receipts](../artifacts/raw/shk-dense-dy-qkv-10x10-002/preflight.json)

| Topology | Global TPS, control → candidate | TPS Δ | Peak total GIFT memory, GiB | Half-window equivalent TPS Δ |
| --- | ---: | ---: | ---: | ---: |
| EP8 / eFSDP1 | 156,742.73 → 156,716.92 | −0.016% | 70.40 → 70.41 | −0.273%, +0.239% |
| EP4 / eFSDP2 | 148,194.37 → 147,875.42 | −0.215% | 73.34 → 73.34 | −0.351%, −0.077% |

The candidate launched during warmup on each rank. Post-run audits found
four compiled sites per rank with both native BF16 GEMM consumers. The
control audit found zero. Loss and gradient norms were finite, compilation
stayed stable in measured windows, and there were zero allocator retries
or OOMs. The largest printed loss and gradient-norm gaps across measured
updates were 0.00023 and 0.0009 on EP8 and 0.00040 and 0.0023 on EP4.
The native pair showed no TPS gain on either topology. Keep the QKV switch
disabled.
[Native matrix](../artifacts/raw/shk-dense-dy-qkv-10x10-002/matrix-results.json) · [Launcher log](../artifacts/raw/shk-dense-dy-qkv-10x10-002-launcher/run.log)

### Routed SwiGLU forward CTA order native 10+10

`shk-swiglu-fwd-10x10-002` compared the original three-fix control with
the feature-first forward grid enabled. Both topologies ran the control
first and measured the 20,971,520 global physical input tokens in
updates 11–20. The resolved configurations matched within each pair
after normalizing the output path.
[Frozen inputs](../artifacts/raw/shk-swiglu-fwd-10x10-002/preflight.json)

| Topology | Global TPS, control → candidate | TPS Δ | Peak total GIFT memory, GiB | Half-window equivalent TPS Δ |
| --- | ---: | ---: | ---: | ---: |
| EP8 / eFSDP1 | 156,734.30 → 157,415.40 | +0.435% | 70.40 → 70.40 | +0.350%, +0.519% |
| EP4 / eFSDP2 | 147,624.61 → 148,484.85 | +0.583% | 73.34 → 73.32 | +0.852%, +0.315% |

Every rank used the expected compiled launch grid during warmup. The
observer removed itself before update 1. All arms had finite loss and
gradient norms, stable observed compilation, and zero allocator retries
or OOMs. The largest printed loss and gradient-norm gaps over measured
updates were 0.00038 and 0.0013 on EP8 and 0.00029 and 0.0010 on EP4.
Each topology has one control-first pair with a gain in both halves.
[Native matrix](../artifacts/raw/shk-swiglu-fwd-10x10-002/matrix-results.json) · [Launcher log](../artifacts/raw/shk-swiglu-fwd-10x10-002-launcher/run.log)

### Routed SwiGLU forward with the six-fix stack

`shk-stack-swiglu-fwd-10x10-001` keeps the six switches from the joint
CuTe and routed BF16 dW trial active in both arms. Only the SwiGLU
forward CTA order changes. Both topologies ran the candidate first.
The frozen configuration retains the original-A workload, compiler and
SAC settings and counts 20,971,520 global physical tokens in updates 11–20.
[Frozen inputs](../artifacts/raw/shk-stack-swiglu-fwd-10x10-001/preflight.json)

| Topology | Global TPS, six → seven | TPS Δ | Peak total GIFT memory, GiB | Half-window equivalent TPS Δ |
| --- | ---: | ---: | ---: | ---: |
| EP8 / eFSDP1 | 157,824.15 → 157,601.69 | −0.141% | 70.28 → 70.30 | −0.466%, +0.187% |
| EP4 / eFSDP2 | 149,450.63 → 149,784.10 | +0.223% | 73.60 → 73.59 | +0.217%, +0.231% |

All four arms completed with stable observed compilation and zero
allocator retries and OOMs. The six common switches were active in both
arms. The SwiGLU launch audit finished before update 1 on every rank.
The maximum printed loss and gradient-norm gaps were 0.00028 and
0.0017 on EP8, and 0.00028 and 0.0031 on EP4. GIFT peak changes tracked
reservation, while maximum PyTorch allocated fell by about 1 MiB on
EP8 and 8 MiB on EP4.

The EP8 halves had opposite signs. EP4 gained in both halves of its
single matched pair. This comparison leaves a common advantage for the
seventh switch unresolved. Keep the six-fix stack as the shared default
for this comparison. [Native matrix](../artifacts/raw/shk-stack-swiglu-fwd-10x10-001/matrix-results.json) · [Launcher log](../artifacts/raw/shk-stack-swiglu-fwd-10x10-001-launcher/run.log)

### Routed row2col hardware warp maximum

The two private CUDA builds use the same pinned CudaCherry source and
compiler flags. The candidate replaces the shuffle reductions with unsigned
warp maxima of the nonnegative FP32 bit patterns. Disassembly shows
80 shuffle instructions in the control and 16 `REDUX.MAX` instructions
in the candidate. The installed extension also ran as a build anchor.
The timing uses 100 warmup calls and 16 alternating pairs with 256
synchronized compiled calls per arm per pair. The EP4 inputs are derived
from the EP8 capture. Negative changes mean lower latency.

| Shape | dY conversion Δ | Input conversion Δ | Through BF16 wgrad Δ, paired 95% CI | Through wgrad peak GIFT, GiB |
| --- | ---: | ---: | ---: | ---: |
| Captured EP8 W2 | +1.42% | +1.06% | −0.40% [−0.91%, +0.09%] | 2.3934 → 2.3934 |
| Captured EP8 W13 | +2.51% | +0.86% | −0.09% [−0.37%, +0.20%] | 3.1434 → 3.1434 |
| Derived EP4 W2 | +1.54% | +1.71% | −0.23% [−1.07%, +0.54%] | 3.1434 → 3.1434 |
| Derived EP4 W13 | +2.92% | +1.32% | −0.48% [−0.80%, −0.20%] | 4.6434 → 4.6434 |

The converted FP8 bytes, FP32 scales, layout and BF16 wgrad matched
bitwise. Special E4M3 and FP32 values and output canaries matched.
Both isolated converter roles slowed in all four shapes, with all eight
intervals excluding zero. Through wgrad, only the derived EP4 W13
interval excluded zero. Keep the installed reduction with no native trial.
[Component results](../artifacts/raw/routed-row2col-redux-001/component/component.json) · [Control disassembly](../artifacts/raw/routed-row2col-redux-001/component/shk_row2col_control/kernel.sass) · [Candidate disassembly](../artifacts/raw/routed-row2col-redux-001/component/shk_row2col_redux/kernel.sass)

### Routed SwiGLU backward CTA order

The experiment changes the CTA grid from `(1988, 12, 1)` to
`(12, 1988, 1)` on the captured EP8 inputs. Each CTA still computes
the same 32 × 128 tile with the native row-major scale layout.
The fullgraph Inductor component includes the stock routing-gradient
reduction, scale transpose and DeepGEMM dgrad. The FP8 bytes, scale bits,
routing gradients and all 63,616 active dgrad rows matched bitwise.
The two profiles show the expected grid and the same other launches.
The target kernel used 128 registers per thread in each arm. Its single
profiled duration was 328.48 → 263.27 µs.

After 100 warmups per arm, 16 alternating pairs with 256 synchronized
calls per arm gave a median wall time of 1.81264 → 1.77100 ms.
The paired latency change was −2.10% [−3.30%, −0.91%]. Peak total GIFT
memory was 2.0751 GiB in every block. This component uses captured EP8
inputs and ends at dgrad. Advance to an independent native EP8 and
EP4 10+10 pair against the original three-fix control.
[Component and per-pair results](../artifacts/raw/routed-swiglu-bwd-order-001/result.json) · [Control trace](../artifacts/raw/routed-swiglu-bwd-order-001/result.A.trace.json) · [Candidate trace](../artifacts/raw/routed-swiglu-bwd-order-001/result.B.trace.json)

The first two dynamic-shape smoke attempts stopped before a CUDA
launch. Dynamo rejected `SimpleNamespace` inside the compiled graph
and then rejected passing a synthetic context to the class's special
autograd backward method. The third smoke binds each arm's exact
installed backward function outside compilation and calls it with a
typed context. The H100 run used the same compiled graph for 128 and
256 unbacked rows and matched the FP8 data, row scales and routing
gradient bitwise. The first-launch audit saw `(4, 12, 1)` and
`(12, 4, 1)`. This scope checks the dynamic backward implementation
with a synthetic upstream gradient. Native training uses the normal
AOT autograd entry point.
[First smoke](../artifacts/raw/shk-swiglu-bwd-10x10-001-launcher/smoke/run.log) · [Second smoke](../artifacts/raw/shk-swiglu-bwd-10x10-001-launcher/smoke-002/run.log) · [Passing smoke](../artifacts/raw/shk-swiglu-bwd-10x10-001-launcher/smoke-003/run.log)

### Routed SwiGLU backward CTA order native 10+10

`shk-swiglu-bwd-10x10-001` compared the backward-only grid change
against the original three-fix control. Each topology ran the control
first. Both arms measured updates 11–20 with 20,971,520 global physical
tokens. The production graph used the existing forward saves and row-major
scales. All eight ranks confirmed the expected grid for their arm before
the measured window.

| Topology | Global TPS, control → candidate | TPS Δ | Peak total GIFT memory, GiB | Half-window equivalent TPS Δ |
| --- | ---: | ---: | ---: | ---: |
| EP8 / eFSDP1 | 157,165.53 → 156,662.71 | −0.320% | 70.41 → 70.41 | −0.443%, −0.199% |
| EP4 / eFSDP2 | 148,086.93 → 148,468.27 | +0.258% | 73.32 → 73.34 | +0.262%, +0.253% |

Compilation remained stable and all arms reported zero allocator retries
and OOMs. The maximum printed loss and gradient-norm gaps were 0.00022
and 0.0028 on EP8 and 0.00024 and 0.0021 on EP4. Both halves slowed on
EP8 and gained on EP4. Keep the backward CTA switch disabled in the common
accepted package.
[Frozen inputs](../artifacts/raw/shk-swiglu-bwd-10x10-001/preflight.json) · [Native matrix](../artifacts/raw/shk-swiglu-bwd-10x10-001/matrix-results.json)

### Final native 10+10 acceptance

`shk-final-10x10-001` ran on the same eight H100 80 GB GPUs on
`sr008-gpu1` using the `shk-fix` worktree based on original A
`e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f`. Each topology ran one
unprofiled trial per arm in the order original A, original three, then
all six accepted fixes. The trials kept the same production-tokenizer
`data-002` crop, compiled FP8 and selective AC settings. The resolved
config matched within each topology after excluding the output directory.
TPS counts all 20,971,520 physical input tokens across updates 11–20.
Peak memory uses the maximum native GIFT memory over all ranks in the
same window.

| Arm | CuTe forward residual fusion | Dense FP8 weight tail | Statistics-only SAC | CuTe backward skip-gradient | CuTe residual F2 → next F1 | Routed BF16 dW overwrite |
| --- | --- | --- | --- | --- | --- | --- |
| Original A | Off | Off | Off | Off | Off | Off |
| Original three | On | On | On | Off | Off | Off |
| All six accepted | On | On | On | On | On | On |

| Topology | Arm | Global TPS | TPS Δ vs A | Peak total GIFT memory, GiB | Memory Δ vs A, GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| EP8 / eFSDP1 | Original A | 156,083.35 | — | 70.34 | — |
| EP8 / eFSDP1 | Original three | 156,723.21 | +0.410% | 70.41 | +0.07 |
| EP8 / eFSDP1 | All six accepted | 157,461.08 | +0.883% | 70.30 | −0.04 |
| EP4 / eFSDP2 | Original A | 147,435.71 | — | 73.38 | — |
| EP4 / eFSDP2 | Original three | 148,061.97 | +0.425% | 73.34 | −0.04 |
| EP4 / eFSDP2 | All six accepted | 149,297.43 | +1.263% | 73.65 | +0.27 |

The direct six-versus-three TPS changes were +0.471% on EP8
(+0.583% and +0.357% by half) and +0.834% on EP4 (+1.064% and +0.608%
by half). Peak GIFT memory changed by −0.11 GiB and +0.31 GiB,
respectively. All six runs passed with stable compilation and zero
allocator retries and OOMs. All eight ranks reported each of the
three follow-up activation markers in both six-fix arms. The maximum
printed loss and gradient-norm gaps between the six-fix arm and
original A were 0.00024 and 0.0028 on EP8 and 0.00034 and 0.0009 on EP4.
The native run measures training throughput and memory. Bitwise parity
checks are scoped to the isolated components above.
[Frozen inputs](../artifacts/raw/shk-final-10x10-001/preflight.json) · [Executed order](../artifacts/raw/shk-final-10x10-001/matrix.tsv) · [Native matrix](../artifacts/raw/shk-final-10x10-001/matrix-results.json)

## Other sites inspected

| Site | Remaining work | Decision |
| --- | --- | --- |
| Three-input gradient merge before CuTe norm backward | The historical trace records an 82.7 µs merge at one repeated site. The native skip-candidate wrapper `coc4zcxrgkuo74m7hvzeo3qwav6a4ysexqk3glegescbfxlrpzrs.py` loads three BF16 tensors, adds them in order in FP32, rounds once to BF16, and reuses the first GEMM output as the destination. Fusing B1 can avoid one 64 MiB read per call. | The isolated fused designs were slower. Native integration would also need to check the two extra 64 MiB branch lifetimes around the opaque norm backward. [Matched norm boundaries](../../../../../runs/compile-baseline/matched-blocks.md#cute-backward-input-assembly-in-moe-layers) |
| GEMM epilogues before the two CuTe gradient merges | In the same generated wrapper, the four-input merge finishes with the FP32 router dgrad. The three-input merge finishes with CudaCherry dgrad after its split-accumulator result rounds to BF16. Each merge adds full-width gradient matrices in a fixed FP32 order and reuses its first input for the BF16 store. | Keep the Triton merges. The available epilogues support one full-matrix accumulator, and the cuBLASLt bias modes broadcast vectors. These interfaces cannot express the required order and intermediate rounding. [cuBLASLt matmul](https://docs.nvidia.com/cuda/archive/13.0.0/cublas/index.html#cublasltmatmul) · [cuBLASLt epilogues](https://docs.nvidia.com/cuda/archive/13.0.0/cublas/index.html#cublasltepilogue-t) · [CudaCherry adapter](../../../gift/torchtitan/ops/float8/_cudacherry_graph_safe.py#L33) |
| CuTe norm auxiliary launches | Across three microbatches of the historical EP8 five-fusion trace, 255 `GnPrep` launches took 1.624 ms. The 129 `GnBwdFin` and 129 `GnBwdWFin` launches took 0.350 and 0.824 ms. `GnPrep` prepares the weight layouts consumed by the forward and backward kernels. | Keep the current preparation and reductions. Their trace budgets are small, and caching preparation through SAC would extend the prepared tensors' lifetimes. [Historical trace](../../../../../artifacts/4fix_qla_embed_selector_gpu_5fusions.json) · [CuTe forward preparation](../../../gift/torchtitan/models/common/norms/cute_gated_norm/forward.py#L64) |
| CuTe F1 sum-square loads | The archived historical capture has 237 H4096 F1 launches totaling 10.555 ms. On the native H4096/R16 split, the SM90a binary has 32 scalar `LDS.U16` instructions in the loop. The experiment emits four `LDS.128` instructions with the same 32 FP32 FMA operations, 168 registers per thread and zero stack or local bytes. | The isolated native-fixture replay above verified parity and measured the reduction. [Baseline disassembly](../artifacts/raw/norm-f1-sass-001/baseline.sass) · [Candidate disassembly](../artifacts/raw/norm-f1-sass-001/candidate.sass) · [Experiment patch](norm_fwd_stats_vector.patch) |
| CuTe B1 gate-weight layouts | B1 stages both the gate weight and its prepared transpose. The experiment uses a second WGMMA descriptor for the same SW32 shared tile. It removes one 2 KiB TMA transfer per stage and 16 KiB of dynamic shared memory per CTA. The compiled loop has 16 → 12 static TMA loads with identical WGMMA instruction counts and 168 registers per thread. | The candidate retained the prepared tensors and ascending K order. The isolated replay found similar latency in both arms. [Baseline disassembly](../artifacts/raw/norm-b1-sass-001/baseline.sass) · [Candidate disassembly](../artifacts/raw/norm-b1-sass-001/candidate.sass) · [Experiment patch](norm_b1_gate_layout.patch) · [Pinned BF16 WGMMA instruction](https://github.com/NVIDIA/cutlass/blob/v4.4.1/include/cute/arch/mma_sm90_gmma.hpp#L2334-L2368) |
| Consecutive post-attention and pre-FFN norms | The three-fix Inductor cache contains 104 direct residual-to-plain norm edges in generated EP8 wrappers and 64 in EP4 wrappers. These are counts in compiled code, with multiple rank and graph variants. At native shape, each residual F2 chunk covers exactly two of the next F1's 512-column splits. Both stages use the same K-major SW128 shared layout for the BF16 tile. | Evaluate the single-kernel resident F2 + next F1 path. It can avoid one 64 MiB logical read while keeping the residual store and all saved tensors. The candidate may need about 22.5 KiB more shared memory and 4.77 MiB of earlier temporary workspace at the pair boundary. [Decoder boundary](../../../gift/torchtitan/models/common/decoder_blocks/transformer_block.py#L118) · [CuTe output kernel](../../../gift/torchtitan/models/common/norms/cute_gated_norm/forward.py#L421) · [Matched component harness](norm_pair_tile_reuse_component.py) |
| Same-layer F1 and F2 | The eight-CTA DSM probe matched the CPU FP32 left fold on every rank in both cancellation-sensitive rounds. Compute Sanitizer reported zero racecheck hazards and zero synccheck errors for that probe. The full plain-norm kernel compiled for SM90a with both WGMMA phases, three cluster rendezvous, local lane-zero TMA releases, and no stack frame. It retains one BF16 input tile per CTA through both phases and exchanges the rounded BF16 gate. | The full resident kernel passed bitwise parity and slowed both timers. The measured kernel used 147,456 bytes of dynamic shared memory per CTA. [DSM racecheck](../artifacts/raw/norm-cluster-dsm-runtime-001/racecheck.log) · [DSM synccheck](../artifacts/raw/norm-cluster-dsm-runtime-001/synccheck.log) · [Full-kernel PTX](../artifacts/raw/norm-same-layer-cluster-offline-001/compiled.ptx) · [Full plain kernel patch](norm_same_layer_cluster.patch) · [Component harness](norm_same_layer_cluster_component.py) |
| CuTe output followed by FP8 input quantization | Rank 0 of the compiled five-fusion trace has 42 same-stream quantizers immediately after `GnFwdOut` at `8192 × 4096`. Thirty-three QKV cuts took 1.512 ms and nine dense FFN cuts took 0.407 ms across three microbatches. Both halves of a 128-column group are in one CTA, in separate warpgroups. | A dense replay wrapper (`c7ee72…:1355–1423`) has a quant-only BF16 consumer. QKV still needs the BF16 store for the decay gate. Shared MoE quantizes after routing, dispatch, and combine, so hoisting would extend a 33 MiB FP8 allocation across those operations. Statistics-only replay has no activation to quantize. The resident prototype keeps the BF16 store and checks all five output planes. Any resident win needs replay through the QKV and gate consumers. [Historical trace](../../../../../artifacts/4fix_qla_embed_selector_gpu_5fusions.json) · [GDN gate and projections](../../../gift/torchtitan/models/common/attention/linear/gdn.py#L76) · [CuTe epilogue](../../../gift/torchtitan/models/common/norms/cute_gated_norm/forward.py#L759) · [Component harness](norm_out_quant_component.py) |
| Dense FP8 dY row quantization and wgrad transpose | Three-fix generated wrappers on both topologies pass the output of `quantize_vec128` to both native dgrad and `scaling_aware_fp8_transpose`, including `8192 × 4096` and `8192 × 8192` gradients. The quantizer rounds BF16 to rowwise E4M3. The transpose chooses the largest row scale in each 128 × 128 tile and shifts E4M3 bytes with flush-to-zero. | The compiled probe checks both exact output layouts using one tile-local kernel. It could remove one rowwise global read and one launch. Emitting the second layout earlier can add 33 or 66 MiB of live data before the original transpose. A native cut must keep the same saved tensors. [FP8 GEMM backward](../../../gift/torchtitan/ops/float8/cuda_gemm.py#L164) · [Pinned CudaCherry converter](../../../../../../hybrid-ep-overlap/sources/cudacherry/src/cudacherry/csrc/quantization/blockwise/row2col_kernel.cu#L372) · [Transformer Engine layout options](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/fp8_blockwise_scaling/fp8_blockwise_scaling.html) · [Compiled probe](dense_dy_quant_transpose.py) |
| Routed FP8 weight transpose before dgrad | The historical rank-0 capture contains 4.559 ms of weight-byte transposes and 0.069 ms of weight-scale transposes over three microbatches. The byte transposes already move about 576 MiB per EP8 projection pair. | The generated transpose leaves this traffic unchanged and slowed w13 in the compiled component. [Current transpose calls](../../../gift/torchtitan/ops/float8/grouped_gemm.py) · [Component result](../artifacts/raw/routed-weight-transpose-001/component.json) |
| Routed FP8 operand descriptors | In pinned DeepGEMM `f093a4cd`, grouped `NN` forwards transpose views to `NT`. The SM90 entry point and TMA encoder require K-major FP8 operands. A transpose view could cover the FP32 scale plane alone. The SM100 grouped `TN` entry point accepts MN-major data. CudaCherry's activation conversion recomputes the column scale and FP8 values from the decoded rowwise inputs. | Keep the established Hopper layouts. The available descriptor interfaces cannot replace the FP8 byte transform, and the scale-only time was 0.069 ms across three historical microbatches. [Grouped API](../../../../../../hybrid-ep-overlap/sources/donors/deepgemm-baseline/csrc/apis/gemm.hpp#L143) · [SM90 TMA setup](../../../../../../hybrid-ep-overlap/sources/donors/deepgemm-baseline/csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp#L141) · [Activation converter](../../../../../../hybrid-ep-overlap/sources/cudacherry/src/cudacherry/csrc/quantization/row2col_dq/row2col_kernel.cu#L180) |
| Routed row2col warp maximum | The stock CudaCherry converter reduces 16 column maxima per warp with five shuffle-and-max rounds. Every lane reaches the reduction on live tiles. The unchanged local calculation starts with positive zero and applies `fmaxf` to absolute values. The resulting FP32 encodings can use an unsigned warp maximum. The four converters accounted for 667.26 µs of summed kernel time in the captured EP8 producer-to-wgrad component. | Keep the installed reduction. The hardware maxima slowed each isolated converter. [Component results](#routed-row2col-hardware-warp-maximum) · [Pinned converter](../../../../../../hybrid-ep-overlap/sources/cudacherry/src/cudacherry/csrc/quantization/row2col_dq/row2col_kernel.cu#L183) · [CUDA warp reduction](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-reduce-functions) |
| Routed row2col expert offsets | HybridEP exposes padded group sizes. GIFT's device loop uses a running offset to write expert IDs and emits only the IDs. The four row2col calls receive the same group counts and IDs. The converter's CTA reads an average of about 7.5 prefix entries at the captured EP8 distribution and 15.5 in its derived EP4 split. Thread zero issues the reads during tile input loading. | Keep the existing prefix loop. The active GIFT interface has no padded-offset tensor to reuse, and no trace establishes a barrier stall from the loop. [HybridEP dispatch](../../../gift/torchtitan/distributed/deepep/hybridep.py#L390) · [Expert ID kernel](../../../gift/torchtitan/ops/float8/triton_kernels/utils.py#L54) · [Converter prefix loop](../../../../../../hybrid-ep-overlap/sources/cudacherry/src/cudacherry/csrc/quantization/row2col_dq/row2col_kernel.cu#L165) |
| Dense w2 direct logical-width transpose | The installed CudaCherry kernel requires a 128-aligned input and derives its output stride from the padded width. A new store specialization could preserve the existing reduction and write the 10,944-column data and scales directly. The current compact-scale copy took 3.744 µs in the component profile. | Defer the vendor specialization. The isolated copy is small and the current logical-width candidate slowed the EP8 training pair. [Component profile](../artifacts/raw/dense-wgrad-width-001/result.json) |
| Shared FFN w1/w3 input FP8 transpose | Original A and the three-fix Inductor wrappers already emit one transpose of the shared quantized input. Both weight gradients consume the same FP8/FP32 output pair. The two distinct gradient operands each require their own transpose. The three-fix launch windows emitted 48 EP8 and 32 EP4 SwiGLU backward wrappers with no duplicate transpose on the same inputs. | Keep the compiler's existing reuse. The historical capture measured the one shared-input transpose at 27.8 µs for shared FFN and 32.9 µs for dense/MTP. [Shared input producer](../../../gift/torchtitan/models/common/ffn/__init__.py#L91) · [Matched dense FP8 paths](../../../../../runs/compile-baseline/matched-blocks.md#decoder-residual-adds-and-dense-fp8-weight-preparation) |
| Shared FFN input FP8 transpose shared-memory layout | The pinned converter uses 132-byte shared rows. A compact 128-byte XOR-swizzled layout could spread each scalar scatter across 32 banks and vectorize the shared-memory reloads while retaining the scale reduction and global output layout. In the historical five-fusion capture, 21 shared input transposes used 0.595 ms on rank 0 and 0.606 ms on rank 1 across three microbatches. | Keep the current converter. Eliminating all of those launches would save about 6.5 ms of summed kernel time over 32 microbatches. A shared-layout change can save only a fraction, before accounting for overlap. The swizzle is a source-level observation with no GPU timing. [Pinned converter](../../../../../../hybrid-ep-overlap/sources/cudacherry/src/cudacherry/csrc/quantization/blockwise/row2col_kernel.cu#L481) · [Historical rank-0 trace](../../../../../artifacts/4fix_qla_embed_selector_gpu_5fusions.json) |
| Dense and shared FFN FP8 GEMM host dispatch | Both historical compiled rank-0 traces show a Python dispatch around the vendor call. Out of 117 matched SwiGLU-neighborhood calls, 114 and 115 had already submitted the vendor work before the previous same-stream operation finished. The remaining submission gaps total 18.2 and 5.6 µs. CudaCherry reuses the plan and workspace. | Keep the current dispatch. A direct fixed-plan call would remove host overhead that the traces show is mostly overlapped. Changing the cuBLASLt plan can change the reduction order and BF16 output. [Baseline trace](../../../../../artifacts/baseline_compile.json) · [Five-fusion trace](../../../../../artifacts/4fix_qla_embed_selector_gpu_5fusions.json) · [Pinned CudaCherry source](../../../../../../hybrid-ep-overlap/sources/cudacherry/src/cudacherry/csrc/gemm/gemm.cu#L683) |
| Dense and shared FFN gradient transpose after SwiGLU backward | The two transposes read distinct rowwise FP8 gradients for gate and up. They took 1.569 ms across all 27 sites in three historical microbatches. Both the dgrad and the wgrad layouts are consumed. An exact dual-output producer must derive the column layout from the rounded rowwise FP8 values and scales. | Keep the staged conversion. The pinned Transformer Engine Hopper implementation derives both scales directly from the source values, and the exact routed counterpart slowed its compiled producer-and-converter chain from 492.00 to 615.74 µs. [Historical trace](../../../../../artifacts/4fix_qla_embed_selector_gpu_5fusions.json) · [NVIDIA quantizer](https://github.com/NVIDIA/TransformerEngine/blob/7457e47d26d9d02dfb783007335dfbe75e1b90a8/transformer_engine/common/transpose/quantize_transpose_vector_blockwise.cu#L343-L386) · [Routed test](../../../../../runs/kernel-fusion/report.md#implementation-audit) |
| Dual-output dense FP8 weight quantizer in SAC replay | One dual-output quantizer could remove an 85.5 MiB BF16 read per dense weight. Twelve historical gate/up pairs separated their normal and transposed launches by 2.37–3.67 ms. In the accepted wrapper, producing both orientations early would extend 86 MiB of FP8 gate/up lifetimes through replay, with up to 129 MiB live when down also replays. | Defer. Transformer Engine has a Hopper dual-output kernel with different shape and scale rules. The added FP8 lifetimes are estimates from the wrapper. [NVIDIA kernel](https://github.com/NVIDIA/TransformerEngine/blob/7457e47d26d9d02dfb783007335dfbe75e1b90a8/transformer_engine/common/transpose/quantize_transpose_square_blockwise.cu#L490-L597) · [Scale rule](https://github.com/NVIDIA/TransformerEngine/blob/7457e47d26d9d02dfb783007335dfbe75e1b90a8/transformer_engine/common/recipe/recipe_common.cuh#L15-L68) |
| Shared activation tiles across CuTe B1 and B2 | H4096 with an SM budget of 116 uses four 1,024-column chunks in B1 and eight 512-column chunks in B2. Preserving the B1 shape takes 256 KiB per 64-row block to keep both BF16 inputs, before weights, statistics or output. B1 and B2 also split their row groups differently. | A simple inter-stage fusion cannot retain the original CTA partition on Hopper with its 227 KiB per-block limit. A cluster variant would need a different tiling and an explicit proof of the existing reduction order and signed rounding. [CuTe partitions](../../../gift/torchtitan/models/common/norms/cute_gated_norm/backward.py#L1202) · [Hopper shared memory limits](https://docs.nvidia.com/cuda/archive/13.1.1/pdf/Hopper_Tuning_Guide.pdf) |
