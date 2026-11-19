# Compile on and off on the selected training baseline

Both runs completed all 21 updates on all eight H100s. With compile, median
unprofiled update time in updates 11–20 fell from 13.313 s to 12.180 s
(−8.52%, equivalent to +9.31% tokens/s at the same batch size). Across the
three profiled microbatches, compile shortened the GPU span by 189.614 ms.
Less time with every GPU stream idle accounted for 109.566 ms of that
difference. The cuDNN, FlashQLA, CuTe, and cross-entropy kernels were present
in both arms. The strongest remaining custom-kernel candidate is the MLA
rotary and Q/K packing path.

[Full layer and kernel comparison](matched-blocks.md) links the historical
PyTorch code to the matched launches, records the selected five fusions,
and ranks the opportunities identified at that point. The [profile PDF](../../docs/compile-fusion-profile.pdf) accompanies this historical analysis. For the current selection, use the [production recipe](../../docs/baseline.md#current-final-task-baseline).

## Comparison contract

Both arms ran the selected S plus GPU capacity selector recipe on one node with
eight NVIDIA H100 80 GB GPUs on `sr008-gpu1`. The source checkout for both arms
is `b9a36408a5795f40503217ed199129687ac4f6a1`. Its GIFT tree fingerprint matches
the preserved S experiment (`6fc4f3c7e6deba5875d13d675704dbdee27b4e09ca3fd106f3a0f61ee6b2b2f5`).
The reused S bridge loads the original 128-counter capacity module in each
process. The comparative adapter verifies that module against the `766fcc4a0`
Git blob. Neither arm imports the grouped-counter candidate.

The two arms keep `data-002`, the production tokenizer, seed 2048, local batch
2, sequence length 4096, accumulation 32, EP8, dense FSDP8, eFSDP1, and PP/TP/CP1.
Both use CuTe norms, FlashQLA `external_gate`, FP8, selective activation
checkpointing, MTP, EMA, the same GFSDP and row-owner embedding controls, and the
saved D compiler and vendor-kernel cache. The eager arm sets
`compile.enable=false` and `TORCHDYNAMO_DISABLE=1`, which also covers helpers
decorated directly with `torch.compile`. Custom CUDA, Triton, and TileLang
kernels still run in eager mode. The compile arm enables model and loss
compilation and CUDA Graph capture is disabled in both arms.

The two native invocations run serially. Each completes 20 warmup updates and
one diagnostic update. Profiling is enabled on ranks 0 and 1 for the first
three complete microbatches of update 21. The trainer still completes the
remaining microbatches and optimizer update. The analysis and downloadable
Chrome traces use rank 0. The trace window excludes the optimizer and the
remaining 29 microbatches. The earlier unprofiled cadence is a separate scope.

From the task root on the verified eight-GPU allocation, use
`bash runs/compile-baseline/campaign.sh <new-campaign-id>`. The retained
campaign ID is `compile-ab-001`. Raw evidence lives in
`runs/compile-baseline/artifacts/raw/` on SR008. The downloaded traces live in
the local ignored `artifacts/` directory as `baseline_compile.json` and
`baseline_no_compile.json`.

## Measured result

| Rank-0 trace, update 21 | Eager | Compile | Compile minus eager |
| --- | ---: | ---: | ---: |
| GPU span through the last linked kernel | 1,338.882 ms | 1,149.269 ms | −189.614 ms (−14.16%) |
| All GPU streams idle | 140.568 ms (10.50%) | 31.003 ms (2.70%) | −109.566 ms |
| Any GPU stream busy, interval union | 1,198.314 ms | 1,118.266 ms | −80.048 ms |
| Communication overlapping compute, interval union | 122.163 ms | 150.820 ms | +28.657 ms |
| Kernels linked to the three microbatches | 11,815 | 8,269 | −3,546 (−30.0%) |
| CUDA kernel-launch API time, host sum | 77.251 ms | 46.882 ms | −30.369 ms |
| DtoH copies during the three microbatches | 6 | 6 | 0 |

The three individual GPU spans were 460.816, 438.871, and 438.041 ms in eager
versus 386.069, 380.447, and 381.656 ms with compile. Time with all GPU streams
idle accounts for 57.8% of the integrated span difference. The host launch-time
sum overlaps GPU work and is separate from this interval accounting. There
are 45 CUDA host synchronization calls in each arm. Six DtoH transfers inside
the window are matched to the embedding lookup in both arms.

The console samples for updates 11–20 show maximum GPU memory of 73.43 GiB
in eager and 72.99 GiB with compile, including non-PyTorch usage. Subtracting
the logged non-PyTorch component gives estimated PyTorch reserved peaks of
68.03 and 67.58 GiB. Both arms ran serially in one allocation. The cadence
numbers are from the unprofiled warmup window and are one comparison at
this fixed workload. Loss and gradient norm differed in update 21
(14.70740 and 2.9722 in eager, 14.70742 and 2.9961 with compile). This
comparison provides timing evidence. Compiled SAC suppresses replay of some
forward side effects, including MoE bookkeeping where enabled, so equal
launch sizes alone cannot establish equal routed work.

## What compile improves

The largest direct change is in the operations around the native kernels.
With compile, the trace includes 1,923 Inductor kernels with 41.140 ms
of summed device duration. The reductions below cover launches with the
listed ATen CPU owner. Some of that work moved into Inductor kernels, so
these are changes in standalone ATen work and should not be added as
independent savings.

| ATen CPU launch owner | Eager device launches / sum | Compile device launches / sum |
| --- | ---: | ---: |
| `aten::copy_` | 1,330 / 33.275 ms | 244 / 10.751 ms |
| `aten::cat` | 108 / 24.303 ms | 21 / 9.884 ms |
| `aten::fill_` | 915 / 17.947 ms | 114 / 1.665 ms |
| `aten::add` | 570 / 12.141 ms | 30 / 0.463 ms |
| `aten::add_` | 345 / 36.991 ms | 75 / 26.331 ms |
| `aten::mul` | 792 / 12.801 ms | 102 / 2.553 ms |

The launch reduction appears both before and within autograd. Original
forward goes from 3,784 to 2,536 kernels (310.377 to 288.401 ms summed).
Backward and checkpoint replay together go from 7,974 to 5,676 kernels
(984.004 to 949.907 ms summed). The latter combines real backward, SAC
replay, and forward work inside backward. 57 kernels per arm have
insufficient scope metadata for either total. The two phase sums count
concurrent streams separately.

MLA provides a local example. Across nine original forward sites, the
stream-7 interval from the end of the final KV projection to the start of
cuDNN attention fell from 2.352 to 1.398 ms with compile. The eager ATen
concatenation kernels total 1.902 ms, compared with 1.317 ms for the compiled
packing kernels. The intervals have matching endpoint symbols and launch
geometry. They exclude the earlier Q concatenation and SAC replay, where
the launch order can differ. One matched site's projection and cuDNN
External IDs are 3268 and 3303 in eager, and 1635 and 1639 with compile.

Matching the full CUDA symbol, grid, block, and count within each microbatch
shows which major native kernels keep the same implementation.

| Matched kernel family | Launches per arm | Eager sum | Compile sum |
| --- | ---: | ---: | ---: |
| CuTe gated norm, forward and backward | 1,281 | 49.144 ms | 51.112 ms |
| cuDNN MLA flash attention and its two backward helpers | 45 | 31.456 ms | 32.321 ms |
| FlashQLA TileLang gated-delta scans | 72 | 20.800 ms | 22.317 ms |
| Stable cross-entropy forward and backward | 24 | 14.032 ms | 14.011 ms |
| MoE capacity selector and dispatch routing metadata | 81 | 0.331 ms | 0.329 ms |
| DeepGEMM routed 1d1d | 42 | 55.950 ms | 55.973 ms |
| DeepGEMM routed 1d2d, matched signatures | 66 | 67.825 ms | 67.874 ms |

HybridEP combine shows 60 launches in each arm and 110.439 → 91.589 ms with
compile. This duration includes effects from inputs, peer arrival and
overlap. It is a schedule observation rather than an instruction-level
speedup of the transport kernel. The longer eager GPU span and smaller
communication/compute overlap are consistent with the launch gaps in
the table above.

## Initial custom-kernel targets before the five fusions

The following budgets are measured in the compile trace across all three
microbatches. Each budget is the sum of the named device kernels. A replacement
still has to perform their computation, and the step-time effect depends on
stream dependencies, resource contention, and any layout work it removes.

| Priority | Boundary in the compiled path | Device budget | Proposed component check |
| --- | --- | ---: | --- |
| 1 | MLA rotary application to Q and K, followed by cuDNN Q/K packing, at 18 original or replayed sites | 4.729 ms (36 packing launches, 36 casts or rotary preparation launches, 36 complex multiplies, 18 table gathers) | Apply RoPE in FP32 and write the final BF16 Q/K layouts directly. Preserve the shared K rotary dimension across query heads, its gradient accumulation, the cuDNN scale for head dimension 192, the V path, and the existing strides and SAC behavior. Compare combined output and backward with the compiled path. |
| 2 | CuTe gated norm output followed by a standalone large Inductor residual add, at 36 original or replayed sites | 2.441 ms for the residual adds | Extend the CuTe output epilogue to add the residual while reproducing the current BF16 rounding boundary. Measure the output and backward independently, including gradient accumulation, gating statistics and SAC. |
| 3 | MLA output layout and sigmoid gate followed by FP8 projection-input quantization, at the same 18 sites as priority 1 | 1.930 ms (1.192 ms gate/layout and 0.738 ms quantize) | Feed the existing rowwise E4M3 quantizer directly from the gate/layout computation, preserving FP32 sigmoid and multiplication, BF16 rounding, FP32 scales, and values required by autograd. Measure bytes written and GPU duration. |

At one MLA site, CUDA External IDs 1636 and 1637 are the two packing launches
and 1639 is the cuDNN attention launch. External IDs 1645 and 1646 are separate
layout/gate and FP8 quantization launches. The gate launch's name contains
`quantize` even though the trace shows the actual quantizer as a separate
kernel. The median gate-to-quantize gap is 1.760 µs. The main opportunity there
is the intermediate BF16 write and reread. At one norm site, External ID 735
is the CuTe output kernel and 757 is the residual add.

Implementation entry points are
[MLA](../../gift/torchtitan/models/common/attention/softmax/mla.py),
[rotary embedding](../../gift/torchtitan/models/common/embeddings/rope/rope.py),
[FP8 projection](../../gift/torchtitan/ops/float8/cuda_gemm.py), and
[decoder residuals](../../gift/torchtitan/models/common/decoder_blocks/transformer_block.py).
NVIDIA's [Transformer Engine fused RoPE API](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.19/api/c/fused_rope.html)
and [cuDNN RoPE operation](https://docs.nvidia.com/cudnn/latest/operations/RoPE/)
are implementation references. A Hopper component test should first establish
the value of writing the final MLA layout and the cost of its backward. Use the
current compile arm as the comparison point for all three ideas.

The three measured compiled-path budgets sum to 9.100 ms, or 0.79% of the
sampled GPU span. That sum describes time spent in the kernels selected for
investigation. Actual savings depend on the replacement's cost, memory
traffic, and overlap. The compile-on versus eager difference does not provide
another 9% estimate for introducing custom kernels on top of compile.

## Reproducing the trace analysis

[Machine-readable results](results.json) retain trace SHA-256 hashes, source
and resolved-config identities, per-rank cadence and step completion, and
the device accounting. Run from the task root with both local traces present.

```sh
python3 runs/compile-baseline/analyze.py \
  artifacts/baseline_compile.json artifacts/baseline_no_compile.json \
  --output artifacts/compile-analysis.json
```

The analyzer uses the CUDA launch and Kineto flow links to assign GPU work
to the three CPU microbatch annotations, then extends through the last
linked GPU activity. It reconciles all 9,275 captured device activities with
compile and all 13,079 in eager. GPU idle uses the union across all CUDA
streams. Device kernel sums are useful for comparing specific work and can
exceed elapsed time.
