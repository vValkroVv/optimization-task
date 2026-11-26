# Projection, dHidden and consuming CE for the B1 length grid

[Linear catalog](linear-plans.final-grid.json) · [All 21 cells and receipt paths](coverage.json) · [Fixture and binary hashes](provenance.sha256)

This campaign covers local batch one with main plus MTP1, vocabulary 128,256, and 100B C2, 100B C8 and 700B C8 at sequence lengths 4,096 through 262,144. The projection and dHidden calibration keys use the actual physical folded rows, matrix dimensions, strides, layout, alignment, capability and cuBLASLt version. The folded row count is twice the sequence length divided by the loss-chunk count. The same key can serve multiple cells when those descriptors match. The different whole-sequence loss denominators for those cells are visible in the fixture metadata. [Linear dispatch](../../../tuning/linear_policy.py) · [NVIDIA cuBLASLt 13.1.1](https://docs.nvidia.com/cuda/archive/13.1.1/cublas/index.html#using-the-cublaslt-api)

The replay reads fixture row zero, repeats the 4,096-position tile and preserves packed positions and ignored tile ends. The BF16 activations and weight are seeded. The first chunk has the actual folded projection shape and consuming CE generates the BF16 dLogits for dHidden. Historical receipts cover previously measured matching descriptors, including the 100B 2,048-row short-replay geometry and the long C8 measurements. Reused receipts establish the component measurement at their recorded input and descriptor. [Prior short linear calibration](../../../tuning/proj/README.md) · [Long C8 linear calibration](../../region-tuning/linear/report.md)

## Measurement and acceptance

The control uses allocating ATen linear or matrix multiply. The candidate calls the deployed cached cuBLASLt bridge using the original owning output shape, BF16 operands and BF16 result, with FP32 computation. The bounded screen checks up to 40 algorithms in each cuBLASLt layout, then retimes up to 12 and keeps at most three distinct bitwise-matching finalists. Allocating confirmation loads the exact key, checks the full BF16 output again, and requires no descriptor fallback. The large projection compares every output element with a CPU-held control in bounded tiles. CPU transfers, plan enumeration, first-use policy preparation and parity are outside confirmation time and peak memory accounting.

Eager samples time one allocating call on one stream through synchronization of its trailing event. The trials rotate two identical controls and the candidate arms for 15 rounds after five warmups per arm. A plan must beat twice the median absolute A/A difference in paired wall time, win at least 12 of 15 pairs, improve both halves, show no allocator retries and preserve GIFT total within observed A/A memory variation. The peak includes the live operands and maximum Torch reserve. Three cache-cleared rounds check isolated peak memory. Host submission, final wait, eager event spans and isolated peak memory are in the receipts. Graph spans use resident inputs and one captured arm at a time after eager measurement. Two short 700B plans with an initial graph disagreement repeated the exact candidate in a fresh 24-round A/A/B comparison with rotated graph replay and cleared the repeated wall criterion and improved the graph median. Individual kernel, idle, copy and communication attribution were unmeasured. These are isolated component replays. [Harness](measure.py) · [Fresh confirmation](recheck.py)

The measurements ran on the replacement SR008 H100 80 GB HBM3 node, job `60634e19-cb86-4e8b-a27c-60078635cd4b`, using GPUs 4–7 after the idle checks. The wheel was Torch `2.15.0.dev20260822+cu130` at revision `8b353065ce260336c5fdfa74d754e8bd9b91b062` with cuBLASLt 13.1.1. The artifact stores raw eager and graph samples, GPU visibility, exact descriptors and source hashes. [Linear launcher](run.sh) · [Catalog generator](summarize.py)

## Projection and dHidden

The catalog retains the eight earlier choices and adds seven projection plans. All seven newly screened dHidden descriptors keep ATen. The catalog still includes the earlier B3 700B dHidden entry at 3,072 rows, which is outside this B1 grid. For all 21 cells, the coverage record lists the selected route, complete descriptor and supporting receipt.

| Model | Folded rows | Region | ATen / finalist wall ms | Paired Δ / A/A ms | Wins | ATen / finalist graph ms | GIFT GiB | Selected |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 100B | 1,024 | dHidden | 1.423 / 1.438 | +0.014 / 0.003 | 0/15 | 1.251 / 1.275 | 1.962 | ATen |
| 100B | 1,024 | Projection | 1.621 / 1.540 | -0.081 / 0.004 | 14/15 | 1.368 / 1.335 | 1.942 | Lt 66/197 |
| 100B | 4,096 | dHidden | 5.636 / 5.648 | +0.012 / 0.003 | 4/15 | 5.567 / 6.779 | 2.724 | ATen |
| 100B | 4,096 | Projection | 6.071 / 5.688 | -0.293 / 0.004 | 15/15 | 6.062 / 5.771 | 2.704 | Lt 66/201 |
| 100B | 131,072 | dHidden | 192.963 / 189.261 | -5.914 / 3.856 | 13/15 | 196.451 / 190.253 | 35.009 | ATen |
| 100B | 131,072 | Projection | 194.645 / 190.122 | -4.611 / 2.070 | 14/15 | 195.688 / 191.527 | 34.013 | Lt 66/201 |
| 100B | 262,144 | dHidden | 389.702 / 382.395 | -8.517 / 6.322 | 11/15 | 393.900 / 384.523 | 68.307 | ATen |
| 100B | 262,144 | Projection | 388.127 / 380.004 | -7.392 / 3.433 | 14/15 | 390.023 / 382.289 | 66.315 | Lt 66/201 |
| 700B | 1,024 | dHidden | 2.483 / 2.486 | +0.010 / 0.008 | 2/15 | 2.542 / 2.559 | 2.704 | ATen |
| 700B | 1,024 | Projection | 2.850 / 2.705 | -0.145 / 0.003 | 14/15 | 2.381 / 2.350 | 2.684 | Lt 66/184 |
| 700B | 2,048 | dHidden | 5.349 / 5.250 | -0.110 / 0.090 | 14/15 | 5.740 / 6.223 | 2.977 | ATen |
| 700B | 2,048 | Projection | 5.132 / 4.984 | -0.147 / 0.004 | 22/24 | 5.327 / 5.155 | 2.936 | Lt 66/201 |
| 700B | 4,096 | dHidden | 10.085 / 9.952 | +0.008 / 0.024 | 7/15 | 9.960 / 10.816 | 3.505 | ATen |
| 700B | 4,096 | Projection | 10.328 / 9.954 | -0.361 / 0.013 | 24/24 | 10.398 / 9.915 | 3.464 | Lt 66/201 |

When ATen is selected, the table shows the fastest finalist by paired wall delta. The two 700B rechecks supply fresh eager and graph results. The two 100B C2 projections at 131,072 and 262,144 rows selected bitwise-identical cuBLASLt plans. The largest output contained 33,621,540,864 BF16 values in a 62.625 GiB allocation and used 66.315 GiB total during isolated projection. Its warmed wall median fell by 2.09%, computed against the ATen median. The result covers allocating one component projection on an H100.

The measured linear harness and the imported replay source are preserved byte for byte. The maintained harness uses assignment to release a captured graph, which clears a static undefined-name finding on the earlier deletion. A local isolated check verified identical sample and error results and graph release for both versions. The recorded replay and its local counterpart agree on the hidden resize function used by these runs. [Measured harness](runtime-sources/measure.py) · [Imported replay](runtime-sources/replay.py)

## Consuming CE

Keep the general tuned binary `d3343cd518a05a6d5bea58b0493fa33373e040f1ce3dd5864b14c03033528929`. The added comparisons rotate tuned, identical tuned, the preceding CUDA 13 artifact and the unrolled alternative. All four arms produced the same FP32 raw row loss bits, both FP32 head sums, weighted scalar and in-place BF16 dLogits over every checked element. The tuned graph span was lower than the preceding artifact at all eight new descriptors. At 1,024 rows the eager paired difference was within A/A variation for both models. Unrolling had no consistent paired wall advantage beyond A/A noise. Each case had the same GIFT total peak across all four arms. The largest case restored a single logits allocation with an out GEMM before the timing marker. The other cases retained a pristine logits copy. The 131,072- and 262,144-row memory figures include those different restore live sets.

| Preset | Sequence | Folded rows | Tuned wall ms | Previous wall ms | Tuned graph ms | Previous graph ms | GIFT GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100B C8 | 4,096 | 1,024 | 0.405 | 0.409 | 0.227 | 0.235 | 1.220 |
| 100B C8 | 8,192 | 2,048 | 0.574 | 0.590 | 0.417 | 0.431 | 1.708 |
| 100B C8 | 16,384 | 4,096 | 0.956 | 0.984 | 0.796 | 0.819 | 2.684 |
| 700B C8 | 4,096 | 1,024 | 0.425 | 0.421 | 0.227 | 0.235 | 1.220 |
| 700B C8 | 8,192 | 2,048 | 0.578 | 0.591 | 0.416 | 0.430 | 1.688 |
| 700B C8 | 16,384 | 4,096 | 0.959 | 0.982 | 0.796 | 0.821 | 2.684 |
| 100B C2 | 131,072 | 131,072 | 24.704 | 25.434 | 24.492 | 25.220 | 63.349 |
| 100B C2 | 262,144 | 262,144 | 53.992 | 57.718 | 53.339 | 56.864 | 66.317 |

Wall includes separate FP32 head sums, weighted normalization, autograd and final event synchronization. The CUDA Graph measures the same CE path and excludes Python submission and logits restoration. Host submission and wait are in the raw samples. [New CE receipts](ce-results) · [CE harness](measure_ce.py) · [CE launcher](run_ce.sh) · [Earlier long C8 comparison](../../region-tuning/ce/README.md)
