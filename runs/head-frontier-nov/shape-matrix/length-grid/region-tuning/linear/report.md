# Isolated projection and dHidden on the C8 length grid

The recommended catalog adds five plans to the three existing choices. Seven plans cleared the original warmed-eager screen and confirmation. Fresh A/A/B checks reproduced one of the three borderline results, and the other two remain provisional.

[Recommended catalog](linear-plans.recommended.json) · [Original selection](selected.json) · [Fresh A/A/B results](sensitivity/summary.json)

## Measurement

The 16 cells cover projection H × Wᵀ and allocating two-dimensional dHidden
dZ × W. Each uses B1, main plus MTP1, vocabulary 128,256, and widths 4,096
and 7,168. A chunk contains S/8 physical tokens and S/4 folded rows. The
largest projection output has 8,405,385,216 BF16 values (15.656 GiB).

The original length-grid route is the control. At 100B/32K, projection uses
the calibrated catalog and its cached policy. The other projection controls
call ATen linear, and every dHidden control calls ATen mm. The candidate
uses the same cached policy, cuBLASLt bridge and borrowed ATen workspace
as the current calibrated route. It uses FP32 compute and BF16 inputs and
outputs. The output owner has the original folded shape for projection
and the original contiguous matrix shape for dHidden.

Fixture row 0 is repeated from the 4,096-token real data-002 target/position
row and seeded BF16 tensors. The harness shifts MTP labels over the full
repeated sequence with GIFT and derives the original denominator from the
fixture multiplier. Only chunk 0 activations and logits are materialized.
dHidden receives the BF16 gradient from the length-grid's tuned compiled
CUDA CE, reusing the logits storage. Every candidate that reaches warmed
comparison matched all BF16 bits of the real representative output, both
through the enumerated plan and after loading its exact cached descriptor.
The harness checks at most 128 rows at a time during equality.

The first screen launches up to 40 checked algorithms per layout, selecting
both cuBLASLt heuristic entries and distinct algorithm/tile families found
by exhaustive enumeration. The 12 fastest successful screen entries receive
three retiming calls and a bitwise comparison. Up to three distinct
matching configurations receive allocating eager confirmation. The
screen is bounded to the measured choices at these exact descriptors.

Confirmation uses five warmups per arm and 15 measured rounds, rotating two
identical control slots and the candidate slots. The table reports the
median synchronized wall time, median paired candidate minus the midpoint
of both controls, and the median absolute A/A difference. A candidate must
beat twice the A/A difference, win at least 12 of 15 pairs, and show a
negative median in each half. GIFT total is the maximum reserved peak plus
current non-Torch memory. Confirmation checks that the candidate peak has
no increase beyond the observed A/A difference for both the warmed series
and three isolated rounds that release allocator cache before each arm.
Allocator retries disqualify a candidate.

The synchronized eager call includes allocating one output, submitting
the operation and waiting for a trailing CUDA event. Host submit time and
host wait time are recorded separately in the receipts. Graph medians
use five rounds of event spans with three replays each and keep graph
storage out of the eager memory accounting. They are diagnostics for the
resident inputs and captures. Communication, full-head timing, and attribution of copies and device idle require separate measurement.

[Length-grid routes](../../report.md) · [Cached policy](../../../tuning/linear_policy.py) ·
[Prior bridge and search](../../../tuning/proj/README.md) ·
[cuBLASLt 13.1.1](https://docs.nvidia.com/cuda/archive/13.1.1/cublas/index.html#using-the-cublaslt-api) ·
[Pinned ATen GEMM](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/aten/src/ATen/cuda/CUDABlas.cpp)

## Original matrix

Each row shows the selected candidate or the bitwise-matching candidate with the best paired wall delta. The decision column incorporates the fresh confirmations. Wall, paired delta, A/A and graph values are in milliseconds per chunk. GIFT total is the warmed peak in GiB, which matched exactly between the two arms in every row. Host submit and wait, allocator diagnostics, raw rounds and exact descriptors are in the receipts.

| Model | Sequence | Region | Eager wall | Paired Δ / A/A | Pair wins | Graph | GIFT GiB | Decision |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 100B | 32,768 | Projection | 11.633 → 11.623 | -0.011 / 0.050 | 10/15 | 11.674 → 11.532 | 3.720 | Keep control |
| 100B | 32,768 | dHidden | 11.392 → 11.569 | +0.038 / 0.015 | 2/15 | 11.696 → 11.873 | 3.778 | Keep control |
| 100B | 65,536 | Projection | 23.940 → 23.427 | -0.998 / 0.197 | 13/15 | 24.430 → 22.977 | 5.731 | Recommend |
| 100B | 65,536 | dHidden | 24.019 → 22.842 | -1.186 / 0.281 | 14/15 | 23.986 → 26.777 | 5.849 | Provisional |
| 100B | 131,072 | Projection | 48.025 → 46.664 | -1.749 / 0.764 | 13/15 | 48.088 → 47.347 | 9.774 | Recommend |
| 100B | 131,072 | dHidden | 49.398 → 48.516 | -0.353 / 0.537 | 11/15 | 49.971 → 47.855 | 10.028 | Keep control |
| 100B | 262,144 | Projection | 96.993 → 95.483 | -1.304 / 1.582 | 13/15 | 96.000 → 98.200 | 17.841 | Keep control |
| 100B | 262,144 | dHidden | 99.162 → 96.330 | -2.607 / 2.990 | 12/15 | 100.674 → 95.522 | 18.349 | Keep control |
| 700B | 32,768 | Projection | 21.418 → 21.360 | -0.541 / 0.153 | 13/15 | 20.841 → 20.130 | 4.501 | Recommend |
| 700B | 32,768 | dHidden | 20.762 → 20.485 | -0.138 / 0.197 | 9/15 | 20.963 → 20.364 | 4.599 | Keep control |
| 700B | 65,536 | Projection | 43.593 → 41.957 | -1.284 / 0.296 | 13/15 | 43.165 → 42.469 | 6.552 | Provisional |
| 700B | 65,536 | dHidden | 43.066 → 42.238 | -1.127 / 0.867 | 11/15 | 43.991 → 41.878 | 6.786 | Keep control |
| 700B | 131,072 | Projection | 85.239 → 84.681 | +0.068 / 1.618 | 7/15 | 84.570 → 87.574 | 10.692 | Keep control |
| 700B | 131,072 | dHidden | 87.036 → 83.348 | -6.986 / 2.707 | 14/15 | 90.303 → 84.852 | 11.122 | Recommend |
| 700B | 262,144 | Projection | 172.128 → 170.814 | -1.207 / 4.089 | 8/15 | 171.783 → 168.150 | 18.954 | Keep control |
| 700B | 262,144 | dHidden | 175.142 → 165.952 | -18.290 / 6.975 | 14/15 | 192.867 → 166.827 | 19.833 | Recommend |

Both 262,144-token projection runs exhausted device memory when capturing a fifth graph. The graph-only reruns reused one capture for both identical control slots to fit all four unique graphs. Those reruns supplied the graph values in the table. The original eager results and decisions stayed valid.

## Fresh A/A/B checks

The repeat used 24 alternating rounds on GPUs 6 and 7. It required 20 pair wins, a paired median improvement larger than twice A/A noise, a negative median in both halves, bitwise parity and the same GIFT memory criterion.

| Model | Sequence | Region | Eager wall, ms | Paired Δ / A/A, ms | Pair wins | GIFT GiB | Result |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 100B | 65,536 | dHidden | 23.091 → 22.240 | -0.998 / 0.228 | 19/24 | 5.849 | Provisional, 19 wins against the 20-pair threshold |
| 700B | 32,768 | Projection | 20.908 → 20.372 | -0.603 / 0.193 | 23/24 | 4.499 | Reproduced |
| 700B | 65,536 | Projection | 42.734 → 41.016 | -1.398 / 0.935 | 22/24 | 6.550 | Provisional, paired gain below twice A/A |

## Reproduction

Use the existing SR008 host with physical GPUs 3, 4 and 5. Run from its
`shape-matrix-1125/length-grid/region-tuning/linear` directory. The runner
sources `/etc/profile.d/wm.sh`, checks the designated GPU is idle, uses its
UUID to set CUDA visibility, and runs the pinned Torch interpreter. The node was SR008 job `60634e19-cb86-4e8b-a27c-60078635cd4b` with H100 80 GB HBM3 GPUs, driver 570.133.20, Torch `2.15.0.dev20260822+cu130` at `8b353065` and cuBLASLt 13.1.1.

```bash
bash run.sh 3 100b:32768:dhidden
bash launch.sh
# Wait for the three GPU lane logs to report completion.
bash run_graph.sh 3 700b-s262144-projection
bash run_graph.sh 4 100b-s262144-projection
bash run_sensitivity.sh 6 700b-s32768-projection 700b-s65536-projection
bash run_sensitivity.sh 7 100b-s65536-dhidden
python3 -B summarize.py --sensitivity
```

[Harness](measure.py) · [Runner](run.sh) · [Matrix launcher](launch.sh) ·
[Summarizer](summarize.py) · [Graph recovery](recover_graph.py) · [Sensitivity script](confirm_sensitivity.py) · [Source and fixture hashes](provenance.sha256)
