# Exact fused CE on the current native LM head

Both fused kernels reduced isolated head wall time against their matched
controls at C2 and C8. A direct interleaved comparison left methods 13
and 16 tied at the resolution of this replay. Peak GIFT memory and all
five reset-output digests agreed in all series.

## Matched controls

| Chunks and GPU | CUDA method | Control wall, ms | Fused wall, ms | Wall change | Adjacent pair changes, ms | GIFT peak in both arms, GiB |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| C2 / 0 | 13 | 75.803 | 74.787 | −1.34% | −0.526, −1.109, −0.325 | 11.6376 |
| C2 / 0 | 16 | 75.678 | 74.738 | −1.24% | −0.949, −0.931 | 11.6376 |
| C8 / 1 | 13 | 79.532 | 78.100 | −1.80% | −0.206, −1.528, −1.789 | 10.1337 |
| C8 / 1 | 16 | 78.823 | 78.482 | −0.43% | −0.551, −0.130 | 10.1337 |

Wall is the median of the fresh-process medians. Changes use each series'
control wall as denominator. Method 13 used three fresh processes per arm,
in ABBAAB order at C2 and BAABBA at C8. Method 16 used two per arm in ABBA
order for both shapes. Control process ranges were 0.549 and 0.981 ms for
method 13 and 0.362 and 0.225 ms for method 16, at C2 and C8 respectively.
Every adjacent fused process ran faster than its control. The C8 control
median shifted by 0.709 ms between series.

## Direct kernel comparison

| Chunks and GPU | Method 13 wall, ms | Method 16 wall, ms | Change vs. 13 | Adjacent pair changes, ms | GIFT peak in both arms, GiB |
| --- | ---: | ---: | ---: | --- | ---: |
| C2 / 0 | 74.744 | 74.657 | −0.12% | −0.042, −0.481, −0.237 | 11.6376 |
| C8 / 1 | 78.209 | 77.726 | −0.62% | −0.894, +1.438, −0.708 | 10.1337 |

The direct series interleaved three fresh processes per kernel in ABBAAB
order at C2 and BAABBA at C8, with method 13 as A. At C2 the three pair
differences favored method 16. The 0.088 ms difference between arm
medians was below the process range of either method, 0.301 and 0.500 ms.
At C8 one pair ran 1.438 ms slower with method 16, and its 1.982 ms process
range exceeded the 0.482 ms difference between arm medians. The direct
results provide no clear integrated advantage for either kernel.

## Execution and parity

The matched replays use the current `lm10-c1` working tree at
`e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f`. Source hashes pin the
actual files used in both arms. The rank-zero capture came from the
production-tokenizer `data-002` run with seed 2048. The CE fixture hash is
`15227882b1da44c2881b63622b6249bcddd307d90575216290f29ab9f720c727`.
It includes main and MTP hidden states and targets, the native denominator,
and the head weight, with B2/S4096, width 4096 and vocabulary 128256.
[Input capture provenance](../layout-order/report.md#writer-launch-order-and-numerical-verification)

Both arms use native BF16 projection, staged dW, direct BF16 dHidden,
early logits release, hidden z-loss and FP32 GFSDP dW accumulation.
The native writer used cuBLASLt plans 9.14 at C2 and 5.5 at C8.
Each process observed 24 native NCCL singleton all-gathers with separate
BF16 input and output storage. Each arm compiled one Inductor graph and
had no recompiles, allocation retries or OOMs in its ten timed calls.
CUDA Graphs were disabled.

The candidate binds the existing fused factory to the native consuming CE
route. Method 13 computes max on unpacked values. Method 16 computes the
local max on packed BF16 pairs before converting to FP32. Both retain the
native loss sums and BF16 gradient rounding in the tested path. The
generated Inductor wrapper passes the folded logits directly to the
custom op, with 8192 rows at C2 or 2048 at C8. The CUDA launcher ran
46 times per C2 process and 184 times per C8 process, including setup and
the reset comparison. The diagnostic backward's FP32 scales matched
the prepared pair bitwise, at approximately 4.7695363e−7 and 1.4308610e−7.
Every run matched all five complete reset-output SHA-256 digests against
its fresh controls and the previous isolated controls. The hashes cover
loss, dHidden, FP32 dW, z-loss and z-loss gradient.
[Generated wrapper](inductor-consume-boundary.txt) · [Summary and digests](summary.json)

## Scope and reproduction

Each process warms up for ten head calls and measures ten more. The timer
includes target packing, submission, native forward and backward, and
the final device synchronization. Memory is the maximum native GIFT
total estimate over the ten timed calls in every replicate. Host
submission and final host synchronization wall times are recorded
separately in the receipts and may overlap GPU activity. This comparison
supplies no separate warmed device-compute, device-idle, or copy timing.
The collective path is singleton, so wire latency and training overlap
remain open. Production use also requires validating the forward overwrite
around consumers such as the optional logits argmax observer and validating
the precomputed scales with the actual training gradient path.

Both series used H100 80 GB HBM3 GPUs on the existing SR008 allocation
`lm-mpi-job-84fd3579-7bd2-4969-ae47-b7af17b73ca2`. The assigned GPUs were
empty at lane start, restricted by UUID, and checked for active compute
before each process. PyTorch reported `2.15.0.dev20260822+cu130`, CUDA
13.0, NCCL 2.30.7 and driver 570.133.20. The compact raw receipts and
launch logs are under `raw/`. The remote run path is
`/workspace-SR008.nfs2/vkropotin/optimization-task/runs/head-frontier-nov/native-ce-finalist`.
The reproduction commands reuse its neighboring native and fused replay
helpers and the two pinned shared objects.

```sh
bash lane.sh 0 2 native control fused fused control control fused
bash lane.sh 1 8 native fused control control fused fused control
NATIVE_CE_METHOD=cuda-16 bash lane.sh 0 2 native-opt control fused fused control
NATIVE_CE_METHOD=cuda-16 bash lane.sh 1 8 native-opt control fused fused control
bash lane.sh 0 2 native-direct cuda-13 cuda-16 cuda-16 cuda-13 cuda-13 cuda-16
bash lane.sh 1 8 native-direct cuda-16 cuda-13 cuda-13 cuda-16 cuda-16 cuda-13
```

Run `python3 summarize.py raw summary.json` in the local directory to
validate the preserved receipts and regenerate the summary.
