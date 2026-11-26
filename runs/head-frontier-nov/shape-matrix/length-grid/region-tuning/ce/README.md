# Consuming CE at long C8 lengths

Keep the current tuned CUDA CE binary for this grid. Across the eight cases, it reduced synchronized CE wall time by 2.43–2.90% and the graph device span by 2.68–2.94% from the previous generalized cuda13 binary. These changes divide the difference of medians by the previous binary's median. Peak GIFT total was identical among all arms in each case.

## Median synchronized wall time

Each row measures the final chunk at local batch one and vocabulary 128,256. All time columns are milliseconds. The memory column is the common eager GIFT total peak in GiB.

| Preset | Sequence | Folded rows | Current tuned | Previous cuda13 | 512 threads | Unrolled | GIFT total |
|---|---:|---:|---:|---:|---:|---:|---:|
| 100B | 32,768 | 8,192 | 1.73211 | 1.77986 | 1.90348 | 1.72931 | 4.65710 |
| 100B | 65,536 | 16,384 | 3.27077 | 3.35346 | 3.60964 | 3.26691 | 8.54382 |
| 100B | 131,072 | 32,768 | 6.33898 | 6.51440 | 7.02508 | 6.34046 | 16.37585 |
| 100B | 262,144 | 65,536 | 12.46861 | 12.82857 | 13.85746 | 12.46835 | 32.03992 |
| 700B | 32,768 | 8,192 | 1.73539 | 1.78171 | 1.90420 | 1.73559 | 4.63757 |
| 700B | 65,536 | 16,384 | 3.27201 | 3.35366 | 3.60819 | 3.26421 | 8.58289 |
| 700B | 131,072 | 32,768 | 6.35011 | 6.53594 | 7.06363 | 6.35686 | 16.41492 |
| 700B | 262,144 | 65,536 | 12.47514 | 12.84771 | 13.89263 | 12.47320 | 32.05945 |

Matched cycles rotated the tuned binary, an identical tuned control and three alternative binaries on the same card. Each arm used ten eager warmups and 24 measured eager calls. The median cycle-paired wall changes from current tuned ranged from −0.253% to 0.000% for the identical control, +2.505% to +2.994% for previous cuda13, +9.754% to +11.352% for the 512-thread variant and −0.112% to +0.111% for unrolling. Unrolling gave no consistent improvement across the grid.

## Timing and memory scope

The timer covers consuming CE on a preprojected BF16 chunk, separate FP32 head sums, weighted normalization and the autograd call that returns an in-place BF16 dZ. The projection, restoration copy, loss-metric bookkeeping and all collectives are outside the timed region. Eager submission includes Python, dispatch and kernel launches. Final wait starts after the stop event is submitted. CUDA Graph events cover the same single-stream operation without Python submission. The graph restores the input before its start event and discards its first replay. Individual kernel time, GPU idle time, and physical memory traffic were unmeasured.

Each cell below gives current tuned / previous cuda13 in milliseconds. The receipts contain the same scopes for both optional variants and the identical control.

| Preset | Sequence | Host submission | Final host wait | Eager CUDA events | Graph device span |
|---|---:|---:|---:|---:|---:|
| 100B | 32,768 | 0.43679 / 0.43952 | 1.29578 / 1.34003 | 1.71626 / 1.76366 | 1.56035 / 1.60570 |
| 100B | 65,536 | 0.44972 / 0.44808 | 2.82534 / 2.90875 | 3.25483 / 3.33776 | 3.08816 / 3.17306 |
| 100B | 131,072 | 0.46905 / 0.46161 | 5.87051 / 6.04946 | 6.31986 / 6.49613 | 6.14632 / 6.31971 |
| 100B | 262,144 | 0.49891 / 0.48933 | 11.97728 / 12.33681 | 12.44925 / 12.81003 | 12.25262 / 12.61010 |
| 700B | 32,768 | 0.43972 / 0.44080 | 1.29436 / 1.34086 | 1.71941 / 1.76536 | 1.55997 / 1.60336 |
| 700B | 65,536 | 0.45287 / 0.44879 | 2.81914 / 2.90550 | 3.25491 / 3.33654 | 3.08606 / 3.17125 |
| 700B | 131,072 | 0.47656 / 0.48068 | 5.87264 / 6.04912 | 6.33139 / 6.51754 | 6.15034 / 6.33101 |
| 700B | 262,144 | 0.48840 / 0.49141 | 11.99204 / 12.36092 | 12.45616 / 12.82789 | 12.26757 / 12.63896 |

GIFT total combines the PyTorch reserved peak and non-PyTorch device usage at readout. It includes the retained pristine BF16 logits and the working buffer for a single chunk. Projection storage, graph pools and the parity reference are outside the eager measurement. All measured samples had zero allocator retries. These values describe the isolated region with its restore buffer. [Full-head length sweep](../../report.md)

## Inputs and verification

Both models used the exact first 4,096-position `data-002` row repeated by the B1 length-grid replay, with its packed positions and ignored tile ends. BF16 hidden states and weights come from the same seeded synthetic fixture used in the head sweep. Head weights were `(1.0, 0.3)` for 100B and `(1.0, 0.1)` for 700B. The fixture-wide main and MTP1 valid counts and the denominators matched the existing rank-0 head receipts at every length. The denominators were 8,386,560, 16,773,120, 33,546,240 and 67,092,480, in ascending sequence order.

At 100B, sequence 32,768, the staged head projection activated the deployed cuBLASLt algorithm 66, tile 197, with zero fallbacks. It matched all 1,050,673,152 BF16 logits from `F.linear` on the final chunk. The sampled logit hash and label hash matched the original CE receipt, with the same head weights and denominator. That establishes input parity for the existing CE timing at this calibrated shape. [Projection parity receipt](100b-32768-projection-parity.json) · [Check](check_projection.py) · [Log](100b-32768-projection-parity.log)

Every arm activated the compiled consuming wrapper, checked its actual loss scales and returned dZ in the working BF16 logits storage. Against current tuned, every raw FP32 row loss, both FP32 head sums, the weighted loss and every BF16 dZ entry matched bitwise in all eight cases. The largest case checked 65,536 row losses and 8,405,385,216 dZ entries per arm. The untimed check holds one gradient reference and compares tiles of 128 rows. [Harness](bench_long.py) · [Original CE tuning and binary designs](../../../tuning/ce/README.md)

The SR008 job was `60634e19-cb86-4e8b-a27c-60078635cd4b`. The 100B and 700B cases used H100 80 GB HBM3 GPUs 6 and 7, respectively, with driver `570.133.20`. The runner verifies fixture hashes and checks device occupancy. Each receipt contains the resolved GPU UUID, PyTorch and CUDA versions, source and DSO hashes, timing samples and numerical checks. [Runner](run.sh) · [100B first-case log](100b-first.log) · [100B remaining cases](100b-rest.log) · [700B log](700b.log)

- 100B receipts by sequence. [32,768](100b-32768.json) · [65,536](100b-65536.json) · [131,072](100b-131072.json) · [262,144](100b-262144.json)
- 700B receipts by sequence. [32,768](700b-32768.json) · [65,536](700b-65536.json) · [131,072](700b-131072.json) · [262,144](700b-262144.json)
