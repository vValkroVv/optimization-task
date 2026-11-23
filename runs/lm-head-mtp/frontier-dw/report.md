# Tuned cuBLASLt for isolated LM-head dW

The tuned FP32 writer lowered compiled replay wall time by **3.21% at 8**
and **3.72% at 16** fixed loss chunks. At 2 and 4 chunks, the difference
fell within the spread between repeated cuBLAS controls. Total peak GPU memory
matched within each of the four comparisons. The full replay passed its
post-timing precision checks at every chunk count.

| Loss chunks | GEMM K | GemmEx → Lt dW event span, ms | GemmEx → Lt replay wall, ms | Wall change | GemmEx repeat spread, ms | Total peak per arm, GiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 8192 | 24.941 → 23.713 | 76.780 → 76.399 | −0.50% | 0.417 | 14.194214 |
| 4 | 4096 | 26.215 → 24.434 | 78.115 → 77.981 | −0.17% | 0.323 | 12.237183 |
| 8 | 2048 | 29.027 → 23.979 | 81.848 → 79.216 | −3.21% | 0.431 | 11.260620 |
| 16 | 1024 | 38.762 → 33.231 | 93.206 → 89.742 | −3.72% | 0.127 | 10.280151 |

[Verified summary](verified-summary.json) ·
[Native replay at 8 and 16](replay-g4-001) ·
[Native replay at 2 and 4](replay-g5-001)

## Writer and precision

GemmEx and Lt consume the same contiguous BF16 dLogits and hidden tensors.
They use FP32 accumulation and write the same native GFSDP FP32 slot with
beta zero on the first chunk and beta one on every later chunk. The output
matrix is 128256 by 4096 (1.957 GiB). Each invocation processes 16,384 total
folded rows across two heads. The replay retains the production tokenizer
capture, packed labels, loss weights, denominator, Inductor CE, dHidden,
fused hidden z-loss and singleton GFSDP SUM. Only the isolated writer
constructor is replaced. [Native replay source](../replay.py) ·
[Benchmark wrapper](../frontier_dw_replay.py)

Both row-major and equivalent column-major descriptors were screened at
0 and 32 MiB caps, requesting up to 16 heuristics each. Every returned
algorithm used 4 bytes of scratch. Zero-cap queries returned no algorithms on
this runtime. The selected plans reuse one 4-byte allocation through PyTorch.
They have split-K one and no split-K reduction. Each uses algorithm 66.
The 2-chunk plan selects row layouts for beta zero and one, using tile IDs
24 and 495 respectively. At 4 chunks the row plan uses tile 29 for both.
At 8 and 16 chunks the column plan uses tile 29 for both. The serialized
algorithm applies to the verified cuBLASLt 13.1.1 runtime version.
[NVIDIA cuBLASLt documentation](https://docs.nvidia.com/cuda/archive/13.1.1/cublas/index.html#cublasltmatmul) ·
[Writer source](../frontier_dw_lib.py) · [Tuner](../frontier_dw_tune.py)

| Loss chunks | Native FP64 dW relative L2, both arms | Native FP64 max absolute error, both arms |
| ---: | ---: | ---: |
| 2 | 1.2949190011e−5 | 3.9889389265e−9 |
| 4 | 6.9668919114e−6 | 2.0098784555e−9 |
| 8 | 3.8229412289e−6 | 9.7042595292e−10 |
| 16 | 2.1640022342e−6 | 4.9021275038e−10 |

The oracle computes FP64 GEMMs on the actual BF16 writer callback inputs in
their native sequence, outside the timed window, and compares the full reset
GFSDP FP32 gradient. Both arms produced finite gradients and identical
error metrics. SHA-256 of loss, dHidden, z-loss and z-loss gradient matched
between arms. The exact-operand isolated tuner returned the same FP64 error
metrics. Each native invocation activated one folded loss and one direct writer
call per chunk and every run had zero allocator retries and OOMs.
[Checks](../frontier_dw_summary.py) · [Verified summary](verified-summary.json)

## Timing and mechanism

The tuner used real replay operands with preallocated FP32 output. Each
comparison alternated six seeded random arm orders, including a second
identical GemmEx control. Unprofiled CUDA events spanned the full eager
dW chain, including any device gaps while Python submitted work. Host
submission was 0.028–0.213 ms across the final arms. Synchronized wall
time was within 0.056 ms of event spans. Those are overlapping scopes.
[Unprofiled timing and candidates](tune-16.json)

The native replay used two fresh processes per arm on each fixed GPU
in ABBA or BAAB order, with ten warmup and ten measured invocations.
The time shown is the median of the two process medians, including
CPU work and device synchronization. Total peak is the maximum native
allocator-plus-external estimate across invocations and replicates.
The native replay does not independently attribute time to host work,
copies or collectives. The pure dW chains are single stream and have no
memcpy or collective in the captured windows. The matched replay used
GPU 5 for 2/4 chunks and GPU 4 for 8/16 chunks.

A separate Kineto trace on GPU 4 showed one GEMM and one tiny device memset
per chunk in both arms. At 16 chunks, Lt changed the observed kernel from
the 320×128, 3-stage NVJET family to the 128×160, 5-stage family. Summed
device GEMM time moved 38.429→33.071 ms, with 0.015→0.014 ms of memsets and
0.071→0.070 ms of inter-event device gaps. At 8 chunks the same kernel
family change appeared in the trace, with summed device GEMM time
26.095→24.996 ms. Its GemmEx trace span was about 10% below the unprofiled
measurement, so use the repeated native timing for effect size. The 2- and
4-chunk traces ran on a different H100 from their timing comparisons and
showed kernel families only. The 2-chunk trace reversed the unprofiled
dW ordering. [Trace scopes and kernel names](profile-001/trace-summary.json)

## Reproduction

Run from `/workspace-SR008.nfs2/vkropotin/optimization-task/.worktrees/lm-head-mtp-fix`
on `sr008-gpu`, source `/etc/profile.d/wm.sh`, and use the existing
`/opt/torchtitan-venv/bin/python3.12`. Source revision
`2c67c182e8de6145f3796d62203714ab1b815446`, H100 80 GB HBM3, driver
570.133.20, PyTorch `2.15.0.dev20260822+cu130`, cuBLASLt 13.1.1. The fixed
fixture is the same capture documented in [the parent replay report](../report.md).

For each tuner arm set `CUDA_VISIBLE_DEVICES` to its verified idle GPU UUID,
`CUDACHERRY_DISABLE_JIT=1` and `OMP_NUM_THREADS=2`. Replay lane commands run
the same numerical checks, use the preserved tuning plans and create a distinct
per-lane compiler cache.

```bash
run=runs/lm-head-mtp
out=$run/frontier-dw
python=/opt/torchtitan-venv/bin/python3.12

"$python" -Bu "$run/frontier_dw_tune.py" 16 "$out/tune-16-repeat.json"

bash "$run/frontier_dw_lane.sh" 4 replay-g4-repeat \
    "16:5.5:cublas-num lt-num lt cublas" "8:5.5:lt-num cublas-num cublas lt"
bash "$run/frontier_dw_lane.sh" 5 replay-g5-repeat \
    "2:9.14:cublas-num lt-num lt cublas" "4:13.13:lt-num cublas-num cublas lt"

"$python" -Bu "$run/frontier_dw_profile.py" "$out/tune-16.json" 5.5 "$out/trace-16-repeat"
python3 -B "$run/frontier_dw_summary.py" "$out"
```

These checks cover isolated compiled head and loss execution. Production
distributed throughput, overlap, GPU memory peaks and optimizer behavior
remain unverified.
