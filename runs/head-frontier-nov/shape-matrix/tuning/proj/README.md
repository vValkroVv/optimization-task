# cuBLASLt projection and dHidden

The first screening round advanced 100B C2 projection tile 197 and 700B C8
dHidden tile 495 to the eight-rank head replay. The 700B C8 tile 31 candidate
writes directly into the final BF16 gradient slices in the component replay.

## Scope and layout

The comparison uses the original folded `linear` and allocating `torch.mm`
as controls. It reads the shape-matrix fixtures with real tokenizer labels and
seeded synthetic BF16 activations and weights. It regenerates BF16 dLogits with
the exact CUDA 13 CE binary used in the head control. Projection uses
`X @ W.T`; dHidden uses `dLogits @ W`. The dHidden-direct control also copies
the flat gradient into its final sequence slice. All candidates use BF16
inputs and output, FP32 compute and scaling, and the default epilogue.

| Geometry | Folded rows per chunk | Hidden | Vocabulary | TFLOP per GEMM | Temporary dHidden |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100B C2 | 8192 | 4096 | 128256 | 8.607 | 64 MiB |
| 100B C8 | 2048 | 4096 | 128256 | 2.152 | 16 MiB |
| 700B C8 | 3072 | 7168 | 128256 | 5.648 | 42 MiB |

The search checked leading dimensions, pointer alignment, row-major and
transpose-equivalent column-major descriptors, default and FP32-only
reductions, and algorithms bounded by the existing ATen workspace. Selected
kernels have split-K 1 and request four bytes from that workspace. The direct
path uses a strided batched GEMM with broadcast weight. Its batch stride equals
the spacing between sequence slices for adjacent batch rows in the full
gradient. There is no additional CUDA allocation for the descriptors or the
borrowed workspace. Unsupported or unseen descriptors use ATen.

The binding keys measured choices by cuBLASLt version, compute capability,
dimensions, leading dimensions, transpose flags, order, batch strides and
alignment. It prepares and caches descriptors and accepts Torch's current
stream. The binding is for an existing explicit backward or custom autograd
function, where the caller owns gradient semantics. The full-head hook calls
the 2D interface. Direct output has separate component coverage. Using it in
the full head requires giving the backward path an accumulator-owned view
and bypassing the accumulator copy. Returning the view as a leaf gradient
can cause autograd to clone it to satisfy its gradient layout contract.
[PyTorch gradient layouts](https://docs.pytorch.org/docs/stable/autograd.html#default-gradient-layouts) · [Vendor API](https://docs.nvidia.com/cuda/archive/13.1.1/cublas/index.html#using-the-cublaslt-api) · [ATen GEMM](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/aten/src/ATen/cuda/CUDABlas.cpp) · [Existing Lt search](../../../../fusion-followup/head_lt.cpp)

## Numerical parity

Each selected descriptor matched all BF16 output bits of the allocating
control on every chunk. This includes the underlying logits before CE.
Direct dHidden filled its assembled final tensor correctly across all chunk
offsets. Loading the choices in the confirmation required an exact descriptor
match. Projection and dHidden element counts are over each complete replayed
microbatch.

| Geometry | Projection elements | dHidden elements per route | Choice receipt |
| --- | ---: | ---: | --- |
| 100B C2 | 2,101,346,304 | 67,108,864 | [100B C2](100b-c2-confirm.choices.json) |
| 100B C8 | 2,101,346,304 | 67,108,864 | [100B C8](100b-c8-confirm.choices.json) |
| 700B C8 | 3,152,019,456 | 176,160,768 | [700B C8](700b-c8-confirm.choices.json) |

## Corrected regional timing and memory

All timing entries are milliseconds for one external chunk on one CUDA
stream. Each arm has 24 measurements, following five warmup calls per arm and
rotating the two identical controls and candidate. The change columns compare
the pooled control median with the candidate median. Paired Δ is the median
candidate minus the midpoint of both controls in the same round. A/A is the
median absolute difference between the controls. Wall covers allocation,
submission and synchronization. The host intervals split at submission of the
trailing event. Graph timings divide a five-replay CUDA-event span by five.
They cover the device operation with resident operands and preallocated
destinations. The direct dHidden control also runs a device copy from its
temporary output. The individual copy duration is unmeasured.

| Geometry and route | Tile | Wall | Paired wall Δ / A/A | CUDA Graph | Paired graph Δ / A/A | Host submit | Host wait |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100B C2 Projection | 197 | 11.931 → 11.603 | -0.314 / 0.057 | 12.495 → 11.873 | -0.603 / 0.344 | 0.101 → 0.254 | 11.826 → 11.345 |
| 100B C2 dHidden | 24 | 11.164 → 11.342 | +0.164 / 0.011 | 11.852 → 11.580 | -0.102 / 0.104 | 0.070 → 0.248 | 11.089 → 11.090 |
| 100B C2 Direct dHidden | 24 | 11.971 → 12.106 | +0.135 / 0.020 | 11.907 → 11.775 | -0.088 / 0.081 | 0.112 → 0.309 | 11.860 → 11.793 |
| 100B C8 Projection | 197 | 3.133 → 3.129 | +0.025 / 0.010 | 3.098 → 2.932 | -0.150 / 0.007 | 0.069 → 0.237 | 3.069 → 2.893 |
| 100B C8 dHidden | 24 | 2.831 → 3.010 | +0.181 / 0.006 | 2.802 → 2.808 | -0.000 / 0.012 | 0.049 → 0.237 | 2.779 → 2.777 |
| 100B C8 Direct dHidden | 24 | 2.867 → 2.945 | +0.079 / 0.009 | 2.885 → 2.842 | -0.023 / 0.005 | 0.077 → 0.158 | 2.792 → 2.792 |
| 700B C8 Projection | 24 | 7.388 → 7.616 | +0.249 / 0.025 | 7.778 → 7.826 | +0.067 / 0.093 | 0.093 → 0.274 | 7.298 → 7.343 |
| 700B C8 dHidden | 495 | 7.873 → 7.502 | -0.370 / 0.009 | 8.184 → 7.611 | -0.596 / 0.069 | 0.063 → 0.210 | 7.805 → 7.285 |
| 700B C8 Direct dHidden | 31 | 8.287 → 7.966 | -0.335 / 0.032 | 8.467 → 7.881 | -0.650 / 0.262 | 0.104 → 0.202 | 8.184 → 7.761 |

The corrected 700B C8 dHidden result saves 0.370 ms of warmed wall time per
chunk with the same total peak. The strided route saves 0.335 ms against the
allocate-and-copy control and removes its 42 MiB temporary. The 100B C2
projection saves 0.314 ms. For 100B C8 projection, Python submission
offset the 0.150 ms graph gain and increased the paired wall time by 0.025 ms.

The isolated memory pass releases each result and clears unused allocator
segments before the next measurement. GIFT total adds peak reserved memory
and current non-Torch device memory. The three columns report the control
peak followed by the candidate peak. The direct route removes the temporary
in each geometry. At 100B C8 that saves 16 MiB allocated while the reserved
peak stays the same. No sample recorded an allocator retry.

| Geometry and route | Allocated MiB | Reserved MiB | GIFT total GiB |
| --- | ---: | ---: | ---: |
| 100B C2 Projection | 3294.12 → 3294.12 | 3302.00 → 3302.00 | 3.893 → 3.893 |
| 100B C2 dHidden | 3358.13 → 3358.13 | 3362.00 → 3362.00 | 4.020 → 4.020 |
| 100B C2 Direct dHidden | 3358.13 → 3294.13 | 3382.00 → 3302.00 | 4.040 → 3.962 |
| 100B C8 Projection | 1791.12 → 1791.12 | 1802.00 → 1802.00 | 2.429 → 2.429 |
| 100B C8 dHidden | 1807.13 → 1807.13 | 1822.00 → 1822.00 | 2.516 → 2.516 |
| 100B C8 Direct dHidden | 1807.13 → 1791.13 | 1822.00 → 1822.00 | 2.516 → 2.516 |
| 700B C8 Projection | 3209.19 → 3209.19 | 3222.00 → 3222.00 | 3.815 → 3.815 |
| 700B C8 dHidden | 3251.19 → 3251.19 | 3262.00 → 3262.00 | 3.923 → 3.923 |
| 700B C8 Direct dHidden | 3251.19 → 3209.19 | 3302.00 → 3262.00 | 3.962 → 3.923 |

A 700B C8 trace observed one workspace memset and one GEMM for ordinary
ATen, ordinary cuBLASLt and batched cuBLASLt. The ATen kernel uses a 192×192 tile, ordinary tile 495 uses 256×96, and
batched tile 31 uses 192×128. The repaired trace gave single-call GEMM
durations of 7.548, 7.354 and 7.444 ms respectively. The earlier trace had
different durations with the same kernel names and ordering. The
rotated unprofiled samples provide the time comparison.
[Repaired 700B trace summaries](recheck/700b-c8-confirm.json)

[100B C2 measurements](recheck/100b-c2-confirm.json) · [100B C8 measurements](recheck/100b-c8-confirm.json) · [700B C8 measurements](recheck/700b-c8-confirm.json)

## Reproduction

The CUDA device lane mapping was physical GPU 5 for 700B C8, GPU 6 for 100B C8,
and GPU 7 for 100B C2. All were H100 80 GB HBM3 with 132 SMs, driver 570.133.20,
Torch `2.15.0.dev20260822+cu130` at `8b353065` and cuBLASLt 13.1.1. The
fixture and CE artifact hashes, GPU visibility, shapes, algorithm attributes,
status values and screening samples are in the three screen receipts.
[100B C2 screen](100b-c2-p0.json) · [100B C8 screen](100b-c8-p0.json) · [700B C8 screen](700b-c8-p0.json)

The scripts use the live, personal SR008 checkout. Run from its
`shape-matrix-1125` directory after sourcing `/etc/profile.d/wm.sh`, with
`PYTHONPATH="$PWD:$PWD/integrated/gift"`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
`OMP_NUM_THREADS=1` and `PYTORCH_ALLOC_CONF=expandable_segments:True`.


```bash
gemm_python=/opt/torchtitan-venv/bin/python3.12
"$gemm_python" -Bu tuning/proj/lt_ops.py
CUDA_VISIBLE_DEVICES=5 "$gemm_python" -Bu tuning/proj/gemm_bench.py \
  --model 700b --chunks 8 --exhaustive --rounds 15 \
  --output tuning/proj/700b-c8-p0.json
CUDA_VISIBLE_DEVICES=5 "$gemm_python" -Bu tuning/proj/gemm_confirm.py \
  --screen tuning/proj/700b-c8-p0.json --profile \
  --output tuning/proj/700b-c8-confirm.json
```

The first confirmation retained the previous return tensor across samples.
Its memory peaks are superseded by the dedicated memory pass in
`recheck.sh`. The defect did not affect the earlier all-chunk bitwise
comparisons or the preallocated graph calls. The recheck validates the screen
hash and recorded choice before reusing the parity result.
[Recheck launcher](recheck.sh)

[Device preflight](recheck/preflight.csv) · [Instrumented source and binary hashes](recheck/source.sha256)
