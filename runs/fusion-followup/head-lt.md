# cuBLASLt tuning for the staged LM-head backward

Keep native `torch.mm` for both staged-head backward GEMMs. The best pitched
dW candidate matched both BF16 outputs and their merged FP32 result and held
the same peak memory. Its confirmation saved 0.620 ms on the two-chunk eager
backward component, with 0.636 ms of variation between identical controls.
The dX winner also stayed within control variation.

The [staged GFSDP head](../../gift/torchtitan/distributed/gfsdp/_staged_head_dw.py)
at `ab62730a9` computes BF16 `dX = dLogits @ W` and writes BF16
`dW = dLogits.T @ X` directly into two half rows of the FP32 accumulation
buffer. The native Triton merge reads both halves before overwriting each row.
The replay runs against identical staged-head source bytes.

The component uses the retained rank-zero production-data capture
`runs/kernel-fusion/artifacts/raw/backward-inputs-001/component-inputs/rank0.pt`
with SHA-256 `15227882b1da44c2881b63622b6249bcddd307d90575216290f29ab9f720c727`.
The eager head in the existing [operand replay](../fusion-pack/head_dw_component.py)
regenerates the two actual folded BF16 logit-gradient chunks from the captured
hidden states, weights, labels, packed positions and global denominator.

| Role | cuBLAS column-major dimensions M × N × K | Operations A, B | Leading dimensions A, B, C/D |
| --- | --- | --- | --- |
| dX | 4096 × 8192 × 128256 | N, N | 4096, 128256, 4096 |
| dW staged | 4096 × 128256 × 8192 | N, T | 4096, 128256, 8192 |
| dW contiguous diagnostic | 4096 × 128256 × 8192 | N, T | 4096, 128256, 4096 |

The staged dW destination offsets are 0 and 4096 BF16 elements. All three
descriptors use BF16 A/B/C/D, `CUBLAS_COMPUTE_32F`, FP32 scaling, alpha 1,
beta 0 and the default epilogue. The search permits FP32 split-K reduction
and the default no-reduction scheme. The staged dW control writes into the
leased FP32 buffer through the original `torch.mm(..., out=half)`. The staged
dX control allocates its original BF16 result in `torch.mm`. The screen
preallocates it for both arms and checks that the native call matches the
original allocating call bitwise before comparing candidates.

The bounded sweep follows NVIDIA's algorithm-ID enumeration, capability
queries and descriptor validation. It takes up to 96 heuristic results per
workspace budget, and checks each exposed algorithm's base and advertised
tile/stage configurations with swizzle 0 and 1 where legal, custom options
0 to 2, and divisible split-K counts 2, 4, 8 and 16. The manual enumeration
keeps cluster and inner shape defaults. It also tries the distinct descriptor
configurations returned by the heuristics. Algorithms that need more than the
existing ATen workspace are excluded before launch. [NVIDIA cuBLASLt API](https://docs.nvidia.com/cuda/archive/13.1.1/cublas/index.html#using-the-cublaslt-api) · [NVIDIA sample at 4ad81eea](https://github.com/NVIDIA/CUDALibrarySamples/blob/4ad81eea85bcd344e1d148028ba6d508687cda50/cuBLASLt/Common/LtMatmulCustomFind.h)

The pinned ATen BF16 GEMM uses FP32 compute with the native BLAS preference.
The native classic cuBLAS workspace on Hopper defaults to 32 MiB. In the
default unified mode, cuBLASLt obtains the same classic workspace for its
current handle and stream. The replay verifies that borrowing returns the
same pointer twice and changes neither allocated nor reserved memory, then
queries heuristics at zero, the standard Lt ceiling, and the verified
existing classic workspace size. The candidate uses ATen's existing Lt handle
and borrows its workspace on the execution stream. [Pinned ATen GEMM](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/aten/src/ATen/cuda/CUDABlas.cpp) · [Pinned ATen workspace pool](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/aten/src/ATen/cuda/CublasHandlePool.cpp)

## Algorithm screen

The run used only physical GPU 0 of `sr008-gpu1`, an H100 80 GB HBM3 with
132 SMs and UUID `GPU-c94b0c9a-14ca-0083-dcc3-08cc6e0367a9`. It loaded Torch
`2.15.0.dev20260822+cu130` at `8b353065`, cuBLASLt 13.1.1 and the native
`Cublas` BLAS preference. The BF16 reduced-precision reduction option was
enabled for the native control. The explicit candidates allowed only compute
precision reductions. All shortlisted candidates used split K = 1 and
reduction scheme 0.

For each matrix layout, cuBLASLt exposed 20 algorithm IDs. Of 58,932 distinct
configurations submitted to `cublasLtMatmulAlgoCheck`, 87 passed within the
32 MiB inherited budget and 79 needed zero workspace. All 87 launched on
both real chunks, and 80 matched both chunks bitwise. The fastest designs
used 4 bytes of the existing workspace. The best zero-workspace designs
took 37–38 ms across two calls during screening, versus approximately
21–22 ms for the interleaved native controls.

The shortlist used 12 A/B/A samples of two preallocated calls per arm. A/B
deltas compare each candidate to the midpoint of the two controls. The A/A
column gives the median absolute difference between identical controls.
All times in the table are milliseconds.

| Role | Algorithm / tile | Control A CUDA event | Control A repeat | Candidate | Paired delta | A/A variation |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dX | 66 / 23 | 22.819 | 23.136 | 22.778 | −0.171 | 0.242 |
| dW staged | 66 / 495 | 23.704 | 23.667 | 23.031 | −0.755 | 0.393 |
| dW contiguous diagnostic | 66 / 495 | 23.682 | 24.509 | 23.092 | −0.843 | 0.163 |

The first integrated comparison advanced staged tile 495. It saved 0.722 ms
eager against 0.625 ms A/A variation and 0.709 ms graph replay against
0.717 ms A/A variation. The confirmation therefore reran all three
shortlisted staged dW tiles without screening new algorithms.

## Balanced confirmation

Each arm executed two native dX calls, two staged dW calls and the existing
Triton merge on a single stream. The eager arm used native dX allocation.
The graph arm reused buffers and excluded Python and dX allocation inside
each replay. Both arms ran 30 rounds, rotating and reversing the order so
each candidate and each of two identical controls occupied each position
six times. The capture always used the same real-data operands. These
component calls exclude CE, collectives and the rest of the training step.

| Staged dW arm | Eager CUDA-event median, ms | Eager paired delta, ms | Eager rounds faster | Graph CUDA-event median, ms | Graph paired delta, ms | Graph rounds faster |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native control | 48.038 | — | — | 48.132 | — | — |
| Native repeat | 48.055 | — | — | 47.967 | — | — |
| Algorithm 66, tile 495 | 47.271 | −0.329 | 20/30 | 47.434 | −0.505 | 19/30 |
| Algorithm 66, tile 29 | 48.122 | −0.094 | 15/30 | 48.270 | +0.144 | 15/30 |
| Algorithm 66, tile 24 | 47.256 | −0.620 | 21/30 | 47.196 | −0.724 | 20/30 |

Identical-control A/A variation was 0.636 ms eager and 0.667 ms in graph
replay. Tile 24's six balanced-cycle medians ranged from −1.259 to +1.396 ms
eager and −1.364 to +0.036 ms in graph replay. Its median eager wall time was
47.302 ms versus 48.058 and 48.094 ms for the controls. Its median host
submission was 86.8 μs versus 79.8 and 82.3 μs, and median host wait was
47.140 ms versus 47.968 and 48.003 ms. The candidate event spans are useful
for comparing the component, with possible GPU queue gaps inside the span.

All three shortlisted dW candidates again matched 525,336,576 BF16 elements
for each chunk and 525,336,576 FP32 elements after merge. The baseline and
each candidate peaked at 7,690,256,384 bytes allocated and reserved in
confirmation with zero allocator retries and OOMs. The 4-byte workspace
request used the same preexisting 33,554,432-byte ATen allocation.

CUPTI confirmed the native dW kernel's 320×128 tile was replaced by 256×96
for tile 495, 128×160 for tile 29 and 256×128 for tile 24. The dX kernel and
merge stayed the same. In three profiled calls per arm, median summed device
kernel time was 47.094 and 48.446 ms for the identical controls and 48.002 ms
for tile 24. Median non-kernel device work was four workspace memsets totaling
3.17, 3.23 and 3.23 μs. Median single-stream inter-event gaps were 18.37,
15.71 and 16.06 μs. The only CUDA allocation call in the confirmation trace
occurred inside the initial profiled native control. The profiler establishes
activation and launch shape. The separated device timings and balanced
unprofiled calls leave a repeatable eager improvement unresolved.

An opt-in implementation that borrowed scratch would need to guard the pinned
library, shape, strides and alignment, cache algorithm descriptors per device,
and look up the existing ATen workspace on the actual autograd stream without
allocating it. In the staged backward, native dX runs before dW on that stream.
The measured result did not justify adding that binding or a version-specific
runtime selector.

## Reproduction and receipts

The [compact result](results/head-lt.json) records the screened layouts,
shortlist, balanced timing summaries, allocator peaks and trace signatures.

Run from the synchronized checkout at
`/workspace-SR008.nfs2/vkropotin/optimization-task` on `sr008-gpu1`. Source
`/etc/profile.d/wm.sh` and run with
`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`. The original commands
used `/opt/torchtitan-venv/bin/python3.12 -B`.

```bash
/opt/torchtitan-venv/bin/python3.12 -B runs/fusion-followup/head_lt_sweep.py \
  --fixture runs/kernel-fusion/artifacts/raw/backward-inputs-001/component-inputs/rank0.pt \
  --fixture-sha256 15227882b1da44c2881b63622b6249bcddd307d90575216290f29ab9f720c727 \
  --output runs/fusion-followup/artifacts/raw/head-lt-001

/opt/torchtitan-venv/bin/python3.12 -B runs/fusion-followup/head_lt_confirm.py \
  --screen runs/fusion-followup/artifacts/raw/head-lt-001/result.json \
  --output runs/fusion-followup/artifacts/raw/head-lt-confirm-001
```

| Run under `runs/fusion-followup/artifacts/raw` | Source revision | `result.json` SHA-256 | `profile.trace.json` SHA-256 |
| --- | --- | --- | --- |
| `head-lt-001` | `cca7bb107` | `22c2218db4ba9fd346ed310f1118a8b56338566e8a738dbec7fc71c7ffee11c6` | `48337feb18cfde5f0177b893ea4933820cae827f46204ea14ab9cdb26e22ee3c` |
| `head-lt-confirm-001` | `dce3fd0e0` | `6fd9013d1b157d301ec29d4d9467d7cdad27e4abbc272d230cc527406d60d248` | `59c70203d73fbb96ce1c4e41b4f52f6ebb66effccac22c27c03cfb75cccce67b` |
