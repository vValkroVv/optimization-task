# LM-head hidden packing and backward GEMM order

The isolated C2 and C8 replays retain the native writer and `dHidden → dW`
order. None of the tested changes produced a verified Pareto improvement.
All variants matched the five output digests and the native FP64 dW error.

## Matched wall time and GIFT memory

| Fixed chunks | Arm | Wall, ms | Change | Process spread, ms | GIFT total, GiB | Allocated, GiB | Reserved, GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C2 | Control, order series | 76.105 | — | 0.201 | 11.6376 | 10.3482 | 10.3535 |
| C2 | dW first | 76.263 | +0.21% | 0.186 | 11.6376 | 10.3482 | 10.3535 |
| C2 | Control, packing series | 76.532 | — | 0.238 | 11.6376 | 10.3482 | 10.3535 |
| C2 | Packed hidden | 77.645 | +1.45% | <0.001 | 11.6962 | 10.4107 | 10.4121 |
| C2 | Packed hidden and dW first | 77.468 | +1.22% | 0.001 | 11.6376 | 10.3482 | 10.3535 |
| C8 | Control, order series | 79.208 | — | 0.320 | 10.1337 | 8.8336 | 8.8496 |
| C8 | dW first | 79.602 | +0.50% | 0.483 | 10.1337 | 8.8336 | 8.8496 |
| C8 | Control, packing series | 80.057 | — | 0.582 | 10.1337 | 8.8336 | 8.8496 |
| C8 | Packed hidden | 79.489 | −0.71% | 0.531 | 10.1532 | 8.8492 | 8.8691 |
| C8 | Packed hidden and dW first | 79.973 | −0.11% | 1.553 | 10.1337 | 8.8336 | 8.8496 |
| C8 | Control, reversed repeat | 78.823 | — | 0.078 | 10.1337 | 8.8336 | 8.8496 |
| C8 | Packed hidden, reversed repeat | 79.253 | +0.55% | 0.019 | 10.1532 | 8.8492 | 8.8691 |

Each estimate uses two fresh processes per arm, ten warmup and ten timed
calls per process. The order series ran control, order, order, control. The
packing series ran control, packed, combined, combined, packed, control.
Wall is the median of the process medians, with synchronization within the
timer. The spread is the absolute difference of the process medians. The
resolved allocator used expandable segments. GIFT total is the maximum
native estimate over both replicates and all timed calls. It adds the
allocator's peak reserve and observed external usage at the sample boundary.
The reserve and allocated columns show their own maxima. Each process had
zero allocation retries and OOMs. [Matched summary](summary.json)

The C2 packed tensor increased allocated bytes by exactly 64 MiB and total
peak by 60 MiB. C8 increased allocated bytes by 16 MiB and total peak by
20 MiB. Running dW first removed these increases from the measured peak.
C8 packing's 0.568 ms apparent improvement was within the 0.582 ms control
spread with higher peak memory. The confirmation used packed, control, control, packed
and disabled the diagnostic stage hooks. Packing was 0.430 ms slower with
the same extra 20 MiB. No candidate advanced to an all-gather alias
confirmation.

The latest training control uses regular all-gather. The previous
memory-priority singleton record uses the same optimization stack with a
BF16 all-gather input alias. Its published values were 76.615 ms / 10.681
GiB at C2 and 78.820 ms / 9.177 GiB at C8. Those values are historical,
with different replay source and physical GPU assignments. The regular
controls in this experiment matched the older regular-gather replay's
five reset-output digests at both chunk counts.

## Writer, launch order and numerical verification

The production source `lm10-c1` at
`e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f` ran the native `new` mode
and its configured writer plans 9.14 and 5.5. File hashes in the fresh
control receipts pin the active uncommitted trial source. The replay uses
the retained production-data rank-zero capture with B2/S4096, hidden
width 4096 and vocabulary 128256. The capture SHA-256 is
`15227882b1da44c2881b63622b6249bcddd307d90575216290f29ab9f720c727`.
Each arm exercised BF16 projection, main and MTP losses with the captured
targets and denominator, weights 1.0 and 0.3, compiled Folded CE, in-place
BF16 dLogits, early logits release, BF16 dHidden, FP32 GFSDP dW and
independent hidden z-loss. NCCL singleton SUM kept the native GFSDP lease
and reduction. Two distinct BF16 gather storages and 24 native gathers
were verified in each full paired process, including setup and diagnostic
calls. The one Inductor graph stayed stable and CUDA Graph replay was off.

Packing uses `hidden.T.contiguous().T` with logical shape unchanged and
strides changed from `(4096, 1)` to `(1, rows)`. It reads and writes
256 MiB per microbatch. The search tried four correct cuBLASLt descriptor
encodings, querying 0 and 32 MiB workspace limits and measuring 32 distinct
returned configurations per shape. The returned configurations used
4 bytes of scratch. The selected beta zero and beta one plans were
21.29 for C2 and 21.13 for C8 in the tuning records. Both shapes selected
algorithm 66 / tile 29 / split K 1 for both beta values. Beta zero
overwrites the native FP32 slot on the first chunk and beta one accumulates
each later chunk into that slot. The single-stream full dW chain includes
packing and allocation. [NVIDIA cuBLASLt API](https://docs.nvidia.com/cuda/archive/13.1.1/cublas/index.html#cublasltmatmul) · [C2 tuning](tune-c2.json) · [C8 tuning](tune-c8.json)

| Fixed chunks | Native and repeat dW-chain event medians, ms | Packed dW-chain median, ms | FP64 dW relative L2, all arms | FP64 max absolute error, all arms |
| --- | ---: | ---: | ---: | ---: |
| C2 | 24.404 / 24.495 | 24.653 | 1.294919e−5 | 3.988939e−9 |
| C8 | 25.100 / 25.452 | 25.328 | 3.822941e−6 | 9.704260e−10 |

After timing, an invocation with the optimizer-facing gradient reset hashed
the entirety of loss, dHidden, FP32 dW, z-loss and z-loss gradient. Every
arm matched all five digests. The FP64 oracle used the actual BF16
writer callback operands and original chunk order. Relative L2 and maximum
absolute error were identical to the native writer, and all results were
finite. The full replay recorded the expected beta sequence in every process.

## Warmed device trace

Nsight Systems captured one separate post-warmup invocation per arm. CUDA
launch correlation matches the per-chunk NVTX operation ranges to device
kernels. All labeled kernels in the table below use one stream. The last
column sums the listed durations. Tiny workspace memsets are reported in
the trace summary. [Kernel and launch evidence](trace-summary.json)

| Chunks | Arm | dHidden GEMMs, ms | dW GEMMs, ms | Pack kernels, ms | Combined, ms |
| --- | --- | ---: | ---: | ---: | ---: |
| C2 | Control | 20.965 | 21.282 | — | 42.247 |
| C2 | dW first | 21.884 | 22.114 | — | 43.999 |
| C2 | Packed hidden | 22.689 | 22.711 | 0.513 | 45.913 |
| C2 | Packed hidden and dW first | 22.294 | 22.184 | 0.509 | 44.988 |
| C8 | Control | 20.255 | 22.407 | — | 42.662 |
| C8 | dW first | 21.245 | 22.635 | — | 43.880 |
| C8 | Packed hidden | 21.437 | 22.488 | 0.528 | 44.454 |
| C8 | Packed hidden and dW first | 20.608 | 22.394 | 0.523 | 43.526 |

At C8, moving dW first preserved the dHidden kernel name and launch shape.
The two GEMM totals rose by 1.218 ms in the capture, so it did not
reproduce the full-training trace's dHidden recovery hypothesis under
singleton isolation. The packed dW trace showed one physical copy kernel
per chunk in both shapes. The profiler establishes active order, layout,
launches and device durations for the captured invocation. Use the matched
unprofiled wall series for effect magnitude. The paired receipts report host
submission and synchronization wait. Those intervals overlap GPU activity.
Stage CUDA events from one additional reset invocation were more variable
than the warmed trace at C2. Device idle and dependency attribution were
outside this check. There was no inter-GPU communication in this run.

## Reproduction

Use the existing SR008 allocation
`lm-mpi-job-84fd3579-7bd2-4969-ae47-b7af17b73ca2`. All C2 replays used
physical H100 2 and all C8 replays used physical H100 3. Both were idle
before the lanes started. The device UUIDs and runtime versions are in the
raw receipts. Driver 570.133.20, PyTorch `2.15.0.dev20260822+cu130`,
cuBLASLt 13.1.1, NCCL 2.30.7, Nsight Systems 2026.1.2.63.

The repro directory on `sr008-gpu` is
`/workspace-SR008.nfs2/vkropotin/optimization-task/runs/head-frontier-nov/layout-order`.
Run the commands on the verified GPU allocation. The lane script selects
the assigned GPU, checks for existing compute processes and starts each
fresh process with the same allocator configuration.

```sh
source /etc/profile.d/wm.sh
export OMP_NUM_THREADS=2 CUDACHERRY_DISABLE_JIT=1 PYTORCH_ALLOC_CONF=expandable_segments:True
python=/opt/torchtitan-venv/bin/python3.12
export CUDA_VISIBLE_DEVICES=$(nvidia-smi -i 2 --query-gpu=uuid --format=csv,noheader)
"$python" -Bu packed_dw.py 2 tune-c2/plan.json
export CUDA_VISIBLE_DEVICES=$(nvidia-smi -i 3 --query-gpu=uuid --format=csv,noheader)
"$python" -Bu packed_dw.py 8 tune-c8/plan.json
bash lane.sh 2 2 paired-order control,fp64,stages order,fp64,stages order,stages control,stages
bash lane.sh 3 8 paired-order control,fp64,stages order,fp64,stages order,stages control,stages
bash lane.sh 2 2 paired-layout control,fp64,stages packed,fp64,stages packed-order,fp64,stages packed-order,stages packed,stages control,stages
bash lane.sh 3 8 paired-layout control,fp64,stages packed,fp64,stages packed-order,fp64,stages packed-order,stages packed,stages control,stages
bash lane.sh 3 8 paired-layout-repeat packed control,fp64 control packed,fp64

# Example trace of the C8 packed writer. Set the verified visible UUID first.
nsys profile --trace cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop --output=trace-c8/packed \
  "$python" -Bu profile_native.py 8 packed trace-c8/packed.json --plan tune-c8/plan.json
nsys export --type=sqlite --output=trace-c8/packed.sqlite trace-c8/packed.nsys-rep
"$python" -B summarize.py paired-order-c2 paired-order-c8 paired-layout-c2 paired-layout-c8 paired-layout-repeat-c8
"$python" -B read_trace.py trace-c2 trace-c8
```

The first C8 smoke found version-specific gather-probe attributes from the
historical replay. The compatible probe and corrected activation checks passed
the retry. The first C2 tuner completed screening but could not serialize the
GPU UUID object. The rerun recorded the verified device UUID string and
completed both beta screens. Raw logs and per-process receipts remain in
the remote reproduction directory. The tests cover isolated memory and
execution on captured inputs. Training overlap and distributed step time
remain outside this measurement.
