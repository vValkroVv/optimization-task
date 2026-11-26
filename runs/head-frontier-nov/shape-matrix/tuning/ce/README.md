# CUDA consuming cross-entropy tuning

The selected Hopper binary reduced synchronized regional wall time by about
2.5% on all three replayed shapes at the same GIFT total GPU memory. Every
raw FP32 row loss, separate FP32 head sum, weighted loss and BF16 gradient
matched the generalized cuda-13 control bitwise. This comparison covers the
CE region on the final projected chunk. Full-head and training throughput
are separate measurements.

## Scoped comparison

| Preset | Folded rows × vocabulary | Scored rows, main / MTP1 | Wall, ms | Wall change | GIFT total peak, GiB |
|---|---:|---:|---:|---:|---:|
| 100B C2 | 8,192 × 128,256 | 4,094 / 4,087 | 1.76288 → 1.71925 | −2.47% | 4.63757 → 4.63757 |
| 100B C8 | 2,048 × 128,256 | 1,022 / 1,019 | 0.56797 → 0.55377 | −2.50% | 1.72742 → 1.72742 |
| 700B C8 | 3,072 × 128,256 | 1,533 / 1,529 | 0.76839 → 0.74955 | −2.45% | 2.19617 → 2.19617 |

| Preset | Host submission, ms | Final host wait, ms | Eager CUDA events, ms | Graph device span, ms |
|---|---:|---:|---:|---:|
| 100B C2 | 0.42249 → 0.42188 | 1.34435 → 1.29777 | 1.74560 → 1.70147 | 1.60614 → 1.56024 |
| 100B C8 | 0.38052 → 0.37963 | 0.18900 → 0.17503 | 0.55208 → 0.53813 | 0.43082 → 0.41494 |
| 700B C8 | 0.38912 → 0.39326 | 0.37979 → 0.35878 | 0.75243 → 0.73330 | 0.62541 → 0.60542 |

The A/A/B arms were rotated on the same card for each case, with ten eager
warmups and 32 measured samples per arm. Wall changes divide the difference
of medians by the control median. Median cycle-paired wall changes for A/A
were −0.03%, +0.32% and +0.12%. The corresponding A/B changes were −2.52%,
−2.39% and −2.36%. Host submission includes Python, autograd and launch
submission. The host wait begins after the last event is submitted. Eager
CUDA events can include gaps between submissions. The graph diagnostic
excludes Python submission and includes all device work and intervening gaps
from the CE kernel through the separate loss sums, normalization and
autograd. Graph changes from displayed medians are −2.86%, −3.68% and
−3.20%. Individual kernel time, GPU idle time and physical HBM traffic were
unmeasured. Communication is outside this single-GPU region.

The input projection and restoration copy finish before the timing window.
The pristine logits stay live for restoration and the working BF16 logits
become the BF16 gradient. GIFT total memory includes the PyTorch reserve
peak and device memory outside the reserve. Allocator retries were zero.
The fixtures use the real packed `data-002` tokenizer rows and the
matrix's seeded synthetic BF16 weights and hidden tensors. The receipts
record the actual inputs and sample series.
[100B C2](raw/final-100b-c2.json) ·
[100B C8](raw/final-100b-c8.json) ·
[700B C8](raw/final-700b-c8.json) ·
[Matrix setup](../../report.md)

## Selected changes and checks

Tuning 3 retains eight CTAs of 256 threads at the largest vocabulary tier.
A packed BF16 maximum compares the original representable values without
an intervening conversion. Each warp lane prepares one FP32 rescaling factor
from adjacent running maxima. The denominator still merges tiles in the same
ordered FP32 recurrence, using the same exponent, multiplication and addition.
The implementation preserves the CUDA libdevice logarithm, original target
subtraction, FP32 upstream scaling and one BF16 rounding. The aligned path
receives the vocabulary at runtime and selects the cluster size from the native
4,096-element reduction tiles. Odd or unaligned storage takes the existing
bounded scalar load and store path. Runtime dimensions and the scale tensor
use the existing `centered_ce_launch_folded_v1` signature.
[CUDA packed-max semantics](https://docs.nvidia.com/cuda/archive/13.0.3/parallel-thread-execution/index.html#half-precision-floating-point-instructions-max) ·
[Hopper occupancy and distributed shared memory](https://docs.nvidia.com/cuda/archive/13.0.2/hopper-tuning-guide/index.html)

Against cuda-13, comparisons covered 1,050,673,152, 262,668,288 and
394,002,432 BF16 gradient entries and all FP32 row losses. Nine additional
direct ABI cases covered vocabulary widths 7 to 131,072, all cluster tiers,
both storage alignment paths, signed and zero scales, ignored tokens,
signed-zero logits and target indices at both vocabulary edges. A compiled
GIFT wrapper case used 11 heads, 13 rows per head and odd vocabulary 16,385
with a one-element BF16 storage offset. Changing the denominator and weights
preserved bitwise parity for each separate head sum, the weighted loss and
the complete in-place gradient.

The launch count, shared-memory staging, row-loss allocation and consumed
logits lifetime are the same as cuda-13. The largest aligned CTA uses 38
registers per thread versus 32 in the control, with zero compiler-reported
spills in both. The optimized CE adds no host dispatch, host synchronization,
device allocations, copies or collectives. Deployment uses the standard
compiled custom-op wrapper and requires one rebuilt shared library and its
digest. It inherits the existing Hopper, BF16, first-order gradient and
exclusive-logits guards. The candidate is an experiment overlay outside
GIFT's training-source tree. [Compiler receipt](raw/build-3.log)

Two 14-sample screens compared the retained shared-memory schedule, 512
threads, register retention, integer warp max, independent packed maxima
and unrolling the denominator loop. Register retention and 512 threads
regressed device time. Both integer and independent packed maxima were
slower than tuning 3. Unrolling the loop gave nearly identical spans in all
three cases. Tuning 3 kept the smaller change.
[First screen, C2](raw/screen-100b-c2.json) ·
[First screen, C8](raw/screen-100b-c8.json) ·
[First screen, 700B](raw/screen-700b-c8.json) ·
[Second screen, C2](raw/max-100b-c2.json) ·
[Second screen, C8](raw/max-100b-c8.json) ·
[Second screen, 700B](raw/max-700b-c8.json)

## Reproduce and integrate

The SR008 job was `60cdca27-79a8-46e2-ade4-2adcbb2e59a3`. The replay used
H100 80 GB HBM3 GPU 3 for both 100B cases and GPU 4 for 700B C8, driver
570.133.20, CUDA compiler 13.2.51 and PyTorch
`2.15.0.dev20260822+cu130`. The same Python harness and GIFT wrapper ran
all arms with the expandable-segment allocator. The baseline DSO SHA-256
was `17b006cc257566200112b089199c97bb76f7e7ad95118dde923eebf845adcac8`.

The measured candidate lives at
`/workspace-SR008.nfs2/vkropotin/optimization-task/runs/head-frontier-nov/shape-matrix-1125/tuning/ce/centered-tuned.so`
and its SHA-256 is
`d3343cd518a05a6d5bea58b0493fa33373e040f1ce3dd5864b14c03033528929`.
Pass the absolute path and its digest through
`GIFT_EXACT_FOLDED_CE_LIBRARY` and `GIFT_EXACT_FOLDED_CE_LIBRARY_SHA256`.
Keep the existing harness's `cuda13` variant for the generic GEMM plan.

From the remote `shape-matrix-1125` directory, build from the verified
baseline source and run the benchmark as follows. The build directory must
be new. The rebuild in a separate directory produced a different DSO
digest and identical CUDA SASS and resource dumps. Use the digest of the
specific DSO under test.

```bash
bash tuning/ce/build.sh integrated/gift/torchtitan/ops/csrc/exact_folded_ce tuning/ce/new-build

source /etc/profile.d/wm.sh
export PYTHONPATH="$PWD:$PWD/integrated/gift"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3
export PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2
/opt/torchtitan-venv/bin/python3.12 -Bu tuning/ce/bench.py \
  --case 100b --chunks 2 --fixture inputs/100b.pt \
  --control native-ce/centered-folded-v1.so --candidate tuning/ce/new-build/centered-tune-3.so \
  --output tuning/ce/new-build/100b-c2.json --graph --wrapper-edge
```

The [build script](build.sh) verifies the baseline source hashes and applies
the [first-sweep patch](candidate.patch). Tuning 1 enables packed max.
Tuning 3 adds rescale preparation. Tunings 7, 11 and 15 add register
retention, 512 threads per CTA, and both to tuning 3, respectively. Phase 2 uses
the [second-sweep patch](second-sweep.patch) with tunings 19, 35, 51 and
67 for integer max, independent packed maxima, both, and unrolling,
respectively. The [benchmark harness](bench.py) records all regional scopes
and parity checks. Full replay is required to resolve the integrated
effect, since the absolute CE saving is 14–44 µs per tested chunk.
