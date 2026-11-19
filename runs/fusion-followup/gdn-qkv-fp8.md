# GDN short-convolution FP8 gradient epilogue

The fused epilogue preserved the native gradient bits and reduced allocated
memory, but increased complete backward latency by 4.0% on the captured
production microbatch. Keep the baseline in the production source. The
[measured patch](gdn-qkv-fp8.patch) records the opt-in implementation at
`da67a369b`. [Machine result and raw trace hashes](results/gdn-qkv-fp8.json).

## Scope

The rank-zero capture came from the selected eight-H100 production recipe,
using the production tokenizer, real data and the existing short-convolution
backward. It saved the projection input and weight, the convolution input and
weight, and incoming dQ/dK/dV from the same native microbatch. The replay on
H100 physical GPU 2 reconstructed the captured QKV tensor bitwise. It used two
sequences of 4,096 tokens, input width 4,096 and QKV width 8,192. The convolution
had width four, 16 Q/K heads, 32 V heads and head width 128. The compiled replay
used Inductor fullgraph, nonreentrant SAC, an empty save-category list and
`early_stop=False`. It exercised the actual projection and convolution modules
without the surrounding distributed training step.
[Capture](gdn_capture.py) · [Replay](gdn_component.py).

The candidate rounds dQKV to BF16 in registers before deriving CudaCherry's
power-of-two vec128 scales and E4M3 codes. It writes the codes and block-first
scales directly to the existing FP8 projection backward. The ordinary
projection and convolution forward calls, saved tensors, dgrad and wgrad GEMMs,
gates and fallback remain on their existing paths. The compatibility guard
keeps the packed gradient on the direct projection edge and requires aligned
BF16 input and output. CudaCherry's installed
`row_quantization_kernel.cu` SHA-256 is
`e62efe4691fc3bd3fdea79443ca1150a0928c0994d6a4dfda1cb9abae26ecbbd`.
[GDN output-norm FP8 donor](../../gift/torchtitan/models/common/attention/linear/gdn_output_fp8.py)
· [NVIDIA E4M3 conversion](https://docs.nvidia.com/cuda/archive/12.3.2/parallel-thread-execution/index.html).

## Timing and memory

The backward replay starts with captured convolution inputs and the projection
state that forward saves. It ends after convolution dW, projection dX and
projection dW. Inputs and saved-state preparation precede timing. CUDA Graph
values are medians of seven spans of 20 warmed replays from distinct captures
of baseline, candidate and identical control in alternating order. The
synchronized call measurements run the same backward without capture.

| Backward scope, µs | Baseline | Candidate | Identical control |
|---|---:|---:|---:|
| CUDA Graph elapsed | 1,778.144 | 1,849.533 | 1,776.083 |
| CUDA event elapsed around host submission | 1,869.376 | 1,923.712 | 1,856.672 |
| Host call | 332.062 | 360.784 | 342.462 |
| Event record and wait | 1,532.257 | 1,596.924 | 1,520.434 |
| Synchronized call | 1,875.766 | 1,930.354 | 1,862.896 |

The single-stream backward profile attributes 641.628 µs to stock convolution
backward and 71.039 µs to vec128 quantization. The fused kernel costs
775.871 µs and replaces both launches. The first consumer GEMM costs 406.750
versus 406.816 µs and the weight-gradient GEMM costs 516.317 versus 514.367 µs.
Kernel sums are 1,766.422 versus 1,823.294 µs, device gaps 15.711 versus
14.688 µs, and copies and memsets 1.600 versus 1.568 µs. The compiled SAC
forward/backward profile has device spans of 3,029.517 versus 3,062.660 µs
and kernel sums of 2,882.507 versus 2,934.304 µs. No collectives run inside
either component replay. Local profiles attribute launches and do not measure
training-step latency.

| Peak memory, MiB | Baseline allocated / reserved | Candidate allocated / reserved |
|---|---:|---:|
| Backward | 935.188 / 952.000 | 871.188 / 908.000 |
| Compiled SAC forward and backward | 1,171.439 / 1,260.000 | 1,109.439 / 1,196.000 |

The paired memory samples start with equal allocations and reserved memory.
Removing the materialized BF16 dQKV removes the overlap between that tensor
and its FP8 projection payload.

## Correctness and variants

The final replay matched FP8 codes and scales, convolution dW, projection dX
and dW, and forward Q/K/V bitwise. The comparison passed in eager and compiled
SAC modes and under BF16 AMP with a frozen projection weight. A separate
diagnostic passed packed boundaries `[0, 1, 3, 68, 128]` and a 129-token case
that exercises the unaligned fallback.

The first Triton implementation changed implicit reduction layouts when it
used an FP8 store. The native replay found 48 different FP8 bytes and 16
different convolution-weight-gradient bytes. Explicit Gluon layouts restored
bitwise equality. The first shared-memory version was 7.3% slower for complete
backward. Swizzling the shifted tap reads and reusing the coalesced input load
reduced that regression to 4.0%. The final native kernel uses 255 registers,
8 reported spills and 80 KiB shared per CTA. Stock uses 255 registers, 130
reported spills and 32 KiB shared. The packed diagnostic had zero reported
spills on the fused path. Early Gluon compile failures involved the broadcast
API, constexpr unrolling and the shared descriptor extent. The corrected kernel
compiled off-GPU for SM90 and passed the GPU comparisons.

To reproduce on the synchronized task server, use revision `da67a369b` or
apply the [patch](gdn-qkv-fp8.patch) to the restored training source. Use an
assigned GPU and fresh output directory.

```bash
source /etc/profile.d/wm.sh
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
  PYTHONPATH="$PWD/gift:$PWD/gift/contrib/flash-linear-attention" \
  /opt/torchtitan-venv/bin/python3.12 -B runs/fusion-followup/gdn_component.py \
  --fixture runs/kernel-fusion/artifacts/raw/followup-fixtures-001/component-inputs/gdn-qkv-rank0.pt \
  --output runs/fusion-followup/artifacts/raw/gdn-replay
```
