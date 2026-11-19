# Final-norm hidden z-loss component

On one H100, fusing the final-norm hidden z-loss reduced median eager
forward-plus-backward time by **74.3%** and both measured absolute CUDA
allocator peaks by about **512 MiB**. The native FP32 loss and every BF16
activation-gradient element matched bitwise.

The crop applies a coefficient of `1e-7` to the final norm input. The replay
uses rank 0's `mtp.hidden` from the real-data `data-002` component capture.
`Decoder._forward_mtp` sends the same trunk tensor to the final norm and to the
MTP module. The norm's GFSDP wrapper has forward input casting disabled. The
captured tensor is contiguous BF16 with shape `[2, 4096, 4096]`. The first
microbatch contains 8,192 positions. The z-loss includes every position.
[Crop preset](../../gift/configs/presets/models/gift_moe/gift_3_5/100b_crop8.yaml) ·
[Decoder call site](../../gift/torchtitan/models/common/decoder_blocks/decoder.py) ·
[Capture and fixture verification](../production-data-abcd/component_fixtures.py)

## Arithmetic and storage

`GIFT_FUSED_HIDDEN_Z_LOSS=1` selects the CUDA kernel for a plain, nonempty,
contiguous BF16 tensor of width 4,096. The eager ATen expression remains the
default and handles other device types and layouts. The candidate casts the
input to FP32 in registers, computes the magnitude and row maximum, and writes
one FP32 exponential temporary. ATen computes the row sum in its original
reduction order, followed by the original log, add, square and mean. The
backward retains the BF16 input and FP32 row log-sum-exp, then rounds once when
storing the BF16 gradient. Both Triton launches use libdevice `exp` with
subnormal arithmetic enabled. The higher-order derivative uses the equivalent
differentiable ATen expression. [Implementation](../../gift/torchtitan/ops/hidden_z_loss.py) ·
[Pinned ATen forward](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/aten/src/ATen/native/ReduceOps.cpp) ·
[Pinned ATen backward](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/torch/csrc/autograd/FunctionsManual.cpp) ·
[Triton row-loading reference](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)

The existing pre-hook runs once per forward microbatch outside the compiled
repeated blocks and their activation checkpoint regions.
The gradient splice, token weighting, and metric recording use the same
operators in both arms. The change adds no collectives, host reads of device
values, checkpoint state, or long-lived buffer. The native training trace is
the evidence for full-recipe activation, memory pressure and overlap.
[Regularizer and splice](../../gift/torchtitan/models/common/regularization.py) ·
[Compile boundary](../../gift/torchtitan/distributed/compile.py) ·
[Auxiliary accounting](../../gift/torchtitan/components/aux_loss.py)

## Numerical checks

At `799a2cb7a`, the FP32 scalar loss, all 8,192 FP32 row log-sum-exp values,
and all 33,554,432 BF16 gradient elements matched bitwise on the native
fixture. Zero incoming gradient, signed zeros, BF16 subnormal gradients, and
large finite inputs were covered. Infinity and NaN cases matched with NaNs
compared by classification. Training-mode hook checks matched the unchanged
primal output, BF16 gradient, and fractional-scale auxiliary total. The
hook also bypassed regularization in evaluation. The existing CPU z-loss
suite passed six tests with the switch enabled and CUDA hidden.

## Timing and memory

The timing run used source revision `f065ab3b6`, PyTorch
`2.15.0.dev20260822+cu130` at `8b353065ce`, Triton `3.8.0`, and physical GPU 0
on `sr008-gpu1`. Timing rows give the two run medians in order and ranges for
profile regions captured once per arm. Eager and graph percentages compare
the midpoints of the two medians.

| Scope | Baseline | Fused |
| --- | ---: | ---: |
| Eager F+B wall time, ms per call | 1.1595, 1.1611 | 0.2987, 0.2984 |
| Warm CUDA Graph F+B span, ms per replay | 1.1229, 1.1228 | 0.1808, 0.1809 |
| Absolute peak allocated, MiB | 704.0635 | 192.0625 |
| Absolute peak reserved, MiB | 706 | 194 |
| Forward GPU kernel work, µs | 514.5–514.8 | 126.6–127.3 |
| Backward GPU kernel work, µs | 616.3–617.5 | 48.2–48.7 |
| Forward host scope, µs | 2659–2740 | 2451–2460 |
| Backward host scope, µs | 2622–2626 | 2194–2396 |
| Forward between-event GPU gaps, µs | 220–257 | 88–112 |
| Backward between-event GPU gaps, µs | 2204–2226 | 2053–2257 |
| Backward device memory operations, µs | 1.1–1.2 | 1.1 |
| GPU kernel launches, forward / backward | 14 / 11 | 7 / 6 |

The eager wall-time reduction was 74.3% and graph replay was 83.9% shorter.
Baseline medians differed by 0.14% for eager and 0.011% for graph replay.
Allocated and reserved peaks fell by 72.7% and 72.5%. Both arms began the
profile with the same 64 MiB fixture allocation. `_shifted_exp_kernel` and
`_magnitude_grad_kernel` appeared in the candidate trace. Forward copies and
memsets took zero time, and both arms had zero CUDA synchronization time
inside the capture regions. The host scopes and GPU gaps include profiler
overhead. All figures apply to this one-device component replay.

## Replay method

The runner verifies the serialized native fixture and the input-ID hash,
then runs separate processes in baseline, fused, baseline, fused order. Each
memory sample records absolute allocated and reserved peaks across one profiled
forward and backward after warmup and allocator-cache release. The Kineto
trace correlates kernels to forward and backward on one CUDA stream. Unprofiled
wall time includes Python, dispatcher, autograd and submission, with one
completion synchronization per ten calls. CUDA Graph events measure replay of
the same activation on the same device and exclude per-call Python and
autograd. The hook and its accounting are tested separately. [Runner](zloss_component.py)

```sh
source /etc/profile.d/wm.sh
fixture=runs/production-data-abcd/artifacts/raw/proddata-component-capture-001/native-inputs
out=runs/fusion-pack/artifacts/raw/zloss-component-002
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/gift" TRITON_CACHE_DIR="$PWD/$out/triton-cache" \
  /opt/torchtitan-venv/bin/python3.12 -B runs/fusion-pack/zloss_component.py \
  check --fixture "$fixture" --output "$out/1-check.json"
```

For timing and traces, use the same command with `baseline` or `fused` and a
fresh output filename. JSON receipts retain exact environment and fixture
identities and each timing sample. On the task host, the numerical receipt is
`runs/fusion-pack/artifacts/raw/zloss-component-002/1-check.json`. The four
timing receipts and corresponding Kineto traces are under
`runs/fusion-pack/artifacts/raw/zloss-component-003/`. The verified fixture
tensor archive SHA-256 is
`dced479c2bea4ff5cc5ade9dad04531d075d74f4bbc5c79e3f069c049d8863a1`.

The initial CUDA initialization needed the image compatibility profile. The
first fused pilot exposed a Triton version mismatch in the max invocation.
After both were corrected, the numerical check passed. The initial timing
collector missed backward kernels submitted by PyTorch's autograd worker.
The final collector matches CUDA launches by process and phase timestamps and
verifies the fused kernel names in the resulting device events. The pilot logs
and partial traces remain under `zloss-component-001` and `zloss-component-002`.
