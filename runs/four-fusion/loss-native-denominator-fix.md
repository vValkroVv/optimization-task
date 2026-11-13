# Native FLCE denominator correction (K2)

K2 is resolved in source and scoped CUDA checks. The earlier implementation's
Tensor-only denominator assumption failed on the actual DP producer, and its
Tensor-only tests did not cover the accepted DP8 boundary. This note owns the
correction evidence; [kernel review](kernel-review.md#k2-flce-has-the-same-native-denominator-mismatch)
owns the finding and [initial implementation](loss-implementation.md) retains
the earlier checks. Source remains based on task revision `33b336a5`.

Finite acceptance: reproduce native float failure; preserve exact normalization
and R1 metadata guards; avoid H2D scalar constants and per-count compilation;
verify eager/compiled MTP loss/components/dH/dW, hooks and GFSDP state; restore
core and check independent patches. All these scoped items pass. Production
DP8 performance and trace gates remain open with the coordinator.

## Producer and arithmetic

`Trainer.train_step` obtains the global count from `dist_utils.dist_sum` when
DP is enabled (`trainer.py:765–769`). `_dist_reduce` returns
`float(funcol.all_reduce(...).item())` (`distributed/utils.py:32–74`), preserving
an existing host normalization boundary. Non-DP supplies a CUDA FP32 Tensor.
The accepted B2/S4096/accumulation32/DP8 workload has at most 2,097,152 physical
tokens per update; the regression uses the native valid count 2,096,640.

Inspected the installed Torch build commit
`8b353065ce260336c5fdfa74d754e8bd9b91b062` and its pinned official
[CUDA true-division source](https://raw.githubusercontent.com/pytorch/pytorch/8b353065ce260336c5fdfa74d754e8bd9b91b062/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu).
For an FP32 tensor divided by a CPU scalar, ATen computes a double reciprocal,
rounds that reciprocal to FP32, then multiplies. Consequently the CE backward's
unit upstream is weighted first, then multiplied by that rounded reciprocal;
that FP32 incoming scale multiplies the centered `(p - target)` before BF16
storage. Direct `weight / denominator` can differ in rounding.

The corrected opaque op takes an optional Tensor denominator and a `Scalar`
host value (`torch.types.Number` schema). Its body converts the host value to
the same Python double, computes the reciprocal there, and passes it as a
runtime FP32 Triton argument. The row kernel multiplies the FP32 head weight
by the rounded reciprocal before multiplying the centered gradient. For the
Tensor path, the kernel retains a device scalar load and uses `tl.div_rn`
for the head-weight division. Loss values retain `FP32 SUM / divisor * weight`
with the original CPU-scalar operation order. No host-to-device Tensor constant
is created. Default denominator `None` uses host count one.

## Compiler boundary and explicit fallback

The installed defaults are `specialize_float=False`,
`assume_static_by_default=True`, `triton.cudagraphs=False`; no setting was
mutated. `builder.py::wrap_symfloat` considers those flags, but the installed
`fx/passes/_tensorify_python_scalars.py` cannot tensorify custom-op scalar
arguments, and `output_graph.py::remove_tensorify_specialized_graphargs` then
specializes the remaining float. This occurred for both a `float` schema and a
`Scalar` schema, including callable-local `dynamic=True`. Guard logs record
`denominator == 2096640.0`, then `denominator == 2096632.0`. This would compile
again as ignored-target counts changed.

The selected adapter therefore converts the producer's **integral-valued host
float** to a Python integer before entering the compiled function. Conversion
requires signed-int64 range and an exact float round trip. The original numeric
value is preserved; the opaque body restores the same double for reciprocal
and loss arithmetic. The loss callable uses local `dynamic=True`, so the host
count enters as a dynamic SymInt. Tensor denominators retain their original
path. Fractional, nonfinite or out-of-range host values use the original
logits/CE path, preserving their behavior without truncating or copying them.
The low-level unit-upstream helper remains restricted to its documented native
bridge; this is not a generic arbitrary-upstream FLCE API.

The existing target int64, target-shape and input/target/weight/device checks
remain before the custom op. The denominator-device check applies to Tensor
denominators. No source-level integration patch changed. The shared row kernel's
new host-reciprocal arm is used only with FLCE head weights; CE-only's default
incoming-scale load and arithmetic remain unchanged.

## Reproducible checks

Physical CUDA device **2** was empty before checks and had no compute processes
after release. Runtime/environment are unchanged from
[the initial evidence](loss-implementation.md#executed-verification):
`/opt/torchtitan-venv/bin/python3.12`, Torch
`2.15.0.dev20260822+cu130`, CUDA 13.0, Triton 3.8.0. No installation, all-eight
GPU execution, training, TPS measurement or profiling was performed.

From task root, apply only `linear-ce.patch` and use:

```bash
source /etc/profile.d/wm.sh
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=gift
export TORCHINDUCTOR_CACHE_DIR=$PWD/runs/four-fusion/artifacts/raw/loss-checks/inductor
```

Actual failing regression before the fix:

```bash
/opt/torchtitan-venv/bin/python3.12 -m pytest -q gift/tests/unit_tests/test_fusion_losses.py -k 'native_mtp and native_float'
```

`denominator-before.log`: **2 failed**, eager `AttributeError` on `.device` and
compiled `Unsupported` reporting that same boundary. The test obtains
`2096640.0` by calling the actual `dist_sum` producer; it passes that unchanged
into real MTP/two-chunk loss orchestration. The corrected version passes both
modes and proves two FLCE activations, with exact host-count value preserved.

Final affected regression command:

```bash
TORCH_LOGS=recompiles /opt/torchtitan-venv/bin/python3.12 -m pytest -q -s gift/tests/unit_tests/test_fusion_losses.py -k 'linear_ce or native or hooks'
```

`denominator-affected-final.log`: **26 passed, 8 deselected, none skipped**,
8.32 s. Checks include the existing Tensor known-scale/near-certain-target
checks, all ten R1 eager/compiled rejection cases, both Tensor and native float
MTP loss/component/dH/dW comparisons, exact head/state identity, native float
hook/eval/probe behavior, fractional-denominator fallback, and signed-int64
round-trip boundaries. Native-float tolerances scale by `29/2096640` from the
original denominator-29 fixture; the absolute tolerances do not dwarf the new
smaller gradients.

The changing-count test sends actual producer outputs 2,096,640, 2,096,632 and
2,096,624 through the same selected compiled native MTP instance and shapes.
Its counting backend delegates to real Inductor. Compile callback counts are
**[1, 1, 1]**, with no count-value recompile guards. All three calls execute
forward and backward and preserve loss normalization. Other recompile messages
in that combined suite concern deliberately different dtypes/grad requirements,
Tensor versus scalar routes, or different fixture shapes, not changing counts.

Actual-mesh producer and GFSDP check:

```bash
/opt/torchtitan-venv/bin/python3.12 runs/four-fusion/artifacts/raw/loss-checks/gfsdp_native_float.py
```

`denominator-gfsdp-final.log`: eager and compiled both pass 32 microbatches,
two outer chunks and **64 selected FLCE calls** each. Unlike the original
fixture, it calls `dist_sum(..., mesh=mesh)`, exercising the functional
all-reduce and Python-float return branch on one NCCL rank. Both control and
candidate use FP32 master parameters, BF16 head compute and FP32 GFSDP
accumulation; every dH/dW element is compared. Maximum absolute differences:

| Mode | dH | Accumulated dW |
| --- | ---: | ---: |
| Eager | 5.820766091346741e-11 | 2.9103830456733704e-11 |
| Compiled | 5.820766091346741e-11 | 2.9103830456733704e-11 |

The fixture checks exact parameter identity, state-dict keys, last-backward and
gradient-sync flags after all microbatches. This is single-device native-head
evidence, not eight-rank gather/reduce or full crop memory validation.

Retained diagnostics under ignored `artifacts/raw/loss-checks/`:

- `denominator-dynamic.log`, `denominator-dynamic-local-corrected.log`,
  `denominator-recompile-guards.log` and `denominator-scalar-schema.log` preserve
  failed float-specialization attempts and their guard evidence.
- `denominator-dynamic-local.log` repeated the default-mode test accidentally
  because the mechanical replacement missed the formatted test call; inspection
  corrected the test. This log proves no local-dynamic result.
- `denominator-host-integer.log` reports two initial compilation callbacks with
  no count-value recompile. Its assertion requiring exactly one startup callback
  was stronger than the actual requirement. The final test records the startup
  count and requires it to remain constant across changed counts.
- `denominator-host-count-final.log` contains the intermediate five-pass native
  subset. The 26-pass final subset above covers the completed correction.

After checks, reversed only `linear-ce.patch`; core `loss.py` is restored to
SHA256 `14d546d63b069a69041ec4a379d10a06f9b6a7888f25a68dddcadf27293b7f7c`.
Both independent `git apply --check` commands pass. Scoped Ruff
`F821,F811,F841`, format checks for the three owned Python files, and
`git diff --check` pass. No other workers' source was changed.

## Cost and remaining gates

The correction adds bounded Python type/range/round-trip work per outer chunk
and passes a dynamic host scalar through the existing opaque compiled boundary.
It adds no device allocation, H2D constant, host GPU-value read, collective,
parameter or persisted state. The existing producer's collective and `.item()`
remain where they were. FP32 accumulation and the single shared dW accumulator
are unchanged. Local dynamic compilation keeps native count values out of
value guards; no global compiler configuration or workload feature changed.
The earlier FP32 dW zero-fill, internal GEMM-size, eager reduction-launch,
lifetime and full-shape memory costs still require native measurements.

The CE-only independent path is mathematically unchanged and reuses its prior
verification. The eight deselected tests exercise its unchanged general-scale
CE path and donor defect; they are not relabeled as freshly executed. Open
production, integration-review, eight-rank, performance and trace gates remain
with the coordinator. This correction closes K2's native scalar source/test
gap and does not claim those broader gates complete.
