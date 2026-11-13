# Explicit input-preparation kernel

Source base: `33b336a592746340d327d86e34149a23dc8688ee`. The implementation
selection and full preparation contract are in
[explicit-kernel-design.md](explicit-kernel-design.md); the earlier producer,
consumer and state map is in
[prefix-routing-design.md](prefix-routing-design.md). No native training or
performance measurement was run for this implementation task.

## Selection and implementation

The earlier Inductor helper did not guarantee one kernel and changed one
divide-by-three result by one ULP. The inspected local and pinned donors listed
in the design handoff do not accept supplied positions and labels at this
boundary while producing mask, exact count and the required MTP target layout.
The concrete gaps justify a small explicit Triton kernel without adding a
dependency. The accepted path uses one CTA, `BLOCK=next_power_of_2(max(B*S,1))`
and eight warps, with an 8,192-element crop bound; larger or unsupported inputs
use the eager reference.

The custom op reads int64 labels and explicit label/position strides. It writes
fresh generated int32 positions when requested, a contiguous bool mask, int64
valid-token count, and contiguous `[B,2,S]` int64 MTP targets when positions
exist. The wrapper returns the supplied positions by identity and returns
unchanged labels when packing is unnecessary. Attention-mask construction stays
in the existing eager order. Context-parallel/full-DTensor paths and trainer or
aux-tracker overrides keep the baseline path.

The installed Torch custom-op schema accepts an optional Tensor input but
rejects optional Tensor outputs. A collection-time probe raised
`ValueError: infer_schema ... Return has unsupported type ... Tensor | None`.
The op therefore returns fresh zero-length Tensor sentinels for inactive
generated-position/packed-target outputs; its fake implementation has the same
shapes, and the Python wrapper restores the optional-output and input-identity
semantics. These sentinels carry no data storage and are never read by the
kernel. The custom-op result retains a fixed five-Tensor schema.

### Auxiliary scale rounding

Torch is pinned by the installed runtime to git
`8b353065ce260336c5fdfa74d754e8bd9b91b062` (Torch
`2.15.0.dev20260822+cu130`). Its [BinaryDivTrueKernel.cu](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu)
uses `DivFunctor(a,b) = a / b` when both operands are tensors. For a CPU scalar
second operand, it computes a double-precision reciprocal, casts that to the
float opmath type, then multiplies. This is the actual lowering of
`AuxLossTracker.weigh`'s two divisions when the denominator is a Python scalar.
The active producer matters: `Trainer.train_step` calls `dist_utils.dist_sum`
for DP, and `dist_sum` returns Python `float` through
`distributed.utils._dist_reduce`; without DP it passes `local_valid_tokens.float()`
as a CUDA FP32 Tensor. Thus DP uses scalar-reciprocal multiplication for both
`valid_tokens / global_valid_tokens` and the later `/_splits`; non-DP uses
tensor/tensor division for the first operation and scalar-reciprocal
multiplication for the second.

The one-CTA kernel preserves both producer paths. For Python-float DP counts,
the orchestrator computes `1.0 / global_valid_tokens` in host double precision
and passes that FP32 scalar as a Triton kernel argument; no denominator Tensor
or H2D constant is created. The kernel multiplies the FP32 local count by this
reciprocal, then by `tl.full((), 1.0 / SPLITS, tl.float32)`. For a CUDA FP32
Tensor denominator it uses `tl.div_rn` for the first operation, then the same
split reciprocal multiply. All launches set `enable_fp_fusion=False`.

Pinned-source CUDA probes compared first-division intermediates and final shares
bitwise for each path across counts 7020, 7019, 8192, 4096, 1 and 0 with
denominator 2,096,640 and splits 3. The Python-float reciprocal path matched
both intermediates and shares; the CUDA FP32 tensor path matched its
tensor/tensor baseline. The prior count 7,020 / 100,000 / splits 3 and adjacent
7,019 sensitivity cases also matched with the source-matched reciprocal. Using
a second `tl.div_rn` for the tensor/Python-scalar split division still differs
at 7,019 / 100,000 / splits 3 (`0x3cbfaa5e` eager versus `0x3cbfaa5d` Triton).
Probe commands and output are retained under `artifacts/raw/input-checks/`.

The kernel returns a fresh FP32 share. The orchestrator still eagerly copies
that value into every existing `state.scale`, retaining the state tensor
identity, copy order and timing before forward. It creates no H2D constants.
CUDA regression tests compare the copied scale with a separate real
`AuxLossTracker.weigh` over six split-3 Python-float denominator pairs and test
the non-DP CUDA FP32-Tensor producer path. The native `prepare_forward_batch`
regression uses the actual DP producer type (`float`, value `2096640.0`) at
B2/S4096, asserts that the scalar custom-op path ran, and checks supplied
position identity, MTP packing and bitwise aux state against
`AuxLossTracker.weigh`. The preceding guard dereferenced `.device` on this
Python float, so this assertion covers the prior native-path failure. A second
native preparation test covers the non-DP CUDA FP32-Tensor producer.

## Runtime-cost review

Before implementation, the active baseline was traced from
`Trainer.post_dataloading_process` through `forward_backward_step`: positions
and token mask were produced before attention-mask handling; the forward block
then repeated the ignored-label comparison/count, updated aux scale state and
packed MTP targets. Global-token aggregation/clamping and the Trainer's later
forward, loss, backward, optimizer, evaluation and checkpoint paths are not
changed by the integration patch.

At native B2/S4096, the kernel runs once per microbatch (32 times per accepted
optimizer update). It reads 64 KiB of labels and supplied positions when
packing. For the DP Python-float producer, the global-count reciprocal is a
launch scalar and adds no device-memory read or transfer; the non-DP tensor
producer adds one FP32 scalar load. It writes an 8 KiB mask, optional 32 KiB
generated positions, optional 128 KiB packed int64 targets, an 8-byte int64
count and a 4-byte FP32 share. It replaces the separate positions/mask/count/
pack work and duplicate label comparison with one launch; each CTA performs an
int64 reduction on one SM. It uses no workspace, global scratch, atomics, host
scalar read, explicit H2D tensor copy, synchronization or collective. It
retains one eager `state.scale.copy_` per aux state. The single CTA limits
parallelism and could be slower than multi-stage work; no runtime benefit is
claimed without the parent's matched timing and trace evidence.

The [independent integration patch](input-prepare.patch) replaces only the prep
block in `Trainer.forward_backward_step`. Model/loss compilation, SAC, FP8, MTP,
global count preparation and all subsequent native execution stay enabled.
The temporary Trainer edit was reversed after writing the patch;
`gift/torchtitan/trainer.py` is clean in the working tree.

## Verification

Runtime used `/opt/torchtitan-venv/bin/python` (Python 3.12.3), Torch
`2.15.0.dev20260822+cu130`, Triton 3.8.0. At the device-ownership check,
physical CUDA device 0 was an idle NVIDIA H100 80 GB with no compute processes.
All CUDA tests explicitly use `cuda:0`.

Focused command, from `gift/`:

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/torchtitan-venv/bin/python -m pytest tests/unit_tests/test_fusion_input_prepare.py -q
```

Result after the native-denominator correction: **16 passed, 1 warning in
7.24 s**. The warning is the environment's
existing `pynvml` deprecation warning. CUDA coverage includes generated and
supplied-reset positions, strided label/position views, mixed/all-ignored and
empty/edge shapes, exact count/mask/layout/dtypes, no input mutation or output
aliasing, the B2/S4096 scale cases, two successive batches, aux state identity,
step reset and local regularizer backward, eager fallback over the crop bound,
both custom-op `opcheck`/outer-compile denominator modes, and the native
`prepare_forward_batch` path with the actual Python-float denominator type,
explicit scalar-op activation, supplied positions, block-mask/MTP packing and
bitwise `AuxLossTracker.weigh` state. A separate test covers the non-DP CUDA
FP32-Tensor denominator. CPU fallback and context-parallel/full-DTensor fallback
orchestration tests also pass; they are not CUDA evidence.

Checks, from `gift/`:

```text
ruff check --select F821,F811,F841 --no-cache torchtitan/experiments/four_fusion/input_prepare.py tests/unit_tests/test_fusion_input_prepare.py
ruff check torchtitan/experiments/four_fusion/input_prepare.py tests/unit_tests/test_fusion_input_prepare.py
ruff format --check torchtitan/experiments/four_fusion/input_prepare.py tests/unit_tests/test_fusion_input_prepare.py
```

All passed. In this task workspace,
`git apply --check --whitespace=error runs/four-fusion/input-prepare.patch`
passed without modifying files. `git apply --numstat` reports only the Trainer
integration hunk (4 additions, 9 deletions); the helper and tests are separate
delivered files and are not duplicated in the activation patch. The earlier
full patch is retained under ignored `artifacts/raw/input-checks/` for history.
`git diff -- gift/torchtitan/trainer.py` is empty after reversing the temporary
edit.

An earlier CUDA run failed `test_compiled_crop_shape_on_cuda` by one FP32 ULP
at count 7,020 / denominator 100,000 / splits 1: expected
`0.07020000368356705`, observed `0.07019999623298645`. The provisional
tensor-denominator path was replaced with the pinned ATen tensor/tensor
division lowering, `tl.div_rn`; the final focused run above passes. The earlier
custom-op optional-output schema collection failure and reference-helper
argument errors were also corrected before the final run.

SHA256: `gift/torchtitan/experiments/four_fusion/input_prepare.py`
`db35fc56ff947e486bd35b9fb9afc432a06ea1dd2d0edbfa10dcfc9211f2666f`;
`gift/tests/unit_tests/test_fusion_input_prepare.py`
`fae08dc21812f823c0c7af3d199c4a3a4d19ceb3dc1bbc6519cbe76383bf03f7`;
`input-prepare.patch`
`e0eea03e83e7ea88bdccd90b9c4a3927727753057df89a4843d072245602f7a0`;
`artifacts/raw/input-checks/native-denominator-parity.log`
`12b3493ef57c59d26ea994d72d4ad31a3594ee2fcac6bae81b6303af7355df95`.

The first stdin-defined Triton probe failed because Triton requires JIT source
from a Python file. The file-based probe succeeded. An initial CUDA suite
collection exposed the installed custom-op optional-output schema restriction;
the sentinel adaptation resolved it. The next test run had five reference-
helper argument errors (the test passed `global_valid_tokens` to a helper that
does not consume it); the test calls were corrected, with no kernel change.

## Remaining gates

The CUDA checks establish the explicit op's local behavior and share parity on
the installed stack. They do not establish native eight-rank model/loss compile,
SAC, FP8/MTP1 integration, kernel activation in a training trace, device/host
region timing, throughput or memory effects. The parent owns native execution,
matched traces and performance verdict; no GPU training, full training check,
or TPS claim was made here.
