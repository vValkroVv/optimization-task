# FLCE canonical parallel-wrapper fix

## Result and scope

Implemented a conservative recognition path for the canonical
`Module.parallelize()` redistribution wrapper when every configured boundary
and state placement resolves to no active mesh. The wrapper is accepted only
when the stored setup tuple matches the exact current forward object, its
original bound method belongs to the head and is exactly one of the supported
linear implementations, `full_dtensor` is false, and `local_map` is absent.
All other wrappers retain the full original forward path. This addresses the
inactive production FLCE trace diagnosed in
[`linear-ce-trace-analysis.md`](linear-ce-trace-analysis.md).

Source HEAD: `33b336a592746340d327d86e34149a23dc8688ee`.
The integration loss patch `linear-ce.patch` was applied only around the
focused GPU tests and then reversed. `gift/torchtitan/components/loss.py` is
back at baseline SHA-256
`14d546d63b069a69041ec4a379d10a06f9b6a7888f25a68dddcadf27293b7f7c`.

## Implementation and runtime-cost review

`Module.parallelize()` captures the original bound forward before optional
`local_map` handling. It stores the exact installed wrapper, original bound
forward and `ParallelDims` in one private tuple only when no local map is
configured. The adapter resolves configured placements once, during
`LinearCE.install()`, and accepts only the no-op contract. Its fallback keeps
the entire canonical wrapper. It does not whitelist wrapper names, unwrap
arbitrary callables, or bypass a configured TP/DTensor operation.

The metadata is one Python tuple per eligible configured module and contains
references only. Placement checks run once at adapter installation. This adds
no tensor allocation, device-value read, transfer, collective, kernel launch or
per-chunk mesh resolution. Parameter identity, state-dict keys and tensor values
were checked in the focused tests. As before, dimensions and sharding config
are assumed fixed after `parallelize()`; arbitrary post-install mutation does
not recompute the contract.

The selected FLCE path has the existing candidate's real runtime costs: tiled
projection/CE/dH/dW work, an FP32 dW accumulator, and the existing GFSDP
gradient-buffer lifetime. This fix does not change the custom op or measure
those costs. The first corrected eight-rank activation trace and any renewed
matched TPS/memory comparison remain required before performance acceptance.

The head/GFSDP ordering concern was checked against the active source. In
`gift/torchtitan/models/gift_moe/_base/parallelize.py`, the policy uses the
configured `param_dtype` and `cast_forward_inputs=False`. The adapter invokes
`head(...)`; GFSDP's prepended `_pre_forward` runs before the replacement
`forward`, and calls each parameter group's `pre_forward` to unshard and wait.
`GFSDPParam.init_unsharded_param()` creates the gathered parameter as a view of
the original full shape; the gather buffer uses `param_dtype` when configured.
The native adapter reads `module.weight` and runs its weight device check inside
that post-hook forward. The pre-hook selection checks apply to hidden states;
the production path supplies local CUDA BF16 hidden states and int64 labels,
with no TP input DTensor (actual head configuration is TP1, EP8). These source
facts resolve the ordering concern at source level. The single-device test does
not replace the forthcoming eight-rank activation gate.

## Regression evidence

The old-path failure was reproduced through `LinearConfig.build()`, actual
`Module.parallelize()`, GFSDP wrapping and the native loss integration patch.
It reached `assert loss._linear_ce.supported_head` and failed because the
canonical closure was not recognized. This is the expected pre-fix failure,
not a harness or import failure:

- `linear-ce-parallel-wrapper-old-fail.log`
- SHA-256: `d233e37a1f625327ac91c568f316ac8203f899f3154d2c076a73074c2bd8cdf1`

Focused CPU protocol/adapter coverage passed, including canonical no-op
positive recognition, active state/input/output mesh fallbacks using the real
`ParallelDims` resolver with a size-two TP mesh stub, full-DTensor/local-map
fallback, changed wrapper, copied contract owner, and overridden subclass:

```bash
CUDA_VISIBLE_DEVICES='' /opt/torchtitan-venv/bin/python -m pytest -q \
  tests/unit_tests/test_fusion_losses.py -k 'flce_' --tb=short
```

Result: `8 passed, 37 deselected` (`linear-ce-parallel-wrapper-cpu.log`,
SHA-256 `f9a6887d04c9c54096b6e450edf6f03834c9e9debb1408bd3ec7e0c65139746c`).
The TP case checks the real resolver against a stub mesh; it does not claim an
executed multi-rank TP run.

With `linear-ce.patch` temporarily active, the CUDA world-size-one
`TestLinearCEParallelizedGFSDP` checks passed on GPU 2, an NVIDIA H100 80GB
(UUID `GPU-2f380105-fce8-292b-f1a6-54dbea10840a`). The final command was:

```bash
CUDA_VISIBLE_DEVICES=2 /opt/torchtitan-venv/bin/python -m pytest -q -rs \
  tests/unit_tests/test_fusion_losses.py::TestLinearCEParallelizedGFSDP::test_dtensor_input_and_labels_keep_logits_fallback \
  tests/unit_tests/test_fusion_losses.py::TestLinearCEParallelizedGFSDP::test_parallelized_gfsdp_native_eager \
  tests/unit_tests/test_fusion_losses.py::TestLinearCEParallelizedGFSDP::test_parallelized_gfsdp_native_compiled \
  --tb=short
```

Result: `3 passed` in 24.69 seconds. The eager and compiled cases use the
actual canonical `parallelize()` path and GFSDP `__call__`, two outer chunks,
two microbatches, MTP weight 0.3, BF16 hidden states, a native Python `29.0`
denominator, and four selected custom-op calls. They compare each microbatch's
loss and hidden gradient, accumulated head-weight gradient, all loss
components, parameter identity and state-dict values. They start with
gradient-sync and last-backward flags false, check the first no-sync
accumulation, then sync the final microbatch and check flag restoration,
reduction state, accumulation-buffer clearing, divide factor and reshard state.
The DTensor input/label case confirms fallback retains original logits. Result
log SHA-256:
`c1f28dec986d026cce4f1c0651f8777f32ebe7e9a478a94b6f5f33ba28d80370`.

An earlier parity harness attempt treated GFSDP's `RaggedShardDTensor` as a
regular DTensor and failed in the test comparison. The test now compares its
local view; the original failed log remains at
`linear-ce-parallel-wrapper-gpu.log` (SHA-256
`ea4264765304b8362ddff55a5b3bd89ed63316891f3d242a66ac9acbf5fcec4a`). A
subsequent single-microbatch eager/compiled check passed before the final
two-microbatch test extension; its log is retained at
`linear-ce-parallel-wrapper-gpu-after-fix.log`.

Runtime used `/opt/torchtitan-venv`, Python 3.12, PyTorch
`2.15.0.dev20260822+cu130`, CUDA 13.0 and Triton 3.8.0. Selected lint and
format checks passed:

```bash
/opt/torchtitan-venv/bin/ruff check --select F821,F811,F841 --no-cache \
  torchtitan/protocols/module.py \
  torchtitan/experiments/four_fusion/linear_ce.py \
  tests/unit_tests/test_fusion_losses.py
/opt/torchtitan-venv/bin/ruff format --check \
  torchtitan/protocols/module.py \
  torchtitan/experiments/four_fusion/linear_ce.py \
  tests/unit_tests/test_fusion_losses.py
```

The final inert unified patch is
[`linear-ce-parallel-wrapper-fix.patch`](linear-ce-parallel-wrapper-fix.patch),
SHA-256 `f675510ad62573bd1cf52907f3bebe0badb742956e7b5f486a1db2cd27895e50`.
`git apply --reverse --check --whitespace=error-all` and `git diff --check`
passed. All GPU processes exited; the final `nvidia-smi` query showed all eight
H100s idle with no compute apps.

## Review findings and remaining gates

- The missing `LinearConfig` import is resolved by a local import in the test
  fixture; focused tests and F821 lint pass.
- Canonical wrapper identity/owner checks and active-placement fallbacks are
  covered. Copied metadata from another head and user-replaced wrappers fall
  back to the complete original forward.
- Native BF16 host denominator, loss components, two-microbatch flag and
  accumulated-gradient state, and DTensor input/label fallback are covered by
  focused tests.
- The actual production path has not yet been proven active at eight ranks.
  Confirm custom-op and tile/GEMM counts, reduction/reshard cadence, and memory
  lifetime in the corrected production activation trace. No speed or memory
  improvement is claimed from these tests.
