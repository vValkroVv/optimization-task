# Routing metadata implementation record

## Pre-implementation selection and cost review

**Target:** task checkout revision `33b336a592746340d327d86e34149a23dc8688ee`,
accepted V2 configuration from `docs/baseline.md`, native MoE crop
`T=8192, E=128, K=8`, Torch
`2.15.0.dev20260822+cu130` (`8b353065ce260336c5fdfa74d754e8bd9b91b062`),
Triton `3.8.0+git3f6e4113`, Python 3.12.3. The source remains in the isolated
task copy; no original checkout or baseline is changed.

**Problem and selected implementation.** `hybridep::_dispatch_impl` calls
`indices_to_map` inside its opaque CUDA custom op. The current helper builds two
dense maps through eager comparisons, offsets, masks, zero fills and scatters.
The selected kernel assigns one CTA to each token and directly writes its
`E`-wide bool and FP32 vectors while visiting the `K` scalar ID/score pairs.
This follows [the explicit-kernel handoff](explicit-kernel-design.md) after
local and pinned-source review. The local FLA position/packing kernels, TE
router/permutation kernels at pin
`2d65f35b662c303a3850250893008eb8c3f0ad69`, and Megatron map construction at
`76cf2e2d64bbfa466d50de64d036db23395dc5af` do not accept post-capacity,
preselected IDs/scores and cannot supply the required dense maps. PyTorch's
installed `custom_op`, fake registration, autograd registration and `opcheck`
interfaces were inspected directly; their signatures support the required
opaque boundary. The kernel adds no dependency and does not nest `torch.compile`.

**Affected path and preserved ownership.** Router top-k supplies distinct IDs;
capacity selection sets dropped IDs to `-1` and zeroes their score. This helper
must treat every negative ID as inactive, but a valid zero-score route remains
true. `dispatch_with_permute` receives contiguous bool `[T,E]` and FP32 `[T,E]`
views with one trailing sink element in each backing storage. HybridEP's
dispatch custom op owns the real combine-backward gather from its original IDs;
the new helper's standalone autograd registration is only for direct callers.
Capacity all-reduce, token-drop policy, host predicate, FP8 quantization,
communication stream/lifetimes, handles, SAC and the enclosing custom-op
schema remain unchanged. Positive out-of-range IDs and duplicate retained IDs
remain outside the producer contract; there is no hot-path value validation.

**Expected costs and tradeoff.** At the crop shape each call reads 65,536 IDs
and scores, performs `8192*128*8 = 8,388,608` expert comparisons/selects, and
writes 1 MiB of routing bits plus 4 MiB of FP32 probabilities and two sink
elements. It removes intermediate offset/mask/zero/scatter launches and
workspace, while retaining dense output allocation and store volume. Per-CTA
vector/register pressure and the extra expert comparisons may offset launch
savings. No host synchronization, collective, global atomic, temporary
`[T,E,K]` tensor or persistent storage is introduced. Timing and native
throughput remain unmeasured by this implementation subtask.

## Finite acceptance checklist

- [x] Explicit Triton kernel, signed int64 input/output offset arithmetic,
      fresh dense outputs and one initialized sink per output.
- [x] CUDA custom-op/fake/autograd boundaries match installed Torch; CPU path
      remains explicitly the eager reference.
- [x] CUDA behavior: production shape; int32/int64 IDs; FP32/BF16 scores;
      strided inputs; empty/all-negative/mixed/zero-score/NaN/signed-zero
      routes; exact output layout/storage; nonuniform standalone gradient and
      base-view gradient.
- [x] `opcheck`, outer `torch.compile` forward/backward, and call from the
      opaque HybridEP dispatch body.
- [x] Scoped lint, patch formatting/check, final code and inherited-path cost
      review. Record failures and unavailable gates without broadening scope.

## Results

### Implementation

`gift/torchtitan/experiments/four_fusion/routing_metadata.py` now owns a
single-CTA-per-token Triton kernel, `four_fusion::routing_metadata_cuda`, its
fake implementation, and direct-call autograd registration. The kernel reads
each `[T,K]` row with signed int64 stride arithmetic, loops over static `K`,
ORs route matches, and uses `tl.where` to select the FP32 score. It writes
contiguous bool/FP32 vectors to fresh `T*E+1` buffers and initializes each
trailing sink once from CTA zero. Empty `T` launches one CTA to initialize just
the sinks. CPU inputs and unsupported CUDA metadata shapes/dtypes use the eager
reference. CUDA shapes are bounded to `E<=1024`, `K<=64`, the CUDA X-grid
limit, and signed int64 input/output offsets. Retained IDs must be in range and
distinct per token as guaranteed by the top-k producer; no device-value check
was added.

The standalone backward saves the original IDs and returns the unchanged
`gather_route_gradients` result cast to the input weight dtype. The existing
HybridEP backward still gathers once from `combine_bwd`; no second gather was
introduced. The independent `hybridep.py` hunk is in
[`routing-metadata.patch`](routing-metadata.patch), and only switches the
`indices_to_map` import. It was temporarily applied for the dispatch-body test
and then reversed. `hybridep.py` is byte-identical to task HEAD afterward
(SHA256 `12b1921372f265ede5cca4974f5867f954f84862cc257ff8e02bb7ecafa0814d`).

### Verification

Runtime pin: task revision `33b336a592746340d327d86e34149a23dc8688ee`,
Python `/opt/torchtitan-venv/bin/python3.12` 3.12.3, Torch
`2.15.0.dev20260822+cu130` / git `8b353065ce260336c5fdfa74d754e8bd9b91b062`,
Triton `3.8.0+git3f6e4113`, CUDA build 13.0. Checks used the existing one-node
allocation `lm-mpi-job-8dff1e19-674b-42ed-96a9-bcd042dfefe5`; physical GPU 1
was an idle H100 80GB before each CUDA invocation and `nvidia-smi` showed no
compute process on that GPU. No multi-rank training or benchmark was launched.

- With `CUDA_VISIBLE_DEVICES=` and the task Python, the focused module completed
  **8 passed, 13 skipped**. The skips were CUDA-only cases. See
  `artifacts/raw/routing-checks/cpu-reference-checks-final.log`.
- With `CUDA_VISIBLE_DEVICES=1`, the routing suite excluding its separately
  patched dispatch-body test completed **19 passed, 1 skipped, 1 deselected**.
  This covered int32/int64 IDs, FP32/BF16 weights, strided inputs, `T=8192,
  E=128, K=8`, empty/all-negative/mixed rows, ignored NaNs, valid NaN and
  signed-zero bits, one-sink storage, no output aliases, standalone gradients
  and base-storage gradients. See
  `artifacts/raw/routing-checks/cuda-kernel-checks-final.log`.
- The no-grad full `torch.library.opcheck` returned `SUCCESS` for schema,
  autograd registration, fake tensor and dynamic AOT checks. The grad-enabled
  opcheck returned `SUCCESS` for schema, autograd registration and fake tensor.
  Grad-enabled default opcheck also exposed a Torch test-helper limitation in
  `test_aot_dispatch_dynamic`: installed
  `/opt/torchtitan-venv/lib/python3.12/site-packages/torch/testing/_internal/optests/aot_autograd.py`
  (SHA256
  `59cecedc00e9cd26d07fb5fcd9c317b9edaf12b807c24b93f33eb48c07646886`),
  line 157, starts its aggregate with the bool routing output's int64 sum and
  then in-place adds the FP32 probability sum, which raises a Long/Float cast
  error before the AOT comparison. No global Torch helper was modified. The
  task's separate `torch.compile(..., fullgraph=True)` forward/backward test
  passed with elementwise weight-gradient comparison. Logs:
  `cuda-kernel-checks-final-attempt.log` (failed default grad AOT helper),
  `opcheck-and-outer-compile.log` (supported opcheck checks and outer compiled
  backward pass).
- With `routing-metadata.patch` temporarily applied, the dispatch-body test
  confirmed the experimental helper import and exact maps passed into the
  buffer recorder inside `hybridep::dispatch`; it passed **1 test**. The
  recorder follows installed DeepEP 1.2.1's
  `hybrid_ep_buffer.py::dispatch_with_permute` five-output return
  (SHA256
  `63e425916b3c6b0c881ad402af35a2b00c4c84203fcde611a6715744b89210cb`).
  This is mocked buffer-boundary evidence, not real FP8 dispatch/combine.
  `git apply --check runs/four-fusion/routing-metadata.patch` passed; after
  reversing it, the core file hash matched HEAD.
- Scoped `ruff check --select F821,F811,F841 --no-cache` passed on the two
  changed Python files. `ruff format --check` passed. The patch whitespace/apply
  check passed.

The initial kernel run failed because the kernel decorator was missing; the
decorator was added, after which the native-shape and type/layout checks passed.
Two early dispatch-recorder runs failed before the outer call returned: one had
the wrong return nesting, and one returned the input hidden tensor, violating
the enclosing custom op's no-alias rule. Both were corrected from the pinned
adapter signature; the final patched dispatch-body run passed. Failed logs are
retained alongside the passing logs under
`artifacts/raw/routing-checks/`.

### Final runtime-cost review and open gates

At the production crop, the kernel reads 65,536 IDs and 65,536 scores, executes
8,388,608 expert comparisons/selects, and writes about 1 MiB of bool routing
data, 4 MiB of FP32 probability data, and two sink values. It makes two dense
output allocations and one Triton launch per helper call. It adds no
`[T,E,K]` workspace, output-zero kernel, atomic, host tensor-value read, new
collective, or synchronization. The four-warps-per-CTA vector can increase
register pressure; its comparison/select cost may offset launch savings. Actual
GPU time, memory peaks, dispatch-region elapsed time and training throughput
remain unmeasured here.

The source review followed producer uniqueness and capacity-drop `-1` IDs into
`hybridep::_dispatch_impl`, its `indices_to_map` consumer, the communication
stream wrapper, the DeepEP dense bool/FP32 inputs and the unchanged native
`combine_bwd`/`gather_route_gradients` path. Capacity all-reduce, host predicate,
FP8 quantization, handles, stream recording, SAC state and the enclosing
custom-op schema were not edited. The mocked dispatch test checks the import
boundary only. Parent-owned native acceptance remains open: real FP8 HybridEP
dispatch/combine/backward under the accepted eight-rank configuration, matched
trace evidence and performance comparison. No speedup or production acceptance
is claimed.
