# Bounded sparse-FP8 determinism source localization

## Result and acceptance

The cause of the strict native A/A failure remains **unknown**. This pass found
no demonstrated noncanonical reduction/permutation order, uncovered read on the
supported exact layout, or new supported deterministic backend control. No
training source, recipe, flags, packages or GPU state were changed.

Finite acceptance: (1) trace the active dispatch-to-sparse-wgrad and routing-state
contracts; (2) distinguish a concrete defect from a source-level possibility and
installed binaries from other source revisions; (3) retain the failed gate and
name at most one diagnostic whose result changes localization. All three items
are complete. This is source evidence, not a runtime correctness pass.

The existing [comparison](../comparison.json) records matching float32 loss/norm
bits at update 1 and mismatches at updates 2–10. Equal scalar bits at update 1
do not establish equal elementwise gradients or optimizer updates. The existing
[backend diagnosis](../backend-determinism-diagnosis.json) remains applicable.

## Identity and limits

- Task HEAD/source configuration identity comes from
  `../source-identity.json`: HEAD `446431dfa191ef0300f116257ae918bc85e38c06`,
  GIFT upstream `6fd9a3230ab6361f11bbbf0b6ec5a5f125b78fc2`, original preset SHA-256
  `b341701bd1eef7774c110fbf7ed69cf6280ba8691100b05be457dee3e7d50a7b`.
- Installed packages were read locally under
  `/opt/torchtitan-venv/lib/python3.12/site-packages/`; no imports requiring CUDA
  initialization and no source fetch were used. `source-files.json` pins the
  inspected text bytes. These hashes do not establish that shipped CUDA source
  matches every instruction in a prebuilt extension.
- Installed DeepGEMM `_build.py` declares `2.3.0+f093a4c`. Its active implementation
  is `_C.cpython-312-x86_64-linux-gnu.so`; available generic CUTLASS headers and
  inactive `legacy/` Python implementations do not identify its chosen reduction
  algorithm. Earlier research of maintained main `5081f851...` is not proof about
  this binary. Existing smoke metadata reports DeepEP `1.2.1+1b8f467`; the installed
  Python wrapper is inspectable, but its metadata/permutation CUDA internals are
  not supplied by that package directory.

## Actual supported path

References beginning `gift/` are task source; package references below are
relative to the site-packages directory above.

1. `gift/torchtitan/models/common/moe/router.py:139–175` computes FP32 softmax,
   adds the persistent expert bias for selection only, and gathers unbiased
   scores for top-4 weights. `gift/torchtitan/ops/router.py:8–20` delegates index
   selection to `torch.topk`; no custom random tie-break is present in this
   wrapper. No evidence establishes tied scores as this failure's cause.
2. `gift/torchtitan/distributed/deepep/hybridep.py:243–413` dispatches with fused
   permutation, pad 128 and blocking exact sizing, retaining both padded counts
   and unpadded handle counts. `deep_ep/hybrid_ep_buffer.py:360–428` delegates
   metadata and permutation to native extension methods. Its Python code does
   not establish the native rank/token ordering or padding initialization.
   The fixed-capacity unused-tail case is not active: the target leaves
   `non_blocking_capacity_factor=None`.
3. `gift/torchtitan/ops/float8/triton_kernels/utils.py:36–96` builds expert IDs
   by ordered count prefixes. On exact summed padded rows every row belongs to
   an expert; the `-1` unused-capacity tail is absent. The padded groups are
   multiples of 128, matching the active requantizer's tile assumption.
4. `gift/torchtitan/models/common/moe/experts/gmm.py:278–325` calls both grouped
   GEMMs without overriding `triton_row2col`; its API defaults to **False** at
   `gift/torchtitan/ops/float8/grouped_gemm.py:288`. Wgrad therefore invokes
   `cudacherry/quantization/row2col_dq.py:23–98`, not GIFT's Triton fallback.
5. The inspected installed row2col CUDA source
   `cudacherry/csrc/quantization/row2col_dq/row2col_kernel.cu:118–294` gives each
   128x128 tile one CTA. Thread 0 computes an ordered integer group prefix;
   input/shared stores cover the tile; barriers precede consumption; each
   column amax uses a fixed five-stage warp shuffle; output tile stores and
   scales have unique owners. Its only uncovered output is an intentionally
   skipped `gid<0` capacity-tail tile. That case is inactive here. This pass
   found no uncovered read in this kernel for valid, exact padded groups.
   It cannot prove that the upstream dispatcher supplies initialized padding
   or correct group metadata, nor that the compiled binary matches this text.
6. `gift/torchtitan/ops/float8/grouped_gemm.py:204–259` requantizes both wgrad
   operands and initializes wgrad with `torch.zeros`. The custom op at
   `gift/torchtitan/ops/float8/deep_gemm_handle.py:67–105` reads already-host
   counts in the blocking target, validates alignment/size, and passes both
   CPU counts and device counts to DeepGEMM. Its empty-batch branch retains
   zero wgrad. `deep_gemm/_C.pyi:187–196` exposes no deterministic argument.
   No inspected active implementation proves an unordered floating reduction.
7. The active fused sparse SwiGLU at
   `gift/torchtitan/ops/float8/triton_kernels/fused_swiglu_quantization.py:171–450`
   writes all gate/up gradient and scale entries and all probability partials
   for the exact 128-padded row count; probability gradients reduce the fixed
   hidden-block dimension. Integer monitoring atomics do not demonstrate an
   unordered floating gradient reduction. Its temporary `torch.empty` calls
   alone provide no evidence of an uninitialized read.

## Routing counts, bias and compiled SAC

`gift/torchtitan/models/common/moe/__init__.py:88–98,160–171` initializes counts
and expert bias to zero and accumulates route counts in FP32. SAC can count a
route in forward and recomputation, as the existing comment states. The policy
in `gift/torchtitan/distributed/activation_checkpoint.py:44–86,145–181` MUST_SAVEs
router selections, HybridEP dispatch/combine and selected compiled outputs,
preserving RNG state per resolved configuration. It is not justified to assume
an additional dispatch occurs on recomputation merely because a layer is
checkpointed.

`gift/torchtitan/models/common/moe/load_balance.py:48–122` reduces counts,
computes the mean/sign bias update, centers that update, mutates bias, and zeros
counts before each optimizer step. The exact target resolves coefficient 0.001.
For one source, four accumulation microbatches, and at most two layer executions,
one expert count is bounded by `32768*4*2=262144`. Across eight distinct sources
it is `2097152`, and summed top-4 assignments across experts are `8388608`.
These integer-valued counts are exactly representable in FP32; rounding from
order of integer count addition is not established as the failure mechanism.
Different selected routes or different recomputation counts could change the
next bias, but neither occurred difference is observed in the existing logs.
Runtime verification must confirm those state values rather than blaming the
known double-count comment.

One existing deterministic control is already applied by native setup:
`gift/torchtitan/distributed/utils.py:145–151` replaces the Flex compiled callable
with `torch.compile(flex_attention)`, dropping its class-level autotuning options.
This is not a new fix and must not be proposed as an untried control for these
strict runs. Specifically the replacement does not pass `options`, removing the
original per-call `wrap_inductor_compiled_regions=True`, `max_autotune=True`,
`coordinate_descent_tuning=True` and `triton.cudagraphs=False` overrides. The
replacement inherits installed compiler defaults and process/global policy,
including the `TORCHINDUCTOR_CUDAGRAPHS`-controlled default. The launch script
does not explicitly set that variable; source/launch inspection alone does not
prove inherited environment value or replay. Both strict controls share this
override, whereas the production-policy performance runs with deterministic
debug disabled retain the original Flex options. Strict numerical evidence
cannot alone establish graph coverage or performance under the production Flex
kernel policy; eventual candidate validation must account for both paths.
Lines 129–143 set strict algorithms and disable uninitialized-memory
fill; that latter setting is not evidence that an unread buffer exists.

## One conditional next GPU diagnostic

If the coordinator authorizes a localization run after the current GPU owner
finishes, use **one genuine-path sparse-wgrad boundary replay diagnostic**:
capture immutable clones of both rowwise FP8 operands, FP32 scales, padded group
counts and m-indices for one actual layer/microbatch before its first wgrad.
Retain native model, compile, SAC, routing and FP8 recipe; do not use synthetic
balanced inputs. Replay the same installed cuDACHERRY row2col plus DeepGEMM
wgrad path twice from identical immutable operands and separately zeroed output
buffers, recording exact intermediate/output bits and input hashes.

- Different row2col output from identical inputs localizes a component failure
  before GEMM; investigate its binary/source/stream contract.
- Identical row2col output but different wgrad localizes the installed sparse
  GEMM boundary and changes the backend deterministic prerequisite.
- Both outputs identical leaves the native A/A gate failed and the cause
  unknown; it does not clear dispatch, cross-process algorithm selection,
  dense gradients, optimizer or routing-state behavior.

This is a targeted component diagnostic, not an A/A retry or training-acceptance
replacement. Preflight the extra clone/output memory against the actual layer
row count and headroom; do not shrink the model or silently exceed available
memory. No diagnostic code or GPU run was created by this source pass.
