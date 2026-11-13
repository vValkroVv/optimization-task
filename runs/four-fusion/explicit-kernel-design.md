# Explicit input and routing kernels

Design handoff, 13 November 2026. Scope: experiments 1 and 2 only. This document
supersedes the **implementation selection** in
[prefix-routing-design](prefix-routing-design.md); its producer/consumer,
distributed, state and patch contracts remain applicable. No source edits,
installation, GPU execution or performance claims accompany this handoff.

## Acceptance and selection

- [x] Inspect existing handwritten local kernels and pinned external sources.
- [x] Identify concrete interface gaps before choosing missing Triton kernels.
- [x] Specify grids, arithmetic, storage, custom-op boundaries, integration and
  checks sufficiently for separate implementers.

The source pin is `33b336a592746340d327d86e34149a23dc8688ee`. The provisional
`input_prepare.py` and `routing_metadata.py` use Inductor helpers and must be
replaced. No performance run has validated them. Do not wrap tensor expressions
in `torch.compile` for either candidate. Existing V2 model/loss compilation,
SAC, FP8, MTP, shapes and independent comparison patches stay enabled.

The problem is launch overhead in the input prefix and dense routing-metadata
construction, as attributed in [profile notes](../../docs/profile-my-ideas.md).
The selected approach is two small explicit Triton kernels: one for preparation
and one for routing, with the existing auxiliary scaling and gather-backward
semantics retained. This uses the installed stack and adds no dependency.
Single-CTA input preparation and extra routing comparisons are material runtime
tradeoffs to measure; fewer launches do not establish a speedup.

## Existing handwritten implementations inspected first

Local files below are pinned by task HEAD and SHA256, rather than an invented
upstream revision for the flattened source.

| Inspected implementation | Concrete gap and decision |
| --- | --- |
| [FLA `prepare_position_ids_kernel`](../../gift/contrib/flash-linear-attention/fla/ops/utils/index.py), SHA256 `b679166f8259224f1e7c42c7c238086544f1da1385863b57251bd663bfe47507` | Writes segment-local positions from cumulative sequence boundaries. It neither accepts the existing supplied `[B,S]` positions nor emits mask, count and MTP `[B,2,S]` targets. Its adjacent public wrapper uses a different segmented-arange path. Reuse the simple masked arange/store pattern, not the wrapper or a new boundary-conversion pipeline. |
| [FLA `packunpack_sequence_kernel`](../../gift/contrib/flash-linear-attention/fla/ops/utils/pack.py), SHA256 `f38278eb60558a64f73aa268dfcae0cfbe137fad609f72ed77c7b66a29aa187b` | Removes/inserts padded sequence elements using `cu_seqlens`; MTP packing preserves every label and adds a position plane. The wrapper also reads a GPU endpoint with `.item()`. Adapting its packing semantics would be incorrect and add unnecessary dependencies. |
| [Local TE-derived permutation](../../gift/torchtitan/ops/float8/triton_kernels/te_permutation.py), SHA256 `2b9d1502c82d6ebc051a1658d47f93e1c0850302864cfd929370e7a2e9fd8af0`; [padded permutation](../../gift/torchtitan/ops/float8/triton_kernels/padded_permutation.py), SHA256 `3df7781824f1c5c8e22a18f14b370abb9b7b22db31f1cf7e3db9b56a016f3abe` | `make_row_id_map` consumes an already dense routing map; padded permutation consumes row maps and token payloads. Neither accepts post-drop IDs/scores to produce the required two dense maps. Keep these downstream algorithms untouched. |

Narrow primary-source follow-up confirmed the same boundary mismatch:

- [TE CUDA router at `2d65f35b662c303a3850250893008eb8c3f0ad69`](https://github.com/NVIDIA/TransformerEngine/blob/2d65f35b662c303a3850250893008eb8c3f0ad69/transformer_engine/common/fused_router/fused_topk_with_score_function.cu),
  SHA256 `1de1f8b650a90f7bba47bc265b1a2f15f5b8da758ffed99892603cb3155395f9`:
  `fused_topk_forward_simple_kernel` accepts logits, computes top-k and writes
  IDs. It cannot directly consume this task's already selected, capacity-dropped
  IDs and scores. Extracting its score/top-k logic would alter the experiment.
- [TE Triton permutation at the same pin](https://github.com/NVIDIA/TransformerEngine/blob/2d65f35b662c303a3850250893008eb8c3f0ad69/transformer_engine/common/triton/permutation.py),
  SHA256 `94465e2ef66d4b1683c42f1528d48961cd907b9dde8923f34ac86eec377a432c`:
  `_row_id_map_pass_1_kernel` loads dense routing entries and computes row IDs.
  It starts after the metadata that this experiment must construct.
- [Megatron map construction at `76cf2e2d64bbfa466d50de64d036db23395dc5af`](https://github.com/NVIDIA/Megatron-LM/blob/76cf2e2d64bbfa466d50de64d036db23395dc5af/megatron/core/transformer/moe/moe_utils.py)
  remains the previously inspected tensor scatter implementation. Its fused
  routing API rejects caller-supplied `precomputed_indices`.

Exact-name searches for Triton `pack_targets`, `indices_to_map`, and
precomputed-index routing did not supply a suitable maintained handwritten
kernel. This is a bounded search result, not a claim that no such kernel exists.
These specific interface gaps justify writing the missing kernels. Preserve
license notices if copying donor code; the designs below need only standard
Triton load/store/reduction patterns.

## Experiment 1: one explicit preparation CTA

### Tensor kernel

Replace the provisional compiler helper in
`gift/torchtitan/experiments/four_fusion/input_prepare.py`. Retain its baseline
fallback and eager trainer orchestration. Export a CUDA custom op with inputs
`labels`, optional `supplied_positions`, static `generate_positions` and
`pack_mtp_targets`; return optional **fresh generated positions**, bool mask,
scalar int64 count, and optional **fresh packed targets**. The Python wrapper
returns supplied positions or unmodified labels itself when no new output is
needed; custom-op outputs must never alias inputs.

Native crop: labels int64 `[2,4096]`, generated positions int32, contiguous bool
mask `[2,4096]`, count int64 `[]`, contiguous packed targets int64 `[2,2,4096]`.
Pass label/position strides explicitly so sliced inputs need no `.contiguous()`
copy. Supplied positions retain their dtype, values, document resets, strides
and Python tensor identity outside the op; convert only the packed plane to
labels' int64 dtype, as baseline does.

Launch `grid=(1,)`, `BLOCK=next_power_of_2(max(B*S,1))`, initially
`num_warps=8` for N8192. No autotuning campaign is needed before first evidence.
For flattened offsets `i=tl.arange(0,BLOCK)`, use `i<N` load/store masks; compute
`b=i//S`, `s=i%S` only when N is nonzero. Load int64 label using strided signed
int64 offsets. Set `valid=(i<N) & (label!=IGNORE_INDEX)`.

1. Store `valid` to the contiguous output mask.
2. Generated mode uses `s.to(tl.int32)` and stores the fresh position output;
   supplied mode loads existing positions with their original strides.
3. When packing, store the original label at `(b*2)*S+s` and the position
   converted to int64 at `(b*2+1)*S+s`. Compute output products in int64 before
   pointer addition. Never shift targets or derive the mask from packed data.
4. Compute `tl.sum(valid.to(tl.int64), axis=0)` and store the scalar int64
   count. Integer reduction is exact for the crop; retain int64 rather than
   narrowing its public contract.

For empty inputs launch the same one-block kernel with one masked lane, zero
count, and no label/position/packed stores. Avoid dividing by S=0. If an input
shape would require a block larger than the supported crop bound, use the
existing eager reference for that call, visibly documented as a fallback.
Likewise retain baseline handling of CP/full-DTensor and overridden trainer
methods. This does not disable any required workload feature; the accepted
crop must demonstrably execute the explicit kernel.

### Scaling and state decision

Integration review retains auxiliary scaling in the intended target. Test a
bounded extension of the same explicit kernel: load the existing CUDA scalar
global count, convert local/global int64 counts to FP32, and evaluate two
`tl.div_rn` operations in the original order, the second with the static split
count represented in FP32. Disable FP fusion. Return a fresh FP32 share scalar;
copy it into each existing `state.scale` eagerly, preserving identity, ordering
and backward consumers. No denominator/split H2D allocation is needed.

The bounded probe identified the exact pinned ATen lowering in
`BinaryDivTrueKernel.cu` at Torch `8b353065ce260336c5fdfa74d754e8bd9b91b062`:
tensor/tensor division uses division, while division by the Python CPU scalar
uses an FP32 reciprocal, computed from a double reciprocal, followed by
multiplication. Therefore the selected arithmetic is a first `tl.div_rn`, then
FP32 multiplication by that rounded reciprocal with fusion disabled. Two
`tl.div_rn` operations failed by one ULP for count 7019/global 100000/splits 3;
the source-supported reciprocal sequence matched both intermediate stages in
the bounded probe. The implementation record must retain this source and probe.

Accept this scale fusion only after exact parity with `AuxLossTracker.weigh`,
including B2/S4096, the recorded `_splits=3` sensitivity, and multiple local/global
counts. The previous Inductor rewrite failed by one ULP. If exact parity fails,
retain original eager weighing, preserve the failure and identify scale fusion
as open rather than claim complete prefix fusion. State-copy launches are not
claimed removed in either case.

[Triton `div_rn`](https://triton-lang.org/main/python-api/generated/triton.language.div_rn.html)
provides IEEE nearest-rounded division, but matching both baseline divisions
also depends on its exact int-to-FP32 conversion and scalar-division lowering.
Do not add a separate speculative scale kernel or change rounding merely to
remove two launches. Do not create per-microbatch denominator/split H2D constants, a
persistent state framework, or return a new aux scalar to replace wired state.
The bounded same-CTA scale implementation requires the baseline bitwise rounding
test above before replacing eager weighing.

Keep attention-mask generation and its baseline ordering in the orchestrator.
For non-BLOCK_CAUSAL Decoder paths, regenerate positions even if supplied ones
exist; BLOCK_CAUSAL requires supplied positions. Keep global-valid aggregation,
clamping, transfers, callbacks, validation, MTP boundary handling and native
forward/loss/backward untouched. Preserve the existing independent
`input-prepare.patch` boundary described in the preceding design.

### Expected costs and checks

One CTA reads 64KiB labels and optionally supplied positions and writes 8KiB
mask, 32KiB generated positions and 128KiB packed targets. It performs an
integer reduction on one SM. This restricts parallelism and may lose to a
two-stage implementation despite fewer launches; native timings decide.
There is no workspace, global atomic, new collective, host data read,
synchronization or cross-microbatch cache. Scaling and state copies retain
their existing costs. Frequency remains once per microbatch/rank.

Replace compiler-helper tests with actual CUDA-kernel tests, keeping the eager
baseline as the oracle. Required cases: exact generated and supplied-reset
positions; ignored/mixed/all-ignore labels; noncontiguous inputs; empty and
sequence-edge shapes; exact dtypes, packed layout and no mutation; two changing
microbatches; aux scalar identity, begin-step reset and local regularizer
backward. Include B2/S4096 and `_splits=3` with the known rounding-sensitive
count/global values. Existing CPU orchestration tests remain useful but must
not substitute for skipped CUDA cases. Native model/loss compile, SAC, MTP and
embedding-start trace evidence remain integration gates.

## Experiment 2: directly write dense routing metadata

Replace the provisional helper in
`gift/torchtitan/experiments/four_fusion/routing_metadata.py`; preserve the
`indices_to_map(ids, weights, num_experts)` call signature and the independent
import-only patch in HybridEP.

### Kernel and storage

Inputs: signed int32/int64 IDs and FP32/BF16 weights, matching `[T,K]` shapes,
same CUDA device, arbitrary supported strides. Production T8192/E128/K8 has
distinct valid IDs per row; every retained ID is in `[0,E)`. Every negative ID
is inactive. IDs >= E and duplicate retained IDs are outside that producer
contract; do not invent semantics, add synchronized data validation in the hot
path, or silently narrow IDs to int32.

Allocate fresh one-dimensional bool and FP32 buffers of size `T*E+1`, returning
the first `T*E` entries as contiguous `[T,E]` views. Preserve this trailing sink
storage contract because existing tests check it. Outputs do not alias inputs
or each other. The sink is zero, initialized by one designated lane in CTA0.
Use `grid=(max(T,1),)`, `BLOCK_E=next_power_of_2(E)` and initially four warps
for E128. Empty T still initializes the two sinks and performs no input loads.

Each CTA owns one token and vector `e=tl.arange(0,BLOCK_E)`:

```text
route = false[E]; prob = +0.0f[E]
for k in tl.static_range(0, K):
    id = load(ids[t,k]).to(int64)
    score = load(weights[t,k]).to(float32)
    match = (id >= 0) & (id == e) & (e < E)
    route = route | match
    prob = tl.where(match, score, prob)
store bool route and float32 prob at int64(t)*E + e, masked e<E
```

The K loop uses register vectors and broadcast scalar loads, with no global
`[T,E,K]` intermediate, no output zeros kernel and no atomics. Compare/select
the actual score; do not multiply by an indicator or sum over K. This retains
zero-score routes, signed zero, and a valid NaN score while ignoring NaNs in
dropped slots. All loads, input-stride products, output offsets and sink
offsets use signed int64 arithmetic before multiplication. Unsupported huge
grids/shapes follow a documented baseline fallback selected by host metadata.

### Autograd and compiled boundary

Use `torch.library.custom_op(..., mutates_args=(), device_types="cuda")` and a
matching fake implementation allocating the same view/layout/storage outputs.
The eager body launches the explicit kernel. No nested `torch.compile` call.
The op is opaque to outer model compilation; it is also callable from the
existing opaque `hybridep::dispatch` body without changing that op's schema,
fake outputs, saved state, stream recording or communication handoff.

Register standalone autograd: save the original IDs and weight dtype; ignore
the bool output gradient and return
`gather_route_gradients(grad_probs, original_ids).to(weights.dtype)` for weights,
and no gradients for IDs/E. Use the unchanged gather helper, including negative
masking; handle an absent `grad_probs` without allocating a dense gradient.
This is the baseline scatter gradient for the distinct-ID producer. The native
HybridEP backward already gathers from `combine_bwd`; keep it unchanged and
do not gather twice. The registered backward exists for direct helper callers.

[PyTorch custom-op guidance](https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html)
supports fake/autograd registration and `opcheck`; verify the exact installed
Torch interface rather than introducing a newer API by assumption. For input
preparation use the same opaque/fake boundary, with no autograd registration
needed for integer/bool-only outputs. Its optional fake outputs must agree
with runtime allocation and must not return supplied input aliases.

### Expected costs and checks

The native crop writes 1MiB bool plus 4MiB FP32 and two sink elements. It reads
65,536 IDs/scores and performs about 8.4 million expert comparisons/selects.
Register pressure, broadcasting and instruction cost can offset the saved
zero/scatter/offset/mask launches. Inspect generated kernel resource use and
native region timing; the dense downstream interface makes these 5MiB writes
unavoidable within scope. No capacity, drop, predicate, all-reduce, FP8,
communication-stream or saved-handle change is included.

Retarget the provisional tests to the explicit op and keep baseline tests.
Required exact checks: native T/E/K; int32/int64 IDs; strided IDs/scores;
FP32/BF16 scores; zero-score route true; all-negative and empty rows; negative
NaN scores; valid NaN and signed-zero preservation by bit inspection where
appropriate; dense dtypes/contiguity and one-sink storage; standalone weighted
backward with nonuniform incoming gradients and dropped slots. Compare actual
weight gradients and unchanged `gather_route_gradients`, including base-storage
gradients for input views. Run `opcheck` plus outer compile forward/backward,
and the real FP8 HybridEP dispatch/combine/backward with dropped routes and
SAC. Mocks cannot close the last gate.

## Implementation and evaluation handoff

Workers own separate modules/tests and independent patches; do not add core
experiment flags or stack candidates. Preserve concurrent loss work. Keep
existing CPU fallbacks labeled as reference execution, never CUDA activation.
Run scoped lint and review actual signatures/call sites. Before handoff review
the final path's allocations, arithmetic, output lifetimes, launch counts and
unchanged helpers against these contracts.

Native evaluation remains the parent assignment: matched V2/candidate short
unprofiled runs, bounded matched traces, Perfetto target-region analysis and
retained-candidate local numerical evidence. Trace the named explicit kernel
before embedding/inside HybridEP to prove activation. Report target elapsed
time, kernel sums, gaps, waits and downstream start separately from TPS and
paired memory. Preserve the known Inductor rounding failure and replace its
implementation, not its evidence. No performance, GPU parity, installed-API
compatibility or production-acceptance gate is closed by this design.
