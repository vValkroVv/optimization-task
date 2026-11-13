# Input-prefix and routing-metadata design

**Implementation selection superseded by the owner's explicit-kernel
instruction.** The contract map, inspected pins and correctness/cost findings
remain applicable. The compiler-helper proposal below is historical and does
not satisfy the current assignment; new fusion implementations must use explicit
handwritten Triton/CUDA/CuTe kernels while V2 model/loss compilation stays enabled.

Read-only design, 13 November 2026. No training source, environment or runtime
was changed and no GPU computation was requested by this subtask.

## Finite acceptance and source reconciliation

- [x] Trace the two active producers, consumers and state/gradient contracts.
- [x] Inspect local/preserved implementations before current primary sources.
- [x] Select an existing mechanism and state exact implementation boundaries,
  costs, checks and unresolved gates. This document completes this design task;
  implementation and measurements remain the parent assignment's work.

Active task HEAD is `da6774a1c2191bfec0e1cb4aa1dc34019b3b7df8`; the initial
working tree was clean. GIFT is a flattened snapshot. Keep the accepted V2
prefetch32/epoch51200 workload, including consumer-owned epoch entry. Canonical
YAML still has depth-zero/default epoch settings; runner overrides, resolved
configuration and source manifests must establish the actual candidate/control.
Keep B2/S4096, accumulation32, eight layers, EP8/dense FSDP8/expert FSDP1, FP8,
CuTe, compile, SAC, MTP1, EMA, original LR and monitoring-off behavior.

Read [baseline](../../docs/baseline.md), [status](../../docs/status.md),
[candidate inventory](../../docs/candidates.md), [planning](../../planning/README.md),
[validation](../../planning/validation.md) and the accepted
[V2 report](../v2-prefetch-baseline/report.md). Historical cleanup-only wording is
superseded for the parent assignment by its explicit new objective.
`docs/profile-my-ideas.md` is absent; no source-specific claim relies on its
unavailable contents at design time. The subsequent owner-requested pull to
`33b336a5` supplied the [notes](../../docs/profile-my-ideas.md); their preparation
and metadata findings confirm the selected boundaries without changing this
design. Training source is unchanged by that pull. Re-identify target regions
from actual V2 traces.

Recorded runtime is Python3.12.3, Torch `2.15.0.dev20260822+cu130`, DeepGEMM
`2.3.0+f093a4c`, DeepEP `1.2.1+1b8f467`, NCCL2.30.7 and CUTLASS DSL4.4.1 on
eight H100s. These are existing report pins, not a fresh runtime verification.
Record actual Torch git revision, Triton version/module hashes and generated
code in each new execution receipt before interpreting a compiler candidate.

## Existing implementations and choice

The following inspection order was used: active source, preserved donors,
current official documentation and maintained implementation source.

| Option | Inspected pin/source | Fit and decision |
| --- | --- | --- |
| Existing tensor expressions + TorchInductor | Task HEAD; [torch.compile API](https://docs.pytorch.org/docs/2.15/generated/torch.compile.html), [custom-op semantics](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html) | **First implementation for both experiments.** Compile a small explicit tensor helper; outer model compilation does not cover eager trainer code or internals of opaque `hybridep::dispatch`. No package change. Inspect generated kernels rather than promise one kernel. |
| Vision runtime input path | Preserved `860152d111e4bac25daecbe3cc0cd84c68c8af9c`, [trainer](../../references/vision-runtime/source/torchtitan/trainer.py) | Eager mask/positions/preparation remains; not a fused-prefix donor. It preserves supplied positions for a different path and supports attention-mask/sample-weighting behavior absent from this task. Copying the donor wholesale changes semantics. |
| Kindulov | Preserved `b983803337dfcc34fdd571b38aaf5e01df583bbe`, [manifest](../../references/kindulov/manifest.json) | Selected source is loss/FFN work, not an implementation of either target. |
| Transformer Engine fused router | [router.py at 2d65f35b662c303a3850250893008eb8c3f0ad69](https://github.com/NVIDIA/TransformerEngine/blob/2d65f35b662c303a3850250893008eb8c3f0ad69/transformer_engine/pytorch/router.py), SHA256 `fa0f0b97de7b9f8eba3ff2a4527ff847f8c525bcd3c08183025bdba6f910ddeb` | `fused_topk_with_score_function` accepts logits, computes scores/top-k and returns dense map or indices. `topk_indices` is an output buffer, not post-drop input IDs. Cannot replace only post-drop metadata construction without moving/changing routing and capacity boundaries. Do not add TE for this experiment. |
| Megatron-LM map construction | [moe_utils.py at 76cf2e2d64bbfa466d50de64d036db23395dc5af](https://github.com/NVIDIA/Megatron-LM/blob/76cf2e2d64bbfa466d50de64d036db23395dc5af/megatron/core/transformer/moe/moe_utils.py), SHA256 `5609f56b85c55688a9fa93db0467edaacce1fe15f61461eff5ed5d5d4380f3bb` | `_build_probs_and_routing_map` uses zero+scatter/index_put, similar to task code; no dropped-ID sink. Fused routing explicitly rejects `precomputed_indices`. This is useful corroboration, not a ready post-drop kernel. |

External pins were obtained from GitHub's file-history API and their exact raw
source bytes inspected. No claim of library speed or H100 compatibility follows
from documentation. A handwritten Triton kernel is **not yet justified**: first
test native compiler lowering of the existing correct expressions. If the
compiled metadata helper still leaves material dense zero/scatter launches,
that generated-code/trace evidence would establish the concrete gap for a
single-write row kernel. Do not add that fallback before this decision exists.

## Experiment 1: GPU preparation prefix

### Contract map

1. `Trainer.train_step` collects 32 CPU batches after `after_dataloader`
   mutators. CPU int64 valid counts are stacked, summed, copied to device,
   clamped at one and reduced over the batch mesh. Leave this global count,
   callbacks, V2 iterator, transfers and their timing untouched.
2. `Trainer.post_dataloading_process` splits dictionary keys. For a Decoder,
   BLOCK_CAUSAL requires supplied document positions; otherwise the task's
   current code **always** regenerates int32 arange/repeat positions, even if
   supplied positions exist. Preserve that behavior. BLOCK_MASK construction,
   CP and full-DTensor transforms remain in their present order. The active
   workload has CP/TP/PP1; never silently bypass the other paths if supported.
3. It constructs `token_mask = labels != IGNORE_INDEX`. Immediately afterwards,
   `forward_backward_step` repeats that comparison and sums locally for
   `AuxLossTracker.weigh`. The count remains int64.
4. `AuxLossTracker.weigh` computes `valid/global_valid/_splits`, copying the
   result into every existing `AuxLossState.scale`. `wire_aux_loss` shares each
   scalar's identity with model regularizers. Retain tensor identity, update
   timing before forward, division order, dtype and step/reset behavior;
   replacing the scalar object breaks existing consumers. Aux backward records
   totals through `record_aux_loss`; SAC behavior must remain unchanged.
5. `MTPLoss.pack_targets` stacks int64 labels and positions cast to labels'
   dtype into contiguous `[B,2,S]`. `MTPLoss.__call__` unpacks them;
   `_shift_labels` preserves ignore labels and document boundaries. Decoder
   independently consumes positions and token_mask. Do not pre-shift MTP
   labels, change the global main-target denominator or count MTP targets as
   main targets. Validation has its own preparation path and remains unchanged.

### Selected implementation boundary

Put experiment code in `gift/torchtitan/experiments/four_fusion/input_prepare.py`.
Use a pure tensor helper with explicit inputs `labels, supplied_positions,
global_valid_tokens` and static position-generation/target-packing decisions.
Return positions, bool token mask, int64 local count, FP32 share and packed
targets. Generate arange directly in the desired outputs where Inductor permits;
reuse the same mask for its sum and mask output. Preserve two FP32 divisions;
do not assume compiler reassociation is numerically identical.

Compile with `fullgraph=True`, the installed Inductor backend, and
`options={"triton.cudagraphs": False}`. Keep dictionary/Decoder dispatch,
attention-mask generation and distributed transformations in an eager
orchestrator. The baseline count is after CP/DTensor transformation: a generic
orchestrator must preserve that order and local extraction, or explicitly call
the unchanged baseline for those configurations. The candidate must support
the complete required crop; do not alter any of its features.

Prefer an independently reproducible patch replacing only the preparation
block in `Trainer.forward_backward_step` with this orchestrator; retain the
native remainder of forward, loss and backward. The orchestrator must copy the
returned share into the existing aux states before model execution. Refactor
only this small preparation block, not the full training method. Keep all
experiment implementation in the experiment package. No core experiment flag
or new dependency is needed. The control applies no candidate patch.

An alternative compiler-only first step can compile the existing
`post_dataloading_process` tensor subregion and `pack_targets`, retaining
`weigh` eagerly; label it partial fusion and record the remaining division/copy
launches. Full acceptance still asks which intended operations were removed.

### Preimplementation cost review and checks

- Frequency: once per microbatch, 32 per update/rank. Main arrays are 8,192
  entries; their small size makes host launch gaps a plausible opportunity,
  not a measured speedup. New outputs should replace current outputs, without
  caching batch-dependent masks/counts/packed storage across microbatches.
- Packed targets are 128KiB at B2/S4096/int64; token mask is 8KiB and int32
  positions 32KiB. Avoid a second packed allocation, count D2H reads, extra
  collectives, pinned-transfer changes or persistent GPU buffers.
- Check exact positions, mask, int64 labels/count and packed targets against
  baseline for generated/document-reset positions, ignore-only and mixed
  labels, noncontiguous labels and sequence-edge cases. Do not change baseline
  all-ignore global clamping or scale semantics.
- Check FP32 share against the existing division order, aux scalar identity,
  two successive microbatches with different counts, begin-step reset, and
  an actual local regularizer backward with matched incoming gradient/state.
  Compare eager/compiled helper results; no full-model gradient export.
- A native crop diagnostic must show the helper's generated kernels before
  embedding all-gather, removal of duplicate comparison/work, unchanged scale
  consumers, MTP packing and required model/SAC/compile execution. A fake-only
  trainer test does not close native integration.

## Experiment 2: HybridEP routing metadata

### Contract map

`TokenChoiceTopKRouter` obtains distinct top-k IDs and scores. Dispatcher's
`_drop_routes` calls `select_capacity_routes`, then replaces dropped IDs by -1
and multiplies scores by keep. Capacity code includes int64 counts, group-gate
all-reduce and a host/device predicate branch. **All remain unchanged.**

`hybridep::_dispatch_impl` calls `indices_to_map` inside its opaque CUDA custom
operator. It builds contiguous bool `[T,E]` routing and FP32 `[T,E]` probabilities
from `[T,K]` IDs/weights. Negative IDs are inactive; a valid zero-score route
still has a true routing bit. Existing offset arithmetic casts IDs to int64;
all negative IDs share one trailing sink. Positive out-of-range IDs are not a
supported producer output; do not silently reinterpret them as valid dropped
IDs. Repeated valid IDs are excluded by the top-k producer; arbitrary duplicate
scatter write ordering must not become an invented API contract.

`_launch_on_comm_stream` records tensor lifetimes and makes the communication
stream wait for metadata. `dispatch_with_permute` consumes the bool/FP32 dense
layout; subsequent FP8 quantization, handles, padding and counts remain intact.
`_dispatch_backward` calls `combine_bwd`, then
`gather_route_gradients(dense_grad, original_topk_ids)`; dropped slots get zero.
SAC saves the same custom communication op results. Native gradient ownership
does not use autograd through metadata construction inside the custom-op body.
Standalone `indices_to_map` tests do exercise scatter autograd, however.

### Selected implementation boundary

Create `gift/torchtitan/experiments/four_fusion/routing_metadata.py`, exporting
an explicitly compiled callable with the exact `indices_to_map(ids, weights,
num_experts) -> (routing, probs)` interface. Start by compiling the unchanged
sink/scatter helper; retain the eager reference for tests. This preserves
standalone autograd and avoids inventing duplicate-write semantics.

The independent candidate patch changes only the callable imported by
`gift/torchtitan/distributed/deepep/hybridep.py`. Keep the enclosing custom-op
schema, fake registration, setup_context, autograd, stream handoff and all
capacity logic untouched. Explicit helper compilation is required because
the outer compiled model cannot inspect this opaque body. Use fullgraph and
CUDA Graphs disabled; verify compilation works from the real custom-op body
under the installed Torch version before larger runs.

If that specific call is incompatible with nested compiler execution, retain
the first failure and reassess; do not fix it by moving metadata into outer
model tracing or disabling model compile/SAC. An independently compiled
callable can be prepared before training, but its real inside-custom-op
execution remains a gate. A direct Triton alternative requires the concrete
lowering gap noted above and independent review.

### Preimplementation cost review and checks

- T8192/E128/K8 writes 1MiB bool plus 4MiB FP32 metadata per invocation; these
  dense outputs remain required by HybridEP. It reads 65,536 IDs/scores. Seven
  routed trunk layers plus any routed MTP path and SAC recomputation determine
  actual frequency; count native occurrences, rather than assuming seven only.
- Compiler fusion may remove temporary offsets/masks/casts and launch overhead.
  Dense output allocation/zero/scatter work may remain. Do not replace it with
  broadcast `[T,E,K]` comparison: that multiplies intermediate/work volume and
  is not justified by current evidence. Preserve the one sink element and
  avoid cloning entire dense matrices.
- Reuse `test_token_drop_hybridep.py` tests for zero-score selected routes,
  empty T, all-drop rows, expert0 protection, bool/FP32 output and contiguous
  layout/sink allocation. Extend eager/compiled comparisons with signed ID
  widths, noncontiguous inputs and supported FP32/BF16 score inputs; check
  negative ignored routes cannot inject NaN scores into valid metadata.
- Test standalone weights gradients and `gather_route_gradients` exactly.
  Then execute real FP8 HybridEP dispatch/combine/backward with baseline top-k
  inputs and matched incoming gradients, including dropped routes. Preserve
  handles and stream lifetimes; fake/CUDA-free extension mocks do not close it.
- Native trace activation must identify compiled metadata kernels *inside*
  `hybridep::dispatch`, unchanged capacity collective/predicate costs and
  earlier dispatch availability if observed. Keep mask/probability fill time,
  comm waits and dispatch elapsed time separate.

## Required parent evidence and open gates

### Independent patch ownership and minimal integration hunks

Experiment1 owns `experiments/four_fusion/input_prepare.py`, its focused tests,
and `runs/four-fusion/input-prepare.patch`. Its only core hunk imports
`prepare_forward_batch` and replaces `forward_backward_step`'s consecutive
`post_dataloading_process`, local count/`aux_loss.weigh`, and `pack_targets`
block with the following call (the helper returns already packed labels and
updates the existing scale tensors):

```python
inputs, labels, extra_inputs, extra_kwargs = prepare_forward_batch(
    self, input_dict, labels, global_valid_tokens
)
```

Experiment2 owns `experiments/four_fusion/routing_metadata.py`, its focused
tests and `runs/four-fusion/routing-metadata.patch`. Its only core hunk changes:

```python
from torchtitan.distributed.deepep.routing import gather_route_gradients, indices_to_map
```

to two imports: the unchanged `gather_route_gradients` from the same baseline
module, and `indices_to_map` from the experiment module. Keep the experiment
module's eager reference import distinct to avoid recursion.

Workers may share only an empty experiment-package initializer; neither
worker edits another candidate's module or patch. Save the reviewed integration
patch, then restore core integration to V2 before handoff. The parent runner
applies exactly one candidate patch serially and restores it afterwards. Freeze
source and record hashes throughout each native run. Unapplied experiment files
have no active imports in the V2 control; never stack the integration patches.

Each candidate is V2 plus only that candidate. Preserve source and independent
patch hashes, resolved config diff, actual activation and generated kernels.
Run the authorized native short pair (10 warmup +10 measured full updates),
bounded matched diagnostic traces (at most three complete microbatches on
ranks0/1), local numerical checks for retained candidates and final independent
integration/runtime-cost review. Reuse the existing runner and unprofiled timer
boundaries. Report input/valid-target TPS separately and memory at matching
phases. Repeat only to resolve an actual promising/ambiguous/noise decision.

Report region elapsed time, kernel sums, host launch gaps, stream waits/copies
and communication separately; determine whether embedding/dispatch starts
earlier or another dependency absorbs the saving. Neither fewer kernels nor
faster helper-only replay proves higher training throughput.

All implementation, installed-compiler compatibility, native activation,
GPU/distributed correctness, local precision and performance gates remain open.
No candidate speedup, retain verdict or production readiness is asserted here.
