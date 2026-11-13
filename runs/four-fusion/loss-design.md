# Independent loss-fusion designs

Read-only design judgment, 13 November 2026, starting task revision
`da6774a1c2191bfec0e1cb4aa1dc34019b3b7df8`. Applied `code-rules` and
`perf-experiments`. No training source edits, installs or GPU executions were
performed for this handoff. This document owns source selection and integration
design; the [report](report.md) owns execution and verdicts.

Finite design acceptance: inspect preserved and active implementations; pin
maintained alternatives; select an adaptation with concrete interfaces; reconcile
precision, backward, compile and head-wrapper contracts; identify scoped tests,
costs and open gates. These design items are complete. Runtime acceptance is open.

## Reconciled starting contract

Use accepted V2 with explicit prefetch depth 32 and logical epoch size 51,200,
including consumer-owned epoch entry; do not revert to the historical depth-zero
control. [Baseline](../../docs/baseline.md),
[V2 report](../v2-prefetch-baseline/report.md) and
[validation](../../planning/validation.md) own settings and methods.
The requested `docs/profile-my-ideas.md` was absent at the starting revision.
The subsequent owner-requested pull to `33b336a5` supplied the
[notes](../../docs/profile-my-ideas.md); their input/routing findings do not alter
these independent loss designs. That pull changes no training source.

Keep eight trunk layers, EP8/dense FSDP8/expert FSDP1, model/loss compile,
per-layer SAC, MTP1 weight 0.3, ordinary two sequence chunks, B2/S4096 and
accumulation 32. Preserve all other baseline settings. Experiments 3 and 4 are
alternative selectors or patches against V2; they must never be enabled together.
Positive softcap is forbidden. Detailed monitoring remains off for performance.

The recorded runtime is eight H100 80GB GPUs, Python 3.12,
PyTorch `2.15.0.dev20260822+cu130` and CUDA runtime 13.0; this is historical
evidence, not verification of the currently connected interpreter. Obtain actual
module paths, source hashes/build revision, Triton version and API signatures
from the execution environment before implementing version-dependent calls.

## Actual active path and invariants

1. `trainer.py:320–337` gives `ChunkedCELoss` the actual wrapped `lm_head` and
   sets decoder `_skip_lm_head`. Decoder training returns stacked main/MTP
   post-norm hidden states. The crop FP8 include pattern matches trunk projections,
   not `lm_head`; the head is `Unembedding(Linear(nn.Linear))` with BF16 logits,
   not an FP8 head. Do not claim preserving FP8 means quantizing this head.
2. `MTPLoss.__call__` replicates hidden states on TP, shifts MTP labels with
   packed-position boundaries, folds `[H,B,S,D]` into `[H*B,S,D]`, and calls the
   inherited chunk loop. `IGNORE_INDEX=-100` remains signed int64. MTP1 yields
   `[4,2048,4096]` hidden-state chunks and `[4,2048,128256]` BF16 logits.
3. `_chunked_loss_and_grad` makes contiguous detached sequence-chunk leaves,
   calls **`lm_head(h_chunk)`**, computes each head's CE SUM, divides each by
   the same main-head `global_valid_tokens`, multiplies by 1.0/0.3 and performs
   inner backward. Hidden gradients enter the existing FP32 `GradAccumulator`
   before the final hidden-dtype cast and decoder bridge. Keep this boundary.
4. The loop borrows FSDP reshard/sync/last-backward flags and restores them in
   `finally`; the last chunk enables gradient synchronization. The GFSDP head has
   its own gradient buffer, configured in `_base/parallelize.py:195–223`.
   Gradient divide factor is 1.0: loss normalization and reduction must not be
   replaced by an implicit mean or default FSDP AVG behavior.
5. GFSDP `_param_group.py:710–719` gathers/waits and attaches its post-backward
   function to differentiable **module inputs**. `_state.py:601–608` attaches
   pre-backward hooks to differentiable **module outputs**. Calling a kernel
   directly with `head.weight` outside `head.__call__` bypasses both contracts.
6. The crop's hidden z-loss is attached at norm input, before detached CE chunks;
   retain it there. Head-local hooks that require actual logits cannot be applied
   to scalar losses. Keep their original path when present; do not silently
   reinterpret hook inputs or move the regularizer.
7. Validation has no inner gradient; MTP eval returns main-only hidden states.
   MTP acceptance sinks require argmaxes. Shape probes use `autograd.grad` for
   hidden states without parameter accumulation. Existing DTensor vocabulary
   sharding needs its distributed CE; local-vocabulary CE is not a fallback.
8. Neither candidate adds parameters or persisted state. Preserve state-dict
   names/optimizer parameter identity, loader/RNG state and SAC around trunk/MTP
   blocks. The head/loss lives after those checkpointed blocks. Check state-dict
   key equivalence and callback/flag restoration; no checkpoint-format migration
   is proposed.

## Inspected implementation options and exact pins

| Option | Inspected identity and fit | Decision |
| --- | --- | --- |
| Bundled FLA fused linear-CE | `gift/contrib/flash-linear-attention/fla/modules/fused_linear_cross_entropy.py`, upstream `1d2281ff1b7988d8070f00dba5ffa9747ae3a60e` in [provenance](../../references/provenance.json), SHA256 `75fa652bf35850a89e53065ba2a647f0b2700eb60b18501b2e3003e335f08d7c`; already available in the stack | Reuse its token-chunk/GEMM design for experiment 3, with fixes below; unchanged code is unsuitable |
| Kindulov CE | [Manifest](../../references/kindulov/manifest.json), revision `b983803337dfcc34fdd571b38aaf5e01df583bbe`; row-kernel SHA256 `d56d7dea35d309cad25f026fc258421040fbde272cf27aec60da5b035e86fc56` | Adapt cap-zero row algorithm for experiment 4; preserve original donor bytes |
| Current Liger FLCE | [FLCE source](https://github.com/linkedin/Liger-Kernel/blob/42dc23c41dcaf1f95d858837aed5301fd4faa27a/src/liger_kernel/ops/fused_linear_cross_entropy.py) and [CE kernel](https://github.com/linkedin/Liger-Kernel/blob/42dc23c41dcaf1f95d858837aed5301fd4faa27a/src/liger_kernel/ops/cross_entropy.py), pinned from repository main on inspection | Use maintained target-gradient and FP32-addmm patterns to inform the local adaptation; avoid a new package |
| Apple CCE | [API](https://github.com/apple-aiml-research/ml-cross-entropy/blob/3de376c106a1916bc5e1b619f9c77c87a461ee1c/cut_cross_entropy/linear_cross_entropy.py), [forward](https://github.com/apple-aiml-research/ml-cross-entropy/blob/3de376c106a1916bc5e1b619f9c77c87a461ee1c/cut_cross_entropy/cce_lse_forward.py), [backward](https://github.com/apple-aiml-research/ml-cross-entropy/blob/3de376c106a1916bc5e1b619f9c77c87a461ee1c/cut_cross_entropy/cce_backward.py) at `3de376c106a1916bc5e1b619f9c77c87a461ee1c` | Technically relevant true matrix-product/CE fusion; requires more integration and stable-stat changes than local chunked reuse |
| Native PyTorch linear-CE | [Functional source](https://github.com/pytorch/pytorch/blob/f761c784cf08ce96d52ef3e56e110d289fa1a74a/torch/nn/functional.py#L3808) at `f761c784cf08ce96d52ef3e56e110d289fa1a74a`, [official API](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.linear_cross_entropy.html) | Do not assume existence in the installed development build; inspected current chunked path explicitly falls back to eager under `torch.compile`, so it does not satisfy required compiled loss unchanged |

The [CCE paper](https://arxiv.org/abs/2411.09009) provides an implemented tiled
matrix-product/LSE alternative. Its published Gemma memory/convergence results do
not predict this crop's throughput. Inspected CCE casts projected logits to the
input dtype, supports no gradient filtering and FP32 dH/dW accumulation through
`impl="cce_exact"`, and applies incoming gradients before its dlogits cast.
However, it saves a single absolute FP32 LSE and subtracts it in backward;
the large-offset stability problem is therefore relevant there too. Default
filtering/low-precision accumulation are not acceptable substitutes for exact
mode. Its public API also rejects DTensor activations and may materialize
classifier tensors; preserve head ownership instead of calling it directly on
sharded parameters.

Current Liger retains target-count `.item()` and a backward `torch.equal`
predicate, and default weight-gradient accumulation follows parameter dtype.
Its newer FP32 `addmm(..., out_dtype=torch.float32, out=...)` avoids the full
parameter-size intermediate of the legacy GEMM-plus-cast path. The CE kernel
computes centered probabilities and recomputes the target probability before
subtracting one, but still forms loss from absolute LSE. These are concrete
adaptation points, not evidence of a tested drop-in solution.

## Experiment 3: selected local FLCE adaptation

Use the bundled FLA chunked fused-linear-CE algorithm, with a task-scoped adapter
under `gift/torchtitan/experiments/` and one runner-installed selector. Preserve
contrib/reference source bytes and their licenses; copied adapted functions must
retain attribution. No new dependency installation is needed. The concrete gap
justifying adapted kernels is preservation of centered statistics and GIFT's
pre-cast gradient scaling with no device-to-host scalar reads.

Keep the **outer two sequence chunks**. Additional internal token tiles are an
implementation detail and must not change sequence chunking, MTP labels or
collective cadence. FLA's current default caps its inner chunk count at eight;
for 8,192 folded rows this means 1,024-row BF16 transient logits (250.5 MiB).
Do not present inner tiles as changing the agreed loss chunk count.

### Required local fixes

- FLA's `target.ne(ignore_index).sum().item()` is unnecessary for SUM reduction
  with GIFT's external denominator. Remove it from this selected SUM route.
- Its absolute row LSE has the same stability defect as Kindulov. Keep FP32
  maximum and FP32 log-denominator separate. Use centered loss/probabilities.
- FLA currently stores a rounded target probability to BF16 and reads it back
  before subtracting one. Form `(p-target_one_hot)` completely in FP32 before
  the single BF16 dlogits store. This fixes cancellation near probability one.
- Apply each head's `weight / global_valid_tokens` **before** that dlogits cast.
  FLA/Liger's default unit-gradient precomputation followed by scaling rounded
  dH/dW changes this rounding boundary. Passing the scale into the CE kernel is
  required for the native GIFT route.
- Accumulate all internal token-tile contributions to dW in FP32; do not silently
  accumulate BF16 partial dW. Prefer inspected `addmm` FP32-output capability if
  supported by the installed runtime. Otherwise document the fallback's FP32
  operand casts and parameter-size temporaries before measuring it.
- Avoid a device predicate for unit backward scale. Native chunk backward has
  unit upstream after known head scales have already entered CE. This restricted
  derivative-precomputation adapter must be explicit about that native contract.
  A general reusable FLCE API supporting arbitrary unknown incoming gradients
  needs backward recomputation of logits before the dlogits cast; scaling saved
  rounded dH/dW is insufficient. Do not claim arbitrary-scale equivalence.

### Integration interface

Install experiment-only functions on the selected loss/head instances, or use a
thin common overridable head/loss boundary if necessary; do not add broad
experiment branches across the trainer. Suggested boundary:
`head(input, *, ce_labels, ce_head_weights, ce_denominator)` returns differentiable
per-head scalar loss terms. The head's original `forward(input)` path remains for
evaluation/hooks or unsupported layouts. The argument name `input` must remain:
the head sharding configuration keys placements by that name.

Perform the fused call **inside wrapped `head.__call__`**, after unshard and input
hooks. Return losses whose autograd graph contains the wrapped hidden input and
head parameter so GFSDP output/input hooks and head-gradient accumulation fire.
The outer loss loop retains per-head metric accumulation, inner backward,
GradAccumulator, final bridge and FSDP flag restoration. Already-scaled losses
must not be normalized/weighted a second time by `MTPLoss._chunk_loss`.
Head metadata is task-local tensor/scalar input, not persisted module state.

Use a separately compiled callable for the fused loss work, retaining model/loss
compile settings. A fake-registered custom op with registered autograd is a
supported integration option; the [PyTorch custom-op guide](https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html)
and [Triton compile guide](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)
describe the supported boundaries. A custom op appearing in a compiled graph is
different from a hidden `torch.compiler.disable` fallback; record graph-break
and compiled forward/backward evidence.

Argmax monitoring can retain the original head path, since detailed monitoring
is off for measured runs. Preserve the ability to activate it, evaluation,
shape-probe handling and DTensor fallback. Do not reject required crop features.
If chosen patch structure cannot preserve these contracts, keep integration
open instead of measuring a narrowed workload.

## Experiment 4: stable cap-zero Kindulov adaptation

Keep the original head call, projection GEMMs, MTP loop and all gradient wrappers.
Replace only the compiled local `cross_entropy_loss` callable; retain existing
DTensor/loss-parallel fallback. Adapt donor cap-zero forward/backward into an
experiment module and explicitly compile the wrapper, rather than copying its
loss-class bypass of `self.fn` or adding unsupported head keyword arguments.

Save FP32 `maximum` and `log_denominator` separately; loss is
`-((chosen-maximum)-log_denominator)` and probability is
`exp((logit-maximum)-log_denominator)`. Preserve FP32 reduction/intermediates,
int64 labels/row offsets, FP32 scalar SUM and final logits-dtype gradient.
The backward incoming scale, including 0.3/global-valid-token normalization,
enters before the BF16 gradient store. Ignored rows have exact zero loss and
gradient. Keep first-order-only behavior explicit. Use direct Triton tracing or
a fake/autograd-registered custom op; verify both eager and compiled actual CUDA
paths. The CPU fallback remains stable PyTorch CE, not a substitute GPU test.

[K1a reconciliation](../overnight-streaming-perf/k1a-compiled-contract.md) already
proves the compiled baseline has BF16 logits/gradients and only FP32 row
statistics. Do not claim removing full-vocabulary FP32 intermediates again.
The remaining hypothesis is removing Inductor's vocabulary-wide one-hot-gradient
reduction. Ordinary donor and baseline each use three logical CE kernels; two
saved stable statistics remove the original donor's one-stat memory advantage.
Do not mix optional completed-chunk lifetime cleanup into this comparison.

## Preimplementation runtime-cost review

| Cost | Experiment 3 | Experiment 4 |
| --- | --- | --- |
| Computation | Forward projection, CE, dH and dW GEMMs remain; smaller token GEMMs may lower efficiency; forward gradient precomputation changes timing ownership | Same projection/backward GEMMs; direct CE gradient removes one-hot reduction arithmetic; row-online loop may be slower than Inductor's reduction |
| Memory/lifetime | Smaller transient logits; FP32 full-head dW is about 1.957 GiB plus BF16 final dW about 0.979 GiB and existing GFSDP buffers; do not assume a net saving | Same full BF16 logits/gradient; two row stats, small losses vector; no GiB-scale new saving |
| Synchronization/copies | Remove selected SUM-route scalar reads; new input/output casts and dW zeroing must be counted; precomputed gradients must die after inner backward | No host scalar reads; same local row passes and final BF16 gradient write |
| Communication | Preserve one head gather per microbatch and final-chunk reduction; duration shifts can alter overlap and retained gradient-buffer lifetime | No collective changes; shorter/longer backward can still shift overlap |
| Compilation/SAC | Compile adapter without graph-break fallback; no reentrant head backward inside checkpointed trunk blocks; preserve SAC policies | Compile selected CE function; SAC boundary unchanged |
| Frequency | Two outer chunks per microbatch, both main/MTP heads, 32 microbatches per optimizer update | Four logical head CE calls per microbatch, 32 microbatches per optimizer update |

Historical update-11 trace sums identify a 2,389 ms linear+CE kernel budget and
195 ms CE budget in a 14.338 s trace. These are historical overlapping kernel
sums, not current exposed elapsed-time budgets or speedup predictions. Reidentify
equivalent sites in V2. Moving GEMMs into a forward-labelled custom op is not
work removal: measure the complete head/loss forward plus required backward.

## Scoped checks and unresolved acceptance gates

1. Before performance, test activation and structural behavior through native
   head/loss calls: MTP weights, ignored/boundary targets, two sequence chunks,
   exact denominator ownership, head parameter identity, hooks, flag restoration
   on success/exception, eval and probe behavior. Exercise ordinary 32-microbatch
   accumulation and actual GFSDP head reduction in the authorized workload.
2. Mandatory defect regressions: actual CUDA eager **and compiled** CE with equal
   BF16 `2**28` logits at V128256 must give approximately `log(V)`, not zero;
   compare elementwise gradients against centered baseline and include ignored
   rows. The unchanged Kindulov kernel must fail this regression. For experiment
   3 also test near-certain target probability and nonunit known GIFT scales to
   expose rounded-target subtraction and late-gradient scaling defects.
3. Scoped numerical checks include mixed/all-ignored rows, partial vocabulary
   tiles, representative BF16 head inputs/weights, valid extreme finite logits,
   per-head component totals, hidden gradients and every head-weight gradient.
   Preserve FP32 accumulation and compare baseline/candidate to an FP32/FP64
   local reference where a difference needs interpretation. Use justified dtype
   tolerances; norms alone cannot pass gradient checks. Include nonunit incoming
   gradient for experiment 4's general CE function. Do not claim generic FLCE
   upstream scaling support from the restricted native experiment 3 adapter.
4. Run the required fresh independent 10+10 monitoring-off V2/candidate pairs and
   bounded diagnostic traces even for neutral/negative candidates. Confirm native
   kernels, compiled backward, disappearance/change of original operations,
   head-gather/reduce cadence, complete-region elapsed time, downstream start,
   host gaps/copies and buffer lifetime. Use Perfetto launch correlations and
   overlapping streams. Retained candidates require the representative matched
   head/loss replay; no whole-model gradient/optimizer export is needed.
5. Report complete-window physical-input and valid-target TPS separately,
   distributions and phase-matched memory. Repeat promising/ambiguous effects
   only when needed to resolve the verdict; preserve failures and the initial
   pair's provisional status. Final integration/runtime-cost review must reconcile
   these source findings with the actual patch and runtime evidence.

Open gates: installed API/source verification; exact experiment patch/interface;
CUDA eager/compile regressions; native GFSDP/accumulation integration; independent
performance pairs; matched Perfetto traces; retained-candidate local numerical
assessment. Neither design establishes implementation completion, speedup or
production acceptance.
