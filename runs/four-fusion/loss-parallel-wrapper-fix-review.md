# FLCE canonical parallel-wrapper review

## Verdict and scope

The proposed source correction is conservative and addresses the observed
activation failure. **The reported test import correction is now present;
production acceptance remains open.** This is one independent,
read-only integration and runtime-cost review before source application. No
source was applied, no tests or GPU commands ran, and the concurrent CE run was
not disturbed.

Acceptance for this review: inspect wrapper provenance and no-op proof; trace
actual GFSDP weight access and loss lifecycle; assess added runtime/state costs;
identify the smallest remaining regression checks. The separate proposed design
note `loss-parallel-wrapper-fix.md` did not exist when inspected.

Reviewed source HEAD: `33b336a592746340d327d86e34149a23dc8688ee`.
Reviewed patch SHA-256:
`99c5c9d51edd046078c17ecd42a80fa6602d6edbc55fdd3fe8c98af0315c582f`.
The owner's targeted test corrections were subsequently inspected in patch
`e887d1d1d83bafdd70f12df50210355c8d687c9c10396a879ecd8afc805fc11a`.
This review concerns those inert patches, not the temporary active loss selection.
The [activation analysis](linear-ce-trace-analysis.md) owns the original failure
and invalid prior FLCE comparisons.

## Findings and dispositions

1. **Resolved in the inert patch: missing test import.**
   `_parallelized_linear_config()` returns `LinearConfig`, but neither the patch
   nor the current `test_fusion_losses.py` imports it. Every added fixture using
   this helper raises `NameError` before exercising the fix. Import it from
   `torchtitan.config.schema.models.common.nn_modules`. Reported to the owner
   and coordinator. The owner's revision adds this local import and `DTensor`;
   scoped F821 verification remains pending until the source freeze ends.

2. **No material source bypass defect found.**
   `Module.parallelize` captures the primitive before optional `local_map`, then
   installs its ordinary redistribution closure. The tuple is recorded only
   without configured `local_map`. The adapter requires exact current wrapper
   identity, bound original owner identity, and one of the two exact primitive
   implementations. It rejects full DTensor mode and active state/input/output
   meshes, using the same resolvers as the actual wrapper. Configured buffers
   share `state_shardings`, so the state check covers those too. Both positional
   and keyword inputs use the per-name source/destination union. Structured
   outputs conservatively fall back. The whole original wrapper remains the
   fallback. A changed user wrapper and an overridden primitive stay unsupported.
   Eligibility assumes dimensions and sharding configuration remain fixed after
   parallelization, as stated in the design; arbitrary post-install mutation
   is not newly supported.

3. **Required focused test additions; no new broad matrix.**
   The proposed actual `parallelize` plus GFSDP eager/compiled fixture correctly
   checks two selected calls, loss, dH, dW, parameter identity and state keys.
   The owner's revision uses integral host `29.0`, checks the selected integer
   count, and compares loss components, resolving those initial gaps. Retain an explicit old/fixed
   activation assertion through this same path: old selection is zero, fixed
   selection is two per microbatch. Add one two-microbatch accumulation case
   borrowing nondefault incoming sync/last-backward flags, checking restoration
   after the first microbatch and final accumulated dH/dW after sync. Existing
   synthetic protocol tests establish flag policy, but do not activate native
   FLCE or establish its GFSDP buffer lifecycle. Add copied/stale contract-owner
   identity negatives. The current active-mesh negative mutates a fake resolver
   after parallelization; label it as a guard unit check and add an actual
   `ParallelDims` TP-layout rejection to the existing distributed fixture when
   that fixture is run. Do not describe fake-axis mutation as executed TP.
   Existing direct-head logits-hook/evaluation/probe tests may be reused; add
   actual DTensor input/label fallback checks and confirm that one canonical
   parallel-wrapper fallback retains original logits.

## Integration and runtime-cost assessment

`GFSDPState.init` installs `_pre_forward` with `prepend=True, with_kwargs=True`.
`GFSDPState._pre_forward` applies the configured input casts, then
`GFSDPParamGroup.pre_forward` unshards, waits, and attaches input backward hooks.
The adapter still enters via `head(...)`; its first weight read occurs inside
the replacement forward, after those hooks. The outer chunk loop retains its
existing last-chunk reduction, immediate inner backward, fp32 hidden-gradient
accumulation, flag restoration and final reshard. These source facts support the
integration design; they do not replace native eager/compiled runtime evidence.

The metadata adds one Python tuple per eligible configured module and retains
references already present in the closure. Resolution and eligibility execute
once at installation, outside the custom op. The correction adds no tensor
allocation, dtype cast, device-value read, transfer, collective, kernel launch,
per-tile guard or per-chunk resolver. Existing per-chunk guards are unchanged.
State-dict/checkpoint tensors and parameter registration are unchanged;
`ModelWrapper` persists `get_model_state_dict`, not this tuple. Pipeline splitting
deep-copies the model before `parallelize_fn`, so the inspected supported path
does not deep-copy this new metadata. Whole-module pickling after parallelization
already encounters the original local closure and is not a new checkpoint
contract. Compile support remains a runtime gate for the actual native path.

Activating FLCE exposes its already documented tiled projection/dH/dW work,
fp32 dW accumulator and GFSDP lifetime costs for the first time. None of the
inactive prior traces or TPS measurements establishes those costs. After the
freeze ends: apply the corrected patch, execute scoped static/CPU checks and the
focused actual GFSDP eager/compiled regressions, then capture native production
activation first. Verify two custom-op calls per microbatch, actual tile/GEMM
counts and reduction/reshard cadence under the fixed production accumulation.
Only a passing activation gate permits a renewed matched TPS comparison.

## Verification limits

Executed: source/patch inspection only. Missing note and one mistaken exploratory
schema path were observed; the correct decoder schema was subsequently inspected.
No skipped or unavailable runtime check is counted as passing. Source freeze,
GPU numerical/lifecycle verification, production activation, timing and paired
memory acceptance remain with the coordinator. No publication is requested.
