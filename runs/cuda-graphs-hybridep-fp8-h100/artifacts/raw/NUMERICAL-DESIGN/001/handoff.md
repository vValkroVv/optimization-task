# Bounded numerical and continuation design

Status: design only, 2026-11-11. No source changes, GPU execution, dependency
installation or new implementation. This is conditional work for a runnable
finalist; it adds no unconditional campaign acceptance requirement.

## Finite acceptance checklist

1. Identify the actual native boundaries for all required state, including gaps.
2. Select a separate-process, elementwise comparison route with bounded disk.
3. Specify exact comparison and full-state continuation semantics.
4. Identify genuine empty/skew diagnostics and their integration limits.
5. Estimate instrumentation costs and list the smallest implementation checks.

Scope: source snapshot upstream `6fd9a3230ab6361f11bbbf0b6ec5a5f125b78fc2`
plus the existing campaign continuation/observation extensions. Preserve the exact
3.266B/18-block/32-expert/top-4, dim-1536 preset, EP8/FSDP2, dense and sparse
FP8, SAC, compiled model and chunked loss, batch 8/global 256/sequence 4096,
four accumulation microbatches, and 2,000/30,000 schedule. No throughput result
may use the invasive numerical runs below.

## Decision: stripe one tensor family across ten native updates

Use the existing `CampaignTrainer` and native callbacks, with one opt-in
experimental numerical observer and a thin optimizer-hook binding. Keep
`Trainer.train_step`, loss, clipping, AdamW and checkpoint implementation native.
Do not build another training loop or use GraphTrainer.

Save one native **nonzero-step full checkpoint** as immutable S0, for example
the baseline's update 5. Every stripe independently loads S0 in fresh processes
and executes updates 6 through 15, with all four changing microbatches per update.
The task permits any matched saved start; it does not require updates 1 through 10.
The already required strict deterministic loss/norm A/A remains a separate gate.

Five ten-update stripes cover `preclip_grad`, `postclip_grad`, `exp_avg`,
`exp_avg_sq`, and `parameter`. The parameter stripe also compares actual update
deltas in FP64 from stored native-dtype parameters, without storing FP64 deltas.
For each stripe: baseline writes actual local tensors for ten updates; candidate
loads the same S0 and compares
every element against those files; retain result/manifest and diagnostic mismatch
slices, then delete that stripe's tensors before starting the next. The candidate
does not write a second complete tensor archive. Every run also records the same
small common input/loss/norm/routing/RNG/optimizer metadata to establish the same
trajectory across stripes. Compare these records exactly across all baseline
stripes and all candidate stripes. A mismatch invalidates combination of stripes;
do not treat independent runs as one trajectory without this check.

| Route | Native updates, excluding S0 and warmup | Process launches | Reference payload, FP32 estimate | Decision |
| --- | ---: | ---: | ---: | --- |
| Five FP32 ten-update stripes, reference then candidate; parameter stripe retains P0 once | 100 | 10 eight-rank launches | 133.8 GiB plus S0 | Recommended |
| Prefix-j, retain only update j, for j=1..10, including pre-update parameter for delta | 110 | 20 eight-rank launches | 73.0 GiB plus S0 | Fallback if disk cannot fit a stripe |
| All five families for all ten updates plus P0 | 20 | 2 eight-rank launches | 621 GiB plus S0 | Reject under tight-disk instruction |

The parameter stripe retains P0 and P1 through P10, all in native dtype. For
each tensor, calculate reference delta `float64(Pn) - float64(Pn-1)` while the
consumer calculates its actual post-minus-pre-update delta in FP64. Validate
actual pre-AdamW parameters equal the previous stored post-update parameters,
so unaccounted inter-update or callback weight mutation cannot hide in this
reuse. Release the reference and candidate FP64 delta immediately after that
tensor's comparison. No separate delta archive or extra native checkpoint is
needed. Exact equality of both endpoint parameters already implies equality
of these diagnostic differences, but explicit comparison is inexpensive once
the endpoints are available.

Prefix-j is still a genuine changing-input trajectory: each process executes
the native prefix from S0, observing only its last update. It needs ten more
native updates and ten more compile/launch cycles than striping. Family striping is simpler with
the same observer and imposes less startup/compilation cost. Neither route needs
two full training models resident on the GPUs concurrently. Report prefix
reexecution or stripe instrumentation costs as validation costs, never speed.

The table assumes all parameters, gradients and moments are FP32 and that the
sum of unique local shards across ranks equals 3.266B elements. Measure actual
bytes, dtypes, aliases and replicated copies at runtime before selecting the
route. `3.266e9 * 4 / 2**30 = 12.17 GiB` per complete family. Five ten-update
families are 608.4 GiB; adding P0 gives 620.5 GiB if all were kept simultaneously.
The largest single stripe instead has eleven parameter states, about 133.8 GiB.
A full checkpoint after Adam initialization is approximately three families,
36.5 GiB, excluding metadata and other state. These are storage
arithmetic, not measured checkpoint sizes or GPU memory peaks.

## Existing native extension points and precise observations

All paths below are relative to `gift/`.

| Required observation | Actual boundary / API | Contract and limitation |
| --- | --- | --- |
| Inputs, labels, valid-token denominator | `torchtitan/trainer.py:Trainer.train_step` (720), `after_dataloader`, `after_forward`, `before_optimizer_step` | The step fetches all four batches, counts valid labels, and reduces the global denominator before forward. Save actual input/label elements in correctness runs, not only hashes. `after_forward` runs **after backward** here; its name does not mean pre-backward. |
| Accumulated elementwise preclip gradients | `Event.before_optimizer_step`, emitted immediately before `dist_utils.clip_grad_norm_` (trainer 793) | Iterate the actual optimizer-owned named parameters after all four backwards. Preserve `grad is None` separately from a zero gradient. No autograd hook or per-microbatch gradient replacement. |
| Actual postclip gradients | `torch.optim.Optimizer.register_step_pre_hook` on each actual optimizer in `self.optimizers.optimizers` | Bind once in `CampaignTrainer.__init__`, after native construction. Clipping precedes `self.optimizers.step()`. The container's existing expert-bias hook precedes the inner AdamW pre-hook; this changes router side effects but not gradients. Hook returns `None`, only reads/copies, and is removed on close. Current callbacks alone cannot expose this boundary. |
| Returned norm and clipping coefficient | `after_optimizer_step` / `batch_end` payload `grad_norm`; `distributed/utils.py:clip_grad_norm_` (498), `_clip_grad_norm_with_ep` (579) | Keep the exact returned unrounded norm. The EP implementation combines dense/expert norms and invokes native `torch.nn.utils.clip_grads_with_norm_` separately on the two sets. Do not recompute a local norm and call it global. Derive coefficient using the **installed pinned** implementation's dtype and epsilon/clamp arithmetic; copy its actual scalar only if a justified tiny experiment hook is necessary. Verify actual postclip gradients against its multiplication, including zero/nonfinite cases. |
| Moments, step, hyperparameters, post-update parameter | `after_optimizer_step`; `ObserverState.optimizer_groups()`, `optimizer_state(param)` in `callbacks/base.py` (201,217) | `optimizer_state` exposes actual read-only state entries. `OptimizersContainer.step` (optimizer.py 187) calls the actual optimizers. Native scheduler advances before this event, so record both the LR used at pre-step and next LR afterward. Do not call `state_dict()` every update: its DCP conversion is unnecessary. |
| Actual update delta | Inner AdamW pre-hook plus `after_optimizer_step`, during parameter stripe | Save P0 once; retain native-dtype P1..P10. Keep an immutable CPU copy of candidate pre-update local parameters, validate reference pre-update endpoint reuse, then compare actual post-minus-pre differences in FP64 one tensor at a time. Store only comparison results, not FP64 delta tensors. This measures actual parameter change, not a reconstructed Adam formula. Release pre-copies after each update. |
| Auxiliary loss and weighting | `Trainer.forward_backward_step` (613), `AuxLossTracker.weigh` in components/aux_loss.py (92), `models/common/regularization.py:wire_aux_loss` (65), `aux_loss_args` (101) | Native loss payload excludes grad-only auxiliary splices. Read the already wired modules' `aux_loss_totals` and `aux_loss_scale` tensors after each actual backward; record names and differences of cumulative totals without resetting them. Record per-microbatch real-token count, global count, B/L and per-sequence mask counts. The active sequence auxiliary formula uses B, real L and top-k (`models/common/moe/losses.py:sequence_aux_loss`); it is not merely the CE denominator. Do not infer auxiliary loss from the rounded main-loss log. |
| Router side effects | `MoE.tokens_per_expert_E`, `expert_bias_E`, `models/common/moe/__init__.py` (74,96,160); `load_balance.py:register_moe_load_balancing_hook` | Read counts before optimizer mutation, compare per-layer count/bias elements and post-update zero counts. Native SAC may double count; preserve that behavior rather than divide by two. Compare identical baseline semantics and count evolution. Counts are nonpersistent; zero-at-save proof is required. |
| Exact route assignment / receive counts when needed | `BaseTokenDispatcher.dispatch_monitor`, `protocols/moe_monitoring.py:DispatchObservation`; existing `callbacks/sources/dispatcher_source.py:DispatcherSource` | The slot is the genuine dispatcher extension point, because dispatcher is not an nn.Module. Existing source handles compiled buffers and recomputation. A Python append hook can run only at tracing/capture or cause graph breaks; it is not replay evidence. Reuse existing capture-safe buffers if extra per-forward counts are needed, preserving/rejecting any existing owner. This is more invasive than reading native MoE counters, so enable only for the unresolved routing question. |

`ObserverState` deliberately has no public trainer or optimizer mutation access.
The thin campaign trainer binding is the proper place to register AdamW hooks;
do not reach into its private `_optimizers` from observer code. A weak owner
binding can follow the existing RNG continuation pattern. Normal performance
configuration must construct no numerical observer, hooks, CPU snapshots or
numerical state serializer. The existing continuation trainer need not acquire
a second copied `train_step`.

For FSDP2/EP, retain a canonical FQN and actual parameter identity mapping once
after wrapping, following `callbacks/sources/base.py:canonical_fqn` and
`OptimizerSource.fit_start`. De-duplicate aliases by identity while retaining
alias metadata. For each tensor record global/local shapes, dtype, stride,
device-mesh names/ranks and DTensor placements, parameter/expert identity, and
whether it is a parameter/gradient/state entry. Use `to_local()` for the actual
rank-local shard and copy to CPU; do not `full_tensor()`, redistribute, all-gather
or dequantize it for comparison. Compare corresponding rank/shard records
exactly; inspect full coverage once, so local equality covers the full model.
Reject unexpected wrappers or layouts instead of quietly treating them as plain
Tensors. This route is for verified target FSDP2, not an untested RaggedShard or
GFSDP generalization. Read completion of any asynchronous local tensor before
copy; CPU copies in these runs are intentionally synchronizing.

## FP8: persistent state versus transient operand scales

Inspected active implementation does not use a delayed-scaling amax history or
persistent FP8 weight cache:

- `ops/float8/cuda_gemm.py:FP8Gemm.forward` (64) quantizes current input and
  weight; backward (174) requantizes weight and grad output.
- `ops/float8/fp8_utils.py:SharedActivationQuantize` (48) quantizes the current
  activation and saves shape metadata.
- `ops/float8/quantize_expert_weights.py:_QuantizeExpertWeight` (23) uses current
  weights or quantized planes supplied by the actual dispatch.
- `distributed/deepep/hybridep.py:dispatch_tokens` (940 onward) constructs fresh
  `FP8DispatchState` carrying payload, scales/count metadata and expert weights.
  `ops/float8/fp8_tensor.py:Float8BlockwiseQTensor` owns data and scale planes.

At runtime inventory all relevant model buffers/attributes and candidate-owned
state to confirm this source conclusion. Then persistent amax/cache continuation
can be explicitly marked inapplicable with that evidence; FP8 computation itself
remains required and active. Do not invent TE-style state to serialize.

Current `ObserverState`, `DispatchObservation` and post-step callbacks do **not**
expose the transient FP8 input/weight/gradient scale planes or dispatcher metadata's
`state.fp8`. A post-update requantization would produce new scales, not observe
the planes used by the update. Source inventory and gradient equality cannot be
reported as measured elementwise equality of those transient planes.

If the finalist introduces cached/frozen scale planes, or the acceptance decision
requires their actual numerical values, this is an explicit additional source
instrumentation gate. Select only the relevant producer/consumer boundary after
the finalist is known. Any compiled observer must copy actual planes through
capture-safe, device-written experiment buffers with an explicit mutation
contract, not a Python side effect inside a captured custom op. Prove the
candidate still replays and reads changing data, and that the observation does
not change computation/code generation other than its explicit copies. Do not
preemptively add a universal FP8 tracing wrapper to close an unknown gate.

## Numerical criterion

Default detailed gate is **exact elementwise equality** for a replay-only
same-generated-kernel finalist; preserve all structural invariants exactly.
Fail on missing/extra tensors, differing mesh/shape/dtype/alias/None structure,
step/cursor/count/RNG mismatches, any nonfinite, or any unequal required element.
Enforce the task's bitwise native loss and grad-norm A/A requirement separately;
retain unrounded values/bit patterns rather than parse rounded console output.
Signed-zero/bit-pattern differences can be reported separately from numerical
equality; when the task says bitwise, compare bytes.

If exact floating equality fails, retain the affected reference and mismatch
coordinates, and replay **the same baseline** against that reference before
inventing a tolerance. This additional run answers whether the divergence is
baseline nondeterminism or candidate behavior. Report per tensor/family maximum
absolute error, relative error with an explicitly defined near-zero policy,
relative L2 error, nonfinite/mismatch counts and worst coordinates. A/A maxima
are calibration evidence, not automatically a valid tolerance: two samples do
not establish a statistical error bound. A nonzero acceptance tolerance needs
the differing operation/reduction identified, its dtype/rounding or nondeterminism
bound justified, and the criterion fixed before rerunning the candidate. Until
then the numerical gate is FAIL or INCONCLUSIVE, never an arbitrary 5% pass.
If the strict deterministic loss/norm gate itself is unsupported, keep it open;
do not use warn-only or substitute this diagnostic route for it.

The existing `scripts/loss_compare.py` is appropriate only for its supported
native loss/norm/log equality controls after command review. It does not compare
elementwise gradients or optimizer state. `agent_tooling/numerics_debugging/
compare_numerics.py` compares activation-log hashes/norm/min/max/mean and fuzzy
op matches, so it cannot replace this gate. Native DCP is the checkpoint format;
no new checkpoint format or checkpoint converter is needed.

## Disk, CPU memory, failure and retention bounds

Use standard `torch.save` per rank/update/family and
`torch.load(..., map_location='cpu', weights_only=True, mmap=True)` for references,
subject to a small pinned-runtime support test before choosing mmap. Avoid
another custom binary container. Snapshot only one family's rank-local tensors
at a time, serialize and release them before advancing. Expected CPU staging is
about 1.52 GiB/rank for one FP32 family, about 12.17 GiB across eight ranks;
the parameter stripe additionally holds its pre-update FP32 family until delta
comparison. Compute FP64 deltas one tensor at a time, adding bounded CPU
scratch rather than a full FP64 model family. Approximate combined FP32 staging
is up to 24.3 GiB across ranks if both pre/post families are fully materialized.
Actual max shard size, host available memory, filesystem behavior and dtype
support must be measured. The consumer can compare one live local tensor with
its mapped CPU reference at a time. Do not create full GPU clones.

Preflight actual bytes against an explicit 192-GiB validation-payload cap and
current free space with safety margin; this cap is a proposed disk policy, not
new permission or a guaranteed allocation. Typical family plus S0 is 158.2 GiB;
the eleven-state parameter stripe plus S0 is about 170.3 GiB. If candidate
pre-copies require files instead of host RAM, one additional temporary family
raises this to about 182.5 GiB. Serial continuation checkpoint lifetimes as
below fit the same order. Measured replication/extra buffers may change these
figures; shrink the stripe's selected update block or use prefix-j, never drop
required tensor families/updates or quantize snapshots to fit.

Write manifests atomically after complete payloads, retain source/config/runtime
pins, rank ownership, tensor keys/shape/dtype/bytes, timing/exit status and hashes.
Hashes are integrity checks, never the numerical comparison. Delete payloads
only after all named consumers and comparisons have completed and the result is
persisted; retain deletion inventory and compact worst-error records. On mismatch
retain only the necessary reference/failing slices under the same cap and stop
starting new stripes. On consumer failure keep its unconsumed reference, record
failure, and repair/retry the affected consumer. Do not delete a required resume
checkpoint because a save or load merely started.

## Full-state 5+5 continuation

After the matched ten-update comparison, use native synchronous full-state saves
and the existing `cuda_graph_campaign_continuation.yaml` overlay on both sides.
From S0 run five updates and save Q5, then continue five uninterrupted. Load the
baseline Q5 in a fresh candidate process and candidate Q5 in a fresh baseline
process, each against the correct uninterrupted reference from S0. Native step
numbers may be 10 and 15 if S0 is update 5; the contract is five plus five applied
updates. Compare loaded state before first resumed model execution and every
resumed update's required outputs/state using the same stripes or bounded
selected-update blocks. Each stripe needs only its tail reference while its
consumer runs. Do not retain all final checkpoints from all stripes.

Native `CheckpointManager.save/load`, optimizer/loader/scheduler state and the
existing rank-qualified `RNGContinuationState` own persistence. Existing-folder
`load(step=0)` explicitly sets `model_only=True` (checkpoint.py 807); avoid this
trap by using nonzero S0. For an immutable external checkpoint use
`initial_load_path`, `initial_load_model_only=false`, and a fresh nonexistent
destination checkpoint folder, since an existing destination overrides the
initial path. Native loop saves after the complete `train_step`, including
`ntokens_seen`, and emits `after_checkpoint_save` afterward (trainer.py 942).

Verify model/moments/optimizer steps, scheduler and LR used/next, loader cursor
and next four batches, CPU/Python/NumPy/current-device RNG, persistent router
bias, zero nonpersistent counts, and actual persistent FP8/candidate state
inventory. Existing RNG continuation stages on load and restores once after
lazy iterator creation plus the first deterministic fetch, before resumed
forward. Its CPU tests do not prove CUDA continuation. Check all eight ranks.
Recreate graphs in the new process and prove actual replay after restart; never
serialize CUDA graph executables. If an explicit candidate warmup changes model,
optimizer, loader, RNG or routing state, restore full state before observations;
do not assume restoring weights alone is sufficient.

Keep S0 until all stripes/continuation consumers finish. Keep each Q5 until both
its uninterrupted continuation and opposite-variant restart comparisons finish;
then delete with inventory. Run baseline and candidate Q5 directions serially
if holding both plus references would exceed the measured cap. No full step-15
checkpoint is needed solely for tensor equality: the observer can compare the
actual post-update state. Preserve the real schedule horizon in every short run
and reuse the already reviewed stop/total_steps overlay rather than invent a
new stop mechanism. Evaluation is only through an existing supported route.

## Genuine empty and skewed experts

Natural full-model batches need per-layer genuine received counts before claiming
empty/skew coverage; balanced-looking or aggregate counts do not prove it.
Existing `tests/integration_tests/token_drop_numerics.py:_check_case` demonstrates
real HybridEP transport with an empty receiver, but uses four experts/dim256 and
a manual dense expert matmul. It does **not** establish this FP8 grouped-expert,
SAC/compile path and must not be relabeled as that evidence.

If natural batches do not cover the boundary, use a clearly labelled component
diagnostic with the actual target MoE/GroupedExperts, HybridEP EP8, four local
experts per rank, dim1536/hidden1024, top-4, FP8, pad128 and comm stream. Build it
through the actual target config and wrapping path; preserve SAC and compilation
for the affected MoE boundary. In this isolated fixture, choose deterministic
router parameters/bias so the real top-k selects all four experts on one rank.
This creates 28 genuinely empty experts and seven empty receiver ranks through
the real dispatcher and FP8 grouped GEMMs. A second case changes route skew and
input values across replay, with a balanced case as control. This is a diagnostic
state, never the performance preset or a full-model acceptance replacement.

At the actual 32,768 local tokens per microbatch, maximal concentration gives
1,048,576 received assignments on one receiver across EP8; each of its four
experts receives 262,144 rows. Verify actual counts/128 padding, valid rows,
finite forward/backward, no stale scales/counts, input/router/expert gradients,
and no lost/duplicated routes. Empty experts may legitimately have zero gradients;
AdamW weight decay can still change their parameters. Do not equate zero
gradient with zero parameter update. If worst-case rows OOM or the native backend
rejects empty work, retain the actual result and open the boundary gate; do not
hide it with a smaller/balanced-only case. Lane C additionally needs its proposed
capacity's overflow/rejection/recovery behavior tested without token drops.

## Scoped implementation verification and remaining unknowns

Before any implementation, only unresolved questions that change this route
warrant checks: installed optimizer-hook order/clip coefficient semantics,
actual tensor inventory/bytes, host-memory/mmap support, and the concrete
finalist's persistent state/capture boundaries. Once a finalist exists:

1. CPU behavior tests of observer phase order and complete elementwise coverage;
   equal-norm different gradients must fail; one corrupted element/moment/update,
   missing/None/alias/shard mismatch, and nonfinite must fail. Assert callbacks
   read actual postclip/pre-AdamW tensors and do not change them. Test inactive
   mode installs no hooks and close removes every handle.
2. Test round-trip snapshot/mmap and compare on BF16/FP32/scalar/empty/noncontiguous
   tensors without reducing precision; test disk-cap refusal, partial output,
   failed consumer retention and deletion only after named consumers finish.
3. Exercise real native CPU/Gloo DTensors for local-shard coverage/metadata and
   optimizer state where supported, restoring test RNG state. These checks do
   not establish GPU/HybridEP/NCCL or FP8 correctness.
4. On the authorized target, first run one selected complete update with the
   observer and verify all rank/phase counts before launching the stripes.
   Compare observation enabled/disabled loss/norm/inputs on the same path and
   verify candidate graph replay remains active. A harmless observer can still
   change scheduling, compilation or graph lifetime; no throughput inference.
5. Execute all stripes, then both 5+5 cross-variant continuations and the genuine
   boundary diagnostics. Review integration and final runtime costs once; fix
   material findings and verify the affected behavior.

Predicted costs: one full device-to-host pass per selected tensor family per
update, serialization or CPU comparison, deliberate host/device waits, and CPU
delta arithmetic in the parameter stripe; no additional reductions/all-gathers or changes
to gradient math. Repeated launches recompiling/rebuilding graphs are the main
time tradeoff for bounded disk. Numerical observation changes schedule and may
retain graph buffers longer, so actual GPU peak and replay coverage need
measurement. This design adds no production runtime work today. Exact GPU hook
behavior, deterministic tensor equality, FP8 runtime inventory, disk byte counts,
worst-case routing feasibility and full CUDA continuation remain unexecuted.

No broad harness, training source patch or premature finalist acceptance is
needed to retain this design. Implement only after the candidate has a runnable
actual-target path and the above specific observation gaps are relevant.
