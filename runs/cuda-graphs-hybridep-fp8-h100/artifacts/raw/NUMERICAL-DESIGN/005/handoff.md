# Minimal native inventory and same-run decoder probe

Design handoff, 2026-11-12. No source edits, execution, checkpoint creation,
installation, cleanup or publication. Uses code-rules, perf-experiments and
titan-train-job. This document has no production runtime effect.

## Decision and acceptance

Implement one opt-in experiment observer with thin binding in the existing
campaign trainer. Reuse native training, checkpointing, clipping and fused
AdamW. Observe one real update from a nonzero native S0; decode every local
parameter and its state against hashes from **that same execution**. This
verifies an encoding, not independent A/A equality or numerical A/B acceptance.

Finite acceptance for the implementation:

1. Eight complete rank inventories and exact native S0 restoration records.
2. Four actual backwards, one native clip/optimizer/scheduler update, actual
   preclip payload and actual postclip/P/m/v/step hashes, with structural state.
3. Same-runtime CUDA singleton decoding matches every same-run hash exactly.
4. Bounded retention, explicit failures and no observer work when absent.
5. Focused CPU regressions and scoped integration/runtime-cost review; GPU
   execution remains a separate authorization and acceptance gate.

Source007 stays frozen: [manifest](../../SOURCE-SNAPSHOT/007/manifest.json)
SHA256 `2585e504350276cb4689ae9ea45a33706e9f898bb61969abe5d08f2650c55620`,
92 changed paths, repository base `446431dfa191ef0300f116257ae918bc85e38c06`,
upstream GIFT `6fd9a3230ab6361f11bbbf0b6ec5a5f125b78fc2`.
An implementation needs a new authorized source revision; do not modify this
pin in place. The attribution runner remains on HOLD for its four lifecycle
fixes. Do not compose it into this probe or run concurrent GPU diagnostics.

Keep the exact 3B MoE/FlexAttention/HybridEP EP8, dense FSDP8/expert FSDP1,
FP8, SAC on all 18 blocks, model/loss compilation, batch8/global256/S4096,
accum4, seed42, c4_test and warmup2000/horizon30000. Both controls retain
`mix_order_reduction=0`, `graph_partition=True`, all18 model blocks excluded
from graphs but compiled, and the common step marker/dense adapter/scan fix.
Only the agreed global graph control changes; CE graphs are the candidate.
The minimal encoding probe uses graph-off once. Graph-on observation/replay
inertness is a later gate; do not require an independent bitwise A/A run here.

## Existing interfaces and implementation ownership

Paths below are relative to `gift/`. Read the existing methods; do not copy
`Trainer.train_step` or add experiment branches to core code.

| Owner / inspected symbol | Exact use |
| --- | --- |
| `torchtitan/experiments/cuda_graph_campaign/trainer.py:CampaignTrainer.__init__`, `_bind_rng_continuation` | After `super().__init__`, optionally find/bind the new observer. Bind native inner optimizer hooks here; `ObserverState` has no public optimizer object or trainer. Absence must construct no numerical hooks, serializer, copies or new synchronization. Remove owned handles through observer close, including partial-construction failure. |
| `torchtitan/trainer.py:Trainer.train_step` (720–823) | `before_optimizer_step` follows all four `forward_backward_step` calls and their backwards; capture accumulated gradients here, before native clipping. `after_forward` is after backward. `after_optimizer_step` follows native optimizer **and scheduler**. `batch_end` follows `ntokens_seen.add_`. |
| `torchtitan/callbacks/base.py:ObserverState.optimizer_groups`, `optimizer_state` | Read actual parameter references, inner type, copied hyperparameters and native state entries; do not use private `_optimizers`, metric reductions, or DCP conversion for per-update tensor observation. |
| `torchtitan/components/optimizer.py:OptimizersContainer.step` | Container pre-hooks run before its body; each inner AdamW pre-hook then sees actual postclip gradients. Register inner hooks after construction without replacing existing hooks; return `None`. |
| `torchtitan/models/common/moe/load_balance.py:register_moe_load_balancing_hook` | Existing **container** pre-hook updates bias using collective counts and zeros counters before inner AdamW. Record counts at preclip and actual bias/reset state at inner pre-hook; never reexecute this hook in decoding. |
| `torchtitan/distributed/utils.py:_clip_grad_norm_with_ep` (579–648) | Native dense/expert norm reductions and combined total remain untouched. Native `clip_grads_with_norm_` is called separately on expert/dense sets with the same total. No new norm reduction, clipping wrapper or coefficient API is needed. |
| `torchtitan/experiments/cuda_graph_campaign/continuation.py:RNGContinuationState`, `CampaignRNGContinuationObserver` | Keep rank-qualified checkpoint keys and deferred one-shot restore. A small **proposed** copy-only accessor for the validated pending snapshot can expose loaded RNG bytes to the probe; it must not consume/restore them. Existing `state_dict()` captures current RNG, not pending loaded RNG. |
| `torchtitan/experiments/cuda_graph_campaign/trainer.py:CampaignTrainer.batch_generator` | Native iterator creation and first fetch precede RNG restore; first `after_dataloader` runs after restore. Record loaded pending RNG at `after_load`, then exact restored RNG at first `after_dataloader`, before forward. Do not restore at `fit_start` or consume an extra batch. |
| `torchtitan/trainer.py:Trainer.train`, `state_dict`; `components/checkpoint.py:CheckpointManager` | Load precedes `after_load` and `fit_start`. Save follows completed `train_step`, before validation; synchronous `after_checkpoint_save` is the completed S0 boundary. Trainer owns step/ntokens; native checkpointer owns loader/scheduler/callback state. |

Put proposed `numerical_probe.py`, its decoder/entry, typed callback schema,
focused tests and maintained usage beside the existing experiment. Reuse
campaign trainer selection and registry conventions. A new callback type and
its output-directory setting are **new implementation**, not existing fields.
Keep the probe fixed to one observed update rather than adding a general
recorder framework. The thin binding may provide copy-only snapshots of native
loader/scheduler/callback state because `ObserverState` does not expose them.
Do not serialize the whole checkpointer state every event. Native checkpoint
format, optimizer, core trainer, distributed utilities and model defaults stay
owned by their existing code and need no changes for this route.

The **proposed, not existing** minimal callback schema is
`type: campaign_numerical_probe`, `mode: create_s0|observe|decode`, and
`output_dir: <fresh evidence directory>`, plus `input_dir` required only for
decode and pointing to the observe output; there is no enable flag.
Reject other modes, wrong absolute stop steps, non-eight-rank target settings,
and existing output files. `create_s0` records the completed step1 checkpoint
boundary; `observe` exports loaded step1 and captures step2 only. The decoder
entry loads this same typed native config with `mode: decode`, initializes only
rank/device ownership, then reads the fixed probe output schema; it must not
call `config.build()` or construct a trainer. Reuse native config loading, not
a new argument parser. Paths to S0/observation are recorded in the probe manifest
and native `checkpoint.initial_load_path`; do not infer them from directory scans.

For exact RNG comparison, capture the actual copied dictionary returned by
`RNGContinuationState.state_dict()` during the native S0 serialization, through
an opt-in experiment-owned binding, and retain it in the producer's small record.
Do not substitute a later `state_dict()` call at `after_checkpoint_save`:
that samples current RNG again. On load compare the validated pending payload
to this saved record, then verify its one-shot application after first fetch.
The binding observes serialization output and must restore its original callable
on close; it cannot alter the returned bytes or add a random draw. This is tiny
probe-only integration at the existing experiment owner, not a new checkpoint
format or a serializer replacement.

### Gradient readiness

Installed PyTorch is `2.15.0.dev20260822+cu130`, git
`8b353065ce260336c5fdfa74d754e8bd9b91b062`; repin actual loaded files/library
identities on every rank before executing. Installed
`distributed/fsdp/_fully_shard/_fsdp_state.py:_root_post_backward_final_callback`
(401) finalizes groups on the last backward and orders the compute stream after
retained reduce-scatter events. `_fsdp_param_group.py:finalize_backward` (790)
waits post-reduce events; CPU offload has its own synchronization. The selected
non-PP trainer does not defer synchronization with `set_is_last_backward` or
`set_requires_gradient_sync`. Do not change these policies to capture gradients.

At the native preclip event, use detached local reads under `torch.no_grad()`;
call `DTensor.to_local(grad_placements=original_placements)` and explicitly
`.wait()` if the result is `AsyncCollectiveTensor` (documented by installed
`tensor/_api.py:DTensor.to_local`, 645). A blocking compact CPU copy, preceded
by a deliberate probe-only device drain, closes outstanding device writes.
Do not call private FSDP finalizers, start a side-stream copy without dependencies,
all-gather, redistribute or alter gradient storage. This drain cannot repair a
missing producer dependency: the separate CUDA lifecycle defects remain gates.

## Native checkpoint route and minimal process sequence

Existing supported base is
`configs/experiments/cuda_graph_campaign_continuation.yaml`, importing the exact
3B preset and selecting `trainer.type: cuda_graph_campaign` plus
`callbacks.campaign_rng_continuation.type: campaign_rng_continuation`.
Existing `train_scan_fix.main` installs the campaign scan fix then delegates to
native `torchtitan.train.main`. Use that entry and the existing canonical dense
adapter environment route; do not use the HOLD combine-backward entry.

Required supported fields for the probe overlay/CLI:

```yaml
checkpoint:
  enable: true
  interval: 1
  async_mode: disabled
  keep_latest_k: 0
  create_seed_checkpoint: false
  initial_load_model_only: false
  initial_load_in_hf: false
  last_save_model_only: false
  save_in_hf: false
  last_save_in_hf: false
  exclude_from_loading: []
lr_scheduler:
  total_steps: 30000
  warmup_steps: 2000
```

`training.steps` is an **absolute stop step**, not a resumed update count.
The explicit scheduler horizon preserves the supplied schedule under short
stops. Preserve all other preset fields and existing callback ordering; do not
enable the timing observer whose window requires extra updates.

Use three serial eight-rank invocations, two real optimizer updates in total:

1. **Create S0:** fresh unique dump/checkpoint directory, no initial checkpoint;
   `training.steps=1`, `checkpoint.load_only=false`. Run one real native update
   and synchronous full native `step-1` save. Record exact completed state and
   all-rank hashes. This initializes actual Adam moments/steps. Step0 is unsuitable:
   the native existing-folder route treats `load_step=0` as weights-only.
2. **Observe:** fresh output directory whose configured checkpoint folder does
   **not exist**, `checkpoint.initial_load_path=<S0/step-1>`,
   `checkpoint.load_step=-1`, `checkpoint.load_only=true`, `training.steps=2`.
   Existing `CheckpointManager.load` (739–836) then performs full initial native
   load; an existing checkpoint folder would override `initial_load_path`.
   Require restored step1, full state keys, P/m/v/step/group matches and pending
   RNG match before proceeding. At `after_load`, export compact S0 local P/m/v
   **one canonical parameter and its families at a time**, preserving metadata;
   this small-probe sidecar is deliberately temporary. Observe actual step2,
   write one preclip set plus hashes/records, then exit without saving step2.
3. **Decode:** separate opt-in decoder entry, eight rank-local CUDA devices on
   the same pinned runtime. No trainer/model construction, forward, backward,
   loader, router hooks, graph capture or training RNG use. Stream one canonical
   parameter's S0 P/m/v/step and gradient at a time, verify native clipping and
   fused AdamW, release it before the next parameter. Collect only compact
   completion/status metadata across ranks; missing or failed rank invalidates
   the result. Do not create expanded state archives.

For process1, checkpoint `keep_latest_k=0` disables automatic purging; it does
not authorize indefinite retention. For process2, `load_only=true` prevents
the native final-step save as well as interval saves. No seed checkpoint,
second model, repeated training prefix or scalar substitute is needed.

## Actual records and decoder

Canonical key is `(part_index, canonical_fqn, global_rank, family)` plus update
and phase. Use `callbacks/sources/base.py:canonical_fqn`; enumerate
`named_parameters(remove_duplicate=False)`, map by actual optimizer parameter
identity after wrapping, retain all original names and reject canonical-name
collisions. Record group/index/order, global/local shape, native dtype, strides,
storage offset, mesh axes and rank list, placements, requires-grad and None/empty
status. Distinguish object aliases, overlapping storage views and replicas;
do not deduplicate replicas as aliases or collapse distinct optimizer owners.
Unexpected overlap/layout/wrapper is an explicit unsupported-inventory failure.
Serialize compact logical bytes; reconstruct layout with native strided storage,
not a view that accidentally serializes a larger backing allocation.

At preclip persist actual accumulated gradients/None state and P endpoint hash,
actual router counts, groups/LR used and optimizer state keys/steps. At each
inner optimizer pre-hook hash actual postclip gradients and record router bias
and zero/reset counts. At post-update hash actual P, exp_avg, exp_avg_sq and
step; record next LR separately. At `batch_end`, record native unrounded summed
loss, ntokens, loader/RNG/scheduler/callback state. Record actual four inputs,
labels and extra fields after mutation, per-microbatch loss, valid-token
normalizers and aux tensors/scale/names without resetting native accumulators.
Keep logical byte hashes domain-separated by identity, dtype, shape and phase;
layout/alias metadata is checked separately. None is never encoded as zeros.

Decoder for each active parameter:

1. Recreate its native local P/g/m/v layouts and actual step tensor dtype/device
   from S0 sidecar. One canonical parameter bundle is live: AdamW necessarily
   needs its P, gradient and both moments together; no second model or second
   parameter bundle is staged. Apply `torch.nn.utils.clip_grads_with_norm_`
   with the **recorded actual returned total_norm tensor**, actual `max_norm`,
   `foreach=True`. Never recompute a norm from this subset.
2. Check decoded postclip hash before optimization. Installed `nn/utils/clip_grad.py`
   (120–181) performs native `max_norm/(total_norm+1e-6)`, clamp and foreach multiply.
   This is the exact native decoder route; no observed-coefficient API exists.
   A separately computed coefficient may be recorded as **derived**, never
   mislabeled as an observed internal scalar. Require same dtype/device and
   scalar operation policy; actual postclip hash is the acceptance oracle.
3. Construct the actual observed AdamW type with recorded supported group
   hyperparameters (`fused=True`), install only this parameter's actual native
   S0 state, assign its reconstructed gradient, and call native `.step()` once.
   `optim/adam.py:_fused_adam` increments native steps and dispatches
   `torch._fused_adamw_`; do not reproduce its arithmetic in Python or on CPU.
   No container/router/scheduler hooks are replayed. Hash P/m/v/step and compare
   with actual step2. None-gradient parameters retain state/step unchanged.

Inventory all group defaults and state keys; reject unexpected AMSGrad,
grad_scale/found_inf, complex parameters, optimizer hooks that mutate state,
unsupported layouts or hidden histories instead of silently dropping them.
Different singleton grouping/alignment versus the real optimizer remains an
exact-hash gate, not an assumed equivalence. Any mismatch stops encoding
acceptance with rank/FQN/family/phase evidence; no tolerance or residual fallback
belongs in this minimal probe.

Inventory model persistent/nonpersistent buffers, plain tensor attributes,
FP8 wrapper payload/scale planes, aux aliases, router state and dispatcher-owned
state reachable at these boundaries. Reuse [003](../003/handoff.md)'s source
finding of current-operand FP8 quantization; do not invent amax histories.
Identify immutable reconstructible caches separately from mutable numerical
state. Inventory is bounded to reachable boundary state: ephemeral FP8 planes,
compiled-closure internals and graph-private buffers not exposed here are **not
measured**. Any necessary persistent state lacking a checkpoint/reconstruction
route blocks scaling. No universal tensor walker or FP8 tracer is authorized.

## Retention, costs and later gates

Parent's fresh free-space observation is 7,200,438,747,136 bytes; the previous
12-GiB observation is superseded. The coordinator's newer 05:49:51 UTC statvfs
reports **7,195,643,084,800 available bytes (6,701.46GiB)**; it supersedes the
earlier fresh observation too. This is not reserved space. Keep the original
176-GiB payload cap plus32-GiB free reserve and check before major writes.

Using [003](../003/handoff.md)'s FP32 estimate, this **probe** retains native S0
36.5045GiB + temporary compact S0 export36.5045GiB + one actual gradient
12.1682GiB = **85.1772GiB**, plus measured input/state/metadata/serialization
overhead. Record actual per-file bytes including temporary files and checkpoint
metadata on all ranks, not just logical numel. No postclip/P/m/v full endpoint
archive is required. GPU scratch is one local parameter's P/g/m/v plus native
workspace; host scratch is bounded to that bundle and compact serialization.
Measure the actual largest bundle/layout span before decoder allocation.

This sidecar deliberately resolves the minimal probe without inventing a
partial-DCP reader. It cannot remain beside the proposed full ten-gradient
reference under176GiB: S0+ten gradients is158.186GiB before overhead, and another
S0 export would exceed the cap. **A bounded reader of native S0 local tensors,
or another reviewed way to avoid that duplicate, is a full-recorder integration
gate.** Do not scale this small-probe export unchanged or silently replace
the native continuation checkpoint. No new storage framework is requested here.

Retain S0 for every named future calibration/A/B/continuation consumer; none
exists yet. Delete temporary sidecars/gradient payload only after all named
probe consumers and review finish, with path/hash/bytes/result/deletion inventory
and existing cleanup authorization. Preserve compact failure evidence and stop
on incomplete writes; no automatic broad cleanup or checkpoint pruning.

Before coding and after integration, review added full-tensor reads/hashes,
D2H/H2D traffic, blocking device drains, compact serialization and filesystem
latency. Preclip writes one family; postclip and post-update hashes still read
their complete families. S0 export adds three families once. Decoder adds
native clipping/AdamW and transfers once per parameter. No new model compute,
comparison all-gather or norm reduction is required; existing native collectives
and router update remain. These observation runs are not throughput evidence.
Off-by-absence overhead and graph-on scheduling/replay effects require review
and later representative evidence, not a claim from this design.

CPU regression contract: exercise real callback dispatch and binding with small
native CPU AdamW state, four accumulated backwards and actual native CPU clip
as a **CPU boundary fixture only**; test observed-before-clip versus actual
inner pre-hook/post-step order, used/next LR, None versus zero, noncontiguous
layout, aliases/collisions, bounded serialization, partial files, hash corruption,
missing rank/family, cap/reserve rejection and cleanup of hook handles on failure.
Check native full-checkpoint/load-only selection and deferred RNG restoration
using existing continuation tests and owner doubles that preserve staged versus
current RNG. CPU cannot validate fused CUDA decoding, FSDP2/NCCL dependencies,
FP8, compile or replay. Regressions must fail on deliberately broken capture
order/corrupted bytes, not merely assert source text. Run focused tests and
`ruff check --select F821,F811,F841 --no-cache` on changed Python files; report
unavailable repository lint gates honestly. Independently review integration and
runtime costs before GPU handoff.

Full ten-update calibration/A/B and both continuation directions remain later
gates. [002](../002/acceptance-amendment.md) requires production A/A variability
and defensible precision criteria frozen **before** graph-on numerical
interpretation; no numeric thresholds are frozen here. Exact structure and
checkpoint restoration remain required. No GPU result or production-readiness
claim follows from this brief.
