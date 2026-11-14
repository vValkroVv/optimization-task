# BF16 expert reuse: bounded implementation design

Read-only design, 13 November 2026. This file records proposed interfaces; they
are not implemented or verified by this design assignment.

## Acceptance and choice

The finite design acceptance is: identify the active producer/consumer path;
define eligibility, storage ownership, mutation and stream lifecycle; give a
focused actual-GFSDP fixture; account for runtime costs and open verification.
The implementation experiment and its two native launches remain the root
assignment's responsibility. No GPU work, installation or source edits occurred
in this design assignment.

Use the existing per-group BF16 input as compute storage for the opt-in eligible
expert groups. Preserve the shared output pool and its ownership checks for
other groups. Retaining the existing ownership checks for eligible groups is a
deliberate conservative choice: it avoids changing prefetch/reshard policy in
the same experiment, although eligible outputs no longer physically need that
pool. No new full-size BF16 tensor is needed.

The lifecycle reference is [Megatron optimizer step at the supplied pin](https://github.com/NVIDIA/Megatron-LM/blob/5b069b1f76890bb322c6fdb656419999c0ba073e/megatron/core/optimizer/optimizer.py#L937)
and [main-to-model copy](https://github.com/NVIDIA/Megatron-LM/blob/5b069b1f76890bb322c6fdb656419999c0ba073e/megatron/core/optimizer/distrib_optimizer.py#L2905).
The former updates masters before refreshing model copies; the latter handles
the compute-weight boundary. Adapt this lifecycle by invalidating on update and
lazily refreshing GFSDP's existing buffer. Do not import the optimizer or its
quantized storage machinery. This is compatible with preserving GFSDP FP32
masters and its existing BF16 cast; it establishes no performance claim.

## Active source path and eligibility

`gift/torchtitan/models/gift_moe/_base/parallelize.py:70–112` applies SAC and
compile before `apply_gfsdp`; its `shard_block_unit:162–180` creates one GFSDP
unit with a separate `param_meshes` group for the exact routed-expert parameter
set. This is the correct opt-in boundary. `distributed/fsdp.py`'s
`grad_reduce_kind` stamps belong to the other backend and must not be assumed
present here.

Canonical config: `parallelism.gfsdp_reuse_bf16_expert_weights: bool = False`
in `torchtitan/config/configs.py:ParallelismConfig`. Pass only in the GFSDP
branch to `apply_gfsdp`. At `shard_block_unit`, select a group by equality of its
managed expert parameters with the exact `block.moe.experts` set captured
before sharding, or by the explicitly supplied `edp_mesh` plus an assertion of
that equality. Avoid FQN substring matching. Enable before lazy initialization.

Eligibility is deliberately narrow: GFSDP, PP1, EP>1, an expert-only group,
shard process-group size one, FP32 original data, BF16 mixed-precision params,
the ordinary `None` gather bucket only, no `fsdp_pre_all_gather` /
`fsdp_post_all_gather` extensions, and no YCCL handle. Assert equal local and
unsharded numel lists/padded sizes before aliasing. This retains the accepted
eight-GPU crop's path while keeping noneligible groups on their existing path.
Warn once if an explicit opt-in activates no group. Do not impose new errors
or behavior on default-off configurations.

`_param_group.py:405–452` currently makes unsharded views over shared outputs;
`lazy_init:490–524` subsequently allocates persistent inputs. Allocate and bind
the input buffers before creating the unsharded parameter views, or pass the
already-created inputs into `_init_unsharded_params`. For eligible groups,
`_all_gather_output[None]` aliases `_all_gather_input[None]`. Existing
`GFSDPParam.init_unsharded_param` creates its normal Parameter once over this
view. Do not recreate Parameters per refresh, and do not alias FP32 masters,
gradients or EMA buffers to BF16 storage. Noneligible output and gradient views
remain byte-for-byte equivalent to the original initialization.

`_collectives.py:46–55` already performs the exact FP32-to-BF16 `copy_` under
`_unsafe_preserve_version_counter`; reuse it. For eligible groups, preserve the
valid input across `_param_group.finalize_backward:890`. After preparation,
skip `all_gather_into_tensor` only on the proven size-one/direct-input path.
Retain an `AllGatherResult` event so existing `wait_for_unshard:684–690` and
`_to_unsharded` behavior remains intact. All other groups still clear their
validity and execute their original collective.

`GFSDPParam.to_unsharded:240–260` handles extension metadata. Excluding extension
parameters from this optimization preserves their existing preparation and
per-forward metadata refresh. Routed expert FP8 quantization remains in
`models/common/moe/experts/gmm.py:_quantize_local_expert_weights` and the
dispatcher consumer; do not cache, skip or alter those operations.

## Concrete lifecycle interface and hooks

Suggested group methods and state:

- `enable_bf16_weight_cache()` before lazy init; records the request and
  finalizes eligibility when handles and actual layout are available.
- `invalidate_bf16_weight_cache()` increments a Python integer generation,
  marks the cache invalid and clears its ordinary gather-input validity.
- `record_weight_write()` records a device event on the current mutation
  stream after writes. The next refresh waits this event.
- `_bf16_cache_generation`, `_bf16_cache_prepared_generation`,
  `_bf16_cache_ready_event`, `_bf16_cache_reader_event`, and
  `_bf16_cache_write_event`. These are derived runtime state, never state_dict
  buffers. No device scalar read or `_version` check is required.

The root/module API may expose `invalidate_bf16_weight_cache(recurse=True)`
and a post-write notification, iterating unique groups, for explicit supported
external weight loading. Mutation API calls are quiescent operations, as are
the existing optimizer/load/EMA APIs; concurrent forwards and writes are not
made newly supported.

Exact integration points:

1. **Optimizer.** `components/optimizer.py:OptimizersContainer.__init__` builds
   actual inner optimizers; `step:187–191` executes them. After construction,
   register pre/post step hooks on each actual optimizer for the eligible
   groups whose stable sharded Parameter IDs occur in that optimizer. Pre-hook
   invalidates; post-hook records completed weight writes. This also covers
   direct calls to the inner optimizer. Keep hook handles owned by the
   optimizer/container. Do not rely on a trainer-only notification, and do not
   modify optimizer mathematics, gradient routing or master precision.
   The selected crop uses the standard fused AdamW container. Do not activate
   this cache in optimizer-in-backward paths without equivalent hooks; leaving
   such paths on the original behavior preserves their support.
2. **Ordinary load.** Split `_param_group._register_state_dict_hooks:1006–1028`
   so save only reshards while pre-load reshards and invalidates. Add post-load
   event recording for direct `module.load_state_dict` and
   `set_model_state_dict` consumers. This must operate on existing master
   storage and preserve existing assign/load semantics.
3. **GFSDP DCP load.** `_dcp.py:load_gfsdp_dcp:527–549` builds load views and
   `dcp.load` writes them directly, before ordinary load hooks run. Invalidate
   eligible groups before `dcp.load`, then record writes after its successful
   return and EMA initialization. `apply_main_state_dict:427–445` subsequently
   runs the ordinary hooks. Duplicate invalidation on a load is harmless;
   multiple generation increments before any consumer still cause one refresh.
   Invalidate before the operation so a partial failed load cannot preserve a
   valid cache. DCP's main/EMA checkpoint views and metadata remain unchanged.
4. **EMA swap and restoration.** In `swap_ema_and_weights:621–635`, invalidate
   on each swap and record an event after active master-pointer selection.
   Read refresh input from `_padded_sharded_param_data`, which is the active
   main/EMA buffer, rather than `fsdp_param._sharded_param_data` (always main).
   `initialize_ema_from_main:598` and `update_ema_params:612` mutate EMA; they
   invalidate/record only if EMA is currently active. `BackendBufferStore.swap`
   and `callbacks/builtin/ema.py:swap_weights` already reach the group method.
5. **Supported backing replacement.** `fully_shard.py:GFSDPModule._apply:524–533`
   reshards, applies the transformation, then calls
   `GFSDPParam.reset_sharded_param:352–360`, which copies replacement data into
   the existing flat master storage. Invalidate each affected group before
   `_apply` and record after reset. Preserve the existing reset's identity
   semantics; this optimization must add no Parameter replacement. The stable
   BF16 buffer then refreshes from the same restored master allocation. Do not
   claim support for arbitrary unannounced `.data`, storage or raw flat-buffer
   mutation; callers of any new explicit replacement API must notify this
   lifecycle. Device/dtype transformations must re-evaluate eligibility before
   reuse; they must never retain a valid stale BF16 representation.

Refresh only when prepared generation differs. Cache hits in forward,
pre-backward, SAC recomputation and later microbatches do not change generation.
Do not treat finalize-backward, gradient reset, or normal EMA-shadow updates
while main weights are active as weight mutations.

## Streams, ownership and costs

Keep `unshard:655–664`'s original shared-context ownership and release-event
ordering. Also retain a **per-group** last-reader event, recorded when the group
reshards after compute/backward. The shared context event can be replaced by
another group, so it cannot be the sole lifetime marker for the private cache.
Before refreshing, the selected preparation stream waits both the group's
last-reader event and its master-write event. After casting it records cache
readiness. Every cache consumer, including a different explicit-prefetch
stream, waits cache readiness; then the normal all-gather-result event makes
the compute stream wait before parameter use. Retain the completion event on
an unused speculative preparation finalized in `finalize_backward`, too.

`_state._root_pre_forward:123–129` already waits for post-optimizer/current
stream work on the all-gather stream. Preserve that ordering. Per-group writer
events additionally cover explicit unshard entry points and nondefault
mutation streams. Events are stream dependencies, not host synchronization.
The existing backward-finalization reduction wait remains unchanged. Do not
call `.synchronize`, `.item`, or device-to-host copies in the new lifecycle.

Expected work is one ordinary cast per used group/generation instead of one
per microbatch, and zero eligible input-to-output collective/copy launches.
No extra BF16 allocation is expected: existing persistent inputs survive, and
shared outputs remain allocated for unchanged machinery. Memory reduction is
not claimed. Added host work is integer comparisons, opt-in hook iteration per
optimizer update and event handling; event overhead and inherited ownership
serialization must be inspected in the diagnostic trace. FP32 accumulation,
optimizer, quantization, compilation/SAC policy, gradient communication and
checkpoint byte formats do not change. There is no measured speedup here.

## Diagnostic API agreed for runner plumbing

`GFSDPModule.set_bf16_cache_debug(enabled: bool, recurse: bool = True)` enables
Python counters only; off by default. `bf16_cache_stats(recurse=True)` returns
a tuple of records with `fqn`, `enabled`, `generation`, `refreshes`, `hits`,
`direct_uses`, `invalidations`. Stats are integer metadata and perform no GPU
synchronization. Snapshot before/after each optimizer update for the diagnostic
launch; do not print per microbatch or enable counters during timing. Count
refresh only at the existing cast operation and direct use only at the
collective bypass. Explain that hits/direct uses depend on actual unshard and
prefetch counts; refreshes must be exactly once per consumed generation.

## Focused verification proposed

Use real `gfsdp.fully_shard`, `GFSDPParam`, mixed-precision views and backward
hooks, not stand-ins for parameter groups. A small two-rank CUDA fixture can
combine dense FSDP2 fallback groups with singleton efsdp expert groups and EP
DTensor expert shards. Use at least three sequential checkpointed units so
eligible groups 0 and 2 really share the original alternating working-buffer
context. Assert their context identity and distinct persistent-input pointers.
Alternatively supply an explicit legal context map with another independent
buffer to satisfy existing minimum-buffer checks; do not bypass validation.

Run matched uncached and cached models from identical FP32 initial values and
fixed inputs, deterministic algorithms, and the existing precision criteria.
Perform multiple microbatches per update and at least two optimizer updates;
run a further consumer after update two so the second invalidation is tested.
Use actual nonreentrant SAC/checkpoint recomputation and confirm a recomputed
consumer executes. Compare loss, input gradients, elementwise FP32 master
gradients before each step, and updated FP32 masters. At each weight consumer,
assert BF16 storage equals a fresh cast of that generation's active master;
this exact cast comparison is justified independently of whole-model FP8.

Then load deliberately changed weights through ordinary loading and an actual
temporary DCP save/load, and test EMA swap in and restoration with distinct
values. Verify each next consumer and one refresh per consumed generation.
Exercise same-device supported `_apply` replacement that changes values without
relying on version counters. Assert sharded/unsharded Parameter identities stay
stable across ordinary training refreshes. Cover default-off, dense, mixed,
size>1 and extension paths as noneligible with original output-buffer use.
Actual FP8/compile/SAC preservation is finally checked by the already authorized
native candidate runs; a BF16 fixture does not alone establish it.

Open gates are implementation, feasible fixture execution, scoped lint,
independent final integration/runtime-cost review, and the exact two native
candidate launches/trace analysis specified by the objective. This design does
not close GPU ordering, numerical parity, memory or performance gates.
