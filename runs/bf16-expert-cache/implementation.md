# BF16 expert cache implementation handoff

## Implementation

`parallelism.gfsdp_reuse_bf16_expert_weights` defaults to `false`. The GIFT
MoE GFSDP path requests caching only for the parameter group whose original
parameter set exactly matches `block.moe.experts`; it does so before lazy init.
`fully_shard` resolves the exact set during partitioning, before
`GFSDPParamGroup` eagerly replaces the original Parameters with sharded ones.
The group becomes active only for FP32 masters, BF16 parameter dtype, a
size-one shard group, the ordinary `None` bucket, no all-gather extensions or
YCCL handle, and equal local/unsharded layouts. Other configurations keep the
ordinary path. The config is passed only to GIFT MoE's GFSDP branch, and a
requested but unsupported setup is warned about.

The existing per-group BF16 gather input is allocated before the unsharded
parameter view and is used for both input and output on eligible groups. The
existing FP32-to-BF16 `copy_` remains in `_collectives.all_gather`; the size-one
collective is bypassed while its result event remains in place. Backward
finalization preserves valid cache state. Per-group readiness, reader and
master-write events order refreshes and consumers across normal and explicit
prefetch streams.

Supported writers invalidate or record the cache at the actual inner optimizer
pre/post step hooks, ordinary state-dict load hooks, GFSDP DCP load, active EMA
swap/update/initialization, and GFSDP `_apply`/backing reset. Cache state is
derived runtime state and is not added to checkpoint data. Optimizer-in-backward
groups remain on the ordinary path. Manual external writers must call
`GFSDPModule.invalidate_bf16_weight_cache()` before writes and
`record_bf16_weight_write()` after them.

GFSDP `_apply` copies transformed sharded values into the persistent flat
master, then restores the original sharded `Parameter` identity and its
`RaggedShardDTensor` local alias/spec with `torch.utils.swap_tensors`. The
sharded-gradient wrapper is restored over its persistent gradient storage as
well, preserving optimizer references and later gradient routing.

The diagnostic API is `GFSDPModule.set_bf16_cache_debug(enabled, recurse=True)`
and `bf16_cache_stats(recurse=True)`. Counters are disabled by default; stats
return Python metadata without device reads. Before lazy init, `enabled` reports
known structural eligibility; lazy init finalizes it against the actual buffer
and YCCL layout.

## Runtime cost reconciliation

| Cost | Expected effect | Evidence or remaining gate |
|---|---|---|
| Cast work | One existing FP32-to-BF16 copy per consumed weight generation instead of after each backward finalization | Source path inspected; diagnostic trace must confirm cast frequency |
| Gather work | No `all_gather_into_tensor` on eligible size-one expert groups; the original result event and shared-buffer ownership remain | Source guards and direct-input assertion implemented; trace must confirm the copy/collective is absent |
| Memory | No additional full-size BF16 allocation; existing inputs persist and the shared output pool remains allocated for fallback groups | Source layout inspected; peak allocated/reserved memory remains a native measurement gate |
| Compute, precision and gradients | No arithmetic, FP32 master, gradient buffer, reduce-scatter, FP8 quantization, compile, SAC, EMA or MTP policy change | Source inspection only; focused fixture and native run remain required |
| Host work and events | Opt-in optimizer hooks iterate selected groups once per inner optimizer update; cache paths add generation checks and per-group event waits/records; counters add no work unless enabled | Source inspection only; diagnostic trace must check event overhead and overlap |
| `_apply` replacement | Value copy and wrapper restoration occur only on module conversion; no new optimizer Parameter is created | Installed-runtime wrapper probe passed; fixture must verify a later backward/update |

No GPU performance or distributed runtime claim is made here. The implementation
keeps the old fallback initialization order and collective path when the option
is off. The opt-in path does not add a second BF16 buffer, but retains the
existing shared output allocation, so no memory reduction is claimed.

## Verification and open gates

- Scoped `ruff check --select F821,F811,F841 --no-cache` passed for the nine
  changed Python files.
- Normal scoped `ruff check` and `python3 -m py_compile` passed for the same
  files.
- A task-environment import check passed for `GFSDPModule`,
  `GFSDPParamGroup`, `OptimizersContainer`, and `ParallelismConfig`; the config
  default was confirmed false and the installed optimizer exposes post-step
  hooks.
- A focused one-rank Gloo wrapper probe passed on installed Torch
  `2.15.0.dev20260822+cu130`: `Module.to(dtype=torch.bfloat16)` swaps
  `RaggedShardDTensor` internals while preserving Parameter identity, and a
  wrapper swap preserves identity and restores the expected local `data_ptr`
  and spec. Reproduce with
  `PYTHONPATH=gift python3 runs/bf16-expert-cache/probe_ragged_shard_reset.py`.
- `git diff --check` passed.
- The fixture owner's first focused two-GPU run reported `2 failed, 1 passed`;
  see [the retained raw log](artifacts/raw/pytest-cache-fixture-001.log). One
  failure is a fixture extension stand-in whose `nn.Parameter` methods do not
  survive GFSDP sharding, so the group correctly remains eligible; the other is
  test model initialization reaching a ragged tensor outside its RNG tracker.
  The fixture owner is correcting these harness issues. This implementation
  task did not launch the fixture, training, or a performance run, and did not
  install dependencies.

The final actual-GFSDP fixture passed in main008, including real EP partitioning,
two optimizer updates, exact loss/input-gradient/expert-gradient/master parity,
ordinary and DCP loading, EMA swap/restoration, and a subsequent optimizer step
after value-changing `_apply` on a separate writer stream. The default-off and
extension-fallback checks also passed. See [focused checks](focused-checks.md)
for exact commands and retained failures rather than treating the early harness
failures above as unresolved source defects.

The final targeted `_apply` repair also clears ordinary gather-input validity
for every affected group. It joins prior shared-buffer release events on the
writer stream before transformation, then publishes one post-reset completion
event to the unique initialized contexts. This preserves fallback readers and
orders new casts after writes even with an older root optimizer event. It adds
only mutation-boundary metadata/event work, with no new steady-state branch,
host synchronization or full tensor allocation. Main006 retained the original
post-replacement parity failure; main007 and the final cross-stream main008
passed the repairs. Final scoped normal Ruff, F821/F811/F841 and whitespace
checks passed.

The independent reviewer accepted all material corrections; findings and final
cost dispositions are in [review.md](review.md). Both authorized native
candidate launches and trace analysis subsequently passed; the
[completed report](report.md) owns their results and limits. Performance
expectations and eligibility details are in [design.md](design.md).
