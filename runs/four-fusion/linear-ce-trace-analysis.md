# FLCE activation failure and bounded correction design

**The first FLCE TPS pair and trace fail the mandatory activation gate.**
They establish neither FLCE speed nor FLCE memory savings. The candidate
completed normally, but executed the original head/compiled CE path. The
[pair record](linear-ce-pair-001.json) now says so. No source, runner or
configuration changed during this analysis, and no GPU command was executed.

## Actual Perfetto evidence

The [JSON](linear-ce-trace-analysis.json) owns exact handles, source/config/runtime
pins, normalization receipts and remaining gates. Complete SQL and results are
in `artifacts/raw/linear-ce-trace-queries.json`; canonical/prepared trace paths
and hashes are in its preparation receipts. Actual `mcp__perfetto__load_trace`
and `execute_sql` were used. Both candidate ranks import **9,930 device events,
all flow-linked exactly once, zero importer errors**; counts and duration sums
match the lossless preparation receipts. Synthetic display lanes preserve
original CUDA stream arguments and device fingerprints.

| Per captured rank | Required activated path | Observed |
| --- | ---: | ---: |
| Complete sampled microbatches | 3 | 3 |
| `four_fusion::native_linear_ce` calls | 6 | **0** |
| Centered `forward_kernel` / `backward_kernel` tiles | 48 / 48 | **0 / 0** |
| Projection + dH + dW tile GEMMs | Expected 144; count actual after repair | Not attributable to FLCE |
| Original compiled CE forward-online / NLL / backward kernels | Replaced inside selected path | **12 / 12 / 12** |

The kernel-category query retains exact IDs, bounds and actual stream 7 for
all 36 original CE kernels per rank. Unfiltered names also include host launch
annotations; they were not counted as device kernels. Full head/chunk guards
remain mapped in [baseline-loss-regions](baseline-loss-regions.md). Comparing
inactive candidate timings as FLCE would be invalid, so further FLCE local
latency/copy/communication/downstream attribution stops at this gate.

## Exact producer and failed consumer

1. `DecoderConfig._head()` creates canonical `LinearConfig`; `Config.build()`
   calls `ModuleConfig._attach_config(instance)`, installing `_sharding_config`.
   The canonical head is `torchtitan.models.common.nn_modules.Linear`, inheriting
   `torch.nn.Linear.forward`.
2. `parallelize_gift_moe(...)` calls `model.parallelize(parallel_dims)` whenever
   TP **or EP** is enabled. The actual configuration is EP8, TP1,
   `full_dtensor=False`, CP1. GFSDP then wraps the head; `Trainer` calls
   `loss_fn.set_lm_head(lm_head)` after parallelization.
3. `Module.parallelize(self, parallel_dims)` always replaces a configured
   module's `forward` with plain closure `forward_with_redistribution(*args,
   **kwargs)`. Its closure contains `fn`, `parallel_dims`, `self`.
4. `LinearCE.install(self, head)` tests
   `getattr(head.forward, "__func__", None)` against exactly
   `Unembedding.forward` and `torch.nn.Linear.forward`. The canonical closure
   has no `__func__`, so `supported_head=False`.
5. `LinearCE.chunk_loss(...)` includes that false flag in `selected`, then
   calls `head(hidden)` and `loss._chunk_loss(...)`: the original compiled CE.

The head wrapper is actually a redistribution no-op for this preset. Its state,
input and output placements use dense DP/CP/TP axes, excluding EP. Under
`ParallelDims.resolve_mesh(axes)`, `full_dtensor=False` removes DP/CP; TP1 is
inactive. All head meshes resolve `None`, and `local_map` is absent. The wrapper
still changes callable identity.

The CPU reproduction in `artifacts/raw/linear-ce-wrapper-cpu-repro.json` uses
canonical `Config.build()` and actual `Module.parallelize`, with a stub
representing no active dense mesh. It changes supported-head recognition from
true to false while preserving outputs exactly. This proves the callable/guard
failure, not GPU or distributed correctness. The initial direct-constructor
attempt omitted config attachment and was inconclusive; its evidence is retained.
Existing unit and world1 GFSDP fixtures construct heads directly and do not
execute this canonical config attachment/parallelization path.

## Minimum maintainable correction design

Existing `_parallelized`, `_sharding_config` and `_pos_arg_list` do not prove
which callable currently occupies `forward`. Do not whitelist wrapper names,
accept arbitrary `__wrapped__` chains, or bypass active redistribution.

At the end of canonical `Module.parallelize`, retain one private setup-only
three-tuple, provisionally `_parallelized_forward_contract`:

- `wrapper`: the **exact** installed `forward_with_redistribution` object;
- `original_forward`: the bound callable captured before `_maybe_wrap_with_local_map`;
- `parallel_dims`: the exact dimensions object used by that wrapper.

The adapter may recognize that contract only when current `head.forward is
wrapper`, the recorded original is bound to this head, and its `__func__` is one
of the two exact supported primitive implementations. Reject `full_dtensor=True`
and configured `local_map` unconditionally. Prove no-op state and boundary
behavior using existing resolvers:

- Every named placement in `_sharding_config.state_shardings` resolves to
  `None` via `parallel_dims.resolve_mesh(...)`; inspect configured buffer
  placements too. Reject an active mesh instead of treating replicated
  placements on an active mesh as no-op.
- For every key in the union of `in_src_shardings` and `in_dst_shardings`,
  `resolve_shared_mesh([src.get(key), dst.get(key)]) is None`.
- `resolve_shared_mesh([out_dst_shardings]) is None`; retain conservative
  fallback for unsupported structured output placement configurations.

Keep the **whole** original wrapper in `LinearCE.original_forward` for fallback.
Retain existing DTensor/input dtype/logits-hook/biased-head/softcap/evaluation/
shape-probe/argmax/denominator guards. A user-installed wrapper, stale/copied
contract, nontrivial TP, local-map path or DTensor remains on the original path.
Treat sharding config and dimensions as fixed after parallelization; supported
mutation must invalidate/recompute this contract. Do not infer safety from
`_parallelized=True` alone. This is a design proposal, not implemented code.

### Integration and cost review required with that correction

The new tuple contains Python references only: no parameters, registered buffers,
state-dict keys, checkpoint tensor bytes, device transfers, synchronization or
collectives. It is populated once per configured module. It adds a small Python
object and retains three references for module lifetime; no per-tile work is
needed. Confirm module pickling/deepcopy and any compiler inspection of module
attributes against the actual checkout. Keep recognition outside the compiled
custom op and preserve both original wrapper fallback and `head(...)` GFSDP
hooks. Review changed protocol metadata and adapter together; their existence
alone does not prove runtime or checkpoint compatibility.

### Regression and renewed production gates

Use canonical `LinearConfig.build()` with the production dense head sharding
config, actual `Module.parallelize`, then GFSDP; direct construction is
insufficient. Match compiled loss, native integral host denominator, two outer
chunks and folded main/MTP heads. Prove the old path emits **zero** selected
calls and the correction emits **two per microbatch**. Compare loss/components,
dH/dW, parameter identity, state-dict keys, head flags, accumulation and final
sync/reshard. A CPU no-op mesh fixture can verify identity/provenance checks;
an actual GFSDP GPU fixture remains required for numerical/lifecycle evidence.

Negative fixtures must execute actual `Module.parallelize` for nontrivial TP
and local-map cases, retaining fallback; also check `full_dtensor`, DTensor
inputs/labels, altered state/input/output meshes, custom/stale wrapper identity,
primitive subclasses overriding forward, and logits hooks. Then repeat the
native production trace/TPS gate, count actual tile/GEMM work, and include
complete forward plus both gradients and the enclosing FSDP/accumulation guard.
The coordinator owns source authorization, checks and renewed runs.

## TPS outlier

`artifacts/raw/linear-ce-outlier-evidence.json` retains all 20 updates × eight
ranks per arm, all 32 F/B times/update, cadence, GC, timestamps and native-log
hashes. Candidate update **11**, microbatch **18**, takes **2606.39–2607.53 ms**
on every rank versus normal F/B near 378 ms. It occurs inside F/B after input
postprocessing. Rank-0 update 11 is 15.6194 s versus control 13.3208 s; its
2.2986 s excess accounts for the full 2.1773 s window excess. Later updates in
both arms gradually lengthen rather than showing continuing startup settling.
GC is approximately 0.04 ms and does not explain the stall. Compile observers
and the DeepGEMM guard were disabled; logs supply no compile/recompile event.
The stall's cause is **unknown**. No evidence justifies extending warmup.
Repair activation first, then use an ordinary matched 20-update 10+10 repeat;
retain this failed pair and outlier. Never promote an outlier-trimmed diagnostic
as the measured result.

The observed peak allocation difference, 69.5363 → 67.5651 GiB, is also not an
FLCE effect. Source predicts smaller transient logits for an activated tiled
path, plus one FP32 dW accumulator shared across eight 1024-row tiles and both
folded heads, repeated FP32 `addmm` updates/zero-fill, and one BF16 output cast.
Both dH and dW are computed inside custom-op **forward**. Their costs and the
memory mechanism remain unmeasured for the inactive native run.

Validation: JSON parses, SQL results are complete and untruncated, preparation
fingerprints/counts/flows pass, CPU identity reproduction passes, and changed
document links/whitespace pass. No failed SQL queries were omitted. A failed
raw-file patch operation had no source effect and was corrected; it is recorded
in the JSON analysis failures. Production FLCE acceptance remains open.
