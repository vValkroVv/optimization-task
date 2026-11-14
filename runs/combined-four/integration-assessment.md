# Combined four: source integration assessment

Assessment date: 14 November 2026. Scope: read-only source assessment and this
record only; no source edit, test execution, installation or job launch.
Applied `code-rules` and `perf-experiments`, the repository operating rules,
GIFT contributing/review rules, and scoped config, experiment, model,
distributed and integer-width rules.

## Finite acceptance

1. Identify the current activation state and exact reusable source.
2. Specify the smallest integration that enables exactly the four requested
   mechanisms while preserving the accepted crop and existing controls.
3. Reconcile prior numerical/lifecycle evidence, inherited contracts and costs.
4. Name only the additional checks needed for their combined activation.

All four assessment items are complete. Combined execution and final independent
integration/runtime-cost review remain coordinator-owned acceptance gates.

## Checked identity and present state

Root: `/workspace-SR008.nfs2/vkropotin/optimization-task`; branch/root revision
at initial inspection: `main`, `6c6cd9c0e41a0a5d3d00e1e252b3c120052d3507`.
The worktree was clean. `gift/` is the active flattened source; no original
workspace GIFT checkout was accessed. The production contract is
[baseline](../../docs/baseline.md): B2/S4096, accumulation32, D4096,
V128256, eight trunk layers, EP8/dense-FSDP8/expert-FSDP1, PP/TP/CP1,
FP32 masters/gradient accumulation, dense+sparse FP8, model/loss compile,
per-layer SAC, fixed capacity50000, fused AdamW, EMA and MTP1 weight0.3.
Both timing variants and the diagnostic must use logical epoch102400
(200 optimizer updates per epoch, covering140 updates plus queue) and the agreed
monitoring-off native runner; scheduler65000/warmup2000 is unchanged.

Current source has V2 and BF16 reuse integrated but default off. The active
HybridEP import still selects `distributed.deepep.routing.indices_to_map`;
`CrossEntropyLoss.__init__` and `ChunkedCELoss.__init__` still assign the native
`cross_entropy_loss`. All four-fusion helper modules are dormant on this path.

SHA256 identities checked directly:

| File | SHA256 |
| --- | --- |
| `gift/torchtitan/distributed/deepep/hybridep.py` | `12b1921372f265ede5cca4974f5867f954f84862cc257ff8e02bb7ecafa0814d` |
| `gift/torchtitan/components/loss.py` | `14d546d63b069a69041ec4a379d10a06f9b6a7888f25a68dddcadf27293b7f7c` |
| `gift/torchtitan/experiments/four_fusion/routing_metadata.py` | `2fb70d05edb774b09af93f0e97bd8054a75cd13f741e6071a82bf07e827c2f53` |
| `gift/torchtitan/experiments/four_fusion/stable_ce.py` | `d101584f7abe3ee8c82f39fc758ea47c0c2a9e6abedd72bd0ed71310679e508a` |
| `runs/four-fusion/routing-metadata.patch` | `4a9e17c884701bbdcd6e4fd68979d5aa45a37b27fce54bfdfbd3ee3dde1e7325` |
| `runs/four-fusion/ce.patch` | `d7cc5e5e79a11c1c76434377b5cd2d424f9582f61d3e1a4e894a6f65f873da12` |

The two core hashes equal the restored historical control hashes in the
[routing implementation](../four-fusion/routing-implementation.md) and
[loss implementation](../four-fusion/loss-implementation.md).
`git apply --check runs/four-fusion/routing-metadata.patch
runs/four-fusion/ce.patch` passed together without writing source.

## Minimal implementation specification

Use the existing two activation patches together between completed baseline
and candidate launches. They modify disjoint files and require no new schema,
kernel, model wrapper or environment selector. Freeze the exact patch bytes,
helper bytes, runner and comparison protocol before baseline execution. Keep
every run's source immutable from launch through confirmed group exit; record
revision, complete source diff and resolved config separately for each variant.

| Mechanism | Baseline | Combined candidate | Existing owner |
| --- | --- | --- | --- |
| CPU materialized-batch prefetch | depth0 | depth32 | `BaseStreamingDataLoaderConfig.num_prefetch_batches`; `datasets/streaming/core/dataloader.py::BaseStreamingDataLoader.__iter__` |
| BF16 expert reuse | false | true | `ParallelismConfig.gfsdp_reuse_bf16_expert_weights`; `models/gift_moe/_base/parallelize.py::apply_gfsdp` |
| Routing metadata | original import | routing patch | `distributed/deepep/hybridep.py::_dispatch_impl`; existing `four_fusion.routing_metadata.indices_to_map` |
| Cap-zero CE | original callable | CE patch | `components/loss.py::{CrossEntropyLoss,ChunkedCELoss}.__init__`; existing `four_fusion.stable_ce.cross_entropy_loss` |

The candidate routing patch only changes `indices_to_map` import;
`gather_route_gradients` remains native. The CE patch only changes `self.fn`
before `_maybe_compile`; `MTPLoss` inherits the change. Do not apply the input
preparation or Linear+CE patches. Do not change `Unembedding`, head forward,
trainer input preparation, precision, topology or any crop dimensions.

Use existing `runs/overnight-streaming-perf/run.py` arguments:
baseline `--num-prefetch-batches 0` without the reuse flag; candidate
`--num-prefetch-batches 32 --gfsdp-reuse-bf16-expert-weights`.
`--epoch-size 102400` is common. The old `runs/four-fusion/evaluate.py` selector
hardcodes depth32 and exactly one patch, so it cannot express this comparison.
Use the existing native runner directly with reviewed serial orchestration;
do not grow the old four-independent-candidate matrix or add a new framework.
Reverse both activation patches only after candidate group exit, unless the
coordinator's agreed final source disposition explicitly keeps them active.

## Preserved paths and reusable evidence

- **Loader ownership:** `BaseStreamingDataLoader._iter_prefetched` enters epochs
  on the consumer, uses the existing bounded `PrefetchIterator`, and commits
  the yielded batch's state. `state_dict` pauses its producer and saves the
  committed cursor; `load_state_dict`/`close_iteration` retain existing explicit
  data-world and worker shutdown behavior. No GPU work enters the producer.
  [V2 evidence](../v2-prefetch-baseline/report.md) covers245 affected checks,
  eight-owner Gloo/DCP, rollover and persistent/nonpersistent workers.
- **Expert lifetime:** `apply_gfsdp` selects exact expert parameter groups before
  lazy initialization. Eligibility requires FP32 masters/BF16 inputs, size-one
  shard groups and compatible ordinary layouts; unsupported groups fall back.
  `_param_group`, `_collectives`, `_state` retain shared-buffer ownership and
  reader/refresh/write events. `OptimizersContainer` inner pre/post hooks,
  ordinary/DCP loading, EMA and `GFSDPModule._apply` own invalidation/write
  publication. Neither fusion changes these writers or parameter identities.
  [Final focused evidence](../bf16-expert-cache/focused-checks.md) includes exact
  actual-GFSDP loss/input-gradient/expert-gradient/master comparisons over
  accumulation and updates, SAC, DCP, EMA and cross-stream `_apply`; main008
  passed after retained main006 exposed a real stale fallback cast. The
  [review](../bf16-expert-cache/review.md) reconciles the resolved findings.
- **Routing:** top-k distinct retained IDs pass through capacity dropping before
  `_dispatch_impl`. The kernel retains contiguous bool/FP32 dense maps and
  trailing sinks, signed-int64 offsets and unchanged FP8 dispatch/combine,
  DeepEP stream ownership and native combine-backward gradient gather. Its
  standalone backward is not an extra gather in opaque HybridEP. The
  [implementation checks](../four-fusion/routing-implementation.md) cover crop
  shape8192x128/top8, striding, negative routes, zero/NaN bits, layouts and
  standalone/compiled gradients. Native eight-rank activation and rank0/1
  trace evidence are in the [report](../four-fusion/report.md).
- **CE/MTP:** `ChunkedCELoss._chunked_loss_and_grad` still calls the wrapped
  head, performs two outer sequence chunks and its existing inner backward.
  `MTPLoss.pack_targets` shifts before chunking; `_chunk_loss` splits folded
  main/MTP heads, invokes SUM CE, divides by the shared main valid-token count,
  and applies weights1.0/0.3. FP32 centered maximum/log-denominator and FP32
  incoming-scale arithmetic precede the sole logits-dtype gradient store.
  DTensor uses native distributed CE; CPU uses FP32 PyTorch CE; first-order
  CUDA gradients remain explicit. No optimizer/checkpoint state is introduced.
  [CE replay](../four-fusion/ce-numerics.md) already checks compiled full
  V128256 two-chunk main/MTP blocks with native denominator2096640, all finite
  gradients and exact ignored rows; differences are at most one BF16 ULP and
  selected FP64 rows explain them. Large-offset/partial-vocab/nonunit-scale
  regressions and native activation are established. No new full-model dump or
  rerun of this unchanged numerical campaign is required.

## Pre-code runtime-cost assessment

| Scope | Expected combined effect and inherited cost | Remaining question |
| --- | --- | --- |
| CPU producer | depth32 overlaps materialization with consumer compute; existing queue/thread and batch storage persist; synchronous epoch entry remains | Combined update cadence and host pressure must be observed, not inferred from prior V2 speed |
| Expert casts/copies | reuse existing persistent BF16 input per FP32 generation; bypass eligible singleton output copy; retain FP8 quantization and shared fallback pool | Combined activation and update-boundary refresh must remain visible; no new full BF16 buffer or memory saving is claimed |
| Routing device work | one kernel/call instead of eager metadata chain; T8192/E128/K8 means8388608 compares, about5MiB dense output plus sinks/call; retain two allocations | Dense-map store/compare cost and communication overlap remain; prior trace counted39 calls/rank per3 microbatches including SAC |
| CE device work/memory | rowwise forward/backward plus FP32 SUM; retain BF16 logits and dlogits, two FP32 statistics/row; no head-GEMM fusion, no FP32 full-logits tensor | Combined peak live memory and enclosing head/loss latency remain unmeasured |
| Host/compile | unchanged native compile boundary; opaque custom ops retain existing fake/autograd rules; routing guards use metadata, not device reads | First compilation/cache state and settling must be excluded from the agreed measured window only on actual evidence |
| Synchronization/communication | no added collective, host scalar read or copy from the two patches; cache's existing per-group event waits/hooks remain; capacity checks, collectives and token-drop predicates unchanged | Scheduling interactions can alter waits; kernel-duration sums cannot establish step savings |
| Persisted state/evaluation | no new parameter, buffer key, optimizer state or loader ownership change; CE CPU/DTensor fallbacks retained | Prior lifecycle evidence applies; combined short runs do not establish exact native continuation, convergence or PP4/EP32 behavior |

There is no independent prediction of additive speedups. Prior routing and CE
TPS effects were small versus control variation; historical BF16 comparison
was not contemporary. Use the requested combined unprofiled comparison for
integrated latency/TPS and paired allocator peaks. Traces, if in the agreed
budget, establish activation/mechanism and host/device/wait/copy/communication
scopes separately; device/kernel sums are not elapsed update time.

## Focused checks before implementation handoff

Only integration-specific uncertainty requires new verification:

1. **Composition/control:** verify native runner child arguments for both
   variants and both patches' joint forward/reverse applicability without a
   running group. Confirm only the two core files differ and helper hashes
   remain the above pins; baseline resolves depth0/reuseFalse, candidate
   depth32/reuseTrue. Check epoch102400 and all preserved crop fields in actual
   rank0 resolved configuration. This prevents an accidentally V2-only control.
2. **Actual combined binding:** with both patches active, execute existing
   `test_fusion_routing_metadata.py::test_triton_metadata_runs_inside_hybridep_dispatch_custom_op`
   and `test_fusion_losses.py::test_native_mtp_two_chunks_loss_hidden_weight_and_identity`
   (native_float denominator, eager and compiled arms). Require actual passes,
   not CUDA skips or the tests' inactive-patch skips. These directly cover the
   new joint import boundary and native two-chunk/MTP callback; retain original
   suite numerical tolerances. Run on authorized available GPU only. No
   whole routing/CE/cache/loader campaign needs repeating for unchanged bytes.
3. **Activation in the combined native run:** resolved flags alone are
   insufficient. Confirm the existing kernel/custom-op observables for routing
   and stable CE, CPU queue use, and eligible BF16 cache activation/refresh;
   ensure input-prep and Linear+CE observables are absent. Reuse the existing
   diagnostic hooks/counters only in the agreed diagnostic run; keep counters,
   profiling and numerical capture off in performance runs. Actual success
   through32 microbatches and an optimizer boundary addresses the combined
   scheduling/lifecycle risk that individual historical checks cannot close.
4. **Mechanical checks/final review:** scoped Ruff F821/F811/F841 and normal
   lint/format on changed Python targets, `git diff --check`, and maintained
   evidence links; no source-text test substitutes for items2/3. Independent
   integration/runtime-cost review must reconcile final composition, actual
   resolved execution, unchanged state ownership and the measured result.

The assessment executed source reads, hashes and a nonmutating joint patch
applicability check only. It does not claim these new focused runtime checks,
the combined native comparison or the final review have passed. Their failure
or unavailability is an explicit acceptance gate, not grounds to disable FP8,
compile, SAC, MTP, EMA or accumulation. Stop after the finite combined task;
individual campaigns and new kernels are excluded.
