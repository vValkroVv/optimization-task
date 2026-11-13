# Optimization candidates from the meeting

This is the entry point for three proposed research directions. The default is
still the [verified 100B crop](baseline.md). The sources below are
reference material for future experiments, not enabled changes or measured
speedups on that baseline. The original campaign is cancelled/incomplete. The
newly requested read-only [update-11 Perfetto analysis](../runs/overnight-streaming-perf/report.md#update-11-candidate-analysis-with-perfetto-mcp)
maps these mechanisms to the fresh monitoring-off trace and native logs. It ranks
streaming prefetch first for directly exposed latency, identifies a larger
linear/CE boundary and recurring MoE capacity synchronization, and retains the
compatibility/memory limits of the other directions. Costs and source-derived
constraints are established. The separately authorized
[V2 CPU-prefetch comparison](../runs/v2-prefetch-baseline/report.md) is complete;
other candidate gains remain unmeasured and unaccepted.
Future separately authorized candidates follow the short monitoring-off,
performance-first [validation protocol](../planning/validation.md).

All listed GIFT source files are preserved inside this repository under
[references](../references/README.md).
Each directory contains original source bytes and a manifest with the exact
upstream revision, Git blob IDs and SHA-256 hashes. No upstream checkout or GitLab
access is needed to read these snapshots. They are selected files, not standalone
installable trees; import dependencies outside the selection are not bundled.

## Approaches at a glance

| Direction | Proposed mechanism | Fit with the current crop | Start reading |
| --- | --- | --- | --- |
| Kindulov: loss fusion | Fuse softcap and CE forward/backward; retain low-precision logits and row log-sum-exp instead of full FP32 intermediates; release completed chunks | The crop has no positive logit softcap; compiled baseline already avoids full-vocabulary FP32 CE buffers. Evaluate the remaining cap-zero backward algorithm and completed-chunk lifetime independently; preserve MTP and ordinary two-chunk loss with detailed monitoring disabled | [Source explanation](../references/kindulov/source/docs/softcap_memory.md) |
| Kindulov: local FFN recomputation | Recompute FFN/shared-expert internals rather than retain activations | Dense/shared FP8 is a candidate; interaction with existing per-layer SAC can add duplicate recomputation. Routed-expert variant explicitly rejects FP8 | [FFN implementation](../references/kindulov/source/torchtitan/models/common/ffn/__init__.py) |
| Fused hidden z-loss | Combine SwiGLU, FP8 quantization and gate/up hidden-z-loss gradients | Current crop applies z-loss to final norm input, not gate/up. This is not a direct fusion of the currently active regularizer | [FP8 design](../references/hidden-z-loss/source/torchtitan/ops/float8/README.md) |
| Vision-related runtime work | Compact supervised-token CE, input prefetch, GFSDP buffer/prefetch changes, frozen-weight storage and metric accounting | Mechanisms have different applicability; CompactCE rejects MTP, frozen storage requires frozen parameters, input prefetch requires the streaming loader | [Runtime source map below](#vision-related-runtime-work) |

## Kindulov: loss fusion and local recomputation

Pinned source: `mkindulov/muon_w_unfinished_configs`, revision
`b983803337dfcc34fdd571b38aaf5e01df583bbe`.
[Manifest](../references/kindulov/manifest.json).

Read the implementation in this order:

1. [L8 recipe](../references/kindulov/source/configs/experiments/gift_4_l8.yaml)
   and [design/limitations](../references/kindulov/source/docs/softcap_memory.md).
   This is the source author's recipe, not our default crop.
2. [Softcap wrapper](../references/kindulov/source/torchtitan/ops/softcap.py),
   [CE autograd wrapper](../references/kindulov/source/torchtitan/ops/softcap_cross_entropy.py)
   and [Triton forward/backward kernels](../references/kindulov/source/torchtitan/ops/_softcap_cross_entropy.py).
3. [Loss/chunk integration](../references/kindulov/source/torchtitan/components/loss.py)
   and [Unembedding raw-logit control](../references/kindulov/source/torchtitan/models/common/embeddings/token_debeddings.py).
4. [FFN recomputation](../references/kindulov/source/torchtitan/models/common/ffn/__init__.py)
   and [routed experts](../references/kindulov/source/torchtitan/models/common/moe/experts/__init__.py).
5. [Softcap/memory tests](../references/kindulov/source/tests/unit_tests/test_softcap_memory.py),
   [expert recomputation tests](../references/kindulov/source/tests/unit_tests/test_expert_recompute.py)
   and [loss tests](../references/kindulov/source/tests/unit_tests/test_loss.py).

The fused CE path preserves the softcap cast before FP32 CE and writes a
low-precision logits gradient. It supports first-order gradients; DTensor and
head-local hidden-z-loss cases fall back to the original path. Retain ignored
labels, reduction/normalization, MTP acceptance counters, head FSDP hooks and
supported hook behavior when adapting it; performance runs keep detailed
monitoring disabled. GPU rounding can differ; donor CPU tests
and reported L8 measurements do not establish crop accuracy or speed.

Actual G3 compiler evidence shows BF16 logits/gradients and only FP32 row
statistics in the baseline CE. The donor can still remove its one-hot-gradient
reduction; ordinary kernel count remains three. See the
[compiled contract](../runs/overnight-streaming-perf/k1a-compiled-contract.md)
for pinned generated code, the required compile adaptation and a confirmed
large-common-offset stability defect in the unchanged donor formula. Retain
maximum/log-denominator separately before accepting that algorithm.

Local FFN recomputation trades saved activations for repeated GEMMs and SwiGLU.
Check it against the crop's `experts_up` and `comm` SAC save policies before
combining them. Clip counts and auxiliary-loss totals must not be double-counted.
The BF16-only routed-expert option is not a candidate for enabling unchanged in
our dense+sparse FP8 baseline.

## Fused hidden z-loss

Pinned GIFT source: `gevko/swiglu-zloss-cudacherry`, revision
`81054a3383dc349a6690cc2b4aa0bce0dc73b5e9`.
[Manifest](../references/hidden-z-loss/manifest.json).
The Git source identifies Evko's implementation; it does not establish who was
going to present it at the meeting.

- [CudaCherry adapter](../references/hidden-z-loss/source/torchtitan/ops/float8/_cudacherry.py):
  imports `swiglu_quantize_zloss`, `fused_swiglu_zloss` and `ZLossState`.
- [Dense/shared FFN integration](../references/hidden-z-loss/source/torchtitan/models/common/ffn/__init__.py),
  [routed-expert integration](../references/hidden-z-loss/source/torchtitan/models/common/moe/experts/__init__.py),
  and [FP8 operators](../references/hidden-z-loss/source/torchtitan/ops/float8/).
- [Auxiliary-loss ownership/scaling](../references/hidden-z-loss/source/torchtitan/components/aux_loss.py)
  and [projection-z-loss preset](../references/hidden-z-loss/source/configs/presets/regularization/gift_4_projection_z_loss.yaml).
- [SwiGLU/FP8/z-loss regression tests](../references/hidden-z-loss/source/tests/unit_tests/test_fp8_swiglu_clip_zloss.py).

The mechanism adds the gate/up z-loss gradients before FP8 rounding. When gate/up
z-loss is enabled, the dense path retains BF16 inputs for backward rather than
only FP8 copies. This has a memory cost that must be included in an experiment.
For future local assessment, inspect gradient scaling, accumulated auxiliary
totals and SAC replay in the affected block; performance monitoring remains off.

The current [crop preset](../gift/configs/presets/models/gift_moe/gift_3_5/100b_crop8.yaml)
selects only `hidden_z_loss_patterns.norm` on input. Adding gate/up regularization
would change the objective, not merely accelerate its existing z-loss. A useful
comparison needs an explicitly agreed configuration with matching gate/up
regularization on both sides.

**External source gap:** the actual fused kernels live in CudaCherry, not this
GIFT tree. The preserved [Dockerfile](../references/hidden-z-loss/source/gift_docker/Dockerfile)
pins `CUDACHERRY_COMMIT=585bd50a641857ccae8b5ec691bb15791c9529e7`.
Those external kernel sources have not been located or copied here. The adapter,
call sites, tests and pin are available locally; this collection must not be
presented as a complete runnable kernel implementation. Obtain and preserve that
exact dependency source before a port; do not substitute an unverified installed
version.

## Vision-related runtime work

Pinned candidate source: `avtereshchenko/mr41-omni-perf`, commit
`860152d111e4bac25daecbe3cc0cd84c68c8af9c` (MR41 runtime publication).
[Manifest](../references/vision-runtime/manifest.json).
This is a relevant implementation collection, **not a confirmed copy of the
vision team's promised list**. The snapshot includes code present at that commit;
not every selected mechanism was first introduced by that commit.

| Approach | Local implementation and tests | Applicability / concern |
| --- | --- | --- |
| CompactCE | `CompactCELoss` in [loss.py](../references/vision-runtime/source/torchtitan/components/loss.py); [tests](../references/vision-runtime/source/tests/unit_tests/test_compact_loss.py) | Compute head/loss for selected supervised positions. Rejects MTP and TP/CP/full-DTensor; the crop has MTP1. Savings depend on ignored-label fraction; preserve token/sample normalization |
| Streaming prefetch | Original reference: `PrefetchIterator` in [prefetch.py](../references/vision-runtime/source/torchtitan/datasets/streaming/core/prefetch.py). Active V2: [CPU FIFO](../gift/torchtitan/datasets/streaming/core/prefetch.py), [loader](../gift/torchtitan/datasets/streaming/core/dataloader.py), [tests](../gift/tests/unit_tests/test_streaming_prefetch.py) | Bounded materialized CPU batches overlap finite-epoch production with consumption. Default depth is zero; candidate depth is 32. The consumer owns epoch entry and delivered scalar checkpoint state. Queue storage, GIL, lifecycle and epoch-boundary costs are reviewed in the [V2 report](../runs/v2-prefetch-baseline/report.md), which owns current verification and performance results |
| GFSDP buffers and prefetch chains | [parameter groups](../references/vision-runtime/source/torchtitan/distributed/gfsdp/_param_group.py), [state](../references/vision-runtime/source/torchtitan/distributed/gfsdp/_state.py), [omni chain wiring](../references/vision-runtime/source/torchtitan/models/gift_omni/parallelize.py), [data-pool tests](../references/vision-runtime/source/tests/unit_tests/test_gfsdp_data_pool.py) | Crop already uses GFSDP2. Compare mechanisms with current source before porting; retained buffers, all-gather overlap and backward order must fit GIFT MoE, SAC and ordinary loss backward |
| Frozen BF16 storage | [Design](../references/vision-runtime/source/docs/frozen_parameter_storage.md), [utils](../references/vision-runtime/source/torchtitan/tools/utils.py), [integration tests](../references/vision-runtime/source/tests/integration_tests/frozen_parameter_storage.py) | Helps frozen components; no frozen-component saving established for full crop pretraining. FP32-to-BF16 storage rounds weights and affects checkpoint precision |
| Metric accounting | [Metrics design](../references/vision-runtime/source/docs/metrics.md), [trainer source](../references/vision-runtime/source/torchtitan/callbacks/sources/trainer_source.py), [trainer](../references/vision-runtime/source/torchtitan/trainer.py), [tests](../references/vision-runtime/source/tests/unit_tests/callbacks/test_input_token_throughput.py) | Input-token counts share the loss-count collective. Keep input versus supervised-token rates distinct; instrumenting metrics is not itself a proven speedup |

## How to use this collection

Start with a concrete bottleneck in the default crop. Select one mechanism,
compare the saved source with the active `gift/` implementation, and identify
which parts are already present. Port only the needed behavior into an experiment;
never copy a whole donor configuration over the baseline. Keep model dimensions,
FP8, SAC, microbatch, MTP and token-drop settings fixed unless that experiment
explicitly studies a change.

The saved tests are starting points, not executed validation of these snapshots.
Start future authorized comparisons with a short matched monitoring-off
baseline/candidate performance pair. A single pair is provisional; repeat
promising or ambiguous results when needed. For a candidate worth retaining,
then compare the smallest affected block on matched inputs and needed weights/
state, including affected backward if changed. Analyze precision and rounding
and use a higher-precision local reference when needed; no whole-model gradient/
Adam gate or universal tolerance remains. No experiments or training-source
changes were made while assembling this inventory.

## Other hypotheses retained from earlier reviews

These are source-inspection questions, not measured improvements. The earlier
700B-specific analyses remain in [Git history](../references/README.md); their
settings and performance expectations do not describe the crop.

| Approach | Current source starting point | What must be established |
| --- | --- | --- |
| Fuse linear projection with CE | [Loss/head integration](../gift/torchtitan/components/loss.py) | Larger boundary than Kindulov's softcap/CE fusion; preserve head sharding hooks, FP32 accumulation, MTP, ignored targets and normalization. External linear-CE kernels need pinned source and compatibility review |
| MLA attention backend | [Attention kernels](../gift/torchtitan/models/common/attention/softmax/kernels/) | Actual backend and conversion cost for QK192/V128; supported forward/backward, causal semantics, compile and SAC before a backend comparison |
| GFSDP reshard/prefetch | [GFSDP](../gift/torchtitan/distributed/gfsdp/) | Exposed all-gather/reduce-scatter waits versus live-buffer memory; compare already present behavior before porting vision runtime changes |
| MoE metadata and quantization overlap | [Experts](../gift/torchtitan/models/common/moe/experts/), [FP8](../gift/torchtitan/ops/float8/) | Whether host count transfers or synchronization are exposed on the critical path; preserve capacity, ordering, scores, scales and collective dependencies |
| GDN recomputation | [FLA implementation](../gift/contrib/flash-linear-attention/fla/ops/gated_delta_rule/) | Distinguish block SAC replay from internal backward recomputation, measure saved-state/memory tradeoff and preserve scan/RNG semantics |
| Norm/residual fusion | [Norm implementations](../gift/torchtitan/models/common/norms/) | CuTe is already enabled in the crop. First show a remaining unfused boundary; do not count enabling existing CuTe as a new candidate |

Priorities and model-shape/token-drop diagnostics live in
[the editable work plan](../planning/README.md), not in a second backlog.
