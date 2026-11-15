# Independent integration and runtime-cost review

Subsequent owner decision: accept the investigated MTP/FP8 A/A variability and
stop further MTP/layer-0 numerical investigations. The measured differences
below remain evidence; their former numerical gate no longer blocks the campaign.
Remaining numerical verification is limited to embeddings.

Reviewed 14 November 2026 against root `2ada87125e433e1f76c4cb2bfad244b641a78a93`
plus the integrated working tree. This consolidates the original FFN review,
owner-storage review and targeted standard-autograd correction. The reviewer
ran no GPU operation, training, installation or test suite. Only this review
artifact was written; existing execution evidence is reused.

## Verdict and finite acceptance

No material source correctness defect was found in the reviewed PP1/TP1/CP1,
dense-FSDP8/EP8 crop path. The current owner backward returns an independently
owned FP32 Ragged gradient to ordinary autograd. Direct VJPs and parameter hooks
are supported by the new tests; earlier manual `.grad` mutation and private
engine/hook restrictions are superseded. The FFN buffer and state integration
contracts accepted below remain accepted.

The full-shape owner numerical check closes its specified lookup/gradient
comparison, and the layer-0 native FP8 block evidence supports numerical
preservation within the measured two-update scope. Neither result establishes
speed, full native allocation peaks, convergence or combined feature acceptance.

Finite review checklist, completed:

- Trace the three independent default-off controls through wrapping, compute,
  accumulation, clipping, AdamW, EMA and persisted state; reuse the accepted
  unchanged FFN review.
- Review current owned-gradient lifetime, ordinary autograd behavior, stream
  waits and unchanged borrow/event registry; reconcile the previous restriction.
- Interpret all eight layer-0 numerical JSON records and full-shape owner
  evidence against their actual harness boundaries.
- Reconcile current memory/runtime costs and name the remaining existing gates.

## Current integration contracts

| Contract | Inspected implementation and disposition |
| --- | --- |
| FFN callable/gradient boundary | `distributed/ffn_overlap.py`, common `transformer_block.py` and `mtp.py` retain the registered tree and w1/w2/w3 owner. Eager orchestration exposes real prefix/FFN/suffix compiled calls. MTP retains hidden/embedding concat order, shift/zero tail and both derivatives. No finding. |
| SAC and compilation | `activation_checkpoint.py:274` checkpoints inside individually compiled regions using the original block FQN and resolved policy. The ordinal `mm_half` policy is explicitly rejected. `compile.py:99` compiles each region; `EagerWindow` includes unregistered prefix/suffix holders. No outer checkpoint substitution was found. Native layer-0 FP8/SAC/Inductor evidence is now available below; MTP and combined native gates remain separate. |
| Existing FFN buffer budget | `_state.py:237` removes the child only from slot dealing, reconstructs the merged per-dtype padded budget, and partitions parent at byte zero and child after the parent's padded range. `_state.py:323` rejects excess split padding; `GFSDPBufferContext.region` only narrows existing uint8 storage. Alternating slots and the private head gradient bucket remain. No duplicate staging allocation. |
| Aliasing and completion | `_param_group.py:42` compares byte intervals within a storage context. Whole-slot and region owners/events remain separate; unshard, packing and prefetch wait intersecting release events. Whole-slot reuse visits both regions; disjoint regions can proceed independently. `fully_shard.py:592` also waits intersecting readers before backing replacement. Construction occurs during lazy init outside compiled compute. Reused review: no finding. |
| Owner routing/precision | `row_owner_embedding.py` retains signed-int64/range/device validation, dynamic counts, unique requests, token reconstruction and invocation-local routing. Forward uses disjoint FP32 selection/BF16 send ranges and reuses the dead selection prefix for return. Backward uses native BF16 source-local coalescing, lossless BF16 transport, separate FP32 widening for each source, and FP32 owner SUM. No world-size division or second MTP lookup. |
| Ordinary autograd | `_OwnerLookup.backward:408` zeros an owned FP32 `[rows_per_owner,D]` tensor and adds each source directly into it. Lines 441–448 wrap its flat storage in the unchanged weight spec and return `(grad_weight, None, None)`. No borrowed view escapes; no `.grad` mutation/private engine/hook rejection remains. The existing Ragged `detach`, `clone`, `add_`, `add` and `copy_` dispatches preserve local FP32 storage and metadata for accumulation. The existing `once_differentiable` boundary remains. |
| Stream readiness | `_communication:200–211` orders communication after caller inputs and makes the caller wait after all owner work, including the returned gradient. The returned local allocation is additionally recorded on the caller stream. Borrow release is after final scratch use. `_wait_for_post_backward` preserves newer borrower events by clearing only its exact reduction event. No new host/device synchronization was introduced by this correction. |
| GFSDP/state integration | `parallelize.py` excludes the final owner Parameter from every wrapper/root, omits the old embedding unit and attaches the root communication context. Ragged clipping/fused AdamW consume the unchanged FP32 representation. `gfsdp2.py`/`_dcp.py` cover EMA seed/update/swap/backing selection, flat main/EMA views and missing-EMA reseeding. These unchanged contracts and prior FFN findings remain accepted; no persistent BF16 cache needs invalidation. |

## Current source identity and owner execution evidence

The owner source SHA256 is
`f3a01c6cf99b981db7b4ef66866fe46259af84e9d2c21e2044e9ae18ff1696fd`;
its test SHA256 is
`fb9c08e313c202b58fd7f8c8ab714e943ea0254bb46d8ea58ff0fd88dc57b990`.
All 13 entries in [`owner-standard-autograd-source.sha256`](owner-standard-autograd-source.sha256)
passed `sha256sum -c` during this review, before later documentation updates.
The current source identities supersede historical owner manifests.

[`owner-standard-autograd-eight-rank-nccl.log`](owner-standard-autograd-eight-rank-nccl.log)
records explicit `backend=nccl mesh_device=cuda cuda_initialized=True` on all
eight ranks; the coordinator records launch exit 0. The frozen test entrypoint
runs routing, borrowing and state checks, including native BF16 coalescing,
32 backward calls, reset-to-None and zero-in-place, pending collective reuse,
no scratch/gradient alias, transforming parameter hooks, post-accumulate hooks,
and `autograd.grad` with both absent and existing unchanged `.grad`. It also
covers clipping, untouched-row dense Adam state, EMA and flat DCP next-update
behavior. This is GPU/NCCL evidence, distinct from earlier CPU/Gloo checks.

The [full-shape numerical results](embedding-numeric-results.md) and
[per-rank JSON](embedding-numeric-results.json) exercise real fixture IDs at
V=128256, D=4096, B=2, S=4096, eight H100/NCCL ranks, seed 2048. Matched
nonuniform BF16 native `/32` and MTP `*0.3` derivatives are combined before
ordinary backward. Lookup outputs and FP32 local gradients are exactly equal
on all eight ranks for first backward, second accumulation and reset-to-None;
nonfinite count is zero. No universal tolerance was applied.

The launch exited 1 only in the rank-zero Markdown formatter after complete
JSON was saved. The retained [launch log](embedding-numeric-launch.log) and
results state this failure; Markdown was repaired offline without rerunning
GPU work. Numerical evidence remains usable; the original launch is not
reported as a clean process success.

## Native layer-0 FP8 numerical interpretation

Reviewed all eight `artifacts/raw/ffn-components/layer0-check-002/rank*.json`
records (for example [rank 0](artifacts/raw/ffn-components/layer0-check-002/rank0.json)).
They identify actual FP8Linear w1/w2/w3, B2/S4096/D4096, selective SAC,
Inductor, FP32 masters/reductions, native external-gate GDN, 32 microbatches
per update and two clipped fused-AdamW updates. Software is torch
`2.15.0.dev20260822+cu130`, CUDA 13.0, NCCL 2.30.7, H100 80GB.

The executed harness SHA256 is
`fc57cff24713f285a675169a15956350e4f070981ff686812586fa1beb3aee1a`.
The current FFN, `_state`, `_param_group`, `_collectives`, `_tensor` and backend
source hashes match the records. The harness subsequently changed; retain the
executed receipt identity when interpreting these results. `numeric_check`
constructs A and AA with `split=False` and B with `split=True`, validates equal
initial FP32 logical parameters, then accumulates 32 backwards. The top-level
receipt's `variant=control` and false split config describe its initial outer
configuration, not internal B. This source trace supports B activation for
this check, but future split timing/diagnostic receipts must prove actual
activation explicitly; missing INFO markers are not supplied by this result.

A/A is exact for every captured tensor in both updates. A/B is exact for the
captured output and hidden input gradient, with no nonfinite values anywhere.
Those activation snapshots cover the final microbatch of each update; parameter
gradients cover all 32 accumulated microbatches. Parameter/optimizer snapshots
reconstruct `full_tensor()` on each rank: repeated per-rank records must not
be summed as distinct model elements.

| Unique full-tensor comparison | Update 1 | Update 2 |
| --- | ---: | ---: |
| Unequal pre-update FP32 gradient elements | 62 across 13 tensors | 81 across 12 tensors |
| Maximum absolute gradient difference | 4.8828125e-4 | 3.0517578125e-5 |
| Maximum per-tensor gradient relative L2 | 1.9269e-9 | 7.7182e-9 |
| Unequal post-Adam FP32 weight elements | 7 across 5 tensors | 23 across 12 tensors |
| Maximum absolute post-Adam weight difference | 9.3132e-10 | 3.7253e-9 |
| Unequal optimizer-state elements | 112 | 185 |
| Maximum optimizer-state relative L2 | 2.8031e-9 | 1.3028e-8 |

The update-2 total of 88 pre-update unequal elements includes 81 gradients
and 7 inherited weight differences; its 208 post-update total includes
23 weights and 185 optimizer elements. The model has 202,424,512 parameters.
Both A/A and B report identical clipping norms, 1,220,286.5 and 1,216,632.5.

Affected update-1 gradients include qkv convolution, ba projection, gated-norm
weights and one element of FFN w1. The largest difference occurs in the
convolution (three elements); w1 has one difference of 9.5367e-7 with relative
L2 3.6870e-12. This is not an exclusively FFN-weight discrepancy.

Inspected `_collectives.reduce_scatter` retains FP32 communication followed by
FP32 copy/add to persistent shard gradients; `_param_group.post_backward`
retains its FP32 staging/addition path. Crop wrappers set divide factor 1.0,
so this remains SUM semantics. Splitting changes collective packet/group sizes,
shard packing and compile boundaries. FP32 reassociation across those boundaries
is a plausible explanation for sparse finite differences and the tiny subsequent
Adam-state differences; these records do not isolate or prove that cause.
No reduced-precision accumulation or objective change was found in source.

**Scoped decision:** accept this layer-0 block check as evidence of preserved
supported numerical semantics at the measured precision and two-update scope,
with the differences retained explicitly. Exact A/A, exact captured BF16
outputs/input gradients, extremely sparse FP32 differences, tiny measured state
effects and unchanged arithmetic contract support that decision. No new
universal FP8 threshold or mandatory causality experiment is justified by this
evidence. This does not establish exact gradients, every-microbatch activation
parity, MTP numerics, long-run quality, or all-three native behavior.

## Runtime costs and memory

The [storage record](owner-storage.md) owns detailed formulas and the current
padded per-dtype calculation. The source computes actual group maxima; it does
not preserve an obsolete ordinary bucket by hardcoding 2004 MiB. With owner
excluded, the ordinary FP32 `None` bucket is source-derived as 937,084,928 bytes
(893.673828125 MiB), including split-parent/child recombination. Private
`lm_head` remains 2,101,346,304 bytes (2004 MiB), independent of those controls.
Scratch chooses an actual initialized buffer and checks capacity every call;
no recreation/growth/fallback was introduced.

Current forward worst-case borrow is 1536 MiB. Backward borrow holds selected
BF16, received BF16 and one-source FP32 conversion: 64+512+128 = 704 MiB worst
case, plus alignment. The former 250.5 MiB dense borrowed suffix is now a newly
owned gradient. It is allocated while the native 1002 MiB BF16 dense result is
still live, and can coexist with a previous accumulated `.grad`. Ordinary
autograd may steal/copy the first result and adds later results. Allocator churn,
zeroing, extra live storage and accumulation must therefore be measured rather
than described as free. Owned storage is the explicit requested tradeoff.

Native BF16 coalescing still has approximately 140.8 MiB FP32 partials at
8192 occurrences plus index/segment scratch; other dispatch branches and a
contiguous incoming-gradient copy remain possible. Metadata is invocation-owned
(about 640 KiB conservative crop bound). Sorting/unique, the count-matrix host
read, row exchange, per-source FP32 additions, imbalance and communication-stream
serialization remain actual costs. The correction adds no collective or
host-value read; it changes allocation/ownership and autograd submission work.

Full-shape owner memory evidence after matched cache cleanup reports maximum
warmed allocated peaks 5,781,722,112 bytes owner versus 8,950,373,888 baseline,
and reserved 7,648,313,344 versus 9,003,073,536. This resolves the fixture's
previous reserve-premium concern under the stated equal cleanup policy. Its
single embedding/GFSDP-stem boundary is not the complete crop, final per-dtype
pool inventory or a performance measurement.

FFN cost risks remain extra collective startup/packing, lost fusion,
Python/autograd/compile boundaries, prefetch/gather retention, live BF16
boundary activations (64 MiB per B2/S4096/D4096 tensor), bounded interval scans
and event waits. There is no explicit new host sync or tensor copy in the
registry. Native timing and matched lifecycle peaks remain the deciding gates.
Device compute, host submission, host/device waits, copies/communication and
integrated complete-step latency are not quantified by this source review.

## Reused historical evidence and remaining gates

The original [buffer-regions manifest](buffer-regions-source.sha256) passed
its review-time checksum check. [CUDA/NCCL regressions](buffer-regions-cuda-nccl.log)
record three passes on two ranks with actual gather/reduce-scatter layouts,
pending-reader overwrite protection, BF16/AOT/SAC accumulation-32 continuation,
exact gradients, AdamW and EMA. Both variants used 383,104 pool bytes/rank;
co-resident tiny-root peaks do not establish full-crop parity. The subsequent
[owner helper regression](owner-storage-ffn-cuda-nccl.log) also passed three
tests in 32.18 seconds. This unchanged FFN execution evidence is reused.

Earlier [CPU integration](buffer-regions-integration-cpu.log) and
[owner CPU checks](owner-storage-cpu-final.log) remain diagnostic evidence.
Early CPU/Gloo comparison failures on PREMUL_SUM/shared-view checks, and the
first owner GPU launch's relative-FileStore failure, remain failures. The
absolute-path [historical owner NCCL rerun](owner-storage-eight-rank-nccl-2.log)
was successful but exercised the superseded manual-accumulation design.
Its performance results in `embedding-component-results` and hook/VJP
restrictions do not establish current standard-autograd acceptance or costs.
The [initial private-parent record](buffer-regions.md) owns its superseding
FFN design; earlier private-parent measurements are historical.

Existing remaining gates, without expanding the experiment:

- Complete current-standard-autograd owner component F+B timings/peaks and
  interpret the actual timer boundaries; historical manual-backward timings
  cannot substitute.
- Resolve the failed MTP A/A reproducibility gate before attributing B/A
  differences or accepting native combined numerics. The coordinator reports
  `artifacts/raw/ffn-components/mtp-check-001` completed with exact initial
  parameters but first-update A/A parameter/both-input-gradient maximum
  absolute difference 0.125 and relative L2 up to about 2.212e-4; final-microbatch
  output is exact. Post-Adam weights differ by up to about 2.04e-3, and update 2
  diverges further. B/A differences cannot be attributed while A/A fails.
  Values are finite. This is a coordinator-supplied new receipt, not an
  independently reviewed MTP diagnosis. The parent owns a separate bounded
  investigation. Complete MTP components and all-three accumulation-32 native
  behavior only after this unresolved gate is addressed.
- Inspect initialized full native per-dtype allocation inventory and matched
  lifecycle peaks for the final owner/FFN combination. Source-derived pool
  arithmetic and isolated embedding peaks do not close this gate.
- The specified matched unprofiled native 10+10 comparison remains gated by
  unresolved numerical acceptance, including MTP A/A. This scoped review does
  not authorize it from the owner/layer-0 partial passes. Once the existing
  prerequisites and authorization are satisfied, inspect cadence and repeat
  only when noise or a concrete decision requires it; preserve accumulation,
  the LR schedule and monitoring-off recipe.
- Complete bounded MCP Perfetto evidence for all-rank readiness, FFN early
  reduce-scatter overlap, safe full-slot reuse, removed embedding table
  collectives and combined critical-path interactions; preserve affected
  clipping/EMA/checkpoint/next-update and lint gates.

No speedup, complete-crop memory parity, convergence, long-run stability or
PP4/EP32 support is accepted by this review. No additional experiment was
launched solely to increase evidence volume.
