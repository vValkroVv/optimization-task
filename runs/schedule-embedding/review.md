# Combined integration and runtime-cost review

## Verdict and boundary

The reviewed MTP1 implementation has no confirmed storage-lifetime or arithmetic
defect from source inspection. One support-validation finding requires a focused
fix. Production acceptance remains open until the requested integrated native,
component, memory and profile evidence is complete. This review is not evidence
of a speedup or unchanged production peak memory.

This is one independent review of the combined GFSDP/embedding changes against
`e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f`. Acceptance for the review is finite:
trace actual configuration and consumers; inspect aliases, stream events and
autograd precision; reconcile available focused results; identify actionable
findings and exact remaining gates. No GPU work, source writes, package changes,
or broad test reruns were performed by this reviewer. The unrelated dirty
capacity implementation, tests and campaign documents were excluded.

Reviewed tracked-source diff SHA256, before the requested validation fix:
`8f26d6ea6da3b3a369e60bd07349e9e46ddac9725cfab4619c1f2d061f65dec2`.
This covers configs.py, parallelize.py, _state.py, _param_group.py,
fully_shard.py and row_owner_embedding.py relative to that base. The new kernel
file SHA256 is
`a8474c8da675e48e6a36d6fbbe566315c8a3420a11b30f18576ee4edba43fc1c`.

## Consolidated finding

### R1 — Reject unsupported multiple-MTP transition requests (medium)

At review time, `parallelize.py:98-100` and `:277-284` accept any MTP container
containing layer `0`. The actual transition source at `:471-474` and replacement
head-prefetch targets at `:500-505` hardcode that layer. The unchanged
`models/common/decoder_blocks/mtp.py:MTP.forward` supports sequential multiple
heads, so the generic reverse chain would otherwise start with the last head.
The enabled feature therefore accepts a scheduling configuration outside its
MTP1 design and evidence. This is a missing support guard, not an observed MTP1
runtime failure.

The coordinator accepted the narrow fix: require exactly one MTP layer in both
`_validate_mtp_transition_request` and direct `apply_gfsdp` validation. Preserve
multiple-MTP behavior when the flag is off. Verify enabled multiple-MTP requests
fail before wrapping, the disabled validator remains a no-op, and the valid
MTP1 path still accepts the intended configuration. An extra native timing run
is unnecessary for a validation-only fix that leaves accepted MTP1 execution
unchanged. **Disposition at handoff: root-owned fix and focused verification.**

No other material source defect was confirmed during this bounded review.

## Integration and lifetime conclusions

- Both schema flags default off. The production model entry validates requests;
  `apply_gfsdp` independently validates direct callers and passes the embedding
  control to `RowOwnerEmbedding.from_embedding`. Flag-off retains the old
  explicit/implicit prefetch decision and native embedding operations. The
  backward implementation is reorganized even with the flag off, so D0 remains
  an actual runtime control to measure, not an inferred unchanged binary.
- `_state.py:_init_shared_state` deals and allocates original pools before
  creating the two aliases, and before group `lazy_init` binds collective
  outputs/inputs. `_alias_buffer_region` uses root-relative source offsets.
  Capacity, dtype bucket, shard rank/size, process-group identity, precision and
  shard alignment are checked. Only the matching last-layer dense group moves;
  the expert group is not remapped. Pool size calculation is unchanged.
- Inherited `GFSDPBufferContext.region`, `overlaps`, `conflicting_owner`,
  `wait_for_release` and `borrow` register aliases in the storage-root registry.
  Whole-storage embedding scratch consequently sees source and target aliases,
  including their live owners and delayed reader events. Events are waited on
  without clearing another context's event. A retained unreduced owner blocks
  borrowing or overlapping gradient packing instead of silently releasing it.
- `GFSDPParamGroup.reshard` records completion on the computing stream before
  clearing the data owner. `post_backward` records entry in PRE_BACKWARD,
  requires a real nonempty reduction, resharding and released data ownership
  before calling the new transition. The all-gather stream then waits through
  ordinary `unshard` on every overlapping release event. An unused root callback
  cannot trigger the transition. Existing pending-result/unsharded guards make
  later ordinary prefetch calls idempotent.
- `_OwnerLookup` preserves the communication-stream handoff, input
  `record_stream`, same-stream scratch borrowing, output handoff and returned
  owned zero-initialized FP32 local gradient. It retains ordinary autograd
  accumulation; no manual `.grad` mutation is added. Compact coalescing saves
  inverse IDs instead of original IDs plus unique IDs, while retaining requested
  local rows through backward. The compact result is separately owned and its
  NCCL read is submitted on its allocation stream before deletion.
- `_route` retains sorted unique requests, per-source splits, the count
  all-gather, host-visible validity/split read and request exchange. Forward
  value and backward gradient exchanges keep the same order/count/payload.
  Optimized validation clamps signed int64 extremes before routing; empty and
  unsupported cases retain coordinated handling. Kernels widen row/element
  offsets and counts to int64.
- The native compact guard matches the inspected pinned CUDA dispatch. For
  large inputs it conservatively excludes the low-occupancy atomic alternative;
  the ordered branch retains ten-row FP32 partials and final BF16 output.
  Monotone relabeling preserves sorted duplicate membership/order. The fallback
  retains full-vocabulary native backward plus unique gather. Small-input native
  dispatch is independent of vocabulary size. These conclusions are specific to
  the pinned PyTorch implementation, not an assurance for future versions.
  Sources: [Embedding.cu](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/aten/src/ATen/native/cuda/Embedding.cu#L240-L324),
  [EmbeddingBackwardKernel.cu](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/aten/src/ATen/native/cuda/EmbeddingBackwardKernel.cu#L286-L351).
- Owner-add kernels have one writer per requested row within a source because
  requests originate from sorted unique IDs. Sources launch in rank order on one
  stream. The final load/add/store kernel avoids the rejected atomic candidate's
  subnormal discrepancy. Direct lookup performs FP32 loads and one BF16 store,
  with no accumulation or extra BF16 intermediate. Local bit tests are the
  arithmetic evidence; distributed hooks/state evidence remains separately
  required.

## Final runtime-cost assessment

No parameter/gradient communication pool allocation, precision change or new
collective is introduced. Two bounded alias views add host objects, not GPU
storage. Pairwise retained-mode validation runs during initialization/control
changes. The extra aliases also enlarge **runtime** overlap scans and may add
event waits in inherited unshard, gradient packing and whole-storage borrowing;
their cost is not confined to construction. These waits preserve correctness
and do not host-synchronize. Their actual critical-path impact needs the native
comparison and trace evidence.

Embedding compact coalescing removes the full-vocabulary dense BF16 output and
reduces native partial storage when its dispatch guard succeeds. It still
allocates a compact native output and partials. The inverse replaces equal-sized
saved input-ID storage; the saved unique-ID tensor is omitted on compact calls.
Owner accumulation keeps the required dense owned FP32 gradient/zero-fill and
removes per-source conversion scratch and one launch per nonempty source.
Lookup removes the FP32 selected-row scratch and conversion-copy launch.
Routing replaces compatible preparation with two bounded CTAs while keeping
unique and the existing host read; it allocates safe IDs plus metadata rather
than introducing a persistent pool.

The hot path retains Python stage membership/shape checks and cached CUDA device
property access. Triton row counts are runtime scalars; width, block choice and
fixed mesh metadata still specialize kernels. First-use compilation and allocator
lifetimes can affect warmup and reserved memory. Source-derived byte reductions
cannot establish peak-memory parity, launch latency or end-to-end gain. Measure
host work, joined device span, communication/copies and integrated latency at
their specified scopes; do not add overlapping stream totals.

## Available verification and exact remaining gates

Reused evidence, without rerunning it:

- [GFSDP implementation checks](gfsdp-implementation.md): 69 CPU/Gloo passes,
  one CUDA-only skip and one explicitly deselected baseline-failing CPU state
  continuation; scoped syntax/lint checks passed. The source tests cover alias
  bounds, equal pool storage, overlap/mode validation, unused/stale callbacks,
  head targets and prefetch idempotence.
- [Embedding implementation checks](embedding-implementation.md): native compact
  BF16 parity including dispatch boundaries; direct conversion bits; corrected
  serial FP32 owner-add bits; routing signed limits/empty input; real production
  IDs with synthetic matched derivatives. The failed atomic and failed one-rank
  probes remain recorded. The coordinator additionally reported the final
  owner-add bit checks passing at widths 16 and 4096.
- [Current two-rank CUDA/NCCL log](gfsdp-cuda-nccl.log): **71 passed, 4 warnings
  in 80.95 s**. The executed suite includes delayed region readers,
  actual alias storage, prefetch idempotence, and accumulation-32 AOT/SAC
  baseline-DCP-to-transition continuation with exact gradients, parameter,
  optimizer and EMA checks. The measured tiny fixture widens only its MTP FFN
  for capacity and does not establish production EP8/FP8/Inductor behavior or
  the combined row-owner scratch path. Its coexisting-root memory receipts do
  not establish production peak-memory parity.
- Inspected runner corrections: `run.py:main` captures original hook builders
  before replacing them and restores them in `finally`; `measure.py` verifies
  byte-exact original capacity against its pinned hash and HEAD before canonical
  import. This avoids modifying contributor source. The coordinator's actual
  wrapper-main dry-check receipt is the executed harness evidence.

Required gates still open at this review handoff:

1. Resolve R1 with focused validation checks.
2. Eight-rank optimized owner routing/backward/hooks/autograd.grad and
   accumulation/reset/state verification using the corrected state-check stage
   propagation, including uneven, empty, skewed and invalid inputs. The local
   kernel tests and CPU fallback do not close this gate.
3. The combined SE native full-update smoke at production accumulation 32,
   selective SAC, FP8 and actual compile boundaries. It must cover whole-storage
   scratch borrowing alongside the new gradient alias, subsequent microbatches,
   reset and the unchanged layer-0 split. Separate S and embedding tests cannot
   substitute for this combined lifetime path.
4. Representative eight-rank component results before native timing: individual
   retained stages, real IDs/owner imbalance, ten warmup/ten samples,
   forward/backward, joined multi-stream event span, synchronized wall/host time,
   rank distributions/kernel counts and phase-matched allocated/reserved peaks.
   Verify actual activation and fallback selection, including compact dispatch.
5. Matched D0/S/SE native 10+10 complete-update measurements and any justified
   repeat needed to resolve noise/drift; per-rank memory/retries/OOMs, resolved
   configuration, activation, source identity and settling receipts. This closes
   the requested no-peak-growth and performance claims, if supported by results.
6. Separate native rank-0/1 update-21 profiles, original/import kernel-count and
   error reconciliation, and Perfetto evidence for both MTP gathers preceding
   head reduction, the real FFN-to-last-dense transition, norm packing's correct
   reader wait, unchanged collective counts/payloads, and no displaced layer-6
   or next-microbatch stall. Component/kernel counts alone do not establish
   those dependencies.

The coordinator owns these gates and their final result reconciliation in the
existing report/status records. No additional diagnostic framework or unrelated
audit is required by this review.

## R1 resolution

Resolved in `gift/torchtitan/models/gift_moe/_base/parallelize.py`: both the
public `_validate_mtp_transition_request` and direct `apply_gfsdp` guard now
require `len(mtp.layers) == 1` when MTP transition prefetch is enabled. The
flag-off validator remains an early no-op, leaving multiple-MTP scheduling
unchanged.

Focused CPU validation on 2026-11-15 passed: `test_gfsdp_row_owner_integration.py
-k mtp_transition` reported 9 passed and 13 deselected. Scoped Ruff
F821/F811/F841 and `git diff --check` passed. The earlier 71-test CUDA/NCCL
MTP1 suite remains applicable because accepted MTP1 behavior did not change;
it was not rerun for this support guard. Finding history above is unchanged;
this closes R1 only.
