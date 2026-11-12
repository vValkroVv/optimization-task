# Numerical probe independent integration and runtime-cost review

**HOLD for the one-update native GPU readiness probe.** Two correctness findings
must be resolved before that probe. One material runtime-cost finding also needs
resolution or explicit acceptance with corrected documentation. This is a review
of staging, not numerical acceptance, production acceptance, or fault clearance.

## Scope and finite checklist

Reviewed 2026-11-12 using `code-rules` and `perf-experiments`. The assignment
authorizes this review directory only. No training-source edits, GPU queries or
launches, installs, cleanup, publication, or subagents were performed.

Acceptance checklist derived from NUMERICAL-DESIGN/006 and retained 005 native
interfaces:

1. Trace callback and input capture through four backwards, native clipping,
   router container hook, inner AdamW, scheduler and batch-end capture.
2. Trace native S0 serialization, full restore, deferred RNG and retained state.
3. Check direct payload/readback, all-rank completion, cap/reserve accounting and
   failure propagation.
4. Review enabled and absent-observer runtime costs and distinguish remaining
   target-runtime gates from producer implementation.

The review covered the complete staged observer, schema, thin trainer binding,
continuation change, tests, README, REPORT and APPLY; the 006 contract and 005
native interfaces; relevant source manuals; and actual unchanged native trainer,
observer-state/dispatch, optimizer container, router hook, clipping, checkpoint,
auxiliary-loss and loader/scheduler state helpers.

Reviewed patch SHA256:
`ebfa4b67cf9f587f6f52a67d0649bdfefb2d6b0cbe0653901fbbcc47aae2ab5d`.
Staged observer SHA256:
`9f0775cd4481912fa9d04f97cbd7991f0bf5852ed0c361239aab6e5391c335a2`.

## Consolidated material findings

### R1 — P1: correct native restore fails the auxiliary scale equality check

Location: staged `numerical_probe.py:856–879`, `1494–1495`.

`_buffer_inventory` includes every public direct tensor attribute, including
`aux_loss_scale`. `after_load` requires the entire buffer/direct-attribute list
to equal its S0 observation. Actual `regularization.wire_aux_loss` initializes
this scale to one and assigns it as a plain attribute. It is absent from the
model checkpoint. Actual `AuxLossTracker.weigh` replaces its value with the
microbatch token share, before each forward. Every target MoE declares aux terms
in its constructor, so this attribute exists regardless of coefficient values.

For the supplied batch8/global256/accum4 workload, the completed last microbatch
has scale near 1/32 (the exact value depends on valid-token counts), while the
fresh restored model starts with scale one. The current equality rejects a
valid S0 restore before the observed update can begin.

[CPU reproduction](repro_aux_restore.py) uses the actual `AuxLossTracker`,
`declare_aux_loss`, `wire_aux_loss` and staged `_buffer_inventory`, plus native
`state_dict`/`load_state_dict`. It confirms absent checkpoint ownership, scale
0.03125 versus 1, failed current equality, and equal inventories after the next
native `weigh`. [Output](repro_aux_restore.log), exit 0.

Required resolution: retain the inventory but distinguish this explicitly
reconstructed per-microbatch value from state that must equal S0 at load time.
Verify its actual value before forward after native weighing. Do not restore or
modify model math to make the observation pass, or broadly ignore unknown
nonpersistent state. A fixed behavior regression must exercise this native
producer/restore boundary.

### R2 — P1: failure-file writes can prevent collective failure publication

Location: staged `numerical_probe.py:2135–2153`; analogous unguarded failure
writes at `1409–1420` and `1519–1531`.

`batch_end` catches a storage/write error, then writes a failure JSON through
the same budget and filesystem outside a protective guard. If the cap, reserve,
quota or filesystem remains unavailable, this second write raises before
`_publish_phase`. Other ranks can already be waiting in its collective. The
intended explicit all-rank failure publication is bypassed, and the original
error may be replaced by the failure-record error. The active writer also loses
the intended normal close/abort path. Elastic process termination or a timeout
does not establish this recorder's collective error-handling contract.

[CPU fault injection](repro_failure_publish.py) invokes the actual staged
`batch_end`, injects failure at the native-state write and subsequent failure
record reservation, and confirms that it escapes with **zero** phase
publications. [Output](repro_failure_publish.log), exit 0. This isolates the
control-flow defect; it is not a distributed or NFS validation result.

Required resolution: make secondary failure records best effort, preserve the
first error, close/abort the owned writer, and reach the same collective status
boundary on every live rank even if diagnostic disk writes fail. Apply that
policy to each equivalent boundary. Verify the repaired behavior with fault
injection that fails on the current handler and passes on the correction.

### R3 — P2: repeated post-update hashes add three full-family D2H passes

Location: staged `numerical_probe.py:2022–2040`; inherited recorder path
`_inventory` called by `before_optimizer_step`.

`_TensorPackWriter.add` already copies and computes the exact domain-separated
payload hash. `after_optimizer_step` then calls `_logical_digest` on the device
P, m and v again with the same header. There is no intervening native update.
The existing returned entry hash can supply this metadata without a second
device read.

Using 006's nominal family F=12.168153763 GiB globally, the present large-family
D2H path is approximately **11F/update = 133.85 GiB**: five stored families,
three P/m/v preclip inventory hashes, and three redundant post-update hashes.
The redundant pass alone is **36.50 GiB/update**, or 365.05 GiB over ten updates.
Step scalars, inputs and buffer inventory add further traffic. README's
five-family movement description understates this implemented path; five
families accurately describes the main persisted bytes, not all device reads.

Required resolution: reuse the hash returned by the payload writer for the
unchanged endpoint, or explicitly accept and accurately document the added
traffic and its purpose. Confirm endpoint/payload hashes remain identical in
the affected existing CPU behavior test. No GPU time or speedup is inferred.

## Integration and cost conclusions

The thin bindings otherwise follow the inspected target boundaries: prepared
inputs are copied before consumption without retaining GPU references; preclip
follows four native backwards; inner optimizer hooks follow the existing
container router update; post-update capture follows native AdamW and scheduler.
The actual native returned norm is captured. Explicit DTensor gradient
placements and AsyncCollectiveTensor wait are present. There is no new norm
reduction, tensor all-gather/redistribution, H2D optimizer reconstruction, or
replacement clipping/optimizer arithmetic. Used LR and next LR are separate.

The loaded pending RNG is compared against serialization-time bytes, then the
one-shot restore is checked after the native first fetch. The source path check,
native full-state hash checks and `load_only` contract remain important; actual
DCP execution is still unverified. The recorder does not implement later A/A
elementwise/delta metrics or criteria. No hash result replaces those metrics.

With the callback absent there are no numerical observer hooks, tensor copies,
hashes or new device drains; the campaign binding retains a small Python branch
in prepared-input processing. Enabled observation has blocking local tensor
D2H copies, repeated CPU hashing, synchronous writes/fsync, CPU readback, explicit
preclip/post-update drains and small object collectives. For one inner optimizer,
each observed update adds seven `all_gather_object` calls and a phase barrier.
There is no basis for a timing or peak-memory claim. CPU staging/readback is
bounded by a tensor, with temporary byte-buffer copies; measure actual largest
tensor and host peak on the target. No consumer previous-P cache is implemented
here; its later explicit bounded exception and delta requirement remain intact.

The locked reservation counter includes all accounting roots, pending reserved
bytes and partials, with reconciliation at completed phase boundaries. Native
S0 storage admission currently occurs in `after_checkpoint_save`, after the
native write: the launch-side preflight must therefore establish the projected
S0 plus reserve before permitting that save. This is an explicit remaining
preflight contract, not proof supplied by the observer. Actual all-rank NFS
locking, bytes and free-space behavior remain target gates.

## Evidence and open acceptance items

Reused staging's reported nine passing CPU tests, full Ruff/format, syntax,
patch-application and hash checks; did not rerun the broad suite. The two new
targeted reproductions above are successful demonstrations of broken behavior,
not passing fixed regressions. Both emitted only the known `pynvml` deprecation
warning. No source change has been made to fix these findings.

After R1/R2 are fixed and R3 resolved, affected behavior checks and integration
review of those fixes remain necessary. The actual one-update eight-rank target
probe, native S0/load-only/RNG execution, FP8/router/persistent-state inventory,
CPU lossless payload readback on native records, storage/host peaks, graph-on
observer effects, ten-update generation, calibrated production A/A criteria,
elementwise gradient/delta comparisons and bidirectional continuation are still
open. Source-build attestation and existing CUDA715 fault gates remain separate;
the earlier timeout is inconclusive and grants no fault clearance or retry.

This report makes no production-readiness or numerical-equivalence claim.
