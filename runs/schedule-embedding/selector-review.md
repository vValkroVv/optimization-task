# Native selector and revised matrix review

## Scope and finite acceptance

Independent integration and runtime-cost review for the revised seven-variant
measurement campaign. No GPU execution, source edits, cache changes, package
installation or broad correctness reruns belong to this review.

- Trace typed native configuration through `apply_gfsdp` to the existing
  `RowOwnerEmbedding.from_embedding` selector, including invalid combinations.
- Check exact historical A/B/C recipes, later-control isolation and the sole
  SE/SE-fused routing-stage difference.
- Check timing 140/40/100, separate diagnostic 21/20/0, all-rank completion,
  token/window validation, fresh output IDs and preserved cache/overlay policy.
- Check combined source/configuration/harness freeze coverage and actual
  post-initialization stage receipts.
- Reconcile focused owner verification and report material findings once.

## Inspected inherited contracts

The accepted implementation and its prior numerical/runtime review remain in
[review.md](review.md), [embedding-implementation.md](embedding-implementation.md)
and [gfsdp-implementation.md](gfsdp-implementation.md). This review concerns the
minimal new selector and harness integration, not a reopened kernel audit.

`row_owner_embedding.py:_resolve_optimization_stages` already canonicalizes
the selected immutable tuple, rejects unknown/duplicate stages and rejects
`optimize=True` with every explicit tuple, including empty. The factory stores
that tuple once. `RowOwnerEmbedding._route` chooses the existing routing
preparation only for selected, supported CUDA IDs; signed int64 validation,
sorted unique order, counts, request exchange and coordinated validity remain
on the established path. Compact/lookup/owner-add selections are independent.

`overnight-streaming-perf/measure.py:WindowMeasurement` retains CUDA drains at
window boundaries and lightweight host cadence at update boundaries. The new
stage-receipt hook is called when the measurement is constructed after Trainer
initialization and before the first update. It enumerates module types and
immutable tuples, writes one JSON file per rank and reads no GPU values.

Expected selector cost: construction-time Python validation and metadata only;
no additional tensor allocation, transfer, collective, synchronization, saved
autograd state, optimizer arithmetic or checkpoint state. Enabling three versus
four existing stages deliberately selects existing runtime paths whose native
timing/memory effects remain the campaign's measurement gates.

## Consolidated material finding

### R1 — Diagnostic gating prevents four-profile completion

The reviewed `campaign.py:_execute` compared `completed` for exact equality
with all seven timing names before each diagnostic (reviewed line 1009).
After the first diagnostic, the set also contains its name, so the next
diagnostic is incorrectly skipped. A diagnostic-only invocation also starts
with an empty set despite validating prior timing receipts, so it skips every
diagnostic. Both authorized execution routes therefore fail the requested
four-profile contract.

Required fix: track validated timing completion separately, test containment
of the seven required timing names, and seed the timing gate from accepted
same-round/same-freeze receipts for the diagnostic-only route. Verify the
actual executor with stub child launches for the full eleven-run route,
the four-diagnostic route, and a missing/failed timing control. Preserve
independent timing continuation and never accept an incomplete timing matrix.
The final executor tracks `timing_passed` independently, seeds it from
same-round/same-freeze timing receipts in diagnostic-only mode, and tests
timing-set containment before each diagnostic. Only successful validated timing
runs enter that set; diagnostic names cannot invalidate it. Disposition:
resolved by the owner and independently verified below.

## Prelaunch failures already corrected

- The capacity snapshot is task-untracked. The freeze now hashes it directly
  and obtains Git HEAD bytes using tracked `measure.CAPACITY_RELATIVE_PATH`
  (`gift/.../dispatch/capacity.py`). Both must match the pinned original
  capacity digest. Inspected at `campaign.py:_campaign_freeze`.
- `WindowMeasurement.__init__` has keyword-only `trainer`, and its real call
  site supplies `trainer=trainer`. The final receipt wrapper forwards original
  `*args, **kwargs` unchanged and restores the constructor in `finally`.
  The dry check uses a keyword-only fixture and checks restoration. Inspected
  against the actual constructor and `tracked_train` call site.
- The initially recorded launch command exported empty
  `CUDA_VISIBLE_DEVICES`, which hides all GPUs. The base child environment
  copies that value and neither its shell wrapper nor `/etc/profile.d/wm.sh`
  clears it. The final launch command unsets the variable after sourcing
  `wm.sh`; empty is confined to CPU dry checks. Native runtime preflight
  requires exactly eight CUDA devices before training.

## Stable handoff evidence and cost conclusion

The owner reports 9 selector/parser tests, one actual two-rank Gloo native
factory-wiring check, one H100 fallback/fused routing metadata equivalence
check, eleven native child-command dry checks, seven receipt/restoration
checks, seven runner-main hook checks, scoped Ruff and whitespace checks.
Commands and retained preflight failures are owned by
[selector-integration.json](selector-integration.json). These checks are
reused rather than repeated by the reviewer. No source/kernel arithmetic,
precision boundary, gradient accumulation, state format or communication
protocol changes are needed for the selector.

The campaign freeze covers the GIFT tracked/untracked snapshot, base config
hashes, FlashQLA source identity, three schedule files, production input
contract, executing runner/measurement/parser/compiler observer/streaming file
lock, historical plan and original capacity snapshot/HEAD. Source/config
checks occur before and after runs. Per-rank actual module/stage receipts
are required in completed-run validation. Timing and diagnostics retain
their separate cadence and token/completion contracts. Saved-D absolute
in-place cache reuse is uniform; new entries can still appear and do not
establish execution or settled timing.

## Targeted finding verification

The reviewer inspected the final gate and its three stubbed actual-executor
tests in [test_campaign_diagnostic_gate.py](test_campaign_diagnostic_gate.py).
All three passed when called against the final `_execute`: full eleven-run
order, diagnostics-only after matching timing receipts, and continued later
timings with diagnostics blocked after a timing failure.

To establish regression sensitivity without modifying files, the reviewer
recompiled `_execute` in process with only the original exact-equality gate
restored. The same full and diagnostics-only tests failed; the failed-timing
control still passed. The original function was restored afterward. The
check used `/opt/torchtitan-venv/bin/python3.12`, CUDA hidden, bytecode writes
disabled and temporary run-owned directories; no GPUs, model/cache files,
package state or source files were modified.

Final reviewed combined freeze reported by the stable owner evidence:
`3b3244ae1e759111b8240af929715074b751c946881202a9ff95e9f3b9918ffd`.
The review file is outside its executing source/configuration boundary.

## Final verdict

The native selector and construction-time receipt path have no confirmed
material defect in this bounded review. R1 and the retained prelaunch findings
are resolved. The minimal selector and revised harness pass this independent
integration/runtime-cost launch review. No further broad audit or new smoke is
required by these changes.

Before execution, the coordinator must preserve the final freeze, use the
corrected environment command and repeat the existing live all-eight-GPU
ownership/availability gate. Seven complete timing measurements, all-rank
stages, cadence/tokens/memory and four diagnostic traces remain campaign
execution gates; source review and prior short runs do not establish those
results. Separate profiles must establish routing absence/presence; cache
entries and selected-stage receipts alone do not prove execution.
