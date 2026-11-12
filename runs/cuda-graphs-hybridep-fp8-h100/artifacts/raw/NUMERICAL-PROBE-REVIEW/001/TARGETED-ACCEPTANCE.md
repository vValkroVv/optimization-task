# Targeted verification of R1, R2 and R3 corrections

**ACCEPT for one-update native GPU inventory/readback readiness preparation.**
This supersedes the R1/R2/R3 HOLD in [REVIEW.md](REVIEW.md) for the corrected
patch identified below. It does not accept a GPU result, numerical equivalence,
ten-update generation, production readiness, or CUDA fault clearance.

Corrected `NUMERICAL-PROBE-IMPLEMENTATION/001/numerical-probe.patch` SHA256:
`aa6ea97e4c6455c612a4913a0d624782a383a7fe6ade1342a13763f68002c6e2`.

Corrected staged `numerical_probe.py` SHA256:
`7ef86c1daa60c5d69cebec54f766da0676c700f57443acc3fb3f4ce6fc54bc6e`.

## Bounded verification

Reused the original full integration/runtime-cost review. Inspected only the
three findings' corrected helpers, binding/cleanup, native weighing consumer
boundary, load comparison, phase failure handling, endpoint hashes, related
regressions and revised cost documentation. No broad audit was restarted.
No training-source edits, GPU operations, installs, cleanup, or subagents.

### R1 — resolved

The direct inventory now marks only declared `aux_loss_scale` attributes as
reconstructed by `AuxLossTracker.weigh`. `_same_loaded_buffer_inventory` permits
their hash difference while requiring identical identity/layout and exact
equality of other entries. `after_load` uses this helper, preserves source and
loaded scale hashes, and retains every other state comparison.

The observe-only wrapper calls native `weigh` once, then records the actual
scalar after weighing and before the native forward. It verifies that every
declared module still points to the tracker's scale. Preclip capture requires
this observation for each microbatch. Close restores the borrowed method, and
the absent-observer path does not install the wrapper. This adds one scalar
host read per observed microbatch per model part with declared aux terms; no
tensor-pack family or training arithmetic was added.

The native aux restore regression passes: S0 scale 1/32 versus fresh restored
one, allowed narrow comparison, rejection of an unrelated direct tensor
mismatch, actual post-weigh value/identity capture, and wrapper cleanup. The
original [broken-behavior reproduction](repro_aux_restore.log) remains evidence
of the pre-fix failure; it was not rerun against changed code.

### R2 — resolved

Secondary failure records are guarded best effort and skipped without an
initialized budget. Original errors remain in collective status. Owned active
writers are aborted; failed boundaries close bindings. S0, initial-load checks,
verified-load writes, batch-end errors, and mid-step collective checks route
failures through phase publication. `_publish_phase` reaches its collective
status/barrier path even when no manifest can be written.

Targeted CPU regressions pass for secondary filesystem failure at S0, no-budget
load failure, quota failure after a primary native-state ENOSPC, writer abort,
first-error preservation, and phase publication without a budget. These test
actual corrected methods with mocked collectives. They establish the repaired
control flow, not eight-rank transport or NFS behavior. The original
[broken handler reproduction](repro_failure_publish.log) remains the pre-fix
negative evidence.

### R3 — resolved

Post-update endpoint metadata reuses each tensor writer entry's SHA256. The
second `_logical_digest` device read is removed. Native CPU accumulation and
readback regressions for one and ten updates pass, including equality between
P/m/v/step endpoint hashes and payload-entry hashes.

README and REPORT now distinguish five persisted families from approximately
eight large-family D2H reads per update: five persisted plus three preclip
P/m/v hashes. The corrected nominal movement is 97.35 GiB globally per update,
excluding inputs, buffers, scalars and state reads. This removes the earlier
redundant 36.50 GiB/update pass. No measured timing or peak-memory claim follows.

## Executed check

From the staged `gift/` directory:

```sh
python -m pytest -q -p no:cacheprovider \
  --basetemp=/workspace-SR008.nfs2/vkropotin/optimization-task/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NUMERICAL-PROBE-REVIEW/001/pytest-corrections \
  tests/unit_tests/test_campaign_numerical_probe.py \
  -k 'aux_loss_scale_restore or failure_record or phase_failure_without_budget or native_cpu_accumulation'
```

**7 passed, 7 deselected, 1 warning, 6.09 s**.
[Full output](targeted-corrections.log). The warning is the known PyTorch
`pynvml` deprecation notice. The seven deselected tests were outside these
corrections; reuse the owner's reported full 14-test, Ruff, formatting, syntax,
clean-apply, SHA and link checks rather than rerunning those gates.

## Preserved gates

The native one-update target execution remains unrun and must validate actual
eight-rank inventory/readback, native S0/full load-only restore, serialization
and one-shot RNG, persistent-state completeness, storage bytes and host peaks.
Launch-side admission for native S0 must precede its save; the coordinator owns
that preflight because observer admission happens afterward. Combined source
freeze/build attestation and existing CUDA fault restrictions remain separate.
This acceptance grants no automatic retry or launch authorization.

The later small auxiliary/router payload additions belong to the numerical
comparison contract and remain pre-ten-update gates. They do not block the
one-update inventory/readback preparation reviewed here. Ten-update generation,
calibrated production A/A criteria, elementwise gradient and FP64 delta metrics,
graph-on observer behavior, and bidirectional continuation remain open. Hashes
do not replace those later comparisons.
