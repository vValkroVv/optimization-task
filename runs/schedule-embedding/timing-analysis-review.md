# Timing adjudication integration review

Reviewed 15 November 2026 at repository HEAD
`e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f`, including the current untracked
analyzer and tests. This is one focused integration review of
[analyze_timings.py](analyze_timings.py) and
[test_analyze_timings.py](test_analyze_timings.py) against the timing interface
of [adjudicate.py](adjudicate.py), its native receipt producer and parser.
The diagnostic implementation being developed separately was excluded.

## Acceptance and result

Finite review checklist: authenticate the external summary and immutable raw
receipts; preserve original wrapper/process outcomes; compare timing proof and
fixture schemas with the actual producers; verify the route for all seven final
rows; run the focused CPU suite and scoped lint. The initial review required
changes; the focused follow-up below verifies all three findings resolved.
**PASS for the reviewed offline analysis integration.** Actual final timing
proofs and their analysis remain a separate execution gate.

The coordinator clarified the final policy during review: all seven rows use
`artifacts/raw/adjudication-140-004/<variant>/adjudication.json`, including a
fresh offline adjudication of retained raw A003 into `004/A` after every timing
attempt finishes. The historical `003/A-recovery-003` proof is not the final
analysis input. Do not pin an interim adjudicator method hash; the manifest must
pin the completed method used for all seven new proofs.

## Initial consolidated findings — resolved in follow-up

The descriptions below preserve the original defects and requested remedies.

1. **High: actual PASS timing proofs cannot enter the adjudicated path.**
   `validate_adjudication_record` requires raw FAIL/exit 1, no completion,
   `failure.json`, the exact nullable guard and a populated nullable exception.
   `_current_raw_receipt_hashes` also requires `failure.json`, and
   `validate_run_receipt` repeats the FAIL-only gate. The timing helper's
   `_validate` explicitly supports clean raw PASS with completion and no failure;
   for explicit-selector SE/SE-fused it emits `original_failure: null`,
   `known_strict_guard_failure: null` and `nullable_exception: null`.
   These are expected final rows. Implement a distinct strictly validated PASS
   branch under the same authenticated proof envelope: require status PASS,
   exit 0, no failure/nullable exception, consistent completion and raw/external
   summaries, clean stopped native process and resolved-config hash. Keep the
   exact nullable exception limited to the FAIL branch. Require all seven final
   manifest rows to use the pinned adjudication route so omitting a proof cannot
   switch a row to the less authenticated historical extraction path.

2. **High: the final gate rejects the real parser's empty error list.**
   `runs/overnight-streaming-perf/parse_results.py:summarize_run` emits `errors`
   as a list, including `[]` on success. The adjudicator and new fixture use this
   schema, but `validate_run_receipt` requires `summary.get("errors") == 0`.
   Consequently even the successful new adjudicated fixture cannot pass the
   full campaign gate. Accept the actual empty-list schema while rejecting any
   reported error; retain explicit historical compatibility only where needed.

3. **High: stage provenance uses the wrong type in the analyzer and fixtures.**
   `measure.py:_record_stage_selection` writes `source_sha256` from the schedule
   identity as a path-to-digest mapping; `adjudicate.py:_check_stage_receipts`
   requires equality to the frozen `schedule_source_sha256` mapping.
   `validate_stage_selection_receipts` instead requires one hexadecimal string,
   and extraction collects only strings into `embedding_stage_source_sha256`.
   All eight genuine receipts therefore fail, and SE/fused source identity
   becomes `None`. Validate and compare the actual frozen mapping, preserving
   all eight rank identities. The B fixture also claims one row-owner module,
   whereas both the native producer and adjudicator require zero for A/B/C and
   one for D0/S/SE/SE-fused. Match these counts and enforce them at the analysis
   boundary; self-consistent zero modules must not establish SE activation.

The new success test exercises extraction only, with empty cadence/window/token/
memory bodies; it cannot establish campaign acceptance. Add a small complete
seven-row synthetic campaign using actual producer schemas, including both
wrapper outcomes. Assert acceptance and preservation of originals, then test
targeted rejection of missing/wrong proof, incorrect run/path, nonempty errors,
stage map/count mismatch, nullable misuse and receipt/hash tampering. No live
adjudication is needed to test these integration boundaries.

## Verified boundaries and cost

The FAIL extraction path pins repair-plan SHA256
`1a193606963e2f3c7086a6afb1d6d56ad78692dae54121e257c05c18fc8a5968`
and source freeze SHA256
`3b3244ae1e759111b8240af929715074b751c946881202a9ff95e9f3b9918ffd`.
It checks plan-derived raw run identity/path, sibling external summary path and
hash, manifest/current method hashes, raw receipt hashes, exact original failure
and process, clean native exit, source/config before-final equality and frozen
runtime-file provenance. Original raw files are read, not normalized or rewritten.
The existing all-rank cadence/token/memory/input/cache gates remain called after
extraction. The initial schema findings prevented their successful integration;
the follow-up verifies successful execution through these gates.

This analyzer adds offline JSON parsing and bounded receipt hashing. It adds no
training arithmetic, tensors, communication, synchronization or per-update work.
No training source, configuration, cache or GPU workload was changed or executed.
The method's operational guard must still keep first live adjudication outside
all six repair timing windows. No runtime performance measurement was needed
for this offline-only change.

## Initial checks and limits

- `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s runs/schedule-embedding -p test_analyze_timings.py -v`: **20 tests passed**, 2.214 seconds.
- `ruff check --select F821,F811,F841 --no-cache runs/schedule-embedding/analyze_timings.py runs/schedule-embedding/test_analyze_timings.py`: **passed**.
- Small in-memory/temporary-fixture checks confirmed that the accepted extraction
  fixture's actual `errors=[]` is rejected by `validate_run_receipt`, and replacing
  stage hashes with the producer's mapping is rejected on all eight ranks.
  Changing proof metadata to the helper's PASS shape also hits the FAIL-only
  rejection. These checks did not run live adjudication or campaign analysis.

No GPU, installation, new allocation, actual adjudication, live matrix analysis,
plots, bulk scans or large-log/trace reads were performed. Existing
cadence/statistics/plots work was not re-audited.

## Focused follow-up — all three findings resolved

Verified only the fixes and affected fixture/gate behavior on 15 November 2026:

1. `analyze_campaign` now requires all seven final proof paths and a method pin;
   `_load_adjudication` enforces each `004/<variant>` namespace, plan run ID and
   raw path, including raw A003 with its new `004/A` proof. The shared proof
   authentication remains in place. Separate branches require exact FAIL/1/
   nullable-guard evidence for A/B/C/D0/S, or PASS/0 with no failure or nullable
   exception, valid completion/resolved-YAML hash and raw/external summary
   equality for SE/SE-fused. Raw process receipts still require native exit zero,
   no timeout/orphan and a stopped process group. Receipt hashing now requires
   the appropriate outcome-specific files. **Finding 1 resolved.**
2. Both proof validation and `validate_run_receipt` require canonical
   `errors == []`; a nonempty list is rejected. **Finding 2 resolved.**
3. Stage receipts now require a nonempty path-to-SHA256 map equal to the
   authenticated frozen schedule-source map on every rank. Extraction preserves
   mapping provenance for SE/fused comparison. Fixed module counts are zero for
   A/B/C and one for D0/S/SE/SE-fused, in both proof and raw stage validation.
   **Finding 3 resolved.**

`AdjudicatedCampaignTests` builds a complete seven-row temporary fixture from
retained A003 metadata, source/config/process receipts, all eight native rank
sidecars and the native parser summary. It retains the immutable real repair
plan and source-freeze hashes. It constructs timing helper proof fields with
the correct nullable/PASS distinction, explicit stage lists, source maps and
module counts. A003 remains the raw A input while every proof and external
summary uses a sibling pair in `004/<variant>`. This fixture calls
`analyze_campaign` and its full acceptance validators without mocking any
validation function; only filesystem constants point to temporary files.

The fixture deliberately synthesizes proof generation, variant metadata and
resolved YAML, and reuses A's measured series. It verifies the analysis schema
and gates, not that these synthetic variants satisfy the live adjudicator's
resolved-config checks or represent independent measurements. That distinction
does not leave a gap in the three reviewed fixes. It uses a synthetic method
file to test manifest/current/proof hash equality; it is not evidence that a
live adjudication used the final method.

Executed follow-up checks:

- `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s runs/schedule-embedding -p test_analyze_timings.py -v`: **23 tests passed**, 23.623 seconds.
- The suite includes successful full seven-row acceptance and rejection of a
  missing proof, wrong run path/variant, nonempty errors, changed frozen stage
  map, wrong module count, nullable exception on PASS, inconsistent original
  status, tampered raw hash and unequal raw/external PASS summary.
- `ruff check --select F821,F811,F841 --no-cache runs/schedule-embedding/analyze_timings.py runs/schedule-embedding/test_analyze_timings.py`: **passed**.
- `sha256sum runs/schedule-embedding/adjudicate.py` confirms final method
  `594fb7db1ff5403e6bcb9ac901fe98f85c35d10e09da36dcdf1c5322810780f4`.
  This verifies current bytes only; no live adjudication was executed.

No unresolved material finding remains in this focused review. The exact
remaining execution gate is to wait for every repair timing attempt to finish,
generate all seven final `004` proofs with the completed method above, pin that
hash in the final manifest, then run analysis against those actual receipts.
The follow-up performed CPU checks and small metadata reads only; it did not run
actual adjudication, live matrix analysis, plots, GPU work, bulk scans or cache
writes. The offline-only runtime-cost conclusion above is unchanged.
