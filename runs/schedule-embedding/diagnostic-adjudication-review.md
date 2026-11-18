# Diagnostic adjudication independent review

Reviewed 15 November 2026. **PASS: the sole scheduling finding is resolved.**
The follow-up review verifies only that fix; earlier unaffected evidence is reused.

## Scope and finite acceptance

Review the new diagnostic mode against [diagnostic acceptance](diagnostic-acceptance.md):
mode isolation, frozen recipe and seven final timing prerequisites, native completion,
the narrow nullable exception, real native parser schema, unchanged raw receipts,
and explicit pending asynchronous trace-tail coverage. Reuse the earlier timing
review; do not repeat the unchanged timing or retained-A audit. Apply `code-rules`
and `perf-experiments`. No training-source changes, GPU activity, installation,
actual adjudication, source-tree scan, cache writes, or large raw artifact reads.

Final reviewed adjudicator method SHA256:
`594fb7db1ff5403e6bcb9ac901fe98f85c35d10e09da36dcdf1c5322810780f4`.
Initial reviewed method:
`fecf5d27e081ce561f7426982fe9052387ef9af7c7fe34dea7e175859c9b87c4`.
All seven final timing proofs must use the current final method hash; the initial
hash is retained here as review history only.

## Material finding

**Resolved P2 — Require all four diagnostics to be terminal before offline work.**
The initial method called `_require_diagnostic_attempt_terminal(run_id)` only
for the selected diagnostic. A completed D0 could therefore proceed while S, SE,
or SE-fused is running or has not started. The subsequent timing-proof admission
rehashes raw timing receipts and summaries; imports, input/cache validation,
original-trace hashing and native trace parsing follow. This permits host and NFS
work during a later diagnostic and conflicts with the coordinator's explicit
final policy: run all four serially, then create offline proofs after all four
end. That policy protects profile host/gap comparisons. The original branch
brief's requested-attempt terminal check is implemented; the stricter final
campaign policy required this additional boundary.

Resolution: `_require_all_diagnostic_attempts_terminal(plan)` validates the exact
four diagnostic names/run IDs and checks every attempt's terminal status and
process receipt. Missing process receipts, unstopped groups and explicitly
orphaned groups fail. `adjudicate()` retains the all-seven timing-terminal gate,
then invokes this all-four gate before timing-proof admission/rehashing, runtime
authentication/imports, input/cache checks and trace parsing. The new parameterized
regression covers another diagnostic RUNNING or absent while D0 is terminal;
costly callbacks remain uncalled. Targeted checks below close this finding.

## Contracts found consistent

- CLI defaults to `--mode timing`; diagnostic accepts only D0/S/SE/SE-fused.
  Plan and freeze pins are respectively
  `1a193606963e2f3c7086a6afb1d6d56ad78692dae54121e257c05c18fc8a5968`
  and `3b3244ae1e759111b8240af929715074b751c946881202a9ff95e9f3b9918ffd`.
  Exact plan entries select 21/20/0 and update 21/three microbatches.
- Seven timing proofs come exclusively from `adjudication-140-004/<variant>`,
  including fresh adjudication of retained raw A003 at `004/A`. Admission requires
  the current helper hash for every proof, matching freeze/plan/runtime manifest,
  original raw receipt hashes and offline summary hashes. There is no historical
  helper allowlist. A-recovery003 supplies the authenticated freeze only.
- All-seven timing-terminal checks precede the small 14-file runtime authentication;
  authenticated imports follow. No new full source walk appears on this path.
- Native exit, source/config finals, cleanup, actual child cadence and overrides,
  resolved config, real-data/cache identity, capacity/bootstrap, stage receipts,
  and eight-rank cadence retain their checks in diagnostic mode.
- FAIL recovery is limited to D0/S, wrapper exit 1, the exact nullable guard error,
  no completion or raw summary, explicit null, typed nullable None and eight
  initialized row-owner modules with empty stages. SE and SE-fused require raw
  PASS and respectively three and four stages. The actual stage producer uses a
  `source_sha256` path-to-digest map and module counts 0 for A/B/C, 1 otherwise;
  the consumer matches this schema.
- The native parser validates the all-eight-rank 32 F+B/one optimizer profile
  counters, collection/drain/export receipts and original trace structure. Its
  successful `errors` value is `[]`, matching the diagnostic validator. A real
  parser synthetic 21/20/0 result passes the new consumer; missing drain fails.
- Raw receipts are read and hashed; derived summary/receipt/index writes use
  separate output paths. No synthetic raw completion or raw PASS is created.
  Parser acceptance leaves flow/correlation and third-microbatch asynchronous
  device-tail coverage explicitly pending.

## CLI and output contract

After all seven timing proofs and all four diagnostic runs are complete:

```bash
python runs/schedule-embedding/adjudicate.py --mode diagnostic \
  --variant D0 \
  --run-dir runs/schedule-embedding/artifacts/raw/schedule-embedding-d0-diagnostic-140-004 \
  --output-dir runs/schedule-embedding/artifacts/raw/adjudication-diagnostic-004/D0 \
  --plan runs/schedule-embedding/artifacts/raw/newlaunchrecords/repair-004-plan.json
```

Use the corresponding variant and lowercase variant in each run ID for S, SE,
and SE-fused. Timing output remains `artifacts/raw/adjudication-140-004/<variant>`.
Diagnostic output is `artifacts/raw/adjudication-diagnostic-004/<variant>` with
`summary.json` and `adjudication.json`; its separate root index is
`compact-adjudication-index.json`.

Successful diagnostic parser status is `DIAGNOSTIC_PARSER_PASS_TAIL_PENDING`
(CLI exit 0), with `mode: diagnostic`,
`native_capture_status: NATIVE_PROFILE_PARSER_PASS`, and
`native_capture_proof.trace_tail_coverage.status: PENDING_PERFETTO` plus
`overall_acceptance_complete: false`. The compact index records
`trace_tail_coverage_status`. This does not establish complete diagnostic
acceptance. Rejection remains `ADJUDICATION_REJECTED`/CLI exit 1.

## Executed verification and limits

Follow-up verification against the final method:

- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider runs/schedule-embedding/test_adjudicate.py -k 'all_diagnostic_attempts or diagnostic_prerequisite_gate or terminal_gate_precedes'`:
  **4 passed, 22 deselected**, 0.11 s. This covers missing/active other diagnostic,
  gate ordering, and preservation of the timing gate. The full reported 26-test
  suite was not repeated in this bounded follow-up.
- The same scoped Ruff command below: **passed** on the final method.
- A tiny temporary CPU fixture exercised the actual all-four gate: all terminal
  receipts passed; the last diagnostic's unstopped group, orphaned group, and
  missing process receipt each failed. Temporary fixtures were removed.

Initial review evidence (unaffected paths were not rerun):

- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider runs/schedule-embedding/test_adjudicate.py`:
  **25 passed**, 0.31 s, no skipped modules.
- `python -m ruff check --select F821,F811,F841 --no-cache runs/schedule-embedding/adjudicate.py runs/schedule-embedding/test_adjudicate.py`:
  **passed**.
- CPU-only temporary synthetic fixture through actual
  `overnight-streaming-perf/parse_results.py:summarize_run`, then
  `_validate_diagnostic_summary` and `_check_original_trace_hashes`: **passed**
  for eight ranks, updates 1..21, two original-format tiny JSON traces and all
  required profile counters. Removing rank-0 `early_stop_cuda_drains` produces
  native `incomplete` and diagnostic rejection. Temporary fixtures were removed.

Runtime-cost review: the branch adds offline CPU/JSON/hash/trace I/O, not work to
training forward/backward, communication, precision, optimizer or checkpoint
paths. Hashing and parser work scale with raw receipt/trace bytes and must stay
outside all live diagnostics; the resolved gate enforces that cost boundary.
No actual 004 timing proof, GPU capture, full diagnostic integration, Perfetto
tail reconciliation, timing magnitude, or numerical claim was tested here.
The source review and targeted fix verification are complete. Real offline
proofs and Perfetto acceptance remain gates for the completed campaign.
