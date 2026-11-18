# Repair-004 timing adjudication review

## Verdict

**Bounded PASS after correction.** The single material finding below is resolved.
The timing-only repair path authenticates the retained explicit runtime files
after the seven-attempt terminal gate and before imports, parsing or output.
No material finding remains within this review's scope. Actual timing
adjudication remains separate work.

Reviewed `adjudicate.py` SHA-256:
`52b0b7999d5ad46c9a62af995814d84ea9244ca139726388f7b56422226505f6`.
Reviewed plan SHA-256:
`1a193606963e2f3c7086a6afb1d6d56ad78692dae54121e257c05c18fc8a5968`.

## Finite acceptance

Scope: authenticate the immutable repair plan; admit retained A-003 plus exactly
six named timing-004 attempts; defer expensive work until all seven are terminal;
authenticate the retained freeze without a live whole-source scan; preserve
native source/configuration boundaries, original wrapper failures and provenance.
The [previous accepted review](adjudication-review.md) supplies unchanged A and
nullable-field evidence. No diagnostic extension or broad correctness audit was
performed.

## Resolved finding history

The initial reviewed method was
`e81790f965f327a79e65279980efc33834a6406e11ebf34625a4539366f313eb`.
The finding and original line references below describe that revision.

### Authenticate offline helpers before importing or invoking them

[adjudicate.py](adjudicate.py), line 932, loads current runner/measure modules before
the three schedule-file checks at lines 934–938. Lines 944–948 then load and
execute the current `runs/overnight-streaming-perf/parse_results.py` without
checking its retained hash. The authenticated freeze contains `executed_helpers`
and Flash bridge hashes, but this path never compares those entries with the
current files. The old `campaign._campaign_freeze` did compare them
([campaign.py](campaign.py), lines 702–742).

Consequently a stale or edited parser, base runner, measure module or historical
recipe can change the new acceptance result while the output still carries the
accepted freeze hash. Native before/final receipts authenticate the attempted
training source and configuration; they do not authenticate code executed later
by the offline adjudicator. The adjudicator's own method hash also does not cover
these dependencies.

Required correction: after the terminal gate and before `_load_runtime`, verify
the explicit file hashes already in the authenticated freeze. Preserve the
existing three `schedule_source_sha256` checks; check all eight
`executed_helpers` entries (`production_data_contract`, `overnight_runner`,
`overnight_measure`, `overnight_parser`, `compiler_observer`,
`streaming_filelock`, `historical_campaign_plan`, `original_capacity_snapshot`)
and the three `flash_source_identity.bridge_sources` paths (`capture.py`,
`measure.py`, `run.py` under `runs/flash-qla`). This is a bounded list of files;
it requires no source-tree walk, cache traversal or dataset scan.

Required regression: make an explicit helper's current digest disagree with its
retained digest and assert rejection before runtime imports/parser invocation or
output creation. Include the parser and a runtime dependency. Keep the active
matrix gate ahead of these hash reads.

### Focused correction verification

The corrected `_authenticate_frozen_files` at lines 226–265 requires exact
manifest membership, canonical helper paths and matching digests for all eight
executed helpers, three schedule files and three Flash bridge files. It also
checks the campaign digest against the launch-plan pin. `adjudicate` now runs
the terminal gate at line 991 and this authentication at line 992, before raw
hashing, `_load_runtime` at line 995, parser loading at line 1001 and output
creation at line 1006. It records the authenticated file map in new provenance.

The actual retained canonical manifest was loaded and checked with mocked digest
reads: the returned map covered exactly its 14 paths and expected hashes, with
`Path.rglob` forbidden. No current helper file contents or source trees were
scanned by this verification.

The updated CPU suite passed **16 tests in 0.61 s** using the command below.
New behavior regressions inject parser, base-runner and Flash-runner digest
mismatches through the public orchestration path; each rejects before raw
hashing, imports, parser loading or output creation. A separate regression
requires the active terminal gate to reject before any frozen-file hashes or
runtime imports. The initial finding is resolved; no further audit was opened.

## Passing boundaries and executed checks

- Mandatory `--plan`, exact plan path/hash, ordered timing entries and argv
  identities constrain A to retained 003 and B/C/D0/S/SE/SE-fused to 004.
  Attempt and output paths are separately constrained.
- The retained freeze's canonical JSON hashes to
  `3b3244ae1e759111b8240af929715074b751c946881202a9ff95e9f3b9918ffd`;
  modified freeze content is rejected. Loading this metadata does not call
  `Path.rglob` or `_campaign_freeze`.
- `_check_source_config_identity` still compares actual per-attempt native start
  source identity with the pinned freeze, final source digest/HEAD with start,
  and before/final configuration hashes with each other and the freeze.
  Synthetic native-start and native-final digest mismatches both reject.
- The gate precedes raw hashing, runtime loading, parsing, input and cache checks.
  A monkeypatched active-gate exception stopped the public `adjudicate` path
  before raw hashing or runtime loading, with no output created.
- Exact known nullable failure plus native exit 0 and all existing completion,
  configuration, stage, capacity, cadence and timing checks remain required.
  The code reads raw FAIL receipts and writes new output only; it does not
  rewrite original failures. Provenance accurately says no current full-source
  walk was recomputed and infers no historical scan duration.
- Initial-review command: `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
  /opt/torchtitan-venv/bin/python3.12 -m pytest -q -p no:cacheprovider
  runs/schedule-embedding/test_adjudicate.py`: **11 passed in 0.08 s**.
- Additional in-memory CPU probes passed exact-seven identity, canonical-freeze
  tamper rejection, native boundary mismatch rejection and gate ordering. A
  separate probe reproduced the finding: a fake parser was invoked while its
  deliberately mismatching digest was never requested. Runtime imports, raw
  validation and output writes were stubbed out for that probe.

## Limits

No actual timing attempt was adjudicated. No GPU work, installations, source or
harness edits, cache scans, whole-GIFT traversal, large logs or traces were used.
This review adds no numerical, GPU/NCCL, convergence or performance evidence.
The finding's targeted acceptance gate is closed by the correction and executed
regressions; unchanged contracts reuse the prior review.
