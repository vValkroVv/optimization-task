# Final targeted preparation acceptance

**ACCEPT for fresh preflight, root release, and the exact two-job native
one-update readiness probe, subject to the explicit release gates below.**

This supersedes the R2/R4 HOLD in
[TARGETED-CORRECTION-REVIEW.md](TARGETED-CORRECTION-REVIEW.md) for the identities
in [accepted-sha256.json](accepted-sha256.json). Original reviews and failed
[CPU reproduction](repro_corrections.log) remain historical evidence.

## Scoped verification and dispositions

Inspected only the remaining ownership/topology corrections, their native
call sites, failure-path handling and targeted regressions. Reused accepted
configuration, full-native-checkpoint, software/import and producer evidence;
no broad audit or redundant test run was performed.

- **R2 resolved:** `scan` starts from the owner PID/start tick as well as recorded
  descendants. Direct children adopted by the subreaper receive real parent
  ancestry proof and retained pidfds, independent of mutable entrypoint argv.
  Both false and exceptional initial registration enter bounded release rather
  than waiting and raising outside release handling. Existing ledger verification
  and failure status prevent advancing with surviving recorded identities.
- The actual regression recreates the failed 80-ms parent-exit/two-second child
  window without entrypoint argv, requires `subreaper_adopted_descendant` proof,
  exit 70 rather than false success, and verifies the exact child PID/start tick
  is absent. The initial-registration-failure regression likewise verifies
  adoption, ledger proof and disappearance through the real release path.
- **R4 resolved:** preflight imports `parse_topology_matrix` and `is_exact_nv18`
  from the hashed task-owned helper. The validator indexes GPU-label keys and
  checks all eight rows, lengths, diagonals and NV18 edges. Its regression uses
  retained actual topology output, accepts the valid matrix and refuses a
  changed edge and missing link. The helper is included in the verifier's exact
  artifact set and release template.
- R1 remains resolved. R3 now accounts for the newly covered adopted descendants
  through the same ledger. R5 retains the reviewed 1,800-second operational
  deadline with bounded escalation and no automatic retry/extension.

Reused owner's reported **3/3 targeted regression passes**, prior three process
tests, shell/full Ruff/compile/template-artifact-set checks, Source009 config
resolution with zero warnings/no attempt checkpoint directories, software import
check, and 34 source CPU tests. Matched all six supplied final changed-file
SHA256 pins directly. The earlier failed CPU cases establish why these new
regressions were required; they were not discarded.

No GPU query/run, preparation/source edit, install, cleanup or subagent was
performed. This review writes only review evidence. Blocked NFS/kernel calls
can exceed software timeouts; uninterruptible surviving processes must remain
explicit release failures, never successful resource-release claims. Sampled
RSS is not an instantaneous peak measurement.

## Required launch and result gates

1. Root freezes Source009 manifest
   `cf95f726518062714913b8155dad878b73442fc5a0d68d993b7abdf6a55cf3f8`
   and these final preparation identities, then creates the source-preflight
   release. Preserve the exact accepted preset/adapter/configs throughout both
   phases.
2. Fresh real preflight proves the exact eight idle H100 UUIDs, no GPU apps,
   SM90/NV18 topology, actual software/task import identity approved by root,
   NFS admission of at least 732 GiB and fsync/readback. Root reviews its receipt
   and fills the final launch release with exact hashes and confirmations.
3. Execute only the serial native step-1 full S0 save and fresh-process full
   load-only absolute-step-2 observed update. Verify complete eight-rank
   readback/manifests/compiler records, ownership release, no observe checkpoint,
   storage accounting and actual sampled host/device memory evidence.

Fresh preflight and final root release are intentionally pending gates, not
missing user authorization. Successful execution establishes only this exact
native readiness path. Production runtime, calibrated A/A numerical comparison,
ten-update generation, graph-on behavior, prior CUDA fault clearance and campaign
acceptance remain open.
