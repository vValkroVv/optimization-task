# Targeted correction verification

**HOLD for fresh preflight and native execution.** Two concrete failures remain
on the corrected R2/R4 paths. This preserves the original [review](REVIEW.md)
and reuses its accepted config/native checkpoint/producer findings. Exact
correction identities are in [correction-reviewed-sha256.json](correction-reviewed-sha256.json).

## R2 — adopted descendants can survive a successful owner exit

`PhaseOwner.scan` seeds its ancestry queue from `self.records`, excluding the
owner process. A child that is spawned and orphaned between scans is adopted
by the subreaper but never traversed from the owner. If it no longer carries
the entrypoint argument, `_register_recovered_entrypoints` cannot discover it.
The agent is then reaped, the ledger contains no live identities, and the owner
publishes `natural_exit`, exit 0 and `all_recorded_identities_absent=true` while
that descendant remains alive. This leaves original R2 unresolved and also
omits this descendant from R3 RSS accounting.

The [bounded CPU reproduction](repro_corrections.py) executed the actual
`phase_process_owner.py`: a direct child sleeps 80 ms, spawns a two-second
separate-session Python child without the entrypoint argv, and exits. The owner
returned 0 with record_count 1 while the descendant remained live.
[Observed output](repro_corrections.log). The sleeper exited naturally; no
signals, process cleanup or GPU operations were used.

Seed scans from the owner PID/start tick so all adopted direct children have
genuine subreaper ancestry proof. Retain pidfds and existing per-identity
signals; avoid claiming argv/cwd matching itself proves ancestry. Include the
initial `_register` failure path in release handling: it currently waits and
raises outside the main release `try`, so a fast-exiting agent can also leave
adopted children without a release attempt. Verify this exact between-scan
adoption case and initial-registration failure alongside existing controller
TERM/deadline checks. Do not rerun the unrelated producer audit.

## R4 — every valid topology fixture raises KeyError

The matrix is built with string keys `GPU0` through `GPU7`, but the final
`topology_ready` generator indexes `matrix[row]` where row is an integer from
`range(8)`. A valid exact NV18 matrix reaches `matrix[0]` and raises KeyError.
The extracted actual expression reproduces `KeyError: 0` in the same CPU script
and log above. Index using the actual GPU labels, and verify a valid NV18
fixture passes while a changed edge is refused before running real preflight.

## Other correction dispositions

- R1 is resolved in source: the parent directory is created before its unique
  child. Reuse the reported successful Source009 two-config resolution with
  zero warnings and no attempt/checkpoint directory creation.
- R3 now uses the exact ownership ledger across sessions; that is correct for
  recorded descendants. Complete R2 adoption accounting before accepting rank
  RSS completeness. Sampling is correctly labelled sampled RSS, not a true peak.
- R4 software/import proof and root's exact `approved_runtime` equality gate
  address the earlier missing software review contract. The fresh topology
  check is correctly scoped apart from its indexing defect. Reuse the reported
  CPU software/import check.
- R5 is resolved as an explicit 1,800-second per-phase operational deadline,
  independent owner and bounded TERM/KILL grace. Sampler commands now have
  timeouts. Blocked NFS/kernel calls cannot be guaranteed interruptible, and the
  documented caveat must remain. A surviving/unverifiable process must remain
  a release failure and prohibit phase transition.
- Reuse the reported three process tests, full lint/shell/template/hash/doc
  checks and Source009 integration/34 CPU tests. The additional experiment was
  necessary because their passing outcomes did not cover adopted children
  missed between scans or the actual valid topology expression.

## Remaining release gates

Fix and target-verify these two findings, freeze new preparation hashes, and
obtain root's final review. Then run fresh real preflight with Source009 and
exact approved software/topology/UUIDs/no-apps/NFS admission before root creates
the launch release. Root release and real preflight are intentionally pending;
this review does not question existing authorization.

No successful readiness, production runtime, numerical A/A equivalence,
graph-on behavior or CUDA-fault-clearance result is claimed. No preparation or
training-source edits, GPU queries/launches, installs, cleanup or subagents were
performed in this verification.
