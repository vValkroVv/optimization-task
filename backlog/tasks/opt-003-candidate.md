# OPT-003 — Evaluate one bottleneck-matched candidate

**Status:** Backlog

**Depends on:** [OPT-002](opt-002-profile.md)

## Goal

Evaluate one candidate selected from a measured bottleneck and a pinned implementation reference. Do not assume a preferred optimization in advance.

## Acceptance

- State the bottleneck evidence, candidate hypothesis, pinned reference revision, source fit, expected runtime/memory costs, and required authorization.
- Preserve the [optimization contract](../../docs/optimization-contract.md); name any proposed scope change before implementation.
- Compare baseline and candidate with matched inputs/configuration, A/A noise, alternating unprofiled runs, stable-step series, production lifecycle costs, and paired memory.
- Complete applicable numerical, state, checkpoint-continuation, and quality checks. Distinguish unavailable GPU/NCCL/quality gates from passing results.
- Report results and uncertainty without extrapolating beyond the tested configuration. Record regressions and unresolved gates.

This task does not authorize source edits, dependency installation, or compute.
