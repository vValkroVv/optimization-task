# OPT-001 — Select and measure the representative baseline

**Status:** In progress for the agreed crop configuration and basic smoke; performance baseline work is not authorized in the current scope.

**Depends on:** Source import record in [additional/baseline](../../additional/baseline/README.md)

## Goal

Select the representative training workload and record its identity. The imported upstream source revision is a source identity, not a measured workload baseline. The selected default is the [agreed 100B-dev crop](../../docs/baseline-100b-crop.md). Its required source/config integration, resolved configuration, and authorized basic GPU smoke remain pending.

## Open selections

The 100B-dev crop and its model, precision, parallelism, compile, SAC, token-drop, and microbatch settings are selected in the baseline contract. Confirm effective settings from the final resolved configuration. Record remaining data identity, software/build, topology, quality, and performance details before any later performance or quality study; do not infer them from this selection.

## Acceptance

- Record all selections and exact source/dependency pins.
- Capture the actual rank-zero resolved configuration and prove relevant features are active.
- Record data identity, split, seed, software/build versions, topology and commands without secrets or private host access details.
- For the current authorized scope, run only a basic training smoke without checkpoints or traces. A later performance baseline requires separate authorization, unprofiled stable-step series, A/A noise, and production lifecycle cadence.
- Define correctness comparisons, exact invariants, tolerances and quality gates before candidate work.
- Publish a concise linked campaign report with results, uncertainty, exclusions and open gates.

The owner authorized only the basic smoke. No benchmark or broader campaign is authorized by this task record.
