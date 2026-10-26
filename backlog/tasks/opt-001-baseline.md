# OPT-001 — Select and measure the representative baseline

**Status:** Backlog

**Depends on:** Source import record in [additional/baseline](../../additional/baseline/README.md)

## Goal

Select the production-representative training workload and record a reproducible baseline. The imported upstream source revision is a source identity, not a measured workload baseline.

## Open selections

Model and preset; dataset, tokenizer and split; sequence and packing rules; optimizer and schedule; global batch; precision, activation checkpointing and compilation; parallel degrees, process groups and physical topology; evaluation/checkpoint cadence; quality thresholds; and authorized hardware.

## Acceptance

- Record all selections and exact source/dependency pins.
- Capture the actual rank-zero resolved configuration and prove relevant features are active.
- Record data identity, split, seed, software/build versions, topology and commands without secrets or private host access details.
- Measure representative unprofiled baseline steps after startup, compilation, autotuning and warmup; include A/A noise and production lifecycle cadence.
- Define correctness comparisons, exact invariants, tolerances and quality gates before candidate work.
- Publish a concise linked campaign report with results, uncertainty, exclusions and open gates.

No benchmark or compute allocation is authorized by this task record.
