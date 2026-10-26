# OPT-002 — Profile the baseline and identify a bottleneck

**Status:** Backlog

**Depends on:** [OPT-001](opt-001-baseline.md)

## Goal

Locate the measured limit in the selected workload and describe its mechanism without treating a trace as a speed result.

## Acceptance

- Confirm the rank-zero resolved configuration, active feature markers, source pins, topology, process groups, and representative inputs.
- Establish baseline A/A noise and stable unprofiled step series.
- Use a representative trace to distinguish compute, host submission, synchronization, transfers, and exposed communication waits.
- Measure paired memory in defined lifecycle windows and account for allocator/device counter limits.
- Include data, validation, metric, checkpoint, and other recurring job phases at actual cadence.
- Name one bottleneck with evidence, confidence, limitations, and a next candidate question. If no bottleneck is established, record that result and stop candidate selection.

No profile or training run is authorized by this task record.
