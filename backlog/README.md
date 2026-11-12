# Backlog

Task records define current scope and acceptance. This scaffold uses Markdown only; no tracker configuration or external synchronization is implied.

| ID | Task | Status | Dependency |
| --- | --- | --- | --- |
| OPT-001 | Select and measure the representative baseline | Default and GPU smoke verified; performance measurement not started | Agreed baseline contract; source import record |
| OPT-002 | Profile the baseline and identify a bottleneck | Backlog | OPT-001 |
| OPT-003 | Evaluate one bottleneck-matched candidate | Backlog | OPT-002 |
| [OPT-004](tasks/opt-004-cuda-graphs-hybridep-fp8.md) | Stabilize the historical 3B HybridEP / FP8 baseline | Complete: bounded stability only | Refreshed GIFT main; prior campaign evidence |

A task is Done only when its acceptance criteria are verified. OPT-001's current
authorized source/config integration and basic smoke are complete; the
[baseline contract](../docs/baseline-100b-crop.md) links their evidence. This does
not complete the later performance measurement criteria. OPT-002 and OPT-003 remain
unstarted. OPT-004's recorded stability result remains distinct historical
evidence; its graph performance comparison is paused and unmeasured.
