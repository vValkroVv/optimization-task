# Backlog

Task records define current scope and acceptance. This scaffold uses Markdown only; no tracker configuration or external synchronization is implied.

| ID | Task | Status | Dependency |
| --- | --- | --- | --- |
| OPT-001 | Select and measure the representative baseline | Backlog | Source import record |
| OPT-002 | Profile the baseline and identify a bottleneck | Backlog | OPT-001 |
| OPT-003 | Evaluate one bottleneck-matched candidate | Backlog | OPT-002 |
| [OPT-004](tasks/opt-004-cuda-graphs-hybridep-fp8.md) | Test CUDA Graphs for 3B MoE / HybridEP / FP8 on eight H100 GPUs | Backlog | Refreshed GIFT main; execution access |

A task is Done only when its acceptance criteria are verified. OPT-004 selects a concrete workload and hardware and includes its own baseline, profiling and candidate stages; it need not wait for the earlier generic tasks. Reuse its evidence in OPT-001 through OPT-003 where applicable, without marking their broader criteria complete automatically. All GPU campaign results remain unmeasured.
