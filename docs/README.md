# Technical documents

- [CUDA Graph research plan and goal](../backlog/tasks/opt-004-cuda-graphs-hybridep-fp8.md): planned feasibility, correctness, continuation and performance tests for the exact 3B MoE/HybridEP/FP8 preset on eight NVLink-connected H100 GPUs.
- [Initial task boundary](task-boundary.md): supplied scope exchange, current direction, and unresolved configuration details.
- [Optimization contract](optimization-contract.md): behavior and state that experiments must preserve.
- [Research lessons](research-lessons.md): source-grounded hypotheses and limits on transferring prior findings.
- [GIFT 4 implemented optimization inventory](gift-4-ready-optimizations.md): existing optimization controls, automatic kernels, compatibility limits, and test candidates grounded in the pinned source snapshot.
- [GIFT 4 700B training flow](gift-4-700b-training-flow.md): step-by-step GFSDP, EP, FP8, loss, backward, and compile boundaries, with bounded research-backed trace candidates.

Update these pages when verified evidence changes a conclusion. Keep run-specific results in [runs/](../runs/README.md), and link to evidence instead of copying raw measurements into an overview.
