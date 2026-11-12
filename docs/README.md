# Technical documents

- [Meeting optimization candidates](optimization-candidates.md): approaches, crop compatibility, preserved source code and tests, exact revisions and missing external dependencies.
- [Agreed 100B-dev crop baseline](baseline-100b-crop.md): current model, runtime, SAC, microbatch, token-drop contract, smoke limits, and finite future experiments.
- [Historical OPT-004 stability record](../backlog/tasks/opt-004-cuda-graphs-hybridep-fp8.md): bounded native stability evidence for the separate 3B HybridEP/FP8 workload; CUDA Graph and performance research remains paused.
- [Initial task boundary](task-boundary.md): supplied scope exchange, its original open choices, and the later agreed baseline selection.
- [Optimization contract](optimization-contract.md): behavior and state that experiments must preserve.
- [Research lessons](research-lessons.md): source-grounded hypotheses and limits on transferring prior findings.
- [GIFT 4 implemented optimization inventory](gift-4-ready-optimizations.md): existing optimization controls, automatic kernels, compatibility limits, and test candidates grounded in the pinned source snapshot.
- [GIFT 4 700B training flow](gift-4-700b-training-flow.md): step-by-step GFSDP, EP, FP8, loss, backward, and compile boundaries, with bounded research-backed trace candidates.

Update these pages when verified evidence changes a conclusion. Keep run-specific results in [runs/](../runs/README.md), and link to evidence instead of copying raw measurements into an overview.
