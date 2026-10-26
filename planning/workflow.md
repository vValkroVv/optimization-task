# Task workflow

Tasks are plain Markdown in [backlog/tasks/](../backlog/README.md). No external tracker or Backlog CLI is configured or required.

1. **Select the baseline.** Complete OPT-001 before claiming that a configuration represents production. The imported source revision alone is not a workload baseline.
2. **Profile the baseline.** Complete OPT-002 with resolved configuration, active-feature proof, topology, A/A noise, representative unprofiled steps, traces for mechanism, and paired memory.
3. **Choose a candidate.** OPT-003 starts only after a measured bottleneck and a pinned reference implementation are reviewed for fit and runtime cost.
4. **Implement only within authorization.** Source edits, dependency changes, compute, and publication each follow their explicit scope.
5. **Review once against acceptance.** Record material findings, fixes, executed checks, and remaining gates in the task and linked campaign report.

Keep task status, dependencies, acceptance, and next action in the task record. Use `Backlog`, `Ready`, `In Progress`, `In Review`, `Blocked`, and `Done` only when the state is evidenced. Do not mark a task Done until every criterion is supported by evidence. A timeout or CPU-only result does not close a GPU gate.
