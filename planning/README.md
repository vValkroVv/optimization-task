# Work plan

Edit priorities, scope and next actions here. [Current state](../docs/status.md)
owns completed results; [candidates](../docs/candidates.md) owns mechanisms and
source links. Follow [workflow](workflow.md) and [validation](validation.md).
A planned item does not itself authorize GPU work or a change to the baseline.

| Item | Status | Question / finite output | Next action |
| --- | --- | --- | --- |
| Default crop and basic smoke | Complete | Canonical config and all-rank training execution | Reuse [receipt](../runs/100b-crop8-ep8-smoke/result.json); rerun only after a relevant change or for a new required property |
| Representative performance baseline | Planned; not run | Stable throughput/memory and A/A noise on agreed data and cadence | Agree data, warmup, window, repetitions, resource budget and quality gates |
| Crop bottleneck trace | Planned; not run | Identify actual critical path, overlap and active kernels | Capture a representative warmed step; pair with unprofiled controls |
| One optimization candidate | Awaiting bottleneck | Correctness plus repeated integrated cost comparison | Select one mechanism from [candidates](../docs/candidates.md) |
| Source gaps | Open | Exact vision list; pinned external CudaCherry source; production MR/trace identities | Obtain source evidence before claiming complete implementation coverage |
| Token-drop pressure | Future diagnostic | Routing/dropped-token distribution at production capacity 50,000; compare 12,500 only as a separately agreed diagnostic | Keep default unchanged |
| Shape proxies | Future diagnostics | E32/EP8/B2 gives four experts/rank and balanced M16384; E128/EP8/B8 increases token work fourfold | Separate model/shape changes from computation-preserving speed comparisons; review memory first |
| CUDA Graphs | Low priority / paused | Candidate benefit on current crop, not merely capture viability | Reopen only under [documented criteria](../docs/cuda-graphs.md) |

## Experiment entry template

For an approved item, create one directory in `runs/<experiment>/` with one
report and a machine-readable result. Record:

- Question, hypothesis, expected cost and the decision each outcome changes.
- Baseline/source/config hashes and one deliberate candidate difference.
- Data, seed, hardware/topology, software and authorization/resource limits.
- Required correctness checks and numerical tolerances before measurement.
- Commands, warmup/window, A/A and alternating A/B order, memory definitions.
- Stop conditions; handling of failed, skipped or incomplete runs.
- Results, evidence locations/hashes, exclusions, verdict and remaining gates.

Use statuses Planned, Running, Complete or Blocked. Complete means the stated
acceptance is evidenced, not that the script exited or a component test passed.
Keep run chronology in its report rather than creating separate handoff/review
files for every command. Update `docs/status.md` when the project conclusion changes.
