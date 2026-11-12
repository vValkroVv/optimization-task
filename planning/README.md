# Work plan

Edit priorities, scope and next actions here. [Current state](../docs/status.md)
owns completed results; [candidates](../docs/candidates.md) owns mechanisms and
source links. Follow [workflow](workflow.md) and [validation](validation.md).
A planned item does not itself authorize GPU work or a change to the baseline.

## Authorized overnight campaign

The user has authorized the [streaming/performance campaign](overnight-streaming-perf.md)
for unattended execution, including task-source edits, existing eight-GPU runs,
private dependency installation and owned-artifact cleanup. Its progress ledger
owns execution status. Execution has begun from the
[English launch prompt](overnight-agent-prompt.md); G0 passed, and MDS/native
measurement integration is in progress. No training measurement exists yet.

The campaign supersedes the earlier unstarted single-candidate plan for its
listed reference mechanisms. It preserves the fixed model and creates a validated
MDS data baseline before measuring performance. Follow its concrete scope and
permissions rather than interpreting this generic table as a second authorization gate.

| Item | Status | Question / finite output | Next action |
| --- | --- | --- | --- |
| Default crop and basic smoke | Complete | Canonical config and all-rank training execution | Reuse [receipt](../runs/100b-crop8-ep8-smoke/result.json); rerun only after a relevant change or for a new required property |
| Representative performance baseline | Authorized in overnight plan; not started | Stable throughput/memory and A/A noise on agreed data and cadence | Execute G1–G2 of the overnight plan |
| Crop bottleneck trace | Authorized in overnight plan; not started | Identify actual critical path, overlap and active kernels | Capture a representative warmed step; pair with unprofiled controls |
| Reference optimization candidates | Covered by overnight plan | Correctness plus repeated integrated cost comparison | Complete the overnight reference coverage table, then evaluate applicable mechanisms sequentially |
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
