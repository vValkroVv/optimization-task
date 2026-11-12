# Matched recapture diagnostic pair — preparation

Status: **prepared, not released, not launched** (2026-11-12). This is a
short diagnostic-cost pair for the late A007 window. It is not a viable
candidate measurement, a stable-throughput result, or an A/A/noise estimate.

## Decision and scope

The coordinator reports that all eight ranks show real dense/loss graph
captures and launches during updates 21–23, with no settled dense/loss reuse.
The independent all-rank trace review is complete. The pair asks whether the
recurring graph-policy cost and memory justify further A007 work or the finite
loss-only fallback. It does not test candidate viability. The exact interpretation
and stop rules are in the final clarification of
[MATCHED-PERFORMANCE-DESIGN](../../MATCHED-PERFORMANCE-DESIGN/001/handoff.md).

Run serially and only after root review, a fresh eight-GPU preflight, and an
explicit root release for the exact attempt. No `preflight.json` or
`release.json` was created here. The launcher refuses to proceed without both.
Do not change graph exclusions or substitute the loss-only fallback under this
pair. If root chooses the fallback, stop this pair without launching it.

| Attempt | Variant | Graph setting | Updates | Measured window |
| --- | --- | ---: | ---: | --- |
| `attempts/001` | C0-short | `TORCHINDUCTOR_CUDAGRAPHS=0` | 26 | 21–25; closes at batch-start 26 |
| `attempts/002` | C1-short | `TORCHINDUCTOR_CUDAGRAPHS=1` | 26 | 21–25; closes at batch-start 26 |

The only intended variant difference is the graph environment value. Both use
`TORCHINDUCTOR_MIX_ORDER_REDUCTION=0`, `graph_partition=True`, the same A007
bootstrap, observer overlay, cache policy, model/config, seed and native CLI.
The observer is identical in both (`start_step: 21`, `update_count: 5`), with
the public step-boundary callback. Blocks `layers.1`–`layers.17` are excluded
from graph capture only; all 18 blocks and loss remain compiled. SAC18,
FlexAttention, the full FP8 recipe, HybridEP/EP8, four microbatches, local batch
8/global batch 256, sequence 4096, seed 42, scheduler horizon 30,000 and
warmup 2,000 remain in force. Checkpointing and evaluation retain preset
defaults; the short-run CLI disables checkpoint writes as reviewed. No
profiling flags, `TORCH_LOGS`, history recording, or other debug instrumentation
are used. Native metric cadence stays unchanged.

The exact common CLI is:

```text
--training.steps 26
--lr_scheduler.total_steps 30000
--debug.seed 42
--checkpoint.enable false
--dump_folder <attempt>/native
```

Each process uses fresh `<attempt>/scan-runtime`, offline HF settings and the
same existing shared caches. Do not clear or reset shared caches. Each run has
a 30-minute hard deadline, `--max-restarts=0`, and no automatic retry. Stop the
pair at the first fault, incomplete/invalid observer window, source/config
change, missing activation evidence, OOM, nonfinite update, or failed GPU
release. Preserve the attempt and first error; C1 requires a root-reviewed valid
C0 window and successful all-eight-GPU release.

## Pins and reusable evidence

- Source snapshot manifest: SHA-256
  `3a606c81d6b13692e422da04c333252620e504df9f1c4ba59c4bcf0c1a51f603`.
- Source archive: SHA-256
  `324cad36a75404f235f5dd7de10b571a75ad486276a44c09fe425f3187ebd51b`.
- Preset: SHA-256
  `b341701bd1eef7774c110fbf7ed69cf6280ba8691100b05be457dee3e7d50a7b`.
- Shared A007 `compiler_entry.py` copy: SHA-256
  `ffde3588d53012c7155fb5b6961a06be96095b44b8bc55c97a9808b2446ba065`.
- Both observer overlays: SHA-256
  `368c5d0219c491381e0cdfc9b6ad30b86c97c3b201e0bfa286ecc2f68cfe7c5c`.
- Existing CPU-only prerequisite comparison proves the same A007 bootstrap
  resolves `triton.cudagraphs` to 0/1 with `triton.mix_order_reduction=False`
  and `graph_partition=True`; it reports no CUDA initialization or training:
  [comparison](../../LANE-A/attempts/007/compiler-prerequisite-comparison-v2.json).
  No probe was repeated.
- Reused launch interface and precedent:
  [A007 launcher](../../LANE-A/attempts/007/launch.sh),
  [A007 bootstrap](../../LANE-A/attempts/007/compiler_entry.py),
  [A007 overlay](../../LANE-A/attempts/007/config/observer.yaml), and the
  existing bounded paired-run process-group pattern in
  [BASELINE-AA launch method](../../BASELINE-AA/attempts/001/launch-method.txt).

The launchers recheck the complete 92-file source manifest, archive, preset,
bootstrap and overlay immediately before starting training. The pinned source
is unchanged by this preparation. The bootstrap records per-rank compiler
options under each attempt and asserts the requested graph setting before
importing the unchanged native `train_scan_fix` entry.

## Release and exact commands

The attempt directories deliberately contain neither `preflight.json` nor
`release.json`. Root must create these only after reviewing this preparation
and performing a fresh preflight. The launcher checks:

- `preflight.json`: `status=READY_TO_LAUNCH`; the exact attempt; UTC
  `checked_at_utc` no more than five minutes old; source/preset/bootstrap/config
  hashes; `all_file_hashes_match=true`; the eight A007 H100 UUIDs in order;
  `identity_matches_previous=true`, `all_h100_idle=true`,
  `compute_apps_empty=true`; and all prior task PIDs absent.
- `release.json`: `status=COORDINATOR_CONFIRMED_FOR_LAUNCH`, exact attempt
  scope and all pinned hashes, `pair_decision=DIAGNOSTIC_COST_ONLY`, root
  review and independent all-rank trace review complete, all-rank dense/loss
  captures/launches confirmed, and `settled_dense_loss_reuse_observed=false`.
- C1 additionally requires the release to record a root-reviewed valid C0
  window (five records on all ranks, 1,048,576 global useful tokens per record,
  exit code 0) and the C0 all-eight-idle/app-empty postrun check. The launcher
  also checks C0's actual `exit-status.json` and `gpu-release-check.json`.

Launch one attempt at a time using the existing isolated-session supervisor
pattern: `subprocess.Popen(..., start_new_session=True, stdin=DEVNULL,
stdout=<attempt>/supervisor.log, stderr=STDOUT, close_fds=True)`. Its PID is the
attempt process-group/session ID. Set a 30-minute deadline; at timeout send
TERM to only that attempt's process group, wait 30 seconds, then KILL only that
group if it remains. Review exact recorded PIDs before any further cleanup.
The launcher records its PID and torchrun PID, kills/waits only its own memory
sampler, uses torchrun with zero restarts, and records a postrun check. Do not
use broad process-kill commands.

The exact child commands are:

```bash
env MATCHED_RECAPTURE_SUPERVISED=1 \
  EXPECTED_SOURCE_MANIFEST_SHA256=3a606c81d6b13692e422da04c333252620e504df9f1c4ba59c4bcf0c1a51f603 \
  bash /workspace-SR008.nfs2/vkropotin/optimization-task/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/MATCHED-RECAPTURE/attempts/001/launch.sh

env MATCHED_RECAPTURE_SUPERVISED=1 \
  EXPECTED_SOURCE_MANIFEST_SHA256=3a606c81d6b13692e422da04c333252620e504df9f1c4ba59c4bcf0c1a51f603 \
  bash /workspace-SR008.nfs2/vkropotin/optimization-task/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/MATCHED-RECAPTURE/attempts/002/launch.sh
```

The supervisor must open each `supervisor.log` exclusively and refuse if it
already exists. Do not start C1 as a chain or automatically after C0.

## Expected artifacts and interpretation

Each run retains `supervisor.log`, `launch-invocation.json`, `runner.pid`,
`torchrun.pid`, `lifecycle.log`, `exit-status.json`, `device-memory.csv`
(500-ms samples), `nvidia-smi.stderr`, `torchrun/` per-rank logs,
`native/` resolved/source configs and native outputs,
`native/timing_observer/rank-00000.jsonl` through `rank-00007.jsonl`,
`compiler-options/rank-00000.json` through `rank-00007.json`, fresh
`scan-runtime/`, and `gpu-release-check.json`. Root-owned `preflight.json` and
`release.json` remain input approvals. No source, preset, cache or shared
resource is modified by preparation.

For a valid window, each rank must have five complete global-token records of
1,048,576 each; these counts are already global and must not be summed across
ranks or multiplied by accumulation. Report `5,242,880 / slowest-rank window
seconds` as the observed global useful-tokens/s ratio, completed mean update
latency, labeled observer submission/event series, boundary allocator values,
and 500-ms sampled device memory. Keep captures/recaptures and their memory in
the window. The sampled device maximum is a lower bound, and boundary
allocator values are not window peaks. Do not call this stable throughput,
claim speedup, infer A/A noise, isolate capture cost, or advance a candidate
from this pair. A result inside this finite window only chooses whether to
investigate the fallback or continue the separately approved finalist plan.

Expected measurement overhead is the existing observer's two boundary drains
and aligned reductions, its boundary bookkeeping, the 500-ms external device
sampler, and unchanged native metric logging. There are no profiler traces,
per-update added barriers or new source runtime changes. Integrated elapsed
time includes work in the observer window; attribution scopes not produced by
these logs remain unavailable.

## Checks and current gate

Prepared only. No `nvidia-smi`, GPU query, CUDA initialization, torchrun,
training, source edit, package install, or shared-cache mutation was run here.
The graph0/graph1 CPU prerequisite evidence above was reused. Shell syntax,
bootstrap syntax, file hashes, and documentation links/whitespace are checked
before handoff.

**Pending:** root file review, fresh GPU/PID preflight, and explicit C0 release.
Root's all-rank trace review is complete; its finding is no settled dense/loss
reuse. If root instead selects the loss-only fallback, this pair remains
unlaunched. C1 remains conditionally gated on a valid C0 result and separate
root release.
