# Overnight streaming and performance campaign

Status: RUNNING. The [accepted campaign](../../planning/overnight-streaming-perf.md)
owns execution progress and the finite G0–G7 acceptance checklist. This report
and `results.json` own measured results; raw payloads remain ignored.

## Entry and environment

Execution started 13 November 2026 02:18 Europe/Moscow (12 November 23:18 UTC)
at clean task `main`, revision `a2a59e14545b55ca777f94a28e332759e7c06087`.
The existing single master exposes eight idle H100 80 GB GPUs. The task path
is on SR008 NFS with approximately 2.5 TiB free; no new allocation was made.
Runtime preflight verified `/opt/torchtitan-venv/bin/python3.12`, torch
`2.15.0.dev20260822+cu130`, CUDA 13.0, NCCL 2.30.7, streaming 0.13.0,
transformers 4.57.1, CUTLASS DSL 4.4.1 and TVM FFI 0.1.11 after sourcing
`/etc/profile.d/wm.sh`. Shared packages remain unchanged.

Perfetto MCP successfully loaded a synthetic Chrome JSON at the actual task
absolute path and executed `SELECT name, ts, dur FROM slice`; it returned
`campaign_route_check`, 1000000 ns, 100000 ns. This verifies tool/path access,
not the later required training-trace analysis. No private server is needed.

## Scope and interpretation

Preserve the fixed crop, precision, SAC, MTP1, EMA, optimizer, seed and scheduler.
MDS is a new data baseline because HF and MDS packing differ. Performance will
include input and monitoring at interval 50, excluding validation and checkpoint
saves. Short runs remain in optimizer LR warmup. No convergence, continuation,
multi-node or production-scale performance conclusion is implied.

## Results

No training measurements yet. Candidate applicability is under source inspection.

## Cleanup

No campaign training processes, checkpoints or heavy diagnostic artifacts yet.
