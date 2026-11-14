# FlashQLA trace capture results

- Status: **PASS**.
- Plan: `/workspace-SR008.nfs2/vkropotin/optimization-task/runs/flash-qla/trace-plan.json` (SHA256 `2ec42682418b5572f97a6bb1bde7fcffacc7c3b095821e4aa00af4ae85a7d792`).
- Acquired exactly `external_gate` then `fused_gate`; no new FLA trace was captured.
- Final check: eight GPUs idle with no compute apps; all eight frozen source hashes match the plan.

| Variant | Run | Summary | Rank 0 raw / gzip | Rank 1 raw / gzip |
|---|---|---|---|---|
| `external_gate` | `fqla-trace-external-001` | `/workspace-SR008.nfs2/vkropotin/opt-fq/runs/flash-qla/artifacts/raw/fqla-trace-external-001/summary.json` | `25854443 B / 2020523 B` | `25857043 B / 2018633 B` |
| `fused_gate` | `fqla-trace-fused-001` | `/workspace-SR008.nfs2/vkropotin/opt-fq/runs/flash-qla/artifacts/raw/fqla-trace-fused-001/summary.json` | `25707431 B / 2007148 B` | `25718815 B / 2007493 B` |

## external_gate — PASS

- Controller: `/workspace-SR008.nfs2/vkropotin/optimization-task/runs/flash-qla/trace-capture-controllers/fqla-trace-external-001.json`; run log: `/workspace-SR008.nfs2/vkropotin/optimization-task/runs/flash-qla/trace-capture-controllers/fqla-trace-external-001.log`.
- Attempt status: `/workspace-SR008.nfs2/vkropotin/opt-fq/runs/flash-qla/artifacts/raw/fqla-trace-external-001/status.txt`; summary: `/workspace-SR008.nfs2/vkropotin/opt-fq/runs/flash-qla/artifacts/raw/fqla-trace-external-001/summary.json`; process receipt: `/workspace-SR008.nfs2/vkropotin/opt-fq/runs/flash-qla/artifacts/raw/fqla-trace-external-001/process.json`; completion receipt: `/workspace-SR008.nfs2/vkropotin/opt-fq/runs/flash-qla/artifacts/raw/fqla-trace-external-001/completion.json`.
- Child exit code: `0`; process group stopped: `True`; orphan: `False`; timeout: `False`.
- Training: 21 updates, all eight ranks complete; profile mode `profile_update`, measurement `disabled`.
- Rank 0/1 capture: 3 microbatches each, one stop/drain/token reduction, one train and optimizer step per rank; ranks 2–7 capture zero and finish the update.
- Resolved rank-0 config: `/workspace-SR008.nfs2/vkropotin/opt-fq/runs/flash-qla/artifacts/raw/fqla-trace-external-001/native/config/resolved_config.yaml` (SHA256 `6f53812d039aa431932fd6a31016ab21a44811dd32e3a00ef61c98251e39cac4`); source and config unchanged.
- Actual parent/child argv are stored in `/workspace-SR008.nfs2/vkropotin/optimization-task/runs/flash-qla/trace-capture-controllers/fqla-trace-external-001.json` and this JSON report; generated child argv matched the plan.
- Actual torchrun entry markers: per-rank false/true counts are stored in `/workspace-SR008.nfs2/vkropotin/optimization-task/runs/flash-qla/trace-capture-controllers/fqla-trace-external-001.json`. These show entry invocations, not literal backward counters.
- Raw trace / gzip identity, raw event census, hashes, bytes, marker receipts, and checked config fields are recorded in this JSON report.
- Rank 0: raw `/workspace-SR008.nfs2/vkropotin/opt-fq/runs/flash-qla/artifacts/raw/fqla-trace-external-001/native/profiling/traces/iteration_21/rank0_trace.json` (25854443 B, SHA256 `74fb51354e5eccbee97405a464cd8563062d5eff9f279e2019faf8803cb055be`); gzip `/workspace-SR008.nfs2/vkropotin/opt-fq/runs/flash-qla/artifacts/raw/fqla-trace-external-001/native/profiling/traces/iteration_21/rank0_trace.json.gz` (2020523 B, SHA256 `cde57c4f059d27a9c15b74502afdf6286cd597006643e7955051210439a43cff`; decompressed identity verified).
- Rank 1: raw `/workspace-SR008.nfs2/vkropotin/opt-fq/runs/flash-qla/artifacts/raw/fqla-trace-external-001/native/profiling/traces/iteration_21/rank1_trace.json` (25857043 B, SHA256 `1028095290d8cd988efbbaefcd313c3c1d8e3175002b576365b55691202b7173`); gzip `/workspace-SR008.nfs2/vkropotin/opt-fq/runs/flash-qla/artifacts/raw/fqla-trace-external-001/native/profiling/traces/iteration_21/rank1_trace.json.gz` (2018633 B, SHA256 `caa12f90153a7cabbbcdfe40f5d36f174145d70bd6165b5526018b5ef13602ba`; decompressed identity verified).

## fused_gate — PASS

- Controller: `/workspace-SR008.nfs2/vkropotin/optimization-task/runs/flash-qla/trace-capture-controllers/fqla-trace-fused-001.json`; run log: `/workspace-SR008.nfs2/vkropotin/optimization-task/runs/flash-qla/trace-capture-controllers/fqla-trace-fused-001.run.log`.
- Attempt status: `/workspace-SR008.nfs2/vkropotin/opt-fq/runs/flash-qla/artifacts/raw/fqla-trace-fused-001/status.txt`; summary: `/workspace-SR008.nfs2/vkropotin/opt-fq/runs/flash-qla/artifacts/raw/fqla-trace-fused-001/summary.json`; process receipt: `/workspace-SR008.nfs2/vkropotin/opt-fq/runs/flash-qla/artifacts/raw/fqla-trace-fused-001/process.json`; completion receipt: `/workspace-SR008.nfs2/vkropotin/opt-fq/runs/flash-qla/artifacts/raw/fqla-trace-fused-001/completion.json`.
- Child exit code: `0`; process group stopped: `True`; orphan: `False`; timeout: `False`.
- Training: 21 updates, all eight ranks complete; profile mode `profile_update`, measurement `disabled`.
- Rank 0/1 capture: 3 microbatches each, one stop/drain/token reduction, one train and optimizer step per rank; ranks 2–7 capture zero and finish the update.
- Resolved rank-0 config: `/workspace-SR008.nfs2/vkropotin/opt-fq/runs/flash-qla/artifacts/raw/fqla-trace-fused-001/native/config/resolved_config.yaml` (SHA256 `b5b76c6e6e3055595742e5903b838dc99e60bee79f8fb32b52cb59a4a05907ce`); source and config unchanged.
- Actual parent/child argv are stored in `/workspace-SR008.nfs2/vkropotin/optimization-task/runs/flash-qla/trace-capture-controllers/fqla-trace-fused-001.json` and this JSON report; generated child argv matched the plan.
- Actual torchrun entry markers: per-rank false/true counts are stored in `/workspace-SR008.nfs2/vkropotin/optimization-task/runs/flash-qla/trace-capture-controllers/fqla-trace-fused-001.json`. These show entry invocations, not literal backward counters.
- Raw trace / gzip identity, raw event census, hashes, bytes, marker receipts, and checked config fields are recorded in this JSON report.
- Rank 0: raw `/workspace-SR008.nfs2/vkropotin/opt-fq/runs/flash-qla/artifacts/raw/fqla-trace-fused-001/native/profiling/traces/iteration_21/rank0_trace.json` (25707431 B, SHA256 `98299717bcc2c4c8f1e096abc3ddefe7963eb814c893e06de0a9613c1cc69be0`); gzip `/workspace-SR008.nfs2/vkropotin/opt-fq/runs/flash-qla/artifacts/raw/fqla-trace-fused-001/native/profiling/traces/iteration_21/rank0_trace.json.gz` (2007148 B, SHA256 `de0f05e9f853a73f00e62239cb55663d507f3a7c7cba0a832eafc0ae836d1463`; decompressed identity verified).
- Rank 1: raw `/workspace-SR008.nfs2/vkropotin/opt-fq/runs/flash-qla/artifacts/raw/fqla-trace-fused-001/native/profiling/traces/iteration_21/rank1_trace.json` (25718815 B, SHA256 `4f2a7b4aafac305d19473efd7c7a506f5c0fb08cb1a62112723eac057f1e62b9`); gzip `/workspace-SR008.nfs2/vkropotin/opt-fq/runs/flash-qla/artifacts/raw/fqla-trace-fused-001/native/profiling/traces/iteration_21/rank1_trace.json.gz` (2007493 B, SHA256 `d7b97ac9e96e2a370c05b4ff694061fa195146a762fdccbb686c3374613caf78`; decompressed identity verified).

## Harness failure history

1. External writer exited 1 after writing both gzip siblings because it searched the structured JSONL path for backend-entry strings. The actual markers are in the torchrun `attempt_0` stdout/stderr logs. The existing gzip files were preserved and verified by decompressed SHA256 and byte identity.
2. A fused inline preflight wrapper exited 1 with only the SSH key-exchange warning visible; the failing gate is unknown. It left no fused controller, attempt, process, or GPU allocation. The saved launcher then passed all seven named gates and launched successfully.
3. `chmod` and `sha256sum` were accidentally placed after a local heredoc terminator, so zsh ran them on macOS against a remote-only path and returned 1. The remote launcher file had been created (mode 0644); the root invoked it with the absolute Python interpreter, and the run completed.

## Analysis handoff

Raw pair handoff to `/root/flashqla_new_trace_comparison` was sent at `2026-11-14T18:55:46Z`. Strict coverage and actual MCP import pass with zero import errors; the analysis report is `/workspace-SR008.nfs2/vkropotin/optimization-task/runs/flash-qla/trace-results.json` and `/workspace-SR008.nfs2/vkropotin/optimization-task/runs/flash-qla/trace-results.md`. The three-microbatch capture still does not establish the full 32-microbatch critical path.

Machine-readable record: `/workspace-SR008.nfs2/vkropotin/optimization-task/runs/flash-qla/trace-capture-results.json`.
