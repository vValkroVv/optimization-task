# GPU fault counters and event visibility

- Checked: 2026-11-26 14:08 UTC snapshot.
- Route: authorized SSH alias `sr008-gpu1`; remote hostname `lm-mpi-job-7c9acdc0-339d-480d-b7ea-9d3a43448e3a-mpimaster-0`.
- Hardware/software observed: 8 NVIDIA H100 80GB HBM3 GPUs; NVIDIA-SMI/driver 570.133.20, NVML 570.133, CUDA 12.8.
- Scope: read-only status and retained-log access checks; no GPU compute or operator changes.

## NVML / NVIDIA-SMI snapshot

For all 8 attached GPUs, ECC was enabled (current and pending). Volatile and aggregate counts were zero for SRAM correctable errors, SRAM uncorrectable parity and SEC-DED errors, DRAM correctable errors, and DRAM uncorrectable errors. Aggregate SRAM threshold exceeded was `No`.

For all 8 GPUs, row-remapper correctable and uncorrectable error counts were zero, `Pending` was `No`, and `Remapping Failure Occurred` was `No`. The bank remap histogram reported 2,560 banks at `Max`, with zero in High, Partial, Low, and None. Retired-page fields (`Single Bit ECC`, `Double Bit ECC`, and `Pending Page Blacklist`) were reported as `N/A` for all 8 GPUs.

The `gpu_recovery_action` query returned `None` for all 8 GPUs at the check time. This is a current recovery-action value, not a historical reset record.

Commands used: `nvidia-smi -L`; `nvidia-smi -q -d ECC,PAGE_RETIREMENT`; `nvidia-smi -q -d ROW_REMAPPER`; `nvidia-smi --query-gpu=gpu_recovery_action --format=csv`.

## Kernel event history access

- `dmesg --ctime` was denied: `read kernel buffer failed: Operation not permitted`.
- `journalctl` is not installed or available on this host (`FileNotFoundError`).
- Therefore, host kernel Xid, GPU-lost, reset, and recovery history could not be read. No conclusion that the retained history is event-free is supported.

The NVIDIA-SMI `-q` output contained no reset/recovery field; the explicit query above did expose current GPU recovery action. No event-history source was accessible in this check.
