# 100B crop8 EP8 training smoke

Status: PASS on eight H100 GPUs, 13 November 2026 (Moscow date).

The default is [the agreed baseline](../../docs/baseline-100b-crop.md), implemented
in [the canonical experiment](../../gift/configs/experiments/gift_100b_crop8_ep8_fp8_compile_cute.yaml).
Training source lives in `optimization-task/gift`. This is a basic execution
check, not a throughput benchmark or convergence study.

## Runtime and reproduction

- Entrypoint: `torchtitan.train`.
- Python: `/opt/torchtitan-venv/bin/python3.12`.
- PyTorch: `2.15.0.dev20260822+cu130`; CUDA runtime 13.0; NVCC 13.2.51.
- DeepGEMM: `2.3.0+f093a4c`; DeepEP: `1.2.1+1b8f467`, from the installed environment.
- CUTLASS DSL: 4.4.1; TVM FFI: 0.1.11.
- Hardware: eight H100 80GB GPUs, NV18 between every pair.
- The launcher sources `/etc/profile.d/wm.sh` for CUDA compatibility libraries,
  verifies installed dependency paths/hashes and checks GPU availability.

[smoke.sh](smoke.sh) takes four arguments: task root, Python executable,
installed DeepGEMM package parent, and a fresh attempt ID. The installed package
parent is `/opt/torchtitan-venv/lib/python3.12/site-packages` on the verified server.
The workload retains microbatch 2, sequence 4096, global batch 512, EP8,
GFSDP2, FP8, compile, AdamW, CuTe and the agreed SAC policy. Smoke overrides are
ten updates, monitoring interval 5 and disabled validation; scheduler horizon
remains 65,000. Checkpoints, profiling and memory snapshots are disabled.
Data are bundled `c4_test` and the test tokenizer, not a production mixture.
Future launches require compute authorization and a fresh availability check.

## Observed result

Executed revision: `af02e60448ba7e2ca1b267c792d13c94205a65cc`.
Training source revision: `383d393360d6451021f5d52f11b4646bd1601b28`.
Launch: 2026-11-12 21:59:59 UTC; rank-0 completion: 22:12:10 UTC.

| Check | Result |
| --- | --- |
| Exit code | 0 |
| Every rank | 10 optimizer updates, 320 microbatches, 32 per update |
| Monitoring | Completed at steps 5 and 10 |
| Logged loss and gradient norms | Finite on all ranks |
| Final rank-0 loss / gradient norm | 14.57517 / 7.6661 |
| Maximum native console memory | 72.61 GiB |
| Checkpoints / profiler traces / memory snapshots | None |
| Teardown | All ranks completed; task processes exited; GPU compute-app list empty |

The memory metric combines the PyTorch reserved peak with report-time non-PyTorch
usage; it is not a simultaneous device-wide peak. Monitoring produced 34,084
scalar tags at steps 5 and 10. The ordinary/monitoring 2/32 loss chunks are
supported by resolved settings and the executed source/hook path, not a direct
chunk counter. This short run establishes basic training execution only.

The [result receipt](result.json) records configuration and dependency hashes,
step metrics and completion checks. Raw evidence is retained on the task server
under `runs/100b-crop8-ep8-smoke/artifacts/raw/20261113-003-stock/`: exit status,
metadata, resolved configuration, eight rank logs, structured logs and TensorBoard
events, plus the sibling launch log/receipt. Raw logs retain their original names.

## Source verification

### Source and CPU checks

The current-main runtime now has typed production initialization, adaptive
expert-bias updates, per-layer SAC and the monitoring loss-chunk override. The
canonical configuration retains the production cosine schedule and minimum LR
factor 0.0098. Torch-free checks verify all 43 CuTe gated norms and the exact 48
selected dense FP8 projections, including the expected exclusions.

The donor SAC policy exposed a real eager-mode defect: the mutating grouped FP8
operation returned `None`, so SAC cached no result and skipped the output
mutation during recomputation. Its functional interface now allocates and returns
the same BF16 output once; forward, dgrad and warmup consumers use that result.
No extra copy or allocation is introduced, and k-grouped wgrad is unchanged.

- Typed initialization/loss/trainer checks: 27 passed; subsequent scheduler,
  CuTe and FP8 selection checks: 3 passed.
- Real grouped-FP8 autograd path with CPU backend/quantizer doubles: 13 passed,
  including trainable and frozen weights, eager and compiled execution, exact
  loss/input/weight-gradient parity, and recomputation counts.
- Adaptive routing/config runtime checks: 25 passed.
- Dedicated SAC/baseline checks: 17 passed, one CUDA-only check skipped.
- Ruff checks on changed source and `git diff --check` passed.
- The prescribed experiment/model golden suite passed all 112 checks after
  regeneration, with update mode disabled. Structural review of 87 changed
  existing snapshots found only the expected default additions: SAC save sets,
  per-module modes, baseline load-balance strategy and optional monitoring loss
  chunks. The two crop snapshots are new.
- The exact smoke CLI overrides passed the torch-free checker for eight ranks
  with zero warnings. The reviewed launch script passed `bash -n`; ShellCheck
  is unavailable. The DeepGEMM import path and runtime attributes were
  checked on the target environment.
- A combined local CPU harness had 55 passes, one skip, one deselection and
  12 compiled-router failures inside PyTorch 2.14.1 guard construction after
  fake-Triton preloading. This is not reported as a passing combined suite;
  the isolated new compiled regressions passed. The target GPU smoke below is
  separate execution evidence, not a claim that the combined CPU suite passed.

Independent integration and runtime-cost review found no blocking source-level
issue. It inspected FP8 autograd/storage handling, SAC selection, GFSDP setup,
HybridEP dispatch, initialization, monitoring, loss, EMA and optimizer hooks.
Saving six up/gate outputs adds a balanced-shape estimate of 2.25 GiB per active
microbatch; layer 5 additionally saves communication outputs. This is not a
measured peak or speed claim.
