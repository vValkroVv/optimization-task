# 100B crop8 EP8 training smoke

Status: source integration reviewed; GPU training smoke pending.

## Scope

The user authorized the default baseline, the required source ports in the task's
`gift/` tree, commit/push, pulling that revision on the existing GPU server, and
one eight-GPU training check. Checkpoints, profiler traces and memory snapshots
are disabled. This is a startup/training check, not a performance comparison,
convergence study or checkpoint-continuation claim.

Canonical configuration:
[`gift_100b_crop8_ep8_fp8_compile_cute.yaml`](../../gift/configs/experiments/gift_100b_crop8_ep8_fp8_compile_cute.yaml).
The [baseline contract](../../docs/baseline-100b-crop.md) owns the default and the
separate future experiments.

## Source and runtime

- Task starting revision: `833aa18435f8d06ab143dc6cb837f91ccf8f97bf`.
- Existing GIFT snapshot: upstream main `6fd9a3230ab6361f11bbbf0b6ec5a5f125b78fc2`
  plus the task's already recorded runtime corrections.
- Port donor: `cute_norm_prodrun@ec47fb8e40a9a1e55d3c18dfa704ca104ae773af`, a
  descendant of `msenikolaev_act_manager_100b@f15d33f3fd9204e5fef2adbd625b3caed2241052`.
  The task retains the current typed configuration system and ports only the
  required SAC, adaptive-routing, initialization and monitoring-loss contracts.
- Read-only preflight found eight idle H100 80GB GPUs, 81,559 MiB reported per
  device, and NV18 connectivity between every GPU pair. Ownership is checked
  again immediately before launch.
- Existing runtime: Python at `/opt/torchtitan-venv/bin/python3.12`, PyTorch
  `2.15.0.dev20260822+cu130`, CUDA 13.0, CUTLASS DSL 4.4.1, TVM FFI 0.1.11,
  DeepEP `1.2.1+1b8f467`, DeepGEMM `2.3.0+f093a4c`.
- Reuse the existing isolated DeepGEMM descriptor repair and the existing pinned
  HybridEP scan correction; do not modify shared installed packages. The repaired
  DeepGEMM header SHA-256 is
  `ed8bfb6a182d6a8a225d36b359287f17dba73e95d5cc50150ca177ea979557d2`.

## Bounded check

Run ten optimizer updates with the canonical microbatch 2, sequence length 4096,
global batch 512 and 32 accumulation microbatches. Keep the scheduler horizon at
65,000 updates. The only monitoring-cadence validation override is interval 5
instead of the default 50, exercising ordinary two-chunk loss and monitored
32-chunk loss in the short run. Use existing offline `c4_test` data and the test
tokenizer; retain the model vocabulary size 128,256. These data are smoke inputs,
not the production pretraining mixture.

Acceptance is finite: all eight ranks start with the intended configuration;
the required FP8, CuTe, compile, GFSDP2/EP8 and per-layer SAC paths are active;
ten optimizer updates complete with finite logged loss/gradient norm; ordinary
and monitoring steps execute; no checkpoint, trace or memory snapshot is emitted;
the owned run exits and releases its GPU processes. An OOM or runtime failure is
recorded rather than silently changing widths, expert count, microbatch or SAC.

## Results

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
  is unavailable. The repaired DeepGEMM import path and runtime attributes were
  checked on the target environment.
- A combined local CPU harness had 55 passes, one skip, one deselection and
  12 compiled-router failures inside PyTorch 2.14.1 guard construction after
  fake-Triton preloading. This is not reported as a passing combined suite;
  the isolated new compiled regressions passed. The real target runtime remains
  a separate verification gate.

Independent integration and runtime-cost review found no blocking source-level
issue. It inspected FP8 autograd/storage handling, SAC selection, GFSDP setup,
HybridEP dispatch, initialization, monitoring, loss, EMA and optimizer hooks.
Saving six up/gate outputs adds a balanced-shape estimate of 2.25 GiB per active
microbatch; layer 5 additionally saves communication outputs. This is not a
measured peak or speed claim.

The published source revision, exact launch command, native logs, activation
evidence and final GPU result will be recorded after execution.
