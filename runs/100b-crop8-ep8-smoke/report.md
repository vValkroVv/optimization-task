# 100B crop8 EP8 training smoke

Status: PASS for the bounded eight-H100 training smoke on 13 November 2026
(Moscow date). No performance or convergence claim.

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
  the isolated new compiled regressions passed. The target GPU smoke below is
  separate execution evidence, not a claim that the combined CPU suite passed.

Independent integration and runtime-cost review found no blocking source-level
issue. It inspected FP8 autograd/storage handling, SAC selection, GFSDP setup,
HybridEP dispatch, initialization, monitoring, loss, EMA and optimizer hooks.
Saving six up/gate outputs adds a balanced-shape estimate of 2.25 GiB per active
microbatch; layer 5 additionally saves communication outputs. This is not a
measured peak or speed claim.

### Attempt 20261113-001

Published and pulled revision: `383d393360d6451021f5d52f11b4646bd1601b28`.
Started at 2026-11-12 21:21:09 UTC (13 November in Moscow). The preflight exited
with status 1 before torchrun: PyTorch reported an old driver API version 12080
and no available CUDA devices, while `nvidia-smi` saw all eight H100s.

The noninteractive SSH environment had no `LD_LIBRARY_PATH`. The existing image
profile `/etc/profile.d/wm.sh` supplies the installed CUDA compatibility libraries
under `/usr/local/cuda-13.2/compat/lib`; the launcher now sources that profile
before its first PyTorch import. This uses NVIDIA's established
[forward-compatibility route](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html),
without installing packages or changing the host driver.

No training rank started; no checkpoint, trace or snapshot was created. The
launcher exited, the exact task-process scan was empty, and all GPU compute-app
lists were empty. Original preflight evidence remains under
`artifacts/raw/20261113-001/` and its sibling launch log.

### Attempt 20261113-002 — PASS

Executed revision: `31f2b0a759835ffbd234e8c78876225d62ec3b24`; the training-source
changes are in `383d393360d6451021f5d52f11b4646bd1601b28`, followed only by the
launcher environment correction. Launch was 2026-11-12 21:25:07 UTC; rank 0
reported training completion at 21:37:53 UTC (13 November in Moscow).

| Check | Observed result |
| --- | --- |
| Exit | `0` |
| Every rank, 0–7 | Ten completed optimizer updates and ten completed steps |
| Microbatches | Exactly 32 completed forward/backward spans per update, 320 per rank |
| Numerical observations | All logged loss and gradient norms finite |
| Monitoring | Hooks emitted at steps 5 and 10; ordinary and eager-monitoring paths completed |
| FP8 | Native activation messages report 48 dense and 14 grouped projections |
| Compile / sharding / SAC | Native model/loss compile, fully-shard/EP8 and the heterogeneous layer policies logged |
| CuTe | Resolved CuTe norms and generated Inductor code containing both CuTe forward and backward operators |
| Memory | Maximum native console memory metric across ranks/steps: 72.55 GiB; rank-0 maximum 72.32 GiB |
| Artifacts | Native configs, structured JSONL logs and TensorBoard events only; no checkpoints, profiler traces or memory snapshots |
| Release | All ranks completed distributed teardown; owned launcher/torchrun/rank processes absent and GPU compute-app query empty |

The native memory metric is the PyTorch reserved peak since the last reset plus
non-PyTorch usage sampled at report time. It is not a continuously measured
device-wide maximum, and must not be added to the separately printed non-Torch
field again. This smoke used a cold compilation path and more frequent monitoring;
its console throughput is not a performance baseline.

Monitoring loss chunks are established by the resolved `2`/`32` settings, hook
events at steps 5/10, and the executed native source path: `Trainer.train_step`
selects the monitoring value through `MonitoringCallback.is_monitoring_step`,
`MTPLoss` inherits `ChunkedCELoss`, and the override is restored in `finally`.
There is no direct runtime chunk-count counter. This is source/output correlation,
not a claimed counter measurement.

Rank-0 displayed loss values for updates 1–10:
`15.31904, 15.29531, 15.24754, 15.17967, 15.09524, 15.00440, 14.90702, 14.80481, 14.69808, 14.57506`.
Displayed gradient norms:
`9.5409, 9.4775, 9.3872, 8.9090, 8.1253, 7.4433, 7.1236, 7.2134, 7.3938, 7.6756`.
These rounded logs establish finite training observations, not numerical parity
or convergence.

The parent independently read exit status, metadata, final step/completion/
teardown lines from all eight rank logs, native output inventory and empty GPU
compute-app state after the observer's complete JSONL audit.

Evidence remains on the task server under `artifacts/raw/20261113-002/`:
`exit-code.txt`, `metadata.txt`, `native/config/resolved_config.yaml`, eight
`native/structured_logs/*.jsonl`, `logs/torchrun/**/stdout.log`, and `native/tb/`.
The compact [result receipt](result.json) is versioned; large runtime artifacts
and caches remain outside Git.

Reproduction uses the reviewed [launcher](smoke.sh), with explicit task root,
existing Python executable, existing isolated DeepGEMM repair parent and a fresh
attempt ID. It preserves the canonical workload and applies only the recorded
ten-update stop and monitoring-cadence validation overrides. Future runs require
their own compute authorization and fresh ownership preflight.
