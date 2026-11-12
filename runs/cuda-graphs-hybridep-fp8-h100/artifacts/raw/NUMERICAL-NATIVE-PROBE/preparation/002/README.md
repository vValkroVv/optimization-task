# Native numerical readiness probe

This preparation defines two serial eight-rank native invocations: create a
full native S0 at absolute step 1, then restore it in a fresh process and
observe absolute step 2. Root owns source release, fresh GPU preflight, runtime
safety review, and execution. This document does not authorize a launch.

## Preserved workload and configs

Both overlays import the supported
[`cuda_graph_campaign_continuation.yaml`](../../../../../../../gift/configs/experiments/cuda_graph_campaign_continuation.yaml)
and preserve the supplied 3B MoE preset: HybridEP/EP8, FP8, FlexAttention,
selective activation checkpointing (SAC18), FSDP2, local batch 8, global batch
256, sequence length 4096, four derived accumulation steps, and compilation of
both model and loss. The deferred RNG continuation callback remains first.
They set seed 42, fused native AdamW, and the 30,000-step schedule with 2,000
warmup steps. `TORCHINDUCTOR_CUDAGRAPHS=0`; the canonical dense adapter retains
`mix_order_reduction=0` and graph partitioning, and records the options per
rank.

[`config/create_s0.yaml`](config/create_s0.yaml) runs one update to absolute
step 1 and synchronously saves the full native checkpoint at
`<dump_folder>/checkpoint/step-1` (`interval=1`, `keep_latest_k=0`, no seed
checkpoint, no model-only/HF save or load, and no excluded state). Its numerical
payload is relative to the dump folder at `checkpoint/numerical_probe`.

[`config/observe.yaml`](config/observe.yaml) runs one update to absolute step 2
with full native restore, `load_step=-1`, `load_only=true`, and no excluded
state. The command supplies the absolute S0 path. Before resolution and after
the fresh process, the launcher requires the observe destination `checkpoint/`
not to exist. Its payload is relative to the dump folder at `numerical_probe`;
load-only mode must leave no checkpoint behind.

The two training commands used by the launcher are:

```bash
/opt/torchtitan-venv/bin/python3.12 -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=8 --max-restarts=0 \
  --log-dir=<attempt>/create_s0/torchrun --redirects=3 --tee=3 \
  <attempt>/create_s0/compiler_entry.py \
  <preparation>/config/create_s0.yaml \
  --dump_folder <attempt>/create_s0

/opt/torchtitan-venv/bin/python3.12 -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=8 --max-restarts=0 \
  --log-dir=<attempt>/observe/torchrun --redirects=3 --tee=3 \
  <attempt>/observe/compiler_entry.py \
  <preparation>/config/observe.yaml \
  --dump_folder <attempt>/observe \
  --checkpoint.initial_load_path <attempt>/create_s0/checkpoint/step-1
```

## Process ownership and readiness sampling

Each phase has a 30-minute hard deadline measured from owner startup, with no
retry or automatic extension. The planning assumption reserves about five
minutes for startup and leaves 25 minutes for roughly 100 GiB of output, about
68 MiB/s aggregate, plus native checkpoint/readback traffic. That is an
operational stop budget, not a measured NFS guarantee. TERM and KILL each have
an eight-second escalation window; blocked kernel or storage calls remain an
operational limit for root's safety review.

[`phase_process_owner.py`](phase_process_owner.py) is specific to these two
phases. It becomes a Linux child subreaper, tracks each process by PID/start
tick and ancestry across separate sessions, retains pidfds for exact signaling,
and durably records identities. Lost-parent recovery accepts only a newly born
process with the exact absolute phase `compiler_entry.py` NUL-delimited
argument and expected task cwd. After positive ownership proof, signal-time
checks use the recorded PID/start tick and pidfd; they do not require argv to
remain unchanged. The owner sends bounded TERM/KILL only to recorded
identities and refuses a successful phase transition unless each recorded
identity is absent. An unresolved survivor fails the chain for root's separate
release gate.

The controller samples owned-process RSS (including ranks in separate
sessions), filesystem use/free space, and GPU memory. These are readiness
samples, not timing or throughput data; directory size is sampled every 30
seconds and other values every 10 seconds. Each external sampler query is
bounded. A sampler failure cannot clear the owner ledger or permit phase 2.
After S0, the launcher rechecks the exact eight UUIDs and requires no compute
apps before starting observe.

Projected native S0 plus one observed update is **104,523,644,928 bytes
(97.3452 GiB nominal)**. The 700-GiB payload cap plus 32-GiB free reserve
requires **732 GiB available before S0**. Preflight checks the NFS mount, free
bytes/inodes, and a retained 4 KiB write/fsync/readback probe. It creates no
attempt or checkpoint directory.

## Source pin, preflight, and root release

`root-release.template.json` is intentionally unconfirmed. Root first pins the
next full source snapshot, exact 3B preset/continuation, and preparation
artifact hashes, then sets `ROOT_PINNED_FOR_PREFLIGHT`. Preflight verifies that
pin before GPU queries. It records the authorized UUID order, H100 and SM90
(compute capability 9.0), all-pairs NV18 topology, idle/no-compute-app status,
task cwd/import origins and hashes, Python/Torch/Triton build and Python
distribution-inventory hash, source identity, and the 732-GiB storage
admission. It does not dump environment variables or secret files.

After reviewing the fresh receipt, root copies its exact runtime object into
`approved_runtime`, sets `ROOT_CONFIRMED_FOR_LAUNCH`, fills the receipt
path/hash, and confirms source/integration review, runtime-safety review, and
fresh no-active-GPU status. `launch.sh` revalidates every source and preparation
hash, the receipt (at most 15 minutes old), and absence of the attempt output.
It creates `config-checks/` before a unique child and resolves both configs
before making the attempt root; the pre-submit path must not create either
checkpoint directory.

Root runs fresh preflight after source release:

```bash
bash runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NUMERICAL-NATIVE-PROBE/preparation/002/preflight.sh
```

Root launches only after reviewing the exact release hashes and fresh preflight:

```bash
PREP_DIR=/workspace-SR008.nfs2/vkropotin/optimization-task/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NUMERICAL-NATIVE-PROBE/preparation/002
RUN_LOG="$PREP_DIR/launch-$(date -u +%Y%m%dT%H%M%S%N).log"
setsid nohup bash "$PREP_DIR/launch.sh" > "$RUN_LOG" 2>&1 < /dev/null &
printf 'controller_pid=%s\nlog=%s\n' "$!" "$RUN_LOG"
```

## Preparation evidence and open gates

- Source Snapshot 009 manifest SHA-256 is
  `cf95f726518062714913b8155dad878b73442fc5a0d68d993b7abdf6a55cf3f8`; archive
  SHA-256 is
  `c58638a869893758254ff486300d427455d8733bf6a1ef117efad9fe0275e443`.
- Both configs resolved against integrated Source 009 with
  `/opt/torchtitan-venv/bin/python3.12 -m torchtitan.config.pre_submit`,
  `--world-size 8`, and the observe absolute S0 path. Both reported
  `OK (0 warning(s), model=gift_moe/3B)`. Reports and dump paths stayed under
  `config-checks/`; the attempts root and checkpoint paths remained absent.
- The CPU-only software identity helper resolved Python 3.12.3, Torch
  `2.15.0.dev20260822+cu130`, and campaign/config/numerical module origins under
  the actual `gift/` cwd. It did not query CUDA devices. This is not the fresh
  launch preflight.
- Focused CPU process-tree tests passed 3/3: separate-session child recovery
  after parent exit and exec, deadline cleanup without extension, and
  controller TERM/early exit. The RSS case verified a separate-session worker
  in the ledger and included it in owned-process RSS.
- Canonical adapter copies match the reused loss-only adapter SHA-256
  `ffde3588d53012c7155fb5b6961a06be96095b44b8bc55c97a9808b2446ba065`.
- No GPU query, training launch, installation, source-tree write, or checkpoint
  creation was performed for this preparation.

Still open: root's final source pin and artifact hashes; fresh actual GPU,
topology, runtime and storage preflight; root runtime-safety and no-active-GPU
release; and the native S0/step-2 execution. A successful readiness probe would
establish no throughput, numerical equivalence, graph-on behavior, or campaign
acceptance.


## Attempt history and attempt-002 gate

Preparation 001's first launcher invocation stopped before training because the
`attempts/` parent was absent; its original evidence remains unchanged at
[`../001/root-launch-001-failure.json`](../001/root-launch-001-failure.json). The
corrected 001 runtime then completed the native update and full step-1 S0 save.
Source Snapshot 009 inventory rejected `complex64` on all eight ranks afterward,
so observe did not run. The source codec correction and next source snapshot are
root-owned; the inventory rejection is not attributed to this launcher.

Preparation 002 targets the fresh `attempts/002` child. Its launcher creates only
the task-owned `attempts/` parent after both config resolutions and their
no-checkpoint checks, then claims the unique child. No attempt-001 runtime output,
checkpoint, config-check result, or preflight result was copied here. Existing
process ownership, deadline, and readiness-sampling behavior is carried forward.

The template deliberately has no source pin or runtime approval. Root must fill
the next source manifest and exact artifact hashes, run a fresh 002 preflight,
review runtime safety, confirm no active GPU work, and provide release before
any launch. The prior Source 009 zero-warning config-resolution and software
identity evidence is historical only; config resolution against the next pinned
source and all fresh runtime gates remain open.
