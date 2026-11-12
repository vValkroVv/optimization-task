# Native readiness attempt 001: failed after native checkpoint save

**Readiness is incomplete.** Eight ranks completed the first native update and
synchronous full checkpoint save. The subsequent S0 diagnostic capture failed on
all eight ranks; observe never started. No numerical comparison passed, and this
attempt does not clear the earlier CUDA fault.

## Acceptance checklist and provenance

- [x] Establish native update, accumulation, save and failure evidence per rank.
- [x] Separate requested configuration, activation markers and missing evidence.
- [x] Inventory native checkpoint and small metadata without reading tensor payloads.
- [x] Resolve readiness, retention and existing-observe eligibility; preserve failures.

Analysis follows `perf-experiments`; it changes only evidence and adds no training
runtime cost. Reproduce from the repository root with
`python runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NUMERICAL-NATIVE-PROBE/analysis/001/extract.py`.
[Extraction script](extract.py) writes [extraction.json](extraction.json), including
per-rank event timestamps/counts, compiler receipts, checkpoint file sizes, state
keys, sampled resources and small-input SHA256 pins. It opens the trusted 2.05-MiB
DCP metadata pickle, never `.distcp` payloads, and makes no CUDA query.

Pinned source is [Snapshot 009](../../../SOURCE-SNAPSHOT/009/manifest.json), base
`446431dfa191ef0300f116257ae918bc85e38c06`; manifest SHA256
`cf95f726518062714913b8155dad878b73442fc5a0d68d993b7abdf6a55cf3f8`, recorded archive
SHA256 `c58638a869893758254ff486300d427455d8733bf6a1ef117efad9fe0275e443`.
Source claims below refer to that archive, not the subsequently editable tree.
The accepted [producer review](../../../NUMERICAL-PROBE-REVIEW/001/TARGETED-ACCEPTANCE.md)
and [launcher review](../../review/001/TARGETED-ACCEPTANCE.md) remain prerequisites,
not GPU correctness evidence. Reused 3+3 CPU process tests and 34 source tests are
precedent only; this analysis does not rerun them.

## What executed

The [lifecycle](../../attempts/001/lifecycle.log) starts at 08:58:02 UTC and ends
08:59:53 UTC on 12 November 2026 with exit 1. The earlier 08:56:44 invocation
failed before training because the attempts parent was missing; its
[failure record](../../preparation/001/root-launch-001-failure.json) remains intact.
Invocation 2 used the same accepted scripts/configuration after creating only
that parent, with the retained fresh 08:55 preflight.

Every rank's native structured JSONL contains exactly:

| Absolute step 1 marker | Count per rank |
| --- | ---: |
| `fetching_batch_end` | 4 |
| `fwd_bwd_end` | 4 |
| `optim_end` | 1 |
| `checkpoint_save_end` | 1 |
| `step_error` | 1 |

Each rank records local valid tokens 131,072 and global valid tokens 1,048,576.
These establish four native microbatches and an optimizer call completing before
save. They do not establish observed preclip/clipped gradients or matching updates.
`step_end` also exists on all ranks but follows an error; it is not success proof.
The numerical observe hooks were never active in an observe process.

## Intended configuration and observed activation

The actual [resolved config](../../attempts/001/create_s0/config/resolved_config.yaml)
matches 3B MoE, 18 layers, FlexAttention, HybridEP/EP8, FSDP2 shard 8, local batch 8,
global batch 256, sequence 4096, seed 42 and `deterministic: false`. It retains
float32 training state/reduction, bf16 parameter mixing, selective AC with RNG
preservation, model/loss compilation, fused AdamW, 30,000-step schedule and
2,000 warmup steps. This diagnostic stops at step 1; the schedule horizon is not 1.

Rank-0 stdout records `Compiling the loss function`, 18 compiled blocks,
`Applied selective activation checkpointing`, `Applied fully_shard`, FP8 structure
with 72 dense and 34 grouped projections, native accumulation 4 and HybridEP
buffer initialization (`ep_size=8`, four local experts). All eight compiler
receipts report Torch `2.15.0.dev20260822+cu130`, `triton.cudagraphs=false`,
`triton.mix_order_reduction=false` and `graph_partition=true`; this is graph-off.
The native mesh log reports pp1/replicate1/shard8/cp1/tp1/ep8 on the selected
eight-GPU server. It does not independently dump every process-group membership.

FlexAttention is preserved in both resolved block definitions and its build
path; no fallback substitution is evidenced. Runtime logging prepares `c4_test`
from `tests/assets/c4_test`, with tokenizer `tests/assets/tokenizer`, zero loader
workers and infinite loading. This is the test asset data identity, not production
quality data. No dataset-content digest, prepared-input records, mask equality,
packing-boundary comparison or restored loader-position comparison was produced.

## Failure and checkpoint meaning

The [probe manifest](../../attempts/001/create_s0/checkpoint/numerical_probe/manifest.json)
is `complete=false`, phase `s0`, with eight failed ranks and one unique error:
`ValueError: unsupported tensor layout/dtype: torch.strided, torch.complex64`.
The repeated outer RuntimeErrors aggregate this same failure; they are not 64
independent failures. Classification: post-save diagnostic inventory/serialization
rejection. This finding does not establish CUDA kernel health or fault clearance.
The codec correction belongs to its separate source owner and is not reviewed here.

Native checkpoint path:
`NUMERICAL-NATIVE-PROBE/attempts/001/create_s0/checkpoint/step-1` under this raw root.
It contains eight `.distcp` shards plus `.metadata`: **39,211,598,683 bytes
(36.518647 GiB), nine files**. Metadata SHA256 is
`d37d0fd6f0d04ec43a70daafdb86f76a8d70ca39eb921cabc539552d674f1784`.
All 8,756 metadata storage entries point within the eight existing shard sizes.
No shard payload was hashed or read, and no partial DCP reader was introduced.

Metadata lists 3,842 keys: model state, native optimizer state, trainer step/token
count, scheduler, eight loader ranks plus world size, and eight owned
`continuation_rng_rank_N` objects. Each RNG owner has schema/rank/world/device,
Python/NumPy, CPU and CUDA state fields. Each of 234 optimizer parameter keys has
`exp_avg`, `exp_avg_sq` and step entries; the two moment shapes match their model
keys. This is metadata coverage, **not P/m/v value equality, restore success,
RNG replay or proof of every buffer's persistence**.

Snapshot009 `after_checkpoint_save` checks exactly one RNG serialization before
calling `_s0_report`; only afterward does it publish the actual serialized RNG
record and small native state. The observed failure occurs before those files
are written. Thus no rank S0 report, `s0-native-state.pt`, tensorpack, per-update
direct-reference record or partial payload exists: only the manifest and eight
failure JSON files exist in `numerical_probe`. The actual RNG serialization
comparison record is unavailable despite native RNG metadata being present.

Existing `after_load` requires a complete S0 manifest, successful rank reports,
actual native RNG serialization and hashed small-state records (Snapshot009
`numerical_probe.py:1531–1618`). This old checkpoint **cannot serve the existing
observe protocol**. A new valid S0 record is required through the supported path;
do not invent a repair mode or bypass these checks.

## Resources, remaining gates and retention

Nine samples cover 08:58:02–08:59:38 UTC. Maximum sampled owned RSS is
189,812,375,552 bytes (176.777 GiB), with up to 289 sampled processes. RSS sums can
double-count shared pages. Per-device sampled used-memory maxima span
28,729–33,043 MiB; these are `nvidia-smi` samples, not allocator or instantaneous
peaks. Directory sizes are logical bytes, normally sampled every 30 seconds;
the 08:59:04 phase-size field is empty, while attempt size is 39,221,285,891 bytes.
The unchanged filesystem-free samples do not measure checkpoint write traffic.
Sampling omits the final teardown interval. No device-compute, host-submission,
wait, copy/communication or stable-step timing attribution exists; native save's
rank-0 19.11-second host span is a single startup/save observation, not throughput.

Owner status reports natural exit and all 658 recorded identities absent.
Root independently checked 660 PID/start-tick identities, all absent, and retained
an eight-UUID idle/no-app [postrun query](../../attempts/001/root-postrun-gpu-query.json).
These are completed release observations, not permission for another launch.

Open: valid S0 capture/readback; fresh-process full restore; exact state, input,
accumulation and RNG continuation verification; observed step-2 payload; repeated
nondeterministic A/A calibration before any A/B numerical verdict; ten-update
generation, graph-on behavior, performance and quality acceptance. No independent
bitwise numerical gate is imposed.

**Retain this checkpoint now.** Root, the codec fix owner and its independent
integration/failure-review consumers must explicitly finish their diagnostics
before it becomes eligible for the already authorized later cleanup. Preserve
failed logs/manifests, this extraction, exact path and nine-file size inventory,
small metadata hash and any consumer-required provenance before deletion; record
the cleanup result and storage released. Do not hash tens of GiB without a concrete
consumer need. This analysis deletes nothing and requests no new approval.
