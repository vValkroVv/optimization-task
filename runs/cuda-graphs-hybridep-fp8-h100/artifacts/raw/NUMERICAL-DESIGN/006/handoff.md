# Direct numerical reference after shared storage recovery

Decision only, 2026-11-12. No implementation, GPU use, installation, source
changes, cleanup or campaign-document changes. Prior [003](../003/handoff.md),
[004](../004/handoff.md) and [005](../005/handoff.md) remain historical evidence.
Uses perf-experiments for the scoped cost assessment; this note has no runtime
effect.

## Recommendation and finite acceptance

**Replace the encoded reference with direct actual preclip/postclip/P/m/v
payloads for ten uninterrupted native updates, plus one native full S0.**
Amend only the numerical observation contract: a 700-GiB total live payload cap
and an additional 32-GiB free reserve replace the self-selected 176-GiB cap.
There is no owner-supplied numeric cap; the owner's tight-retention requirement
still applies. No concrete correctness requirement now makes encoding necessary.

This decision checks capacity arithmetic, affected interfaces, observation costs,
retention and preserved gates. It does not accept an implementation or authorize
a run. The [05:49:51 statvfs receipt](../../STATIC-CHECKS/009-storage-recovery/statvfs.json)
shows 6,701.46 GiB available, not reserved. Recovery removes the old capacity
reason for native CUDA reconstruction and the future partial-DCP reader.

## Bytes and retention

Reuse 003's source-derived FP32 inventory: 3,266,363,904 unique elements,
`F = 13,065,455,616 B = 12.168153763 GiB` per complete family across eight ranks.
Actual shard ownership, aliases, layouts and native checkpoint bytes remain
runtime inventory gates; these are uncompressed logical estimates.

| Retained object | Nominal bytes | GiB |
| --- | ---: | ---: |
| Ten actual updates, five families each | 653,272,780,800 | 608.4077 |
| One native S0, P/m/v lower estimate | 39,196,366,848 | 36.5045 |
| Combined reference | 692,469,147,648 | 644.9121 |
| One-update direct probe plus native S0 | 104,523,644,928 | 97.3452 |

The 700-GiB cap leaves 55.0879 GiB beyond the full nominal reference for actual
checkpoint overhead, steps, inputs, persistent state, metadata, logs and bounded
partial files. It is a ceiling, not a target or guaranteed fit. Count all retained
validation files and simultaneous rank writes, not only committed tensor files.
Do not create a second full reference or compact P/m/v S0 export.

Required first-update parameter-delta metrics need P0 values as well as hashes.
Both trajectories load and verify identical native S0. Use 003's consumer
previous-endpoint CPU cache: one FP32 parameter family, F globally, about
1.5210 GiB per rank nominal. Read common verified P0 from the consumer's actual
full native restore for the first reference delta; later reference deltas read
adjacent stored P endpoints. Consumer deltas use its actual previous-P cache.
Compute actual endpoint differences in FP64 tensorwise, preserving the required
delta metrics and their precision criteria; parameter equality alone is no
substitute. This is an explicit bounded host-memory exception to single-tensor
transfer/serialization staging. It needs no P0 disk export, partial-DCP reader,
second GPU model or RAM server. Measure actual host peak/headroom including this
cache, reference endpoints, FP64 scratch and serialization buffers before use.

Before payload generation, verify free bytes/inodes, actual mount and a small
write/fsync. At every major allocation, enforce both total live payload <=700 GiB
and projected remaining free >=32 GiB, including all ranks' pending bytes.
At zero existing payload, the conservative full-cap admission requirement is
732 GiB (785,979,015,168 B). Shared free space can disappear after a check:
ENOSPC/quota/fsync failure or a missing rank invalidates completion and stops new
payload generation. Never consume reserve to finish a run or silently raise cap.

Reuse one reference for production A/A and then graph-on comparison. Retain it
only until all named comparison, diagnosis and review consumers finish. Delete
eligible task-owned payloads with path/hash/byte/result inventories, preserving
compact failures and logs. Generate continuation checkpoints serially afterward
where possible; their live bytes share the same accounting. No automatic pruning
of required S0 or other contributors' files is permitted.

## Reuse and removed machinery

Reuse 005's native training/continuation route, completed nonzero S0, exact
serialization-time RNG record, pending-versus-restored RNG checks, input capture,
canonical ownership metadata, local-shard readiness, native callback ordering,
inner optimizer pre-hooks and hook cleanup. Inspected `Trainer.train_step`
(720–830), `ObserverState.optimizer_groups` and `optimizer_state` confirm:
preclip follows four backwards, inner pre-hooks see actual postclip gradients,
and `after_optimizer_step` follows optimizer and scheduler. Preserve used versus
next LR, router-hook ordering, None/empty semantics and actual state keys.

Change actual postclip/P/m/v hashes into compact payloads plus hashes at the
same native boundaries. Validate each serialized logical tensor by lossless
readback and its same-execution hash, including metadata and all-rank completion.
The final probe phase is a bounded reader/integrity check; it needs no CUDA
optimizer reconstruction, training model or optimizer arithmetic. The native
producer still uses its actual CUDA clipping and fused AdamW unchanged.

Remove acceptance requirements for singleton CUDA decoder bit-exactness,
decoder grouping/alignment equivalence, decoder residual fallback, compact m/v
S0 export and partial-DCP extraction. Preserve exact actual native norm bytes;
label a computed clipping coefficient as derived rather than observed. No new
core clipping hook is justified solely for reconstruction.

## Costs and retained gates

Direct recording adds about 486.7262 GiB of writes over ten updates compared
with retaining only gradients. Existing encoding already copied/hashes the
other four actual families, so this need not add their D2H copies. It removes
native clipping/AdamW decoder passes, reconstructed-state H2D/D2H traffic and
decoder caches. Actual I/O latency, scratch/RAM peaks, blocking drains and
graph-on scheduling effects remain unmeasured. Stage one canonical tensor at a
time; no full-family GPU clones, reference model, all-gather or redistribution.
Observation runs provide no throughput evidence. Absence must incur no observer
hooks, copies, hashes, serialization or synchronization in performance runs.

Retain CPU behavior regressions for real callback order, four accumulations,
None/zero, aliases/layouts, wrong LR/step, corrupted/missing payloads, partial
writes, budget refusal and hook cleanup; exercise the actual readback/comparison
path, not just metadata. Retain scoped independent integration/runtime-cost
review, and the one actual-target one-update probe for native readiness,
inventory, S0/RNG restoration, lossless files and measured bytes before scaling.
CPU checks cannot establish FSDP2/NCCL, FP8, compile or graph replay behavior.

Preserve the exact FlexAttention/SAC/FP8/HybridEP/compilation/accum4 workload.
Keep all native state/input/loader/RNG/router/FP8 inventories and exact structural
invariants from 005. Necessary unexposed persistent state remains an open gate.
Production A/A criteria must be fixed before graph-on interpretation under
[002](../002/acceptance-amendment.md); no arbitrary tolerance or independent-run
bitwise gate is introduced. Native fresh-process bidirectional full-state 5+5
continuation remains separate, including RNG, loader, scheduler, moments and
graph reconstruction. CUDA faults, graph-on observer inertness, ten-update
comparison and all campaign performance gates remain open.

No SOTA search is needed: the serialization option and native interfaces were
already inspected/pinned in 003/005. Recovered capacity changes their tradeoff,
not their API. If actual overhead or concurrent disk pressure prevents the
direct archive fitting its cap, stop and reconsider that concrete constraint;
do not resume encoding machinery preemptively.
