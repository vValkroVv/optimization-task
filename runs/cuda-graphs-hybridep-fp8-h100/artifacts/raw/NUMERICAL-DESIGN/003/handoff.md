# Joint ten-update numerical observation under the current disk limit

Design only, 2026-11-12. No source changes, GPU execution, checkpoint creation,
installation or cleanup. Owns this directory only. This uses the code-rules and
perf-experiments skills. Documentation has no production runtime impact.

## Decision and finite acceptance

**Recommend one encoded reference: native S0 plus the actual accumulated
preclip gradients for ten consecutive native updates, with exact same-run
hash-verified native-kernel decoding. This is a conditional design, not an
implemented or accepted recorder.** It fits the source-derived storage estimate
without assuming compression. One small native inventory/round-trip probe is
required before implementation can claim actual byte feasibility or launch the
full numerical pair. If decoding is not exact or the inventory exceeds the
budget, leave the coherent-ten-update gate open. Do not substitute stripes,
repeated training prefixes, scalar agreement or a second training model.

Acceptance checklist for this deliverable:

1. Reconcile prior designs with the runnable finalist and current storage.
2. Establish source-derived tensor/state ownership, dtype and byte inventory;
   distinguish runtime unknowns and specify one minimal probe.
3. Define a coherent reference, exact observation/decoding boundaries and
   losslessness checks, including persistent non-optimizer state.
4. Bound disk/host/GPU/copy costs, failure behavior and retention.
5. Preserve A/A-based precision criteria and separate native continuation;
   check local links and whitespace.

[002](../002/acceptance-amendment.md) governs floating acceptance. Its independent
stripes/prefix cohorts cannot close the joint trajectory gate.
[001](../001/handoff.md) supplies historical hook investigation, not current
bitwise A/A acceptance or a feasible ten-update archive recommendation.

## Frozen workload and evidence

Use [source006](../../SOURCE-SNAPSHOT/006/manifest.json), repository base
`446431dfa191ef0300f116257ae918bc85e38c06`, snapshot archive SHA256
`324cad36a75404f235f5dd7de10b571a75ad486276a44c09fe425f3187ebd51b`.
Its preset SHA256 is
`b341701bd1eef7774c110fbf7ed69cf6280ba8691100b05be457dee3e7d50a7b`.
[Actual A008 config](../../LANE-A/attempts/008/native/config/resolved_config.yaml)
and [mechanism review](../../LANE-A/analysis/008/mechanism-review.md) establish
the selected execution: exact 3B MoE, 18 compiled model blocks excluded from
graph capture, persistent compiled chunked-loss forward/backward graph replay
on eight ranks. Keep FlexAttention, HybridEP, dense/sparse FP8, FSDP2/EP8,
selective AC on all 18 blocks, local batch 8/global 256/sequence 4096,
accumulation 4, seed/data identity and 2,000/30,000 schedule.

Both controls use `mix_order_reduction=0`, `graph_partition=True`, the public
step marker, dense unsafe adapter and corrected scan. Only the global graph
switch is 0/1. Do not reuse strict/debug compiler policies or old unmatched
controls. Keep the same resolved data/loader state, actual four microbatches,
inputs/labels, masks, valid-token and auxiliary denominators, optimizer groups,
parallel memberships, source and runtime pins in the numerical manifest.

[C1 diagnosis](../../C1-STORAGE-LOGGING/001/diagnosis.md) records ENOSPC log gaps;
console success cannot replace a complete all-rank numerical manifest. No native
checkpoint exists from those checkpoint-disabled runs. The parent supplied
225.6 GB free; this read-only local `df -B1 .` observed **227,271,770,112 bytes
(211.66 GiB)** free on the task's NFS mount. This fluctuating observation is not
a reservation. Do not use the historical roughly 666 GiB allowance.

## Source and runtime inventory

[Native trainer](../../../../../../gift/torchtitan/trainer.py),
[optimizer](../../../../../../gift/torchtitan/components/optimizer.py),
[FSDP ownership](../../../../../../gift/torchtitan/distributed/fsdp.py),
[MoE state](../../../../../../gift/torchtitan/models/common/moe/__init__.py),
and [router update](../../../../../../gift/torchtitan/models/common/moe/load_balance.py)
were inspected. A008 resolves `training.dtype=float32`, mixed parameter BF16,
reduction FP32, `optimizer.implementation=fused`, AdamW, no custom param groups,
LR 0.0004, betas 0.9/0.95, epsilon 1e-20, decay 4e-5. Fused BF16 optimizer-state
mode is **not** selected. Native Adam initializes moments with `zeros_like(p)`;
FP32 master/sharded parameters imply FP32 moments and gradients here. These are
source/config conclusions, not a runtime dtype inventory.

Existing [native model count](../../NATIVE-SMOKE/001/supervisor.log) is
3,266,363,904 parameters. Expert constructors allocate
`w13_E2FD=(32,2,1024,1536)` and `w2_EDF=(32,1536,1024)` for each of 17 MoE
layers: 2,566,914,048 expert and 699,449,856 remaining elements. Expert EP
ownership divides dim 0 across eight ranks; FSDP dense weights shard dim 0
across the eight-rank dense mesh, expert FSDP uses the remaining efsdp mesh.
Do not count the same expert shard eight times or treat dense weights as
replicated merely because TP=1. Verify the actual placements/ranks and padding
in the probe. No tied embedding/head assignment was found in Decoder's separate
constructors; prove aliases by actual storage identity, including optimizer
state, before counting them once. A view's backing storage can exceed its
logical payload: serialize detached compact CPU payloads and keep separate
stride/view/alias metadata; do not accidentally save whole flat storages.

Let `F = 13,065,455,616 bytes = 12.168153763 GiB` for one FP32 family, assuming
unique complete shards and no extra serialization padding/replicas.

| Family / owner | Native dtype and source-derived payload across eight ranks | Retention |
| --- | --- | --- |
| Master parameters / wrapped model | FP32, F; nominal F/8 = 1.5210 GiB/rank | Native S0; thereafter decode only |
| Accumulated preclip gradients / actual optimizer parameters | FP32, up to F/update; preserve None | Ten updates: 130,654,556,160 B = 121.6815 GiB |
| Actual postclip gradients / same owners | FP32, up to F/update | Hash every tensor; decode, no full archive |
| Adam exp_avg and exp_avg_sq / inner optimizer.state[p] | FP32, F each after initialization; step scalar per active parameter | Native S0 and actual per-step hashes; decode later |
| Native S0 P/m/v | 39,196,366,848 B = 36.5045 GiB, plus metadata/state | One nonzero full native checkpoint |
| Router expert_bias_E / 17 MoEs | FP32, 17 x 32 x 4 = 2,176 B per complete replica | Actual values every required boundary, preserve replicas |
| Router tokens_per_expert_E / 17 MoEs | FP32, same logical size, nonpersistent | Actual per-rank counts pre-hook and zero/reset checks after step; not assumed integer dtype |
| Auxiliary totals/scale / wire_aux_loss shared vectors and views | Created FP32; size from wired names | Actual per-microbatch values, names, aliases and normalizers |
| Inputs/labels / native four fetched batches | Actual dtype, commonly int64 | Store actual elements, not hashes alone; two int64 arrays cost about 160 MiB for ten updates across eight ranks; extra fields measured |
| RNG/loader/scheduler/trainer/callback state | Native exact bytes/types; includes rank-qualified RNG continuation | S0 plus small per-step structural records and required floating records |
| FP8/cache/candidate state | See below; no invented BF16/FP8 substitution | Persist actual non-derivable state if present; bounded inventory gate |

FP8 is current-operand quantization in `FP8Gemm`, shared activation quantization,
`quantize_expert_weight` and fresh `FP8DispatchState` in HybridEP. Quantized
payloads/scales are not FP32 masters. `Float8BlockwiseQTensor` carries E4M3 data
and FP32 scales, with metadata; inspect both planes if any becomes persistent.
No delayed amax history or persistent numerical weight cache was found on this
active path. Backend compiled-kernel/transport caches and graph executable pools
are not optimizer numerical histories. The candidate graphs only capture loss
here. Runtime inventory must still inspect model buffers **and** plain attributes,
quantized wrappers, dispatch-owned state, callback state, graph-owned persistent
floating buffers and their lifetimes; `state_dict()` alone misses nonpersistent
buffers and aux attributes. Save actual non-derivable floating bytes if present.
RoPE/frequency caches are nonpersistent, source-derived immutable data: hash
against reconstruction and record ownership. Ephemeral FP8 operand/scale values
remain unobserved by these hooks; do not claim transient plane elementwise parity
or add a universal FP8 tracer. If required persistent/cache state is discovered,
its bytes and checkpoint behavior are an explicit gate, not silently omitted.

### Installed native kernel contract

Read the actual runtime files under
`/opt/torchtitan-venv/lib/python3.12/site-packages/torch/` without importing torch
or using CUDA. `version.py`: **2.15.0.dev20260822+cu130**, CUDA 13.0, git
`8b353065ce260336c5fdfa74d754e8bd9b91b062`.

| File | SHA256 |
| --- | --- |
| optim/adam.py | 6f4a96e7bccd5e233a66f846c39f7a9da0157cf657c2275d2a269a78a6b90921 |
| include/ATen/native/cuda/fused_adam_utils.cuh | b986a984e827593487b8814a756a9548f1a0c2ffcbbec265b30a5d9051e225e4 |
| nn/utils/clip_grad.py | 5e30dac17eb0311ab1a12e46ed2484decb66165427644d2601d13570ab0ee4d7 |

The installed `adam_math` uses native FP32 opmath, including nested FMA for
moments; this differs from older unfused/Python formula evaluation.
[Matching upstream kernel](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/aten/src/ATen/native/cuda/fused_adam_utils.cuh)
confirms per-element reads/writes with scalar LR/betas/epsilon/weight decay and
per-tensor step. No cross-parameter reduction occurs in that kernel. `_fused_adam`
increments step tensors, groups by device/dtype and calls the native fused
AdamW op. Global clipping is the cross-parameter coupling; **record its actual
global scalar and do not recompute the norm from a decoder subset**. The
container pre-hook also updates router bias with collectives; store its actual
state rather than reexecute that hook in the decoder. Native optimizer state
and groups must show no unaccounted AMP scaling/found_inf, AMSGrad or hooks;
if present record/support their exact state or stop. Pin AdamW wrapper, loaded
native library identity and all-rank runtime versions in the probe too.

Installed clipping uses `max_norm / (total_norm + 1e-6)`, clamps at 1 and invokes
native foreach multiplication separately for the expert/dense sets. Save the
actual returned total_norm tensor in its native dtype and the native coefficient
bytes (a tiny opt-in hook at that boundary is needed for literal observation;
no public callback exposes it). Decode with the identical native scalar
arithmetic and foreach multiplication, checking against **actual** postclip
hashes. CPU multiplication, manual Adam math, BF16 casts, CPU AdamW and an older
PyTorch reference are unsuitable exact decoders.

## Established storage options assessed first

[PyTorch serialization](https://docs.pytorch.org/docs/2.15/notes/serialization.html)
uses uncompressed ZIP64 and preserves storage sharing; mmap does not compress
bytes. Plain per-rank tensor serialization is the simple reference format, but
all five ten-step families cost 608.4077 GiB. S0 already contains P0, so avoid a
redundant P0 file; even then the complete payload plus S0 is 644.9122 GiB.

The installed DCP filesystem writer supports experimental `_extensions` and its
`_extension.py:ZStandard` implements the maintained stream transform. Native
GIFT checkpointing passes no compression extension. The optional zstandard and
pyzstd package directories were absent in this inspected environment; no package
was installed. [Official DCP compression design](https://pytorch.org/blog/reducing-storage-footprint-and-bandwidth-usage-for-distributed-checkpoints-with-pytorch-dcp/)
uses zstd with custom writer integration. Its reported savings do not establish
this workload's ratio. Compression is an option only after a lossless native
round trip and measured final bytes; it is not a way to change native S0/Q5
formats during this design.

[ZipNN paper v2](https://arxiv.org/abs/2411.05239v2) and
[maintainer implementation](https://github.com/zipnn/zipnn) provide float-aware
lossless compression. Published model-weight ratios do not establish ratios for
all ten gradients, Adam moments and evolving parameters. Choosing it introduces
a dependency and dtype/layout integration without a measured benefit here;
no install or unpinned port is recommended. A full compressed archive must fit
176 GiB including S0/headroom accounts: that needs roughly 73% reduction even
if all 644.9 GiB are compressible. No such ratio is evidenced. Generic lossless
compression may reduce the selected encoding further, but the budget assumes
**zero** compression savings. No additional compression benchmark is necessary
before selecting this conditional route.

The precise gap justifying the proposed encoding is redundant deterministic
pointwise optimizer state between recorded actual gradients. Established
serialization/compression does not guarantee the required size here. This is an
observation encoding with native arithmetic and exact verification, not a new
optimizer or checkpoint format. Source inspection establishes eligibility;
actual decoder losslessness remains a runtime gate.

## Recorder and reader contract

Use native `Trainer.train_step` through the campaign trainer/callback route.
A small opt-in hook binding is future source work requiring its own authorized
scope; this report implements none. Preserve native loss/optimizer/checkpointer.

| Boundary | Record / validate |
| --- | --- |
| After native load and one-shot RNG restore | Exact S0 P/m/v/steps/groups, aliases/shards, loader/RNG/scheduler/trainer/callback state; common native nonzero-step start |
| after_dataloader / after_forward | Actual four changing input/label batches; main and auxiliary loss/scales/counts after each backward. after_forward occurs after backward in this trainer |
| before_optimizer_step | Exactly four backwards complete; write actual accumulated preclip local shards and None metadata; pre-update parameter hash equals prior endpoint; snapshot actual router counts |
| Native clip boundary | Actual returned global norm and clipping scalar, with native dtype and both set identities; no subset reduction |
| Each inner AdamW step pre-hook | Actual postclip hashes; LR actually used, groups, steps, grad presence; after container router hook, record router bias/reset bytes |
| after_optimizer_step | Actual P/m/v/step hashes, floating persistent state; optimizer completes once; scheduler has advanced, so record next LR separately |
| batch_end and completed checkpoint boundary | Native loss sum, ntokens_seen, loader/RNG/structural state and completion record, all ranks |

Read only actual local DTensor shards after pending work completes. No
all-gather/full_tensor/redistribution or dtype conversion for snapshots. Record
FQN, optimizer part/group/index, global/local shape, stride/storage offset,
dtype/byte order, placements/mesh membership/rank, alias/view map, None/empty
status and state keys. Exclude padding only if source ownership proves it is
not logical state; otherwise retain and describe it. Keep model parameters and
optimizer references mapped by identity after wrapping.

Every baseline update appends preclip payload plus actual phase hashes and
small records from **that same execution**. The evolving training state is never
restored between the ten updates. S0 plus these inputs encodes one actual joint
trajectory. Hash all required actual tensor bytes, with domain-separated
update/rank/FQN/family/dtype/shape metadata, before reuse/mutation. Hash logical
elements in canonical order and validate alias/layout metadata separately.
SHA256 integrity checks are adequate here; they do not estimate A/B errors.

After recording, validate decoding before allowing A/A or A/B comparison:

1. For each actual local tensor, read its S0 P/m/v and actual per-tensor step,
   the ten actual gradients, per-step group values and clipping scalars.
2. On the same pinned CUDA device/runtime, reconstruct native postclip gradients
   using the native foreach path and apply **native fused AdamW**, with the
   recorded optimizer defaults/state and step progression. No container router
   hook, loss, forward, backward, data fetch or training RNG use in decoding.
3. Check every decoded postclip/P/m/v/step against that baseline's actual hash
   at all ten boundaries. Check pre-update parameters, None handling, no hidden
   mutations and state-key coverage. Decode only whole local tensors with their
   native layout; do not split arbitrary tiles and assume aligned/vectorized
   paths match. Different grouping/alignment dispatch is an explicit hash gate.
4. Emit a complete all-rank decode-validation manifest. A single mismatching or
   missing hash invalidates the encoding; do not call it approximate state or
   use it for numerical comparison. Retain affected evidence and stop.

This check tests deterministic decoding of one recorded run. It imposes **no
bitwise acceptance between independent A/A trajectories**.

A separate A/A or candidate process runs its own ten uninterrupted native
updates. Its reader decodes the verified reference one local tensor at a time
using bounded scratch; compare every actual consumer family to decoded native
reference bytes with the frozen floating criteria. Verify a decoded reference
hash before that comparison too. Compute parameter deltas from actual native
endpoints in FP64 per tensor, including the reference endpoints verified above;
do not compare Adam formula estimates. Keep reconstructed reference P/m/v on
CPU between updates (about 36.5 GiB globally, 4.56 GiB/rank nominal), retain
consumer native pre-update CPU endpoints for actual deltas (another F globally),
stage one
whole local tensor at a time to CUDA, then release scratch. This is optimizer
state decoding beside the one live training model, with no second forward/
backward model and no IPC training harness. Verify host RAM and GPU scratch
headroom first. A purely offline decoder can validate each tensor's entire
trajectory without holding whole reference state in GPU memory or writing the
expanded archive. Decoder operations must not advance training RNG or mutate
consumer tensors, optimizer groups or graph storage.

If postclip reconstruction alone cannot match actual bytes, a lossless XOR byte
residual between decoded and actual postclip gradients can be assessed as a
bounded fallback, with an established lossless compressor and exact hashes.
The residual is integer byte XOR, not rounded float subtraction. It has **no
assumed sparsity/compression ratio** and may cost another 121.7 GiB; that does
not fit. Only use it if measured worst-case retained bytes fit the same budget;
otherwise stop. Do not grow infrastructure or silently drop preclip gradients.
A P/m/v decoding failure requires source/runtime diagnosis; arbitrary residuals
for every state are not an automatic expansion of this design.

## Disk, memory, I/O and failure bounds

Proposed validation payload ceiling: **176 GiB (188,978,561,024 bytes)**,
including S0, all retained references, scalar/input/state records, temporary
payloads, checkpoints and logs created for this validation. Keep an additional
**32 GiB free reserve** on the actual mount, verified before every major write.
Use `min(176 GiB, current_free - 32 GiB)` as the live ceiling for new retained
payload, accounting for existing validation files rather than subtracting them
twice. The observed mount has roughly 35.66 GiB beyond the proposed ceiling.
This is a bounded plan, not a reservation or permission to fill NFS.

Estimated S0 + ten raw gradients is **169,850,923,008 B (158.1860 GiB)**.
Remaining payload allowance is **19,127,638,016 B (17.8140 GiB)** for actual
checkpoint overhead, inputs, persistent state, metadata, logs and bounded
partial output. Do not spend it on a second reference/checkpoint or complete
expanded decoded family. Inventory exact serialized/stored bytes, not only
numel, before acceptance. Use atomically committed per-rank/update payloads and
all-rank completion manifests; a renamed partial file still consumes space.
Bound each rank's pending write to one tensor; a common budget/reservation
manifest must count all ranks' simultaneous in-flight bytes. Refuse the next
write before consuming reserve. An NFS ENOSPC, failed fsync, missing rank record
or quota error invalidates completion and stops further payload generation.
Keep compact failure/progress evidence in the bounded available route; ordinary
logging error handling is not a success criterion. No unrelated cleanup.

The recorder still reads/copies actual preclip, postclip, P/m/v and the
pre-update parameter identity check to CPU for byte hashes: about **6F per
update**, roughly **730.1 GiB D2H for ten updates** across
eight ranks, plus S0/state/inputs. Hash-only families save disk, not copy cost.
Writes are about 121.7 GiB plus S0 and small records. Single-tensor synchronous
staging avoids full-family GPU clones and unbounded pinned queues. Decoder adds
native foreach/AdamW kernel launches, H2D/D2H copies of reference state and
reference reads; offline verification adds one full pointwise decoding pass,
and each consumer adds another. Host reference cache is about 36.5 GiB globally; consumer pre-update
endpoints add 12.17 GiB globally (about 6.08 GiB/rank combined nominal), plus
bounded CPU staging. FP64 error/delta scratch is per tensor, not a full family. Largest logical
parameter shard is expected to be an embedding/head slice (about 75 MiB) rather
than an expert w13 shard (48 MiB), but runtime max storage/layout controls the
actual bound. Budget roughly four native tensor buffers plus comparison scratch
per rank before using this route, then measure actual allocator peaks.

Device compute, host work, synchronization/waits, copies and I/O overlap; no
latency is estimated by adding them. Expect substantial validation slowdown and
possible scheduling changes. The observer is absent by default: no hook, hashes,
CPU copies, decoding, compression or checkpoint serialization in performance
runs. Prove graph-on still uses persistent changing-data loss replay with
observation, and that observer lifecycle removes all hooks. Do not report its
timing as throughput evidence.

## One precise inventory/decoder probe and verification gates

No probe is run here. The unresolved decision is whether actual serialized
shards/state and exact native decoding fit the budget. One later authorized
probe should use the frozen native eight-rank target, production flags and four
real microbatches, without new broad tracing or strict determinism work:

- Initialize native optimizer state, create one nonzero native full S0 only
  after the completed native step, then execute **one** observed applied update
  from S0. Keep the real schedule horizon. Report actual state/alias/shard and
  persistent FP8/callback inventories before and after it; actual storage and
  serialized bytes; host available RAM; max local tensor bytes; all-rank native
  runtime/library hashes and optimizer defaults/hooks.
- Retain that one actual preclip payload and compact native postclip/P/m/v and
  coefficient/hash records. Decode it with the same native CUDA paths on
  separate scratch after the real update and check exact hashes. Inspect both
  actual clip sets and FSDP local layouts. Establish no consumer mutation/RNG
  consumption. Report raw timings, D2H/H2D and scratch peaks descriptively only.
- Successful exact decoding + complete inventory within projected ten-update
  ceiling selects this encoding. Mismatch/missing ownership/budget or memory
  excess leaves the gate open. No second diagnostic launch is automatic. Reuse
  S0 and this evidence if source/runtime/observer contracts remain unchanged;
  delete the one-update payload once its consumers finish, recording inventory.

Before ten-update capture, focused recorder/reader tests must catch equal-norm
unequal elements, corruption of one gradient/moment/update, a wrong LR/step or
clip scalar, None versus zero, empty/noncontiguous tensors, alias/shard omission,
nonfinite values and missing rank completion. Demonstrate injected corruption
fails at the actual decode/comparison path. Exercise cap refusal, incomplete
write, failed consumer retention and deletion eligibility. CPU/Gloo tests can
verify metadata/I/O and failure paths, but do not establish CUDA kernel parity.
The one production-shaped probe is the CUDA hook/decoder integration gate;
complete ten-step same-run hash checks remain mandatory even after it passes.

## Criteria, consumers and native continuation

Keep [002's metrics and precision method](../002/acceptance-amendment.md): all
losses, elementwise pre/postclip gradients, actual parameter deltas, moments,
parameters and floating persistent state; per-update and per-layer/expert/shard
visibility; meaningful near-zero handling; exact structural, accumulation,
normalization, data, checkpoint and RNG ownership contracts. Nonfinite or lost
state cannot be calibrated away. Routing near-ties later in evolving independent
trajectories are measured; preserve no-drop/capacity/count/reset semantics.

Use one **production A/A ten-update pair** from S0. Reference A writes the one
verified encoding; independent A' streams comparisons. Freeze metric strata,
near-zero thresholds, absolute-plus-relative element bands, delta/moment/scalar
limits, exceedance rules and uncertainty **before viewing A/B errors**. Justify
numbers with precision/operation conditioning plus observed A/A variation;
never multiply observed maxima by a convenient arbitrary constant. One short
additional calibration pair is allowed only after naming an unstable metric,
its possible outcomes and the changed freeze/verdict decision. It cannot replace
the required coherent ten-update pair or establish long-horizon quality. If
credible margins remain unavailable, the verdict stays INCONCLUSIVE. No new
baseline nondeterminism-elimination project or bitwise independent loss/norm
gate is introduced.

Reuse the **same** A reference for A' and then graph-on B, and for any explicitly
named, decision-relevant calibration consumer. Each consumer is a separate
native evolving trajectory. Preserve mismatch coordinates and compact actual
failing slices before deleting anything. Close the ten-update reference only
after all named consumers finish, all comparison/decoder manifests and frozen
criteria are saved, and no unresolved mismatch needs its payload. Then delete
only this task's unneeded gradient reference with hash/byte/consumer/deletion
inventory. Do not keep redundant full archives or final model checkpoints.

**Native 5+5 bidirectional checkpoint continuation remains separate.** Use
CheckpointManager/DCP model, optimizer, loader, scheduler, train state and
rank-qualified RNG callback ownership. S0 must be nonzero because existing
`load(step=0)` selects model-only; preserve the initial-load destination rules
from 001. Save Q5 after completed train_step/ntokens accounting; in each
direction compare loaded native bytes exactly before resumed execution, then
compare the actual five resumed updates to that variant's uninterrupted tail
with frozen floating criteria. Recreate graphs and verify replay; do not save
graph executables or restore weights alone after capture side effects.

Run the two Q5 directions serially **after** ten-update gradient payload removal.
One direction retains S0 + one native Q5 + five tail gradients, source estimate
**143,720,011,776 B (133.8497 GiB)** plus state/metadata. Generate/verify its one
tail reference from the uninterrupted trajectory, consume by the opposite
variant restart, then close/delete it and Q5 before the other direction. Hash
verification gives the actual tail families, not inference from matching norms.
Do not hold both Q5s, the ten-update reference and both tails together. Keep S0
until every named numerical and native continuation consumer finishes; delete
unneeded task checkpoints only after that inventory is complete.

## Verdict and remaining gates

Source-level native pointwise decoding is eligible and its uncompressed estimate
fits the current bounded allowance. **Actual ten-update storage feasibility,
all-rank dtype/alias/persistent state inventory, same-run CUDA decoding, observer
inertness/replay, numerical criteria and both continuation directions are open.**
This report supplies a finite route and one minimal runtime decision probe; it
does not claim numerical acceptance, production readiness or measured compression.
No additional experiments, diagnostics or implementation are part of this handoff.
