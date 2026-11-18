# Isolated Quack LMhead + cross-entropy experiment

**Result:** no material, robust latency gain on this component. The patched
4096-token Quack path lowers peak allocated memory by 33.3% (3.119 GiB), but
its paired mean latency changes range from 0.06% faster to 0.20% slower. The
8192-token path has about 0.94–1.00% lower paired means and 22.1% lower peak
allocated memory; that small latency difference remains provisional under the
observed periodic tails. Neither result establishes full-training improvement.
Both Quack variants have larger local gradient error against FP32 than native.

## Scope and acceptance

One H100 on the existing authorized job 23027817, native real-data rank-0
fixture, B2/S4096/D4096/V128256, BF16 untied head, MTP1 weight 0.3 and
compiled loss. The component includes complete forward, dX and dW. No optimizer,
GFSDP communication, offload, input loading or trunk/MTP replay is timed.

Finite checks: identify source/runtime/inputs; verify native and scaled-exp
activation; alternating unprofiled controls and candidates with 10 warmups and
30 observations; matched fresh-process memory; representative traces;
FP32-leaf loss/dX/dW oracle and relevant ignore/MTP semantics. These component
checks cannot close GFSDP integration, full training, convergence or resume gates.

## Method and source contracts

`prepare_inputs.py` loads the preserved real-input MTP boundary, initializes
canonical norm, MTP and untied head with seed 2048, and replays norm/MTP once.
It builds a pruned native config on meta before allocating. These are newly
initialized canonical modules; original checkpoint weights were not exported.
Captured positions and masks are restored. Main labels shift input IDs left
once with final ignore; native `MTPLoss._shift_labels` shifts the MTP targets.
The shared denominator is 8190 × 32 × 8 = 2,096,640, including the native final
ignored target; it is not a separate MTP target count.

Native `MTPLoss` folds heads into B4, chunks S into 2048, and applies the shared
head to 8192 rows per chunk. Its constructor selects compiled
`four_fusion.stable_ce.cross_entropy_loss`. The harness retains this actual
native path. A post-accumulate hook copies each BF16 chunk dW to FP32, clears
weight.grad, and adds the second chunk in FP32. This emulates the local
`gfsdp/_param_group.py:post_backward` collection before communication.

The candidate invokes `scaled_exp_linear_cross_entropy` directly, checks
`scaled_exp_lce_supported`, compiles its opaque custom-op route, and runs
immediate backwards for the main and MTP calls. Its internal chunks accumulate
dW in FP32 before returning a BF16 per-head gradient. The harness collects these
in FP32. The candidate therefore changes where BF16 rounding occurs: per head
rather than per mixed-head sequence chunk. Both paths share all inputs/weights,
loss weights, targets and denominator. This deliberate computation difference
requires numerical evidence and cannot establish convergence from local parity.

## Scoped cost review

The native path computes two folded BF16 projections, four compiled stable CE
calls and their backward, dX/dW GEMMs, hidden chunk copies and FP32 hidden
assembly. It materializes B4×S2048×V128256 logits per sequence chunk.

Scaled-exp computes a fused exp epilogue with target logits, Triton glue and
two transformed GEMMs; it avoids conventional full logits/dlogits, but retains
an exp matrix, per-tile metadata, FP32 dW accumulator, hidden scaling buffers
and gradients. Q4096 makes four internal token chunks over main+MTP; Q8192
makes two. Each call does gradient work during forward, so only full
forward/backward measurements are compared. The two calls change shared dW
rounding and launch/lifetime structure. Actual kernel and allocation evidence
is required before attributing any measured difference.

Host scope includes Python/autograd, loss bookkeeping, allocations and launches,
gradient reset and final device synchronization. CUDA-event spans include host
submission gaps; they are not pure device compute. Profiles separately identify
kernel busy intervals, gaps, host sync, copies and absence/presence of collectives.
Fresh subprocesses and equal lifecycle phases bound allocator comparisons.
No training source or shared package is modified. Production GFSDP gather,
reshard and reduction semantics remain an integration gate.

## Results

### Identity and execution

- GIFT task source HEAD: `e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f`.
  Existing contributor changes were preserved. The three relevant source hashes
  and harness hash are in every measured result receipt.
- Quack: `bb0e6b220f6b9ee6b36a638d3d6c6150d52db8d8`, package 0.6.5.
  Original source is preserved under remote `runtime/source/`.
- All campaign arms use the same task venv, Torch
  `2.15.0.dev20260822+cu130`, CUDA 13.0, CUTLASS DSL 4.7.0, and the existing
  `/etc/profile.d/wm.sh` compatibility profile. See the
  [setup manifest](evidence/artifacts/setup/setupmanifest.json).
- The allocation is job `23027817`, one eight-H100 node; only GPU 0 executes
  this experiment. All eight GPUs were idle before the campaigns.
- Prepared input SHA256:
  `52089ab6cc123b9d7f1f41e04cfa742f7d81a543a8a6d9b915f7ac64d91860f4`.
  The [input receipt](evidence/artifacts/input/receipt.json) contains the native
  fixture identities and resolved configuration. Physical tokens are 8192 per
  main/MTP head, with 8190 main and 8188 MTP supervised targets.

The original Q4096 pipeline failed before measurement: with M=D=4096, Quack's
frontend cannot infer whether its `(1,M)` scale vector is a row or column
operand. The preserved [failure log](evidence/artifacts/raw/pipeline-q4096.log)
records this error. The separate task-local
[patch](vendor-patches/0001-pin-scaled-exp-v-axis.patch) explicitly pins
`ColVecLoad("v")`, matching existing nonsquare inference and the per-row dX
scale. It changes no arithmetic, dtype or chunking. An independent source review
confirmed this contract. Q4096 below always means this patched copy; Q8192
uses the unmodified pinned source. The
[patch receipt](evidence/vendor-patches/receipt.json) identifies original,
patched and patch bytes. Patch SHA256 is
`1f6e4fe9d198f65a0186f6fb70f1dbba73aba0cda3601c626415be64a981febf`.

An initial venv import failure was repaired before the campaign; it had selected
system Torch 2.11 instead of production Torch. A transient SSH timeout recovered
on the same authorized route. Neither failure contributes a timing sample.
Pipeline samples are also excluded from the results below.

### Integrated component latency

Each fresh process uses 10 complete warmup calls and 30 measured calls. The
ordering is A0, AA0, Q8192-0, A2, Q8192-1, followed by A3, patched Q4096-0,
A4, patched Q4096-1. The identical A0/AA0 control pair measures allocation-local
noise. GPU input loading, canonical norm/MTP replay, compilation, profiling,
communication and optimizer work are excluded. Both dX and dW are included.
The wall timer includes gradient clearing, allocation, Python/autograd,
submission and final device synchronization. CUDA-event spans also include host
submission gaps. All measurements below are milliseconds per main+MTP call.
P95 uses the nearest-rank definition over 30 observations.

| Arm | Wall median | Wall mean | Wall P95 | CUDA-event median, submission inclusive |
| --- | ---: | ---: | ---: | ---: |
| Native A0 | 79.447 | 80.986 | 90.306 | 79.425 |
| Native AA0 | 79.474 | 81.491 | 91.127 | 79.443 |
| Quack 8192-0 | 78.356 | 80.678 | 91.922 | 78.331 |
| Native A2 | 79.499 | 81.520 | 90.383 | 79.467 |
| Quack 8192-1 | 78.552 | 80.751 | 92.462 | 78.529 |
| Native A3 | 79.196 | 81.262 | 91.233 | 79.172 |
| Patched Quack 4096-0 | 78.599 | 81.210 | 92.308 | 78.575 |
| Native A4 | 79.624 | 81.482 | 89.527 | 79.604 |
| Patched Quack 4096-1 | 79.711 | 81.645 | 89.450 | 79.690 |

The control-pair median changes by +0.033% and mean by +0.623%. The full raw
series has periodic tails near 90–94 ms in every arm; the cause is not
established. Using the immediately preceding native block as denominator:

| Candidate | Paired median elapsed-time decrease | Paired mean elapsed-time decrease |
| --- | ---: | ---: |
| Quack 8192 | 1.19–1.41% | 0.94–1.00% |
| Patched Quack 4096 | −0.11–0.75% | −0.20–0.06% |

Negative means slower. These are ranges of two block comparisons, not confidence
intervals or paired-per-iteration statistics. The small Q8192 mean difference is
close to the measured A/A mean variation; patched Q4096 does not show a robust
latency improvement. All per-iteration values and min/quartile/max statistics
are in the [summary](evidence/artifacts/summary.json) and individual receipts.
For a hypothetical 12,848-ms update with 32 fully exposed calls, multiplying
the observed Q8192 median call difference by 32 gives only about 0.24–0.28%
of that update. This is conditional arithmetic, not measured end-to-end gain;
communication and overlap may hide any component difference. No throughput or
full-step speedup is claimed.

### Matched process memory

GiB means bytes / 2^30. Every timing process follows the same warmup, gradient
clear, garbage collection, empty-cache and peak-reset lifecycle. Each candidate
repeat reproduces the same peaks. These are local component allocator metrics,
including the materialized untied head and FP32 local dW collector; they are not
GFSDP sharded state or total training memory.

| Arm | Starting allocated / reserved GiB | Peak allocated / reserved GiB | Increment allocated / reserved GiB |
| --- | ---: | ---: | ---: |
| Native | 3.123 / 4.072 | 9.370 / 9.943 | 6.246 / 5.871 |
| Patched Quack 4096 | 3.061 / 3.063 | 6.251 / 7.109 | 3.190 / 4.047 |
| Quack 8192 | 3.061 / 3.063 | 7.297 / 9.133 | 4.236 / 6.070 |

Relative to native, Q4096 peak allocated falls 33.28% and reserved 28.50%;
Q8192 falls 22.12% and 8.15%. Incremental allocated falls 48.93% / 32.18%
for Q4096 / Q8192. Incremental reserved falls 31.07% for Q4096 but rises 3.39%
for Q8192 because its starting reserved pool is smaller. Report both absolute
and incremental peaks; do not translate these into a full-training saving.

### Trace mechanism and timing scopes

One post-warmup profile per variant confirms the active implementation. Native
has four `forward_kernel` and four `backward_kernel` launches from the inspected
stable-CE implementation. Both Quack traces contain
`quack::lce_scaled_exp_fwd` and the scaled-exp epilogue/dX/dW GEMMs. There are
no communication kernels. Full trace paths, sizes and hashes are in the
[trace receipts](evidence/artifacts/trace-receipts.json); traces remain on NFS.

| Profile | Kernel launches | Kernel interval union ms | First-to-last kernel span ms | Gaps within span ms | Memcpy count / summed ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native | 71 | 82.960 | 83.680 | 0.721 | 1 / 0.0013 |
| Patched Quack 4096 | 46 | 85.921 | 86.856 | 0.935 | 2 / 0.1031 |
| Quack 8192 | 36 | 84.290 | 85.201 | 0.912 | 2 / 0.1000 |

These are representative profiled device-work observations, not unprofiled
kernel-cost magnitudes. They do not demonstrate a kernel speedup: the changed
GEMM pipeline dominates device work in both paths. The profile spans are larger
than the unprofiled block medians and are not mixed into the timing comparison.
The gaps are idle/wait intervals within this operation's kernel envelope; their
individual causes are not established. Profiled host synchronization is
reported separately in the summary. Standalone nonoverlapping Python/dispatcher/
autograd CPU-work time is unavailable; those costs remain included in the
integrated wall scope and visible as CPU events in the full traces. Copy sums,
kernel work and host waits overlap and must not be added.

### Numerical effects after performance

The oracle uses actual FP32 input/weight leaves, FP32 logits and gradient
accumulation, TF32 disabled, and streams 128-token slices to bound memory.
It uses the same BF16-representable fixture values and shared denominator.
No arbitrary absolute tolerance is used to pass normalized gradients.

| Variant | Loss absolute error | dX relative L2 | dW relative L2 | dX / dW error divided by native error |
| --- | ---: | ---: | ---: | ---: |
| Native | 0 | 0.00178036 | 0.00186778 | 1 / 1 |
| Patched Quack 4096 | 3.7253e−9 | 0.00280379 | 0.00249492 | 1.575 / 1.336 |
| Quack 8192 | 3.7253e−9 | 0.00280379 | 0.00249488 | 1.575 / 1.336 |

Oracle loss is 0.05989603325724602; native matches that FP32 scalar exactly.
Quack loss is 0.059896036982536316. All gradients are finite. The smallest
cosine is 0.999997718. Maximum absolute errors, full cosines and nonfinite
counts are in the [numerical receipt](evidence/artifacts/raw/numeric-oracle-patched4096/result.json).
The larger Quack errors are consistent with its changed gradient construction
and rounding contract, but this experiment does not isolate every error source.
Both Quack chunk sizes have identical dX in the reported metrics and nearly
identical dW error. All three paths pass the all-ignored case with exact zero
loss, dX and dW, and verify main/MTP ignored tails and the shared denominator.
The [native](evidence/artifacts/raw/numeric-native/result.json),
[Q4096](evidence/artifacts/raw/numeric-q4096/result.json) and
[Q8192](evidence/artifacts/raw/numeric-q8192/result.json) receipts record those checks.
These local errors are observations, not a convergence or training-quality pass.

## Reproduction and disposition

The remote experiment root is
`/workspace-SR008.nfs2/vkropotin/optimization-task/runs/quack-lmhead`.
`prepare_inputs.py`, `benchmark.py`, `run_campaign.py`, `run_patched4096.py`,
`apply_vendor_patch.py`, `analyze.py` and `package_evidence.py` are retained
beside this report. The two campaign state files preserve exact child commands:
[unmodified-reference campaign](evidence/artifacts/raw/campaign-state.json) and
[patched4096 campaign](evidence/artifacts/raw/patched4096-state.json).
Both runners source their environment from the invoking shell's native
compatibility profile, use only GPU 0, run serial subprocesses, stop on failure,
and enforce process/campaign deadlines. Their declared PYTHONPATH chooses the
unmodified or patched source explicitly.

All finite component acceptance items are complete: identities, active full
forward/backward, alternating latency/noise, matched memory, profiles and FP32
local numerical comparisons. The original Q4096 failure is preserved and the
minimal adapter is labeled. No GIFT source changes, shared package changes,
optimizer steps or full-training run were performed by this experiment.

Stop here: the candidate is a memory tradeoff with no material robust latency
gain on this workload. Production GFSDP collectives/buffer lifecycle, distributed
accumulation over 32 microbatches, compilation in the full trainer, convergence,
checkpoint/resume and full-step latency/memory remain open gates. A component
measurement cannot close them. No additional campaign is implied.
