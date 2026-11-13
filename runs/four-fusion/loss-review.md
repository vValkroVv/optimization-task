# Independent loss integration and runtime-cost review

Reviewed 13 November 2026 against root `33b336a5`, the two independent patches,
the implementation/design records and their retained CUDA evidence. Applied
`code-rules` and `perf-experiments`. No training source was written during the
initial review. Review acceptance is finite: integration, arithmetic/stability,
autograd/wrappers, optional paths, runtime costs, and disposition of the design's
findings. Its only required correction, R1 below, has since been resolved;
production evaluation remains open.

## Required correction

**R1: Validate FLCE target metadata before the Triton launch.**
[linear_ce.py](../../gift/torchtitan/experiments/four_fusion/linear_ce.py):115–123
flattens labels without validating their dtype, shape or device, and the selector
at lines 191–203 does not establish those invariants. In contrast,
[stable_ce.py](../../gift/torchtitan/experiments/four_fusion/stable_ce.py):165–170
validates them. CUDA eager and compiled FLCE accepted int32 targets while the
baseline rejected them with `expected target dtype to be Long or Byte, but got
Int`. A shorter target buffer can reach `tl.load(Y + row)` without a target-size
mask; the vocabulary assertion cannot validate that allocation.

Required change: enforce signed int64 labels, matching `input.shape[:-1]` and
label shape, and matching device at the native adapter boundary before reshape
and launch. These are metadata checks and require no device scalar read. Cover
wrong dtype, incompatible shape and device explicitly; preserve the valid eager
and compiled native MTP/two-chunk path. No new numerical comparison or broad
suite is needed for this metadata-only correction beyond affected regression
and integration coverage. Root assigned the correction to the original owner.

The focused reviewer reproduction used only previously authorized physical GPU
2 after verifying zero memory usage and no compute applications. It exercised
`LinearCE.chunk_loss` with a BF16 `nn.Linear`, B2/S4/D4/V17, denominator 8 and
int32 zero labels, both eager and compiled, and compared baseline rejection.
Command environment was the implementation record's prefix and interpreter;
exact transcript is retained in ignored
`artifacts/raw/loss-checks/review-int64.log`. The process exited successfully;
the postcheck found no compute applications. No TPS run or profiling occurred.

### R1 resolution

`native_linear_ce` now checks signed int64 labels, `labels.shape == input.shape[:-1]`,
and device agreement among input, labels, weight, and denominator before either
reshape or custom-op dispatch. The checks use dtype, shape, and device metadata;
they do not inspect tensor values or alter the Triton launch. The targeted
regression rejects int32 labels, a mismatched target shape, and mismatched label,
weight, or denominator devices in eager and `fullgraph=True` compiled calls. For
invalid metadata, fullgraph reports the Python validation exception as Dynamo
`Unsupported` while tracing; the test verifies the original validation message.
The existing native two-chunk MTP integration test passes in eager and compiled
modes. The temporary `linear-ce.patch` used to activate that integration path
was reversed, restoring `gift/torchtitan/components/loss.py` to its precheck
SHA256. Exact commands, failure resolution, and scoped checks are recorded in
the [R1 correction note](loss-metadata-fix.md).

This metadata-only correction adds no kernel, tensor allocation, data transfer,
collective, or scalar-value read. It adds host-side metadata comparisons once
per selected chunk call. The reviewed 1536-versus-192 head GEMM count and live
dW/logits overlap are unchanged; no speed or memory measurement was run for R1.

## Integration and arithmetic disposition

No other material defect was found on the required crop path.

- **Stable statistics:** forward and backward keep FP32 maximum and
  log-denominator separate. Centered loss and probabilities resolve the donor's
  large-common-offset defect. Existing V128256 `2**28` CUDA eager/compiled
  regression passes; preserved donor fails. Ignore-index rows are exactly zero.
- **Rounding and scale:** target subtraction and known head weight/denominator
  enter FP32 dlogits before the single BF16 store. Existing near-certain-target,
  partial-vocabulary, all/mixed ignored and nonunit CE upstream checks cover the
  documented defects. The FLCE internal tile accumulator is FP32 across all
  tiles and folded heads; only its final dW is cast to head dtype. Installed
  FP32-output `addmm` was executed in the recorded checks.
- **Native autograd:** the FLCE op precomputes dH/dW under an explicit
  unit-upstream SUM contract; `chunk_loss` returns `terms.sum()` directly to
  the inherited immediate inner backward. It does not claim arbitrary term
  weighting or higher derivatives. The existing decoder gradient bridge already
  ignores external upstream scaling; neither patch changes that inherited
  contract. CE supports ordinary nonunit first-order upstream scale. Both custom
  backward paths reject higher derivatives explicitly.
- **Head wrappers:** FLCE calls wrapped `head(...)`, preserves the `input`
  argument name and accesses weight after unshard. GFSDP
  `_param_group.py:710–718,978–1009` attaches input post-backward functions and
  `_state.py:601–608` attaches differentiable-output pre-backward hooks; vector
  loss terms retain those graphs. No direct `weight` call bypasses the module.
  The actual head uses an own gradient buffer and divide factor 1.0 in
  `gift_moe/_base/parallelize.py:195–233`. The unchanged loop keeps one gather
  across outer chunks and final-chunk reduction when incoming sync permits it.
- **Borrowed flags:** actual single-device GFSDP and existing protocol evidence
  establish canonical teardown plus incoming last-backward/sync handling and
  exception cleanup. The inherited loop restores reshard policies to `True`
  (`loss.py:763–773`); it does not snapshot arbitrary incoming reshard policies.
  The protocol double records setter calls rather than testing those arbitrary
  policies. Do not describe this evidence as universal exact flag restoration.
  Neither patch changes that behavior or the required canonical workload.
- **Optional paths:** FLCE retains original logits for evaluation without input
  gradients, shape probes, argmax acceptance sinks, DTensor inputs/labels,
  biased or softcapped heads, unsupported forward implementations and non-FSDP
  local hooks. Head logits calls outside the loss remain original. CE delegates
  DTensor inputs to native distributed CE. Existing hook/eval/probe tests pass;
  distributed vocabulary sharding was inspected, not newly executed.
- **State/preset:** no parameters, buffers, optimizer state or checkpoint keys
  are introduced. Identity and keys pass existing native/GFSDP checks. MTP1,
  weights 1.0/0.3, shared denominator, packed boundaries, two outer chunks and
  FP32 hidden-gradient accumulator remain intact. Trunk FP8, BF16 head, model
  and loss compile, SAC, accumulation32, prefetch32 and epoch51200 are not
  changed. Int64 row addresses avoid vocabulary-product overflow; R1 remains
  necessary to establish label dtype at the FLCE boundary.

Existing evidence was reused, not rerun: 17 focused checks per independent
patch, 31 lifecycle/monitoring checks, and world-size-one matched GFSDP control/
candidate with 32 microbatches and two outer chunks, eager and compiled. Recorded
native counts are 64 FLCE or 128 CE calls; maximum absolute dH/dW differences
are `1.9073486328125e-6` / `2.384185791015625e-7`. The harness uses actual
FP32-master/BF16-compute GFSDP heads and FP32 reduction/accumulation; its small
H2/B2/S8/D32/V520 shape does not prove production memory or rank ordering.

## Final runtime-cost review

**Experiment 3.** Projection, CE and both backward GEMMs still execute; work
moves into the forward-labelled opaque op. Eight internal 1024-row tiles per
folded outer chunk give 24 GEMMs rather than three large GEMMs. Across two
outer chunks and 32 microbatches this is 1536 versus 192 head GEMMs per rank per
update; arithmetic is broadly retained, while GEMM efficiency and launch costs
can change. There are 16 row-kernel launches per outer chunk, FP32 accumulator
zeroing, a final BF16 dW cast, and eager per-head scalar reductions/scaling inside
the custom-op body. The fullgraph wrapper does not compile away opaque-body
Python dispatch or allocations.

At V128256/D4096 the FP32 dW accumulator is **1.957 GiB**, and its final BF16
dW is **0.979 GiB**. They coexist during the final cast, together with the last
logits tile and the full dH output; existing GFSDP buffers coexist too. The FP32
accumulator dies when the custom-op forward returns. BF16 dW and dH are saved
until that chunk's immediate backward consumes them; they are not deliberately
retained for 32 microbatches. A logits tile is **250.5 MiB**, but the next
`F.linear` allocates before Python rebinds `logits` at line 44, so the previous
tile can coexist transiently: do not treat 250.5 MiB as the op's peak logits
allocation. Full dH is **64 MiB** per folded outer chunk, in addition to the
inherited FP32 hidden-gradient accumulator and contiguous chunk copies.

No target-count `.item()`, per-chunk H2D weight tensor or new collective is
introduced. Moving dW/dH computation before autograd hooks can change overlap,
exposed waits and buffer ownership time. Full-region traces must include this
forward work and the remaining backward/hooks; forward-only or backward-only
timing can misattribute the change.

**Experiment 4.** Head forward/dH/dW GEMMs, full BF16 logits and BF16 dlogits
remain. Two FP32 row-stat vectors plus the small loss vector replace no
full-vocabulary FP32 tensor in the already compiled control. The online row loop
and separate SUM reduction can be slower; the concrete opportunity is removing
the original backward one-hot reduction work. Row-stat storage lives until CE
backward. Folded head slices are contiguous on this workload, so wrapper
`.contiguous()` does not itself add a full logits copy; other layouts may copy.
There are four logical CE calls per microbatch and 128 per update. No new
collective or host-value predicate is introduced, but collective overlap may
shift with CE duration.

Device compute, host dispatch, synchronization/idle, copies/communication,
integrated elapsed time and phase-matched memory remain separate measurement
scopes. Source counts establish expected costs, not magnitude or a speedup.

## Remaining acceptance gates and review limit

Close R1 with affected behavior checks. The coordinator owns the unchanged
production gates: eight-rank activation and GFSDP collective ordering, full-shape
memory fit, independent monitoring-off 10+10 matched control/candidate runs,
matched bounded Perfetto forward/backward diagnostics and paired phase memory;
representative local numerical replay is required for a retained candidate.
No source-level review or world-size-one success closes those gates. Neither
candidate is accepted for performance or production readiness by this review.

Reviewed hashes before correction: core loss
`14d546d63b069a69041ec4a379d10a06f9b6a7888f25a68dddcadf27293b7f7c`,
FLCE `bd171d8f740c1101e5ff8257973322c20241c3c62b31ae9be16b459f710c28fc`,
stable CE `f5864b5df58832a67e013a75b62f7041c619f4e258670fa689dd9fb38ce272df`.
