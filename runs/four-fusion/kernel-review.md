# Independent kernel integration and runtime-cost review

Reviewed 13 November 2026 at task base `33b336a592746340d327d86e34149a23dc8688ee`.
Scope: explicit input/routing kernels, their actual native producers and
consumers, independent activation patches, and the remaining denominator
boundary of the prior loss review. Applied `code-rules` and `perf-experiments`.
No source writes, patch activation, package installation, GPU test or training
was performed by this review.

Finite acceptance: inspect input positions/masks/counts/packing/aux state,
routing storage/autograd/streams, integer widths and arithmetic; reconcile
existing focused evidence; verify patch isolation and preset preservation;
record concrete defects and native performance gates. This is one consolidated
review, not a new loss audit or experiment campaign.

**Final correction disposition:** K1, K2 and K3 are resolved by source inspection
and the owners' focused evidence below. No source-level blocker remains from
this review. Native eight-rank integration, activation, memory and performance
gates remain open. The findings and reviewed costs below retain the initial
snapshot; the correction section supersedes their unresolved status.

## Initial required corrections (resolved below)

### K1: Native input denominator is a Python float

`gift/torchtitan/trainer.py:767` obtains `global_valid_tokens` from
`dist_utils.dist_sum`. `gift/torchtitan/distributed/utils.py:87` delegates to
`_dist_reduce`, whose final `.item()` is returned as a Python `float`. The
accepted dense-FSDP8 workload has DP enabled. The measurement wrapper's
`tracked_token_sum` in `runs/overnight-streaming-perf/measure.py:938` preserves
this result. `input_prepare.py:224` instead reads `.device` directly and the
following conditions require an int64 CUDA scalar. With native CUDA labels,
the input candidate raises `AttributeError` before model forward. Non-DP
training supplies a float32 tensor at `trainer.py:769`, which also cannot
activate the kernel. The successful local input tests supplied synthetic
int64 device denominators and therefore missed this boundary.

Required correction: accept the actual native scalar without changing the
global reduction, its timing, or the agreed workload. Preserve pinned ATen
scalar division lowering; the existing tensor/tensor first division probe
does not establish Python-scalar parity. Avoid per-microbatch H2D constants.
Add a focused regression passing the producer's Python float through real
preparation orchestration, proving kernel execution and exact scale parity;
retain tensor-denominator behavior/fallback explicitly. Repeat only affected
checks. Native input activation is blocked until this is resolved.

### K2: FLCE has the same native denominator mismatch

`linear_ce.py:230` only constructs a denominator when the supplied value is
`None`. The native float reaches `native_linear_ce`, where the R1 device guard
at `linear_ce.py:125` reads `denominator.device`; the underlying custom-op
schema also requires a Tensor. `loss.py` forwards the global value unchanged
through chunk/MTP loss. Thus valid DP8 FLCE training fails despite the recorded
Tensor-denominator MTP/GFSDP tests. CE-only normalization stays outside its
custom op and does not have this new boundary.

Required correction: support the actual scalar producer at the FLCE adapter
without removing R1 target dtype/shape/device checks or introducing unreviewed
per-chunk H2D copies. Verify eager/fullgraph native MTP two-chunk behavior with
a Python float denominator and retain valid Tensor handling. Preserve the
different Python-scalar rounding in reported numerical comparisons. The prior
R1 target-metadata correction itself is present and its 10 rejection plus two
valid MTP tests remain valid for the contracts they exercise.

### K3: The input activation patch cannot apply in the evaluation workspace

Read-only `git apply --check runs/four-fusion/input-prepare.patch` exits 1:
both `gift/torchtitan/experiments/four_fusion/input_prepare.py` and
`gift/tests/unit_tests/test_fusion_input_prepare.py` already exist. The patch
adds these files, whereas the evaluator intentionally keeps helper modules
present and applies the selected patch in place. Its preflight therefore
rejects the input candidate. A check in an empty temporary checkout does not
exercise this actual launch boundary.

Required correction: use an integration-only evaluator patch, as for routing
and loss, or distinguish a standalone distribution patch from that activation
patch explicitly. Preserve working helpers/tests. Verify the actual evaluator
dry run for all variants and the four read-only patch application checks after
the correction. No GPU run is needed to verify this fix.

## Reviewed input contracts and costs

Apart from K1, source matches the intended crop semantics: int64 labels/count,
fresh contiguous bool mask, int32 generated positions, preserved supplied
positions identity/dtype/strides, and contiguous `[B,2,S]` int64 packed targets.
Empty tensors avoid division by sequence zero. Label/position strides and
packed row products widen before address arithmetic; the public wrapper caps
element count at 8192. Larger shapes and CP/full-DTensor/overridden hooks use
baseline orchestration. Packing overrides still call their own implementation.

The mask/packing consume original labels without shift or mutation. Attention
mask creation still precedes aux state updates and model forward. The fresh
share is copied into existing `state.scale` tensors in original order; no state
identity or checkpoint key is replaced. Native MTP shifting/packed boundaries,
aux backward, resets, optimizer, evaluation and loader/RNG ownership remain
in their existing consumers. Moving count/packing earlier than attention-mask
construction does not change these pure tensors, but native traces must measure
the resulting schedule.

At B2/S4096, one CTA on one SM reads 64 KiB labels plus optional supplied
positions, writes 8 KiB mask, optional 32 KiB positions and 128 KiB targets,
and reduces an int64 count. Output scalars are 8-byte count and 4-byte share.
No workspace, atomics, new collective or host scalar extraction is present.
The current implementation's denominator load assumption is invalid on native
DP8; revisit its cost/arithmetic disposition with K1. Existing eager aux
`copy_` calls remain. Single-CTA bandwidth/parallelism can outweigh saved
launches; no speed claim follows from this source review.

Reused evidence: `input-implementation.md` records 14 passed CUDA/CPU checks,
opcheck/outer compile, exact split-3 tensor-denominator rounding, aux identity
and local backward. Their native-denominator gap is K1, not a failed claim
about the tested tensors.

## Reviewed routing contracts and costs

No material source defect was found within the documented distinct, in-range
retained-ID producer contract. Router top-k supplies distinct expert choices;
capacity masking changes dropped IDs to `-1` without creating duplicates.
The kernel widens IDs and all input/output stride products to signed int64.
Wrapper metadata bounds expert vectors, static K, grid X and signed offsets;
unsupported metadata uses the eager reference. No synchronized value check is
added for positive out-of-range or duplicate IDs, which remain outside the
producer contract.

Each token CTA writes every dense bool/FP32 entry; CTA0 initializes the one
trailing sink in each separate backing allocation, including T=0. Selects
preserve valid zero scores, signed zero and NaN while ignoring negative-slot
NaNs. Fake outputs reproduce shape/layout/storage. The custom-op eager body
launches Triton directly and contains no nested compiler-expression helper.
Standalone backward gathers nonuniform dense gradients using original IDs and
casts to score dtype, including base-storage gradients of strided views.
Native HybridEP owns its separate combine-backward gather at
`hybridep.py:727`; the inner helper's autograd is not a second native gather
inside the opaque dispatch forward.

`hybridep.py:103–116,288–340` preserves compute-to-comm waits and stream
recording for routing/probability buffers, communication output joins and
expert-weight quantization overlap. Metadata is produced on the same current
stream as before and recorded before communication consumes it. Dispatch
handles, FP8 row quantization, capacity counts/all-reduce, host predicate and
drop policy remain unchanged. They must remain included in native region
analysis; this patch does not remove their cost.

At T8192/E128/K8 each call writes 1 MiB bool plus 4 MiB FP32, reads 65,536
IDs/scores, and performs 8,388,608 expert comparisons/selects. Dense writes are
required by downstream dispatch. One CTA per token gives useful parallelism,
but unrolled K, broadcasts and register pressure can offset fewer eager
launches. Frequency is per dispatch/layer/microbatch and SAC may replay active
forward work; native trace counts must establish actual frequency. No new
global atomic, `[T,E,K]` allocation, copy, host read, synchronization or
collective is added. Dense metadata lifetimes and communication ownership
are inherited rather than removed.

Reused final routing logs: `cuda-kernel-checks-final.log` records 19 passed,
one skipped, one deselected; `opcheck-and-outer-compile.log` records the
separate opcheck/fullgraph forward-backward pass; the patched HybridEP-body
log records one pass. CPU-only evidence records eight passed and 13 skipped.
The HybridEP-body check uses a recording backend and disables FP8/comm stream;
it proves nesting and values, not real dispatch/combine/SAC/EP8 correctness.

The excluded grad-enabled AOT opcheck failure is reproduced in the retained
failure log and explained by inspected installed
`torch/testing/_internal/optests/aot_autograd.py:157`: `sm += i.sum().abs()`
starts with the bool output's int64 sum then attempts an in-place FP32 add.
Do not edit installed Torch or call this failed subcheck a pass. Full opcheck
on detached inputs, grad registration/schema/fake subchecks, and an explicit
fullgraph weighted-probability backward with exact base-view gradients cover
the intended local compiled-gradient contract without that broken reduction.

## Isolation, disposition and remaining gates

The other three `git apply --check` commands pass. Routing's activation patch
changes only the HybridEP import; loss patches each select their own alternative
loss path. The evaluator rejects dirty core integration targets, holds its
lock, applies exactly one patch, and reverses only that patch after confirmed
owned-group exit. Helper files being present do not activate other candidates.
The input patch packaging is the exception K3. The runner retains prefetch32,
epoch51200, accumulation32, FP8, model/loss compile, SAC, MTP1, shapes and
schedule; no reviewed patch changes those defaults or introduces persisted
state. Record actual resolved configuration and kernel activation for every
native run.

At review snapshot, input helper SHA256 was
`a10ff058594f9687b523433834d597088ef2306b11c5b87cb96d21692f2fb2fb` and
routing helper SHA256 was
`2fb70d05edb774b09af93f0e97bd8054a75cd13f741e6071a82bf07e827c2f53`.
Implementation owners were still finalizing records/patches; corrections must
update provenance and obtain targeted verification before native acceptance.

Native eight-rank activation, real FP8 HybridEP combine/backward with SAC,
full-shape memory, independent 10+10 unprofiled comparisons, matched bounded
Perfetto traces and retained-candidate local replay remain open. Separate
device work, host dispatch, idle/waits, copies/communication, target elapsed
time and TPS/phase-matched memory. Local test success does not establish any
production performance or acceptance verdict. K1–K3 were sent to the
coordinator and resolved with the targeted work below, without a broad audit.

## Targeted correction verification

Only K1/K2/K3 were revisited after source freeze. This verification read the
changed source, regression cases, retained final logs and correction records;
it ran no tests, GPU operations, training or patch applications.

**K1 resolved.** `_supports_explicit_kernel` now branches on Python float
before reading Tensor metadata, accepts the native positive finite host count,
and separately accepts the non-DP CUDA FP32 Tensor producer. The scalar custom
op passes a host-double-computed reciprocal as a runtime FP32 launch argument.
The input kernel performs local-count FP32 multiplication by that reciprocal,
then multiplication by the FP32 split reciprocal with fusion disabled. The
Tensor path retains `tl.div_rn` followed by the split reciprocal multiply.
This matches the distinct pinned ATen scalar and Tensor lowering; neither
producer nor global reduction moved. No denominator Tensor/H2D copy, new
collective or device-value read was introduced. The scalar input prefix runs
in eager Trainer orchestration, so its direct outer-compile probe does not
imply native per-count loss compilation.

The [input record](input-implementation.md#verification) reports 16 passes and
retains the corrected native-denominator probes. Inspected new tests exercise
real `prepare_forward_batch` orchestration with the actual DP producer's
Python-float type/value, explicit scalar-op activation, B2/S4096 packing and
bitwise `AuxLossTracker.weigh` state; a separate case proves the non-DP FP32
Tensor arm. Split-3 count/denominator parity and prior state/position tests
remain covered. The input regression supplies the known producer value
directly; its scope is preparation, not a newly executed DP8 all-reduce.

**K2 resolved.** The native adapter retains R1 target int64/shape/device guards
and restricts the denominator device check to Tensor denominators. Its opaque
op now accepts an optional Tensor plus host `Scalar`. `LinearCE._host_count`
bridges an integral-valued host float to signed-int64 only when an exact float
round trip succeeds; the opaque body restores the original double value.
Fractional/nonfinite/out-of-range values use baseline logits/CE. Native counts
are positive and at most 2,097,152 for this workload. Local `dynamic=True`
allows the integral count to remain symbolic; no global compiler setting or
required workload feature changed.

The FLCE host arm computes the double reciprocal in the opaque body, passes it
as runtime FP32 scale, and multiplies the FP32 head weight before the centered
gradient and final BF16 cast. The Tensor arm uses `tl.div_rn` for the weighted
scale. Loss terms retain SUM/divisor/head-weight order. The shared row kernel's
new branch is selected only by FLCE; CE-only's default incoming-scale path is
unchanged. Fake/autograd signatures match the additional scalar argument, and
backward still returns one entry for each of the six inputs. No new device
allocation, transfer, collective or persisted state is introduced. Bounded
host type/range/round-trip work and dynamic dispatch are added per selected
outer chunk; the earlier GEMM/allocation/lifetime costs remain as reviewed.

The [correction record](loss-native-denominator-fix.md) retains the original
two native-float failures and final 26-pass affected subset (eight unchanged
CE-only tests deselected). Inspected final logs confirm changing producer
counts compile callbacks `[1, 1, 1]`, plus eager/compiled world-size-one actual
`dist_sum(..., mesh=mesh)` and GFSDP accumulation32/two-chunk runs with 64 FLCE
calls each. Recorded maximum dH/dW differences are
`5.820766091346741e-11` / `2.9103830456733704e-11`. These close the native
scalar boundary and local gradient/compile regression; they do not establish
eight-rank communication or full-shape memory. Core loss SHA256 remains the
preserved `14d546d63b069a69041ec4a379d10a06f9b6a7888f25a68dddcadf27293b7f7c`.

**K3 resolved.** The current input patch contains only the Trainer integration
hunk, with no added helper/test files. The input owner records a successful
actual-workspace `git apply --check --whitespace=error` and clean restored
Trainer. The coordinator retains all-variant evaluator checks; the specific
existing-file collision is removed without deleting helper files. The review
did not apply or reverse any patch after source freeze.

Verified current SHA256 values: input helper
`db35fc56ff947e486bd35b9fb9afc432a06ea1dd2d0edbfa10dcfc9211f2666f`,
FLCE helper `32eea5faeaaff4e4ec0c7a16b825e4807be8fd6481123fc3cc5de01722d5e80b`,
and input activation patch
`e0eea03e83e7ea88bdccd90b9c4a3927727753057df89a4843d072245602f7a0`.

All three material findings are resolved for their scoped source/integration
contracts. The remaining native eight-rank, FP8/SAC, activation trace,
full-shape memory and independent performance/trace gates above are unchanged;
no candidate is accepted for speed or production readiness by this review.
