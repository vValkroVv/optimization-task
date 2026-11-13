# Overnight streaming and performance campaign

Status: RUNNING. The [accepted campaign](../../planning/overnight-streaming-perf.md)
owns execution progress and the finite G0–G7 acceptance checklist. This report
and `results.json` own measured results; raw payloads remain ignored.

## Entry and environment

Execution started 13 November 2026 02:18 Europe/Moscow (12 November 23:18 UTC)
at clean task `main`, revision `a2a59e14545b55ca777f94a28e332759e7c06087`.
The existing single master exposes eight idle H100 80 GB GPUs. The task path
is on SR008 NFS with approximately 2.5 TiB free; no new allocation was made.
Runtime preflight verified `/opt/torchtitan-venv/bin/python3.12`, torch
`2.15.0.dev20260822+cu130`, CUDA 13.0, NCCL 2.30.7, streaming 0.13.0,
transformers 4.57.1, CUTLASS DSL 4.4.1 and TVM FFI 0.1.11 after sourcing
`/etc/profile.d/wm.sh`. Shared packages remain unchanged.

Perfetto MCP successfully loaded a synthetic Chrome JSON at the actual task
absolute path and executed `SELECT name, ts, dur FROM slice`; it returned
`campaign_route_check`, 1000000 ns, 100000 ns. This verifies tool/path access,
not the later required training-trace analysis. No private server is needed.

## Scope and interpretation

Preserve the fixed crop, precision, SAC, MTP1, EMA, optimizer, seed and scheduler.
MDS is a new data baseline because HF and MDS packing differ. Performance will
include input and monitoring at interval 50, excluding validation and checkpoint
saves. Short runs remain in optimizer LR warmup. No convergence, continuation,
multi-node or production-scale performance conclusion is implied.

## Results

Native MDS smoke retry `g1-streaming-smoke-002` passed at source `4d7689aa`:
ten optimizer updates and 320 successful forward/backward calls on each rank,
finite metrics and native monitoring at 5/10. Final loss was 14.57432 and
gradient norm 7.6163; maximum native composite memory was 72.57 GiB.
The [separate receipt](g1-smoke-002-receipt.json) verifies unchanged source/config,
clean teardown, no diagnostic outputs and released GPUs. No performance
measurements yet. First attempt001 remains a failed harness attempt, not a
candidate verdict. Source applicability is recorded in the
[reference evidence](reference-coverage.md); candidate runtime gates remain open.

The smoke's rank0 host boundary intervals were about 13.5–14.8 s for ordinary
updates and 173.5/166.7 s for monitoring updates5/10. These are unsynchronized
smoke cadence observations, not baseline TPS or CUDA kernel time. They reinforce
the required normal-cadence calibration through update50 before choosing W;
the 32-chunk shape and monitoring work cannot be omitted from the controls.

## MDS fixture and real loader contract

The [data receipt](mds-data.json) owns corpus/tokenizer, installed streaming
implementation, row and shard hashes and exact check commands. The fixture is
896 full 4096-token uint32 rows in 29 uncompressed shards, made from the first
3,670,016 BOS/text/EOS tokens of the same local C4/tokenizer as the HF smoke.
Index SHA256: `7cc8d53152c801450fdae44ba84c182eb8dcb9106d8bba9e3c3d69ebdaa8a8c2`.
The [helper](prepare_mds.py) refuses existing output before calling the destructive
packer. The [loader fragment](mds-dataloader.yaml) selects installed streaming0.13,
two persistent workers/rank, shuffle/FIM off, canonical nodes1, infinite iteration
and no pinning. After native smoke passed, the canonical experiment's data
mapping was promoted, with all non-data values preserved. Its affected golden
test passed (1 passed,60 deselected) after regeneration and again in ordinary
check mode. Relative `local` resolves from the normal `gift/` working directory
to the same index and all29 shard hashes. Native002 used the equivalent absolute
external fragment; canonical bytes have decode/path evidence, without another
GPU smoke.

Two fresh actual eight-rank Gloo runs each validated 58 batches/rank, including
the 56-batch epoch wrap. Both passed exact row hashes/IDs, all16 worker partitions,
global coverage, stable worker PIDs, shapes/dtypes, EOS-derived positions, label
shift and final ignored labels. Each microbatch has 8190 valid main targets;
global per-update numerators are 2,097,152 input and 2,096,640 valid main targets.
No MTP token multiplier is applied. The separate native GPU smoke passed; this
does not establish parity with the differently packed historical HF input.

Pass1 exited0 but retained 16 Python multiprocessing finalizer Errno16 warnings
from the provider's NFS TMPDIR. Pass2 used local tmpfs and had none. The runner
will create and remove only its uniquely owned local temp directory; all data,
compiler caches and persistent logs stay in the task NFS area. No shared temp
cleanup was performed. Final helper, packer, data-world, loader-fragment and index
hashes were checked against the receipt after worker handoff.

Native resolved/source correlation identifies an important position distinction:
the fixed SDPA attention selects causal, non-varlen behavior and propagates that
mask type to GDN. `Trainer.prepare_inputs` replaces the collator's EOS-derived
positions with int32 per-row `arange` for this configuration. Thus the data check
proves the collator contract, while actual model attention/scan state spans
documents within each packed row. It is causal within rows, without EOS document
isolation. Keep this existing model behavior; switching to block-causal/varlen
would change the identified baseline. Exact native per-batch partition counters
are unavailable; the prior actual eight-rank Gloo loader check plus native mesh
and source correlation support rank partitioning.

## G2 calibration

`g2-calibration-001` passed60 native updates/1920 microbatches on every rank,
with exactly32 calls/update, finite loss/gradient logs, clean teardown and released
GPUs. Source/config stayed unchanged; no checkpoint/trace/snapshot was emitted.
Raw evidence is under `artifacts/raw/g2-calibration-001/`; resolved config SHA256
is `b88ffc3da42a7507fa47bb33f6f54fb22d023fa5bc0978338fce6fa9d3e9b27a`.
Helper bytes used by calibration are preserved at `02cfb427`.

Rank0 host boundary cadence medians (seconds):

| Updates | Median |
| --- | ---: |
| 2–10 | 13.7441 |
| 11–20 | 13.8981 |
| 21–30 | 15.3566 |
| 31–40 | 15.3071 |
| 41–49 | 14.9610 |
| 50, monitoring | 173.9149 |
| 51–55 | 14.9717 |
| 56–60 | 15.0141 |

Updates51–60 range14.5722–15.5668s without continuing monotonic settling;
maximum same-update host-interval spread across ranks was1.843ms. Recommend W60,
then measure all updates61–110 including monitor100. This is not TPS evidence:
the calibration has no synchronized50-update window. Step1 took259.828s,
including245.228s in its first forward/backward. Monitor50's first microbatch
took8.477s, then median4.8147s; its173.9s cannot all be assigned to compilation.
Diagnostic compiler logs were disabled, so later compilation is unknown. A
minimal existing-counter boundary observer is being resolved before controls;
monitor100 and surrounding updates must also be checked directly in A1/A2.

## Native profiler controls

Implemented generic opt-in `profile_ranks` and `record_shapes` controls in the
native typed profiler. Defaults remain all ranks and shape capture enabled;
this campaign selects ranks 0/1 and disables shape capture. Nonselected ranks
return before profiler or postprocessor construction. Native schedule, exports
and full-update callsite remain unchanged. With profiling disabled, no new
per-step runtime work is introduced. Stacks and profiler-memory capture are
explicitly false; memory-history snapshots remain a separate disabled feature.

Focused checks from `gift/`, using the existing interpreter and task source:

```bash
python -m pytest tests/unit_tests/test_profiler.py tests/unit_tests/test_config_torch_free.py::test_profiler_rank_and_shape_controls_decode_torch_free -q
python -m pytest tests/unit_tests/test_experiment_parity.py tests/unit_tests/test_model_arch_presets.py -q
```

Results: 21 and 112 passed respectively; scoped F821/F811/F841, default Ruff,
formatting and whitespace checks passed. Goldens were regenerated through the
native update mechanism, then checked without update mode. The CPU native
schedule test exported only update 6, with 32 microbatch markers and optimizer
coverage. It does not establish CUDA tracing. A broader torch-free run retained
one pre-existing registry mismatch (`campaign_numerical_probe` is registered but
missing from the expected callback set); no registry files changed.

The [independent review](profiler-review.md) accepted the source handoff and
verified all 56 golden files contain only the two expected default additions.
Eight-GPU selected-rank trace integrity, numerical observation and lifecycle
acceptance remain open at G3. Full-tree autofixing hooks were not run across
the flattened task tree because their source-root path filters do not protect
its byte-preserved reference and contrib snapshots; changed-file lint was used.

## Native runner review and prelaunch repairs

No GPU run was started before review. The independent reviewer found two
material integration defects: a missing measured-window completion attribute,
and inherited DeepGEMM startup overriding private cache variables with hardcoded
`/temp/.cache/rank_N/deepgemm` paths. The task adapter now redirects the actual
warmup helper before Trainer construction and fails closed on alternate shared
cache paths, retaining native warmup and source/package bytes. Five focused
tests passed, including actual setup/share-path sentinels for ranks0–7 and patch
restoration, full W..W+50/two-sync timing, incomplete-window rejection and smoke
cadence/microbatch counting. The parser passed a synthetic eight-rank global
envelope/counter check; actual measured-window acceptance remains at G2.
The host CPU core limit was unlimited; task child processes set it to zero
and clear inherited diagnostic/core-dump flags. The
[review](runner-review.md) owns findings and their final resolution. These are
prelaunch harness/inherited-path repairs, not failed candidate measurements.

### First native attempt: census failure

The [runtime receipt](g1-runtime-receipt.json) records exit1 on all eight ranks:
`parameter_census` calls `RaggedShardDTensor.to_local()` with grad enabled,
which this wrapper explicitly rejects. Its same method returns the local shard
under no_grad. Zero native updates or microbatch calls occurred; at that failed
attempt, finite loss, monitoring and execution gates were still open. The scoped repair was to run metadata
census under no_grad, retaining original trainability and restoring caller grad
mode, then test that real method guard before retrying.

The correction is implemented and accepted by targeted independent review.
Two affected tests pass, including the actual ragged-shard method guard and
success/error grad-mode restoration. The exact historical census loaded from
Git into memory reproduces the old exception; the fixed implementation counts
the existing local shard. No model arithmetic or training-source bytes changed.
After owned resources were released, the obsolete failed-run compiler cache was
removed (23,453,549 logical file bytes); logs and compact evidence are retained.

Initialization did verify the resolved fixed contract, all-rank FP8/SAC/compile/
GFSDP/streaming/MTP/EMA builds and native DeepGEMM warmup (39 kernels/node),
with all rank caches under the attempt's private tree and child core limits0.
These are initialization observations, not executed CuTe/model kernels or
training correctness. Source/config/helper hashes stayed unchanged; the owned
group stopped, local temp directory was removed, GPUs were released, and no
checkpoint/trace/snapshot was emitted. Logs and compact failure evidence remain.

### Active compiler-cache scope

TorchInductor, Triton, DeepGEMM rank caches and HybridEP use the run's fresh NFS
cache tree. CUTLASS DSL4.4.1 additionally caches generated MLIR under
`CUTE_DSL_CACHE_DIR`, or by default `$TMPDIR/$USER/cutlass_python_cache`
(`cutlass/base_dsl/cache_helpers.py:66–82,235–306`). Root verified that
`CUTE_DSL_CACHE_DIR` and `TVM_FFI_CACHE_DIR` are unset after the image profile;
thus active CuTe IR cache uses each run's unique private tmpfs directory and is
removed with that directory. Its `COMPILED`/`PLANS` maps are process-local.
The active CuTe JIT exports a function pointer to TVM-FFI rather than invoking
`tvm_ffi.cpp.extension`, so that extension builder's persistent default cache
is not the norm path. This qualifies the cache policy without changing kernels
or introducing another experiment. Shared/personal unrelated caches are intact.

### G3 capture adapter: source checks accepted; GPU gate open

The native rank/shape controls and task runner's opt-in target-update mode are
implemented, with native scheduling, actual-call markers and two-trace checks.
The owner reports22 focused checks passed. Independent review's profile-only
parser false acceptance for extra wrong-update/rank markers is resolved by exact
marker-map equality. Both new regressions failed against `02cfb427` and all9
affected parser cases pass on the fix; scoped/default Ruff and compile checks
pass. Root inspected the targeted delta. No unprofiled regression was found.
The actual local `structured_logging.log_trace_span` emits JSONL logging events,
not Kineto `record_function` scopes. Its native `fwd_bwd` events attest calls in
structured logs; manual markers from the profiler CPU test do not establish
native Kineto microbatch/optimizer coverage. The task adapter must add those
scopes around delegated calls on ranks0/1 at the one selected ordinary update,
without wrapping the outer structured step context across its in-scope export.
G1 has exited, and the task helpers are frozen for unprofiled calibration.
Profile timing cannot enter primary TPS.
This is a known remaining integration item, not a passed G3. Profile mode installs
temporary delegated-call counters/checks on all eight ranks; only selected ranks
enter scopes at the one selected update (32 successful microbatch calls, one
train_step and one optimizer call). Unprofiled runs install no new wrappers.
All eight ranks still train, without extra synchronizations.

## Reused validation interfaces and limits

The historical `CampaignNumericalProbe` is not an unchanged validation route for
this campaign: its trainer schema requires `hf_text/c4_test`, zero workers and
synchronous full checkpointing (`gift/torchtitan/config/schema/experiments/cuda_graph_campaign/trainer.py`).
It also persists multiple full tensor families. Do not switch this streaming
campaign to that trainer or generate its full-model payloads merely to reuse
the old probe. Candidate checks must exercise their actual affected loss, state
or lifetime path with matched inputs and elementwise gradients/state, then native
integration; any unresolved full-workload numerical gate stays explicit.

### K1 implementation contract (design only; no candidate enabled)

Keep completed-chunk release and fused CE independently selectable, default off.
The release point is the existing chunk-loop tail after backward/gradient
collection, preserving head calls, argmax, components, probes and flag restoration.
For CE, select a cap-zero callable before `_maybe_compile`, retain current
normalization and weighted MTP, and route DTensor to the current distributed CE.
Do not copy the donor loss class or change Unembedding. The pinned donor's
softcap tests do **not** test fused CE directly, so its CUDA/Inductor autograd
path needs new evidence, including V128256, signed int64 ignored labels,
all-ignored rows, noncontiguous inputs and explicit higher-order rejection.

Release tests must prove output weakrefs expire before the next projection and
compare the actual current chunked A/B paths through gradients and two AdamW
updates (parameters, moments and steps), ordinary/monitoring chunks, MTP,
evaluation/probes, FSDP protocol and exception restoration. For fused CE, first
freeze baseline error/variation against an FP64 CE/analytic-gradient oracle at
the same quantized logits and output cast; compare elementwise downstream
gradients/state through the same projection. Different reduction rounding is
not subject to an invented percentage or unjustified bitwise gate. Any
uncalibrated difference stays unresolved and the feature stays disabled.
Tiny tests do not replace fixed-crop native GPU/SAC/GFSDP integration.

### V3 partial-backward contract (design only)

The isolated donor guard belongs at the start of the group loop in
`GFSDPState._root_post_backward_final_callback`. During nonfinal inner backward,
skip groups outside PRE_BACKWARD/POST_BACKWARD; preserve final cleanup. Current
MTP already folds its hidden states into the chunk loop and sets shared
`is_last_backward=False`, so no loss-class transplant is needed. Prefetch borrows
PRE_BACKWARD temporarily and restores IDLE: an untouched IDLE group can still own
a gathered buffer/pending all-gather. Regression coverage must preserve that
ownership through repeated head chunks until the group's actual backward.

Keep this default-off guard independently selectable from arena/tail retention.
It adds Python state checks per group/inner backward and may remove premature
reshard/regather, while extending existing buffer lifetimes; collision/overlap
safety needs native evidence. Existing explicit EP8 chains, double buffering and
head private gradients are already present. Startup shard slicing is an
applicable initialization optimization with no warmed-runtime effect from the
inspected mechanism; do not credit it as stable-step TPS or silently drop its
initialization-equality gate.

Current loss restores incoming `is_last_backward`/reduce-grads flags, but its
finally hardcodes both reshard policies to true. This inherited limitation is
outside the fixed crop's required policy and must not be described as arbitrary
policy restoration. A guard port must preserve the current policy and verify
normal/exception paths, MTP0.3, chunks2/32, actual owner/event lifetimes and
elementwise losses/gradients/two optimizer updates. Exact callback/state checks
are justified; native FP8/HybridEP floating-point repeatability is not yet
established. Calibrate unchanged numerical A/A before setting native tolerances;
timing A/A and rounded norms do not establish numerical equivalence.

### Native numerical receipt route (design only)

The smallest identified production-path route is a separate task-only numerical
run, with no instrumentation in TPS runs. Wrappers can observe native
`after_dataloader` callback-mutated CPU inputs/labels, `forward_backward_step`
metadata/denominator and loss, project
`dist_utils.clip_grad_norm_` before/after gradients, `OptimizersContainer.step`
after actual AdamW, and callback `after_load`/`batch_end` state. Preserve all
arguments, returns, hooks and wrapper restoration. Adaptive load balancing is
an optimizer-container **pre-hook**: capture incoming counts/bias before clipping
and updated bias/reset counts after step. Its counts are currently FP32,
nonpersistent; `ntokens_seen` is int64. Do not silently change either dtype.

Optimizer param groups own the sharded parameters actually consumed by fused
AdamW; RaggedShardDTensor local access under no_grad returns its exact shard.
Map identities/FQNs/aliases at sharded boundaries and capture logical local
elements without padding, gathering or casting. One-tensor-at-a-time CPU hashes
plus dtype/shape/shard metadata can establish elementwise byte equality only
when unchanged native A/A repeats exactly. Hash mismatches need bounded temporary
payloads and calibrated elementwise error comparisons, not a tolerance inferred
from hashes. Include actual pre/postclip gradients, params, moments/steps,
loss/components, normalization and routing/counter state. The clipping return is
the norm; an independently computed clip coefficient is derived, not an observed
native coefficient. Actual postclip tensors attest the applied result.

The first isolated observer copied every microbatch to CPU, which would serialize
the next microbatch and could mask asynchronous buffer-lifetime faults. The
source-grounded correction hashes CPU batches after `after_dataloader` (all32
are fetched before GPU compute), retains only detached native scalar losses and
one immutable denominator reference during forward/backward, then hashes those
at preclip. Native loss component totals and mutable aux totals/current scale
are sampled once at preclip, before next-update reset. No per-microbatch D2H,
aux clones or graph-bearing references are needed. The native loss provides one
normalized scalar per microbatch; component tests own elementwise CE evidence.
This preserves microbatch overlap but still perturbs update boundaries. Hash
runs alone cannot validate V3 asynchronous ownership: uninstrumented native
integration and the separate representative trace remain required.

All-parameter captures are expensive: five full FP32 parameter-sized families
are about47.5 GB/rank/update of D2H traffic (about760 GB across8 ranks for two
updates), plus initialization/small state. Copies synchronize and can alter
scheduling; these are numerical diagnostics, never TPS. Preserve monitor50 and
capture updates1,2,50 to cover chunks2/32 rather than treating first-two-update
checks as monitoring coverage. A tiny real CPU AdamW test must verify observation
neutrality, exact counts/dtypes/None, mutation detection, and exception cleanup;
it cannot close native repeatability. This design is not implemented or accepted
as production numerical evidence yet.

## Runtime applicability: V4

The native002 parameter census found 243 trainable tensors and 2,375,764,304
local FP32 elements per rank, with zero frozen tensors/elements on every rank.
EMA storage and buffers are not frozen model parameters. V4 frozen BF16 storage
is therefore INAPPLICABLE to this fixed crop; no precision change is made.

## Cleanup

Both native smoke processes have stopped and released the GPUs. Their logs,
resolved configs and compact receipts are retained; no checkpoint, trace or
snapshot was produced. Attempt001's obsolete compiler cache was removed as
recorded above. Attempt002's obsolete compiler cache was removed after smoke
acceptance (604,228,268 logical file bytes); calibration/controls use fresh
per-process caches. Compact native/cache-path evidence remains.
