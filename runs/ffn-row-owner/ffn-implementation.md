# FFN overlap implementation

## Finite scope and acceptance

Implement independent default-off `gfsdp_split_layer0_ffn` and
`gfsdp_split_mtp_ffn` controls in the task snapshot. Preserve existing parameter
identity, FQNs, FP32 master/reduction/accumulation, BF16 boundaries, FP8 modules,
residuals, MTP hidden/embedding input order, and the resolved SAC policy. Verify
local output/input/parameter-gradient parity through real checkpointed compiled
regions, original/control behavior, buffer ownership and prefetch ordering. Keep
GPU/NCCL/FP8 schedule, component performance and DCP continuation gates open until
executed. The assignment prohibits GPU execution, installation, commit, or
publication in this phase; the unintended GPU test executions below violated
that execution boundary and are disclosed explicitly.

## Preimplementation design and runtime-cost review

Source baseline: root `2ada87125e433e1f76c4cb2bfad244b641a78a93`.
Local PyTorch: `2.15.0.dev20260822+cu130`, existing interpreter
`/opt/torchtitan-venv/bin/python`. References inspected: existing
`apply_ac`, `_apply_ac_to_block`, `_sac_context_fn`, `apply_compile`,
`GFSDPState._init_shared_state`/`validate_shared_state`,
`GFSDPParamGroup._register_post_backward_hook`/`post_backward`,
`GFSDPModule.set_own_grad_buffer`, and the current GIFT parallelization path.
These pinned local APIs supply the established implementation; no new dependency.

Each selected original parent stays an eager GFSDP orchestrator over prefix,
FFN and suffix compute regions. Prefix ends after the attention residual and
FFN norm; suffix retains postnorm, residual scaling/add and (MTP) final norm.
The original FFN module owns w1/w2/w3 together, is wrapped before its parent,
and keeps its identity and FQN. Checkpoint calls are inside each compiled
region, with the original parent FQN and resolved save categories, RNG,
early-stop, determinism and debug settings. There is no eager outer SAC wrapper
around an opaque compiled region. MTP prefix retains [hidden, embedding] concat.

Nested parent+FFN data lifetimes require private parent data contexts; private
parent gradient contexts avoid waiting for the early FFN reduction before the
parent backward can proceed. Ordinary units keep alternating shared data
contexts and a shared grad context. Root finalization already waits for each
post-reduce event. Forward enters parent then FFN; backward enters parent on its
suffix gradient before FFN, so nested units cannot use blind flat reversal for
prefetch.

Expected costs per microbatch: one added dense FFN RS group/startup, changed flat
packing/padding, Python/autograd and compilation boundaries, possible lost
cross-boundary fusion, and extra live BF16 boundary tensors (64 MiB each at
B2/S4096/D4096). Private parent data/FP32 gradient buffers add residency; gather
and prefetch volumes/lifetimes may change. No new dtype conversion or explicit
host/device synchronization is intended. Collective launch readiness depends on
FFN input-gradient and weight-gradient completion via the existing GFSDP input
post-backward node; actual all-rank overlap requires GPU traces. Changed flat
partitioning makes baseline DCP model/optimizer/EMA restoration and the next
update a required gate. No speed claim follows from source or CPU evidence.

## Implemented interfaces and evidence

Both flags live in `gift/torchtitan/config/configs.py`; both default false.
`parallelize_gift_moe` enables `unit._ffn_overlap_regions` before applying AC
and compile. `apply_gfsdp` needs no additional arguments: it detects the region
tuple, wraps the original dense FFN first, then the remainder, and requests
private parent data/gradient contexts. Its `units` list is the top-level entry
order; each selected unit expands to `[parent, FFN]` both forward and backward,
with only top-level units reversed for backward. This is the integration point
for subsequent row-owner stem exclusion. Shared config/parallelize ownership is
released after this handoff.

`enable_ffn_overlap` keeps region holders outside the registered module tree;
`EagerWindow` explicitly discovers them. AC stores a bound checkpoint forward on
each region, preserving original resolved save categories, forced-mm FQNs,
preserve-RNG, early-stop, debug and determinism settings inside compilation.
`mm_half` is explicitly rejected for selected regions because its original
whole-block ordinal policy cannot restart independently per region. Default
blocks retain their existing AC wrappers and compile slots.

Setup activation observables (enabled only):

- `FFN overlap compute regions active: layers.0` / `mtp.layers.0`, followed by
  `(compiled prefix/FFN/suffix; eager parent)`.
- `GFSDP FFN overlap active: layers.0.feed_forward` /
  `mtp.layers.0.block.feed_forward`, followed by `(w1/w2/w3 together); remainder
  uses private data/grad contexts`.

### Executed local checks

Interpreter `/opt/torchtitan-venv/bin/python`, PyTorch
`2.15.0.dev20260822+cu130`. **Correction:** automatic DTensor device selection
used CUDA here (`torch.cuda.is_available() == True`), not CPU/Gloo as initially
assumed. The actual GFSDP test used two CUDA/NCCL ranks. The initial existing
sharding suite used its eight CUDA/NCCL ranks, and the targeted existing
activation-checkpoint suite included its CUDA memory test. These were unintended
GPU executions outside this worker's no-GPU assignment. No native crop training
or performance benchmark ran. The coordinator was notified immediately after
the device-selection mistake was verified. The standalone local block tensors
were explicitly CPU tensors. Seeds 2048 (parameters) and 87 (local block inputs);
small GQA+dense blocks D64, FFN128, B1/S3. These are focused behavioral probes,
not the production FP8/GDN/MLA crop.

Use `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=gift` for these commands.
Inductor commands originally set `TORCHINDUCTOR_CACHE_DIR` to
`runs/ffn-row-owner/inductor-cpu-cache` (absolute task path). The completed cache
and raw logs now live under ignored `runs/ffn-row-owner/artifacts/raw/`.

- `python -m pytest -q gift/tests/unit_tests/test_ffn_overlap.py
  --disable-warnings --tb=short`: **19 passed**. Includes CPU Inductor
  fullgraph+SAC output, each input gradient (both MTP inputs), every parameter
  gradient, AOT backward-graph matmul recomputation evidence for selective
  versus saved-mm policy, no registered module/parameter/FQN replacement,
  EagerWindow restoration, private/shared data mapping and all four independent
  wrapping/prefetch switch combinations. Float64 comparisons use 1e-12 for this
  small CPU arithmetic; this is not a production tolerance.
- `python -m pytest -q gift/tests/unit_tests/test_gfsdp_ffn_overlap.py
  --disable-warnings --tb=short`: **1 passed** on two actual GFSDP CUDA/NCCL ranks.
  Baseline checkpoint loads into changed split flat layouts with exact logical
  model, fused-AdamW state and EMA values; the next update after 32
  microbatches matches all FP32 gradient elements, weights, optimizer state
  and EMA exactly under identical BF16 AOT eager primitives. The final run
  additionally confirms all selected FFN wgrads are populated before the real
  post-backward call and FFN-before-parent post-backward order exactly once per
  microbatch, with no SAC duplicate reductions. Parent/child data and grad
  contexts are distinct. This is CUDA/NCCL readiness/order evidence, not traced proof of
  asynchronous communication/computation overlap.
- Initial affected model/wrapping suite: **109 passed** (`test_mtp.py`,
  `test_residual_scale.py`, `test_gfsdp_buffer_order.py`,
  `test_gfsdp_moe_sharding.py`, `test_compile_cudagraph_exclusions.py`).
- Targeted SAC/DCP/buffer suite: **62 passed** (`test_sac_categories.py`,
  `test_activation_checkpoint.py`, `test_compile_cudagraph_exclusions.py`,
  `test_gfsdp_dcp_param_shapes.py`, `test_gfsdp_dcp_fqn.py`,
  `test_gfsdp_buffer_order.py`). Raw retained results:
  `artifacts/raw/ffn-cpu-local.log`, `ffn-gfsdp-local.log`,
  `ffn-affected-local.log`.
- Scoped Ruff `F821,F811,F841` passes on all 11 changed/new Python files;
  `git diff --check` passes. New test files formatted with Ruff. Full
  pre-commit was not run because it can install hooks/change unrelated files.

Retained failed attempts: the first two multi-policy/backend test-matrix runs
exhausted Dynamo's shared per-code cache (four failed, nine passed). After two
attempts the harness was corrected to reset compiler state between independent
policy/backend cases; production cache limits were not raised. A bound method
replaces the initial checkpoint closure. The first actual-GFSDP test selected
for-loop AdamW, which this checkout's RaggedShardDTensor does not implement
(`aten.lerp_.Scalar`); selecting the required native fused AdamW path passed.
The first formatting check requested formatting of the two new test files;
Ruff formatted them. None of these failed checks is counted as a pass.

### Final runtime-cost review

Meta construction of the actual crop, with only HybridEP dispatcher startup
suppressed, gives the counts below. FFN w1/w2/w3 has 134,479,872 elements for
each selected unit. Remainder counts are divisible by the GFSDP8 alignment of
64 elements (8 ranks x shard_alignment 8), so these predicted private sizes
need no extra padding for the current ordinary BF16 gather bucket.

| Remainder | Elements | Private BF16 data MiB | Private FP32 grad MiB |
| --- | ---: | ---: | ---: |
| layers.0 | 67,944,640 | 129.594116 | 259.188232 |
| mtp.layers.0 | 99,791,360 | 190.336914 | 380.673828 |
| Both | 167,736,000 | 319.931030 | 639.862061 |

The predicted added parent contexts total **959.793091 MiB/rank**. The dense
shared maxima remain controlled by input/output tables: each table's
525,336,576 elements needs 1002 MiB BF16 data and 2004 MiB FP32 gradients.
Replacing each private parent in the ordinary alternating list with its FFN
preserves ordinary parity; both shared data contexts still contain one table,
and the ordinary shared grad maximum is unchanged. Existing expert contexts
are unchanged. These are source-derived sizes, not measured allocator peaks.
The setup's actual dtype buckets/padding, auto-reshard retention and allocator
peaks must be recorded on the native FP8 crop. Extra BF16 boundary tensors can
cost 64 MiB each; changed fusion/compilation and extra RS startup/packing remain
performance gates. No new explicit host reads, device synchronizations or
changes to divide-factor1/FP32 accumulation were introduced.

### Remaining gates and next execution route

The required independent combined integration/runtime-cost review is owned by
the coordinator after row-owner integration. Current production flags retain
all accepted baseline features and FlashQLA external_gate. Do not infer
production acceptance from these focused results. The existing distributed test
currently auto-selects CUDA when available; its invocation requires coordinated
GPU authorization. A CPU-only expectation must explicitly force or guard the
device rather than rely on the machine role.

1. On the authorized eight-H100 allocation, use the existing native campaign
   runner with each FFN flag independently and both together while preserving
   FP8, exact per-layer SAC, model/loss compile, MTP1 and all32 microbatches.
   Verify both setup marker pairs before interpreting results. The common
   source integration is in `parallelize_gift_moe`, so CLI/YAML overrides feed
   the actual GFSDP compiled/SAC production path.
2. A separate bounded rank0/1 capture must verify the actual
   `GFSDP::post_backward_reduce` ranges labeled
   `layers.0.feed_forward` and `mtp.layers.0.block.feed_forward` start after
   their dInput/wgrad completion and before corresponding remainder backward
   ends. Verify collective sequence on all eight ranks and inspect buffer
   release-event waits, gather prefetch, RS packing/launches and remaining
   backward in MCP Perfetto. Compare against the current external_gate trace,
   not historical FLA estimates. Component and native timing remain unrun.
3. For a portable focused CUDA/NCCL continuation check, run
   `PYTHONPATH=. python -m pytest -q
   tests/unit_tests/test_gfsdp_ffn_overlap.py` from `gift/` under the authorized
   CUDA environment (DTensorTestBase selects its available test device). This
   is two-rank BF16/AOT GQA coverage; the actual native eight-rank
   FP8/Inductor/SAC/GDN/MLA continuation and local block numerical comparisons
   remain required. These focused tests cannot substitute for those.
4. Coordinate config golden updates after all three controls land. Existing
   experiment snapshots already omit the prior
   `gfsdp_reuse_bf16_expert_weights` field, so a broad snapshot mismatch is not
   wholly attributable to this change. `gift/tests/unit_tests/config_golden.py`
   documents `TORCHTITAN_UPDATE_GOLDEN=1` with `test_experiment_parity.py` and
   `test_model_arch_presets.py`; review a structural diff rather than blindly
   refreshing unrelated drift.

## Execution-boundary correction and retained timestamps

The erroneous assumption was that local notebook-server tests would select
CPU. Device selection was not verified before invoking existing DTensor and
CUDA-aware tests. No further test was run after discovery and the coordinator's
stop instruction. Exact start/end wall timestamps were not instrumented for
these pytest commands. Do not treat duration-derived estimates as exact times.
The actual owner's grant timestamp is not available in this worker context,
so these tests cannot honestly be classified against that timestamp.

| Command / observed execution | Result | Timing actually retained |
| --- | --- | --- |
| `PYTHONPATH=gift python -m pytest -q gift/tests/unit_tests/test_mtp.py gift/tests/unit_tests/test_residual_scale.py gift/tests/unit_tests/test_gfsdp_buffer_order.py gift/tests/unit_tests/test_gfsdp_moe_sharding.py gift/tests/unit_tests/test_compile_cudagraph_exclusions.py --disable-warnings` | 109 passed; includes five eight-rank CUDA sharding tests | Duration77.33s; exact start/end not recorded; launched and observed complete before this worker received the coordinator's owner-grant message. Output retained in tool session39901, not a file. |
| `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=gift python -m pytest -q gift/tests/unit_tests/test_sac_categories.py gift/tests/unit_tests/test_activation_checkpoint.py gift/tests/unit_tests/test_compile_cudagraph_exclusions.py gift/tests/unit_tests/test_gfsdp_dcp_param_shapes.py gift/tests/unit_tests/test_gfsdp_dcp_fqn.py gift/tests/unit_tests/test_gfsdp_buffer_order.py --disable-warnings --tb=short` | 62 passed; includes CUDA `TestApplyAC.test_mem` | Duration6.73s; retained log final write2026-11-14T20:02:34.220579Z; exact start/end not recorded. Log `artifacts/raw/ffn-affected-local.log`; tool session94674. |
| `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=gift python -m pytest -q gift/tests/unit_tests/test_gfsdp_ffn_overlap.py --disable-warnings --tb=short`, initial for-loop optimizer attempt | 1 failed; both rank processes reported exit10 on unsupported `aten.lerp_.Scalar` | Error timestamp2026-11-14 20:02:34.699Z; duration18.58s. Exact start/end not recorded. Tool session20229. Its log was overwritten by the next invocation; error and cause retained here and in tool output. Launched/observed before worker received owner-grant notice. |
| Same GFSDP command after selecting fused AdamW | 1 passed | Duration19.63s; exact start/end not recorded; tool session30954. Log overwritten by final invocation. Launched before worker received owner-grant notice; completion was observed after that notice. |
| Same GFSDP command after adding wgrad readiness/order assertions | 1 passed | Duration19.91s; retained log final write2026-11-14T20:03:55.779716Z; exact start/end not recorded. Tool session19085; log `artifacts/raw/ffn-gfsdp-local.log`. Launched after worker received owner-grant notice, while its explicit no-GPU assignment still applied. |

The local CPU block tests have retained log final
write2026-11-14T20:00:00.215764Z (`artifacts/raw/ffn-cpu-local.log`).
Tool sessions39901,94674,20229,30954,19085 all returned completed process
exit codes (0,0,1,0,0 respectively). A subsequent read-only `/proc/*/cmdline`
scan matching the exact test-file argv entries for `test_gfsdp_ffn_overlap.py`
and `test_gfsdp_moe_sharding.py` found no matching test process. This is scoped
process evidence, not a claim that all allocation GPUs are idle. No cleanup,
process termination or broad GPU probe was performed. The coordinator owns
the next all-node ownership/availability preflight.
