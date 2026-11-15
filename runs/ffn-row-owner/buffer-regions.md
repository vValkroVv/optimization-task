# FFN regions in existing GFSDP storage

## Acceptance and preimplementation review

Replace the rejected private-parent design with contiguous parent/FFN views in
existing buffers. Keep w1/w2/w3 together, original alternating weight slots,
dtype buckets, FP32 reduction/accumulation, compile/SAC, prefetch and early FFN
reduction. Check actual aliasing, alignment/rank layout, overlapping ownership,
async completion, numerical/continuation behavior and unique GPU storage/phase
peaks. No pool growth or extra staging. No change to accepted baseline flags.

Source base: `2ada87125e433e1f76c4cb2bfad244b641a78a93` plus current task changes.
The existing `GFSDPBufferContext.lazy_init` allocates raw byte tensors; each
`_init_unsharded_params` formerly narrowed from byte zero. `_init_shared_state`
deals two alternating weight contexts per process group, one ordinary gradient
context, and the required separate lm_head gradient context. The private-parent
API and integration introduced for FFN overlap are superseded by this revision.

A parent remains live through its child forward/backward. The child's early
reduce-scatter reads its gradient region while the parent may copy/accumulate
into a different region. Later ordinary groups reuse the whole slot and must
wait for *both* regions' completion. Context identities alone cannot express
this aliasing; an interval-overlap registry checks live owners and retains
per-region release events for all overlapping future streams.

Each parent+child is dealt as its original unit. Each dtype slot is sized using
the original merged parameter count and original rank/alignment padding.
Parent starts at zero; child starts after the parent's padded byte range. An
explicit budget check rejects excess split padding instead of adding storage.
Current crop counts previously inspected are alignment-divisible. The existing
head gradient context, FP8 parameter classes and compute paths are unchanged.
Allocation/views stay in lazy initialization, outside compiled compute.

Expected cost: region ownership queries inspect at most the whole-slot context
and the two region pairs (layer0/MTP); event waits can retain completed events,
with no host synchronization or additional tensor copies. No new compute or
communication; flat GFSDP grouping and extra collective count remain those of
the accepted FFN boundary design. Full-slot users must wait both async readers,
while non-overlapping parent/FFN contexts do not wait on each other. Additional
activation/compile-boundary memory remains separate from pool byte parity.

## Verification and observed memory

Native test command (repository root, 2026-11-14 UTC):

```sh
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=gift \
  /opt/torchtitan-venv/bin/python -m pytest -q -s \
  gift/tests/unit_tests/test_gfsdp_ffn_overlap.py --disable-warnings --tb=short \
  > runs/ffn-row-owner/buffer-regions-cuda-nccl.log 2>&1
```

**3 passed in 33.20 s**, two actual H100 CUDA/NCCL ranks. Pytest parent PID
226232; test children 226237/226238, 226338/226339 and 226397/226398 exited
successfully. All eight H100s were idle with zero process memory before and
after; no compute applications remained. Source checksum verification passed
for all 12 entries in [the source manifest](buffer-regions-source.sha256),
including untracked FFN and row-owner modules. No source changed during ranks.

1. Actual GFSDP all-gather restores each split group's original logical BF16
   values. Actual post-backward reduce-scatter maps SUM contributions to the
   correct local rank's output region and padded/ragged parameter slices.
   Tests assert contiguous region offsets, non-overlap, alignment, per-rank
   input/output sizes, and reduce-scatter output address equal to the existing
   rank chunk of that group's input. No collective packing/index logic changed.
2. Two disjoint regions have actual asynchronous NCCL collectives and reads
   pending on separate CUDA streams. Whole-slot overwrite waits for both region
   release events; both snapshots preserve the exact expected sum. This is
   executable pending-reader lifetime evidence, not a fake event test.
3. The existing BF16 AOTAutograd/SAC 32-microbatch DCP regression passes with
   new regions. Baseline model/fused AdamW/EMA restore exactly into split flat
   layouts; the next update's FP32 parameter gradients, parameters, optimizer
   state and EMA match exactly. FFN post-backward occurs once before parent
   post-backward per selected block per microbatch, with every FFN wgrad ready.
   CPU Inductor input-gradient/parameter-gradient parity from the earlier FFN
   implementation remains applicable because compute/checkpoint code is
   unchanged; only the allocation/view lifetime layer changed here.

The split design already changes flat shard ownership compared with the unsplit
control; this storage revision preserves that existing split layout, ragged
metadata and collective input/output mapping. DCP logical state compatibility
and the next update are tested directly; identical FQNs alone are not the proof.

Actual unique pool storage per rank, deduplicated using untyped storage pointer
and byte size (aliased views counted once):

| Pool | Control bytes | Split bytes |
| --- | ---: | ---: |
| Alternating data slot 0 | 82,176 | 82,176 |
| Alternating data slot 1 | 98,944 | 98,944 |
| Required separate head grad | 4,096 | 4,096 |
| Ordinary grad | 197,888 | 197,888 |
| Total | **383,104** | **383,104** |

The tiny test's warm 32-microbatch forward/backward incremental allocated peaks
were 929,792 bytes control and 909,824 bytes split on both ranks; reserved peak
was 75,497,472 bytes for both. Absolute rank-0 allocated peaks were 73,973,760
and 74,491,392 bytes respectively, because the DCP test retains both model roots,
optimizer state and expected gradient exports. These are scoped test readings,
**not isolated activation-peak or full-crop memory parity evidence**. Startup
control measured 68,038,656 additional bytes including initialization/compile
caches and cannot be compared to the warm candidate. The native FP8 factory
owns complete buffer/staging inventory, lifecycle phase peaks and bounded
Perfetto overlap evidence before native training acceptance.

Other targeted checks:

- `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=gift
  /opt/torchtitan-venv/bin/python -m pytest -q
  gift/tests/unit_tests/test_ffn_overlap.py -k 'regions_alias or nested_wrapping'
  --disable-warnings --tb=short`: **5 passed**, including mixed dtype byte views,
  whole-slot/region ownership conflict detection and all independent FFN flag
  wrapping/prefetch combinations.
- Same CPU environment with `test_gfsdp_row_owner_integration.py` and
  `test_gfsdp_buffer_order.py`: **32 passed**; retained log
  [buffer-regions-integration-cpu.log](buffer-regions-integration-cpu.log).
- Scoped Ruff `F821,F811,F841` and `git diff --check`: pass.

Retained failed/skipped checks:

- First CPU/Gloo DCP attempt fails in the unsplit baseline because Gloo cannot
  execute native NCCL `PREMUL_SUM`; log `buffer-regions-cpu-gloo.log`.
- Test-only SUM with factor 1 resolves that backend gap, then the unsplit baseline
  fails autograd version checking because Gloo writes increment the shared
  CPU buffer version. Log `buffer-regions-cpu-gloo-2.log`; its CUDA-only pending
  collective test is skipped. After two failures the native CUDA route became
  the numerical decision gate. No preserve-version adapter or production
  workaround was added; CPU backward remains an unsupported evidence route.
- A separate CPU construction/rank-layout probe initially failed a test-helper
  `RaggedShardDTensor.to_local()` no-grad precondition. `@torch.no_grad()` fixes
  the helper, and the actual CUDA rank-layout test passes. The failed record is
  `buffer-regions-layout-cpu.log`; no third CPU numeric retry was made.
- Native tests emitted vendor quack disk-cache load warnings and fell back to
  in-process compilation. No cache/package/environment mutation was performed.

## Final cost review and remaining gates

The production diff adds no tensor allocation to runtime compute and no staging
copies. Region views are created once during root lazy initialization. Original
logical-unit ordering preserves two alternating data slots; root/region overlap
validation prevents simultaneous writers. Ownership/release checks add bounded
Python interval checks and stream waits; they do not read CUDA values or host-
synchronize. BF16/FP8 buckets, FP32 accumulation, optimizer/EMA and DCP continue
through their existing consumers. The removed private-parent data/grad requests
and own-data API add no replacement pool. `set_own_grad_buffer` for lm_head is
unchanged. Explicit custom buffer maps/YCCL are rejected for the opt-in split
region API; the agreed NCCL crop uses neither.

The native production crop still requires unique buffer/staging storage
inventory, separate activation/compiled-boundary phase peaks, FP8/SAC component
measurements and at-most-three-microbatch-per-rank Perfetto confirmation of
both early reductions overlapping remainder work. Pool parity in this tiny
regression does not close those gates. Independent integration review remains
with the coordinator's assigned reviewer. No defaults, commits, publication or
native training run occurred in this bounded buffer assignment.
