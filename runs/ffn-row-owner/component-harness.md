# Embedding component harness

Preparation completed 14 November 2026. No GPU work was executed by this
assignment. The coordinator serializes execution after the combined source
freeze. This document owns the harness interface and checks; campaign results
remain in [report.md](report.md).

## Finite preparation acceptance

- [ ] Actual GFSDP baseline and actual `RowOwnerEmbedding` with equivalent
  deterministic FP32 initialization, signed int64 IDs, BF16 derivatives and
  completed dense FP32 accumulated shard gradients. Both roots include the
  same real untied FP32 `[V,D]` `lm_head` GFSDP unit and its private gradient
  buffer. Head-buffer allocation, scratch aliasing and matching forward passed;
  the CPU/Gloo harness could not exercise backward (details below).
- [x] Eight-rank H100/NCCL execution guard, V128256/D4096/B2/S4096, existing local
  MDS fixture plus duplicate, owner-skew, empty-requester, empty-all and boundary
  cases. No downloads, capacity calibration, table caching or config mutations.
- [x] Observer-off complete F+B timing; separate F/B diagnostic timing;
  stage contexts on the actual active stream without per-stage barriers.
  Explicit stream joins precede the completion endpoint.
- [x] Per-rank samples/distributions and slowest-rank distributions; allocated
  and reserved phase peaks; request/count/host-read census and input hashes.
- [x] GPU-hidden plan/parser, actual tiny GFSDP/owner construction and real MDS
  checks; scoped lint.
- [ ] Execute the GPU commands below and interpret the resulting table before
  native timing. Preparation does not close this gate.

## Actual paths and measurement scopes

[`components.py`](components.py) imports torch only when inputs or execution
need it. Running without `--execute` prints a torch-free JSON plan. The runnable
GPU route uses one NCCL group containing all eight ranks and does not alter the
training schedule or compile/SAC configuration. The embedding stem is already
eager outside the compiled/SAC blocks; this harness measures that stem directly.

`build_embedding` creates the actual `nn.Embedding` GFSDP wrapper with BF16
parameter gather, FP32 gradient reduction and division factor 1. Both variants
also create an untied `nn.Linear(width, vocab, bias=False)` named `lm_head`,
with an exact `[V,D]` FP32 weight, wrap it with GFSDP and call the unchanged
`set_own_grad_buffer()` API. The head's forward is not run; its private buffer
is still allocated and resident in both variants. The candidate uses
`RowOwnerEmbedding.from_embedding`, excludes its FP32 parameter from the root
and attaches the real GFSDP communication context. Its scratch view must alias
the existing private head gradient storage. The root has no other compute, and
no toy communication implementation replaces either path.

**Lifetime correction:** a standalone stem would become the last GFSDP buffer
user, causing root auto mode to retain its full table and omit the baseline's
backward all-gather. The harness explicitly sets root resharding and asserts the
embedding group's `post_forward_mesh_info` is present. This reproduces the
nonfinal crop embedding stem's gather/reshard lifecycle. It does not simulate
contention with intervening trunk work; native diagnostics own that interaction.

Every timed sample starts with an existing, zeroed FP32 gradient shard, allocated
by an untimed actual forward/backward warmup. The resident private `lm_head`
buffer and all other initialized GFSDP gradient storage are also present at
sample start. `gradient_buffer_inventory` deduplicates physical backing storage
when GFSDP regions expose aliased views. Gradient clearing and the between-
sample rank barrier are excluded. The full forward/backward path and its
gradient accumulation are included. No optimizer update belongs to the stem
measurement. No per-stage barrier or synchronization is inserted.

- `--scope integrated` measures one complete forward+backward call with the
  observer absent. It has no forward/backward midpoint synchronization.
- `--scope phases` measures F and B separately, with a synchronization boundary
  between them. Those spans are diagnostic and must not be summed to claim
  integrated latency. Forward setup remains live for backward.
- `--scope stages --variant owner` instruments the actual owner stages in a
  separate run. `StageObserver` records CUDA events on each stage's current
  stream, host context spans and `components::<name>` profiler ranges. Collection
  occurs only after the complete operation has finished. These are stage spans
  in their actual path, including submission gaps/waits, rather than standalone
  kernel-only microbenchmarks. No isolated-stage sum is an end-to-end estimate.
- Optional `--profile` is accepted only in diagnostic scopes. It captures one
  additional complete F+B per case on **every rank** to Chrome trace JSON.
  Profiling is outside all reported unprofiled sample windows. Limit `--cases`
  to the concrete question being profiled.

CUDA event spans start on the compute stream. The communication stream waits
for that start; after the call, the compute stream joins recorded completion
events from GFSDP's all-gather and reduce-scatter streams before recording its
endpoint. Host call spans include internal synchronization such as dynamic
count reads. Endpoint host synchronization is reported separately. These spans
include host starvation/device gaps and are not pure kernel cost. Device compute,
copy/collective attribution and internal host wait attribution remain unavailable
until a diagnostic trace is analyzed; no subtraction estimate is supplied.

The input census is computed on CPU outside timing. It reports each requester's
unique rows by owner, exact unpadded request/value/gradient payload bytes, count
all-gather output bytes and the one explicit count-matrix host read. These are
logical application bytes, not topology-dependent bus traffic. The baseline
reports logical full-table collective element counts separately. Per-rank
metadata makes skew and receiver load reconstructible. Fixture rows are direct
MosaicML `LocalDataset`/MDS-reader rows `2*rank` and `2*rank+1`, decoded from the
fixture's recorded little-endian uint32 bytes. They are representative fixture
inputs, not a claim about a specific native loader microbatch position.

Both variants initialize matching FP32 input tables and `lm_head` weights before
sharding. Initial full-parameter setup is outside operation peaks, although its
allocator cache remains. GFSDP lazy initialization then allocates the actual
private `lm_head` gradient buffer, exactly `V*D*4` bytes (2004 MiB at the crop),
in both variants. The harness inventories GFSDP backing storage by underlying
storage identity, so shared views count once. The candidate's scratch selection
must resolve to this private buffer; a separate workspace or allocating
fallback fails the harness. The baseline additionally has its embedding's own
GFSDP gradient storage. This two-component difference does not model contention
with the intervening trunk/MTP units in the full model.

The real native BF16 embedding backward still allocates its dense `[V,D]`
gradient output (1002 MiB at the crop) and its N=8192 partial FP32 scratch
(about 140.8 MiB, plus native index/CUB scratch) in the affected path. These are
ordinary operation allocations, counted by actual allocated peaks separately
from the private head buffer. The owner path reuses borrowed views for selected
lookup rows, BF16 send/receive payloads, source widening and the dense FP32
per-microbatch owner sum. Its retained routing IDs/indices are fully counted
(up to 640 KiB at the crop, excluding the input IDs). No native coalescing or
routing work is hidden in precomputed state.

Run each variant/scope in a fresh process. Peaks are absolute and
phase-specific; `starting_bytes` records live allocation at each sample start,
and reserved peaks include allocator cache high-water marks. Full-table setup is
excluded, while persistent initialized buffers and gradients remain live and
count in allocated peaks. This matches the actual operation lifecycle in both
variants.

## Commands for the serialized execution owner

From the task repository root in the already verified authorized runtime:

```bash
CUDA_VISIBLE_DEVICES='' /opt/torchtitan-venv/bin/python3.12 \
  runs/ffn-row-owner/components.py --variant owner --scope integrated

PYTHONPATH=gift /opt/torchtitan-venv/bin/torchrun --standalone --nproc-per-node=8 \
  runs/ffn-row-owner/components.py --execute --variant baseline --scope integrated \
  --output runs/ffn-row-owner/artifacts/components/baseline-integrated

PYTHONPATH=gift /opt/torchtitan-venv/bin/torchrun --standalone --nproc-per-node=8 \
  runs/ffn-row-owner/components.py --execute --variant owner --scope integrated \
  --output runs/ffn-row-owner/artifacts/components/owner-integrated

PYTHONPATH=gift /opt/torchtitan-venv/bin/torchrun --standalone --nproc-per-node=8 \
  runs/ffn-row-owner/components.py --execute --variant baseline --scope phases \
  --output runs/ffn-row-owner/artifacts/components/baseline-phases

PYTHONPATH=gift /opt/torchtitan-venv/bin/torchrun --standalone --nproc-per-node=8 \
  runs/ffn-row-owner/components.py --execute --variant owner --scope phases \
  --output runs/ffn-row-owner/artifacts/components/owner-phases

PYTHONPATH=gift /opt/torchtitan-venv/bin/torchrun --standalone --nproc-per-node=8 \
  runs/ffn-row-owner/components.py --execute --variant owner --scope stages \
  --cases fixture skew --profile \
  --output runs/ffn-row-owner/artifacts/components/owner-stages
```

Defaults are two warmup iterations and three measured iterations per case; the
initial magnitude is provisional. Use unique output paths and retain process
stdout/stderr. Per-rank files are `<variant>-<scope>-rank<N>.json`; rank zero
writes `<variant>-<scope>-all-ranks.json` with all per-rank evidence and
sample-wise slowest-rank statistics. Trace files are
`<variant>-<case>-rank<N>.trace.json`. Reports record runtime, exact argv, source
revision, tracked GIFT diff hash, all GFSDP Python file hashes, harness hash,
mesh membership, active module type and input/index hashes. Failures after
initialization retain a per-rank `*-failed.json` and propagate nonzero exit.

## Earlier preparation checks

The following result predates addition of the real `lm_head` private buffer and
does not validate the storage-reuse revision. It is retained as historical
harness evidence.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=gift /opt/torchtitan-venv/bin/python3.12 \
  -m pytest -q runs/ffn-row-owner/test_components.py \
  --basetemp=.tmp/components-tests --disable-warnings -x
ruff check --select F821,F811,F841 --no-cache \
  runs/ffn-row-owner/components.py runs/ffn-row-owner/test_components.py
ruff check --no-cache \
  runs/ffn-row-owner/components.py runs/ffn-row-owner/test_components.py
```

Result: **10 passed, one warning, no skipped tests**; scoped and default Ruff
passed. The final CPU run took 2.96 seconds. The old tiny CPU route matched BF16
outputs and FP32 gradients, including a second operation using existing
accumulated-gradient storage. Fixture index SHA256:
`7cc8d53152c801450fdae44ba84c182eb8dcb9106d8bba9e3c3d69ebdaa8a8c2`.

## Storage-revision forced-CPU checks

On the revision that adds the real private `lm_head` buffer, the torch-free plan
test passed. The initial single-rank tiny F+B check could not fit the owner
backward scratch into its actual 512-byte head buffer: it requested 704 bytes.
That one-rank setup has no row sharding and is not the supported crop geometry.
The affected harness was changed to two CPU ranks and Gloo so it could retain
the real existing buffer without fabricating capacity.

The two-rank run constructed both actual GFSDP variants, confirmed the untied
FP32 head, distinct baseline embedding/head buffers, one deduplicated candidate
buffer, candidate scratch aliasing the private head storage, and equal BF16
forward outputs. It then failed in the baseline GFSDP backward before the owner
backward ran: the installed Gloo backend rejects GFSDP's
`ReduceOp.PREMUL_SUM` (`Cannot use ReduceOp.PREMUL_SUM with Gloo`). No further
CPU harness retries were made. The storage-revision scoped Ruff check passed.
This leaves full backward parity and persistent-gradient behavior for the
integrated harness path unverified on CPU; the available GPU/NCCL checks belong
to the serialized coordinator run.

## Remaining required FFN and GPU gates

The coordinator assigned a separate owner to the native FP8 FFN block factory.
This harness deliberately provides no BF16/AOT substitute for that required
measurement. The factory must preserve the crop's native quantization
conversion, FlashQLA external gate, rotary/mask/local state, initialization,
exact SAC policy, separate compile regions and actual GFSDP buffering for both
layers.0 and MTP. It must compare controls independently with matched block
inputs and upstream gradients, output/dInput/all-parameter gradient evidence,
and all-eight-rank reduction readiness/order diagnostics. `StageObserver` is
reusable by that owner, but importing it alone does not establish those facts.

All eight-rank GPU correctness, component results, FFN construction/execution,
MCP trace interpretation, native accumulation/integration and matched native
10+10 gates remain open. The CPU checks establish only harness behavior and the
small Gloo path; they establish no H100/NCCL timing or production acceptance.
