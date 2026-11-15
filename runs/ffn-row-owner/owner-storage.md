# Row-owner storage reuse

## Standard-autograd revision — preimplementation review

The prior owner backward manually mutated ``weight.grad`` and rejected
``autograd.grad`` and parameter hooks. Restore ordinary PyTorch gradient
ownership by returning an owned FP32 local dense gradient wrapped in the
parameter's existing RaggedShard spec; let AccumulateGrad handle accumulation
and hooks. This matches the pinned NeMo Automodel custom all-to-all contract:
its backward returns a gradient tensor, and the owner lookup routes IDs,
performs local embedding work, and routes value gradients back to owners
([custom all-to-all backward](https://github.com/NVIDIA-NeMo/Automodel/blob/3ea1de8d0aa902c3329b585d194c2e5274a85f3f/nemo_automodel/components/models/qwen3_8_flash_next/engram.py#L192-L261),
[owner lookup](https://github.com/NVIDIA-NeMo/Automodel/blob/3ea1de8d0aa902c3329b585d194c2e5274a85f3f/nemo_automodel/components/models/qwen3_8_flash_next/engram.py#L879-L932)).

Keep the existing native BF16 source-local coalescing and BF16 row transport,
then widen each source to FP32 and SUM it directly into the owned output.
Borrowed scratch still holds the selected BF16 rows, received BF16 rows and
one-source FP32 conversion only. At the accepted crop this removes the 250.5
MiB dense-gradient suffix from the 954.5 MiB backward borrow; the returned
250.5 MiB gradient replaces that suffix as an owned allocation. The usual
accumulated ``.grad`` remains live across microbatches, so this does not claim a
lower full-step peak. Later backward calls now allocate the owned return tensor
before AccumulateGrad adds it to the existing gradient; this can add allocator
churn, while collective count/volume, FP32 reduction arithmetic and host
synchronization remain unchanged. The ordinary shared GFSDP bucket is sized
from its actual remaining groups in ``_state._init_shared_state``; no hardcoded
capacity or pool growth is part of this change.

## Historical manual-accumulation acceptance and preimplementation review

Task base is `2ada87125e433e1f76c4cb2bfad244b641a78a93` plus the current
integrated changes. Preserve the accepted FP8/compile/SAC/MTP/accumulation,
optimizer, EMA and checkpoint contracts. Acceptance is: reuse existing GFSDP
storage without a new pool or allocating fallback; protect overlapping users
with ownership and explicit stream events; keep retained routing independent
of reusable scratch; retain source-local native BF16 rounding and per-source
FP32 SUM; keep exactly one persistent dense owner gradient; test aliasing,
reuse after delayed readers, resets/32 backwards, and affected state behavior.
Actual production component latency and allocated-memory comparison, including
native scratch and retained metadata, remain required before native timing.

The inspected `_state._init_shared_state` sizes every dtype bucket to the
largest original unit assigned to it. Removing the embedding changes both the
ordinary gradient maximum and alternating weight assignment. Its old 2004 MiB
ordinary gradient slot is therefore **not assumed available**. The unchanged
`gift_moe/_base/parallelize.py` separately wraps `lm_head` and invokes
`head.set_own_grad_buffer()` for the supported PP1 crop: chunked CE retains
unreduced gradients between its chunks. With V=128256, D=4096, eight shards,
alignment 8 and FP32 reduction, this existing bucket is exactly
`128256*4096*4 = 2,101,346,304` bytes (2004 MiB). Borrowing its root storage
does not change allocation size or shard layout. The actual initialized bucket
capacity and live owners must be checked, never synthesized or grown.

The embedding stem runs before head forward and after all trunk/MTP backward;
the head buffer is not holding a live unreduced gradient at either point.
Pending previous reductions can still read it. Acquisition must reject live
owners across all overlapping FFN/full contexts and wait all recorded release
events on the borrowing stream. Release records an event after its final
consumer; later GFSDP consumers wait that event. No `record_stream`-only reuse.

## Explicit byte layout

Let N be local token occurrences, U local unique IDs, Q total requests received
from all sources, C maximum receive count from one source, R=V/8 and H=D.
At the crop N=8192, U<=8192, Q<=65536, C<=8192, R=16032. Every offset below is
an integer byte offset aligned for its viewed dtype. Capacity is checked for
every invocation, independent of earlier counts; insufficiency is an error.

Forward borrows one existing gradient bucket:

| Bytes | Live purpose |
| --- | --- |
| `[0, 4QH)` | FP32 `index_select(out=...)` of owner master rows |
| `[4QH, 6QH)` | BF16 values converted with `copy_`, then all-to-all input |
| `[0, 2UH)` after conversion | BF16 all-to-all output, reusing dead FP32 rows |

Required size is `max(4QH, 2UH)+2QH`; the returned prefix must not
overlap the still-live send range, so send starts at `max(4QH, 2UH)`.
The worst crop extent is 1536 MiB. Output reconstruction allocates the required
BF16 `[B,S,H]` activation, as baseline lookup does. No FP32 lookup temporary or
independent BF16 payload storage remains.

Backward uses disjoint views in the same bucket, in order: BF16 selected native
rows (`2UH` bytes), BF16 receive (`2QH`), one-source FP32 conversion (`4CH`),
and dense per-microbatch FP32 owner sum (`4RH`). Crop worst bound is
64+512+128+250.5 = 954.5 MiB, before a possible contiguous incoming BF16 copy.
Widen each source into the FP32 view with `copy_`, then `index_add_` that source
into the zeroed dense per-microbatch view. Add that completed FP32 sum once to
the persistent dense owner gradient, preserving the previous source/microbatch
rounding boundaries. Scratch must never survive as persistent `.grad`.

Original IDs, unique IDs and owner-local request rows survive until backward in
their own invocation-owned allocations; none aliases the borrowed bucket.
Their worst crop footprint is `(N+U+Q)*8 = 655360` bytes (640 KiB), excluding
the original IDs' existing input allocation. The forward-only inverse and count
matrix are not retained through trunk execution. Multiple live invocations must
remain valid after intervening GFSDP writes.

## Native implementation and runtime-cost findings

Inspected runtime: `/opt/torchtitan-venv/bin/python`, torch
`2.15.0.dev20260822+cu130`, git `8b353065ce260336c5fdfa74d754e8bd9b91b062`.
The installed dispatcher registers `embedding_dense_backward.out` only as
CompositeExplicitAutograd, while `index_select.out` has a native CUDA kernel.
Do not use the embedding `out` overload to claim allocation elimination.
Pinned primary sources:
[Embedding.cu](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/aten/src/ATen/native/cuda/Embedding.cu),
[EmbeddingBackwardKernel.cu](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/aten/src/ATen/native/cuda/EmbeddingBackwardKernel.cu).

Retain native dense BF16 backward as the allowed 1002 MiB output allocation.
At N=8192 its two-pass FP32 partial buffer is
`(8192//10+8192)*4096*4 = 147636224` bytes (140.796875 MiB), plus sorted-index,
segment/CUB scratch. This also exists in baseline. For <=3072 occurrences the
native feature kernel uses no full FP32 partial buffer. Other native dispatch
branches must be accounted for in actual measurements; the low-parallelism
atomic branch can allocate a full FP32 table and must not be hidden.

Compared with the superseded owner implementation this removes FP32 selected
lookup, BF16 send/receive, selected-gradient, each `.float()` source conversion,
and fresh dense per-microbatch owner-gradient allocations. It adds bounded
Python ownership checks and event waits, and no new collectives. Host count
reads, unique synchronization, native coalescing allocations and fresh output
activation remain measured costs. Pool parity alone does not prove total peak
parity. GPU execution is serialized by the coordinator after source freeze.

The original design issue was ordinary AccumulateGrad lifetime: returning
a borrowed view can let it escape into `.grad` or let two lookup invocations
combine before the shared leaf's post-accumulate hook. The selected resolution
must cover first/later backwards, reset-to-None and zero-in-place, and multiple
live invocations without an allocation fallback.

## Historical manual-accumulation implementation and checks (superseded)

The coordinator approved explicit owner accumulation for the supported normal
training `.backward()` path. `_OwnerLookup.backward` forms the native BF16
source boundary, borrowed FP32 per-microbatch SUM, then copies once on first
backward or adds once to the existing persistent FP32 Ragged gradient. It
returns no weight gradient to AccumulateGrad, so borrowed bytes cannot escape
through that leaf. A private pinned engine check rejects direct VJP calls;
parameter gradient hooks are rejected explicitly. Inspection found normal
`trainer.py` loss `.backward()` and fused AdamW in the crop. The separate
optimizer-in-backward container uses post-accumulate hooks and is not selected;
ChunkedCE's `autograd.grad` path is only its PP shape probe on detached hidden
chunks, outside the supported PP1 owner crop. Optimizer, clipping, EMA and DCP
consume the unchanged dense FP32 Ragged `.grad`/master representation.

`GFSDPBufferContext.borrow` checks every overlapping live owner, waits release
events, and records a new release event after final scratch use. An affected
inherited bug was fixed: `_wait_for_post_backward` clears only its own exact
reduce event, preserving a newer borrower's event. No buffer allocation,
region-map change, collective change, or pool growth occurs in this interface.

CPU command (process-scoped hidden CUDA):

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=gift \
  /opt/torchtitan-venv/bin/python -m pytest -q -s \
  gift/tests/unit_tests/test_row_owner_embedding.py \
  gift/tests/unit_tests/test_gfsdp_row_owner_integration.py \
  gift/tests/unit_tests/test_gfsdp_buffer_order.py --disable-warnings --tb=short
```

**34 passed in 15.94 s**, log `owner-storage-cpu-final.log`; both explicit
Gloo ranks report CPU mesh and no CUDA initialization. The tests retain native
rounding, routing, multiple live invocations, coordinated invalid inputs,
32-microbatch accumulation, clipping, Adam moments/untouched rows, EMA and flat
baseline/owner DCP continuation. New assertions prove pointer-stable persistent
gradient storage across microbatches and zero-in-place resets, allocation only
for the persistent gradient after reset-to-None, no pool alias in `.grad`, full
scratch overwrite between live lookup forwards and their backwards, exact
byte-view aliasing, nested/live-region rejection and capacity failure without
fallback. The explicit NCCL entrypoint also exercises pending NCCL reads on
another stream, borrower writes, subsequent-reader waits and preservation of
the newer borrower event against an older group's final callback.

Retained first attempt: `owner-storage-cpu.log` has 1 failed/1 passed because
`get_gradient_edge` internally views the Ragged leaf, an unsupported operation.
The direct engine edge already available as `ctx.next_functions[0][0]` avoids
that view; `owner-storage-cpu-2.log` passed both original owner tests before the
final alias/event checks were added. No kernel or precision workaround was
introduced. Scoped Ruff F821/F811/F841, full default Ruff on owner/test files,
and `git diff --check` pass.

## Historical CUDA/NCCL receipt for manual accumulation

The coordinator granted the sole GPU lease for these two checks. Preflight
found all eight H100 80GB devices at 0 MiB/0%, no compute applications, and the
task checkout on the verified SR008 NFS mount. The eight-rank tiny fixture's
maximum forward/backward leases are 768/640 bytes within its actual 4096-byte
non-embedding GFSDP gradient bucket; the production dimensions were unchanged.

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=gift \
  /opt/torchtitan-venv/bin/python -m torch.distributed.run --standalone \
  --nnodes=1 --nproc-per-node=8 gift/tests/unit_tests/test_row_owner_embedding.py \
  --backend nccl --checkpoint-dir \
  /workspace-SR008.nfs2/vkropotin/optimization-task/runs/ffn-row-owner/artifacts/owner-storage-nccl-checkpoints-2
```

**Passed, torchrun exit 0**, all eight ranks explicitly report CUDA/NCCL.
Parent PID 229622; ranks 229629–229636. Durable log:
`owner-storage-eight-rank-nccl-2.log`. This executes all routing/rounding,
accumulation/reset/no-alias, pending collective/borrow/event, clipping, Adam,
EMA, baseline flat DCP and continuation checks described above.

The first invocation used a relative `--checkpoint-dir`, which the existing
test concatenates into a `file://` URL. Pinned PyTorch's
`distributed/rendezvous.py::_file_rendezvous_handler` takes `urlparse(url).path`,
dropping that URL's relative netloc and waiting on a nonexistent root path.
No process group was established. Only task torchrun PID 229447 was terminated
with SIGTERM; its workers 229459–229466 exited and no compute applications
remained before the unchanged-source absolute-path rerun. Its exit 1/log
`owner-storage-eight-rank-nccl.log` are retained. This is an invocation failure,
not a numerical/kernel pass or a hidden production failure.

Then the existing affected FFN regressions ran serially:

```sh
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=gift \
  /opt/torchtitan-venv/bin/python -m pytest -q -s \
  gift/tests/unit_tests/test_gfsdp_ffn_overlap.py --disable-warnings --tb=short
```

**3 passed in 32.18 s, exit 0**. Parent PID 230262; two-rank test pairs
230312/230313, 230721/230722 and 230827/230828. Log
`owner-storage-ffn-cuda-nccl.log`. This preserves the established region budget,
actual collective/rank layout, pending disjoint-region readers, and exact
BF16 AOTAutograd/SAC 32-microbatch Adam/DCP/EMA continuation evidence after the
release-event helper change. Vendor quack cache-load warnings fell back to
in-process compilation; no environment/cache mutation was performed.

All five entries in `owner-storage-source.sha256` matched before and after GPU
work. No source changed while ranks were active. Both owned groups exited,
the final compute-applications query was empty, and the GPU lease was released
immediately to the coordinator. No optional suite or benchmark was launched.

Further required gates belong to the integrated campaign: targeted independent
review and actual production component latency/allocated comparison including
native internal scratch and retained routing. Tiny alias/pointer tests establish
no extra scratch pool and safe named lifetimes; they do not establish full-crop
total peak memory parity or a speedup. Native timing remains behind the actual
component/correctness gates.

Additional pinned out-kernel inspection: CUDA `Indexing.cu::index_select_out_cuda`
uses `resize_output` on the exact-shaped ordinary output view; its <=16-index
path launches directly, while the larger path passes index `view`/`expand` into
`gather_out`. `TensorAdvancedIndexing.cpp::gather_out` dispatches CUDA
`ScatterGatherKernel.cu::gather_cuda_kernel`; its TensorIterator uses
`resize_outputs(false)` with the provided output, ordinary source/index views
and no tensor-storage allocation. Contiguous same-device `copy_` provides the
FP32/BF16 conversions. Allocation claims therefore use the pinned bodies and
exact output shape, beyond the overload names alone. Actual allocator peaks
still count native embedding output, sort/CUB scratch and conversion/contiguity
work within the baseline-equivalent measurement scope.

## Standard-autograd restoration — implementation and verification

`_OwnerLookup.backward` now creates an owned FP32 `[rows_per_owner, hidden]`
gradient, widens each received BF16 source into the borrowed one-source view,
and adds that source directly into the owned result. It returns a flat
`RaggedShardDTensor` using the existing parameter spec. The dense sum no longer
occupies a borrowed suffix, and `_accumulate_gradient`, the private engine-node
guard, and parameter-hook rejection are removed. The existing
`@once_differentiable` boundary remains. Routing IDs/local rows remain
invocation-owned; scratch still protects its existing native/inter-group
readers with the same borrow events. No communication or reduction-order change
was made.

The full-shape gradient-bucket budget is source-derived from the accepted crop
meta plans and the current GFSDP sizing code. The plans identify the actual crop
groups and parameter counts; the all-gather key and storage arithmetic come
from `GFSDPParamGroup` and `GFSDPState._init_shared_state`:

| Gradient bucket | Option state | `all_gather_dtype` key | Padded FP32 elements | Capacity | MiB |
| --- | --- | --- | ---: | ---: | ---: |
| Ordinary shared | Control: owner and FFN splits off | `None` | 525,336,576 | 2,101,346,304 bytes | 2004 |
| Ordinary shared | Owner on, FFN splits off | `None` | 234,271,232 | 937,084,928 bytes | 893.673828125 |
| Ordinary shared | Owner and layer-0/MTP FFN splits on | `None` | 234,271,232 | 937,084,928 bytes | 893.673828125 |
| Private `lm_head` | All states | `None` | 525,336,576 | 2,101,346,304 bytes | 2004 |

The accepted workload uses dense shard world 8 and shard alignment 8, so each
group's padding divider is 64 elements. The actual native crop
[MTP plan](artifacts/raw/ffn-plan/mtp-plan.json) records 234,271,232 MTP
parameters; its selected FFN has 134,479,872 parameters and the remainder has
99,791,360. All three counts are exact multiples of 64. The
[layer-0 plan](artifacts/raw/ffn-plan/layer0-plan.json) records 202,424,512
parameters, also divisible by 64. Thus MTP's split parent and child sum to the
original padded group exactly. The input and output tables each have
`128256 * 4096 = 525,336,576` elements, also divisible by 64. No alignment
padding is required for these groups.

The GIFT `GMixedPrecisionPolicy` sets BF16 parameter gathers and FP32
reductions, with `all_gather_dtype_to_param_cls=None`; the group has the single
`None` dtype key and its gradient byte count uses four bytes per padded
parameter. `_init_shared_state` forms ordinary slots from the current
`grad_param_groups` and the per-dtype maximum `original_buffer_sizes`; the
owner Parameter is excluded from every wrapper by the shared `ignored_params`
set. This is why the ordinary slot shrinks from the input-table 2004 MiB to the
largest remaining dense group, MTP at 893.673828125 MiB. With FFN splits,
`original_buffer_sizes` recombines each parent and child before taking that
maximum. The unchanged `lm_head.set_own_grad_buffer()` keeps its separate
2004 MiB FP32 reduction bucket. The production code retains the computed-max
algorithm; these bytes are not hardcoded.

This is an exact padded-count calculation from the actual native crop meta
plans plus inspected dtype/group-sizing code, not an observation of the final
full-crop bucket allocation or total allocator peak. Root's native integration
check still needs to confirm the initialized per-dtype capacities and peak on
the final owner/split source. The newly owned 250.5 MiB local FP32 gradient is
allocated before native BF16 `dense` is deleted and while that 1002 MiB result
is live. It replaces the removed 250.5 MiB borrowed suffix, but this does not
prove a lower peak; include both tensors, the existing accumulated `.grad`, and
the retained GFSDP pools in that check. No source or pool-size change is needed
for correctness unless this measured gate fails.

The pinned NeMo Automodel reference at commit
[`3ea1de8d0aa902c3329b585d194c2e5274a85f3f`](https://github.com/NVIDIA-NeMo/Automodel/tree/3ea1de8d0aa902c3329b585d194c2e5274a85f3f/nemo_automodel/components/models/qwen3_8_flash_next)
returns a gradient tensor from `_FixedCapacityAllToAll.backward` and uses that
custom exchange in its owner lookup. The source and tests now apply that
standard-autograd ownership boundary while retaining the local native
BF16-coalescing contract required here.

Focused NCCL command on the granted eight-H100 lease:

```sh
timeout 600s env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=/workspace-SR008.nfs2/vkropotin/optimization-task/gift \
  /opt/torchtitan-venv/bin/python -m torch.distributed.run --standalone \
  --nnodes=1 --nproc-per-node=8 \
  gift/tests/unit_tests/test_row_owner_embedding.py --backend nccl \
  --checkpoint-dir \
  /workspace-SR008.nfs2/vkropotin/optimization-task/runs/ffn-row-owner/artifacts/owner-standard-autograd-nccl-checkpoints
```

The command exited 0; the durable log
[`owner-standard-autograd-eight-rank-nccl.log`](owner-standard-autograd-eight-rank-nccl.log)
contains all rank 0–7 CUDA/NCCL markers. It covers native source-local BF16
duplicate coalescing and FP32 owner SUM, simultaneous live lookups, 32-step
gradient accumulation, zero-in-place and set-to-None reset, direct VJPs with
both missing and existing `.grad`, mutable parameter hooks, post-accumulate
hooks, no borrowed output storage, clipping, fused AdamW, EMA and baseline/owner
flat DCP continuation. The source and affected test hashes remained
`f3a01c6cf99b981db7b4ef66866fe46259af84e9d2c21e2044e9ae18ff1696fd` and
`fb9c08e313c202b58fd7f8c8ab714e943ea0254bb46d8ea58ff0fd88dc57b990` through
the run. Supporting source, tests, crop plans and GPU records are pinned in
[`owner-standard-autograd-source.sha256`](owner-standard-autograd-source.sha256).
Before and after checks found all eight H100s idle and no compute applications
after release. Scoped Ruff F821/F811/F841, full Ruff, Ruff format check and
`py_compile` also passed. CPU/Gloo reruns were skipped under the
coordinator's instruction; the existing older CPU/Gloo evidence above applies
to the historical manual-accumulation revision only. No full-crop owner/split
peak or performance measurement was run as part of this restoration.

The coordinator's full-shape embedding component check is recorded in
[`embedding-numeric-results.json`](embedding-numeric-results.json). It used
`[2, 4096, 4096]` BF16 lookup outputs, an eight-H100 dense-shard NCCL group,
two output consumers and exact seeded BF16 upstreams. All eight ranks reported
exact output and gradient equality for the first gradient, accumulated second
backward, and backward after `grad=None`; no nonfinite values were found. After
the same post-setup cache policy, the warmed complete forward/backward fixture
peaked at 8,950,373,888 allocated / 9,003,073,536 reserved bytes for baseline
and at most 5,781,722,112 allocated / 7,648,313,344 reserved bytes for owner.
This confirms the full-shape embedding fixture's output/gradient lifecycle and
peak under that isolated setup. It does not instantiate the full native model's
ordinary per-dtype GFSDP pool groups or establish full-step peak parity; the
native integration inventory remains an acceptance gate before 10+10.
