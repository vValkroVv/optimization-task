# Row-owner embedding optimization handoff

## Scope and implementation

Implemented in the task-owned checkout at source base `e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f`. The root/GFSDP worker owns config, schema and `parallelize.py`; this change adds the default-off `optimize: bool = False` factory keyword and a read-only `optimization_stages` tuple for direct diagnostic construction. The enabled stage order is `compact`, `owner_add`, `owner_lookup`, `routing`. Construction logs the tuple once, with no per-microbatch logging. Root wiring should call `from_embedding(..., optimize=gfsdp_row_owner_embedding_optimizations)`.

The new [kernel module](../../gift/torchtitan/distributed/gfsdp/_row_owner_embedding_kernels.py) is imported only on a selected, supported CUDA path. CPU, non-CUDA, unsupported layout/type, and out-of-range routing-size paths keep the existing native implementation.

### 2A: compact source-local coalescing

The backward saves the int64 inverse instead of token IDs when compact native dispatch is safe. It keeps host shape/unique-count metadata and the existing local row request tensor, and does not save the unique-ID tensor. The native compact BF16 result already follows the sorted unique-row order, so NCCL reads it directly; the tensor is deleted after that same-stream read is enqueued.

Pinned PyTorch `Embedding.cu` dispatches `N <= 3072` to `embedding_backward_feature_kernel`, independently of `num_weights`; relabeling with inverse IDs is safe there. For larger `N`, compact native dispatch is enabled only if the exact pinned `sum_scatter_grid` is at least `4 * num_sms`, preventing the native unordered BF16 atomic path. The calculation uses `stride_warped = ceil(D/32)*32`, `block = min(stride_warped, 1024)`, and `max_segments = min(N, U)`. Otherwise it retains full-vocabulary native backward plus unique gather. References: [Embedding.cu](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/aten/src/ATen/native/cuda/Embedding.cu#L240-L324), [EmbeddingBackwardKernel.cu](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/aten/src/ATen/native/cuda/EmbeddingBackwardKernel.cu#L31-L34), [grid guard](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/aten/src/ATen/native/cuda/EmbeddingBackwardKernel.cu#L330-L346), [MAX_BLOCK_SIZE](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/aten/src/ATen/native/cuda/SortingCommon.cuh#L12-L16).

For the real-data rank-0 capture below (`N=8192`, `U=2141`, `D=4096`), the compact BF16 output is 17,539,072 bytes (16.73 MiB), versus 1,050,673,152 bytes (1002 MiB) for the full-vocabulary BF16 table. Native FP32 partial storage also scales with `N/10 + max_segments`; it is 48,496,640 bytes at `U=2141`, versus 147,636,224 bytes at the full-vocabulary `max_segments=8192`. These are source-derived tensor sizes, not measured peak-memory or latency claims.

### 2B: owner-side BF16-to-FP32 add

The optimized owner path launches one source-specific kernel for each nonempty source on the shared communication stream, preserving source-rank order and omitting both the BF16-to-FP32 scratch copy and the following `index_add_` launch. It keeps the required zero-initialized owned FP32 local gradient. `_route` creates a sorted unique request list per requester and all-to-all preserves the per-source split, so each source segment has one writer per destination row; launches for successive sources are serialized.

The first kernel draft used `tl.atomic_add` and failed exact parity: BF16 `0x0001` converted to FP32 and added to `+0` produced native FP32 `0x00010000`, but the global atomic produced `0x00000000`. That failed evidence is retained at [.tmp/row-owner-embedding/gpu0-atomic-owner-add-failure.log](../../.tmp/row-owner-embedding/gpu0-atomic-owner-add-failure.log). A focused CUDA profiler run of contiguous unique dim-0 FP32 `index_add_(alpha=1)` observed native kernel `kernel_cutlass__kernel_CopyAtom_ThrID10_TVLayoutSrc112801_TVLayoutDst112801_Valuetypef32_tensor00o1110_tensorptri64gmemo1_tensorptrf32gmemoi641_____0`, consistent with its subnormal-preserving path. The final kernel uses a single-writer FP32 load/add/store, not an atomic. It passed raw-bit comparison for ordered cancellation, signed zeros, subnormals, infinities, and signed NaN payloads. The pinned dispatch condition is in [Indexing.cu](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/aten/src/ATen/native/cuda/Indexing.cu#L1095-L1120); the runtime bit comparison remains the correctness evidence.

### 2C: owner lookup and BF16 conversion

FP32 indexed loads now write one BF16 conversion directly into the existing send region. The borrowed scratch layout is returned rows `[0, 2UD)` and send rows `[2UD, 2(U+R)D)`, disjoint during NCCL; no FP32 selected-row temporary is created. The NCCL exchange and reconstruction stay in their current order. The returned autograd gradient remains an owned dense FP32 local tensor.

### 2D: bounded routing preparation

For CUDA signed-int64 inputs with `N <= 8192`, one CTA validates and clamps into safe signed-int64 IDs, writing validity into the existing metadata scalar. After `torch.unique`, a second CTA counts each contiguous owner interval directly into the metadata vector with int64 counts. Empty input writes valid counts/metadata. `INT64_MIN/MAX` are clamped before division on the fallback path and before unique/owner interval counting on the optimized path. Unsupported ID dtype/device and `N > 8192` retain the coordinated native path. Both paths keep the fixed count all-gather, host-visible split read, request/value exchanges, inverse reconstruction and coordinated invalid-input rejection before payload exchange.

## Runtime-cost review

Before the change, the affected path used full-vocabulary BF16 native backward plus a unique gather; for each owner source, BF16-to-FP32 scratch conversion plus `index_add_`; FP32 `index_select` plus BF16 copy for forward owner lookup; and tensor-based validation/clamping/owner counting before the fixed count exchange. After the change:

- Source-local coalescing allocates the native compact result only when its exact dispatch guard allows it. The inverse replaces saved IDs at equal `8N` bytes and drops the saved `8U` unique-ID tensor on compact calls. No full-vocabulary dense table is made on the compact path.
- Owner reduction keeps the required dense FP32 gradient zero-fill, and removes per-source FP32 scratch and one kernel launch per nonempty source. The source kernels remain serial in rank order.
- Owner lookup removes the selected-row FP32 scratch and copy launch. It writes the BF16 send buffer into the same borrowed GFSDP storage, with the receive and send ranges disjoint.
- Routing adds no collective or host read. It replaces the prior validation/clamp/owner-count intermediates with two bounded Triton launches, retains `torch.unique`, and retains the one fixed all-gather plus host-visible splits. Larger inputs fall back.
- There is no persistent GPU pool or borrowed scratch escaping its existing storage lifetime. Existing CUDA stream handoff, NCCL order, `record_stream` calls, and the ordinary accumulated gradient are retained.
- Triton imports are lazy. Dynamic ID/unique/source row counts are runtime scalar arguments; only power-of-two `BLOCK`, hidden width and fixed mesh metadata specialize kernels. This avoids per-count JIT variants. First-use compile cost and production allocator peaks still need the root-owned integration/performance campaign.

No component timing, peak-memory run, matched native 10+10 comparison, or Perfetto campaign was run here. The change count and byte calculations above do not establish a speedup or production memory peak.

## Verification

Runtime identity: `/opt/torchtitan-venv/bin/python3.12`, PyTorch `2.15.0.dev20260822+cu130`, git `8b353065ce260336c5fdfa74d754e8bd9b91b062`, CUDA 13.0, Triton 3.8.0, GPU0 NVIDIA H100 80GB. The installed PyTorch git revision exactly matches the pinned source.

- CUDA-hidden behavior: `CUDA_VISIBLE_DEVICES='' PYTHONPATH=gift pytest -q gift/tests/unit_tests/test_row_owner_embedding.py gift/tests/unit_tests/test_row_owner_embedding_kernels.py --disable-warnings -x` — **2 passed, 5 skipped**. This covers CPU/Gloo route/invalid-ID coordination, duplicate and empty inputs, repeated accumulation, ordinary hooks and `autograd.grad`, optimizer/clipping/EMA/checkpoint state, and stage-selector validation. GPU kernel cases skip when `ROW_OWNER_CUDA_KERNEL_TESTS` is unset.
- Scoped lint/format: `ruff check --select F821,F811,F841 --no-cache gift/torchtitan/distributed/gfsdp/row_owner_embedding.py gift/torchtitan/distributed/gfsdp/_row_owner_embedding_kernels.py gift/tests/unit_tests/test_row_owner_embedding.py gift/tests/unit_tests/test_row_owner_embedding_kernels.py` and `ruff format --check` on the same files — **passed**.
- Syntax: `/opt/torchtitan-venv/bin/python3.12 -m py_compile` on those four files — **passed**.
- GPU0 compact parity: `CUDA_VISIBLE_DEVICES=0 ROW_OWNER_CUDA_KERNEL_TESTS=1 PYTHONPATH=gift /opt/torchtitan-venv/bin/python3.12 -m pytest -q gift/tests/unit_tests/test_row_owner_embedding_kernels.py --disable-warnings -x` first reported **2 passed, then the initial atomic 2B implementation failed**; the atomic failure is preserved above. The compact cases included `N=0/3072/3073/8192`, safe and guarded fallback cases, and skewed duplicate groups of 9/10/11/19/20/21.
- After replacing atomics, the focused 2B check passed: `CUDA_VISIBLE_DEVICES=0 ROW_OWNER_CUDA_KERNEL_TESTS=1 PYTHONPATH=gift /opt/torchtitan-venv/bin/python3.12 -m pytest -q gift/tests/unit_tests/test_row_owner_embedding_kernels.py -k owner_add --disable-warnings -x` — **1 passed**.
- The route metadata and grid-guard check passed: the same pytest command with `-k 'routing or guard'` — **2 passed**, including signed INT64 limits, empty IDs, and the 8192-ID bounded block.
- After removing dynamic row counts from constexpr specialization, affected CUDA0 checks were rerun with `-k 'owner_add or owner_lookup or routing'` — **3 passed**. The runtime scalar arguments compile with bounded power-of-two block choices.
- Corrected production IDs: `runs/production-data-abcd/artifacts/raw/data-002/train`, MDS rows 0 and 1, shape `(2,4096)`, flattened `N=8192`, `U=2141`, ID SHA256 `606edf2e732c9777eb7c891349fbebd33cede74d213b6af2c531683292b49ec0`. Compact native BF16 output matched full-vocabulary native output plus unique gather bitwise (0 mismatches). The derivative used was a **synthetic matched gradient pattern on real production IDs**: seed 2048, BF16 native values divided by 32 plus BF16 MTP values multiplied by 0.3 and added in BF16. These were not captured native upstream gradients.

## Distributed handoff and open gates

Two one-rank NCCL launches were attempted on GPU0; neither produced the worker's first `row_owner_checks rank=0` receipt. The first direct-Python launch hung before process-group/mesh initialization. The torchrun attempt emitted c10d `Address family not supported by protocol` socket warnings and also produced no worker receipt; it was stopped at the parent instruction. Console records with exact argv/environment, last marker and exact process cleanup are [gpu0-one-rank-direct.log](../../.tmp/row-owner-embedding/gpu0-one-rank-direct.log) and [gpu0-one-rank-torchrun.log](../../.tmp/row-owner-embedding/gpu0-one-rank-torchrun.log). GPU0 was clear after cleanup. Neither is a supported NCCL result; this worker did not source `/etc/profile.d/wm.sh`.

The root owns the established sourced launch path and the later all-eight-rank NCCL verification, including standard backward/hooks/`autograd.grad`, accumulation, uneven/empty/skewed routing, scratch lifetime, and optimizer/checkpoint gates. This handoff does not claim all-rank correctness, compile/SAC integration, performance magnitude, memory peak, convergence, or production acceptance. The full task's native comparisons and Perfetto evidence remain with the root campaign.
