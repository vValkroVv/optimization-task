# Compact source gradient coalescing

Original A computes source-local gradients in a full-vocabulary BF16 output,
then gathers the source's sorted unique rows into the send buffer. The
historical `compact` selector at `3860b1eb5` already saves the inverse IDs and
uses `aten.embedding_dense_backward(grad_bf16, inverse, U, -1, False)` when its
CUDA dispatch guard passes. It sends the result directly. The sorted unique IDs
define the increasing mapping from global IDs to compact row labels. The
flattened inverse keeps original token order. BF16 request-source gradients,
peer splits, donor order and the native FP32 owner accumulation are identical
when `compact` is the only selector active.
[Original A backward](../../gift/torchtitan/distributed/gfsdp/row_owner_embedding.py#L387) · [Historical implementation](../../../../runs/schedule-embedding/embedding-implementation.md#2a-compact-source-local-coalescing)

## Ordering on the pinned CUDA build

- At `N <= 3072`, `scale_grad_by_freq=False` selects the feature kernel. It
  groups duplicates at the same token positions within each chunk, accumulates
  the contributions in FP32 and updates the BF16 output for that chunk. The
  increasing relabeling keeps the chunk groups and BF16 rounding points.
  [ATen dispatch](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/aten/src/ATen/native/cuda/Embedding.cu#L240-L281) · [Feature kernel](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/aten/src/ATen/native/cuda/Embedding.cu#L39-L116)
- For larger `N`, ATen stably radix-sorts row labels with original offsets as
  values. Relabeling can change the number of radix bits and preserves the
  order of segments and positions within each segment. The two-pass path
  forms FP32 partial sums in slices of up to ten positions, adds those partials
  in segment order, and writes the BF16 output. Launch bounds can shrink
  without changing those sums. Odd widths use the same rounded-up warp stride
  in both calls and mask lanes beyond the actual width.
  [ATen sort call](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/aten/src/ATen/native/cuda/Embedding.cu#L283-L295) · [ATen CUB wrapper](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/aten/src/ATen/cuda/cub-RadixSortPairs.cuh#L30-L51) · [CUB stability contract](https://github.com/NVIDIA/cccl/blob/v3.0.0/cub/cub/device/device_radix_sort.cuh#L103-L106) · [ATen partials and final sum](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/aten/src/ATen/native/cuda/EmbeddingBackwardKernel.cu#L109-L143) · [ATen final write](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/aten/src/ATen/native/cuda/EmbeddingBackwardKernel.cu#L217-L245)
- Let `W=ceil(D/32)*32`, `B=min(W,1024)`, `M=min(N,num_weights)`,
  `P=floor(N/10)+M`, and `G=ceil(M*W/B)`. The historical selector requires
  `G >= 4*SMs` for large `N`, so the compact and full tables take the ordered
  path. The exact pinned atomic predicate for dense embeddings is
  `G < 4*SMs && P > 4*G && !deterministicAlgorithms()`. Here the atomic path
  adds into **FP32** for BF16 inputs, then casts to BF16. Its unordered partial
  additions can change BF16 bits. At `N=8192, D=4096, SMs=132`, full vocabulary
  has `G=32768`. The compact selector falls back for `U<132`, and unguarded
  compact can take the atomic path at `U<=54`. Real loader batches reached
  `U=12`.
  [ATen dispatch and atomic accumulation](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/aten/src/ATen/native/cuda/EmbeddingBackwardKernel.cu#L317-L387) · [Real-data census](../../../../runs/production-data-abcd/integration-review.md)

## What was measured

The archived `d0` versus `d1-compact` component comparison used the same eight
signed-int64 input hashes as the current [embedding replay](embedding-replay-result.json).
The compact path ran on all eight H100 ranks with unique counts
`2234, 1888, 1559, 3208, 1955, 1741, 1553, 2859`. Median unprofiled rank-max
synchronized forward/backward fell from `2.739` to `2.496` ms (8.9%). Observed
component peak allocated/reserved fell from `5.52/17.64` to `4.49/15.88` GiB.
This was one serial pairing. The historical native `SE` versus `S` result
combined compact with two other embedding changes, so it cannot isolate compact.
[Archived component report and raw receipts](../../../../runs/schedule-embedding/component-results.md) · [Historical native comparison](../../../../runs/schedule-embedding/report.md#direct-embedding-routing-comparison)

At the current rank-0 `U=2234`, the source-derived BF16 output sizes are
`1,050,673,152` and `18,300,928` bytes. The ordered FP32 partial-buffer sizes are
`147,636,224` and `50,020,352` bytes. The compact result stays live through its
same-stream NCCL read. It removes the selected-row gather and its borrowed send
view. The head gradient pool and dense owned FP32 local gradient remain the
same. Requester traffic stays `8U` ID bytes and `2UD` gradient bytes. These sizes
describe individual tensors and messages. The allocator peaks above cover the
measured component.

Historical CUDA checks covered empty input, `N=3072/3073/8192`, repeated groups
around the ten-position boundary, and guard fallback at `D=4096`. A separate
raw MDS rank-0 fixture with `U=2141` matched the BF16 payload bitwise using a
seeded BF16 derivative. It differs from the current eight-rank capture. The
historical optimized NCCL suite covered owner accumulation and reset, clipping,
optimizer and checkpoint state using its scoped test fixtures. A local CPU
ATen check matched BF16 bytes for 30 relabeling cases spanning empty, repeated
and odd-width inputs. CPU dispatch differs from CUDA. GIFT's eager embedding
precedes the checkpointed blocks, and the backward receives the gradient
already combined by autograd from trunk and MTP. Current isolated NCCL payload
parity, odd-width CUDA parity and full compiled SAC continuation remain open.
[Historical numerical and distributed evidence](../../../../runs/schedule-embedding/report.md#focused-correctness-results) · [Decoder forward](../../gift/torchtitan/models/common/decoder_blocks/decoder.py#L115)

## Follow-up decision

Reuse the historical selector for an isolated current native memory test if
needed. Repeating the matched high-`U` component timing gives little new
dispatch evidence. A distinct low-`U` experiment can call ATen with
`C=max(U, floor((4*SMs-1)*B/W)+1)` output rows and send `out.narrow(0,0,U)`.
This only applies on large-`N` shapes where both `N` and the full vocabulary
reach the minimum row count, which keeps both branches ordered. At the current
H100 shape and `U=12`, `C=132` uses a `1,081,344`-byte BF16 output and sends the
same 12 rows. Test raw BF16 payload and FP32 local gradients on real low-`U`
inputs and the guard boundaries before using it. The current eight high-`U`
inputs exercise the historical compact path identically in both variants and
cannot distinguish the proposed low-`U` treatment.

### Low-unique fixture and integration

For the real fixture, use the eight `depth0/rank*.json` paths in the preserved
`data-002-loader-summary.json`. Select the earliest microbatch index with the
lowest `records[].unique_token_ids` across ranks, with rank as the final tie
breaker. Use the `records[microbatch]` entry from all eight ranks so the same
natural group exchanges rows. Read its `row_indices` from `data-002/train` with
`LocalDataset`, verify each little-endian uint32 row against `row_hashes`, then
convert to signed int64 B2×4096 and record each rank's ID hash and unique count.
The local summary contains the receipt paths and reported minimum, while the
individual rank receipts are on SR008. The exact row numbers remain unresolved.
[Loader summary](../../../../runs/production-data-abcd/data-002-loader-summary.json) · [Receipt fields and hash](../../../../runs/production-data-abcd/verify_data.py#L39) · [MDS row reader](../../../../runs/ffn-row-owner/components.py#L88)

For shape and order checks, hold `N=8192` and `D=4096` and use `U` in
`1, 12, 54, 55, 131, 132`. Use sorted unique IDs crossing owner row boundaries,
shuffle repeated IDs at a fixed seed and include runs of 9, 10, 11, 19, 20 and
21 where cardinality allows. Give at least one selected ID contributions with
large signed cancellation and small terms. Include `N=0, 3072, 3073` and an
odd width in the separate source-op parity check.

The historical integration points are `_compact_native_is_safe`, the
forward's saved inverse and unique count, and `_native_source_coalesce` in
`3860b1eb5:gift/torchtitan/distributed/gfsdp/row_owner_embedding.py`. For the
padded variant, save the selected output row bound and narrow the owned BF16
buffer to `U` before the existing gradient exchange. Keep `owner_add`,
`owner_lookup` and `routing` disabled in this comparison. Reuse the existing
exact replay's message capture to compare real rank-local BF16 output, send
and receive bytes with splits, and initial, accumulated and reset FP32
gradients. Complete one short native compiled/SAC continuation before
interpreting whole-training memory. [Exact replay utilities](row_owner_route_numeric.py)
