# Block quantizer source assessment

## Result and finite acceptance

No explicit device trap or demonstrated input, registration, stream or lifetime
violation was found on the inspected `quantize_blk128x128` path. The originating
kernel remains unknown. This assessment supports no quantizer repair, dependency
upgrade or additional fault sweep. Retain the already selected single native
completion-prefix diagnostic and its separate loaded-quantize-binary receipt.
Its output may narrow an interval; it cannot establish causality from the
asynchronous stack.

Completed checklist: inspect installed wrapper/dispatcher/kernel and direct
helpers; compare selected call metadata with native/Meta contracts; examine
local stream/ownership contracts; state the justified decision and provenance
limits. Only this artifact directory was written. No source/package edits,
imports, CUDA queries, training, benchmark, installation or process action was
performed. `code-rules` and `perf-experiments` were applied to this scoped source
review. This documentation adds no production runtime work.

## Established source route

Installed package root is
`/opt/torchtitan-venv/lib/python3.12/site-packages/cudacherry` (metadata 0.1.0).
Exact SHA-256/size pins for the inspected files are in
[source-receipts.json](source-receipts.json). Local installed source is not proof
of the loaded binary's build source. Existing provenance research found local
wheel origin `/opt/cudacherry`, without an installed source commit; its upstream
repository is GigaLLMs CudaCherry. That unresolved identity prevents claiming a
version-matched upstream fix. No new upstream checkout or recursive download was
made. GIFT upstream pin supplied by the coordinator is
`6fd9a3230ab6361f11bbbf0b6ec5a5f125b78fc2`; frozen task snapshot 007 has base
`446431dfa191ef0300f116257ae918bc85e38c06`. Its two changed FP8 integration files
match the manifest; other inspected source bytes are independently pinned here.

- GIFT `torchtitan/ops/float8/_cudacherry.py` directly imports the public
  `quantize_blk128x128` wrapper. `cuda_gemm.py:118` quantizes weights for forward,
  including SAC recomputation; `:191` separately requests transposed weights for
  dgrad. Saved weight tensors use `ctx.save_for_backward` at `:156`.
- Installed `quantization/blockwise.py:75` calls `load_family("quantize")`,
  makes noncontiguous inputs contiguous, then invokes the dispatcher operation.
  `_loader.py` caches a prebuilt extension or JIT fallback. This does not prove
  what failed rank 3 loaded.
- Installed `csrc/quantization/blockwise/bindings.cpp:228` implements
  `quantize_blk128x128_raw`; `:587` defines the plain tensor schema and `:598`
  registers the CUDA body, with a separate Meta body. The schema declares fresh
  outputs and no mutation/alias. The raw body allocates fresh FP8/FP32 tensors;
  its uint8 dtype view shares the returned FP8 tensor's owned storage. No hidden
  persistent quantize workspace appears on this path. Pybind entry-point checks
  are not dispatcher checks; the active dispatcher instead reaches the CUDA
  entry's own `TORCH_CHECK`s.

## Selected call and explicit trap assessment

The preserved cached callable is
`NATIVE-SMOKE/001/cache/torchinductor/b3/cb3phbwhwqlg3hsapbyxedotxx7g54mlajsocxjdeioufefknbd7.py`,
whose exact name and line 4345 match the later failure stack. It is a reused
cached source artifact, not a newly recovered failing-run binary.

At lines 4343–4360 it checks input `(2048,1536)`, stride `(1536,1)`, calls
`quantize_blk128x128(primals_28,None,True,False,False,0.0)`, checks outputs
`(2048,1536)` FP8 and `(16,12)` FP32, and passes both to
`torchtitan.fp8_dense_gemm`. The local references persist through that consumer;
`primals_28` is retained until its later transposed quantization at 4494–4495.
Native and Meta output shapes/dtypes/contiguous strides agree for this call.

In installed `block_quantization_kernel.cu`, this selects
`quantize_blk128x128_kernel_nt_computed_scale<true,false,false>` via
`launch_128x128_compute_scale_impl<true,false>`, with grid `(12,16)` and 256
threads. Both dimensions are exact tile multiples; no tail or precomputed-scale
branch is selected. Dimensions/offset products here are far within the used
integer ranges; global offsets multiply in int64. The kernel performs BF16
loads, shared-memory staging and warp/block reduction, FP32 scale arithmetic,
FP8 conversion and padded output stores. Its barriers are uniform on this
specialization. There is no explicit runtime `assert`, trap instruction,
architecture rejection, device printf or timeout in this kernel or its seven
direct project helper headers. `static_assert`s are compilation checks, not a
runtime device-assert mechanism. This is a source observation, not a disassembly
claim about the actual loaded quantize binary or all compiler-generated code.

The launcher uses `CUDAGuard(x.device())`, `getCurrentCUDAStream()` and
`C10_CUDA_KERNEL_LAUNCH_CHECK()`. The installed Torch header defines the last
macro as `C10_CUDA_CHECK(cudaGetLastError())`; it is not a completion wait. This
provides an ordinary reporting point for an earlier asynchronous error and does
not identify quantize as the source. The same cached callable has earlier FP8
GEMMs, quantizers and fused recomputation work before line 4345.

## Concrete contracts without a demonstrated violation

The native boundary checks CUDA/BF16/2D/contiguous input and fresh output
shape/dtype/contiguity. The selected `None` precomputed scale is consistent with
`use_precomputed_scale=False`. No device-value predicate branches to a trap;
nonfinite inputs have no explicit assertion on this path.

Vector BF16 loads require suitable pointer alignment (the included BF16 helper
states an 8-byte precondition), and uint32 output stores require four-byte
alignment. Padded outputs and step-four indexing satisfy output-offset
alignment. Shape/stride alone does not establish the input base alignment:
`primals_28` has no immediate `copy_if_misaligned` call in this cached callable,
whereas some other primals do. There is no recorded input address/storage offset
showing misalignment, and no evidence connecting this to the fault. Do not turn
this missing runtime fact into a finding or add unconditional copies. NVIDIA's
[PTX address contract](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#addresses-as-operands)
requires natural memory alignment; it does not establish which alignment error,
if any, occurred here.

Quantize allocates and launches on the caller's current stream; inspected GEMM
consumption also uses the current CUDA stream (`csrc/gemm/gemm.cu:1026`). No
private quantize stream, asynchronous callback, borrowed workspace, detached
raw-pointer output, or early release is evident in this local path. PyTorch's
[2.15 stream semantics](https://docs.pytorch.org/docs/2.15/notes/cuda.html#cuda-streams)
require ordering and allocator lifetime protection for cross-stream use. This
review found no such violation inside the quantizer; it does not certify FSDP
input production, prior custom operations or unrelated streams. The current
completion-prefix diagnostic intentionally perturbs scheduling and lifetime,
so nonreproduction there cannot clear those unresolved alternatives.

## Decision and limits

Outcome (c): source evidence does not select a repair. Do not edit installed
CudaCherry/framework code, disable FP8/SAC/compilation/HybridEP, or infer a
quantize origin from the observed launch stack. Consume the already authorized
single native diagnostic and exact mapped quantize/GEMM receipts. A future
source decision needs concrete evidence of a violated contract or a narrowed
fault interval; no undirected follow-up experiment is proposed here.

Verified evidence is static source, cached-call metadata/order, source hashes
and existing failure records. Actual quantize provider/build origin, executed
machine code, originating PC/kernel, live input address/lifetime and recurrent
fault repair remain unverified. No GPU correctness, timing, memory-peak or
speed claim is made.
