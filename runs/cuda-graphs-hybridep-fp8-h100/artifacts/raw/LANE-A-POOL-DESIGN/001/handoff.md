# Lane A003: first dense block graph-pool ownership diagnosis

## Result and finite acceptance

**The first failing region is dense decoder block 0, not HybridEP.** Its compiled
forward finishes Trees warmup, then the allocator guard detects one persistent
allocation absent from tracked outputs. CudaCherry's stream-keyed FP8 GEMM
workspace is a specific source-supported suspect. The failed pointer's owner is
not yet proved: A003 saved no allocation history or block size. Select **one
bounded allocation-ownership diagnostic**, preserving the guard, before choosing
an implementation. This is not evidence that CUDA Graphs are universally unsafe.

Acceptance for this read-only design: identify the first failing region and guard;
trace plausible persistent producers and consumers there; inspect installed API,
stream, and binary provenance plus established remedies; specify one diagnostic
with decisions and finite subsequent gates. These four items are complete.
No source/package changes, GPU queries/initialization/runs, installation, remote
operations, or delegation were performed. Only this evidence directory was written.

## Frozen inputs and observed failure

Repository reported HEAD `446431dfa191ef0300f116257ae918bc85e38c06`, source003
archive SHA-256 `ea215abaac14d98a21c36e8684dbd8ce88a1d48f47cec5fe4f85f61be5dd4e27`.
[Source pins](source-pins.json) identify 22 inspected source/artifact files.
The exact target remains 3B MoE, one dense plus 17 MoE blocks, EP8/FSDP2,
FlexAttention, FP8, selective AC, model/loss compile, batch 8/global 256, S4096,
four accumulation microbatches, schedule horizon 30000, seed 42. No attention,
fusion, checkpoint, RNG or trainer-policy change is proposed here.

[Rank-4 full stack](first-error.json) is retained without truncation. All eight
[rank error records](rank-errors.json) report the same pool-guard class with
second-resolution timestamp `1794439338`. Rank 5's traceback appears first in
supervisor output order; rank 4 is one complete example, not an established
chronologically first failure. Its pointer is `597168226304`.
Rank-4 stderr records mask `[0/0]` INFERENCE warmup at 23:22:15.638, then decoder
`[1/0]` FORWARD warmup at 23:22:17.439. The old invalid mask-storage exception is
absent at this boundary. The new stack reaches post-model `CUDAWarmupNode.run`
line 844; it is not a replay, backward, or optimizer failure.

The stack's actual generated code is
`NATIVE-SMOKE/001/cache/torchinductor/rp/crpavfztke5wmvkq4ztm2pl57hvh5qbm5ynacsqaesxygi2vdth7.py`.
Its `Runner.call:1374` invokes `partition_0:1143`. That partition contains MLA
attention and four dense FP8 GEMMs: output projection (line 1268), FFN gate/up
(lines 1299/1314), FFN down (1342), plus quantization and fused SwiGLU. It has no
HybridEP/DeepEP/grouped GEMM dispatch. Resolved configuration lists `mla_dense`
repeat 1 before `mla_moe` repeat 17 (lines 153–157). Compiled shapes include dense
FFN 1536→6144→1536, corroborating dense block 0. The partition returns normally
before the pool check; its warmup operators executed, but no useful CUDA Graph
capture/replay or native backward/update is established.

## Actual allocator and candidate owner

Installed torch is `2.15.0.dev20260822+cu130`, reported revision
`8b353065ce260336c5fdfa74d754e8bd9b91b062`; installed file hashes, rather than
revision alone, govern this analysis. Trees behavior:

- `run_eager:2710` creates a `CUDAWarmupNode`, sets `current_node` to that node and
  `path_state=WARMUP`, then invokes it. Mode here is FORWARD, function partition_0.
- `_use_cuda_memory_pool_manager:655` switches to the manager stream and directs
  current-thread allocations into its private graph pool, without stream capture.
- `CUDAWarmupNode.run:805` executes the wrapped compiled model there, records
  output-storage weak references, then `check_memory_pool:2140` checks all live
  allocations against tracked path storages. It retries after GC and synchronization
  before reporting the extra block. This is a correctness check, not a skip hint.
- `CUDAGraphTreeManager:2315` creates its own `torch.cuda.Stream()`. Warmup and
  recording both use this dedicated `self.stream`, not ordinary eager compute's
  current stream. Ordinary default-stream prewarm cannot seed a stream-keyed cache.

The active producer/consumer chain is native dense `FP8Linear`/
`_Float8LinearFunction.forward` in `ops/float8/cuda_gemm.py:139` → `_cudacherry.py`
→ installed Python GEMM wrapper → `torch.ops.cudacherry.gemm.default` → C++ CUDA
GEMM → `get_cached_workspace` → `cublasLtMatmul`. The compiled calls use functional
`out=None`, `use_split_accumulator=True`, and workspace size 33554432. No residual
out-schema mutation is active in these four generated calls.

Installed `cudacherry/csrc/gemm/gemm.cu:234–276` has a process-lifetime
`static thread_local` heap map from `(device, stream)` to `at::Tensor`.
A miss allocates `at::empty({workspace_size + 256}, uint8)` and keeps the tensor in
that map. The extra bytes allow `aligned_workspace_ptr` alignment. All four calls
on one thread/device/stream can share one cached allocation. The returned GEMM
output and Meta kernel expose only the result, not this scratch tensor. Its
`ExcludeDispatchKeyGuard` suppresses Python dispatch visibility but does not
redirect the CUDA allocator. The caller uses the workspace in cuBLASLt and drops
its temporary reference; the map remains its strong owner.

The existing `check_not_capturing` tests `cudaStreamIsCapturing` only on a miss.
That rejects lazy allocation during capture, but permits Trees private-pool
warmup, where the stream is not capturing. This explains how one hidden live
workspace can survive the return and fail the guard. Shipped quantization/SwiGLU
sources on this region did not reveal another persistent tensor cache. Native
cuBLAS workspaces remain an alternative: Trees clears its own cuBLAS workspace
manager around warmup; that cleanup does not clear CudaCherry's independent map.
The one-block count fits the CudaCherry hypothesis; it does not prove the address
belongs to it. C++ retained tensors also need not appear in `gc.get_objects()`.

## Installed wrapper/binary provenance

CudaCherry metadata says version 0.1.0 and installation from local
`file:///opt/cudacherry`; that directory is absent in this environment and no Git
revision/build attestation is supplied. `_loader.load_family:133–169` normally
loads the existing prebuilt family unless FORCE_JIT is enabled; A003 logs contain
no JIT-build message. The actual extension candidate is
`cudacherry/gemm/_C_gemm.cpython-312-x86_64-linux-gnu.so`.
Its hash and shipped source hashes match installed wheel RECORD entries, and its
symbols/strings include the workspace Tensor map, WorkspaceKey and capture guard.
This supports the shipped-source explanation but does not certify a reproducible
source-to-binary build or a runtime module path absent an execution receipt.
The next diagnostic may record loaded GEMM module `__file__` and its hash only if
the existing observation route exposes it without adding a harness; absent that,
loaded-module provenance remains explicit. Do not set FORCE_JIT or install/rebuild for diagnosis.

## Established remedies and choice

The [PyTorch 2.15 Trees documentation](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html#cudagraph-unsafe-custom-ops)
says opaque custom ops are assumed capture-safe unless tagged unsafe, and explains
partitioning. The installed `utils.is_cudagraph_unsafe_fx_node:4202` checks that tag;
`scheduler:9979` excludes such nodes from captured partitions. With partitioning
off, `compile_fx:1999` excludes the containing graph instead. Installed config
uses `TORCHINDUCTOR_GRAPH_PARTITION` (default on outside fbcode), and the failing
artifact already has a partition runner. Verify resolved policy before a change.
The [custom-op API](https://docs.pytorch.org/docs/2.15/library.html#torch.library.custom_op)
supports tags at registration; modifying a private `.tags`/manager object is not
the public integration route. The installed C++ registration currently has no
unsafe tag. No inspected public vendor API accepts an explicit Tensor workspace
or reserves it for a caller-provided stream; only `workspace_size` is exposed.

| Approach | Fit, cost, and decision |
| --- | --- |
| Allocation-history diagnostic | Selected next action: proves or rejects the exact ownership hypothesis with no computation/partition change. Adds diagnostic CPU stack/memory overhead; never use its timings for performance. |
| Ordinary eager prewarm | Reject as a fix: wrong stream key. Public prewarm cannot select Trees' internally created stream. Do not create/access a private manager merely to fetch that stream. |
| Explicit caller-owned persistent workspace | Plausible library fix after proof: eager-owned storage passed as an op input makes lifetime visible; requires a supported vendor API/schema and stream/concurrency ownership, resize/capture checks, Meta and autograd/SAC validation. It is not currently available and is not a trivial campaign callback. |
| Tag only actual GEMM unsafe, retain compile and partition | Public fallback after proof if vendor workspace integration is unavailable. Four dense GEMMs become uncaptured boundaries, as do other uses of the same op; more graph fragments/host dispatch may reduce benefit. Tag at supported registration, not by mutating private operator state. Requires separate source/package authorization and graph-off/on controls. Do not tag all FP8, attention, or HybridEP. |
| Change workspace_size to zero | Reject: changes GEMM algorithm availability/selection and is not an ownership repair. |
| Disable guard or patch Trees manager | Reject: masks the allocator correctness contract. |

This document has no production runtime cost. No speedup, numerical equivalence,
or production readiness is claimed. The former native71 illegal-instruction failure
at update33 remains separate; source002/003 marker reasoning does not explain it.

## One bounded ownership diagnostic, not executed

Unresolved question: **does the bad private-pool allocation originate in the
CudaCherry persistent workspace, or another owner?** Outcome changes the fix owner
and avoids a blind custom-op workaround.

Use one newly authorized graph-on diagnostic with exact frozen source003/preset,
public full-update marker, existing eight-rank runner and first-error stop. Preserve
all four microbatches and the full model. Before entering the existing native
entrypoint, set the installed config option
`torch._inductor.config.triton.cudagraph_trees_history_recording=True`.
No dedicated environment-variable binding exists for this option in the inspected
installed config: do not invent one. An inline launcher setup invoking the same
native module and arguments can set it without source edits or a new harness.
Do not disable `slow_path_cudagraph_asserts`.

This is the diagnostic directly suggested by the actual exception.
`get_history_recording:216` enables allocator history around warmup/recording,
using installed `_record_memory_history()` defaults (context all, stacks all).
`get_cudagraph_segments:2113` automatically takes `torch.cuda.memory_snapshot()`
on failure; `check_memory_pool` filters the bad pool/address and prints its Python/
C++ allocation frames in the original error. Thus the first check needs no new
snapshot wrapper, GC scan, package installation, or private-manager access.
The built-in path does **not** save a standalone snapshot or print block size;
do not claim otherwise. Preserve full per-rank error output as already done.
No added exception interception should replace or swallow the original error.

Source comments warn allocator history can affect cleanup; use this only for the
single early-failure diagnostic, never timing/parity evidence. The recording API's
default event history limit is sys.maxsize, so retain the existing bounded attempt
deadline and first-error stop, with the normal 25-update upper limit if it advances;
do not turn this into a long profiling run. Its context disables recording when
it exits if recording was initially disabled; the process then terminates normally.

Proof requires the guard's failing address and allocation frames naming the actual
producer (for example `get_cached_workspace`/`at::empty` inside CudaCherry).
One hidden block and the expected 33554688-byte request are corroboration, never
ownership proof. If C++ frames are unavailable or unsymbolized, explicitly report
that remaining gap; no automatic second diagnostic or new harness is authorized.

Decision and stop after this single diagnostic:

1. Workspace allocation origin plus matching address confirmed: request the narrow
   vendor workspace ownership integration, or supported GEMM unsafe-registration
   fallback if that integration is unavailable; present its concrete scope first.
2. Another allocation origin: assign only that demonstrated producer/owner for a
   scoped fix. Preserve the CudaCherry finding as a separate potential hazard.
3. Missing frames or changed failure: report the exact missing ownership evidence;
   do not claim the hypothesis verified, auto-retry, or broaden the kernel audit.

Before any implementation, CPU checks must cover the chosen real schema/Meta
contract, default-off integration, and unchanged full-update marker; a vendor
workspace change additionally needs cache key/resize, explicit ownership and
stream-concurrency contracts. CPU checks cannot establish private-pool safety.
After a fix, the finite runtime prerequisites are the same native graph0 control
and graph1 25-update diagnostic, first/second update generation transitions,
all ranks, four forwards/backwards per update, real optimizer events, capture/
replay and partition attribution, and matched memory. Keep later numerical,
checkpoint/continuation and performance acceptance separate. Establish A/A
variability and precision-based criteria before interpreting graph-off/on
numerics; independent-run bitwise loss/norm equality is diagnostic, not a gate.
No nondeterminism-elimination project, FlashAttention migration, or extra fusion
prerequisite is introduced.
