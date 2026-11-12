# Selected fused HybridEP device fault sites

## Result

No reachable explicit device assertion, trap or timeout was found in the inspected single-node fused HybridEP backend source and its direct project device helper, `utils.cuh`. All 15 runtime device assertions in the backend belong to multi-node RDMA code excluded by the pinned one-node JIT compiler route. This does not rule out a generated illegal instruction, memory corruption, an invalid asynchronous-operation contract, or a fault in another module. It selects no source fix or additional whole-workload retry.

Completed finite checklist: establish selected native call/template route; locate explicit device fault sites and their guards; compare concrete conditions with retained runtime evidence; select a justified action or state that none follows. The `code-rules` and `perf-experiments` skills informed this bounded review. Only this artifact directory was written. No imports, compiler invocation, CUDA/GPU query, launch, install, source/library edit or cleanup occurred. This documentation adds no production runtime work.

## Selected call chain and pins

Existing [primary-source research](../../BASELINE-AA/fault-research/001/handoff.md) establishes installed DeepEP `1.2.1+1b8f467`, the byte-identical upstream backend, pinned JIT compiler and native binding. Reused primary source is [DeepEP compiler at 1b8f467](https://github.com/deepseek-ai/DeepEP/blob/1b8f467/csrc/hybrid_ep/jit/compiler.cu). No new upgrade research was needed.

The task entry `train_scan_fix.py` calls `install_hybrid_ep_scan_fix` before native training. `backend_scan_fix.py` pins the installed wrapper/header SHA-256, stages an unchanged wrapper and helper tree, corrects only the scan-array bound, then selects that relocated wrapper at the GIFT buffer factory. The unchanged wrapper passes its own directory as native `base_path`. Both task adapter files match snapshot 009's manifest. Inspected source and retained runtime-file hashes are in [source-receipts.json](source-receipts.json).

GIFT `torchtitan/distributed/deepep/hybridep.py` routes:

- `_dispatch_impl` → `_buffer.dispatch_with_permute`, FP8 row data/scales, routing map/probabilities, pad 128 and fused permutation; metadata preprocessing runs on a new handle.
- `_combine_impl` → `combine_with_unpermute` without probabilities.
- Backward of combine, `_dispatch_bwd_impl` → FP8 row quantization → `dispatch_with_permute`, reusing the handle.
- Backward of dispatch, `_combine_bwd_impl:631–665` → contiguous token gradient and FP32 probabilities → `combine_with_unpermute`, reusing the handle. Native `with_probs=True` selects the backward-combine channel. This is the completion diagnostic's selected custom-op body.

The selected communication-stream helper waits for prior compute, records input use on communication, invokes the native callable there, joins it into compute and records output use on compute. This establishes the Python stream contract; it does not independently attest all installed native internals.

Retained completion-probe rank-2 runtime manifest identifies wrapper SHA-256 `63e425916b3c6b0c881ad402af35a2b00c4c84203fcde611a6715744b89210cb` and corrected backend `ff3a22e1c4e15785d9beca011a457f138dd1ce6502c7240ce6432d0f0985daba`. Byte comparison verifies the sole bound correction relative to installed `b507f65cfa4580f16342076fc2d00afe81e43f64ddefcc4a3cf43681aef11b4b`; wrapper and `utils.cuh` remain byte-identical. The retained same-process cache keys decode using the pinned compiler's key producers as follows:

| Kernel | Selected configuration |
| --- | --- |
| Common | hidden 1536, max tokens/rank 32768, four experts/rank, eight ranks/node, one node, fusion true, nonblocking false |
| Scan | pad 128, chunk 64, 256 threads, 108 blocks |
| Dispatch | uint8, stages 10/10, in-flight 8/8, pad 128, additional in-flight 6, chunk 64, 16 dispatch + 108 permute blocks, forward true and false, device sync true |
| Combine | stages G2S/S2G 10/2, unpermute stages 2/2, chunk 64, token group 4, 16 combine + 108 unpermute blocks, additional in-flight 1/1, backward false and true, device sync true |

The JIT compiler selects `intra_node_flags` for `num_of_nodes == 1` (`compiler.cu:142–152`). Those flags lack `HYBRID_EP_BUILD_MULTINODE_ENABLE`; fusion adds only its own macro, while token dropping requires fused **and nonblocking**, false here. Single-node combine uses two data pipelines; backward enables probability handling. Cache keys and staged source are strong route evidence, not proof of every loaded instruction or native extension build source. No new binary attestation requirement is introduced.

## Explicit sites and relevant invariants

| Site/condition | Reachability and existing evidence |
| --- | --- |
| NIXL transfer statuses must equal SUCCESS/IN_PROG; atomic statuses must be nonnegative (`backend:758–1006`, 12 assertions) | All guarded by MULTINODE + USE_NIXL. Excluded on selected one-node route. No matching NIXL device-status evidence. |
| DOCA completion-queue poll status must be nonnegative (`:1187`, `:2873`, `:4227`) | All guarded by MULTINODE and non-NIXL. Excluded. This includes `rdma_sync_kernel`; its name does not make it active on NVLink-only execution. |
| Scan local-expert sum array (`:5128`) | Previously concrete four-expert out-of-bounds bound is corrected in retained runtime bytes. Recurrence with this correction prevents treating it as the established repair. No second demonstrated bound defect is identified here. |
| Device barrier (`:4145–4174`) | Reachable because device sync true. Each parity's expected uint32 value increases by eight; polls require equality with the shared flag. No deadline or explicit trap exists. At 1385 calls the count is nowhere near uint32 wrap; no observed flag values demonstrate divergence. |
| Fused chunk flags and shared mbarriers | Reachable waits require producer notification/expected parity (`:3659`, `:3942`, `:3888–4136`). Two-pipeline unpermute notifications increment each target chunk twice; expected unpermute value increments by two (`:5818`). These loops can fail to progress if a producer fails or metadata/order is invalid, but have no explicit timeout/trap. No captured flag/parity or originating PC demonstrates a violated invariant. |

The entire backend and direct project device helper contain no explicit `trap`, breakpoint, device `abort`, clock-based timeout or timeout error branch. `static_assert` checks reject compilation and cannot explain a delayed runtime assertion. Wrapper Python assertions and host `CUDA_CHECK` reporting/abort are different mechanisms; `cudaGetLastError` is not a completion wait. Compiler/CUDA-header-generated fault behavior is outside this source-level absence claim.

## Decision and remaining uncertainty

The [existing completion-prefix assessment](../004/handoff.md) remains applicable: all ranks completed through ordinal 1384; ordinal 1385 entered without recorded completion, and rank 6's event wait reported CUDA 715. That boundary includes intervening recomputation/backward and joined communication. It does not isolate combine or identify the originating kernel. A stalled peer can explain missing completion elsewhere without eight origins.

The recurrent graph-OFF failures at updates 22/33/48, asynchronous quantizer reports, and the single sanitizer attempt's 20-minute timeout after 16 updates remain failures/inconclusive attribution. Capped NCCL-init reports do not prove full diagnostic coverage or clear the fault. Earlier silent corruption remains possible.

**No further diagnostic or scoped source repair is justified by these explicit fault sites.** In particular, a device-timeout assertion patch would add scheduling/register costs while testing an unobserved flag violation, and another sanitizer or workload retry would repeat unresolved attribution. Preserve current failures and the separate one-update numerical probe's scope. A concrete future source action requires an originating PC/kernel or observed metadata/flag violation; none is supplied by this review. Installed source-to-binary correspondence and the fault origin remain honest unknowns.
