# DeepGEMM B-descriptor fault: bounded causal assessment

## Result and acceptance

The fault is a **B-operand TMA multicast load in expert w2 weight-gradient computation**. No causal repair is established. Two concrete descriptor-publication/lifetime defects are reachable in this installed fork, but neither is demonstrated on faulting CTA82 with four positive groups. The next useful action is the exact-template standalone fixture below, without changing training or installing/upgrading packages.

Completed: map the actual ELF instruction to operand and call; inspect its scheduler and descriptor helpers; compare pinned upstream repairs; inspect retained registers; specify a finite fixture and conditional minimal repairs. No GPU work, training-source edits, dependency writes or process actions occurred. Documentation adds no production runtime cost. Caller/scratch lifetime is separately reviewed in [caller review](../../caller-review/001/REPORT.md).

## Identity and selected path

Repository base `446431dfa191ef0300f116257ae918bc85e38c06`, dirty main, Source010 manifest SHA256 `b2e9144af50e8673e0b8646a4f03f629bc240e6fc3ed1aaccfe6814491fb7d20`; no reviewed source was changed. Reuse the exact preset and runtime configuration in [attempt analysis](../../analysis/002/REPORT.md): 3B MoE, HybridEP EP8/FSDP2, dense/sparse FP8, Flex, SAC, model/loss compile, chunked CE, local/global batch8/256, S4096, accumulation4, graph0/mixorder0. Native002 completed35 updates on all8 ranks, then failed during36.

Runtime pins: Torch `2.15.0.dev20260822+cu130`, git `8b353065ce260336c5fdfa74d754e8bd9b91b062`; DeepGEMM `2.3.0+f093a4c`; Triton3.8.0; DeepEP `1.2.1+1b8f467`; CudaCherry0.1.0. Installed `_build.py` declares f093a4c; package provenance is a locally built wheel, not a verified public upstream revision. [Source receipts](source-receipts.json) pin inspected installed headers, extension, and GIFT wrappers.

The [completed core/module receipts](../../analysis/002/binary-identities.json) pin core SHA256 `1e9b5a60a5ba7d550ed07c8b10205b07be9f7693a375d17bf0acfb4613cb6c71`, relocated ELF `2b2ea175760f2de67139c11c61bcc40409c798a0c413a7701f208de0f990a224`, cache cubin `f4bbbae842de6bf676e66bf846713dbe626d9c5a08434c8e993aca15ad0a1bdf`. The exact template is associated with cache key `500d411276c82c1ae1681a9672269982`; relocated/cache byte identity is not claimed.

`GroupedExperts._fp8_experts_forward` calls handle2 with w2 shaped `(4,1536,1024)`. `GroupedGemmFp8Wrapper.backward` converts dY and SwiGLU output into packed column layouts, allocates BF16 wgrad, and invokes `_DeepGemmHandle.k_grouped_fp8_gemm_contiguous`. Its CUDA custom op calls installed `deep_gemm.k_grouped_fp8_gemm_nt_contiguous((a_data,a_scale),(b_data,b_scale),d,ks,ks_tensor,d)`. Thus M1536/N1024 identifies w2 wgrad; w13 wgrad is M2048/N1536. Accumulation=true follows `c=d`, initialized to zero. This maps the kernel family, not its layer or microbatch ordinal.

The exact generated kernel is KGroupedContiguous, M1536/N1024/Kdynamic, groups4, block128/152/128, stages5, 384 threads, grid116, two-CTA multicast on B, BF16 CD and accumulation=true. Cache metadata K128 is the compilation representative, **not the failed invocation's observed counts**.

## Instruction and retained operands

Triggered PC `0x7fd81aff1cc0`, function+`0x4fc0`, is `UTMALDG.2D.MULTICAST [UR4],[UR8],UR10,desc[UR14]`. Grid578088, block82, thread257, SM64, warp8, lane1; cluster41 has two CTAs. [Selected actual-ELF SASS](selected-sass.txt) shows the preceding addition of `0x1da00`: exactly `512 + 38912 + 5*16384`, the kernel's B staging base. This selects the dynamic B tensor map, not scales, output, or a WGMMA instruction.

Offline CUDA-GDB `info cuda lanes; info registers`, then `cuda lane 0; info registers`, found producer lane0/thread256 divergent at +24272, waiting on an empty barrier; lanes1–31 are at +23568/EXIT. The retained uniform registers include:

| Register | Snapshot value |
| --- | --- |
| UR4 / UR5 | `0x30e00` / `0x370a0` |
| UR8:UR9 | `0x0000000a03d58900` |
| UR10 | `0x30000` |
| UR14:UR15 | `0x1000000000000000` |

These are **current-state values, not proven trigger-time operands**. With the compiled shared base `0x400`, UR4 matches B stage4 (`0x400+0x1da00+4*0x4c00`); UR5 matches full barrier4 (`0x400+0x36c80+4*8`). Descriptor pointer is128B aligned and multicast mask encoded by the adjacent instruction is3. No invalid operand is demonstrated. Shared memory requires128B alignment; lack of1024B phase alignment alone is not invalid ([NVIDIA swizzle contract](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#the-swizzle-modes)). Global/shared/constant payloads were omitted, so the actual tensor map and ks cannot be reconstructed from this core. No corrupted descriptor bytes are asserted.

## Reachable source defects and causal limits

Installed `sm90_fp8_gemm_1d1d.cuh:185–245` has two A and two B descriptor slots per CTA. It prepares the next group, repairs skipped-prefetch mismatches, and acquires current maps. Actual ELF has generic `STG.E.128` publication at +0x3020–0x32d0 and +0x3480–0x3790, release fences at +0x37a0, then map invalidation/acquire at +0x3900/+0x3940. No descriptor-read drain precedes publication.

For four positive groups there are `ceil(1536/128)*ceil(1024/152)=84` tiles/group. Persistent scheduling uses `iteration*116+blockIdx.x`:

| CTA | Scheduled groups | Descriptor consequence |
| --- | --- | --- |
| 0–19 | 0→1→2 | Preparing3 overwrites slot1 previously used by1 without a completion guarantee. |
| 20–51 | 0→1→3 | Corrective final-group3 publication has no following release, because no next group exists. |
| 82 | 0→2→3 | Group0 uses by-value base. Group2 first publishes slot0, also prepares3/slot1 and releases both; group3 uses that prefetch. Neither defect above is demonstrated here. |

The initial hypothesis that CTA82 overwrites a live group0 slot0 was **wrong**: group0 uses the by-value base. It is withdrawn. If an expert count is zero, scheduling can make CTA82 jump directly to a final third nonempty group; that could reach the missing-release branch. Failed ks are unavailable, so this remains conditional, not a claim that production had an empty expert.

[Upstream #343](https://github.com/deepseek-ai/DeepGEMM/pull/343), merged commit `1f6f3f378920ccb5cc036ef43eb3f5972e921713`, drains descriptor reads before GMEM overwrite and uses a single-lane `__syncwarp` to stop ptxas store reordering. The installed two-slot prefetch fork differs from its single-slot current-group implementation. [Open #375](https://github.com/deepseek-ai/DeepGEMM/pull/375), pinned head `4e4e54036becc2a1ee9b5470015566673045acfd`, adds a pre-modification drain; it is **not merged or established for this fork**. Its claim that tensormap.replace joins the same bulk async group is not directly established by the current PTX contract, which lists bulk copy/reduce operations. Do not promote that explanation from an open PR into a verified mechanism. Both patches/API status are archived here.

Smallest conditional changes, only after a fixture establishes the affected failure:

- Missing publication release: place `tensor_map_release_cta()` immediately after the corrective current-map A/B stores, before acquire. One CTA-scoped publication fence per corrective mismatch; no host wait, new storage, collective or precision change.
- In-flight descriptor overwrite: port #343's descriptor-read completion plus scheduling barrier **before every reused GMEM descriptor publication**, including both fork branches. Keep two slots, scheduler and arithmetic. Review actual generated SASS ordering. This can wait at group transitions, up to a few transitions per CTA/GEMM; magnitude is unknown. It is not per-token or process-wide serialization.
- Enlarging to pipeline-indexed descriptor slots changes the host scratch-layout contract and memory (currently4*128 bytes/CTA). No maintained matching implementation was found in this bounded search; it is a larger, unselected change.

Neither candidate is a production fix yet. Hardware, bad runtime counts, earlier corruption and compiler/driver behavior remain unexcluded. The independent host review found same-current-stream scratch allocation/launch/release and no specific lifetime defect.

## Next decisive experiment: exact-template baseline fixture

Implement only a task-owned fixture and, if later selected, an isolated copy of this exact installed dependency. Preserve installed package bytes. No training framework, tensor recorder, CP/RNG changes, broad synchronization or new dependency. One authorized H100, set116 SMs, native current stream, graphs off. Hold explicit operand references for the fixture lifetime. This isolates inner-kernel behavior; it does not test communication/SAC/autograd lifetimes.

Use the **actual GIFT custom op** with `d=(4,1536,1024)` BF16 zeros, `ks` a Python integer list, `ks_tensor` the identical CUDA int32 values, `ks_cpu=None`, A/B FP8 e4m3fn packed group-by-group as contiguous `[M,k_g]` / `[N,k_g]` flattened to1D. Scales are FP32 views of contiguous `[sum(ks)/128,M]` / `[sum(ks)/128,N]` buffers transposed to `[M,blocks]` / `[N,blocks]`, matching the GIFT row2col consumer layout. `c=d` remains selected through the unchanged custom op.

First admissibility call for each case must produce the exact template above (verify generated kernel.cu/meta and successful synchronization **outside** its async body). A different template makes that case inconclusive for this fault; do not silently alter block sizes/SM budget to force a pass.

Fixed cases and maximum reach, stopping immediately on an exception or wrong result:

1. `ks=[128,128,128,128]`,1024 calls: short pipelines expose descriptor-transition ordering.
2. `ks=[128,128,128,0]`,1024 calls: legal empty final expert makes CTA82's final third nonempty-group corrective-publication path reachable.
3. `ks=[32768,32768,32768,32768]`,64 calls: production-scale K and the same template; not claimed to equal failed counts.

Use deterministic manually constructed FP8 values0/±1, unit scales, and at most128 nonzero K positions per dot product, with distinct per-group/channel signs/support. Expected BF16 integer results are exactly representable; compute that explicit oracle and check elementwise values, including zero group's unchanged output. This tests wrong-group/stride reads without inventing tolerance or requiring training-loss comparisons. Reset d to its documented initial state between calls on the same stream. Keep launches asynchronous in fixed chunks; synchronize/check at chunk boundaries, not inside the kernel. Cap the complete fixture at120 seconds and2112 GEMM calls, with no automatic retry. Save case/launch/chunk, source/template identities, and exception core; no timing/speed claim.

Use the already validated native exception settings: `CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1`, flags `skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory,skip_abort`, `CUDA_COREDUMP_SHOW_PROGRESS=1`, and a fresh task-owned `CUDA_COREDUMP_FILE` with `%p.%t`. These omit descriptor contents; any later need for small metadata should be addressed explicitly in the fixture, not by claiming this core contains it.

**Decision:** an unmodified-fixture fault/wrong result at this template permits a matched one-cause candidate comparison. Empty-group-only failure selects the release placement first; descriptor-reuse failure with positive groups selects #343 adaptation. Require original fail and candidate pass with the same inputs/limits, inspect resulting SASS, then review cost. A clean fixture only bounds nonreproduction; it selects no production patch or whole-training pass. Actual integration and two fresh native200 passes remain open after a causal fix. No GPU validation was performed in this assessment.
