# Native diagnostic attempt 002: fault attributed to DeepGEMM

## Result

The native graph-off 200-update diagnostic failed during update 36 after **35 completed optimizer updates on each of eight ranks**. The completed CUDA exception core attributes the exception instruction to a DeepGEMM K-grouped FP8 GEMM kernel. This establishes the faulting kernel and instruction, not the first corrupting operation or a repair.

The controller exited naturally with code 1 at 2026-11-12 15:34:05.597838 UTC. The owner recorded all 696 process identities absent; the eight worker PIDs and controller were also absent when checked. No retry, training-source change, live debugger attachment, launch blocking, or new GPU execution was performed by this analysis.

## Acceptance and actual configuration

- Exact native preset/import chain and resolved configuration reviewed: gift_moe_3b_hybridep_fp8_compile.yaml, 3B MoE, HybridEP EP8/FSDP2, FlexAttention, selective AC, model and loss compilation, original FP8 settings, local/global batch 8/256, sequence 4096, accumulation 4, seed 42; stop 200 and scheduler 30000.
- Checkpoint, evaluation and profiler remained disabled for this diagnostic. No observer, custom timing, step marker or memory monitor was added. Native structured logs and native metrics remained active.
- All eight recorded worker environments contained the four exact core-capture settings, graph0 and mixorder0. All eight native logs proved FP8 projections, HybridEP buffer, model/loss compilation, selective AC, fully_shard, resolved mesh degrees and accumulation4. All eight runtime manifests matched the existing bound-corrected HybridEP header SHA256 ff3a22e1c4e15785d9beca011a457f138dd1ce6502c7240ce6432d0f0985daba.
- Inherited warnings remain explicit: Inductor complex-op fallback for RoPE, checkpoint context warning, record_function ignored and quack disk-cache loader fallback to in-process compilation. These did not change the selected configuration.

See [configuration and coverage checks](configuration-capture-checks.json) (33 passing checks), [native resolved configuration](../../attempts/002/native/config/resolved_config.yaml), [imports](../../attempts/002/native/config/imports.yaml) and [recorded worker capture coverage](../../attempts/002/live-rank-capture-coverage.json). Hardware/runtime/source authorization evidence is retained in the coordinator's release and reconciliation receipts; this analysis did not reassign hardware.

## Completion evidence

All eight JSONL files parsed completely, with unique strictly increasing sequence IDs. Each rank has successful step and optimizer spans for exactly steps 1–35, each with four successful fwd_bwd spans: **140 successful microbatches per rank**. Each rank entered update36. Rank3 emitted fwd_bwd_error and step_error, followed by their normal end records; those end records are rejected as successful completion. Other ranks stopped inside the first microbatch of update36, without an optimizer span for 36.

Native rank0 metrics at steps 1,10,20,30 had finite loss and gradient norm. Loss values were 11.63658, 10.55154, 9.56865 and 9.32132; corresponding gradient norms were 28.2154, 15.8034, 9.9 and 9.2464. Native token metrics were finite (local131072/global1048576). This is finite observed logging, without a new numerical-parity or quality acceptance gate.

[Completion evidence](completion-evidence.json) retains per-step span counts, all errors, last events, native metrics and terminal receipts. Native logging confirms failure during36; successful host spans are native completion evidence and do not prove that no preceding asynchronous device error existed.

## Core and instruction

Rank3/PID74585 generated the only core, 343873935 bytes. Its stderr reports `Writing out global memory (0 bytes)`, `Writing done`, then `All done` at15:33:44.917006 UTC. The filename timestamp is not used to order first-error time. Peer cleanup happened after generation completed. The completed core opens successfully offline.

| Field | Observed value |
| --- | --- |
| Exception | Warp Illegal Instruction Parameter |
| Rank/device | 3/3 |
| Grid / block / thread | 578088 / (82,0,0) / (257,0,0) |
| SM / warp / lane | 64 / 8 / 1 |
| Triggered exception PC | 0x7fd81aff1cc0 |
| Function offset | +20416 (0x4fc0) |
| Triggered instruction | UTMALDG.2D.MULTICAST [UR4][UR8],UR10,desc[UR14] |
| Focused lane current PC | 0x7fd81aff2910, +23568, EXIT |

The exception PC and focused current lane PC differ. The `EXIT` at the current PC is not the attributed fault instruction. The full triggered specialization is:

```text
deep_gemm::sm90_fp8_gemm_1d1d_impl<1536u,1024u,0u,4u,128u,152u,128u,128u,128u,5u,128u,256u,2u,false,116u,(deep_gemm::GemmType)3,true,cutlass::bfloat16_t>
```

The exact mangled template occurs in retained DeepGEMM JIT cache `kernel.sm90_fp8_gemm_1d1d.500d411276c82c1ae1681a9672269982`. Its kernel.cu names `GemmType::KGroupedContiguous`; metadata specifies M1536/N1024/K128/groups4, accumulation=true, block128/152/128, stages5, multicast2 on B, 224472 shared bytes and384 threads. Metadata selects the installed deep_gemm include `deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh`.

The relocated ELF was copied from the debugger's extracted module before debugger exit as [fault-module.elf](fault-module.elf). Its hash differs from the cached cubin; this report establishes exact-template association and preserves both identities, without claiming byte identity between relocated and cache images. [Binary identities](binary-identities.json) records core, relocated ELF, cached cubin, kernel.cu and kernel_meta hashes.

[Essential offline decode](decode.log) and [triggered-PC SASS decode](decode-trigger.log) retain commands and output. Both debugger invocations exited0. No omitted tensor/global/shared memory was read; ordinary ELF instruction decoding was used without `/r`. Descriptor values and first-corruption causality remain unestablished. The NCCL watchdog's illegal-instruction report is a later host reporting site and does not select NCCL as the cause.

## Remaining gate and handoff

The first-kernel localization question is resolved. The next owner must inspect this exact KGroupedContiguous specialization and TMA multicast instruction, its descriptor/scheduler/producer contracts and applicable pinned fixes before selecting a causal repair. Required later validation remains the agreed causal fix, affected-path verification and two fresh native200 passes after the fix. This failed diagnostic is not a post-fix pass. No performance, numerical-parity, convergence or checkpoint-continuation claim is made.

This evidence-only analysis has no production-runtime impact. Core capture itself was a diagnostic runtime setting; its overhead is unmeasured, so elapsed-time comparisons are not used.
