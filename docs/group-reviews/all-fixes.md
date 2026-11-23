# Combined final fixes

## 10+10 results

Each pair ran compiled Original A (`e9b7cdbb4`, CPU prefetch depth 0) followed
by `final-all-fix` (`5e3f2ed52` plus the LM-head chunk update now recorded in
`3783bf33e`) on the same eight H100s on SR008. All runs used the 100B crop, `data-002` with
the production tokenizer, dense FSDP8, selective activation checkpointing
and MTP1. C2 and C8 denote the number of loss chunks.

| Topology | Chunks | Global TPS, baseline → all fixes | TPS Δ | Peak total GPU memory, baseline → all fixes (GiB/GPU) | Memory Δ (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| EP8 / eFSDP1 | 2 | 155,886 → 179,715 | +15.29% | 72.30 → 62.41 | −9.89 |
| EP4 / eFSDP2 | 2 | 146,965 → 170,613 | +16.09% | 76.02 → 64.61 | −11.41 |
| EP8 / eFSDP1 | 8 | 151,163 → 178,564 | +18.13% | 68.85 → 62.43 | −6.42 |
| EP4 / eFSDP2 | 8 | 143,601 → 174,680 | +21.64% | 72.52 → 64.25 | −8.27 |

Window TPS divides 20,971,520 physical tokens by drained all-rank elapsed
time over unprofiled updates 11–20. Memory is the maximum native GIFT total
across the same updates and all eight ranks. Each arm ran once.

## How the fixes fit together

- **Input and routing.** V2 CPU prefetch fills a 32-batch queue. HybridEP
  uses the GPU capacity selector in both GDN and MLA MoE blocks.
  [Prefetch](prefetch.md) · [HybridEP](hybridep.md#gpu-capacity-selector)
- **Attention.** The native FLA scan uses direct GDN backward, fused short-conv
  backward and fused output norm → FP8 with its direct backward. MLA packs
  Q/K during RoPE. [FLA and MLA](fla.md#changes-under-review)
- **Dense GFSDP and embeddings.** `dg-n4c` splits layer-0 and MTP FFNs into
  their own GFSDP groups. LM-head backward prefetches both MTP groups.
  Release events allow the last trunk dense group and final norm to reuse the MTP
  FFN's BF16 weight and FP32 gradient regions, respectively. Input embeddings
  exchange compact BF16 rows on a dedicated four-channel NCCL communicator
  and accumulate gradients in FP32 at the row owners. [Dense GFSDP](dense-gfsdp.md#changes-under-review)
- **Expert GFSDP.** Both topologies cache FP8 weights and retain tail groups
  across intermediate LM-head backwards. EP8 writes BF16-rounded expert `dW`
  straight into the FP32 slot. EP4 writes into a compact BF16 slot, exchanges
  with its peer and accumulates in FP32 at the owner. The deferred reuse
  wait precedes the first slot write, including EP4 zeroing. [Expert GFSDP](expert-gfsdp.md#changes-under-review)
- **LM-head and MTP.** One BF16 GEMM per chunk produces main and MTP1 logits.
  The `cuda-13` CE kernel applies each head's scale in FP32 and overwrites logits with
  BF16 `dLogits`. cuBLASLt writes `dW` directly into the FP32 GFSDP slot with
  β=0 on the first chunk and β=1 thereafter. C2 uses plan `9.14`. C8 and C32
  use `5.5`. Each BF16 `dH` chunk fills its own slice of the BF16 buffer passed
  to model backward. The recipe includes fused hidden z-loss and regular BF16
  weight all-gather. [LM-head and MTP](lm-head-mtp.md#final-fused-ce-acceptance)
- **Shared SM90 kernels.** Dense FP8 weight-tail quantization and the four
  CuTe norm optimizations cover forward, SAC replay and backward. The routed
  BF16 overwrite is a fallback for calls without a direct expert `dW` slot.
  [Shared SM90 kernels](shared-dm90-kernels.md#overlap-with-the-other-optimization-groups)

The [evidence index](../../runs/final-all-fix/artifact-index.json) records
per-pair results and checksums for all eight separate 10+1 captures. Each
capture covers the first three microbatches of update 11 on ranks 0 and 1.
The raw captures are in the local `artifacts/all-fix-traces` directory.
