# Dense GFSDP and input embedding group review

Updated 22 November 2026. Against compiled Original A, the final package
raised throughput by **2.18%** and lowered peak total GPU memory by
**2.81 GiB** on EP8/eFSDP1. On EP4/eFSDP2 it raised throughput by **1.29%**
and lowered the peak by **3.49 GiB**. Both final arms combine the two dense
FFN splits, the MTP backward schedule, and row-owner embeddings with a
four-channel NCCL communicator and compact source gradients.
[EP8 results][ep8-results] · [EP4 results][ep4-results]

## Source branch and tested variants

The integration branch is `dense-gfsdp-fix` in `.worktrees/dense-gfsdp-fix`.
Its HEAD is the Original A commit `e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f`,
with GIFT tree `0f44b971069af55d6d573628129900483ca3e1dc`. Original A has the
two FFN splits and ordinary row-owner available as disabled options. The
integration worktree carries the scoped MTP transition port from `3860b1eb5`.
Both final training arms used `dg-n4c` on SR008, staged by applying the
archived `dg-n4c.patch` to a detached checkout at the same base revision.
The local counterpart is `.worktrees/dense-gfsdp-row-owner-nccl4-compact-001`.
[Reviewed port manifest][port-manifest] · [Staging script][stage] · [Final patch][patch]

| Variant | SR008 checkout | Staged GIFT digest prefix |
| --- | --- | --- |
| Compiled Original A | `dg-a` | `8450bbe7e8f2` |
| Four original changes, shared communicator | `dg-fix` | `da8fdaca51a3` |
| Dedicated four-channel communicator | `dg-n4` | `5fe3b16adb1b` |
| Dedicated eight-channel communicator | `dg-n8` | `cb2cb7f319ab` |
| Dedicated four-channel communicator and compact gradients | `dg-n4c` | `be51c5be2791` |

Both final run receipts pin GIFT source SHA-256
`be51c5be2791a042dbda2dfb8df015090e004634b777fa50026eb5331de429b6`.
The final patch SHA-256 is
`347faaa2c0deeb9975dba59d428645f9d366529367a61f20959177862ba7b648`.
[EP8 source receipt][ep8-source] · [EP4 source receipt][ep4-source] · [Bundle checksums][bundle]

## Changes under review

| Change | Original behavior | Final behavior and intent |
| --- | --- | --- |
| **Split `layers.0.feed_forward`** | One GFSDP group held attention, norms and the dense FFN. Gradient reduction waited for the backward boundary of the entire group. | Compile the block prefix, FFN and suffix separately and give the dense `w1/w2/w3` MLP its own GFSDP group. Its FP32 reduce-scatter can start while the prefix still runs backward. Both groups shard over the same eight ranks and use disjoint views of the existing pool. [Sharding][sharding] · [FFN boundaries][ffn] |
| **Split `mtp.layers.0.block.feed_forward`** | The MTP projection, attention, norms and dense FFN shared one GFSDP group. | Apply the same split to the MTP FFN. Projections, attention and internal norms stay in the parent. The FFN gets its own gradient-ready point and releases its BF16 weight and FP32 gradient regions separately. [Sharding][sharding] · [FFN boundaries][ffn] |
| **Prefetch MTP backward weights and reuse its buffers** | The LM head followed the default backward prefetch order. The last trunk dense group and trunk final norm used their standard slots. | The LM head requests both MTP groups earlier. After the MTP FFN releases its BF16 region, the dense group of `layers.7` gathers there. After the FFN releases its FP32 gradient staging, the trunk's `model.norm` packs its gradient there. Release events protect overlapping views on the same process group. [MTP schedule][sharding] · [Trace evidence][mtp-trace] |
| **Shard input embeddings by row owner** | GFSDP all-gathered the full BF16 input embedding table and reduce-scattered its full FP32 gradient. | Each owner stores full rows. Requesters send unique int64 IDs, receive BF16 rows, and return locally coalesced BF16 gradient rows. Owners accumulate the received contributions in FP32 in donor-rank order. The untied `model.lm_head` keeps its GFSDP path. [Row-owner module][owner] |
| **Isolate embedding NCCL traffic and cap it at four channels** | All four row-owner exchanges used the dense process group. First use of sparse P2P on that communicator allocated buffers for 32 channels per peer. | Route all four exchanges through a dedicated process group with the same rank order and `max_ctas=4`. NCCL INIT showed four embedding channels alongside 32 dense channels. [Process group][owner-nccl] · [Effective channels][channel-receipt] |
| **Coalesce source gradients directly in compact BF16 storage** | Row-owner backward produced a full vocabulary-sized BF16 tensor and copied the requested rows into the send buffer. | Call ATen embedding backward with compact row labels that preserve token order and sorted row groups. Pad the internal row count for the ordered CUDA kernel when needed, then send only the unique rows. The existing path handles unsupported shapes and runtimes. Owners still accumulate in FP32. [Coalescing implementation][owner] · [CUDA dispatch evidence][compact] |

The MTP schedule depends on the MTP FFN split. `layers.0` is the first
trunk block and is independent of that buffer reuse. The donor's BF16 region
is 268,959,744 bytes, of which `layers.7` uses 170,460,160. The trunk norm
borrows 540,672 bytes of FP32 gradient staging. Both tested topologies use
the same dense eight-rank group for these operations. Later MoE blocks keep
expert weights on their configured expert mesh and the rest of the block on
dense GFSDP. The expert topology still changes competition for compute and
communication. [Buffer layout and scope][report]

The four original settings under `parallelism` are `gfsdp_split_layer0_ffn`,
`gfsdp_split_mtp_ffn`, `gfsdp_mtp_transition_prefetch`, and
`gfsdp_row_owner_embeddings`. The tested `dg-n4c` source selects the
four-channel group and compact coalescing on the row-owner path.
[Settings and activation][sharding] · [Final patch][patch]

## Final 10+10 on compiled Original A

The ten unprofiled runs used the eight-layer 100B crop on eight H100 80 GB
GPUs, with dense FSDP8, TP/PP/CP1, real `data-002`, the production tokenizer,
seed 2048, local batch 2, sequence length 4096, global batch 512 and 32
accumulation microbatches. All arms kept FP8/BF16/FP32, Inductor model and
loss compilation, selective activation checkpointing, MTP, EMA, native FLA,
Original A routing and cross-entropy, and fresh compiler caches.
[Run protocol][validation] · [Runtime][runtime] · [Experiment scope][report]

Window TPS divides 20,971,520 physical input tokens by elapsed time over
updates 11–20. Peak memory is the largest native GIFT value across those
updates and all eight ranks. It combines the interval's peak PyTorch reserved
memory with non-PyTorch memory sampled at logging time. All ten arms completed
with stable observed compilation and zero allocator retries or OOMs in the
measured window. [EP8 results][ep8-results] · [EP4 results][ep4-results]

| Topology | Arm | Window TPS | Median update TPS | Peak total GPU memory, GiB | Non-PyTorch at peak, GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| EP8 / eFSDP1 | Original A | 155,695.87 | 155,670.5 | 72.32 | 2.56 |
| EP8 / eFSDP1 | All four, shared communicator | 158,691.75 | 158,728 | 73.34 | 5.24 |
| EP8 / eFSDP1 | All four, dedicated four channels | 158,671.52 | 158,808 | 71.09 | 3.02 |
| EP8 / eFSDP1 | All four, dedicated eight channels | 158,944.54 | 159,016 | 71.52 | 3.43 |
| EP8 / eFSDP1 | **Final, four channels and compact gradients** | **159,093.39** | 159,168 | **69.51** | 3.02 |
| EP4 / eFSDP2 | Original A | 146,782.87 | 146,834 | 75.98 | 2.91 |
| EP4 / eFSDP2 | All four, shared communicator | 149,390.29 | 149,468 | 76.32 | 5.60 |
| EP4 / eFSDP2 | All four, dedicated four channels | 148,765.33 | 148,912 | 74.09 | 3.39 |
| EP4 / eFSDP2 | All four, dedicated eight channels | 148,737.55 | 148,704 | 74.51 | 3.80 |
| EP4 / eFSDP2 | **Final, four channels and compact gradients** | **148,678.95** | 148,744.5 | **72.49** | 3.39 |

The final attempt IDs are `ep8-ep8-combined-36872437` and
`ep4-ep4-combined-d93609e6`. Compared with the four original fixes on the
shared communicator, the final variant saved **3.83 GiB** on each topology,
with **+0.253% TPS** on EP8 and **−0.476% TPS** on EP4. Adding compact
gradients to the four-channel arm saved another 1.58 GiB on EP8 and 1.60 GiB
on EP4, with the same sampled non-PyTorch memory. The sub-percent TPS changes
are observations from one run per arm. [EP8 results][ep8-results] · [EP4 results][ep4-results]

## Isolated and earlier training experiments

| Change | Matched 10+10 control | EP8 TPS / peak delta | EP4 TPS / peak delta | Isolated evidence |
| --- | --- | ---: | ---: | --- |
| `layers.0` FFN split | Original A | −0.11% / +0.16 GiB | −0.22% / +0.09 GiB | Block forward and backward 25.274 → 23.375 ms (−7.51%). Peak PyTorch allocated 3.377 → 3.126 GiB. [FFN replay][ffn-replay] |
| MTP FFN split | Original A | −0.19% / −1.83 GiB | −0.47% / −2.51 GiB | Block forward and backward 32.221 → 30.136 ms (−6.47%). Peak PyTorch allocated 3.609 → 3.358 GiB. [FFN replay][ffn-replay] |
| MTP backward schedule | MTP FFN split | +0.85% / 0.00 GiB | +0.79% / 0.00 GiB | The MTP FFN gather moved into the LM-head backward window. The last trunk dense gather began sooner after FFN slot release. [MTP trace][mtp-trace] |
| Row-owner on the shared communicator | `layers.0` FFN split | +1.77% / +0.75 GiB | +1.23% / +0.24 GiB | Embedding forward and backward 12.894 → 2.777 ms. The embedding fixture dropped one 2,101,346,304-byte backing per rank. [Embedding replay][embedding-replay] |

The native `layers.0` pair measured each recipe with its own initialization.
A separate check found 18 of 23 logical FP32 tensors in the block initialized
differently. The component replay restored the same logical weights for its
bitwise comparison. The direct native row-owner arms reached Dynamo's
eight-specialization cap for `CheckpointWrapper.forward` before update 1.
With the `layers.0` split in both arms, the conditional row-owner pairs
completed using the same compiler limit and checkpointing policy.
[Per-fix results][per-fix] · [Initial weights][initial-weights] · [Conditional row-owner pairs][owner-recovery]

In the earlier matrix, the four original changes together gave **+2.13% TPS**
and **+1.01 GiB** on EP8, and **+1.46% TPS** and **+0.24 GiB** on EP4. The
row-owner replay reproduced **2.686 GiB** per GPU of extra non-PyTorch memory
at the first sparse ID exchange. Of this, 2,818,572,288 bytes are accounted
for by 224 pairs of 10 MiB and 2 MiB NCCL P2P buffers. Keeping the dense
communicator warm, the separate four-channel communicator saved 2.213 GiB
of residency against the shared path. Eight channels saved 1.807 GiB, and a
separate default-width communicator added 0.506 GiB.
[Earlier matrix][earlier] · [Allocation replay][external-memory] · [Communicator replay][narrow-memory]

At 2,234 unique rows, compact coalescing shrank the source BF16 output from
1,050,673,152 to 18,300,928 bytes before exchange. At the pinned H100 training
shape, padding to at least 132 rows preserved the ordered CUDA kernel at low
unique counts. The compact-only EP8 run on the shared communicator saved
1.60 GiB with −0.49% window TPS. The fixed-shape routing experiment removed
an extra host sync and device-to-host copy. It ran slightly slower in both
isolated pairs. The donor-add experiment exposed subnormal differences in
the pinned TMA scatter-add path. Neither routing nor donor-add entered the
final package. [Compact dispatch][compact] · [Compact-only timing][compact-native] · [Other follow-ups][follow-ups]

Two scheduling investigations stayed separate from the final matrix. The
EP8 control captured the first trunk FFN gather starting 2–54 µs after the
`layers.1` dense reduction, which motivated a prefetch trial from `layers.1`.
A later MoE block would need a different shared-FFN split. In the archived
layer-4 trace, the existing dense reduction already overlapped compute for
96.62–97.74% of its duration on the sampled ranks and microbatches.
[First FFN control trace][layer0-prefetch] · [Later-layer investigation][follow-ups]

## What the MTP backward trace showed

The EP8 profiles sampled three microbatches on ranks 0 and 1. The control
already finished the MTP parent gather before the LM-head reduce-scatter.
The FFN gather started 0.003–0.263 ms after that reduction ended. With the
new schedule, the FFN gather finished 22.60–28.69 ms before it began. Each
MTP group still issued one backward gather per microbatch.
[Trace comparison][mtp-trace]

After the MTP FFN reduce-scatter, the `layers.7` dense gather started
13.58–14.04 ms later in the control and 0.071–0.440 ms later with the new
schedule. Its span outside classified rank-local compute fell from
0.423–0.829 to 0.011–0.055 ms. The trunk norm began gradient accumulation
before the MTP parent reduce-scatter ended. `layers.7` backward still began
after the MTP parent reduction in all captured arms. The available trace
does not identify the exact CUDA event responsible for the remaining wait.
[Timing and buffer dependencies][mtp-trace] · [Stream boundary][mtp-boundary]

## Numerical evidence

| Changed path | Scoped bitwise result |
| --- | --- |
| `layers.0` FFN split | Given the same logical weights, local gradients matched before NCCL. Moving a logical gradient to a different destination shard changed the FP32 ring addition order. Captured inputs differed in 23 of 202,424,512 FP32 words, with max absolute delta `5.82e-11`. Seeded inputs reached `3.05e-5` near magnitude 410. The ring reconstruction matched all 48 sampled words. [Root-cause replay][layer0-numeric] |
| MTP FFN split | In the captured block, unchanged cuDNN attention gave different output and log-sum-exp values between forward and checkpoint replay, with identical Q/K/V. Reusing the original attention outputs during that replay made the FFN input, incoming cotangent and local `w1/w2/w3` gradients bitwise equal on all eight ranks. A separate identical-arm test also isolated cuDNN backward `dQ` drift. [Causal intervention][mtp-causal] |
| MTP backward schedule and buffer reuse | The eight-rank shape-isolated replay matched bitwise with compiled selective checkpointing, chunked MTP loss and repeated accumulation, including gradient reset and reuse across shapes. [Schedule replay][mtp-exact] |
| Dense embedding → row-owner | BF16 lookups and FP32 owner gradients matched for the checked ordinary inputs. An adversarial cancellation fixture with contributions at `±2^24` differed in 34 FP32 words after one accumulation pass, with max absolute delta `2.0`. NCCL ring and row-owner rank-order sums reconstructed the difference exactly. [Ordinary-input replay][owner-exact] · [Cancellation reconstruction][owner-order] |
| Shared row-owner → four channels → compact gradients | Lookup, exchanged payloads and FP32 gradients matched bitwise across eight ranks in six scenarios, including real high- and low-cardinality inputs, all unique IDs, a single remote owner, and padding and token-count boundaries. Initial, accumulated and reset states matched. [Final exact replay][final-exact] |

The two reproducible FP32 addition-order differences were accepted for this
review. The component checks cover the edited paths with controlled inputs.
Full-training bitwise parity and convergence remain outside the measured
scope. The cuDNN attention implementation was unchanged by this package.

[port-manifest]: ../../.worktrees/dense-gfsdp-fix/runs/dense-gfsdp-fix/reviewed-port.sha256
[stage]: ../../.worktrees/dense-gfsdp-fix/runs/dense-gfsdp-fix/artifacts/reviewed-port-001/embedding-combined-001/stage_sources.sh
[patch]: ../../.worktrees/dense-gfsdp-fix/runs/dense-gfsdp-fix/artifacts/reviewed-port-001/embedding-combined-001/dg-n4c.patch
[bundle]: ../../.worktrees/dense-gfsdp-fix/runs/dense-gfsdp-fix/artifacts/reviewed-port-001/embedding-combined-001/bundle.sha256
[ep8-source]: ../../.worktrees/dense-gfsdp-fix/runs/dense-gfsdp-fix/artifacts/raw/embedding-combined-001/ep8/combined.json#L2270
[ep4-source]: ../../.worktrees/dense-gfsdp-fix/runs/dense-gfsdp-fix/artifacts/raw/embedding-combined-001/ep4/combined.json#L2270
[ep8-results]: ../../.worktrees/dense-gfsdp-fix/runs/dense-gfsdp-fix/artifacts/raw/embedding-combined-001/ep8/results.json
[ep4-results]: ../../.worktrees/dense-gfsdp-fix/runs/dense-gfsdp-fix/artifacts/raw/embedding-combined-001/ep4/results.json
[runtime]: ../../.worktrees/dense-gfsdp-fix/runs/dense-gfsdp-fix/artifacts/raw/embedding-combined-001/runtime.json
[validation]: ../../planning/validation.md
[report]: ../../.worktrees/dense-gfsdp-fix/runs/dense-gfsdp-fix/report.md
[sharding]: ../../.worktrees/dense-gfsdp-row-owner-nccl4-compact-001/gift/torchtitan/models/gift_moe/_base/parallelize.py
[ffn]: ../../.worktrees/dense-gfsdp-row-owner-nccl4-compact-001/gift/torchtitan/distributed/ffn_overlap.py#L16
[owner]: ../../.worktrees/dense-gfsdp-row-owner-nccl4-compact-001/gift/torchtitan/distributed/gfsdp/row_owner_embedding.py
[owner-nccl]: ../../.worktrees/dense-gfsdp-row-owner-nccl4-compact-001/gift/torchtitan/distributed/gfsdp/row_owner_embedding.py#L155
[channel-receipt]: ../../.worktrees/dense-gfsdp-fix/runs/dense-gfsdp-fix/artifacts/raw/embedding-final-bitwise-001/combined/effective-channels.json
[compact]: ../../.worktrees/dense-gfsdp-fix/runs/dense-gfsdp-fix/compact-source-coalescing.md
[ffn-replay]: ../../.worktrees/dense-gfsdp-fix/runs/dense-gfsdp-fix/ffn-replay-result.json
[embedding-replay]: ../../.worktrees/dense-gfsdp-fix/runs/dense-gfsdp-fix/embedding-replay-result.json
[per-fix]: ../../.worktrees/dense-gfsdp-fix/runs/dense-gfsdp-fix/artifacts/raw/native-001/per-fix/results.json
[initial-weights]: ../../.worktrees/dense-gfsdp-fix/runs/dense-gfsdp-fix/artifacts/raw/native-001/initial-weights/comparison.json
[owner-recovery]: ../../.worktrees/dense-gfsdp-fix/runs/dense-gfsdp-fix/artifacts/raw/native-001/row-owner-recovery/results.json
[earlier]: ../../.worktrees/dense-gfsdp-fix/runs/dense-gfsdp-fix/artifacts/raw/native-001/final/results.json
[external-memory]: ../../.worktrees/dense-gfsdp-fix/runs/dense-gfsdp-fix/artifacts/raw/row-owner-external-memory-001/comparison.json
[narrow-memory]: ../../.worktrees/dense-gfsdp-fix/runs/dense-gfsdp-fix/artifacts/raw/row-owner-narrow-communicator-001/comparison.json
[compact-native]: ../../.worktrees/dense-gfsdp-fix/runs/dense-gfsdp-fix/artifacts/raw/compact-native-002/timing/results.json
[follow-ups]: ../../.worktrees/dense-gfsdp-fix/runs/dense-gfsdp-fix/report.md#follow-up-experiments
[layer0-prefetch]: ../../.worktrees/dense-gfsdp-fix/runs/dense-gfsdp-fix/artifacts/raw/native-001/validation/layer0-prefetch-control.json
[mtp-trace]: ../../.worktrees/dense-gfsdp-fix/runs/dense-gfsdp-fix/artifacts/raw/native-001/validation/mtp-trace-analysis.json
[mtp-boundary]: ../../.worktrees/dense-gfsdp-fix/runs/dense-gfsdp-fix/artifacts/raw/native-001/validation/mtp-parent-boundary.json
[layer0-numeric]: ../../.worktrees/dense-gfsdp-fix/runs/dense-gfsdp-fix/artifacts/raw/bitwise-root-cause-001/layer0/summary.json
[mtp-causal]: ../../.worktrees/dense-gfsdp-fix/runs/dense-gfsdp-fix/artifacts/raw/bitwise-root-cause-001/mtp-causal-summary.json
[mtp-exact]: ../../.worktrees/dense-gfsdp-fix/runs/dense-gfsdp-fix/artifacts/raw/all-final-bitwise-001/mtp-transition-shape-isolated/result.json
[owner-exact]: ../../.worktrees/dense-gfsdp-fix/runs/dense-gfsdp-fix/artifacts/raw/native-001/validation/embedding-exact.json
[owner-order]: ../../.worktrees/dense-gfsdp-fix/runs/dense-gfsdp-fix/artifacts/raw/bitwise-root-cause-001/embedding-order-reconstruction.json
[final-exact]: ../../.worktrees/dense-gfsdp-fix/runs/dense-gfsdp-fix/artifacts/raw/embedding-final-bitwise-001/comparison.json
