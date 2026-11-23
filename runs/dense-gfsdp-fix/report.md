# Dense GFSDP and input embedding validation

## Scope and source

Original A is revision `e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f` and its
GIFT tree is `0f44b971069af55d6d573628129900483ca3e1dc`. This is the
revision used by the existing Linear-fixes control and the Original A
post-promotion comparison recorded at `529d8f27d:runs/fusion-followup/report.md`.
Original A contains the two FFN splits and ordinary row-owner embeddings as
disabled options. The candidate uses that code and a scoped port of the MTP
transition from `3860b1eb5`. The control runs from its own pristine checkout.
Both preserve native FLA, original routing and original CE, real data-002,
FP8/BF16/FP32 policy, selective SAC, model/loss compile, and the 32-microbatch
accumulation schedule. Expert optimization flags stay at Original A values.
Both topologies keep the same eight-rank dense group. The FFN split groups,
row owners, and MTP buffer transition use that group. The layer-7 target
selects the parameter group on the MTP donor's exact process group. Changing
EP and eFSDP can still alter GPU work and competition for links, so the
observed training effect is measured separately at each topology.

The [reviewed file manifest](reviewed-port.sha256) pins the transition port
and the two native drivers. The Original A checkout and reviewed candidate
are on SR008 under `dg-a` and `dg-fix`. The first timing attempt stopped
before training because the empty ignored output directory did not exist.
The rerun used the same source and inputs after creating that directory.

## What each comparison measures

| Candidate | Matched control | Check on the active path |
| --- | --- | --- |
| First trunk FFN split | Original A | Separate BF16 gather and FP32 gradient reduce-scatter for `layers.0.feed_forward`. Check timing against the remaining block backward and shared storage capacity. |
| MTP FFN split | Original A | The same comparison at `mtp.layers.0.block.feed_forward`, including the effect on readiness of the head-side backward gather. |
| Row-owner input embedding | Original A | Check request and return traffic, source coalescing and FP32 owner accumulation on real IDs. Check the documented BF16 transport boundary against the ordinary path. |
| MTP transition scheduling plus MTP split | MTP split only | Check the two backwards gathers issued by the head, the earlier layer-7 dense gather and the norm gradient packing after the FFN region releases. Measure layer-7 backward overlap separately. |
| All four changes | Original A | Four final runs, one control and one candidate at each of EP8/eFSDP1 and EP4/eFSDP2. |

The splits add collectives and change the ready points within the same dense
group. The transition borrows existing unsharded buffers and adds event
dependencies. Row-owner lookup exchanges sparse row requests and gradients,
with cost sensitive to real token skew and host-side count synchronization.
Use component or trace evidence for the mechanism and unprofiled native runs
for whole-training effects. Preserve complete 20-update logs, exact resolved
configuration, active feature receipts and all-rank window data. Report global
physical tokens/s by dividing the 20,971,520 physical tokens from updates
11–20 by the interval from the earliest rank start to the latest rank end.
Report the maximum native GIFT total GPU memory estimate over every rank and
those updates. The GIFT console reports
the interval's peak PyTorch reserved bytes plus non-PyTorch memory sampled
when the update is logged, rounded to 0.01 GiB. A single comparison gives an
observed difference. Identical controls and repeats are needed when that
difference is close to run variation.

## Source changes and intended effect

| Change | Before | After |
| --- | --- | --- |
| First trunk FFN | `shard_block_unit` wrapped `layers.0` once. Its attention, norms and dense `w1/w2/w3` FFN shared one GFSDP group. The parent backward boundary determined when their gradients could be reduced. | `enable_ffn_overlap` exposes prefix, FFN and suffix compute boundaries. `shard_block_unit` wraps `block.feed_forward` first and wraps the remainder second. The FFN reduce-scatter can start while backward works through the prefix. Both groups still shard over the same eight dense ranks. |
| MTP FFN | `mtp.layers.0` grouped its input projection, attention, internal norms and dense FFN into one GFSDP unit. | The same split wraps `mtp.layers.0.block.feed_forward` separately. The projections, attention and MTP norms stay in its parent. The separate FFN group can submit its reduce-scatter sooner and provides two reusable buffer regions. |
| Input embedding | `g_fully_shard(model.tok_embeddings, ...)` gathered the BF16 table and reduced its full gradient. | `RowOwnerEmbedding` holds whole rows on their GFSDP owners, exchanges unique requested IDs and BF16 rows, and returns coalesced BF16 gradients to the owners for FP32 accumulation in donor order. `model.lm_head` stays on its existing GFSDP path. |
| MTP transition | The head's default backward prefetch targeted the next unit in the chain. The last trunk dense group and trunk norm used the standard shared slots. | The head explicitly prefetches both MTP groups. The last trunk dense group uses the released MTP FFN BF16 region, and `model.norm` uses its released FP32 gradient region. Aliasing follows the exact dense process group, and overlapping regions wait on release events. |

The FFN is the MLP submodule inside the transformer block. Its split changes
the collective boundaries and gives the two groups disjoint views of existing
buffer slots. In later MoE blocks, the configured expert parameters use the
expert mesh and the remaining parameters use dense GFSDP. The transition
targets the last trunk layer and has no dependency on the first trunk split.
[Sharding and prefetch](../../gift/torchtitan/models/gift_moe/_base/parallelize.py) ·
[FFN compute boundaries](../../gift/torchtitan/distributed/ffn_overlap.py) ·
[Row-owner lookup](../../gift/torchtitan/distributed/gfsdp/row_owner_embedding.py)

## Component evidence

The [FFN replays](ffn-replay-result.json) used captured real training tensors and
paired component initialization on Original A. Each arm used eight H100s, ten
warmups and ten measured samples. Wall time covers the complete isolated block
forward and backward with selective checkpoint replay and GFSDP.

| Split | Median rank-max wall time | Change | Peak PyTorch allocated, max rank | GFSDP pool backing per rank |
| --- | ---: | ---: | ---: | ---: |
| `layers.0` FFN | 25.274 → 23.375 ms | −7.51% | 3.377 → 3.126 GiB | 1,214,547,072 B in both arms |
| MTP FFN | 32.221 → 30.136 ms | −6.47% | 3.609 → 3.358 GiB | 1,405,627,392 B in both arms |

These are isolated allocator peaks. The two subgroups use disjoint views of
the original pool, so the split leaves the measured backing capacity unchanged.

The [row-owner replay](embedding-replay-result.json) used the same captured
real-data token IDs and seeded backward gradients in both arms. On the pinned
SR008 runtime and eight H100s, median synchronized forward plus backward time
over ten measured rank-max samples fell from 12.894 to 2.777 ms (−78.46%).
The joined CUDA-event span fell from 12.878 to 2.757 ms. The median rank-max
host-call span rose from 1.045 to 2.581 ms, while the median rank-max explicit
endpoint wait fell from 11.918 to 0.277 ms. Each scope has its own rank max.
The owner arm borrowed the same private LM-head gradient storage and dropped
one 2,101,346,304-byte backing storage in the embedding-only factory on each
rank. Complete training memory remains a separate native measurement.

## Native 10+10 timing

| Final acceptance pair | Original A TPS | All four TPS | TPS change | Original A peak | All four peak | Peak change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EP8 / eFSDP1 | 155,570 | 158,884 | +2.13% | 72.32 GiB | 73.33 GiB | +1.01 GiB |
| EP4 / eFSDP2 | 146,496 | 148,628 | +1.46% | 76.08 GiB | 76.32 GiB | +0.24 GiB |

All four arms passed with eight ranks, 20 complete updates, stable
compiler observations, and zero allocator retries or OOMs in the measured
window. The fresh controls differ from the per-fix controls by +0.07% TPS
and +0.02 GiB on EP8 and by −0.18% TPS and +0.11 GiB on EP4. These are
single observations of run variation under the two topologies.
[All four run receipts](artifacts/raw/native-001/final/results.json) ·
[Launch order and exit codes](artifacts/raw/native-001/final/attempts.jsonl)

| Final acceptance pair | Max PyTorch allocated, Original A → all four | Max PyTorch reserved, Original A → all four | Non-PyTorch at total peak, Original A → all four |
| --- | ---: | ---: | ---: |
| EP8 / eFSDP1 | 69.15 → 65.89 GiB | 69.77 → 68.09 GiB | 2.55 → 5.24 GiB |
| EP4 / eFSDP2 | 72.21 → 68.35 GiB | 73.16 → 70.72 GiB | 2.91 → 5.60 GiB |

The layer-0-only to layer-0-plus-row-owner pairs reproduce the full
2.69 GiB increase outside PyTorch's allocator on both topologies. The
other individual changes leave this metric within 0.01 GiB of Original A.
The NCCL replay below identifies the external allocation.
[Peak-step console receipts](artifacts/raw/native-001/validation/native-peak-memory.json) ·
[EP8 Original A allocator receipt](artifacts/raw/native-001/final/native/ep8-a/summary.json) ·
[EP8 all-four allocator receipt](artifacts/raw/native-001/final/native/ep8-all/summary.json) ·
[EP4 Original A allocator receipt](artifacts/raw/native-001/final/native/ep4-a/summary.json) ·
[EP4 all-four allocator receipt](artifacts/raw/native-001/final/native/ep4-all/summary.json)

The table below uses the separate per-fix controls and reports the paired
effect of the named change with any prerequisite present in both arms.

| EP / eFSDP | Isolated change | Window global TPS, control → candidate | Window TPS | GIFT total peak, control → candidate |
| --- | --- | ---: | ---: | ---: |
| EP8 / eFSDP1 | `layers.0` FFN split | 155,459 → 155,284 | −0.11% | 72.30 → 72.46 GiB |
| EP8 / eFSDP1 | MTP FFN split | 155,459 → 155,162 | −0.19% | 72.30 → 70.47 GiB |
| EP8 / eFSDP1 | MTP transition, with the MTP split in both arms | 155,162 → 156,479 | +0.85% | 70.47 → 70.47 GiB |
| EP8 / eFSDP1 | Row-owner, with the `layers.0` split in both arms | 155,284 → 158,038 | +1.77% | 72.46 → 73.21 GiB |
| EP4 / eFSDP2 | `layers.0` FFN split | 146,760 → 146,441 | −0.22% | 75.97 → 76.06 GiB |
| EP4 / eFSDP2 | MTP FFN split | 146,760 → 146,071 | −0.47% | 75.97 → 73.46 GiB |
| EP4 / eFSDP2 | MTP transition, with the MTP split in both arms | 146,071 → 147,232 | +0.79% | 73.46 → 73.46 GiB |
| EP4 / eFSDP2 | Row-owner, with the `layers.0` split in both arms | 146,441 → 148,239 | +1.23% | 76.06 → 76.30 GiB |

The transition improved the logged TPS at all ten corresponding measured
updates in each topology. Each topology has one timed run per arm.
[All direct per-fix receipts](artifacts/raw/native-001/per-fix/results.json)

For the EP8 `layers.0` pair, median logged update TPS moved +0.21%. On rank 3, the peak
reserved amount rose from 69.752 to 69.908 GiB, which accounts for the
observed total memory change at console precision. The two runs also
reported different first-update loss and gradient norm. A fresh diagnostic
compared initialized parameter bytes with `gfsdp_match_fsdp2_init` enabled
in both arms. It reconstructed all 23 logical FP32 tensors in `layers.0`.
Eighteen differed, including all three FFN weights and random attention
and norm parameters. The other 225 tensors matched and no layout or name
comparisons were unresolved. The native pair measures the two recipes as
they initialize today. The same-weight component replay isolates the block
effect more closely, and a same-weight native comparison is still open.
[Pair receipts](artifacts/raw/native-001/per-fix/layer0-memory-details.json) ·
[Initial-weight comparison](artifacts/raw/native-001/initial-weights/comparison.json)

The direct row-owner arms at EP8 and EP4 activated their eight-rank owner
groups and failed before completing update 1. Dynamo reached its full-graph limit of eight
specializations of `CheckpointWrapper.forward`. The final guard involved
the selective checkpoint policy arguments. Row-owner is installed after
checkpointing and compilation are configured. The guard report alone
leaves the earlier specialization reasons unresolved. The conditional
comparison uses the first-FFN split in both arms and preserves SAC and the
existing compiler settings.
[EP8 attempt](artifacts/raw/native-001/per-fix/ep8-row-owner-failure.txt) ·
[EP4 attempt](artifacts/raw/native-001/per-fix/ep4-row-owner-failure.txt) ·
[Conditional receipts](artifacts/raw/native-001/row-owner-recovery/results.json)

The EP8 transition startup receipt names the donor as
`mtp.layers.0.block.feed_forward`. The dense group of `layers.7` borrows
170,460,160 bytes from the donor's 268,959,744-byte BF16 region. The trunk's
`model.norm` borrows 540,672 bytes from the corresponding FP32 gradient
region. Those values describe the alias layout. The trace comparison checks
when the gathers and gradient packing actually run.

## What changed in MTP backward

The EP8 captures cover three complete microbatches from update 11 on ranks
0 and 1 in each arm. The MTP-split control isolates the scheduling change.
The all-four capture checks that the same ordering also occurs in the
combined candidate.

| Observed device timing | MTP split only | MTP split and transition | All four |
| --- | --- | --- | --- |
| MTP FFN BF16 gather relative to LM-head reduce-scatter | Starts 0.003–0.263 ms after the head reduction ends | Finishes 22.60–28.69 ms before the head reduction starts | Finishes 22.58–27.40 ms before the head reduction starts |
| Dense `layers.7` BF16 gather starts after the MTP FFN reduce-scatter ends | 13.58–14.04 ms | 0.071–0.440 ms | 0.006–1.748 ms |
| Dense `layers.7` gather time outside classified rank-local compute | 0.423–0.829 ms | 0.011–0.055 ms | 0.016–0.076 ms |
| Trunk norm gradient accumulation starts relative to the MTP parent reduce-scatter end | 0.003–0.058 ms after | 0.143–4.711 ms before | 3.003–7.164 ms before |
| `layers.7` compiled backward overlap with MTP parent reduce-scatter | 0 ms | 0 ms | 0 ms |

The MTP parent BF16 gather already finished before the head reduction in
the control. Each MTP group performed one backward gather per microbatch in
all three arms. The trace therefore supported the earlier FFN and layer-7
gathers but did not show duplicate MTP gathers across head chunks. CUDA
kernel timing alone cannot establish that a particular wait event blocked a
consumer. The trace shows norm gradient accumulation within its GFSDP scope,
and the startup receipt supplies the actual FP32 alias assignment.
[Trace analysis and per-microbatch timings](artifacts/raw/native-001/validation/mtp-trace-analysis.json)

In the all-four capture, `layers.7` started its first compiled backward
kernel 0.400–0.461 ms after the MTP parent reduce-scatter. The CUDA launch
had already been submitted, and the compute stream was idle in this interval.
A 576 MiB device-to-device copy under the layer-7 expert gather occupied
0.395–0.400 ms on the collective stream and ended 2.30–7.55 µs before that
first backward kernel. The trace records the dense and expert group wait
calls but omits the event handles needed to match a particular wait to its
record. The expert gather is outside the dense changes under test.
[Raw stream boundary analysis](artifacts/raw/native-001/validation/mtp-parent-boundary.json)

## Numerical and source checks

The pinned runtime passed all 63 tests covering the FFN split, buffer order
and sharing, and GFSDP checkpoint continuation. A separate eight-rank replay
comparing Original A to row-owner on data-002 matched BF16 lookup results
and FP32 local gradients exactly on every rank, including accumulation and
`grad=None` reset. This replay used fixed seeded upstream gradients and
did not cover full-model numerical equality.
[Source test log](artifacts/raw/native-001/validation/transition-tests.log) ·
[Embedding numerical receipt](artifacts/raw/native-001/validation/embedding-exact.json)

## Follow-up experiments

The fixed-shape routing replay removed the extra host synchronization and
device-to-host copy from ID partitioning on all eight ranks, while matching
the BF16 lookup, gradient messages, and FP32 accumulated gradients exactly.
The rank-max synchronized forward plus backward medians were 2.717 → 2.734
ms and 2.690 → 2.753 ms in the two pairs. Peak allocated bytes and the GFSDP
pool were unchanged. The replay supplied no throughput case for integrating
this routing variant.
[Fixed-shape routing timing and trace receipts](artifacts/raw/fixed-shape-route-001/summary.json)

The padded compact coalescing replay matched the selected BF16 rows on all
eight ranks for the two real fixtures and the synthetic dispatch cases.
The natural low-unique-ID batch included ranks at 12 and 14 IDs. Its
rank-max median fell from 0.723 to 0.291 ms, and the peak allocated increment
fell from 1,198,710,272 to 84,503,552 bytes. The source-op replay excludes
NCCL and owner accumulation. The next trial compares identical BF16 wire
bytes and FP32 local gradients through the owner module before native timing.
[Coalescing implementation and dispatch evidence](compact-source-coalescing.md) ·
[Source-op replay](artifacts/raw/compact-coalescing-001/summary.json)

The native traces close the duplicate backward gather hypothesis for the
captured EP8 workloads.

The EP8 compiler diagnostic kept full-graph compilation, selective AC,
and the eight-specialization limit. Original A used `CheckpointWrapper.forward`
at nine blocks and logged five recompiles of that function per rank. Row-owner
alone used the same nine wrappers, logged eight recompiles per rank, and stopped
on all eight ranks before update 1. With the first FFN split, `layers.0` used
three compiled regions and the other eight blocks kept the wrapper. That arm
logged four wrapper recompiles per rank and finished update 1. The logs show
guards on the per-block checkpoint policy, module type, and argument count.
They establish which shared function hit the limit. The source of the additional
specializations activated by row-owner remains open.
[Original A](artifacts/raw/compile-guard-001/compile-guard-001-compile-guard-ep8-a-e9f71265.json) ·
[Row-owner alone](artifacts/raw/compile-guard-001/compile-guard-001-compile-guard-ep8-row-owner-560ce633.json) ·
[Row-owner with the first FFN split](artifacts/raw/compile-guard-001/compile-guard-001-compile-guard-ep8-layer0-row-owner-b0b96e17.json)

The [donor-add replay](owner_add_replay.py) reuses `owner_add_bf16_` from
`3860b1eb5:gift/torchtitan/distributed/gfsdp/_row_owner_embedding_kernels.py`.
It widens each donor's BF16 rows and adds them to the destination FP32
gradient in one kernel, preserving the existing serial rank order. All
eight donors contributed to each owner in the archived matched capture.
The archived component replay measured 2.479 → 2.100 ms end to end. The
new verification matched the captured inputs, BF16 payload and FP32
initial, accumulated and reset gradients on all eight ranks. It failed
the adversarial gradient bitwise check on every rank, so timing did not
start. The pinned Torch fast path uses Hopper TMA FP32 scatter-add, which
flushes subnormal operands and results. The next component trial uses an
explicit matching add and retains the full bitwise check.
[Failed parity receipt](artifacts/raw/owner-add-001/parity.json) ·
[Pinned TMA instruction](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/torch/_native/ops/scatter_add/tma_kernel.py#L74-L86) ·
[CUDA 13.0 instruction semantics](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk)

Fresh eight-GPU processes warmed the same dense BF16 all-gather and FP32
reduce-scatter, then ran the ordinary row-owner protocol or a no-exchange
control. All eight ranks first added 2,883,584,000 bytes (2.686 GiB) outside
the PyTorch allocator when the int64 request exchange issued grouped
`ncclSend` and `ncclRecv`. The logs show 32 P2P channels per nonself peer.
The 224 pairs of 10 MiB and 2 MiB shareable buffers account for
2,818,572,288 bytes per rank. Subsequent BF16 exchanges and freeing PyTorch's
cache left the delta unchanged. Destroying the dense communicator released
those logged buffer bytes and left a 65,011,712-byte residual. The measurement
reproduces the +2.69 GiB rounded non-PyTorch delta from both native topologies.
[Eight-GPU replay measurements](artifacts/raw/row-owner-external-memory-001/comparison.json) ·
[NCCL allocation source](https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/transport/p2p.cc)

The compact trial's first staged launcher exited before replay because it
could not import the task's `campaign.py`. The corrected launch matched real
high- and low-unique-ID fixtures on all eight ranks through the actual
row-owner module, including BF16 messages and FP32 local gradients. Its EP8
native comparison measured 158,690 → 157,910 window TPS (−0.49%) and
73.34 → 71.74 GiB total peak (−1.60 GiB). Median logged step TPS was
158,744 → 159,080. The metrics cover different timing scopes. This one pair
establishes an observed memory improvement and leaves window TPS parity open.
[Exact module comparison](artifacts/raw/compact-native-002/exact/comparison.json) ·
[Native timing](artifacts/raw/compact-native-002/timing/results.json)

An eight-GPU replay kept the dense communicator warm and moved all four
row-owner exchanges to a separate communicator. All wire bytes and rank
ordering matched in all arms. With both communicators live, a default-width
embedding communicator cost an extra 0.506 GiB per GPU against shared
routing. Caps of eight and four CTAs produced eight and four P2P channels
per peer, saving 1.807 and 2.213 GiB per GPU respectively. This measures
communicator residency. The new compile-enabled 10+10 campaign compares
fresh Original A and all-four shared controls with both caps and cap four
plus compact coalescing at EP8 and EP4. The candidates use the same GPU
stream order and wire precision. Their dedicated communicator uses the
PyTorch NCCL default timeout. The existing GIFT timeout transition applies
to the world and mesh communicators.
[Communicator memory replay](artifacts/raw/row-owner-narrow-communicator-001/comparison.json) ·
[Exact replay driver](narrow_owner_exact.py) ·
[Native timing driver](narrow_native_acceptance.py)

### Combined embedding memory campaign

| Topology and variant, compile enabled | Window TPS | Total peak per GPU | Non-PyTorch at total peak | TPS vs all-four shared | Peak vs all-four shared |
| --- | ---: | ---: | ---: | ---: | ---: |
| EP8 / eFSDP1 · Fresh Original A | 155,696 | 72.32 GiB | 2.56 GiB | — | — |
| EP8 / eFSDP1 · All four, shared row-owner communicator | 158,692 | 73.34 GiB | 5.24 GiB | — | — |
| EP8 / eFSDP1 · All four, 4-channel row-owner communicator | 158,672 | 71.09 GiB | 3.02 GiB | −0.013% | −2.25 GiB |
| EP8 / eFSDP1 · All four, 8-channel row-owner communicator | 158,945 | 71.52 GiB | 3.43 GiB | +0.159% | −1.82 GiB |
| EP8 / eFSDP1 · All four, 4-channel communicator and compact gradients | 159,093 | 69.51 GiB | 3.02 GiB | +0.253% | −3.83 GiB |
| EP4 / eFSDP2 · Fresh Original A | 146,783 | 75.98 GiB | 2.91 GiB | — | — |
| EP4 / eFSDP2 · All four, shared row-owner communicator | 149,390 | 76.32 GiB | 5.60 GiB | — | — |
| EP4 / eFSDP2 · All four, 4-channel row-owner communicator | 148,765 | 74.09 GiB | 3.39 GiB | −0.418% | −2.23 GiB |
| EP4 / eFSDP2 · All four, 8-channel row-owner communicator | 148,738 | 74.51 GiB | 3.80 GiB | −0.437% | −1.81 GiB |
| EP4 / eFSDP2 · All four, 4-channel communicator and compact gradients | 148,679 | 72.49 GiB | 3.39 GiB | −0.476% | −3.83 GiB |

All ten arms completed with stable compilation and zero allocator
retries and OOMs in updates 11–20. Against fresh Original A, the combined
arm delivered +2.182% TPS and saved 2.81 GiB at EP8. At EP4 it delivered
+1.292% TPS and saved 3.49 GiB. Against the otherwise identical four-channel
arm, compact gradients saved another 1.58 GiB at EP8 and 1.60 GiB at EP4,
with the same sampled non-PyTorch memory. The A/B replays passed bitwise
checks for lookup, all four exchanges and FP32 gradient states on both
real fixtures across all eight ranks. NCCL INIT logs recorded 32 channels
in the shared path, and 4 or 8 in the dedicated path alongside dense's
32.
[EP8 training results](artifacts/raw/embedding-combined-001/ep8/results.json) ·
[EP4 training results](artifacts/raw/embedding-combined-001/ep4/results.json) ·
[Combined exact receipt](artifacts/raw/embedding-combined-001/exact/combined/comparison.json) ·
[Combined channels](artifacts/raw/embedding-combined-001/exact/combined/effective-channels.json)

The [matched-weight diagnostic](matched_layer0_weights.py) captures the 18
changed layer-0 masters and their EMA shadows from Original A before the
first forward, then loads them through GFSDP's checkpoint adapter into the
split layout. It checks all 243 logical weights before the first update
and compares native microbatch inputs, losses, and pre-clip FP32 layer-0
gradients. This is a numerical diagnostic. Its instrumentation adds host
copies and is unsuitable for TPS comparison.

An [isolated scheduling experiment](queue_layer0_prefetch_profile.sh) requests
the first FFN's BF16 backward gather from `layers.1`. With the current schedule,
the FFN gather starts 2–54 µs after the `layers.1` dense reduce-scatter and
ends 2–4 µs before the FFN's first backward kernel in the six captured EP8
microbatches. Its median duration without other observed device work is
0.742 ms. Both layer-0 groups already occupy disjoint regions of the same
data buffer. The experiment reuses those regions and checks whether the gather
moves before the layer-1 reduction without adding a second gather.
[All-four control mechanism](artifacts/raw/native-001/validation/layer0-prefetch-control.json)

The largest gradient buffer holds 2,101,346,304 bytes for the untied LM head.
Its FP32 gradient survives across the two detached chunked-loss backwards.
Row-owner borrows bytes from that buffer while it is idle. Reducing its
temporary scratch use leaves the head allocation in place. Sharing the
whole 937,084,928-byte ordinary dense gradient buffer with the head would
serialize the early MTP FFN pack, which starts before the head's last device
work in five of the six captured microbatches. Keeping the MTP FFN and its
trunk-norm view in a 537,919,488-byte backing would make up to 399,165,440
bytes (380.674 MiB) per rank available for reclamation if later dense groups
can use the head backing. The late-only layout still requires verification
of callback ownership, the head's terminal event and native memory.
[Gradient buffer allocation](../../gift/torchtitan/distributed/gfsdp/_state.py) ·
[Rank 0 capture](artifacts/raw/native-001/diagnostic/ep8-all-traces/native/profiling/traces/iteration_11/rank0_trace.json) ·
[Rank 1 capture](artifacts/raw/native-001/diagnostic/ep8-all-traces/native/profiling/traces/iteration_11/rank1_trace.json)

Splitting the dense shared FFN out of later MoE blocks would require a
different graph boundary. In layer 4, the shared FFN accounts for 72 MiB
of the 333.19 MiB FP32 dense reduce-scatter input. The current fullgraph
backward exposes its gradients at the block boundary, and the existing
buffer subdivision expects a parent with one parameter group. In the
archived EP8 trace at `1f02311a5`, 96.62–97.74% of the existing layer-4
dense reduce-scatter kernel overlapped rank-local compute activity across
three microbatches on two ranks. A new split would need evidence of an
exposed wait large enough to justify the extra collectives and graph work.
