# Profile research: my ideas

Research origin: the repository owner's own investigation of the baseline profile.
Recorded: 13 November 2026.
Status: historical baseline-profile findings, followed by an owner-selected
MTP experiment and capacity-selection research ideas from later BF16-cache
profiling. A row-owner input-embedding design was selected on 14 November 2026;
it is a research proposal only, with no source implementation or benchmark.
The later
[fusion experiments](../runs/four-fusion/report.md) and
[BF16 reuse experiment](../runs/bf16-expert-cache/report.md) own implementation,
measurements and current conclusions. Earlier observations retain their
original trace scope; the MTP section records a separate, unimplemented and
unbenchmarked experiment, and the capacity ideas below remain unimplemented
research with no measured speedup.

On 18 November 2026, the owner set two high-priority candidates for reducing
memory and step time at `eFSDP > 1`:
[FP8 all-gather for sharded routed-expert weights](#fp8-all-gather-for-sharded-routed-expert-weights)
and [direct expert weight gradients into the FP32 reduce-scatter buffer](#direct-expert-weight-gradients-into-the-fp32-reduce-scatter-buffer).
Two follow-ups from the same review are
[BF16 expert gradient exchange with FP32 owner reduction](#bf16-expert-gradient-exchange-with-fp32-owner-reduction),
selected for an experiment, and
[keeping tail weights live across chunked LM-head backward](#keep-tail-weights-live-across-chunked-lm-head-backward),
selected as a scheduling fix. The weight path remains local FP8 shard caching
with two shared gather slots. A cache of all fully gathered layers is deferred
because its memory grows with model depth.

## Finding and evidence scope

The first microbatch has a short GPU preparation prefix before the embedding
weight all-gather. Its eager PyTorch operations generate ten small kernels with
gaps between launches. CPU/GPU launch correlations show that host submission
limits this prefix. CPU core saturation is not established by this observation.

Evidence comes from Perfetto inspection of `monitorfree-profile-update11-001`,
optimizer update 11, rank 0, first CPU microbatch marker `slice[707]`. The execution
revision is `16b13c2548317945712ded2e66f136b169130d95`. The
[baseline report](../runs/overnight-streaming-perf/report.md#fresh-traces-and-cleanup)
owns configuration/runtime identities and canonical trace paths/hashes. The
inspected artifact is `rank0_trace.json.gz` under that run's
`native/profiling/traces/iteration_11/` directory. Times below use the trace UI's
relative clock. SQL IDs refer to this trace import.

The [baseline](baseline.md) has microbatch size 2, sequence length 4096 and 32
microbatches per optimizer update. Model and loss compilation are enabled.
Shapes below follow those settings and inspected source; input-shape and stack
recording were disabled in this trace.

## What the small kernels do

Attribution uses each kernel's CUDA correlation ID, its CPU launch, the enclosing
CPU operator and the corresponding source operation.

| GPU slice(s) | Operation and purpose | Source |
| --- | --- | --- |
| 714 | `arange`: generate int32 positions `[0, ..., 4095]` for the active causal-mask path | [Position generation](../gift/torchtitan/trainer.py#L584) |
| 728 | `repeat` / `copy_`: materialize positions for both samples as `[2, 4096]` | [Position replication](../gift/torchtitan/trainer.py#L586) |
| 751 | `ne`: create `token_mask = labels != IGNORE_INDEX` | [Token mask](../gift/torchtitan/trainer.py#L594) |
| 754, 763, 765 | Repeat the validity comparison, cast its Boolean result to int64 and sum valid targets for this microbatch | [Auxiliary-loss token count](../gift/torchtitan/trainer.py#L630) |
| 768, 771 | Two divisions: `valid_tokens / global_valid_tokens / self._splits` | [Auxiliary-loss scaling](../gift/torchtitan/components/aux_loss.py#L93) |
| 780 | Cast positions from int32 to the labels' int64 dtype | [MTP target packing](../gift/torchtitan/components/loss.py#L932) |
| 785 | `stack`, lowered to `cat` and a view: pack labels and positions into `[B, 2, S]` | [MTP target packing](../gift/torchtitan/components/loss.py#L932) |

The validity comparison is computed twice: once for the model's token mask and
once for the auxiliary-loss count. Auxiliary setup also copies the resulting
scale into its state through `state.scale.copy_(share)`; that copy is outside the
ten-kernel duration sum below.

## Why this prefix is limited by host submission

The first listed kernel starts at **1.010759288 s**; the last ends at
**1.011164346 s**. This spans **405.058 microseconds**, while the ten listed kernel
durations sum to **18.401 microseconds**. This sum is kernel execution time, not
the elapsed cost of the entire preparation path or a device-wide utilization
measurement.

A concrete launch correlation supports the host-submission diagnosis:

| Event | Relative time |
| --- | --- |
| GPU kernel 751 finishes | 1.010911769 s |
| CPU begins launching kernel 754 | 1.010999422 s |
| GPU kernel 754 starts | 1.011008953 s |
| GPU kernel 754 duration | 1.344 microseconds |

There are **87.653 microseconds** after kernel 751 finishes before the CPU even
submits kernel 754. The next kernel then starts 9.531 microseconds after launch
entry. This is evidence of delayed host submission, rather than a backlog of
these tiny GPU kernels. The trace does not isolate Python, PyTorch dispatch,
profiling and host scheduling costs from one another.

The conclusion applies to this prefix. The subsequent embedding and layer weight
all-gathers have their own dependencies; this observation does not establish that
the whole forward or optimizer update is CPU-bound.

## Existing compilation does not cover this region

- Model compilation wraps repeated blocks through
  [`segment.module.compile(...)`](../gift/torchtitan/distributed/compile.py#L107).
- Loss compilation wraps
  [`self.fn`, the cross-entropy function](../gift/torchtitan/components/loss.py#L162).
  It does not compile every method of the loss object.
- The trainer calls
  [input postprocessing, auxiliary scaling and `pack_targets`](../gift/torchtitan/trainer.py#L624)
  before the model forward. These calls remain eager, including MTP packing even
  though that method belongs to the loss object.

Enabling the existing `model` and `loss` compile components therefore does not
fuse the operations listed above.

## Fusion hypothesis

Fusion could reduce the separate kernel launches and associated host-submission
overhead in this preparation region. The current model/loss compile settings do
not cover it. No fused variant or speedup has been measured.

## Reuse BF16 expert weights across microbatches

### Observed baseline behavior

The same trace contains preparation of layer 2's routed expert weights on stream
24 while layer 1 is computing:

| GPU slice | Operation | Duration |
| --- | --- | --- |
| 1572 | Cast the local expert weights from FP32 into a BF16 input buffer | 588.002 microseconds |
| 1575 | Copy 603,979,776 bytes (576 MiB) from that BF16 input into the BF16 output buffer | 398.274 microseconds |

The baseline has `efsdp=1`: these operations prepare weights already resident on
the same GPU. Slice 1575 is a local device-to-device copy under the
`nccl:_all_gather_base` annotation.

The [BF16 input buffer is allocated per parameter group](../gift/torchtitan/distributed/gfsdp/_param_group.py#L499)
and remains allocated between microbatches. For each MoE layer, it contains all
experts owned by this rank. The
[BF16 output uses the shared working-buffer pool](../gift/torchtitan/distributed/gfsdp/_param_group.py#L418).
The current path is therefore:

```text
FP32 expert weights -> per-layer BF16 input -> shared BF16 output -> compute
```

After backward, [GFSDP clears the input-validity flag](../gift/torchtitan/distributed/gfsdp/_param_group.py#L880),
without releasing the BF16 input storage. The next microbatch repeats the
[FP32-to-BF16 cast](../gift/torchtitan/distributed/gfsdp/_collectives.py#L54).
Perfetto inspection found 32 such casts per MoE layer in this update. The
[optimizer runs after all 32 microbatches](../gift/torchtitan/trainer.py#L807),
so the expert weights do not change between them.

### Idea

Prepare the BF16 expert-weight representation once per optimizer update and reuse
it across the 32 microbatches, including backward and activation recomputation.
Refresh it when the underlying weights change, while keeping the FP32 master
weights.

For `efsdp=1`, compute could consume the persistent per-layer BF16 buffer directly
to also avoid the extra input-to-output copy. Simply skipping later copies with
the current arrangement would not work: other layers overwrite the shared output
buffer. This idea reuses BF16 storage already allocated by the current baseline;
repeated casting currently does not free that storage.

Follow-up, 14 November 2026: the opt-in implementation and focused GFSDP checks
pass. Rank-0/1 traces confirm one cast per expert group per update and removal
of the singleton copies. One timing candidate is 1.99% above historical V2;
causal speedup remains inconclusive without a contemporary control/noise estimate.
The [completed report](../runs/bf16-expert-cache/report.md) owns the evidence and
limits. The feature stays disabled by default. These crop results do not
establish behavior or performance of production PP4/EP32.

## FP8 all-gather for sharded routed-expert weights

**[IMPORTANT] High priority for multi-rank expert FSDP.**

### Scope and current path

Owner-selected research candidate, 18 November 2026. Start on one NVLink-connected
node with eight H100s, PP/TP/CP1, dense FSDP8, EP4 and eFSDP2, with no separate DP
replicas. Use the selected recipe as the starting point, including the production
tokenizer and real-data mixture, 32 microbatches per update, model and loss compile,
the per-layer SAC policy, MTP and EMA. Compare the existing BF16 expert gather
against FP8 expert gather on that same EP4/eFSDP2 topology.
[Selected recipe](baseline.md#current-final-task-baseline) ·
[Data contract](baseline.md#global-requirement-real-data-and-production-tokenizer)

EP assigns 32 of the 128 routed experts to each EP lane. The eFSDP pair shards the
weights for that lane across two GPUs. For rank 0, eFSDP gathers weights and
reduces weight gradients on `{0, 4}`, while HybridEP exchanges routed activations
with `{0, 1, 2, 3}`. Target only `w13` and `w2` of the routed experts in the seven
MoE layers. Attention, routers, shared experts, and dense FFNs use dense FSDP.
[Mesh construction](../gift/torchtitan/distributed/parallel_dims.py#L164) ·
[Expert parameter groups](../gift/torchtitan/models/gift_moe/_base/parallelize.py#L284)

```text
Current   FP32 master shard -> BF16 input shard -> BF16 weight all-gather
          -> quantize the full weights -> DeepGEMM

Proposed  FP32 master shard -> BF16 rounding -> local E4M3 + FP32 scales
          -> store until the next weight write -> all-gather data and scales
          -> DeepGEMM reads views of the gathered output
```

The current configuration enables FP8 expert GEMMs and still uses BF16 weight
all-gather. The BF16 expert cache requires an expert shard group of size one.
With eFSDP2, the ordinary gather casts the local FP32 shard into an existing BF16
input buffer and reuses that input within the microbatch. Backward finalization
clears its validity for the next microbatch. HybridEP quantizes the full BF16
weights when its dispatch path runs, including SAC replay where that dispatch
is recomputed. GFSDP has hooks for alternate gather representations, but the
current parameter extension accepts one input tensor and does not gather these
block scales. The expert FP8 gather requires source integration.
[FP8 config](../gift/torchtitan/config/configs.py#L678) ·
[BF16 cache eligibility](../gift/torchtitan/distributed/gfsdp/_param_group.py#L619) ·
[Gather input](../gift/torchtitan/distributed/gfsdp/_collectives.py#L54) ·
[Input invalidation](../gift/torchtitan/distributed/gfsdp/_param_group.py#L1130) ·
[Gather extension](../gift/torchtitan/distributed/gfsdp/_param.py#L287) ·
[Current expert quantizer call](../gift/torchtitan/distributed/deepep/hybridep.py#L323)

### Exact EP4 layout and precision

The model has width 4,096 and expert intermediate width 1,536. In the current
flat expert group, `w13` precedes `w2`. The equal split is at element 301,989,888,
exactly after the first 24 `w13` expert matrices. The second shard contains the
remaining eight `w13` matrices and all 32 `w2` matrices. There is no padding at
this split. Both ranks own complete 128×128 tiles, so the local tile maximum is
the same maximum used after a BF16 gather. No inter-rank reduction of tile
maxima is needed for this layout. Check the actual flat slices during setup.
For other shapes or shard degrees, require complete tiles in each matrix's
physical layout and keep the existing BF16 route when the layout is ineligible.
[Weight shapes and registration order](../gift/torchtitan/models/common/moe/experts/gmm.py#L121) ·
[Parameter collection order](../gift/torchtitan/distributed/gfsdp/_init.py#L164) ·
[Flat shard layout](../gift/torchtitan/distributed/gfsdp/_param_group.py#L327)

| Tensor | Value shape for the local EP lane after gather | FP32 scale shape | FP8 data elements | FP32 scale elements |
| --- | --- | --- | ---: | ---: |
| `w13` | `[32, 3072, 4096]` | `[32, 24, 32]` | 402,653,184 | 24,576 |
| `w2` | `[32, 4096, 1536]` | `[32, 32, 12]` | 201,326,592 | 12,288 |
| Full group | | | 603,979,776 | 36,864 |
| One eFSDP shard | | | 301,989,888 | 18,432 |

Use exactly the current `FP32 → BF16 → E4M3FN` numerical sequence. Compute the
tile maximum and the power-of-two FP32 scale from **BF16-rounded** values, then
use the same scaling arithmetic, E4M3 conversion, and scale layout as the
existing quantizer.
The first reference can explicitly cast to BF16 and invoke the existing
128×128 quantizer on each contiguous local matrix range. The production
implementation should fold BF16 rounding into the quantizer and write into
the existing cache slices, after proving equality against that reference.
Direct FP32-to-FP8 quantization can change both scales and FP8 values.
[Current scale arithmetic and output layout](../gift/torchtitan/ops/float8/triton_kernels/block_quantization.py#L64)

### GFSDP integration and buffer ownership

1. **Cache each local shard per weight generation.** Replace the persistent
   BF16 gather input with E4M3 bytes and FP32 scales. Refresh once before first
   use of a new FP32 master version. Invalidate on optimizer writes, checkpoint
   load, EMA swap or swap-back, and backing replacement. Order the refresh
   after the master write and after any pending reader of the old cache. Keep
   FP32 masters and persisted optimizer, model and EMA state in their current
   formats. Treat the FP8 cache as derived runtime state.
   [Existing invalidation and event pattern](../gift/torchtitan/distributed/gfsdp/_param_group.py#L655)
2. **Replace the two shared receive slots.** Gather FP8 bytes as `uint8` and
   scales as FP32 in two fixed-order collectives on the same eFSDP group. Split
   the resulting byte and scale arrays at 402,653,184 bytes and 24,576 floats
   and take contiguous `w13` and `w2` views. The weight data requires no repack.
   Represent the two planes as one expert group's weight gather. Keep the
   FP32 gradient bucket layout independent because GFSDP currently uses the
   gather dtype map to iterate gradient buckets.
   Each consumer waits for both planes. Retain GFSDP's completion and
   output-slot release rules, including asynchronous `Work.wait()` semantics
   where applicable. The small scale all-gather adds a launch whose cost
   belongs in the measurement.
   [Gather completion](../gift/torchtitan/distributed/gfsdp/_collectives.py#L24) ·
   [Gradient bucket iteration](../gift/torchtitan/distributed/gfsdp/_param_group.py#L1094) ·
   [Slot ownership](../gift/torchtitan/distributed/gfsdp/_param_group.py#L866) ·
   [NCCL rank order](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allgather)
3. **Feed the existing DeepGEMM path.** Use the gathered `(data, scale)`
   planes at the expert-weight integration point and bypass the full-weight
   quantizer in HybridEP. Make the prepared weights available to the expert
   call so that it also bypasses the fallback quantizer. The existing
   `quantize_expert_weight(w, (data, scale))` accepts precomputed values and
   returns the weight gradient to the original edge. Adapt GFSDP's
   unsharded-parameter construction, which now builds that edge from the
   gathered BF16 view. Provide the logical BF16 gradient edge without a
   full BF16 weight backing. Pass the prepared planes to grouped GEMM
   alongside the dispatch metadata and leave the dispatch weight outputs
   empty on this path. SAC can retain those outputs across slot reuse.
   Preserve BF16 `dW` and the same FP32 packing, reduce-scatter and gradient
   accumulation. The gathered weights and scales use the existing DeepGEMM
   format. Activation dispatch, combine and token-drop behavior stay on
   their current paths.
   [Unsharded parameter construction](../gift/torchtitan/distributed/gfsdp/_param.py#L209) ·
   [Autograd adapter](../gift/torchtitan/ops/float8/quantize_expert_weights.py#L23) ·
   [Expert fallback](../gift/torchtitan/models/common/moe/experts/gmm.py#L252) ·
   [DeepGEMM consumer and weight-gradient path](../gift/torchtitan/ops/float8/grouped_gemm.py#L109)
4. **Resolve backward weights from the active slot.** Grouped GEMM currently
   saves actual FP8 data and scale tensors for `dgrad`. The SAC policy for
   layer 5 also retains the dispatch output under `save:experts_up+comm`.
   Saved references to a recycled receive slot can read another layer's
   weights. Bind backward and SAC replay to the current layer and weight
   generation, using the existing `pre_backward` unshard to supply live
   views before either consumes them. Keep a slot owned until its last
   consumer completes, and keep the existing dgrad transpose in the cost
   accounting. Verify this path explicitly under compile and slot reuse.
   [Saved weight tensors](../gift/torchtitan/ops/float8/grouped_gemm.py#L122) ·
   [Backward unshard](../gift/torchtitan/distributed/gfsdp/_param_group.py#L989) ·
   [Per-layer SAC](../gift/configs/experiments/gift_100b_crop8_ep8_fp8_compile_cute.yaml#L99) ·
   [Dispatch classification](../gift/torchtitan/distributed/activation_checkpoint.py#L87)

Megatron Core documents `--fp8-param-gather` with its blockwise E4M3 recipe
and sharded FP32 main weights. Use it as a reference for quantized weight
storage and gather lifecycle. GIFT uses its own BF16 rounding boundary,
GFSDP flat layout, DeepGEMM wrappers and SAC lifetimes, so the proposed
integration follows those contracts.
[Megatron Core 0.20 MoE guide](https://docs.nvidia.com/megatron-core/developer-guide/0.20.0/user-guide/features/moe.html) ·
[Megatron blockwise recipe](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/low_precision_training.html#blockwise-deepseek-style-block-scaling)

### Calculated GPU memory savings

Each local input is 576 MiB in BF16 or 288 MiB of FP8 plus 72 KiB of FP32
scales. Each full output is 1,152 MiB in BF16 or 576 MiB of FP8 plus 144 KiB
of scales. Count one persistent local input per MoE layer and two shared
output slots per expert shard group.

| Routed-expert weight buffers per GPU | BF16 | FP8 + FP32 scales | Saved |
| --- | ---: | ---: | ---: |
| Seven persistent local inputs | 3.9375 GiB | 1.9692 GiB | 1.9683 GiB |
| Two shared full outputs | 2.2500 GiB | 1.1253 GiB | 1.1247 GiB |
| **Total** | **6.1875 GiB** | **3.0945 GiB** | **3.0930 GiB** |

This is the planned reduction in resident gather storage relative to the
EP4 BF16 control. Holding one additional full FP8 group costs 576.14 MiB.
Holding all seven in addition to the two shared slots costs 3.94 GiB and
can erase the storage saving. A prototype with both BF16 and FP8 output
pools would save only about 0.84 GiB after replacing the seven local inputs,
before scratch. The integrated target replaces both pools. Measure peak
allocated and reserved separately, including cache refresh, saved tensors,
SAC, and transpose scratch. Count any reduction in quantizer temporaries
only after verifying their lifetime and contribution to the peak.
[Persistent gather inputs](../gift/torchtitan/distributed/gfsdp/_param_group.py#L707) ·
[Shared output pools](../gift/torchtitan/distributed/gfsdp/_state.py#L344)

### Modeled step-time effect

The selected EP8 source and trace are evidence for operation counts and
kernel costs. For the trace use `capacity-selector-001-trace`, update 21,
rank 0, three microbatches, SHA-256
`dbd9da0ddfb950c5c7f4e28e388bbac48879651596782d88945eab7fb687a5ad`.
The capture has 42 CPU expert all-gather scopes, two per layer per
microbatch. This includes layers 6 and 7. The root callbacks from the two
head-loss backward chunks reshard expert groups before decoder backward,
including the tail groups eligible for automatic retention after forward.
Use **448 logical gathers per rank per update** as the EP4 planning case
with the same schedule. If a schedule retains both tail groups through
those callbacks, the sensitivity case is 384 gathers. Record the actual
EP4 count and forward/backward split.
[Selected trace](../artifacts/4fix_qla_embed_selector_gpu.json) ·
[Trace receipt](../runs/capacity-selector/report.md#native-trace-mechanism) ·
[Chunked head backward](../gift/torchtitan/components/loss.py#L761) ·
[GFSDP root callback](../gift/torchtitan/distributed/gfsdp/_state.py#L694)

| Work removed | Estimate per rank per update | Basis |
| --- | ---: | --- |
| Weight all-gather traffic | **125.97 GiB** received and the same sent at 448 gathers | Peer bytes per gather fall from 576 to 288.07 MiB. At 384 gathers, the saving is 107.97 GiB per direction. |
| Serialized payload transfer time | **0.45 s** at 448 gathers, **0.39 s** at 384 | Model at 300 GB/s effective directional bandwidth. The large dense head gathers in the EP8 reference trace give 242–295 GB/s. The EP4 pair and additional scale launch need measurement. |
| Repeated full-weight FP8 quantization | **About 0.26 s** of kernel work | The trace shows 39 `w13`/`w2` pairs totaling 12.383 ms across three microbatches. The model doubles bytes for the EP4 full group and scales to 32 microbatches. Replacing these calls with seven local shard refreshes still costs a few milliseconds. |
| Repeated local FP32-to-BF16 casts | **Up to about 0.13 s** of separate kernel work | Seven one-time casts in the reference total 4.143 ms for the same local shard size. The EP4 ordinary input path refreshes once per microbatch. The upper estimate assumes the new quantizer fuses BF16 rounding. |

For planning, assume 25–50% of the gather and local-cast savings and
50–75% of the FP8-quantization savings translate into distinct reductions in
critical-path time after overlap. This gives approximately **0.26–0.49 s
per update** across the two gather-count cases. With the measured EP8
reference of 12.2301 s per update, the illustrative throughput gain is
**+2.2–4.2%** using `T / (T − saved_time) − 1`. The EP4 step time,
effective bandwidth, exposed waits, launch overhead and memory peaks need
their own A/B. The FP32 expert reduce-scatter remains in the measurement.
[EP8 native timing](../runs/capacity-selector/report.md#native-1010-result) ·
[NCCL all-gather bandwidth convention](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allgather)

### Validation sequence

1. **Isolate the EP4/eFSDP2 component first.** Compare the real `w13` and
   `w2` shard shapes on all four pairs. Measure GPU and host time for
   baseline gather plus full quantization, FP8 data plus scale gather, and
   local cache refresh separately. Capture allocated and reserved peaks,
   physical buffer sizes, message volumes, and the slowest rank. Compare
   gathered FP8 bytes and FP32 scales bitwise against
   `BF16 all-gather → existing quantizer`, including tile values at
   rounding boundaries.
2. **Validate the affected training path.** Compare expert forward outputs
   and `dX`/`dW`, the FP32 reduction and accumulation across 32
   microbatches, and the resulting FP32 master update. Exercise the normal
   same-slot reuse across layers 1 → 3 → 5 → 7. Verify layer 5 backward
   after layer 7 overwrites its old forward slot. Exercise ordinary SAC
   replay and layer 5's `save:experts_up+comm` path with compile enabled, both weight
   orientations, empty expert token groups, and the unchanged token-drop
   behavior. Check repeated cache hits, optimizer refresh, EMA swap and
   swap-back, and checkpoint continuation.
3. **Measure native EP4 against native EP4.** After the component passes,
   run matched unprofiled 10-warmup + 10-measured pairs on the production
   tokenizer and real data. Keep the same accumulation, LR schedule,
   parallelism, SAC and compile. Record tokens/s and per-rank allocated and
   reserved peaks. Use a matched profile to verify FP8 payload and scale
   gathers, removal of the repeated full-weight casts, weight gathers
   during forward and backward, and whether shorter gathers reduce compute
   waits or contention. Check the step series for drift, and repeat the
   pair if drift or noise prevents resolving the effect.
   [Validation policy](../planning/validation.md)

The priority is to test the multi-rank path with the existing mathematical
recipe and shared memory budget. Implementation and EP4 GPU validation are
open. The calculated buffer reduction and throughput scenario are planning
inputs.

## Direct expert weight gradients into the FP32 reduce-scatter buffer

**[IMPORTANT] High priority for multi-rank expert FSDP.**

Start with the same eight-H100 crop as the FP8 all-gather candidate, with
PP/TP/CP1, dense FSDP8, EP4, eFSDP2, 32 microbatches per optimizer update,
compile, and the current per-layer SAC policy. Scope the change to routed
experts' `w13` and `w2` in the seven MoE layers. This targets gradient
production and can be tested with either BF16 or FP8 weight all-gather.
[Expert parameter groups](../gift/torchtitan/models/gift_moe/_base/parallelize.py#L284)

### Current path and proposed change

DeepGEMM's K-grouped weight backward accumulates into zero-initialized
BF16 `dW13` and `dW2` tensors. Autograd delivers those gradients to GFSDP.
In `post_backward`, GFSDP acquires the shared gradient slot and uses
`torch._foreach_copy_` to widen the gradients into its flat FP32
reduce-scatter input. The two parameter-shaped FP32 views already exist.
With PP1, the routed-expert group reduces each microbatch's gradients
and accumulates the resulting FP32 shard across the 32 microbatches.
[DeepGEMM backward](../gift/torchtitan/ops/float8/grouped_gemm.py#L221) ·
[FP32 input views](../gift/torchtitan/distributed/gfsdp/_param_group.py#L519) ·
[Gradient packing](../gift/torchtitan/distributed/gfsdp/_param_group.py#L1000) ·
[Sharded accumulation](../gift/torchtitan/distributed/gfsdp/_collectives.py#L177)

Write `dW13` and `dW2` from the SM90 GEMM epilogue directly into those
views. Apply the current BF16 round-to-nearest-even to the FP32 accumulator
and widen that rounded result to FP32 before storing it. The current
DeepGEMM FP32 store skips this rounding. The current BF16 store rounds and
uses TMA reduce-add into a zeroed destination. Implement an overwrite for
the eligible single-contribution case and explicitly zero the regions for
experts with no tokens, including the all-empty batch. Preserve the current
BF16 baseline values, including its zero and special-value behavior.
Keeping a full FP32 zero fill and destination reduce-add can consume the
bandwidth saved by removing packing.
[GIFT accumulation call](../gift/torchtitan/ops/float8/deep_gemm_handle.py#L99) ·
[Pinned SM90 epilogue](../../hybrid-ep-overlap/sources/donors/deepgemm-baseline/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh#L350)

Acquire the existing expert gradient slot before the first `dW` write,
after its previous reduce-scatter and follow-up shard accumulation release
it. Keep independent `dX` and weight-gradient operand preparation ahead of
that wait where dependencies allow. Publish readiness once per weight to
GFSDP and enqueue the same FP32 reduce-scatter after both writes complete.
Keep the collective's precision, scaling, ordering, and FP32 shard
accumulation. The slot remains owned until reduction and accumulation finish.
Retain the ordinary gradient route for unsupported layouts, deferred
reductions, multiple contributors, and consumers that require a materialized
autograd gradient. Verify the direct path's mutation and readiness under
compile and SAC. A second full FP32 slot would add 2.25 GiB per GPU, so the
candidate uses the existing slot budget.
[Current buffer acquisition and release](../gift/torchtitan/distributed/gfsdp/_param_group.py#L1000) ·
[Reduction waits and release event](../gift/torchtitan/distributed/gfsdp/_collectives.py#L123)

### EP4 traffic, memory, and timing estimate

The 32 routed experts per EP lane produce 603,979,776 gradient elements
per MoE layer. There are 224 group reductions per rank per update, with
two `dW` producers per group.

| Cost per GPU | Per MoE layer per microbatch | Per optimizer update |
| --- | ---: | ---: |
| Current packing, BF16 read plus FP32 write | 3.375 GiB | 756 GiB |
| Additional producer writes from wider FP32 stores | 1.125 GiB | 252 GiB |
| Net logical output and packing traffic removed | 2.250 GiB | 504 GiB |

The direct path removes up to **1.125 GiB** of live BF16 `dW` storage for
one layer on a GPU. The existing **2.25 GiB** FP32 reduce-scatter input and
the sharded gradients remain allocated. The peak allocated and reserved
savings depend on where the temporary gradients are live. The traffic
calculation excludes initialization, TMA destination reads, and network
traffic. The reduce-scatter payload is the same.

In `capacity-selector-001-trace`, update 21, rank 0, the three captured
EP8/eFSDP1 microbatches contain 21 expert
`FSDP::post_backward_accumulate` regions. Their 63 BF16-to-FP32 packing
kernels total **13.656 ms** on stream 7, with zero interval overlap with
kernels, copies, or memsets on other streams in that capture. Doubling the
gradient size and scaling to 32 microbatches gives **291 ms** of modeled
packing work on EP4. At the same effective bandwidth, charging for the
wider GEMM stores leaves **194 ms** of modeled savings. If half to all of
this budget reduces elapsed time, the planning range is **about 100–195 ms
per update**. The observed EP8 step of 12.2301 s gives an illustrative
**+0.8–1.6% throughput**. EP4 may expose an earlier wait on the shared slot,
and the new epilogue can change GEMM performance. Measure both on EP4.
[Source Kineto trace](../artifacts/4fix_qla_embed_selector_gpu.json) ·
[Trace receipt](../runs/capacity-selector/report.md#native-trace-mechanism) ·
[Native timing reference](../runs/capacity-selector/report.md#native-1010-result)

### Validation sequence

1. **Benchmark the EP4 component.** Use the actual 32-expert `w13` and `w2`
   shapes. Compare current BF16 GEMM plus packing, BF16 overwrite plus
   packing as a control for the store change, and rounded direct FP32
   overwrite. Compare the full FP32 collective input to the current path.
   Include empty experts, all-empty batches, and BF16 rounding boundaries.
   Measure from `dW` operands ready to reduce-scatter launch, including
   explicit zeros and slot waits, and verify that the direct path has no
   full-size temporary or hidden compiler copy.
2. **Validate the training path.** Run with the existing compile and SAC
   policy. Verify single delivery of both gradients, safe slot reuse across
   adjacent expert layers, and the ordinary fallback. Compare local FP32
   shards after reduction, accumulated gradients after 32 microbatches,
   clipping, and the FP32 master update.
3. **Measure native EP4 A/B.** Use matched unprofiled 10-warmup + 10-measured
   runs on real data with the production tokenizer. Compare throughput and
   per-rank allocated and reserved peaks. In a separate matched profile,
   confirm the expert packing kernels disappear and compare the timing of
   each first `dW` write, previous slot release, and reduce-scatter launch.
   [Validation policy](../planning/validation.md)

## BF16 expert gradient exchange with FP32 owner reduction

**[IMPORTANT] Test on multi-rank expert FSDP with strict FP32 update parity.**

### Current path and proposed change

For routed-expert `w13` and `w2`, DeepGEMM already produces BF16 `dW`. GFSDP
widens each rank's full contribution to FP32, then runs FP32 reduce-scatter.
Each rank accumulates only its resulting FP32 shard across microbatches.
Expert and dense groups use different meshes. Change only the expert gradient
collective, independently of the precision used to gather expert weights.
[DeepGEMM output](../gift/torchtitan/ops/float8/grouped_gemm.py#L221) ·
[GFSDP gradient packing](../gift/torchtitan/distributed/gfsdp/_param_group.py#L1000) ·
[Reduction and sharded accumulation](../gift/torchtitan/distributed/gfsdp/_collectives.py#L123)

Use the current flat gradient layout to split each BF16 contribution into
equal slices, one per eFSDP owner. Exchange the slices on the **eFSDP group**.
Each owner receives one contribution from every peer and includes its own
local contribution. Convert each slice exactly to FP32, sum in FP32 and
produce the same kind of FP32 shard as today. Preserve padding, zero
contributions from empty experts, the current gradient divide factor of one,
and the local shard's existing copy-or-add sequence. The first microbatch
writes the reduced shard, and later microbatches add a separately computed
FP32 reduction result. Keep the existing optimizer, clipping and persisted
state contracts. The initial scope is PP1, TP1, CP1, with no separate DP
replica group and the selected compile and SAC settings.
[Expert mesh](../gift/torchtitan/distributed/parallel_dims.py#L164) ·
[Gradient scaling](../gift/torchtitan/models/gift_moe/_base/parallelize.py#L368)

This is an alternative to the direct FP32 GEMM store for the same gradient
collective. With BF16 exchange, the natural producer is the existing BF16
`dW`. Widening the full gradient in the GEMM and narrowing it before exchange
would add traffic. Keep the FP32 collective as the reference and fallback.

### Scaling estimate

For a group of size `P`, a full BF16 contribution of size `B` and an FP32 ring
reduce-scatter on the same elements, the per-rank, per-direction peer payload
is `B × (P − 1) / P` for BF16 exchange and `2B × (P − 1) / P` for the ring.
This is a logical payload calculation. Fabric bytes, message startup, NCCL
SM use, packing, the owner sum and lost overlap all affect elapsed time.

| PP1 topology | Routed experts per rank | Full BF16 `dW` | Current FP32 ring per direction | BF16 exchange per direction |
| --- | ---: | ---: | ---: | ---: |
| 8 GPU, EP4, eFSDP2 | 32 | 1,152 MiB | 1,152 MiB | 576 MiB |
| 128 GPU, EP32, eFSDP4 | 4 | 144 MiB | 216 MiB | 108 MiB |

At EP32/eFSDP4, an owner keeps its own 36 MiB BF16 slice and receives three
36 MiB peer slices. The saving is 108 MiB per layer per microbatch in each
direction. With seven MoE layers and the current global/local batches of
512/2, the 128-GPU configuration has **two microbatches per update**, so the
saving is **1.48 GiB per GPU per direction per update**. With 32 microbatches
and a global batch of 8,192 it is **23.625 GiB**. The 8-GPU EP4 run keeps 32
microbatches at global batch 512 and saves **126 GiB** per direction per
update. Under the current rank order, rank 0's expert shard peers at EP32 are
0, 32, 64 and 96. Confirm the physical placement at launch. Contiguous rank
packing on eight-GPU nodes puts those four peers on different nodes.
[Crop batch sizes](../gift/configs/experiments/gift_100b_crop8_ep8_fp8_compile_cute.yaml#L77) ·
[Accumulation formula](../gift/torchtitan/trainer.py#L223)

Domyn implemented this approach for a 1T MoE on 384 GH200s with EP4 and
cross-node expert FSDP. After adopting FP8 weight gathers, they reported
**+14% tokens/s per GPU** from the BF16 gradient exchange. They reported
bitwise agreement in their four-rank tests and described the general result
as dependent on summation order. This supports testing the approach on Hopper.
Our step-time gain and exact agreement on our topology remain open.
[Domyn implementation report](https://www.domyn.com/blog/training-1t-moe-eu-public-infrastructure-pytorch#bf16-gradient-all-to-all-with-fp32-accumulation)

### Precision, memory and integration

- **Match the actual FP32 reduction order.** Lossless BF16 transport preserves
  the producer's bits. The FP32 sum of four contributions can round
  differently when the additions are reassociated. Establish NCCL's actual
  channel-specific order for each owner and shard range on the pinned runtime
  and topology.
  Compare FP32 shards bitwise after one microbatch and after accumulation,
  then compare clipping and the FP32 optimizer update. Use the FP32 collective
  on configurations where exact parity has not been established. A BF16
  output collective would add rounding even with FP32 internal accumulation.
  [NCCL ring reference](https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/device/reduce_scatter.h#L11)
- **Keep the existing gradient-slot budget.** The current flat FP32 slot
  holds `2B` bytes. A candidate layout uses two disjoint BF16 views of size
  `B` for packed sends and received slices, including the self slice. Pack
  the ordinary `w13` and `w2` outputs directly into the send view. At EP32
  this is 144 MiB plus 144 MiB inside the existing 288 MiB slot. GFSDP's
  existing FP32 output alias points at the rank's 72 MiB slice. To reuse
  it, put sends in the first half on group ranks 0 and 1 and the second
  half on group ranks 2 and 3. Put receives in the other half. After the
  exchange has finished reading the send half, a kernel can widen and sum
  the received BF16 slices in registers and write the FP32 result to the
  existing output alias. Then use the existing FP32 sharded accumulator.
  This avoids materializing a widened receive plane. The row-owner embedding
  already passes disjoint views of one GFSDP allocation to all-to-all.
  Verify this layout's ownership, alignment, allocation and compile behavior
  in the expert path. Measure BF16 packing before changing the producer.
  A direct BF16 write can remove packing if acquiring the slot earlier
  preserves backward overlap.
  [FP32 output aliases](../gift/torchtitan/distributed/gfsdp/_param_group.py#L550) ·
  [Row-owner exchange](../gift/torchtitan/distributed/gfsdp/row_owner_embedding.py#L353)
- **Preserve asynchronous reuse.** Acquire the slot after the previous
  consumer releases it, publish the BF16 inputs from the compute stream,
  order the exchange before local reduction, and release the slot after
  FP32 shard accumulation completes. Ensure all ranks issue the same
  collective sequence with empty experts and during SAC. Measure overlap
  with weight all-gathers, expert compute and HybridEP. Avoid an extra
  full-size send, receive or FP32 staging allocation.
  [Existing gradient-slot acquisition](../gift/torchtitan/distributed/gfsdp/_param_group.py#L1000) ·
  [Existing reduction stream](../gift/torchtitan/distributed/gfsdp/_collectives.py#L123)

First compare the native FP32 collective and BF16 exchange plus FP32 sum at
the actual EP4/eFSDP2 tensor sizes. Time packing, exchange, local reduction,
accumulation and slot waits. Use the four-rank shape and target rank placement
to verify the scaling and exactness condition before selecting EP32. Follow
with matched native training on the current production tokenizer, real data,
compile and SAC. Record throughput, per-rank allocated and reserved peaks,
and a separate matched trace. The EP8/eFSDP1 trace cannot measure this
multi-rank expert exchange.
[Validation protocol](../planning/validation.md)

## Keep tail weights live across chunked LM-head backward

**[IMPORTANT] Fix the nonfinal GFSDP callback with the existing weight slots.**

### Trace evidence and cause

The selected `capacity-selector-001-trace`, update 21, rank 0, contains three
microbatches on EP8/eFSDP1. Each has two inner LM-head backward callbacks and
one outer callback. The first inner callback reshards the `layers.7` and
`layers.6` expert groups. The outer decoder backward later calls their gather
paths again. Across the capture there are 42 expert gather scopes. The expert
groups are singletons in this capture, so these are logical unshard paths.
The same capture shows **two actual dense DP8 NCCL all-gathers inside the
second head chunk of each microbatch**, one for the MTP parent and one for its
FFN group. Their BF16 output lengths are 99,791,360 and 134,479,872 elements.
[Trace artifact](../artifacts/4fix_qla_embed_selector_gpu.json) ·
[Trace receipt](../runs/capacity-selector/report.md#native-trace-mechanism) ·
[Singleton expert cache path](../gift/torchtitan/distributed/gfsdp/_collectives.py#L46)

GFSDP deliberately keeps the last forward owner of each shared weight slot.
At decoder forward exit, the two expert slots hold the weights of layers 7
and 6. The chunked loss detaches slices of `hidden_states`, runs backward
separately on each head chunk, then propagates the accumulated gradient into
the decoder graph. It marks the inner calls with `is_last_backward=False`.
The current root callback uses that flag for finalization but calls
`post_backward()` for groups that have not entered the inner backward. Their
`reshard()` clears ownership. That is how a head-only backward can evict
expert weights and invalidate materialized or prefetched dense MTP weights.
[Forward retention](../gift/torchtitan/distributed/gfsdp/_state.py#L420) ·
[Detached head backward](../gift/torchtitan/components/loss.py#L709) ·
[Decoder backward bridge](../gift/torchtitan/components/loss.py#L839) ·
[Root callback](../gift/torchtitan/distributed/gfsdp/_state.py#L694)

### Proposed fix and lifetime

Adapt the preserved `vision-runtime` guard. For each nonfinal callback,
leave groups in `IDLE` or `FORWARD` alone. Run the existing fallback
`post_backward()` for a group in `PRE_BACKWARD` whose regular hook has not
run, including the case where its forward inputs did not require gradients.
Reset `POST_BACKWARD` groups without running reduction twice. Preserve
final callback behavior, forward-order tracking, profiling drains, and head
gradient accumulation across chunks. The donor includes a unit test of the
per-group state transitions.
[Donor callback](../references/vision-runtime/source/torchtitan/distributed/gfsdp/_state.py#L607) ·
[Donor unit test](../references/vision-runtime/source/tests/unit_tests/test_gfsdp_partial_backward.py#L11)

The current PP1 order fits in the existing two expert slots.

| Point in the schedule | Expert slot A | Expert slot B |
| --- | --- | --- |
| Decoder forward complete, then both head chunks | Layer 7 retained | Layer 6 retained |
| Layer 7 backward | Layer 7 reads, then releases | Layer 6 retained |
| Layer 6 backward | Prefetch layer 5 into the released slot | Layer 6 reads, then releases |
| Layer 5 backward | Layer 5 reads | Prefetch layer 4 into the released slot |

For dense MTP, preserve the parent and FFN regions and their pending gather
results through the inner head callbacks. These regions already share a
physical slot and the saved buffer inventory shows they are disjoint. Dense
layers 7 and 6 still need their own gathers. Before preserving dense layer-7
prefetch overlap, check its actual destination range against the retained
MTP regions. The selected trace launches a dense layer-7 prefetch after the
MTP FFN release and before the MTP parent release. The inspected source has a
different explicit prefetch chain. The trace order alone cannot establish
that those destinations can coexist.
[Saved region inventory](../runs/ffn-row-owner/native-memory-results.json#L139) ·
[Current explicit prefetch chain](../gift/torchtitan/models/gift_moe/_base/parallelize.py#L373)

Track the buffer owner, the byte range, the release event and any pending
all-gather as one lifetime. The current `reshard()` can clear the slot owner
even when another group owns it, and `wait_for_unshard()` can install a
pending result without reacquiring ownership. Check that a nonparticipant's
cleanup cannot clear another group's slot or release it early. Skip
speculative prefetches whose range is still occupied and preserve the
required wait before a later writer reuses that range. Handle prefetched but
unused groups at the proper final boundary. The fix must fit the existing
slot budget for both BF16 and FP8 expert gathers.
[Ownership transitions](../gift/torchtitan/distributed/gfsdp/_param_group.py#L932) ·
[Prefetch conflict check](../gift/torchtitan/distributed/gfsdp/_param_group.py#L1170)

### Expected effect and validation

For the seven-MoE-layer crop, the proposed schedule removes **two expert
weight gathers per microbatch**, from 14 to 12. This is independent of the
separate opportunity to eliminate the two dense MTP re-gathers recorded in
the trace. The table counts logical expert weight gathers. If the FP8 path
uses separate collectives for values and scales, avoiding a logical gather
also avoids both of those collectives.

| Topology and accumulation | Expert gathers per update | BF16 expert payload avoided per GPU per direction | FP8 expert payload avoided per GPU per direction, including scales |
| --- | ---: | ---: | ---: |
| 8 GPU, EP4/eFSDP2, 32 microbatches | 448 → 384 | 36 GiB | 18.00 GiB |
| 128 GPU, EP32/eFSDP4, 2 microbatches | 28 → 24 | 432 MiB | 216.05 MiB |
| 128 GPU, EP32/eFSDP4, 32 microbatches | 448 → 384 | 6.75 GiB | 3.38 GiB |

At DP8, eliminating both dense MTP re-gathers would additionally avoid about
**391 MiB per GPU per direction per microbatch**, or **12.22 GiB** at 32
microbatches. These calculations use the full output size multiplied by
`(group size − 1) / group size`. They carry no assumption that every avoided
millisecond shortens the step. On a deeper decoder with the same two slots,
the expert opportunity stays at two gathers and becomes a smaller fraction
of total gathers.

1. **Check the callback and buffer lifetime.** Cover two head chunks, both
   retained expert slots, split MTP regions, an in-flight prefetch, an unused
   prefetch and the no-input-gradient fallback. Verify ownership and release
   events across consecutive updates. Exercise existing nonfinal-backward
   callers, including PP no-sync and shape probes, when changing the common
   GFSDP callback.
2. **Check native correctness.** Preserve the selected SAC policy, compile,
   head/MTP losses, accumulation and FP32 optimizer update. Verify head
   reduction occurs once at its existing boundary and each participating
   group publishes its gradients once. Check that the two expert slots and
   the split MTP regions retain and release their actual storage safely.
3. **Check the mechanism and speed separately.** Use a matched EP4/eFSDP2
   rank-level trace to check that expert layers 7 and 6 enter backward with
   valid weights and no extra gather, and that layer 5 prefetch waits for the
   correct release. Check dense MTP separately for the two recorded
   re-gathers and inspect any shift in layer-7 prefetch. Then use matched
   unprofiled native runs to measure throughput and peak memory.
   [Validation protocol](../planning/validation.md)

The callback and gather counts come from the saved trace. The proposed
schedule, memory behavior and performance at eFSDP greater than one still
need native validation.

## Fuse MoE routing metadata preparation

### Observed baseline behavior

The owner's next finding concerns the small kernels after router top-k and before
FP8 token quantization, in the same first microbatch and rank-0 trace identified
above. GPU slice 2276 (`triton_poi_fused_scatter_zeros_like_8`) starts at
1.035268302 s. The capacity check then launches all-reduce slice 2285, followed
by small kernels and long gaps before slice 2370
(`_rowwise_1d_quantize_kernel`) starts at 1.035891728 s.

### What the all-reduce does

Slice 2285 sums the token-assignment counts used by the token-drop capacity
check. Its NCCL metadata records 128 `Long` (int64) elements, eight ranks
`[0, 1, 2, 3, 4, 5, 6, 7]` and a duration of 32.416 microseconds. The input
contains 1,024 bytes of counters.

The [baseline enables `drop_by_group_capacity`](../gift/configs/experiments/gift_100b_crop8_ep8_fp8_compile_cute.yaml#L19).
The [capacity-selection code](../gift/torchtitan/models/common/moe/dispatch/capacity.py#L124)
builds an expert-count histogram, all-reduces it, sums the global counts for each
destination group, and combines the local expert-overload and group-overload
conditions into `needs_drop`.

| GPU slice | Operation |
| --- | --- |
| 2285 | All-reduce the 128 expert counts across the eight EP ranks |
| 2286, `triton_per_fused_slice_sum_view_15` | Sum the global counts into eight destination-group loads, with 16 experts per group |
| 2287, `triton_per_fused_any_bitwise_and_gt_16` | Compute the final Boolean capacity predicate |

### The capacity predicate stops CPU submission

The [conditional capacity custom operator](../gift/torchtitan/models/common/moe/dispatch/capacity.py#L63)
contains `if needs_drop`. Evaluating this GPU Boolean on the CPU produces the
following trace events:

| Slice | Recorded event | Evidence |
| --- | --- | --- |
| CPU 2233 | `torchtitan::conditional_capacity_mask` | Encloses the predicate read and the subsequent clone |
| CPU 2234–2236 | `aten::is_nonzero`, `aten::item`, `aten::_local_scalar_dense` | Convert the device predicate into a host scalar |
| GPU 2288 | `Memcpy DtoH (Device -> Pinned)` | One-byte transfer; 2.208 microseconds |
| CPU 2239 | `cudaStreamSynchronize` | 1.032911660–1.035354507 s; 2.442847 milliseconds |
| CPU 2289 / GPU 2293 | `aten::clone` / device-to-device copy | The no-drop branch returns `valid.clone()` in this occurrence |

Most of the CPU synchronization overlaps the preceding GDN computation. Its full
2.442847 milliseconds must not be counted as additional GPU idle time. It stops
the CPU from submitting later work until the capacity predicate is available,
allowing the GPU queue to drain.

### Eager routing operations then leave the GPU waiting for CPU launches

CPU slice 2297 is `hybridep::dispatch`. Its
[custom-operator implementation](../gift/torchtitan/distributed/deepep/hybridep.py#L243)
calls [the routing-metadata helper](../gift/torchtitan/distributed/deepep/hybridep.py#L288)
before token quantization. The helper,
[`indices_to_map`](../gift/torchtitan/distributed/deepep/routing.py#L6), constructs
dense routing and probability matrices from the top-k expert IDs and scores:

```text
compare IDs -> arange -> add offsets -> mask invalid IDs
-> zero routing map -> scatter -> zero probabilities -> mask scores -> scatter
```

Although the surrounding model is compiled, these operations execute eagerly
inside the opaque custom operator. CPU events include individual `aten::ge`,
`aten::arange`, `aten::add`, `aten::masked_fill_`, `aten::zeros` and
`aten::scatter_` calls with separate CUDA launches. GPU slices 2300–2360
correlate with these launches; their CPU launches are enclosed by dispatch
slice 2297.

A concrete launch correlation establishes delayed host submission:

| Event | Relative time |
| --- | --- |
| GPU scatter 2360 finishes | 1.035803408 s |
| CPU launch 2369 begins submitting quantization kernel 2370 | 1.035883653 s |
| GPU quantization 2370 starts | 1.035891728 s |

The GPU finishes the scatter 80.245 microseconds before the CPU starts submitting
the quantization kernel. The trace does not separate Python, dispatcher,
profiling and host-scheduling overhead, or establish CPU core saturation.

From the start of slice 2286 to the start of slice 2370, Perfetto records:

| Measurement | Duration |
| --- | --- |
| Elapsed interval, 1.035335406–1.035891728 s | 556.322 microseconds |
| Execution of 14 GPU kernels | 27.711 microseconds |
| Execution of three GPU copies | 4.735 microseconds |
| Time without a recorded kernel, copy or memset | 523.876 microseconds (94.2%) |

All recorded GPU work in this interval runs on stream 7 without overlap;
no kernel, copy or memset on another recorded rank-0 CUDA stream overlaps this
interval. The table excludes kernel 2370 itself, which quantizes token
activations for FP8 dispatch and takes 38.176 microseconds.

### Fusion idea and limits of the evidence

Fuse the routing-metadata operations into one GPU kernel that constructs the
routing mask and probabilities directly from the expert IDs and scores. The
observed CPU launch gaps make this a concrete opportunity to reduce overhead in
this region.

The capacity-check all-reduce and the CPU read of `needs_drop` are separate
costs. Fusing metadata construction alone does not remove either. Avoiding the
host predicate read would also require keeping the capacity decision on the GPU
while preserving token-drop behavior.

No fused implementation or speedup has been measured. The 523.876 microseconds
of observed gaps are not established as fully recoverable, and this one interval
does not establish an optimizer-update or whole-training speedup.

## Overlap MTP FFN gradient reduction with the remaining MTP backward

### Decision and evidence scope

The owner selected the next experiment: put MTP FFN `w1`, `w2` and `w3` in a
separate compile and GFSDP group, then launch their reduce-scatter while the
remaining MTP backward executes. This is a selected experiment, not an
implementation or benchmark result. The initial target is FFN
gradient-reduction overlap.

The evidence is from `bf16-cache-trace-001`: V2 with the BF16 expert cache,
optimizer update 21, rank 0, three complete microbatches, including the MTP
reduce-scatter at Perfetto slice 9233. It matches
the events analyzed in the
[native readiness evidence](../runs/bf16-expert-cache/mtp-gradient-readiness-native.json)
and [Perfetto readiness evidence](../runs/bf16-expert-cache/mtp-gradient-readiness-perfetto.json).
The source revision is `6c6cd9c0e41a0a5d3d00e1e252b3c120052d3507`; the native
trace SHA-256 is
`fdcf72769a9ddbad51db80b0bf7663784f7bbf4c72971e3ee8d7925cfc800203`.

### Why a separate FFN boundary is a candidate

The current MTP is one GFSDP unit inside a fullgraph-compiled block. The GFSDP
post-backward hook waits for gradients of the wrapped module's inputs. With the
whole MTP under one parent boundary, that hook withholds the reduce-scatter
opportunity until the MTP input gradients return, even though local FFN weight
gradients are ready earlier. The backward dependency runs through FFN, MLA,
MTP input projection, then input norms and the end of MTP. A separate FFN unit
could expose its input-gradient readiness while those later operations continue.

The relevant integration points are the
[MTP wrapping path](../gift/torchtitan/models/gift_moe/_base/parallelize.py#L241),
[GFSDP post-backward hooks](../gift/torchtitan/distributed/gfsdp/_param_group.py#L1186)
and [gradient readiness handling](../gift/torchtitan/distributed/gfsdp/_param_group.py#L1347),
[block compilation](../gift/torchtitan/distributed/compile.py#L95), and the
[FP8 GEMM path](../gift/torchtitan/ops/float8/cuda_gemm.py#L201).

### Payload and producer timing

Each FFN weight contributes 171 MiB of FP32 reduce-scatter input: 513 MiB for
`w1`, `w2` and `w3`, or about 57.4% of the 893.674 MiB MTP total. These sizes
are the FP32 conversion and packing payload; the producer gradients are BF16.
For comparison, the MLA projections contribute 249 MiB, the input projection
128 MiB, and eight norms 3.674 MiB.

First-microbatch producer times below are trace-relative milliseconds within the
MTP backward interval. “Remaining” columns are elapsed time and recorded kernel
union from that point to MTP backward end. Rows for individual and cumulative
producers overlap; they are subsets and must not be added together.

| Producer point | Time (ms) | Remaining elapsed (ms) | Remaining kernel union (ms) |
| --- | ---: | ---: | ---: |
| `w2`, including unpad/crop clone | 192.305304 | 12.128140 | 11.931374 |
| One of `w1` or `w3` (exact order not recorded) | 194.188287 | 10.245157 | 10.072103 |
| All three FFN weights | 195.800677 | 8.632767 | 8.478657 |
| MLA `wo` | 197.000681 | 7.432763 | 7.316990 |
| Candidate completion of all MLA projection GEMMs | 202.002555 | 2.430889 | 2.388588 |
| Candidate completion of input projection | 203.854210 | 0.579234 | 0.558948 |

Producer-gradient completion is not itself the module-input hook. A conservative
candidate FFN boundary is after fused add/norm-prep kernel ordinal 97. On the
first microbatch that point is 195.864773 ms, leaving 8.568671 ms elapsed and
8.417153 ms of recorded kernel union. The exact add operands are unavailable,
so this is a candidate boundary inferred from the trace, not a measurement of a
separate-module hook.

| Microbatch | Remaining elapsed (ms) | Remaining kernel union (ms) |
| --- | ---: | ---: |
| 1 | 8.568671 | 8.417153 |
| 2 | 8.848800 | 8.700988 |
| 3 | 8.760927 | 8.613922 |

### Existing communication and possible overlap

On the first microbatch, MTP backward ends at 204.433444 ms. Whole-group packing
runs from 204.435620 to 204.938597 ms. The existing MTP reduce-scatter,
Perfetto slice 9233, runs from 204.946757 to 212.433632 ms, a duration of
7.486875 ms. The preceding communication completes at 192.540089 ms, and stream
24 is free when the whole FFN is ready; the same stream-availability pattern
holds across all three microbatches.

This shows a candidate interval in which an FFN group's reduce-scatter could
overlap later MTP backward work. It does not establish that the candidate
8.6 ms window equals 8.6 ms of step reduction: partial-group packing, collective
duration, contention and dependencies have not been measured.

### Trace limits and implementation requirements

All three MTP graphs have 163 native kernels with identical names and order.
The native trace retains the full graph intervals because the Perfetto import
omits some kernels; selected timestamps and ownership were cross-checked between
the native and Perfetto evidence. Mapping parameter groups to source modules and
the backward dependency is supported by the source and trace. Tensor shapes and
generated FX outputs are unavailable, so BF16 per-projection labels are less
certain than the aggregate FFN payload.

The experiment must expose the FFN as its own callable, compile boundary and
GFSDP unit with input-gradient readiness. Splitting only the parameter list
under the opaque parent compile would not establish the required boundary.
Preserve the existing FP8/BF16/FP32 numerical path, selective activation
checkpointing (SAC), gradient accumulation, optimizer behavior and checkpoint
semantics. Changes to compile or SAC boundaries can change fusion, recomputation,
memory use and launch overhead.

Before drawing a performance conclusion, verify per-rank readiness and
consistent collective ordering, FP32 packing/casting and buffer lifetimes,
partial reduce-scatter duration and contention, and loss and gradient agreement
for affected blocks. Then compare repeated, matched unprofiled throughput and
peak memory. A finer split for `w2` may be considered later; it is outside this
initial experiment.

## Investigate lower-volume capacity counters and SAC reuse

### Status and evidence scope

These are research ideas worth investigating, not implementation decisions or
measured speedups. Preserve the current conditional capacity-drop conditions,
quotas, score-based selection, and tie ordering. Do not change the capacity
policy or bypass its current path.

The evidence is `bf16-cache-trace-001`: V2 with the BF16 expert cache, optimizer
update 21, rank 0, and three microbatches. It records 39 capacity all-reduces,
each with 128 int64 counters (1,024 bytes). Their summed intervals are 1.206983
ms across 21 forward calls and 13.840274 ms across 18 backward recomputations,
for 15.047257 ms total in a 1,269.710 ms window (about 1.19%). The trace records
no other local GPU work overlapping the collectives. The
[capacity-window budget](../runs/bf16-expert-cache/capacity-window-budget.json)
and [screening summary](../runs/bf16-expert-cache/capacity-optimization-screening.json)
contain the measurements and scope limits.

These are exposed collective intervals, not predicted step savings. The
collectives retain the same EP participants and synchronization point, and
their lifetimes can include peer waits. Removing or shortening this work may
also move a wait to the following EP dispatch.

### Idea 1: reduce group counters before the EP all-reduce

The current code builds 128 local expert counts, all-reduces all 128, then
reshapes the global counts into `(EP, E / EP)` and sums each group. Keep the
local expert counts for the existing local expert-overload condition, but sum
each destination group locally before the all-reduce. Reducing those `EP`
int64 counters yields the same global group sums and threshold result, without
dividing counts, quotas, or capacity by `EP`. Preserve the current gate exactly:
local expert overflow and global group overload must both hold. The aggregation
is localized in [`select_capacity_routes`](../gift/torchtitan/models/common/moe/dispatch/capacity.py#L134).

For `E=128`, the per-rank collective input becomes 64 bytes at EP=8 (16x
smaller), 128 bytes at EP=16 (8x), and 256 bytes at EP=32 (4x); generally the
reduction factor is `E / EP`. This reduces payload only. It keeps the same
participants and barrier, so it does not predict a 16x, 8x, or 4x reduction in
the measured 15.047 ms.

### Idea 2: reuse the exact capacity result during SAC replay

Run the existing capacity selection in the original forward and retain its
Boolean keep mask for recomputation. Recompute differentiable routing scores
and gradients as usual, and apply the saved mask to the same expert assignments.
The existing mask has shape `[8192, 8]` and costs 64 KiB per layer,
microbatch, and rank. Reuse must preserve mask/assignment alignment, lifetimes,
the current no-drop behavior, capacity quotas, and the existing score selection
and tie ordering.

This needs a granular SAC policy. Saving only
`conditional_capacity_mask` does not ensure that its upstream collective inputs
are reused. Either save the targeted collective and existing mask operation, or
place the current capacity selection behind a functional saved boundary; then
verify that the compiled graph actually reuses both. The existing
[`comm` SAC category](../gift/torchtitan/distributed/activation_checkpoint.py#L85)
also saves large EP dispatch/combine outputs, so enabling it broadly would not
be a metadata-only change.

### Cross-rank limits and next checks

On both recorded ranks, the selected long backward collectives end within 0.5
microseconds: rank 1 starts 115.866 microseconds before rank 0 for Perfetto
slice 58892, and 167.205 microseconds before rank 0 for slice 60411. Ranks 2–7
are absent, so peer arrival, protocol progress, and resource effects remain
unresolved. See the [cross-rank evidence](../runs/bf16-expert-cache/capacity-crossrank-wait.json).
This makes the collective intervals poor proxies for saved step time; any
remaining wait may appear at the following EP dispatch.

If investigated, first check counter and mask identity for no-drop, overload,
and tied-score cases; then confirm compiled SAC graph reuse and all-rank
collective order, measure retained memory, and compare matched unprofiled
performance. This note authorizes research only: no training-source or
configuration edits and no new compute runs are included.

## Owner-selected row-owner input embedding

### Selection, scope, and status

On 14 November 2026, the owner selected full row-owner routing for the input token
embedding as the next research design. The objective is to remove both the full-table
weight all-gathers and the dense full-table gradient reduce-scatter from the embedding
path. Every row remains owned by one rank in the embedding's data-parallel process
group. Ranks route token IDs to those owners for lookup, return the selected embedding
rows, and route coalesced row gradients back to their owners.

The first target is the eight-GPU crop only. It is a proposal, not an implementation,
runtime result, or performance claim. The production topology and its owner group remain
a later compatibility gate. No source, configuration, runner, or training change is
authorized by this design note.

Keep the existing model and training contract: FP32 persistent parameters and gradients,
BF16 embedding compute, `acc32`, one MTP branch, the selected per-layer selective
activation checkpointing (SAC) policy, enabled model and loss compilation, dense fused
AdamW, EMA, and checkpoint/resume. The design is acceptable only if it integrates with
those enabled settings. A result that requires disabling SAC, compilation, EMA, or
another production setting does not meet the selected scope.

### Current embedding and rank contract

The input parameter is a plain, untied `nn.Embedding` table of shape `[128256, 4096]`,
with 525,336,576 elements. `lm_head` is separate. The embedding is one GFSDP parameter
group. The crop's eight-way contiguous row partition contains 16,032 rows per owner, or
65,667,072 elements. This gives 250.5 MiB per owner for the FP32 parameter/master shard
and its dense FP32 gradient, 125.25 MiB for a BF16 local row shard, 1,002 MiB for a
gathered BF16 compute table, and 2,004 MiB for the full FP32 table or gradient.

For this divisible crop only, the row mapping is `owner = token_id // 16032` and
`local_row = token_id % 16032`. The owner is the token embedding's dense FSDP
process-group rank. It is not an EP-local-expert rank and must not be derived from a
global-rank assumption. Resolve the actual mesh, group order, rank membership, and shard
metadata at integration time. The general GFSDP layout can include padding and
alignment, so the crop's simple division is not a universal mapping.

The decoder performs one token lookup before the repeated block loop. That output feeds
the trunk and is also carried into MTP, which shifts the same embedding tensor and
zero-fills its tail; MTP does not perform another lookup or introduce token IDs. Trunk
and MTP derivatives must therefore be combined at the shared embedding output before one
row-gradient exchange. Sending a first-branch gradient early would omit the later MTP
contribution or require a separate, numerically validated accumulation path.

Relevant local contracts are recorded in the [embedding source and trace
contract](../runs/bf16-expert-cache/embedding-local-contract.json). The lookup and MTP
dataflow are visible in [the
decoder](../gift/torchtitan/models/common/decoder_blocks/decoder.py#L135), [the
embedding module](../gift/torchtitan/models/common/embeddings/token_embeddings.py#L12),
and [the MTP block](../gift/torchtitan/models/common/decoder_blocks/mtp.py#L100).

### Baseline communication and numerical boundaries

The baseline keeps original/master shards in FP32, gathers BF16 compute weights, and
uses dense embedding backward. The native dense BF16 weight gradient is copied into a
full FP32 buffer and summed with reduce-scatter into the FP32 local shard. The
configured gradient divide factor is 1.0: this is a SUM with factor one, not an average
by eight or an extra division by the 32 microbatches. Global valid-token normalization
is owned by the loss path.

The crop runs 32 microbatches per optimizer update. After each synchronized microbatch,
the local FP32 shard gradient is initialized from its first contribution and accumulated
with later contributions. Preserve the same normalization, 32-microbatch cadence, FP32
accumulation, dense gradient materialization, optimizer timing, and update order.

Native trace evidence identifies `embedding_dense_backward`,
`compute_grad_weight<c10::BFloat16,long>`, and `sum_and_scatter<c10::BFloat16,long>`
with a BF16 output pointer. This confirms the BF16 output boundary but does not
establish every internal reduction detail of the installed PyTorch build. The
source-rank local duplicate coalescing and BF16 rounding that occur before cross-rank
FP32 communication are part of the numerical contract. FP32 owner accumulation alone
does not prove parity if either that local coalescing or its rounding boundary changes.

In particular, autograd through a deduplicated lookup, inverse map, or `index_select`
expansion is not evidence that duplicate token gradients are coalesced in the same
order, precision, or rounding point as native dense embedding backward. The
implementation must compare that path against the baseline or explicitly provide an
equivalent local accumulation and BF16 output boundary. It must not silently replace
source-local BF16 coalescing with per-occurrence FP32 accumulation at the owner.

One no-extra-rounding seam is available to a narrower backward-only adapter: after
native dense BF16 `unsharded_param.grad` is complete and before GFSDP copies and clears
it, select touched rows and route those already-coalesced BF16 values. This seam is not
the lifecycle of the selected full-owner design: that design excludes the embedding
parameter from GFSDP and must not depend on an unsharded GFSDP gradient buffer.

For full owner routing, produce source-local duplicate-coalesced BF16 row contributions
from the completed shared embedding-output gradient, matching the baseline's local
accumulation and BF16 output boundary. Native dense embedding backward can serve as the
numerical reference or an initial staging path if it can be invoked without the
embedding weight gather. That staging path materializes a full 1,002 MiB BF16 gradient
table per rank before rows are selected, so account for its allocation and lifetime. A
compact local coalescing path may replace it only after affected-path numerical parity
is demonstrated. Widen each source-rank BF16 contribution to FP32 before the owner SUM
and before adding it to the FP32 local gradient. Do not use BF16 collective SUM or BF16
owner accumulation. If a source contribution has already accumulated in FP32, including
a deferred or no-sync path, it cannot be cast to BF16 under this rule; that path needs
separate handling or an explicit support guard.

The owner reduction factor remains exactly one. Do not add `/8`, `/32`, a second loss
normalization, or another averaging step. The order of owner-side FP32 additions can
differ from the current NCCL reduce-scatter order, so assess the resulting elementwise
numerical difference under the scoped deterministic comparison. Do not promise arbitrary
bitwise identity without evidence.

### Proposed forward route

Keep the FP32 row shard as the registered owner parameter. For each actual lookup,
validate signed int64 token IDs against `[0, 128256)` before computing owners or
offsets. Empty-request markers and padding sentinels must not be treated as valid
vocabulary rows. Partition the valid IDs by the embedding owner process group, retaining
the original sequence position and the inverse mapping needed to restore output order.

Within each requester-to-owner segment, deduplicate repeated IDs for the forward lookup.
Send each unique row ID to its owner, let that owner gather the corresponding local
rows, and return values to the requester. Use the retained inverse map to expand the
returned values to the exact original token order. Forward values must preserve the
baseline BF16 compute dtype and shape. The owner can cast selected FP32 rows to BF16 for
lookup. A cached BF16 owner view or selected-row representation is optional; begin
without a cache unless invalidation is proven for optimizer updates, checkpoint loads,
and EMA swaps.

The route metadata is per lookup invocation. Keep immutable IDs, counts, permutation,
unique-row indices, and inverse mapping alive until that invocation's backward has
completed. Do not store a single mutable route slot that a later microbatch can
overwrite. Masked or loss-zero tokens may produce zero derivatives, so the input IDs
define a conservative candidate touched set, not guaranteed nonzero gradient support;
this is sufficient for correct zero-filled dense local gradients.

### Proposed backward route and local accumulation

Wait for the gradient of the shared embedding output after both trunk and MTP branches
have contributed. Sum occurrence gradients for each source-rank, owner, and row using
the baseline-compatible local coalescing and BF16 rounding boundary described above.
Keep an inverse relation from every occurrence to its source-local unique row; the
values sent to an owner must represent the sum of all occurrences of that row on that
source rank.

Route one coalesced row contribution per source-rank/owner/row to the row owner. Widen
every received source contribution to FP32 before summing across the owner group. For
each microbatch, write the result into the owner's dense FP32 row shard: zero rows not
present in the contribution set on the first microbatch, and add the current
contribution to the existing FP32 shard gradient on later microbatches. Preserve a
materialized dense `.grad` even when every contribution is zero. A `None` gradient can
change optimizer behavior and is not a valid substitute for a zero tensor.

The owner exchange must participate in a rank-consistent order, including empty
requesters, empty owners, all-zero gradients, and ranks with no unique rows for a peer.
If an implementation omits empty payload collectives, it must still make that decision
consistently for the entire process group. Error and no-gradient paths must not leave
peers waiting in unmatched collectives.

### Preserve dense optimizer, clipping, EMA, and checkpoint state

Keep the registered owner parameter and a dense FP32 gradient with its current FQN and
full logical shape. Continue using the configured dense fused AdamW path with its
existing moments, weight decay, clipping, and once-per-update step. Do not switch to
`SparseAdam`, LazyAdam, row-only optimizer updates, or optimizer-in-backward. Untouched
rows still take part in dense AdamW semantics, including momentum and weight decay;
zeroing rows in the gradient is not the same as skipping those parameter updates.

The owner shard still needs to participate in global norm clipping. A standard
`Shard(0)` DTensor is a possible ownership representation, but its interaction with
GFSDP `RaggedShard` DTensors, fused AdamW, foreach operations, and clipping has not been
established. Validate the mixed representation before choosing it. Optimizer
construction enumerates registered parameters, so this proposal does not by itself
require a new optimizer.

EMA is a required integration item. Current EMA setup, update, and swap code tracks
GFSDP parameter groups. A parameter excluded from GFSDP will not be covered
automatically. The owner shard needs explicit EMA registration, initialization,
per-update maintenance, swap behavior, checkpoint save/load, and invalidation of any
cached BF16 row representation after a swap. Treating EMA as a final verification-only
check would leave the selected design incomplete.

The main DCP path starts with ordinary module state and substitutes GFSDP-managed
entries; an ordinary owner parameter is not automatically dropped. Preserve the model
FQN, global shape, owner placements and group metadata. Verify a full save/load
continuation across main weights, dense Adam state, and EMA state through the next
optimizer update. A state-dict roundtrip without optimizer/EMA continuation is
insufficient.

### GFSDP integration boundary

GFSDP exposes `ignored_params`, but the current wrapping path includes the token
embedding and root wrapper and builds a prefetch chain. The selected parameter must be
excluded from every enclosing GFSDP wrapper that could shard or reduce it, including the
token-embedding and root paths. Update the prefetch relationship so it does not continue
to assume the owner is a normal GFSDP parameter. Do not let GFSDP and the owner path
both reduce or update the same gradient.

Exclusion also bypasses GFSDP's precision conversion, FP32 accumulation, post-backward
reduction, completion events, and gradient-buffer lifecycle for that parameter. The new
path must own source-row collection, dense FP32 local gradient storage,
first-versus-later microbatch accumulation, asynchronous buffer lifetime, collective
ordering, and readiness before clipping and the optimizer. It must handle no-grad/eval,
frozen parameters, absent backward, exceptions, and zero contributions. Guard
unsupported layouts or execution modes explicitly until they are integrated, including
tied weights, multiple parameter groups, incompatible mesh/placement/padding,
PP/TP/CP/HSDP, offload, and deferred/no-sync accumulation.

Source references for this seam are [`fully_shard` ignored-parameter
support](../gift/torchtitan/distributed/gfsdp/fully_shard.py#L76), [token embedding and
root wrapping](../gift/torchtitan/models/gift_moe/_base/parallelize.py#L215), [root
wrapper construction](../gift/torchtitan/models/gift_moe/_base/parallelize.py#L250),
[optimizer parameter enumeration](../gift/torchtitan/components/optimizer.py#L108), [EMA
setup and update](../gift/torchtitan/distributed/gfsdp/fully_shard.py#L533), and
[ordinary DCP state handling](../gift/torchtitan/distributed/gfsdp/_dcp.py#L383). The
[local owner integration
contract](../runs/bf16-expert-cache/embedding-local-contract.json) records the remaining
source-level limits.

### Compilation and SAC are required behavior

The current decoder performs the embedding lookup before the repeated block loop. Model
compilation wraps eligible repeated blocks, and the SAC policy selects blocks; neither
requirement implies that the whole model or the existing embedding stem is compiled as
one full graph. Keep the embedding at its current stem boundary if the integrated path
works with the enabled block compilation and the exact selected per-layer SAC policy. Do
not add a blanket requirement for a new whole-model fullgraph compile.

At the same time, compilation and SAC are not optional compatibility checks. Test the
owner path with the current enabled model and loss compilation, the same SAC selection,
MTP enabled, FP8 enabled, and `acc32` unchanged. Do not silence a compile or replay
failure by disabling a component or changing the SAC policy. Preserve the graph boundary
the baseline actually uses and record which part of the embedding route remains eager or
captured.

Any mutable routing state, owner count, permutation, or communication tensor must belong
to one forward invocation and remain valid until its backward. SAC replay must use the
same route and values, and must not replay or mismatch mutable collectives. Activation
recomputation must not issue duplicate owner communication or alter collective order.
Because MTP reuses the original embedding output, replay must not invent another token
lookup or a second set of embedding IDs.

If a custom operator is needed, provide correct schema, autograd, fake/meta behavior,
and eager/compiled tests for that operator. Do not rely on a static dummy zero-gradient
tensor, stale global route context, or unverified graph capture behavior. A custom
operator is not required if a module boundary and the supported eager stem already meet
the integration contract.

The source integration points are the [decoder
lookup](../gift/torchtitan/models/common/decoder_blocks/decoder.py#L135), [block
compilation](../gift/torchtitan/distributed/compile.py#L95), [SAC block
selection](../gift/torchtitan/distributed/activation_checkpoint.py#L362), and [MTP input
reuse](../gift/torchtitan/models/common/decoder_blocks/mtp.py#L100).

### Routing counts, padding, skew, and memory bounds

The crop has 8,192 input tokens per rank (`B=2`, `S=4096`), with signed int64 IDs. The
absolute unique-row upper bound per requester is therefore 8,192. At hidden width 4,096,
an unpadded compact payload at this upper bound is 64 MiB in BF16 or 128 MiB in FP32;
the ID array itself is at most 64 KiB. These are compact useful-row bounds before owner
padding and protocol overhead.

The inspected NeMo exact-count route gathers a `P x P` count matrix, uses equal-split
padding to the global maximum count, and reads count metadata to the host. At `P=8`, the
count matrix is 512 bytes. Its strict validation adds more device-host synchronizations.
With one padded segment for each of eight owners, the BF16 row-value buffer can reach 8
x 8,192 x 4,096 x 2 bytes, or 512 MiB per rank for one route phase. Returned forward
values and backward gradient values are separate phases. FP32 staging would double the
corresponding value-buffer bound. Actual allocation and peak overlap depend on
implementation lifetimes and must be measured.

Raw token IDs are not a uniform hashed workload. A frequent token or skewed owner can
make one requester-to-owner segment set the maximum padding for every peer. Count
metadata, useful row bytes, padded bytes, local duplicates, and owner skew must be
measured separately. The 64 MiB compact BF16 bound does not bound NeMo's padded network
buffer and cannot be read as a speedup prediction.

Capacity mode is not a safe general answer by itself. The inspected NeMo mode calibrates
from the maximum count of its first lookup, multiplies by 1.25, aligns to 64 with a
minimum of 64, then uses a fixed capacity with an overflow assert. It does not establish
that a first batch bounds later batches. Never drop rows silently. A chosen count
strategy must be rank-consistent, handle arbitrary supported batches, and report
overflow before any contribution is lost. Dynamic exact counts also need an explicit
treatment of host reads and synchronization rather than an assumption that moving
`unique` earlier removes the synchronization.

### Historical trace evidence and its limits

The timing reference is `bf16-cache-trace-001`, optimizer update 21, rank 0, three
microbatches, with source revision `657e77a4f8ddb116b75925e5671b00642a0e4d32`. The
current inspected task source revision is `6c6cd9c0e41a0a5d3d00e1e252b3c120052d3507`; do
not claim the historical trace directly executes every current source byte. The raw
resolved config is not present locally. Its retained receipt records SHA-256
`a71150cc5997eafd3d6d6cedc6bfab13c5c1472c7a06e0befdc1051f4cc5777e`.

The rank-0 Perfetto embedding backward gradient reduce-scatter is slice 43466, starting
at 853.516399 ms and lasting 5.182387 ms in the trace-relative clock. The layer-0
gradient reduce-scatter is a distinct slice (43435); do not attribute that layer event
to the embedding. The three recorded forward embedding all-gathers last 2.649900,
2.825164, and 2.809420 ms. The three backward embedding all-gathers last 4.214321,
3.941328, and 4.047825 ms. The dense FP32 gradient reduce-scatters last 5.198802,
5.182387, and 5.217619 ms.

These are collective intervals, not exposed loss, additive step time, or measured
recoverable speedup. The payload sizes are not wire-byte counts, and the trace is rank 0
only. The trace and local source/trace contract are linked in the [local embedding
contract](../runs/bf16-expert-cache/embedding-local-contract.json) and [native tail
record](../runs/bf16-expert-cache/tail-optimization-native.json). Treat these timings as
motivation to test mechanism removal, not a performance estimate.

There is a smaller, distinct existing control:
`model.tok_embeddings.set_unshard_in_backward(False)` after wrapping skips the GFSDP
backward weight unshard for the plain embedding. Native embedding backward needs IDs,
incoming gradients, and saved shape, not weight values. The forward weight all-gather
and dense gradient reduce-scatter remain. The BF16 shared-buffer allocation may be
reused after forward; reshard releases ownership but does not promise that storage was
freed or that its contents remain intact. This narrow setting is source-level feasible
but untested with the required compiled/SAC path. It is not a substitute for the
selected row-owner design and must be evaluated separately if pursued.

### Reusable implementation research and alternatives

The strongest donor inspected is NVIDIA NeMo Automodel's generic
`Qwen3_8_FlashNextOwnerShardedEmbedding`, pinned at
[`3ea1de8d0aa902c3329b585d194c2e5274a85f3f`](https://github.com/NVIDIA-NeMo/Automodel/blob/3ea1de8d0aa902c3329b585d194c2e5274a85f3f/nemo_automodel/components/models/qwen3_8_flash_next/engram.py).
Its generic owner table accepts ordinary global IDs; it does not require n-gram hashing.
It sorts and routes IDs to owners, runs owner-local `F.embedding`, returns values
through a reverse all-to-all and inverse permutation, then reverses the value exchange
in autograd. Its backward routes occurrence gradients and lets the owner-local embedding
backward coalesce duplicates. It sends every occurrence and has no requester-side
deduplication.

Reuse its owner-routing and inverse-route concepts, not its numerical or optimizer path
unchanged. BF16 owner parameters demote the baseline FP32 persistent storage and
accumulation. FP32 owner `F.embedding` changes compute and local duplicate-coalescing
boundaries; casting only the returned values to BF16 does not demonstrate backward
parity. Owner sharding also bypasses GFSDP's accumulation and reduction. Adaptation is
required for this workload.

NeMo's exact-count route uses count all-gather and host count reads, while its capacity
mode fixes padding after first-batch calibration and asserts on later overflow. Its
[pinned official Qwen
guide](https://github.com/NVIDIA-NeMo/Automodel/blob/3ea1de8d0aa902c3329b585d194c2e5274a85f3f/docs/model-coverage/llm/qwen/qwen3-8-flash-next.mdx)
documents an eight-node/64-H100, 100-step owner-sharding run, but the owner block is
eager and outside activation checkpoint recomputation and MTP is unsupported there. A
separate pinned DeepSeek guide reports 100-update H100 validation under its own
Torch/NCCL, owner-count, BF16-gradient/moment, accumulation, network-plugin, and PXN
conditions. This is real external owner-path evidence, not validation of our `[128256,
4096]`, DP8, 32-microbatch `acc32`, GFSDP, EMA, MTP, compile, or SAC contract. The
[alternatives research
record](../runs/bf16-expert-cache/embedding-alternatives-research.json) contains the
pinned paths and exact conditions.

Other inspected candidates do not provide a drop-in implementation:

| Candidate | Reusable part | Gap for this design |
| --- | --- | --- |
| [TorchRec row-wise sequence sharding](https://github.com/meta-pytorch/torchrec/blob/16c16e982bde3eb48a8d8505305c70019cb362ec/torchrec/distributed/sharding/rw_sequence_sharding.py#L121) and [pinned FBGEMM backward](https://github.com/pytorch/FBGEMM/blob/81de8c84ee278ca16a65d3f8ae0618573a0280d8/fbgemm_gpu/codegen/training/backward/embedding_backward_split_template.cu#L811) | Owner routing and inverse autograd all-to-all | Inspected DENSE backward is limited to embedding dimensions up to 2,048; our width is 4,096. The inspected weight dispatcher lacks BF16 weight support, and dense AdamW/EMA/GFSDP integration remains. |
| [NVIDIA cuEmbed](https://github.com/NVIDIA/cuEmbed/tree/792db47311f4556bda205da2e9c7a00df5bbcf5b) | Unique-row compressed backward and inverse mapping concepts | Inspected support is FP32/FP16 and single-GPU; BF16 and multi-GPU are future work. H100 examples are forward embedding-bag workloads at widths 32–128, not this D4096 distributed backward. |
| [NVIDIA DynamicEmb](https://github.com/NVIDIA/recsys-examples/tree/85e628db7a6ac4d5f5f070d1e2d15d51eead0534/corelib/dynamicemb) | Maintained BF16 row routing and H100/NVSwitch examples | Backward/update separation is on its roadmap; non-rowwise optimizer states follow embedding dtype in the inspected API. D4096 and the external FP32 dense AdamW/32-microbatch contract were not established. |
| [DeepSpeed sparse embedding path](https://github.com/deepspeedai/DeepSpeed/blob/892ed83e417df01d9d4f21f559a0b88183cf5e4c/deepspeed/runtime/engine.py) | Sparse gradient all-gather support | This is not a GFSDP owner-directed reduce-scatter replacement; ZeRO routing and optimizer handling differ, and a dense AdamW seam remains. |
| [PyTorch sparse NCCLX path](https://github.com/pytorch/pytorch/blob/e63b330cbcbcc136d3aab1895a183579dbb80dde/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp) | Sparse tensor API surface | The inspected NCCLX operation unpacks into dense output and runs dense all-reduce; it is not sparse-wire transport. |
| [DeepEP Engram buffer](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/buffers/engram.py) | Remote BF16/FP8 fetch/write and wait concepts | The inspected interface has no backward gradient push/reduce API; its current feature also requires a newer CUDA baseline than the recorded run. |

The inspected NCCLX [sparse block all-reduce
implementation](https://github.com/meta-pytorch/torchcomms/blob/8ecc3488fb132278b406f2abb755f1e8ba8a16e8/comms/ncclx/meta/collectives/AllReduceSparseBlock.cc#L91)
clears a dense receive tensor, unpacks sparse blocks, then performs dense all-reduce.
This is not sparse-wire communication.

The [kernel and transport research
record](../runs/bf16-expert-cache/embedding-kernel-research.json) and [alternative
survey](../runs/bf16-expert-cache/embedding-alternatives-research.json) preserve the
pinned source revisions and limitations. The choice is therefore to adapt the NeMo
owner-routing pattern around our local precision, optimizer, SAC, compile, EMA, and
checkpoint contracts. No donor is a drop-in, and no performance advantage has been
established.

### Finite implementation and acceptance gates

If source work is separately authorized, implement and review against these
affected-path gates before any production claim:

1. **Lookup and routing.** Compare owner-routed outputs with baseline BF16
   embedding outputs for valid int64 IDs, repeated IDs within one requester,
   IDs repeated across ranks, frequent-token skew, empty requesters, empty
   owners, and all-zero/ignored-loss cases. Check original token order, row
   owner, local row, inverse mapping, range validation, and empty sentinels.
2. **Precision and gradients.** Compare the source-local duplicate-coalesced
   BF16 rows at the native output boundary, owner-side FP32 SUM with factor 1,
   and dense FP32 shard gradients. Include combined trunk plus MTP gradients,
   per-microbatch accumulation over 32 microbatches, and the configured
   `acc32` behavior. Record observed numerical differences and their cause;
   do not impose a universal tolerance without the affected-path evidence.
3. **Dense update semantics.** Verify materialized zero gradients and dense
   fused AdamW behavior for untouched rows, including existing momentum and
   weight decay. Check global norm clipping when owner shards coexist with
   GFSDP ragged shards. Confirm one optimizer update after the same 32
   microbatches.
4. **EMA and persistence.** Exercise EMA initialization, every update, swap,
   swap-back, and cached-row invalidation. Save and reload main weights,
   Adam state, and EMA state, then compare the next update with uninterrupted
   execution. Preserve FQN, global shape, and owner group metadata.
5. **Compilation, SAC, and ordering.** Run the enabled compiled model/loss
   path with the exact current per-layer SAC selection, MTP1, FP8, and `acc32`.
   Confirm backward/recompute uses the original immutable route once, all
   ranks issue collectives in matching order for empty and nonempty traffic,
   and buffers are not reused before the last reader completes. Do not disable
   compilation or SAC to make this gate pass.
6. **Performance mechanism and cost.** Verify by trace that both weight
   all-gathers and dense gradient reduce-scatter are absent for the target
   embedding, that owner payloads match useful/padded row counts, and that no
   new host synchronization or staging allocation dominates the path. Measure
   peak memory and unprofiled throughput, including sorting, deduplication,
   padding, casts, pack/unpack, collective startup, and lookup work.

Use the affected embedding block and deterministic repository settings for numerical
comparisons. Compare with an identical control pair to measure A/A noise. For
performance, use the agreed production baseline and candidate, paired within the same
job, with 10 warmup and 10 measured steps after any startup or compilation. Record
software, configuration, seeds, topology, owner group, job identity, measured step
series, peak memory, and useful and padded traffic. Use traces to establish the
mechanism and unprofiled runs to measure magnitude. Neither the 5.2 ms historical
collective interval nor the 64 MiB compact payload is an expected step-time saving.

CPU/Gloo checks can cover route semantics and selected state behavior but do not close
H100/NCCL collective ordering, kernel, compilation, memory-lifetime, or performance
gates. Those checks require separate authorization and the actual target hardware. This
proposal itself authorizes no GPU job.

### Evidence and open status

The local evidence is read-only and dated 14 November 2026:

- [Embedding source, precision, trace, and integration contract](../runs/bf16-expert-cache/embedding-local-contract.json)
- [Pinned owner-sharding alternatives survey](../runs/bf16-expert-cache/embedding-alternatives-research.json)
- [Kernel and transport survey](../runs/bf16-expert-cache/embedding-kernel-research.json)
- [Native update-21 tail record](../runs/bf16-expert-cache/tail-optimization-native.json)

No implementation, test execution, GPU run, or benchmark is reported here. This document
records a proposal only; implementation and GPU validation require their own
authorization. The unresolved gates are actual runtime configuration and mesh order,
numerical equivalence of the source-local BF16 coalescing seam, dynamic count/padding
cost, integrated compiled/SAC behavior, dense optimizer/clipping compatibility, EMA and
DCP continuation, and H100/NCCL performance.
