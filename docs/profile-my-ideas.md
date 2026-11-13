# Profile research: my ideas

Research origin: the repository owner's own investigation of the baseline profile.
Recorded: 13 November 2026.
Status: baseline-profile findings; fusion and expert-weight reuse remain untested
hypotheses.

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

The expected reduction in repeated work is an untested hypothesis. Neither a
modified implementation nor a training speedup has been measured. The size-one
copy observation belongs to this baseline; it does not establish the behavior of
the production PP4/EP32 configuration.

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
