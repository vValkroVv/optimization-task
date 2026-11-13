# Fresh baseline loss regions

Baseline only; no candidate trace or acceleration verdict. The JSON manifest
records exact Perfetto IDs, CPU bounds, flow-linked device intervals, original
CUDA streams, and kernel names through each rank's `name_dictionary`.
Raw SQL and complete results are in
`artifacts/raw/baseline-loss-region-queries.json` (ignored artifact).

## Reusable boundaries

- Two ranks × three microbatches × two native sequence chunks are mapped.
  Every chunk folds main and MTP heads into one head forward GEMM, then runs
  two CE forwards and immediate inner backward. Each chunk has three `aten::mm`
  roots: head forward, then the two head backward GEMMs. Their device ID lists
  include associated support launches, not just GEMM kernels.
- CE forward root: `Torch-Compiled Region: 1/0`, enclosing `CompiledFunction`
  and a `## Call CompiledFxGraph ... ##` root. Each head launches
  `triton_red_fused__log_softmax__to_copy_prepare_softmax_online_view_0` and
  `triton_red_fused__log_softmax__to_copy_nll_loss_forward_view_1`.
- CE backward root: `CompiledFunctionBackward` enclosing its recorded graph
  root and
  `triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_arange_eq_expand_nll_loss_backward_scalar_tensor_view_0`.
  Graph hashes differ across ranks; use recorded roots and named launches.
- `ce.compiled_core_device_ids` contains six CE kernels per chunk. The wider
  `ce.device_ids` also includes scaling, weights, scalar losses, backward
  normalization and `SplitBackward0` concatenation. CE core sums range from
  2.754 to 3.861 ms/chunk; wider CE device unions range from 4.208 to 5.429 ms.
  These are baseline device costs, not elapsed training time or speedups.
- Whole chunk bounds begin at the head's `GFSDP::root_pre_forward` and end
  at the next head root, or `_DecoderOutputGradientBackProp` for chunk two.
  This retains FSDP gradient accumulation, callbacks, hidden-gradient fp32
  conversion/copy, and final output cast. The enclosing loss guard begins
  after `GFSDP::post_forward (mtp.layers.0)` and includes folding/setup.

## Attribution limits and comparison rules

Head all-gather is prefetched during MTP forward. Its exact two linked kernels
are recorded separately; gather→bridge elapsed includes remaining MTP work.
The second inner backward prefetches MTP weights, so its interval contains
communication beyond the head. Root callbacks traverse all module states.
Retain these costs in enclosing guards and report their ownership separately.

The manifest separates CPU elapsed, CPU-root duration sums, device span,
device interval union, kernel-duration sum and CPU-start→device-tail elapsed.
CPU loss submission takes 6.406–9.844 ms, but its linked work completes
94.798–97.768 ms after that CPU start; asynchronous device backlog makes
CPU submission duration unsuitable as loss runtime. Do not sum overlapping
chunk spans or interpret a launch flow as a CUDA data dependency.

Source mapping: `loss.py:644` defines setup, chunk loop, immediate backward,
sync/reshard and final cast; `loss.py:1009` defines folded-head loss weights;
`GradAccumulator.add` converts to fp32 and copies chunk gradients. Stable CE
must match CE forward **and** backward plus normalization/casts/reductions,
with whole head/loss guard. Linear CE computes dinput/dweight in
`four_fusion::native_linear_ce` **forward**, including tiled GEMMs and fp32
dweight accumulation/cast: compare entire chunks and guard, not only autograd
backward. Exact producer→consumer waiting cannot be established solely from
launch flows; no downstream-wait causality is asserted here.

Validation: JSON parses; both ranks have three microbatches, two chunks each,
two CE forward and backward roots, six CE core device kernels and three head
mm roots per chunk. An initial check incorrectly expected three device events
per three mm roots; each has support launches, so that assertion failed and
was corrected to validate roots. No source, runner, GPU or test execution changed.
