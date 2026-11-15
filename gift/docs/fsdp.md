# FSDP1 -> FSDP2

## Why FSDP2?
PyTorch's fully sharded data parallelism (FSDP) API, [`FullyShardedDataParallel`](https://pytorch.org/docs/stable/fsdp.html), looks to offer a performant eager-mode implementation, including communication bucketing and communication/computation overlap. It defines a `FlatParameter` by flattening and concatenating a group of parameters to represent a communication bucket. However, this `FlatParameter` complicates applying different behaviors to individual parameters within the `FlatParameter`, e.g. parameter freezing, parameter casting, etc., hurting composability, and it complicates the internal implementation, e.g. making state dict logic thousands of lines and requiring additional communications.

With these limitations in mind, we designed and implemented an FSDP rewrite removing the `FlatParameter`.  We refer to this rewrite as FSDP2 and the original as FSDP1. FSDP2 targets the same use cases as FSDP1 plus more, and FSDP2 still strives for good performance in eager mode, using several of the same techniques.

Compared to FSDP1:
- FSDP2 represents sharded parameters as `DTensor`s sharded on dim-0, allowing for easy manipulation of individual parameters, communication-free sharded state dicts, and a simpler meta-device initialization flow.
- FSDP2 implements an improved memory management system that achieves lower and deterministic GPU memory by avoiding `recordStream` and does so without any CPU synchronization.

In the future, FSDP2 will offer an extension point to customize the all-gather (e.g. for fp8 all-gather for fp8 linears) and improved `torch.compile` support.

We have validated FSDP2 numerics and performance using torchtitan (e.g. see this [PR](https://github.com/pytorch/torchtitan/pull/165)). For example, on some Llama-7B runs on 8x H100s, FSDP2 achieves higher MFU with 7% lower peak memory than FSDP1, matching the same loss curve.

For more details on motivation, API, and system design, refer to [here](https://github.com/pytorch/pytorch/issues/114299). In this README, we try to provide more user-facing info and less system design details.

## FSDP1 <> FSDP2 API Differences
We go over some API differences between FSDP1 and FSDP2. Overall, we hope to minimize the API surface (including the number of arguments) to avoid having a monolithic API.
```python
@contract(state_cls=FSDPState)
def fully_shard(
  module: nn.Module,
  *,
  mesh: Optional[DeviceMesh] = None,
  reshard_after_forward: Union[bool, int] = True,
  mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
  offload_policy: OffloadPolicy = OffloadPolicy(),
) -> nn.Module:  # returns `module` for `contract` checks
```

| FSDP1 | FSDP2 |
| ----- | ----- |
| `module` | `module` |
| `process_group`/`device_mesh` | `mesh` |
| `sharding_strategy` | `reshard_after_forward` |
| `cpu_offload` | `offload_policy` |
| `auto_wrap_policy` | removed |
| `backward_prefetch` | removed |
| `mixed_precision` | `mp_policy` |
| `param_init_fn` | removed |
| `device_id` | removed |
| `sync_module_states` | removed |
| `forward_prefetch` | not yet implemented |
| `limit_all_gathers` | removed |
| `use_orig_params` | removed |
| `no_sync` | `set_requires_gradient_sync` |
| `ignored_modules`, `ignored_states` | `ignored_params` |

- `fully_shard(module)` is similar to `FullyShardedDataParallel(module)`, constructing one communication bucket from `module.parameters()` except those already assigned to a nested `fully_shard`/`FullyShardedDataParallel` call.
    - `fully_shard(module)` adds an `FSDPState` object on `module`, accessible via `fully_shard.state(module)`, instead of being an `nn.Module` wrapper. This is done via the `@contract` decorator.
    - Calling `model.named_parameters()` for a `model` with FSDP2 applied returns unchanged parameter names and `DTensor` sharded parameters. This means that the optimizer and gradient norm clipping see `DTensor`s.
    - `fully_shard(module)` performs a dynamic class swap on `module`. E.g., if `type(module) is Transformer`, then FSDP2 constructs a new class `FSDPTransformer` that inherits from a class `FSDPModule` and `Transformer` and sets `module.__class__` to be `FSDPTransformer`. This allows us to add new methods and override methods via `FSDPModule` without constructing an `nn.Module` wrapper.
- FSDP1's `sharding_strategy` and `process_group`/`device_mesh` maps to FSDP2's `mesh` and `reshard_after_forward`.
  - `mesh` should be 1D for FSDP and 2D for HSDP. For HSDP, we assume replication on the 0th mesh dim and sharding on the 1st mesh dim. If `mesh is None`, then FSDP2 initializes a 1D global mesh over the default process group.
  - `reshard_after_forward=True` or `False` determines whether parameters are resharded (freed) after forward. If `True`, then they are re-all-gathered in backward. This trades off saving memory at the cost of extra communication.
  - (Experimental) `reshard_after_forward: int` means that parameters are resharded to a smaller world size after forward (e.g. `reshard_after_forward=8` can mean intra-node) so that the backward all-gather is over a smaller world size.
  - | FSDP1 | FSDP2 | DeepSpeed |
    | --- | --- | --- |
    | 1 `process_group` + `FULL_SHARD` | 1D `mesh` + `reshard_after_forward=True` | ZeRO-3 |
    | 1 `process_group` + `SHARD_GRAD_OP` | 1D `mesh` + `reshard_after_forward=False` | ZeRO-2 |
    | 2 `process_group`s/2D `device_mesh` + `HYBRID_SHARD` | 2D `mesh` + `reshard_after_forward=True` | MiCS |
    | 2 `process_group`s/2D `device_mesh` + `_HYBRID_SHARD_ZERO2` | 2D `mesh` + `reshard_after_forward=False` | - |
    | - | 1D/2D `mesh` + `reshard_after_forward=8` (`int`) | ZeRO++ hpZ |
- FSDP2 maps `mixed_precision` to `mp_policy` and `cpu_offload` to `offload_policy`.
  - For `mp_policy`, we remove `buffer_dtype`, simplify `cast_forward_inputs` and `cast_root_forward_inputs` into just `cast_forward_inputs`, and add an `output_dtype`.
  - For `offload_policy`, we add a `pin_memory` option to avoid pinning CPU memory. (This feature may not have landed yet.)
- FSDP2 removes `auto_wrap_policy`, `backward_prefetch`, `param_init_fn`, `device_id`, `sync_module_states`, `limit_all_gathers`, and `use_orig_params`.
  - `auto_wrap_policy` provides a syntactic sugar for calling `FullyShardedDataParallel` on modules based on a predicate given by the policy and assigning the wrapped module to its parent. FSDP2 is no longer an `nn.Module` wrapper, so there is need to assign the module back to its parent. We prefer for this functionality to exist above `fully_shard`, and we may provide a utility like `auto_wrap_policy` in the future.
  - FSDP2 always follows `backward_prefetch=BACKWARD_PRE` without option since that is the only way to overlap collectives in backward correctly. `BACKWARD_POST` can prefetch [incorrectly](https://github.com/pytorch/pytorch/issues/108190) in nested-module cases.
  - FSDP2 supports a new meta-device initialization flow that does not require materializing a module on GPU *before* sharding it, removing the need for `param_init_fn`. See [Meta-Device Initialization](#meta-device-initialization) for more details.
  - FSDP2 always moves managed parameters/buffers to the `mesh`'s corresponding device, removing the need for `device_id`. For example, if `mesh.device_type` is `"cuda"`, then FSDP2 uses the current CUDA device.
  - FSDP2 uses a new memory management system that preserves communication/computation overlap while achieving deterministic and lower memory usage than FSDP1. This system does not require any CPU synchronization, so there is no need for `limit_all_gathers`.
  - FSDP2 always "uses the original parameters" since there is no more `FlatParameter`, removing the need for `use_orig_params`.
- How to implement `forward_prefetch` in FSDP2 is under discussion.

| FSDP1 | FSDP2 |
| ----- | ----- |
| `model.state_dict()`: full state dict | `model.state_dict()`: sharded state dict (no communication) |
| `optim.state_dict()`: local state dict | `optim.state_dict()`: sharded state dict (no communication) |
| `summon_full_params()` | use `DTensor` APIs like `full_tensor()` |
| `FSDP.clip_grad_norm_()` | `nn.utils.clip_grad_norm_()` |
| `ShardedGradScaler` | `amp.grad_scaler.GradScaler` |


## Meta-Device Initialization
Before with FSDP1:
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
with torch.device("meta"):
    model = Transformer()
policy = ModuleWrapPolicy({TransformerBlock})
# Call `reset_parameters()` on every module
model = FSDP(model, auto_wrap_policy=policy)
# Call `param_init_fn` on every module
def param_init_fn(module: nn.Module) -> None: ...
model = FSDP(model, auto_wrap_policy=policy, param_init_fn=param_init_fn)
```
After with FSDP2:
```python
with torch.device("meta"):
    model = Transformer()
for module in model.modules():
    if isinstance(module, TransformerBlock):
        fully_shard(module)
fully_shard(model)
for tensor in itertools.chain(model.parameters(), model.buffers()):
    assert tensor.device == torch.device("meta")
# Allocate buffers and sharded parameters on GPU
model.to_empty(device="cuda")
# Run user-defined initializers
model.init_weights() # or `model.apply(init_weights)`
```
FSDP1 requires either `reset_parameters` or `param_init_fn` to materialize a module onto GPU immediately before sharding. To do this correctly without re-initializing any tensors requires care and can be unwieldy. However, FSDP2 allows materializing tensors onto GPU _after_ sharding (taking advantage of `DTensor` and a new `swap_tensors` path for `nn.Module._apply` methods).

## GIFT GFSDP opt-in controls

The three independent controls below default to `false` and are only applied by
the GIFT MoE GFSDP2 path. The FFN controls require TP/PP/CP1. Row-owner
embeddings also require no HSDP replicas, no CPU offload, untied weights, and a
supported complete-row input table and lookup.

| Parallelism option | Default | Behavior |
| --- | --- | --- |
| `gfsdp_split_layer0_ffn` | `false` | Give `layers.0.feed_forward` a separate compiled GFSDP boundary. |
| `gfsdp_split_mtp_ffn` | `false` | Give `mtp.layers.0.block.feed_forward` a separate compiled GFSDP boundary. |
| `gfsdp_row_owner_embeddings` | `false` | Exchange requested input-embedding rows with dense GFSDP owners. |

For example, add the selected options to an existing experiment override:

```yaml
parallelism:
  fsdp_backend: gfsdp2
  gfsdp_split_layer0_ffn: true
  gfsdp_split_mtp_ffn: true
  gfsdp_row_owner_embeddings: true
```

Each FFN keeps `w1`, `w2` and `w3` together in its original module. Its eager
parent calls separately compiled prefix, FFN and suffix regions. Activation
checkpointing runs inside those regions with the original block's resolved
policy and options. The ordinal `mm_half` SAC policy is incompatible with these
boundaries and raises an explicit error. Model and loss compilation remain
independent existing controls.

The FFN remainder and its FFN child form separate compiled/GFSDP regions. The
selected groups keep `w1`, `w2` and `w3` together; setup logs name both compute
boundaries and actual GFSDP groups. The ordinal `mm_half` SAC policy is
incompatible with these boundaries and raises an explicit error. Model and loss
compilation remain enabled through their existing controls. FFN overlap is a
schedule opportunity, not a measured speedup. Each selected parent and FFN use disjoint contiguous views per dtype in the
original alternating all-gather slot and ordinary gradient slot. Allocation
sizing uses the original unsplit unit; padding that exceeds that budget raises
before compute. Reuse waits for every overlapping region's release event, so
early FFN reduction can proceed alongside writes to the disjoint remainder.
Allocator peaks and actual overlap require component measurements.

With `gfsdp_row_owner_embeddings`, the registered FP32 parameter keeps the
`tok_embeddings.weight` FQN and global table shape. On the accepted crop's
eight dense owners, each shard contains 16,032 complete rows. Forward routes
unique signed int64 token IDs to owners and restores their original order; the
trunk and MTP reuse this one eager lookup. Backward uses native dense BF16
embedding coalescing before lossless BF16 row transport, then widens each source
contribution before FP32 owner summation. The owner returns that sum as an owned
local FP32 gradient wrapped in the parameter's RaggedShard spec, so ordinary
AccumulateGrad owns `.grad` accumulation. `torch.autograd.grad`, parameter
gradient hooks, and post-accumulate hooks retain normal PyTorch behavior;
higher-order differentiation remains unsupported. Dense FP32 accumulation,
division factor one, global clipping, fused AdamW, EMA, and flat DCP restore
remain in use. No full-table embedding gather or dense embedding reduce-scatter
is issued, and no persistent BF16 row cache is kept.

The current native backward path still materializes a full-vocabulary BF16
gradient and native scratch: 1,002 MiB for the BF16 output at vocabulary
128,256 by hidden size 4,096; the low-parallelism branch can additionally use a
2,004 MiB FP32 temporary. The returned owner gradient is another 250.5 MiB per
rank at this shape. It replaces the former dense-gradient suffix in borrowed
scratch, reducing the maximum backward borrow from 954.5 to 704 MiB, but the
owned output is allocated while the native BF16 gradient is still live; this is
not evidence of a lower allocator peak. Per forward, dynamic unique counts use
one count all-gather and explicit host read before variable-size request/value
exchanges; later batches are not bounded by a first-batch capacity. Skewed
traffic and any additional synchronization from dynamic `torch.unique` remain
runtime costs to measure. Checkpoint/optimizer/EMA ownership is integrated
through the existing GFSDP state adapter.

The embedding remains eager before compiled/SAC blocks, while model and loss
compilation and the selected SAC policy stay enabled. Unsupported modes fail
explicitly: TP/PP/CP greater than one, HSDP replicas, CPU offload, tied input
weights, padding/max-norm/frequency-scaled/sparse lookup, incomplete or padded
owner rows, deferred/no-sync reduction, and higher-order differentiation. These
controls preserve logical parameter names, precision, gradient division, and
optimizer/EMA traversal. Flat shard offsets can change; use the GFSDP DCP
adapter when restoring model, optimizer, and EMA state.

Performance remains pending. The native scratch, dynamic host metadata,
owner imbalance, and communication-stream costs require the authorized H100/
NCCL component measurements; no speed claim follows from the CPU/Gloo checks.

Focused tests are `tests/unit_tests/test_ffn_overlap.py`,
`tests/unit_tests/test_gfsdp_ffn_overlap.py`,
`tests/unit_tests/test_row_owner_embedding.py`, and
`tests/unit_tests/test_gfsdp_row_owner_integration.py`. The FFN GFSDP test
exercises 32 microbatches, fused AdamW, EMA, and baseline-to-split DCP
continuation; row-owner tests cover routing/state and actual wrapper exclusion.
CPU/Gloo checks do not validate H100 FP8 kernels or the production eight-rank
schedule.
