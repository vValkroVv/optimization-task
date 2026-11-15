# BF16 expert-cache coverage when EFSdp is greater than one

This note records one implementation-coverage limit of the opt-in BF16 expert
weight cache. The [baseline](baseline.md) and [current state](status.md) remain
the owners of the selected recipe and experiment results.

## Concrete topology and limitation

The selected eight-GPU crop uses PP=TP=CP=1, EP8, dense shard-DP8 with
replication 1, and expert FSDP (EFSdp) 1. The current cache covers this
size-one expert-shard case. If the same eight ranks use EP4 while retaining
`dp_shard=8`, `dp_replicate=1`, and PP=TP=CP=1, EFSdp becomes 2:
ranks (0,4), (1,5), (2,6), and (3,7)
each shard one logical group of 32 experts across a pair of ranks, then gather
that group for computation. These are flat parameter-buffer shards; a shard
need not contain whole experts.

The general scaling limitation appears when EP is much smaller than the GPU
count and expert weights are sharded across multiple ranks. The precise condition
is EFSdp>1: the [mesh construction](../gift/torchtitan/distributed/parallel_dims.py#L164)
uses `efsdp = dp_shard * cp * tp // ep`. With PP=TP=CP=1 and no DP replication,
this reduces to `GPU_count / EP`. With other mesh axes, EP below the total GPU
count alone does not establish EFSdp>1.

In [`_bf16_cache_layout_is_eligible`](../gift/torchtitan/distributed/gfsdp/_param_group.py#L548),
the cache is rejected when the all-gather process group is larger than one and
when local and full unsharded parameter sizes differ. Those guards match the
implemented singleton direct-input path. On rejection, `finalize_backward`
clears ordinary all-gather input validity, so a later microbatch prepares the
local BF16 input from its FP32 master again. See
[`finalize_backward`](../gift/torchtitan/distributed/gfsdp/_param_group.py#L1067)
and the ordinary input preparation in
[`all_gather`](../gift/torchtitan/distributed/gfsdp/_collectives.py#L46).

**[WARNING] The current cache leaves repeated local BF16 preparation in the EFSdp>1 path.**
This is an optimization-coverage limitation; ordinary GFSDP remains active.
No EP4/EFSdp2 run or slowdown measurement is recorded here.

## Approximate extension

A narrow extension could reuse the existing per-group BF16 input for EFSdp>1
without bypassing the collective:

1. Split “local BF16 input may persist” from “the local input can also serve as
   the full output.” Keep the exact expert-parameter selection and existing dtype,
   ordinary-bucket, extension, YCCL, and device checks. For a multi-rank shard
   layout, validate the actual padded local-shard input and gathered-output shapes;
   do not retain full-equals-local size checks as a general eligibility condition.
   Reserve those equality/alias checks for the size-one direct-use path.
2. Keep one BF16 input per group and weight generation. Reuse the existing
   optimizer, load, EMA, and backing-replacement (including `_apply`) invalidation boundaries, along
   with their writer/reader ordering. The existing implementation and its lifecycle
   are described in the [implementation record](../runs/bf16-expert-cache/implementation.md).
3. For EFSdp>1, continue the normal `all_gather_into_tensor` into the shared full
   output. Wait for the collective before compute and retain its output-buffer
   ownership rules. Only EFSdp1 can keep the current direct input/output alias and
   skip the singleton collective. Do not promise the measured singleton
   input-to-output copies disappear at EFSdp2.
4. Track BF16-cast readiness and completion of actual collective input reads
   separately from the full output's compute lifetime. Preserve `Work.wait()`
   semantics for asynchronous collectives; an event recorded after launch alone is
   not proof that communication has finished.
5. Keep FP32 masters and gradients, BF16 collective inputs, existing FP8 compute
   conversion, optimizer math, checkpoint state, compilation, and activation
   checkpointing unchanged. This local-input extension needs no additional
   persistent full-weight cache.

This approach targets repeated local FP32-to-BF16 preparation. It does not by
itself remove all-gather communication, full-output storage, output-release
waits, or reduce-scatter dependencies. Caching the entire gathered BF16 weight
would be a different memory tradeoff: it requires persistent remote portions
per group and more invalidation/storage than the local-input extension.

## Scheduling dependencies and limits

The current communication context assigns all-gather and reduce-scatter to the
same stream in [`GFSDPCommContext`](../gift/torchtitan/distributed/gfsdp/_param_group.py#L74),
and reduce-scatter waits for current-stream work in
[`reduce_scatter`](../gift/torchtitan/distributed/gfsdp/_collectives.py#L105).
Unshard waits for a shared output-buffer release event, while reshard records
that release after the current consumer; see
[`unshard`](../gift/torchtitan/distributed/gfsdp/_param_group.py#L795) and
[`reshard`](../gift/torchtitan/distributed/gfsdp/_param_group.py#L875). With
normal resharding, [`_state.py`](../gift/torchtitan/distributed/gfsdp/_state.py#L236)
assigns two output buffers per shard process group in round-robin order; the
current dense and expert shard groups have separate pools. Root forward also
orders the all-gather stream after the optimizer event or preceding current
stream work in [`_root_pre_forward`](../gift/torchtitan/distributed/gfsdp/_state.py#L115),
normally once per microbatch. The exposed wait duration is not measured.

These are source-level dependencies, not measured delay attribution. Reusing a
local input does not remove them. Earlier gathering is only a possible later
scheduling change when the output buffer, stream dependencies, and peer ranks
are ready. A separate stream alone does not establish overlap when collectives
share a communicator; see PyTorch's
[`Work.wait()` documentation](https://docs.pytorch.org/docs/2.15/distributed.html)
and [FSDP all-gather process-group guidance](https://docs.pytorch.org/docs/2.15/distributed.fsdp.fully_shard.html#set_separate_all_gather_group).
No output-release or collective-ordering wait should be removed without a
specific ownership/completion proof and measurement.

## Evidence and validation status

The current evidence is bounded to the selected EFSdp1 crop. The
[BF16 cache report](../runs/bf16-expert-cache/report.md) and
[trace analysis](../runs/bf16-expert-cache/trace-analysis.md) record 21 to 7
FP32-to-BF16 casts and 42 to 0 local input-to-output copies per traced rank over
three microbatches, with the existing FP8 quantization path still active. This
confirms the singleton mechanism only; it does not predict EFSdp2 behavior or
prove a training-step speedup. The report's historical TPS difference is
provisional and is not EP4 evidence.

Before accepting an EFSdp>1 extension, validate at least:

- Requested versus actually enabled cache groups under EP4/EFSdp2, and one
  refresh per weight generation with reuse across microbatches.
- The normal multi-rank all-gather remains active and produces the expected
  gathered weights; check loss, block gradients, and updated FP32 masters against
  uncached EP4 under the same deterministic settings.
- Optimizer, load/DCP, EMA, and supported backing-replacement invalidation,
  including writer/read ordering across ranks.
- A representative EP4 trace for casts, all-gather calls/bytes, output release,
  reduce-scatter, and peer/stream waits; then paired stable unprofiled timing and
  memory measurements to estimate magnitude.

The EFSdp>1 extension, EP4 correctness, and EP4 runtime/performance remain
unimplemented or unverified. These checks are a validation outline, not results.
