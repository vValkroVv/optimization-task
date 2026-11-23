# BF16 peer transport for EP4 expert gradient reduction

This component uses one local eFSDP2 pair from the EP4 crop. Each rank
contributes a contiguous slot of 603,979,776 FP32 elements. The two equal
destination slices contain 301,989,888 elements each. The native arm invokes
FP32 NCCL reduce-scatter with `PREMUL_SUM(1.0)`, as selected by GFSDP for this
gradient division factor and no replica reduction. The candidate narrows the
peer-owned slice to BF16, sends and receives one contiguous BF16 chunk with
`all_to_all_single`, and uses one Triton kernel to widen the received lanes
and perform the owner add in FP32. It writes directly into the requested FP32
output view. The local slice stays FP32 throughout.

The owner exchange uses split sizes `[0, N]` and `[N, 0]` with the pinned
PyTorch NCCL backend. The zero entry is the local row. The pinned NCCL wrapper
selects its uneven all-to-all implementation based on build capabilities.
[PyTorch NCCL process group](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L5047-L5145) ·
[PyTorch NCCL wrapper](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/torch/csrc/cuda/nccl.cpp#L743-L799) ·
[NCCL reduction operator](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/ops.html#ncclredopcreatepremulsum)

## Correctness gate

The producer domain comes from the direct dW epilogue, which applies
round-to-nearest-even BF16 conversion and BF16 addition to a positive zero
before storing the widened result. The deterministic input pattern uses
different magnitudes by source rank to make some FP32 sums require more
precision than BF16. Each rank compares every regular FP32 output element
as integer bits, including first-microbatch out-of-place output and the later
in-place output into its rank-local slot slice. There is a populated case and
a case with an entirely zero second rank. Both ranks participate in both
cases. Each alias comparison restores the input outside timing.
[Direct dW epilogue](../direct-dw/deepgemm-rounded-f32.patch) ·
[GFSDP native reduce-scatter](../../../gift/torchtitan/distributed/gfsdp/_collectives.py)

Receipts list operand and output hex bits for signed zeros, cancellation,
subnormals, a non-BF16 FP32 sum of BF16 operands, overflow, infinities, and
NaNs. The two deliberately injected FP32-only `1 + 2**-8` probes must retain
the tie bit in native output. The candidate must retain it when it stays
local and lose it when transmitted as BF16. The deliberately injected raw
NaN-payload diagnostic requires NaN on both paths and reports the output
bits separately. The ordinary and remaining special-case bits must match
on both ranks before timing. The raw NaN diagnostic can show a difference
from input packing or reduction evaluation order and does not by itself
identify which mechanism occurred.

## Run

Use only an authorized idle H100 pair from the same node. Select the
physical pair matching the profiled eFSDP2 ranks and point `output_dir` at a
fresh existing task-owned artifact directory. Run from this task source root
in the pinned image with its workload profile initialized.

```bash
source /etc/profile.d/wm.sh
CUDA_VISIBLE_DEVICES=0,4 timeout --signal=TERM --kill-after=15s 10m \
  /opt/torchtitan-venv/bin/python3.12 -B -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0 \
  runs/efsdp-optimizations/experiments/bf16_owner_transport.py \
  --output-dir "$output_dir"
```

Replace `0,4` only if the verified job mapping uses different device
ordinals. The script checks the pinned Torch revision and both H100s.
NCCL's process-group timeout defaults to 60 seconds. Keep
`TORCH_NCCL_NAN_CHECK` disabled for the embedded NaN probes. Add
`--verify-only` to exercise both layouts without collecting latency.
Preserve combined launcher output and its exit status. A nonzero exit
invalidates timing. Keep the failed directory and use a fresh directory
for a permitted retry.

## Timing and memory

The default runs two ABBA cycles for each of 1 and 8 chunks, with five
warmups and twenty samples per block. A is native reduce-scatter and B is
the candidate's full pack, exchange, and owner add sequence. Both arms
reuse the same out-of-place FP32 tensors on the same stream. Allocation,
correctness readbacks, block barriers, and event completion synchronization
are outside the measured host interval. The device interval runs between
events on the caller stream and includes communication dependencies,
kernel work, and any device gaps between submits. The host interval
measures the synchronous c10d API call path and Python submission work.
The timings do not decompose kernel, fabric, or wait durations. The
component covers transport through the owner result, before any optimizer
or microbatch accumulation update.

The candidate has two BF16 staging tensors per rank. One chunk uses
1,207,959,552 bytes (1.125 GiB) and eight chunks use 150,994,944 bytes
(144 MiB). Live and peak Torch allocations are recorded against the same
FP32-buffer baseline. Reserved memory and whole-device usage snapshots
also appear in the rank receipts. Device-wide usage snapshots include
allocator and runtime overhead at the sampled points and do not establish
a device-wide peak.

`rank-0-verification.json` and `rank-1-verification.json` are written before
timing. Final rank receipts include runtime identity, ordered samples,
memory observations and full verification rows. `summary.json` preserves
the median of the samplewise rank maxima for each ABBA block and its
worst-rank incremental Torch allocation. Compare the two B blocks with the
bracketing A controls for each chunk choice. The result measures only this
component on the selected pair.

## Receive into the dead peer slice

`--alias-peer-receive --chunks 1 2` compares native reduce-scatter, the owner
route with disjoint staging, and the owner route with a receive view into
the just-packed FP32 peer chunk. The outgoing BF16 buffer is separate from
that view. Each chunk packs, exchanges, and adds on the same priority -1
stream. The receive overwrites the first half of the current peer chunk's
bytes. The owner input, untouched half, and next chunk stay outside the
receive range. In this mode the full producer slot is restored for all
arms before every warmup and timed sample, outside both timers. The route
order is `A B C C B A` per repeat. Omitting the flag retains the 1/8-chunk
ABBA invocation above.

In addition to the full owner and special-value checks, each rank verifies
the exact BF16 bytes written to the aliased ranges, preservation of the
unwritten peer ranges, full peer preservation in the disjoint arm, and
canaries around the slot, output, and staging. A same-group collective
immediately reads owner lanes. A separate check overwrites the producer
slot and reuses it for a second exchange, comparing the full accumulated
FP32 result against native reduction.

Use the same authorized two-H100 preflight with NCCL 2.30.7. Point
`output_dir` at a fresh existing task-owned directory and set `DEVICE_PAIR`
to the verified eFSDP2 pair.

```bash
source /etc/profile.d/wm.sh
CUDA_VISIBLE_DEVICES="$DEVICE_PAIR" timeout --signal=TERM --kill-after=15s 10m \
  /opt/torchtitan-venv/bin/python3.12 -B -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0 \
  runs/efsdp-optimizations/experiments/bf16_owner_transport.py \
  --alias-peer-receive --chunks 1 2 --output-dir "$output_dir" \
  > "$output_dir/launcher.log" 2>&1
```

| Owner chunks | Disjoint staging per rank | Aliased staging per rank |
| --- | ---: | ---: |
| 1 | 1,207,959,552 bytes (1,152 MiB) | 603,979,776 bytes (576 MiB) |
| 2 | 603,979,776 bytes (576 MiB) | 301,989,888 bytes (288 MiB) |

The aliased receive bytes are reported separately from allocated staging.
Rank receipts and the summary report observed incremental live and peak
Torch bytes above the common FP32 baseline. Use the block timings from this
campaign to compare all three routes under the matched restoration policy.
