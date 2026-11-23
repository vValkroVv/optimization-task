# EP4 FP8 all-gather coalescing

Run this component benchmark on one node with two H100s from the target
eFSDP2 fabric. Each rank sends 301,989,888 uint8 payload bytes and 18,432
FP32 scales and receives two rank-ordered shards into separate typed planes.
Both routes reuse the same send and receive allocations. A submits two
`all_gather_into_tensor` calls. B wraps those same calls in
`_coalescing_manager`, which groups two typed NCCL operations without a
packing buffer. Both use the current Expert GFSDP synchronous submission mode
on a dedicated CUDA stream. See the pinned
[c10d manager](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/torch/distributed/distributed_c10d.py#L3353-L3448)
and [NCCL implementation](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L4684-L4724).

From the task checkout in the authorized training image, set `DEVICE_PAIR`
to the two allocated device selectors and `ATTEMPT` to a new directory on
task-owned storage. Check the selected GPUs are idle and share the intended
NVLink route before starting.

```bash
source /etc/profile.d/wm.sh
mkdir "$ATTEMPT"
nvidia-smi topo -m > "$ATTEMPT/topology.txt"
timeout --signal=TERM --kill-after=20s 180s \
  env CUDA_VISIBLE_DEVICES="$DEVICE_PAIR" OMP_NUM_THREADS=1 \
  /opt/torchtitan-venv/bin/python3.12 -B -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0 \
  runs/efsdp-optimizations/experiments/fp8_gather_coalescing.py \
  --output-dir "$ATTEMPT" > "$ATTEMPT/launcher.log" 2>&1
```

The script requires the deployed Torch source revision and exactly two H100
ranks. The NCCL group timeout defaults to 45 seconds. A failed dtype or
rank-order check stops the experiment before timing. A nonzero launcher exit
or missing `summary.json` invalidates the attempt. Keep its log and any
partial rank receipts, and use a new directory for a retry.

Each rank checks every received byte and FP32 scale bit against independently
generated shard values for both routes before timing. The default is three
ABBA cycles with five warmups and twenty measurements per block.
`host_submit_us` measures the wall time of the native calls and, for B, the
coalescing context. `device_span_us` is CUDA event elapsed time on the caller
stream, with the end event synchronized after each pair. It includes any
stream idle time between recorded events. The barrier between blocks and the
correctness reads are outside both timing windows.

`rank-0.json` and `rank-1.json` preserve sample order and runtime identity.
`summary.json` reports the median of the samplewise rank maxima for every
block and the median of those block results for each route. Compare the A
blocks with each other to assess drift before interpreting A/B differences.

Inputs model already-prepared weight planes. Quantization is outside this
experiment. The component result measures serialized all-gather costs on
the selected pair. Training overlap, shared pool capacity and first-refresh
cost require their respective production measurements.

## EP4 FP8 all-gather on copy engines

`fp8_gather_copy_engine.py` compares three routes for the same two typed
all-gathers on the same NVLink-connected H100 pair.

| Arm | Allocation | NCCL communicator |
| --- | --- | --- |
| A | Ordinary PyTorch buffers | Default CTA policy |
| B | NCCL symmetric, registered send and receive buffers | Same default communicator as A |
| C | NCCL symmetric, registered send and receive buffers | Zero CTA policy |

The script uses the deployed Torch revision and NCCL 2.30.7. Both collectives
use the dedicated caller stream with `async_op=False`. A separate stream is
required for the documented copy-engine path. Each rank compares both
received planes bitwise with reference shards in all three arms. The probe
requires rank 0 to log one `[Copy Engine]` AllGather for each of C's two
payload sizes and none for A and B. A nonzero exit or missing probe summary
prevents timing. The timing run matches the probe's GPU UUIDs, versions and
NCCL environment fingerprint. See the [pinned Torch setup](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/docs/source/symmetric_memory.md#L367-L430),
[NCCL requirements](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2307/user-guide/docs/usage/bufferreg.html#zero-cta-optimization),
and [NCCL selection log](https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/ce_coll.cc#L1280-L1304).

Use the same device preflight as above. Set `DEVICE_PAIR` and `ATTEMPT`,
with `ATTEMPT` a fresh task-owned directory. Keep `NCCL_CTA_POLICY` unset
and verbose NCCL logging off in the launch environment. The probe enables
its own bounded logs. Both scripts in this directory must be present.

```bash
source /etc/profile.d/wm.sh
mkdir "$ATTEMPT"
mkdir "$ATTEMPT/probe" "$ATTEMPT/timing"
nvidia-smi topo -m > "$ATTEMPT/topology.txt"
launcher=(timeout --signal=TERM --kill-after=20s 180s
  env CUDA_VISIBLE_DEVICES="$DEVICE_PAIR" OMP_NUM_THREADS=1
  /opt/torchtitan-venv/bin/python3.12 -B -m torch.distributed.run
  --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0
  runs/efsdp-optimizations/experiments/fp8_gather_copy_engine.py)
"${launcher[@]}" probe --output-dir "$ATTEMPT/probe" \
  > "$ATTEMPT/probe/launcher.log" 2>&1 &&
"${launcher[@]}" bench --activation "$ATTEMPT/probe" \
  --output-dir "$ATTEMPT/timing" > "$ATTEMPT/timing/launcher.log" 2>&1
```

The default timing order is two ABC CBA cycles, with three warmups and
twelve samples for each mode and arm. Rank receipts retain individual
samples. The summary takes the median of the two-rank sample maxima for
each duration and the median of the rank minima for event-span overlap.
It reports gather-only host and device timing, BF16 GEMM-only time, and
concurrent gather, GEMM, joint and overlap timings separately.

The GEMM is a contention proxy using one BF16 `w13`-shaped multiply with
2048 rows, input width 4096 and output width 3072. Those widths match the
crop and the row count models balanced EP4 routing. The timed pair enqueues
gathers and then the GEMM on a separate stream. Use the recorded overlap
before interpreting contention results. The component omits real FP8
grouped GEMM, quantization, other collectives, and shared training buffers.

Each arm has 864 MiB plus 216 KiB of logical send and receive planes per
rank. All three sets stay live during timing. Device-used and PyTorch
allocator snapshots record allocation stages and both allocator peaks over
timing. Device-used snapshots include symmetric and NCCL memory that the
PyTorch allocator may omit. The stage deltas include validation or cached
allocations and setup costs, so the snapshots do not isolate training
memory overhead.

## EP4 FP8 gathered scale cache

`fp8_gather_cached_scales.py` uses the same two-H100 eFSDP2 route and exact
301,989,888 uint8 bytes and 18,432 FP32 scales per rank. Two distinct expert
fixtures alternate on a shared payload slot. A performs both typed gathers
on every unshard. B keeps one separate rank-major scale output per fixture,
gathers both planes on the first unshard after an input generation changes,
and gathers only uint8 on a hit. Source scale planes are identical across arms.

Use the same idle NVLink pair and task-owned `DEVICE_PAIR` and `ATTEMPT`
settings as above. Keep `fp8_gather_coalescing.py` alongside the script and
use a fresh directory for each attempt.

```bash
source /etc/profile.d/wm.sh
mkdir "$ATTEMPT"
nvidia-smi topo -m > "$ATTEMPT/topology.txt"
timeout --signal=TERM --kill-after=20s 180s \
  env CUDA_VISIBLE_DEVICES="$DEVICE_PAIR" OMP_NUM_THREADS=1 \
  /opt/torchtitan-venv/bin/python3.12 -B -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0 \
  runs/efsdp-optimizations/experiments/fp8_gather_cached_scales.py \
  --output-dir "$ATTEMPT" > "$ATTEMPT/launcher.log" 2>&1
```

The preflight requires pinned Torch and NCCL 2.30.7. Each rank checks both
planes bitwise across changed inputs and fixture handoffs. Async correctness
covers consumed and discarded first gathers on a separate consumer stream.
The discard drains both `Work` handles before recording release, then a
different fixture takes the slot before the discarded fixture's hit. The
default timing run uses three ABBA cycles. Each block has three warmup and
twelve measured sweeps with `refresh 0, refresh 1, hit 0, hit 1` ordering.
Rank receipts preserve host submission and synchronized CUDA event spans
on a priority -1 caller stream. The summary reports refresh and hit timings
separately using samplewise rank maxima. A failed launcher or missing
summary invalidates the attempt. Compare A blocks for drift before reading
A/B differences. Fixture updates and correctness reads are outside the
timed windows. The measurement isolates transport of prepared weight planes.

Both arms stay allocated during timing. The two persistent outputs add
288 KiB per rank to the live fixture, and rank receipts include their Torch
allocator delta. In EP4 deployment, seven persistent 144 KiB outputs
replace two shared scale pools while local scale inputs stay allocated,
adding 720 KiB per rank before rounding. With `G` equal eligible groups,
`P` reclaimable shared scale pools, eFSDP size `R > 1`, and `L` local
FP32 scale bytes, the separate-plane increase is `(G - P) × R × L` per
rank. Adjust the subtraction for pools still held by other FP32 consumers.
eFSDP1 already uses local scales.
