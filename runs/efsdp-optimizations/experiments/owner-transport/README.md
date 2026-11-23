# EP4 two-chunk BF16 expert owner transport

This task-only overlay targets the combined Expert GFSDP fixes on the
original-A EP4 recipe. It replaces the NCCL FP32 reduce-scatter for the
seven routed expert groups with the two-chunk BF16 peer exchange. The
exchange overwrites only the dead half of each outgoing peer chunk after
packing into disjoint BF16 send storage. The owner adds received values in
FP32, using the same kernel shape and addition as the passing component.
One 301,989,888-byte BF16 scratch tensor is shared across all expert groups
on the existing priority -1 stream. The native first-microbatch output,
later in-place output, accumulation, and slot-release events are used
through the existing GFSDP reduction path.
[Peer transport component](../bf16-owner-transport.md)

The overlay itself selects the route. Use the unmodified combined source as
the matched control. The overlay rejects other recipes, validates the
eight-rank EP4 mesh and its exact eFSDP2 pairs, and binds the seven groups
only after checking the direct dW buffers and FP8 consumers. It requires
NCCL 2.30.7, the pinned Torch build, H100, native two-rank groups, no
replica or YCCL, BF16 parameter and FP32 reduction, the native
PREMUL_SUM(1.0) contract, and NCCL NaN checking off. It inherits the
original-A runner's resolved recipe and input validation.

The direct dW lease already requires both projection writes and rejects an
additional gradient. Selected groups with a missing, incomplete, partial,
or additional contribution fail. The runtime validates
`layers.7[efsdp]` through `layers.1[efsdp]` before each exchange, and the
shared final callback checks the full sequence. Intermediate head
backwards must have zero expert reductions. Stage identical source bytes
for all ranks. A failing rank stops the distributed attempt and the
ordinary process-group watchdog handles a waiting peer.

## Check and stage

Run the local check from the combined-fix source root with Torch and Gloo.
It applies the patch in a temporary tree. It uses host substitutes for
CUDA stream/event and Triton operations around real two-rank Gloo
collectives, exercises the overlay's production reduction function and
rejects invalid leases and order. GPU math and scheduling need the native
gate below.

```bash
python -B runs/efsdp-optimizations/experiments/owner-transport/verify_cpu.py
```

Apply only in the task's dedicated treatment checkout, using the accepted
combined GIFT source as the input. The patch adds one GIFT module and
changes four source paths. Keep the unmodified combined checkout for the
control. Save the patch hash and each checkout's GIFT source hash with the
attempt.

```bash
patch_path=runs/efsdp-optimizations/experiments/owner-transport/owner-transport.patch
git apply --check "$patch_path"
git apply "$patch_path"
```

To remove the staged overlay from an idle treatment checkout:

```bash
git apply -R --check "$patch_path"
git apply -R "$patch_path"
```

## Native two-H100 gate

Choose an authorized idle eFSDP2 device pair and set `DEVICE_PAIR`. Run the
check from the staged treatment checkout in the pinned image. Keep the
sibling `bf16_owner_transport.py` fixture beside the overlay directory.

```bash
source /etc/profile.d/wm.sh
PYTHONPATH="$PWD/gift" CUDA_VISIBLE_DEVICES="$DEVICE_PAIR" \
  timeout --signal=TERM --kill-after=15s 3m \
  /opt/torchtitan-venv/bin/python3.12 -B -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0 \
  runs/efsdp-optimizations/experiments/owner-transport/verify_transport.py
```

Require exit zero and one JSON result from each rank. The gate compares
every raw FP32 output bit after the first out-of-place reduction and after
14 reductions into the same persistent accumulator. It runs the exact
production reduction entry point with the same slot and completion event
between each reduction and the next producer overwrite. One rank
contributes zeros throughout the second sequence. The fixture supplies the
BF16-rounded producer domain. The compiled direct dW lease integration
still needs a native training smoke before comparing 10+10 from the two
checkouts. Require the activation marker and preserve the original
matching source, cache policy, data, topology, and run windows.
