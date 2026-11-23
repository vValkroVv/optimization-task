# Compact BF16 expert gradient staging

This task-only overlay targets the deferred-dW EP4 treatment. It replaces the
seven expert groups' shared 2,304 MiB FP32 staging slot with one 1,152 MiB BF16
slot and adds one disjoint 288 MiB receive tensor per rank. The physical
staging saving is 864 MiB per rank. The allocator peak and training throughput
require a matched training check.

The overlay validates the full eFSDP context before allocation. Only the seven
unpadded expert groups on the pinned two-rank eFSDP pairs qualify. Dense and
head groups continue to use their own FP32 layouts. The compiled, selective-AC
recipe, FP8 weight path, optimizer and persistent FP32 gradients use the
deferred-dW treatment's configuration.

The existing mutating direct dW op waits for the old slot users before zeroing
the BF16 projection. For nonempty K it then calls the stock DeepGEMM BF16 writer
with the same destination as its zero accumulator. With empty K it publishes
positive zeros. Every rank follows the same seven-group order in the last head
backward, including ranks with no local expert work. The intermediate head
backwards run zero expert exchanges. Missing, partial or additional gradients
fail the attempt. Run the same patch on every rank.

Two BF16 exchanges send peer-owned slot views to a disjoint receive tensor on
the existing communication stream. The owner widens exactly and rounds the
sum in FP32. Later microbatches add that rounded sum to the persistent FP32
gradient with a second FP32 rounding. Both exchanges and owner writes precede
the slot release. This matches the tested producer domain of native FP32
NCCL PREMUL_SUM(1.0), including empty routing and the separate accumulation
rounding. The two-H100 precursor compared the stock producer and the pinned
rounded FP32 writer as well as the transport arithmetic.
[Component receipts](../../artifacts/raw/bf16-owner-slot-01/evidence/summary.json)

## Stage and verify

From the unpatched task checkout, run the local verifier with a Torch build
that supports Gloo. It applies both patches in a temporary directory. It
exercises real two-rank Gloo exchanges, the real GFSDP allocator, leases and
post-backward reduction, and uses CPU doubles for CUDA streams and kernels.
It also checks fake dispatch for both BF16 slot mutations. Pinned NCCL, the
actual DeepGEMM writer and the compiled training path remain native gates.

```bash
python -B runs/efsdp-optimizations/experiments/owner-bf16-slot/verify_cpu.py
```

Use dedicated, idle treatment trees. Apply the deferred patch in both arms,
and this compact patch in the compact arm. The older packed owner-transport
overlay is a different treatment.

```bash
compact=runs/efsdp-optimizations/experiments/owner-bf16-slot/owner-bf16-slot.patch
git apply --check "$compact"
git apply "$compact"
shasum -a 256 "$compact"
```

Reverse only the compact overlay with `git apply -R --check "$compact"` and
`git apply -R "$compact"`. Save the patch hash and both staged GIFT source
hashes with the campaign.

## Native gate and comparison

Use the same pinned H100 image, stock DeepGEMM installation, direct dW build,
original-A reference, data and compiler-cache policy as the accepted deferred
campaign. The task's existing runner supplies all four expert flags. Run from
each arm's staged source root with its own run ID.

```bash
python -B runs/efsdp-optimizations/run.py run "$run_id" --ep 4 \
  --optimization bf16-cache --optimization fp8-cache \
  --optimization direct-dw --optimization tail-retention \
  --direct-dw-build "$build_root" --reference "$reference"
```

Require exactly one `GFSDP compact BF16 owner transport active` line in
each compact-arm rank log, covering global ranks 0 through 7. Each line must
report `slot_bytes=1207959552` and `receive_bytes=301989888`. The deferred-only
control must have zero compact activation lines. Inspect rank logs from the
native runner, which enables INFO for all eight ranks. The default
`expert-gfsdp-rank-*.json` receipts report direct-dW slot presence and the
other expert flags. Both arms can produce the same JSON fields.
[Native runner](../../README.md)

In the first native compiled trace, check seven
`GFSDP::bf16_slot_owner_reduce` calls per final head backward, ordered
`layers.7[efsdp]` through `layers.1[efsdp]`, on both ranks of the profiled
eFSDP pair. Check for zero owner calls during intermediate head backwards
and zero native expert reduce-scatters for those seven groups.

The EP4 deterministic comparison is pending. The current
`--numerical-ep8` diagnostic is specific to EP8/eFSDP1. Before accepting
training timing, require the EP4 diagnostic to check retained FP32
gradients, first and later microbatches, and optimizer parity. Compare
matched stable windows and report allocated and reserved CUDA peaks. The
component timings exclude training overlap and compiler costs.
[EP8 diagnostic](../../numerics/README.md)
