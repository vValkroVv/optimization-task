# Native inventory and compiler-cache preflight

## Acceptance checklist

- [x] Add one explicit, default-off GFSDP inventory switch. It is valid only for
  `--steps 1 --warmup 0 --window 1`; normal timed A/B commands do not enable it.
- [x] After the first native `Trainer.train_step()` returns, record one JSON file
  per rank with initialized GFSDP storage views, per-group padded dtype
  requirements, actual data/gradient contexts, and allocator current/peak stats.
- [x] Route FlashQLA TileLang through the already validated selected compiler
  cache root. Reuse existing TileLang directories only when the outer runner
  selected a completed cache, while retaining task containment, symlink, rank,
  and temporary-directory checks.
- [x] Verify command routing, post-step capture, storage deduplication, and
  reused-cache guards with CPU-only tests and scoped Ruff.
- [ ] Run the prepared diagnostics and training comparisons under the sole GPU
  lease after the component performance table is complete.

## Implementation and runtime cost

The full-model inventory uses the existing
[`buffer_inventory`](ffn_components.py#L367) helper. It reads initialized
`GFSDPModule` buffer metadata; it does not read tensor values, allocate CUDA
memory, synchronize, or run compute. Each rank writes
`measurement/gfsdp-buffer-inventory-rank-NN.json` after its first
`train_step()` returns. The record retains parameter FQNs, data/gradient kind,
dtype, backing pointer/size, view offset/length, and unique backing bytes.
Group records distinguish ordinary and private gradient ownership from data
contexts, list the actual buffer context type and backing views, and report
each padded dtype requirement. The byte requirements follow
`GFSDPParamGroup.padded_unsharded_param_numel` and the dtype selection used by
[`_init_unsharded_params`](../../gift/torchtitan/distributed/gfsdp/_param_group.py#L502).

The existing `WindowMeasurement` peak sampler now supports a zero-warmup
single-update diagnostic. It starts the measurement boundary immediately
before `Trainer.train()`, after model construction, and ends after update 1.
The allocator peak therefore includes first-update compilation and lazy GFSDP
initialization. The inventory file also records allocator current/peak values
immediately after the first `train_step()` return. This one-update duration is
a compilation and initialization diagnostic; it is not throughput evidence.

Instrumentation remains off by default. The one-update-only flag cannot be
combined with a timed A/B window, profile capture, or additional updates. Its
extra work is one metadata walk, one allocator-stat query, and one small JSON
write per rank. The selected TileLang cache path adds directory checks during
runner setup and rank bootstrap. Ordinary runs with no cache reuse retain fresh
per-attempt TileLang directories. Timed A/B runs keep the accepted monitoring,
profiler, validation, checkpoint, and numerical-capture settings disabled.

The selected compiler cache is
`/workspace-SR008.nfs2/vkropotin/optimization-task/runs/overnight-streaming-perf/artifacts/raw/combined-four-baseline-003/cache`.
The task runner's completed-cache validation already checked its producer
receipt, stopped state, runtime fingerprint, and canonical configuration hash.
Its TorchInductor, Triton, DeepGEMM, and HybridEP paths remain under the same
root. It had no TileLang directory before this change; the first authorized
run can create `tilelang/` there, and later runs can reuse it. A and B select
the same root. No cache probe or GPU command was run here.

No training-source file was changed for this preflight. The source-owner
receipt supplied for the active standard-autograd snapshot is
`f3a01c6cf99b981db7b4ef66866fe46259af84e9d2c21e2044e9ae18ff1696fd`; the
prepared plan's base revision is `2ada87125e433e1f76c4cb2bfad244b641a78a93`.

## Prepared one-update commands

These commands are prepared only. They preserve the accepted crop, seed,
external-gate backend, prefetch depth 32, BF16 expert-weight reuse, FP8/BF16/
FP32 precision, compile, SAC, MTP1, and existing parallelism. Both have
monitoring, profiler, validation, checkpoint saves, and memory snapshots off.

Baseline inventory and peak:

```bash
/opt/torchtitan-venv/bin/python3.12 /workspace-SR008.nfs2/vkropotin/optimization-task/runs/flash-qla/run.py \
  --gdn-backend external_gate \
  --run-id ffn-row-owner-memory-a-001 \
  --steps 1 --warmup 0 --window 1 \
  --num-prefetch-batches 32 --epoch-size 102400 \
  --gfsdp-reuse-bf16-expert-weights \
  --gfsdp-buffer-inventory \
  --output-root /workspace-SR008.nfs2/vkropotin/optimization-task/runs/ffn-row-owner/artifacts/raw \
  --compiler-cache-root /workspace-SR008.nfs2/vkropotin/optimization-task/runs/overnight-streaming-perf/artifacts/raw/combined-four-baseline-003/cache
```

Candidate inventory and peak, with the three controls enabled together:

```bash
/opt/torchtitan-venv/bin/python3.12 /workspace-SR008.nfs2/vkropotin/optimization-task/runs/flash-qla/run.py \
  --gdn-backend external_gate \
  --run-id ffn-row-owner-memory-b-001 \
  --steps 1 --warmup 0 --window 1 \
  --num-prefetch-batches 32 --epoch-size 102400 \
  --gfsdp-reuse-bf16-expert-weights \
  --gfsdp-split-mtp-ffn --gfsdp-split-layer0-ffn --gfsdp-row-owner-embeddings \
  --gfsdp-buffer-inventory \
  --output-root /workspace-SR008.nfs2/vkropotin/optimization-task/runs/ffn-row-owner/artifacts/raw \
  --compiler-cache-root /workspace-SR008.nfs2/vkropotin/optimization-task/runs/overnight-streaming-perf/artifacts/raw/combined-four-baseline-003/cache
```

The generated native config arguments for both runs are the current canonical
crop config plus these common overrides:

```text
--dataloader.num_prefetch_batches 32
--training.steps 1
--lr_scheduler.total_steps 65000 --lr_scheduler.warmup_steps 2000
--debug.seed 2048 --metrics.log_freq 1
--checkpoint.enable false --validator.enable false
--profiler.enable_profiling false --profiler.enable_memory_snapshot false
--parallelism.gfsdp_reuse_bf16_expert_weights true
--dataloader.dataset.epoch_size 102400
```

Candidate B additionally resolves all three new controls to `true`:
`gfsdp_split_mtp_ffn`, `gfsdp_split_layer0_ffn`, and
`gfsdp_row_owner_embeddings`. CPU command replay verified that the outer
FlashQLA runner forwards the inventory switch to the measurement helper and
that these native config overrides are unchanged apart from the requested
controls and one-update duration.

## Verification and remaining launch gate

The CPU-only command below passed: **77 passed, 3 warnings**. It ran with
`CUDA_VISIBLE_DEVICES=''`; the fake post-step metadata test also asserts CUDA
was not initialized. The warnings are the installed `pynvml` deprecation and
two Python fork warnings in existing streaming-lock tests.

```bash
CUDA_VISIBLE_DEVICES='' /opt/torchtitan-venv/bin/python3.12 -m pytest -q \
  --basetemp=runs/overnight-streaming-perf/artifacts/raw/pytest-native-preflight-final \
  runs/overnight-streaming-perf/tests/test_ffn_row_owner_runner.py \
  runs/overnight-streaming-perf/tests/test_measure.py \
  runs/flash-qla/tests/test_native_bridge.py
```

The first Ruff pass found one unused local and missing `Any` annotations in the
new CPU fixture. Those were fixed; the final scoped lint passed:

```bash
ruff check --select F821,F811,F841 --no-cache \
  runs/overnight-streaming-perf/run.py \
  runs/overnight-streaming-perf/measure.py \
  runs/overnight-streaming-perf/tests/test_ffn_row_owner_runner.py \
  runs/overnight-streaming-perf/tests/test_measure.py \
  runs/flash-qla/run.py runs/flash-qla/measure.py \
  runs/flash-qla/tests/test_native_bridge.py
```

Changed Python-file SHA-256 receipts after the edits:

| File | SHA-256 |
| --- | --- |
| `runs/overnight-streaming-perf/run.py` | `50d873d7394cb4907e1cb08a86582ec3e7ee0b278115fa50d03850661646bb2f` |
| `runs/overnight-streaming-perf/measure.py` | `8b1dcd56b822d73e0be045f4ab71734a50e72320d04321ec946433566dd5f2d1` |
| `runs/overnight-streaming-perf/tests/test_ffn_row_owner_runner.py` | `af2fe7bb16a5aeedae03b81e90a6b2fc1a1828567bf16be11f917fcf3dee554a` |
| `runs/overnight-streaming-perf/tests/test_measure.py` | `37692fdd8636324d00beeb3274a084c3a132f348a0070aa724a78a447648c958` |
| `runs/flash-qla/run.py` | `d5aaae525a5b732c0e03c31799724fa7a46c472c337f7aad2e3f7a8c94712a00` |
| `runs/flash-qla/measure.py` | `aa5720c49feb37f5d6bf807e1bc4dd1dd6292830111ceedd44818867de06aa23` |
| `runs/flash-qla/tests/test_native_bridge.py` | `fcdb5e9700773c4498b35eafa104f8ceda02bce83b6ad7f077211fdc952bad61` |

No native update, benchmark, cache probe, or GPU verification was launched by
this deliverable. The coordinator reports that all six FFN component GPU runs
exited successfully and the lease is released. The owner accepted the
investigated MTP/FP8 A/A variability and ended all further MTP/layer-0 numerical
investigation. Existing embedding-specific eight-rank NCCL, full-shape, state,
and memory evidence is retained; no numerical gate remains. The next worker
should run the prepared baseline/candidate inventory and peak commands, then the
requested combined 32-microbatch update and A/B 10+10 sequence under the sole
GPU lease. These preflight commands do not change the accepted config or set a
new tolerance policy.
