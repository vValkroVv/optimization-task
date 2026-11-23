# Deferred direct expert dW slot wait

This experiment keeps the existing GFSDP host claim and captures the distinct
release events for overlapping gradient-slot regions at acquisition. The first
writer waits on those captured events on the lease stream, after Python
validation and operand view construction. The pinned native writer still runs
its own C++ validation before launching. Both projections use the existing
checks, FP32 destination and writer. The ordinary gradient path and native
reductions remain unchanged.

A rejected lease finish joins outstanding captured events and records its
replacement fence on the original stream. A stale lease cannot release a newer
claim. Unexpected exceptions during native backward retain the inherited
training fail-stop behavior.

## Overlay identity

Use an isolated, task-owned campaign copy synced from the expert-GFSDP
candidate working tree. That tree includes changes on top of
`e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f`. The patches use paths relative
to the outer task repository.

| Patch | SHA-256 |
| --- | --- |
| `deferred-dw.patch` | `8f87b29e4ecf8804c0947ae7687599bbaab23bddec02af4bbbe1f6ae05ff39b1` |
| `deferred-dw.reverse.patch` | `c54141a7ada2ae71aa64bbbbd7e175ae3b294fad0e0a339c5798fd3cdbfe9b26` |

| Source file in `gift/` | Base SHA-256 | Applied SHA-256 |
| --- | --- | --- |
| `torchtitan/distributed/gfsdp/_param_group.py` | `64f43a0d85b14d2552c07437bc1dd10ea15215c2e7fb63660e950832204f2732` | `3c1a94b7a2a3e07993bc0c72b961659f6f923f3bcb1bdb3eb69242e580d769eb` |
| `torchtitan/ops/float8/direct_expert_dw.py` | `a228168d731af14347e2258d86e24a5d7d4b0aa6988b896f9fa1b2cf20d833e5` | `ea883a2b0e37a3a34ec6fa94b1d60b07f0d6c5e791bb2f411c7200b4c7c204b8` |
| `tests/unit_tests/test_direct_expert_dw.py` | `7a25d6ad2ec4ec04e713250c8c50e7879b7b23ad15edadaf4401e897b4d139d2` | `08e8d680060793c71fd629abbe2e05efdd2bf1af5fa97c6916ecc5b3e01ca8eb` |

From the isolated copy, apply and later reverse with the matching patches.

```bash
experiment="$PWD/runs/efsdp-optimizations/experiments/deferred-dw"
git apply --check "$experiment/deferred-dw.patch"
git apply "$experiment/deferred-dw.patch"
# Run the applicable verification below before reversing.
git apply --check "$experiment/deferred-dw.reverse.patch"
git apply "$experiment/deferred-dw.reverse.patch"
```

## CPU check

In the disposable patched copy, put the focused test beside the existing
GIFT fixture so pytest loads the repository's Triton import shim when needed.
Use the existing environment Python.

```bash
cp "$experiment/test_wait.py" gift/tests/unit_tests/test_deferred_dw_wait.py
PYTHONPATH="$PWD/gift:$experiment" python -B -m pytest -q -p no:cacheprovider \
  gift/tests/unit_tests/test_direct_expert_dw.py \
  gift/tests/unit_tests/test_deferred_dw_wait.py
```

The isolated local copy passed all 25 CPU and mock tests. The reverse patch
restored the three patched files byte for byte. The deferred-order test then
failed on the restored baseline at the early-acquire wait assertion.

## Native two-rank gate before training

Use two authorized idle H100s, the campaign's initialized pinned Torch
environment and the exact current direct dW build manifest. Supply an absolute
`build_root` and a fresh task-owned `evidence` directory. The command creates
one JIT cache per rank and links to the verified build artifacts. The component warms both
writer paths, the delay kernel, marker and the exact NCCL shape on explicit
compute and reader streams. It submits the reduce-scatter through Torch's
synchronous API, which enqueues NCCL directly on the priority −1 reader
stream in the pinned build. [Native GFSDP submission](../../../../gift/torchtitan/distributed/gfsdp/_collectives.py) · [Pinned Torch NCCL stream choice](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L3803-L3810)

Each first-writer order has a matched scheduling control with the same delayed
NCCL reader and unrelated marker. The control skips lease acquisition and slot
writes. Both cases require the marker to finish before the reader and verify
every prior NCCL output bit. The leased case also requires the reader to finish
before the writes complete and verifies every new slot bit. The test clears the
previous release field after acquisition to exercise captured-event lifetime.

```bash
DG_CACHE_FORCE_SKIP=1 PYTHONPATH="$PWD/gift:$PWD/runs/efsdp-optimizations/direct-dw" \
  python -B -m torch.distributed.run --standalone --nproc-per-node=2 --no-python \
  /bin/bash -c '
    set -eu
    rank_build="$1/build-rank$RANK"
    mkdir -p "$rank_build/jit"
    for name in manifest.json source extension; do
      ln -s "$2/$name" "$rank_build/$name"
    done
    exec python -B "$3" --build-root "$rank_build" --num-sms 116 --evidence "$1"
  ' bash "$evidence" "$build_root" "$experiment/component.py"
```

Each rank records both cases, runtime and source identities, the effective
scheduling environment, host submission times, nonblocking reader readiness
queries and device event times. Receipts survive a failed numerical or ordering
gate. A failed scheduling control leaves the overlap question unresolved. Use
the host NCCL submission time and reader readiness to distinguish late host
submission from device scheduling. CUDA can serialize independent streams that
share a work queue. Keep the task's normal CUDA settings for the first check.
If the control indicates device serialization, `CUDA_DEVICE_MAX_CONNECTIONS=8`
is a separate diagnostic setting on a host without MPS. [CUDA work queues](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-device-max-connections)

The earlier asynchronous fixture put both rank markers just after the roughly
51 ms reader completion in its first case. Those timestamps do not identify
where submission was delayed. The revised fixture passed both writer orders
on two H100s in attempt `deferred-dw-component-04`. The matched EP4 campaign
covers compiled backward, full GFSDP accumulation, timing, and training numerics.
[Rank 0 receipt](../../artifacts/raw/deferred-dw-component-04/rank0.json) ·
[Rank 1 receipt](../../artifacts/raw/deferred-dw-component-04/rank1.json)
