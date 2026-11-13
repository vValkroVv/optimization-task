# FLCE target metadata correction

Status: R1 from the [loss integration review](loss-review.md) is resolved.
The source base is task root `33b336a5`; the preserved native loss file was
restored after the focused integration check.

## Change and contract

The public `native_linear_ce` adapter now validates, before flattening or
dispatching the handwritten custom op:

- `labels.dtype == torch.int64`;
- `labels.shape == input.shape[:-1]`;
- `input`, `labels`, `weight`, and `denominator` share a device.

These checks read tensor metadata only. They preserve the signed int64 target and
row-address contract, and they prevent the unchecked Triton target load from
receiving a short or incompatible target view. The native unit-upstream backward,
valid MTP/two-chunk path, compile settings, and fallback selection are unchanged.

## Runtime-cost disposition

The selected adapter is called once per outer sequence chunk and microbatch
(64 calls per optimizer update with 32 microbatches and two chunks). The change
adds Python metadata comparisons at that boundary. It adds no tensor-value
reads, allocations, copies, kernels, or collectives, and causes no device
synchronization. The 1536 versus 192 head GEMM count and the reviewed live dW /
logits overlap remain unchanged. No performance or memory measurement was run
for this metadata-only correction; its host dispatch cost is not quantified.

## Verification

Only physical GPU 2 was used: H100 80 GB, UUID
`GPU-2f380105-fce8-292b-f1a6-54dbea10840a`. Before the focused runs it showed
0 MiB used, 0% utilization, and no compute applications; it was again idle after
the integration run. Runtime was `/opt/torchtitan-venv/bin/python3.12` (Python
3.12.3, Torch `2.15.0.dev20260822+cu130`, CUDA 13.0, Triton 3.8.0). Commands used
the task-root working directory and this prefix:

```bash
source /etc/profile.d/wm.sh
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=gift
export TORCHINDUCTOR_CACHE_DIR=$PWD/runs/four-fusion/artifacts/raw/loss-checks/inductor
```

The final focused commands and results were:

```bash
/opt/torchtitan-venv/bin/python3.12 -m pytest -q gift/tests/unit_tests/test_fusion_losses.py::test_native_linear_ce_rejects_invalid_metadata
# 10 passed: five invalid metadata cases in eager and fullgraph-compiled mode.

# Applied runs/four-fusion/linear-ce.patch temporarily to activate this existing path.
/opt/torchtitan-venv/bin/python3.12 -m pytest -q gift/tests/unit_tests/test_fusion_losses.py::test_native_mtp_two_chunks_loss_hidden_weight_and_identity
# 2 passed: eager and compiled native MTP/two-chunk integration.

# Reversed the temporary activation patch and checked the core-file hash.
sha256sum gift/torchtitan/components/loss.py
# 14d546d63b069a69041ec4a379d10a06f9b6a7888f25a68dddcadf27293b7f7c
```

Fullgraph reports Python validation exceptions as Dynamo
`torch._dynamo.exc.Unsupported` during tracing. The test checks that the
diagnostic contains the corresponding metadata validation message. The eager
path raises `ValueError` directly. Both pytest commands emitted the environment's
existing `pynvml` deprecation `FutureWarning`; neither skipped a test.

The baseline int32 behavior was checked with this CPU command; exit status 1 was
expected because native cross entropy rejects int32 targets:

```bash
source /etc/profile.d/wm.sh && /opt/torchtitan-venv/bin/python3.12 -c 'import torch; import torch.nn.functional as F; x=torch.zeros(4,17); y=torch.zeros(4,dtype=torch.int32); F.cross_entropy(x,y,reduction="sum")'
# RuntimeError: expected target dtype to be Long or Byte, but got Int
```

The first metadata-test attempt expected eager `ValueError` from both modes. It
reported 5 passed and 5 failed because all five fullgraph cases correctly
rejected during tracing with Dynamo `Unsupported`, retaining the validation
message. The test expectation was updated for that compile-time behavior; the
final command above passes all ten cases. The initial format check also exited 1
because the new test needed formatting; `ruff format` corrected it.

Final scoped quality checks:

```bash
ruff check --select F821,F811,F841 --no-cache gift/torchtitan/experiments/four_fusion/linear_ce.py gift/tests/unit_tests/test_fusion_losses.py
ruff format --check gift/torchtitan/experiments/four_fusion/linear_ce.py gift/tests/unit_tests/test_fusion_losses.py
git diff --check
```

All three passed. No full-module suite, training, TPS run, profiling, install, or
publication was performed. The production integration, full-shape memory, and
performance gates retained in the review remain open.
