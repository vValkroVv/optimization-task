# Capacity group-counter correctness

## Scope

Implementation is limited to `gift/torchtitan/models/common/moe/dispatch/capacity.py`
and the existing focused token-drop test module. Baseline task revision is
`e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f`; original source SHA256 is recorded
in [pre-implementation cost review](preimplementation-costs.md). The task
root's concurrent planning edit and other workers' evidence are preserved.

## Required behavior

For each destination, local int64 expert counters are summed before the
functional EP all-reduce. The collective receives exactly EP int64 group
loads, including from empty sources. The original local expert counters remain
available to the local overload predicate and selector. The gate remains local
expert overload AND global group overload; capacity, score ordering/ties,
validity masks, routed IDs and scores are unchanged.

## Results

Candidate `capacity.py` SHA256 is
`164af05bd72ceb0675f0d9f756ecf04926ced17430a58dd2e38bc160c514dfdc`; the
affected test SHA256 is
`2abffc21929ed2dd1c4731f67a8602d051866ea496aecc20abdeaf4c476260b6`.
The exact original [source snapshot](capacity.original.py) matches the GIFT
capacity source at the starting task revision and uses the
[GIFT BSD 3-Clause license](../../gift/LICENSE). Snapshot SHA256 is
`2f034b072fdaf02ac844936366134fafcd5e815588ce750b0dc7dfcc50467120`.
The current reverse patch `original-order.patch` applies cleanly in check mode.

Runtime used: `/opt/torchtitan-venv/bin/python3.12`, Python 3.12.3,
PyTorch `2.15.0.dev20260822+cu130`, build
`8b353065ce260336c5fdfa74d754e8bd9b91b062`. Every CPU command below set
`CUDA_VISIBLE_DEVICES=''` and `PYTHONPATH="$PWD/gift"`.

| Check | Result |
| --- | --- |
| `pytest -q gift/tests/unit_tests/test_token_drop_fixed_capacity_group.py` after final formatting | 13 passed, including E128/EP8 payload, exact bounded oracle cases, and real two-rank Gloo; see `group-tests-formatted.log`. |
| `pytest -q gift/tests/unit_tests/test_token_drop_dispatchers.py::test_local_capacity_dispatch_and_gradient gift/tests/unit_tests/test_token_drop_dispatchers.py::test_all_to_all_with_real_collectives_empty_and_unequal_sources` | 9 passed, including exact gradients and real Gloo with empty/unequal sources; see `dispatcher-gradient-gloo.log`. |
| `pytest -q gift/tests/unit_tests/test_token_drop_compile.py::test_fullgraph_matches_eager_masks_and_gradients -k cpu` | 12 passed; 12 CUDA parameter cases were deselected; see `compile-tests.log`. |
| `pytest -q gift/tests/unit_tests/test_moe_checkpointing.py::test_dtensor_routing_and_sac_issue_one_real_group_collective` | 1 passed; the test exercises Gloo, compilation, gradients, and full/selective checkpoint policies; see `sac-gloo-tests.log`. |
| Scoped Ruff F821/F811/F841; full Ruff on both changed Python files; Ruff format check | All passed on final formatted files. |

The original-order negative control was run by applying the then-current
`original-order.patch` and verifying the source hash matched the exact original
snapshot. The E128/EP8 payload test failed as required: the mock observed shape
`[128]` where it required `(8,)`; full traceback is in
`original-order-regression.log`. The shell trap restored the candidate, and the
same payload test passed afterward (`payload-candidate.log`). The source was
then formatting-only adjusted; the final group test and payload assertion
passed again. The final patch was regenerated and `git apply --check` passed.

Pytest emitted the environment's non-failing `pynvml` deprecation warning.
The first Ruff format check requested formatting for both changed files; Ruff
format was applied and the final format check passes. No candidate correctness
failure remains.

## Reproducible original-order control

Applying `original-order.patch` to the candidate restores the original
E-counter all-reduce. To reproduce the demonstrated negative control, run:

```bash
git apply runs/capacity-group-allreduce/original-order.patch
sha256sum gift/torchtitan/models/common/moe/dispatch/capacity.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/gift" \
  /opt/torchtitan-venv/bin/python3.12 -m pytest -q \
  gift/tests/unit_tests/test_token_drop_fixed_capacity_group.py::test_ep_all_reduce_receives_eight_int64_group_counts_for_128_experts
git apply --reverse runs/capacity-group-allreduce/original-order.patch
```

The original hash is `2f034b072fdaf02ac844936366134fafcd5e815588ce750b0dc7dfcc50467120`;
the expected failure is the `(128,)` versus `(8,)` payload assertion. The
candidate hash after restoration is recorded above.

The delegated GPU checks also passed on the final candidate bytes:

| GPU check | Result |
| --- | --- |
| `pytest -q gift/tests/unit_tests/test_token_drop_compile.py -k cuda` | 13 passed, 82 deselected; log: `artifacts/raw/token-drop-compile-cuda.log`. |
| Existing Triton selector and unsupported-shape/dtype fallback targets below | 3 passed; log: `artifacts/raw/token-drop-triton-cuda.log`. |

```bash
PYTHONPATH="$PWD/gift" /opt/torchtitan-venv/bin/python3.12 -m pytest -q \
  gift/tests/unit_tests/test_token_drop_triton.py::test_selector_launches_triton_without_reading_device_scalars \
  gift/tests/unit_tests/test_token_drop_triton.py::test_cuda_fallback_preserves_unsupported_shapes_and_dtypes
```

The separate real EP Gloo/SAC path already exercised by this worker is
`gift/tests/unit_tests/test_moe_checkpointing.py::test_dtensor_routing_and_sac_issue_one_real_group_collective`.

## Remaining review and execution gates

The [independent integration/runtime-cost review](review.md) is complete with
no material finding. The [eight-rank NCCL component](component-results.md)
also passes exact group-load checks. GPU unit/compile/SAC checks pass as recorded
above. Native timing and the final trace remain open; focused checks do not
establish those gates.
