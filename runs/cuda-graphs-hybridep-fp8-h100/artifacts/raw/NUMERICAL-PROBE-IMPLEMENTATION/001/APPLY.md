# Applying the staged numerical probe

The files under `staged/gift/` are review copies. The patch is based on the current Source008 `gift/` tree and preserves the phase-memory README append. Do not apply it to the source tree as part of this staging handoff.

From the optimization-task repository root, check the exact patch first:

```sh
git apply --check --whitespace=error runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NUMERICAL-PROBE-IMPLEMENTATION/001/numerical-probe.patch
```

Once root accepts this corrected patch for integration, the existing task source-edit authority covers the named `gift/` paths. Apply the patch with:

```sh
git apply runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NUMERICAL-PROBE-IMPLEMENTATION/001/numerical-probe.patch
```

It adds or updates only these seven paths:

- `gift/torchtitan/config/schema/experiments/cuda_graph_campaign/numerical_probe.py`
- `gift/torchtitan/config/schema/experiments/cuda_graph_campaign/trainer.py`
- `gift/torchtitan/experiments/cuda_graph_campaign/continuation.py`
- `gift/torchtitan/experiments/cuda_graph_campaign/numerical_probe.py`
- `gift/torchtitan/experiments/cuda_graph_campaign/trainer.py`
- `gift/torchtitan/experiments/cuda_graph_campaign/README.md`
- `gift/tests/unit_tests/test_campaign_numerical_probe.py`

The README hunk was generated against Source008 with base SHA256 `079c7156fd77389fad3fda0e84c262a6b035ea9917b3c3f16a1d449c6b4f160b`. If that README has changed, stop and rebase the README hunk while retaining both the existing phase-memory section and the numerical-probe section.

After integration, run the scoped CPU test with a fresh artifact-local basetemp, then run full Ruff and format checks on the six changed Python paths:

```sh
cd gift
python -m pytest -q -p no:cacheprovider --basetemp=/workspace-SR008.nfs2/vkropotin/optimization-task/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NUMERICAL-PROBE-IMPLEMENTATION/001/pytest-after-integration tests/unit_tests/test_campaign_numerical_probe.py
ruff check --no-cache torchtitan/config/schema/experiments/cuda_graph_campaign/numerical_probe.py torchtitan/config/schema/experiments/cuda_graph_campaign/trainer.py torchtitan/experiments/cuda_graph_campaign/continuation.py torchtitan/experiments/cuda_graph_campaign/numerical_probe.py torchtitan/experiments/cuda_graph_campaign/trainer.py tests/unit_tests/test_campaign_numerical_probe.py
ruff format --check torchtitan/config/schema/experiments/cuda_graph_campaign/numerical_probe.py torchtitan/config/schema/experiments/cuda_graph_campaign/trainer.py torchtitan/experiments/cuda_graph_campaign/continuation.py torchtitan/experiments/cuda_graph_campaign/numerical_probe.py torchtitan/experiments/cuda_graph_campaign/trainer.py tests/unit_tests/test_campaign_numerical_probe.py
```

The numerical-comparison contract still requires separate small auxiliary/router payload additions before any ten-update generation; they are outside this patch. The CPU test does not establish target GPU behavior. Before using `updates: 10`, complete the mandatory one-update target readiness probe and review its inventory, readback, actual storage use, and host/device costs. Keep the CUDA fault gates and production A/A acceptance unset until their separate evidence exists.
