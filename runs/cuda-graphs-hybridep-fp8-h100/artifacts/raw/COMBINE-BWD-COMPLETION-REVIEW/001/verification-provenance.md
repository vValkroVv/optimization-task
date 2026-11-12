# Reused verification provenance

Owner `/root/combine_bwd_completion_implementation` supplied freeze notification after final source/tests/docs changes. Final source hashes are [source-pins.json](source-pins.json).

Owner focused003 ran in `gift/` using `/opt/torchtitan-venv/bin/python3.12`, `CUDA_VISIBLE_DEVICES=''`, importing torch first, asserting `torch.cuda.is_available() is False` and `torch.cuda.device_count() == 0`, replacing `torch.cuda.init` and `torch.cuda._lazy_init` with a function raising AssertionError, and only then importing pytest/test modules. Pytest arguments:

```text
-q tests/unit_tests/test_campaign_combine_bwd_completion.py tests/unit_tests/test_config_torch_free.py::test_config_package_imports_torch_free tests/unit_tests/test_config_torch_free.py::test_component_types_register_torch_free_and_are_pinned
```

[Recorded output](../../COMBINE-BWD-COMPLETION-IMPLEMENTATION/001/pytest-focused-003.log): 38 passed in 4.28s. The installed pynvml deprecation warning is retained. This proves the stated CPU/double/AOT behavior, not CUDA/NCCL/SAC/FP8 native runtime correctness.

Owner reports this scoped command passed (`All checks passed!`), cwd `gift/`:

```text
ruff check --select F821,F811,F841 --no-cache torchtitan/experiments/cuda_graph_campaign/combine_bwd_completion.py torchtitan/experiments/cuda_graph_campaign/train_combine_bwd_completion.py torchtitan/config/schema/experiments/cuda_graph_campaign/combine_bwd_completion.py tests/unit_tests/test_campaign_combine_bwd_completion.py tests/unit_tests/test_config_torch_free.py
```

Owner reports diff whitespace and trailing-whitespace checks clean. This reviewer did not repeat that suite/lint. Independent final assertions evaluated only the pure path expression from AST, compared real file/hash, confirmed ordinary source diffs empty and base preset lacks the diagnostic callback. See final-targeted-assertions.log. This source-level check specifically closes the discovered real-path defect; it does not replace behavior tests.
