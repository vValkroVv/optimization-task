# Native runner controls and command-plan checks

## Scope

Added three independent default-off switches to the task runner:

- `--gfsdp-split-mtp-ffn` → `parallelism.gfsdp_split_mtp_ffn`
- `--gfsdp-split-layer0-ffn` → `parallelism.gfsdp_split_layer0_ffn`
- `--gfsdp-row-owner-embeddings` → `parallelism.gfsdp_row_owner_embeddings`

When selected, each emits only its typed native `true` override. The runner
expects every field in the resolved config, including explicit `false` defaults,
and records all three requested values in attempt metadata. `measure.py` and its
timing hooks are unchanged. Actual activation is gated on resolved rank-zero
configuration and the source setup markers recorded in [native-plan.json](native-plan.json).

## Checks

Final scoped checks used the task interpreter and kept CUDA hidden for the
process. The tests replay the wrapper, native runner, generated child command,
and helper parser; they cover default behavior, every independent switch, the
combined A/B invariants, the three-microbatch diagnostic, resolved config, and
FlashQLA `external_gate` forwarding.

```text
CUDA_VISIBLE_DEVICES='' /opt/torchtitan-venv/bin/python3.12 -m pytest -q --basetemp=runs/overnight-streaming-perf/artifacts/raw/pytest-native-controls runs/overnight-streaming-perf/tests/test_ffn_row_owner_runner.py runs/overnight-streaming-perf/tests/test_bf16_cache_runner.py runs/overnight-streaming-perf/tests/test_measure.py runs/flash-qla/tests/test_native_bridge.py
77 passed, 3 warnings

ruff check --select F821,F811,F841 --no-cache runs/overnight-streaming-perf/run.py runs/overnight-streaming-perf/measure.py runs/overnight-streaming-perf/tests/test_ffn_row_owner_runner.py runs/overnight-streaming-perf/tests/test_measure.py
All checks passed.

git check-ignore -v runs/ffn-row-owner/artifacts/raw/
.gitignore:19:/runs/**/artifacts/raw/
```

The plan test regenerates each `child_torchrun_argv` through `_parse_args`,
`_training_overrides`, `_command`, and `measure._parse_args`; the outer arrays
also pass through the FlashQLA bridge parser. It confirms that A and B share
the accepted config and measurement settings, with B adding only the three
feature overrides. A has profiling and per-step instrumentation off. The
separate diagnostic captures update 21 on ranks 0/1, requests at most three
microbatches, and retains the native 32-microbatch optimizer update.

After adding the final verification record to the plan JSON, its command arrays
were replayed once more:

```text
CUDA_VISIBLE_DEVICES='' /opt/torchtitan-venv/bin/python3.12 -m pytest -q --basetemp=runs/overnight-streaming-perf/artifacts/raw/pytest-native-controls runs/overnight-streaming-perf/tests/test_ffn_row_owner_runner.py::test_campaign_plan_commands_match_runner_and_measurement_hooks
1 passed
```

## Failed harness attempts retained

The first pytest invocation used the environment's default temporary directory:
69 passed and 7 failed. One new fixture had a mismatched `dump_folder`; the
other failures came from existing task-path checks rejecting temporary paths
outside the task root. A retry initially named a basetemp whose parent did not
exist (15 passed, 61 setup errors). A retry under this campaign's raw root ran
71 tests successfully but five existing adapter checks rejected its noncanonical
artifact layout. The final command uses the native runner's own ignored raw root
and passes all 77 tests. No failing check remains in this scoped runner suite.

## Execution gates

No GPU probe, training, allocation, install, or publication was performed.
Owner authorization for the existing eight-H100 allocation is recorded in the
plan, but component correctness and H100/NCCL measurements, integrated source
readiness, a full 32-microbatch all-three smoke, source freeze, and the root
coordinator's serialized GPU lease remain ahead of A/B and diagnostic launches.
The plan makes no speed or feature-activation claim.
