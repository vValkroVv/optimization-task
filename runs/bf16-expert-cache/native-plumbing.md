# Native runner plumbing

## Scope

The task-local native [runner](../overnight-streaming-perf/run.py) and
[measurement adapter](../overnight-streaming-perf/measure.py) now have two
opt-in controls:

| Runner option | Default | Native effect |
| --- | --- | --- |
| `--gfsdp-reuse-bf16-expert-weights` | off | Sets `parallelism.gfsdp_reuse_bf16_expert_weights=true`; the runner verifies the resolved value. |
| `--bf16-cache-counters` | off | Enables GFSDP cache counters only for a profile-update diagnostic. It requires the feature option and `--profile-update`. |

The unprofiled path does not set the debug-counter flag, register a microbatch
hook, or snapshot counters per update. After training it records one
`activation_metadata` snapshot to verify that eligible groups were active. The
diagnostic enables `set_bf16_cache_debug(True, recurse=True)` before training,
records a pre-training snapshot, then snapshots all groups after each completed
optimizer update. These boundaries cover all 32 microbatches; the three
microbatch profiler cap does not shorten training or counter collection. The
rank receipts keep `fqn`, `enabled`, `generation`, `refreshes`, `hits`,
`direct_uses`, and `invalidations`. No source-owned printing was added.

The runtime-cost impact from this plumbing is limited to one native boolean
override and receipt fields in the timing run, plus one post-run activation
snapshot outside the measured window. Diagnostic mode adds one setter call and
one stats snapshot at each optimizer boundary, with no per-microbatch callback
or extra synchronization. Source-side cache compute and memory costs remain
owned by the implementation and its performance review.

## Commands for the two authorized native runs

These commands were constructed and parsed by the tests below; they were not
launched as part of this plumbing task.

```bash
/opt/torchtitan-venv/bin/python3.12 runs/overnight-streaming-perf/run.py \
  --run-id bf16-cache-tps-001 --steps 60 --warmup 20 --window 40 \
  --num-prefetch-batches 32 --epoch-size 51200 \
  --gfsdp-reuse-bf16-expert-weights

/opt/torchtitan-venv/bin/python3.12 runs/overnight-streaming-perf/run.py \
  --run-id bf16-cache-trace-001 --steps 21 --warmup 20 --window 0 \
  --profile-update 21 --profile-microbatches 3 \
  --num-prefetch-batches 32 --epoch-size 51200 \
  --gfsdp-reuse-bf16-expert-weights --bf16-cache-counters
```

Parsed child schedules are exactly 60/20/40 and 21/20/0. The diagnostic child
sets profiler ranks to `[0, 1]`, skip-first to 19, and microbatch capture to 3.
Both preserve prefetch depth 32, logical epoch 51,200, LR horizon/warmup
65,000/2,000, the existing native config, and no `--capture-microbatches` host
counter. The runner's resolved-config check verifies local/global batch sizes,
FP8, compile, selective SAC, EMA, the feature flag and profile activation. The
config-level regression check also confirms the existing recipe computes 32
accumulation steps and retains MTP. The base preset was not rewritten, and no
four-fusion override was added.

## Checks

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/gift:$PWD/runs/overnight-streaming-perf" \
  /opt/torchtitan-venv/bin/python3.12 -m pytest -q \
  runs/overnight-streaming-perf/tests/test_bf16_cache_runner.py
```

Result: **6 passed** in
[`test_bf16_cache_runner.py`](../overnight-streaming-perf/tests/test_bf16_cache_runner.py).
The checks cover opt-in defaults and invalid combinations,
both exact helper schedules and child overrides, rank selection, default-off
timing instrumentation, diagnostic snapshots across 32-microbatch optimizer
updates, and fallback-group receipt rows.

Existing affected runner checks also passed:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/gift:$PWD/runs/overnight-streaming-perf" /opt/torchtitan-venv/bin/python3.12 -m pytest -q --basetemp=runs/overnight-streaming-perf/artifacts/raw/.pytest-native-measure-001 runs/overnight-streaming-perf/tests/test_measure.py`: **50 passed**.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/gift:$PWD/runs/overnight-streaming-perf" /opt/torchtitan-venv/bin/python3.12 -m pytest -q runs/overnight-streaming-perf/tests/test_bounded_profile.py`: **13 passed**.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/gift:$PWD/runs/overnight-streaming-perf" /opt/torchtitan-venv/bin/python3.12 -m pytest -q runs/overnight-streaming-perf/tests/test_epoch_size.py::test_epoch_size_cli_validation_and_existing_run_schedules`: **1 passed**.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/gift:$PWD/runs/overnight-streaming-perf" /opt/torchtitan-venv/bin/python3.12 -m pytest -q --basetemp=runs/overnight-streaming-perf/artifacts/raw/.pytest-native-config-002 runs/overnight-streaming-perf/tests/test_epoch_size.py::test_child_receives_native_epoch_override_and_default_stays_null`: **1 passed** after adding the opt-in flag to the fixture.
- After the config field landed, the full `test_epoch_size.py` and the actual resolved-config/MTP test in `test_short_performance.py` were included in the combined run below.

The first full `test_measure.py` attempt used pytest's default temporary path,
which resolves through the external `.sr008-gpu/.cache` symlink. Six existing
task-path guard tests consequently failed because their temporary roots were
outside `TASK_ROOT`; the other 44 passed. Rerunning with the task-local ignored
raw-artifact `--basetemp` above passed all 50. This was a test-environment path
issue, not a runner assertion failure.

The first actual-config attempt found that emitting an explicit native `false`
added a key to the `parallelism` mapping and broke equality with the canonical
recipe. The runner now omits that override when disabled; the schema default is
false. The combined rerun then reported **75 passed, 1 failed**: the new
opt-in config assertion had not yet added the opt-in flag to its test fixture.
After correcting that fixture, the exact child-config test passed (**1 passed**).
Thus all 76 unique tests across the focused BF16 runner suite, full
measurement/bounded-profile/epoch-size suites, and the actual native
config/MTP check passed across the two invocations.

The combined command was:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/gift:$PWD/runs/overnight-streaming-perf" \
  /opt/torchtitan-venv/bin/python3.12 -m pytest -q \
  --basetemp=runs/overnight-streaming-perf/artifacts/raw/.pytest-native-final-002 \
  runs/overnight-streaming-perf/tests/test_bf16_cache_runner.py \
  runs/overnight-streaming-perf/tests/test_measure.py \
  runs/overnight-streaming-perf/tests/test_bounded_profile.py \
  runs/overnight-streaming-perf/tests/test_epoch_size.py \
  runs/overnight-streaming-perf/tests/test_short_performance.py::test_short_command_resolves_preserved_recipe_without_monitoring
```

Scoped lint:

```bash
ruff check --select F821,F811,F841 --no-cache \
  runs/overnight-streaming-perf/run.py \
  runs/overnight-streaming-perf/measure.py \
  runs/overnight-streaming-perf/tests/test_bf16_cache_runner.py \
  runs/overnight-streaming-perf/tests/test_epoch_size.py \
  runs/overnight-streaming-perf/tests/test_measure.py
```

Result: **passed**.

## Integration gate at plumbing handoff

No GPU run, allocation, installation, or source edit was performed here. These
tests use fake model-part methods with the same signatures as the now-present
`GFSDPModule.set_bf16_cache_debug(enabled, recurse=True)` and
`GFSDPModule.bf16_cache_stats(recurse=True)` implementation. The config field,
source wiring and API signatures are present. The source-owned actual-GFSDP
fixture must still exercise this runner path; native activation and GPU
measurement remain open for the parent task. No runtime result is claimed here.

Subsequently, the actual-GFSDP checks and both authorized native launches passed.
The [completed report](report.md) owns the integrated results and limits.
