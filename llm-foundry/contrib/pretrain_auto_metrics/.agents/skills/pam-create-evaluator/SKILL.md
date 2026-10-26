---
name: pam-create-evaluator
description: >
  Add a new evaluator (benchmark/metric) to pretrain-auto-metrics (PAM).
  Always use this skill before creating any new evaluator in PAM: it encodes
  the required patterns, config, and conventions.
---

# PAM: create an evaluator

The best new evaluator is the closest existing one with only the
benchmark-specific parts changed. Every field, method, and config that already
lives in a Reference Evaluator is a decision someone made for you — don't
re-make it.

Priorities, in order: **correctness → consistency with the Reference Evaluator
→ minimal diff → maintainability.** When they conflict, prefer what a future
maintainer reads fastest.

## Read first

Read these fully before editing — understand the flow before touching the diff:

1. `node_runner/src/metrics/evaluators/interface.py` — base classes and the methods you must implement.
2. `node_runner/src/config_model.py` — config shapes and `apply_defaults()` flow.
3. `node_runner/src/metrics/metrics_manager.py` — registration buckets and phase wiring.

## Pick the Reference Evaluator

Pick one reference and mirror it. Phase shape matters more than benchmark branding.

| Need | Base | Best Reference |
| --- | --- | --- |
| evaluator around an existing eval framework | `MetricsEvaluator` | `lm_eval.py` |
| standalone benchmark CLI: set endpoint env, run external tool, parse artifacts | `MetricsEvaluator` | `harness_bench.py` |
| RLI / terminal environment | `MetricsEvaluator` | `terminal_bench.py` |
| judge model scores saved generations | `ScoringMetricsEvaluator` | `dredd.py` |
| evaluated model runs in parallel against secondary models | `ParallelInferenceEvaluator` | `tau2.py` |

## Files to touch

A normal new evaluator touches only these:

- `node_runner/src/metrics/evaluators/<name>.py`
- `node_runner/src/metrics/evaluators/__init__.py`
- `node_runner/src/config_model.py`
- `node_runner/src/metrics/metrics_manager.py`

Plus the config YAMLs: declare the new evaluator in every config that lists
evaluators — all the base `configs/*.yaml`, and the `configs/alignment_prod/` presets it
belongs to. Mirror how `terminal_bench` / `tau3` / `swe_bench` are declared. An
evaluator wired only into the domain config its author ran is invisible to
everyone measuring from a base config.

Touch tests only when the user asks for tests; then follow `/pam-tests`:
`.agents/skills/pam-tests/SKILL.md`.

## The rule of extension

Mirror the Reference Evaluator. Do not improvise structure.

- A field or method that copy+pastes from the reference — copy it.
- A method the reference lacks — borrow the closest version from another evaluator, then minimal edits.
- Drop reference-only fields and methods this benchmark doesn't need.
- Preserve each method's signature and the order of commands inside it.
- Same logic flow as the corresponding method. Deviate only where the benchmark
  differs, and only as much as it differs.
- Keep master/worker behavior identical to the matching phase.
- Run external tools via `SubprocessManager` / `SubprocessPoolRunner`, like the evaluator with the same run shape.
- Reuse `MetricsEvaluator` helpers for metrics dirs, checkpoint parsing, JD
  payloads, tar archives, transfer symlinks. Don't re-implement them.

## No raise inside the evaluator

An evaluator class does not `raise` at runtime — not for validation, not for
"impossible" states, and never a `try/except` that re-raises. Where to put each
case instead:

- **Subprocess failed** → `SubprocessManager.log_error()`, not `raise_for_exit_code()`.
- **Artifacts missing or unparseable** → return empty metrics and `Logger.error` with the path.
- **Config is wrong** → validate in the config model, where raising at startup is correct.
- **A phase failed and nothing is worth saving** (generation of a `ScoringMetricsEvaluator`, scoring whose outputs are unusable) → raise: the scheduler then skips the scoring stage instead of booting the judge, and saves no metrics.

The reason is what a raise throws away. `DynamicScheduler._run_task` catches
`Exception` per evaluator step and then skips `save_evaluator_metrics`, so a raise
discards whatever the benchmark had already measured.

The obligation that comes with it: do not vanish quietly.
`save_evaluator_metrics` returns silently when `get_metrics()` is empty, so a
swallowed failure disappears from the report with no trace. Log the absence, with
the reason, as an explicit error line.

An infrastructure failure — dead proxy, verifier crash, unreachable API, missing
DNS — removes the metric. It never lowers it. A benchmark that scored the model on
requests that never reached it must publish nothing and say why.

## Config

- One evaluator → one config class, inheriting `BaseConfigModel`.
- Serializing the config into benchmark args is a method on the config model, not on the evaluator, and it uses `model_dump` with an explicit `exclude` set — not a hand-rolled overrides builder, not `setdefault` chains over a dict.
- The evaluator never mutates the config.
- Validation belongs here and may raise. Meaningless values must not validate: zero retries, zero timeout, empty task list. `ge=0` on a retry count is a bug.
- Reuse field names, aliases, defaults, and `apply_defaults()` shape from the reference.
- Drop reference-only config fields this evaluator doesn't need.
- Keep runtime-only fields out of the serialized benchmark CLI args.
- Defaults that depend on inference instances, the inference base URL, or the
  checkpoint path go through the existing `apply_defaults()` flow.
- Dict-shaped configs (e.g. `dredd`) stay dict-shaped unless the change needs typed validation.

## Required knobs: timeout and parallelism (mandatory)

Every new evaluator MUST expose two configurable knobs for requests to the
served model. An evaluator missing either is incomplete.

- **request timeout** — a `timeout` field (seconds) defaulting to the shared
  `DEFAULT_REQUEST_TIMEOUT` constant in `config_model.py` (never hardcode the
  number), threaded into whatever the underlying tool reads:
  a CLI arg, an env var, or an emitted config key. A small/absent client timeout
  is what killed forge on a slow reasoning model under load. Mirror an evaluator
  that already does it: forge / dredd (`config_model.py`), tau2
  (`Tau2SamplingConfig.to_llm_args`), bfcl (`OPENAI_TIMEOUT` env), OBM /
  lmms_eval (`timeout` in the shooter `api` block).
- **parallelism** — a field controlling request concurrency to the served model
  (`workers` / `parallel_api_calls` / `num_threads` / `concurrency`), usually
  scaled off `inference_instances` in `apply_defaults()`.

If the underlying tool has no hook for one of these, add the tool-side hook (the
tools are vendored in `contrib/`); if that is genuinely out of scope, say so
explicitly in the PR rather than shipping silently — and see **Committing**.

## Committing

If the user asks you to commit an evaluator that lacks the timeout and/or
parallelism knob, prepend this exact line to the FIRST line of every commit
message AND to the TOP of the MR/PR description:

`⚠️ MISSING KNOBS: evaluator committed without configurable <timeout|parallelism|timeout+parallelism> (see /pam-create-evaluator).`

Fill in which knob is missing. This is mandatory and overrides the usual clean
commit-message convention.

## Endpoints

The evaluator must not know where it shoots. Do not accept an endpoint as a
parameter, do not branch on SLURM env vars, do not detect internet availability
per shooter. PAM decides RLI vs proxy once, through the existing availability
check in the evaluator interface, and hands the shooter a single well-known env
var. Adding a shooter must not add a second place where that decision is made.

## Skipping work

Skip an expensive phase on the metrics themselves, or on a marker written as the
last step of the phase. An output directory is not proof: a preempted or killed
run leaves one behind, and the next run will report a truncated result as a real
one.

## Registration

Register in the `EVALUATORS` dict in `metrics_manager.py`. The base class decides
which engines each stage needs through `stages()`:

- `MetricsEvaluator` — one `Stage` on the evaluated model (`run()`).
- `ScoringMetricsEvaluator` — `run_phase("generation")` on the evaluated model, then `run_phase(SCORING_PHASE)` on the single judge from `secondary_model_names`.
- `ParallelInferenceEvaluator` — one stage per secondary model, each next to the evaluated model (`run_on_model_name`).

`get_metrics()` is called once, after a metrics-producing stage finished, so it needs no phase checks.
Judge URLs and instance counts come from `config.judge_endpoints[name]`; the manager fills them before `apply_defaults()`.

Then export it from `node_runner/src/metrics/evaluators/__init__.py`.

## Metrics

- `get_metrics()` returns `dict[str, float]` for TensorBoard / internal reporting.
- `after_run()` sends JD metrics, matching the reference. Master-only gating is
  handled by `metrics_manager.py` (only the master runs evaluators) — do not add `is_master` checks
  inside the evaluator.
- JD keys are explicit and stable. Normalize only where an existing evaluator already does it for the same metric kind.
- Keep raw benchmark artifacts in `metrics/<evaluator_name>` unless the reference uses another layout.
- Match PAM convention on scale: values are usually fractions, not 0–100.

## Final check

- Wired in config, exports, and exactly one manager bucket.
- Declared in all base `configs/*.yaml` and the relevant `configs/alignment_prod/` presets.
- Exposes a configurable timeout knob and a parallelism knob (see **Required knobs**).
- Mirrors the reference; no new shape unless the benchmark forces it.
- Skips expensive benchmark runs when their metrics already exist.
- Preserves raw model answers when the benchmark supports it.
- Leaves no publishable metrics after a technical benchmark failure.