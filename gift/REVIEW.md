# Review Checklist

Use this checklist for merge request reviews. Focus first on correctness,
compatibility, tests, and maintainability. Style comments are secondary unless
they affect readability or future changes.

## Scope And Design

- The MR matches the task scope and does not include unrelated refactors.
- The design is explained for non-trivial changes.
- The code reuses existing repository logic instead of duplicating similar
  helpers, builders, model components, distributed utilities, or tests.
- Core upstream-derived files are changed only when necessary and the reason is
  documented.
- Core code is production-quality. Experimental or research-only logic stays in
  `torchtitan/experiments/` and does not leak into core paths.
- Unsupported or intentionally untested scenarios fail fast with explicit errors.

## Architecture Boundaries

- Config/schema code stays lightweight and does not import runtime, models,
  distributed code, `torch`, `docker/`, `scripts/`, or root-level `contrib/`.
- Runtime builders create objects from typed configs and do not read YAML or
  apply CLI overrides.
- Trainer code orchestrates runtime objects and callbacks instead of building
  models, optimizers, or dataloaders inline.
- Docker and submit logic stay outside core `torchtitan/*` modules.
- New functionality is modular and lives in the owning core package: config in
  `torchtitan/config/`, runtime components in `torchtitan/components/`, train loop
  logic in `torchtitan/trainer.py`, distributed primitives in
  `torchtitan/distributed/`, shared model pieces in `torchtitan/models/common/`,
  and model-specific code in the relevant model package. Config owners build
  their runtime objects; `torchtitan/config/build.py` assembles the model spec.
- Imports follow dependency direction: config does not import runtime/models/
  distributed/torch; models do not import trainer or submit/docker helpers;
  core does not import root-level `contrib/` directly; optional heavy imports are
  local to builders/callbacks. There are no circular dependencies.

## Config, Registry, Builders

- New config fields have validation and a YAML example or preset when user-facing.
- A new config class is declared with `@config` in the `torchtitan/config/schema/`
  module mirroring its owner's, named `<Owner>Config` with `_owner_spec`; the
  owner aliases it as `Config`. Nothing under `schema/` imports torch or a
  runtime module at import time (`tests/unit_tests/test_config_torch_free.py`).
- Config values are data: initializers are `InitSpec`s, placements are spec
  objects, coefficients are numbers. No callables or torch objects in a config.
- Local invariants live in `__post_init__`; checks that need the stage, the
  runtime or a sibling section live in `validate(ctx)` and return `Issue`s.
- A change that alters what a YAML decodes to re-pins the golden fingerprints
  and says why; a move that must not alter it shows the class-stripped dump
  diff is empty.
- New component `type` values are registered with `@config(type=...)` in the
  torch-free schema; trainer types have entries in `registry.TRAINER_TYPES`.
- YAML uses stable `type` names, not Python class paths.
- Builders import optional/heavy dependencies lazily and raise clear errors when
  dependencies are missing.

## Training Correctness

- Checkpoint save/load/resume behavior is covered when affected.
- Trainer/callback changes preserve callback order and checkpoint callback state
  when state affects training.
- Distributed changes validate world size, mesh axes, placements, and unsupported
  parallelism combinations.
- Distributed features have integration coverage or a documented manual GPU run.
- New distributed functionality preserves torch-native distributed design: it
  should compose with supported parallelisms and must not hard-code assumptions
  that block FSDP/TP/PP/CP/EP unless the unsupported combinations fail fast with
  explicit errors.
- Integer counters, accumulators and collectives that can pass 2**31 at
  production sizes (per-rank tokens x hidden, sums over ranks, microbatches or
  steps) are int64 end to end, kernel atomics included. See
  `.claude/rules/integer_overflow.md`.
- Model changes preserve checkpoint compatibility or document the breakage.
- Model architecture changes update the relevant HF/checkpoint conversion scripts
  or explicitly document why conversion is not affected.

## Tests And Evidence

- The MR includes focused tests or a clear reason why tests are not practical.
- Almost every new feature should add unit tests. Missing unit tests require a
  concrete explanation in the MR.
- Test commands and manual validation are documented in the MR.
- Numerics/loss evidence is provided for non-trivial training changes.
- Performance-sensitive changes report relevant metrics or explain expected risk.

## Code Quality

- Code is simple, local, and maintainable. If the same behavior can be expressed
  with less branching, less indirection, or a clearer existing abstraction, prefer
  that version.
- Names, errors, and validation messages describe the actual concept and include
  config paths or runtime context when helpful.
- The MR does not introduce avoidable global state, hidden import side effects,
  or broad behavior changes outside the touched feature.

## Documentation

- Public config/API/CLI/checkpoint changes update documentation or examples.
- Docker and submit flow changes include command examples.
- Follow-up work is documented when the MR intentionally leaves gaps.
