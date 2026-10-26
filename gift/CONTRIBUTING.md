# Contributing

This repository is a fork-first GitLab codebase based on TorchTitan. We do not
regularly sync with upstream TorchTitan. Upstream changes are cherry-picked or
ported when they are useful for our production training workflow.

For agent/developer operating rules, see [AGENTS.md](AGENTS.md).

## Setup

```bash
pip install -r requirements.txt -r requirements-dev.txt
pre-commit run --all-files
pytest tests/ -x
```

## Before Starting Work

Before implementation, create or update the task with:

- scope and expected behavior;
- proposed design for non-trivial changes;
- affected areas: config, runtime, trainer, model, distributed, data, checkpoint,
  Docker, submit scripts, or CI;
- testing plan and expected risks.

Keep changes scoped. Do not mix broad refactors with feature work unless the
refactor is required for the feature and described in the MR.

## Merge Requests

Use GitLab merge requests and fill the default MR template:
[.gitlab/merge_request_templates/Default.md](.gitlab/merge_request_templates/Default.md).
The template is the source of truth for title format, compatibility sections,
testing evidence, and final checklist.

Before requesting review, complete the MR template and make sure testing
evidence or known gaps are documented.

Core changes must be reviewed carefully. CODEOWNERS is the source of truth for
required ownership, but high-risk changes should also get domain review.

## Architecture Boundaries

Use the most specific layer that owns the responsibility:

- `torchtitan/config/`: lightweight schema, config presets, CLI overrides,
  validation hooks. Must not import `torch`, models, or distributed code.
- `torchtitan/trainer.py`: train loop, trainer state, events, callbacks.
- `torchtitan/models/`: model definitions and model-local helpers.
- `torchtitan/distributed/`: distributed primitives and parallelism utilities.
- `torchtitan/components/`: reusable runtime components.
- `gift_docker/`: Dockerfile, image dependency pins, environment.
- `gift_scripts/`: ML Space job submission and cluster wrappers.
- `scripts/ci/`: CI helpers, including the SR008 integration job submitter.

Core code must not import `scripts/*`, `gift_scripts/*`, `gift_docker/*`, or
root-level `contrib/*` directly.

## Config, Registry, And Builders

New user-facing config fields require:

- typed dataclass field;
- lightweight validation when applicable;
- YAML example or preset when the field is intended for users;
- tests or an explicit reason why tests are not practical.

New model/data/optimizer/scheduler/callback types should go through typed config,
explicit registry registration, and builder functions. YAML uses stable `type`
names, not Python class paths.

Builders create runtime objects from typed configs. They must not read YAML,
apply CLI overrides, or start training. Heavy optional imports should happen
inside the builder/callback function that needs them.

## Contrib And Submodules

External repositories, vendored code, and git submodules live under root-level
`contrib/<name>`.

Integration adapters live under `torchtitan/contrib/<name>` and connect external
code to our configs, registries, builders, callbacks, and runtime.

Submodule updates must be intentional:

- update the pinned commit in a dedicated MR or dedicated commit;
- link the upstream commit/MR;
- explain why the update is needed;
- ensure lightweight config/preflight paths do not import the submodule.

## Testing Expectations

Run focused tests for the changed area and report exact commands in the MR.

Recommended minimums:

- config changes: config/unit test and YAML example;
- registry/builder changes: runtime smoke test when dependencies are available;
- trainer/callback changes: callback order and state save/load tests;
- checkpoint changes: save/load/resume test;
- distributed changes: GPU/integration test or documented manual run;
- model changes: checkpoint compatibility and numerics/loss note;
- Docker/submit changes: command example or dry-run when available.

If a scenario is unsupported, incompatible, or intentionally left untested, add
an explicit validation check, assertion, or user-facing error in the relevant
code path. Do not let unsupported combinations silently run.

## Numerics And Performance

Non-computation changes should preserve loss with fixed seeds and deterministic
settings. Computation changes should show convergence or an explicit reason why
numerics are expected to change.

Performance changes should report the relevant metric: memory, throughput, MFU,
profile traces, or multi-node results when applicable.

## Upstream TorchTitan Notes

We keep the useful parts of upstream TorchTitan guidance, but adapt them to this
fork-first repository:

- prefer PyTorch-native abstractions when they fit, but third-party production
  integrations are allowed when isolated behind clear boundaries;
- keep model changes minimal and document why they are needed;
- after model changes, verify checkpoint compatibility when applicable;
- keep shared training code modular instead of adding experiment-specific
  branches to core paths;
- justify non-trivial changes with tests, loss/numerics evidence, performance
  data, or an explicit explanation of residual risk.

## Documentation

Update documentation when changing:

- public config schema or YAML examples;
- CLI behavior;
- checkpoint format or resume behavior;
- model/runtime/trainer APIs;
- Docker images or job submission flow.

Known limitations should be documented in the MR and, when useful, tracked as
follow-up tasks.
