# Operating rules

These rules make this clone usable on its own. The task record and current user authorization define the permitted scope.

## Current prioritization

- Target GFSDP, compilation of both model and loss, FP8, and SAC/full activation checkpointing. Keep `gift 4` as supplied until its exact model/preset meaning is clarified.
- Focus initial optimization research on small models on one or two nodes; exact hardware and configuration remain to be selected.
- Deprioritize PP for now. It is not excluded or a blocker.
- Improve training speed without trading away numerical stability or model quality. Preserve the selected production settings when comparing candidates.

See the [task boundary](docs/task-boundary.md) for the original brief and unresolved details.

## Objective and invariants

Improve representative end-to-end training throughput while preserving the training objective, data, precision, distributed behavior, checkpoint continuation, and agreed model quality.

Before implementation, record the production workload and resolved configuration: model and preset, data identity and split, sequence and packing behavior, optimizer and schedule, precision, activation checkpointing, compilation, parallel groups and physical topology, evaluation and checkpoint cadence, and quality thresholds.

Preserve target shifting, masks, packed-document boundaries, attention context, convolution and scan resets, loss weights and denominators, and valid-token counts. Preserve loader position and RNG ownership, validation state, routing side effects, optimizer state, and checkpoint semantics. Distinguish exact continuation from weights-only initialization.

Use FP8 in the intended target baseline after its recipe is resolved. Preserve the selected precision recipe, activation checkpointing, compilation, monitoring, and required checkpoint behavior in optimization candidates. Changes to the FP8 recipe or objective need explicit agreement, an updated contract, and appropriate quality validation.

For computation-preserving changes, compare loss and elementwise gradients before clipping, then the clipping norm and scale, clipped gradients, optimizer moments, and parameter updates. Equal norms do not establish equal gradients. Separate exact structural invariants from floating-point values with agreed tolerances and stated nondeterminism. Compare eager and compiled paths where both are supported.

## Source manuals and workspace guides

Before an authorized GIFT source change, read [gift/AGENTS.md](gift/AGENTS.md), [gift/CONTRIBUTING.md](gift/CONTRIBUTING.md), [gift/REVIEW.md](gift/REVIEW.md), and the applicable scoped rules: [config](gift/.claude/rules/config.md), [distributed](gift/.claude/rules/distributed.md), [models](gift/.claude/rules/models.md), [integer overflow](gift/.claude/rules/integer_overflow.md), and [experiments](gift/.claude/rules/experiments.md). Before an authorized Foundry source change, read [llm-foundry/.agent/agentic.md](llm-foundry/.agent/agentic.md). These manuals govern their source trees. Verify environment-specific paths and launch assumptions before applying them; their examples do not authorize source edits, package installation, remote access, or compute.

Optional workspace-only skill guides live outside a standalone clone. In the original workspace, consult `../.agents/skills/code-rules/SKILL.md`, `../.agents/skills/perf-experiments/SKILL.md`, and `../.agents/skills/git-gitlab/SKILL.md` for source changes, performance review, and repository operations. For authorized execution, the relevant guides are `../.agents/skills/llmf-train-job/SKILL.md`, `../.agents/skills/titan-train-job/SKILL.md`, and `../.agents/skills/mlspace-gpu-job/SKILL.md`. For trace or bounded no-GPU analysis, use `../.agents/skills/trace-analysis/SKILL.md`, `../.agents/skills/trace-timeline/SKILL.md`, `../.agents/skills/comm-interactions/SKILL.md`, `../.agents/skills/pp-trace/SKILL.md`, and `../.agents/skills/no-gpu-checks/SKILL.md`. These paths are workspace-only routing hints, not dependencies of this clone or authorization for any action. The rules in this file remain available when those guides are absent.

## Source, compute, and Git boundaries

- `gift/` and `llm-foundry/` are flattened source snapshots, not nested repositories or submodules. Run Git operations from this repository root. Preserve upstream metadata; do not initialize submodules or add remotes inside snapshots.
- A source import or baseline pin does not authorize later source edits. Change training source only after explicit authorization naming the target tree and scope.
- Do not install packages, fetch data or models, allocate compute, launch training, benchmark, or mutate shared resources without authorization for that action.
- Keep credentials, private keys, `.env` contents, private host details, and access instructions out of tracked files. Use non-secret identifiers for hardware and restricted data.
- Preserve other contributors' edits. Do not commit or publish unless the active user instruction authorizes that action.

## Evidence and performance claims

Verify the actual rank-zero resolved configuration and prove each relevant feature is active. Record process-group membership and physical topology, not just mesh degrees.

Separate startup, compilation, autotuning, warmup, stable steps, validation, checkpointing, and shutdown. Estimate noise with an identical A/A pair; alternate baseline and candidate in unprofiled runs; compare stable-step series. Use traces to explain mechanisms, not to estimate speed. Measure paired memory at the same lifecycle window.

Report host submission, device work, waits, copies, communication, and integrated elapsed time with their timer definitions. These intervals can overlap; do not sum overlapping kernel or communication durations as elapsed time. Include job lifecycle costs at their real cadence. Report uncertainty and all exclusions. Do not claim a speedup from one run, one trace, an isolated kernel, or a component-only fixture.

Keep raw large traces, checkpoints, datasets, and profiles outside Git. Track concise, reviewed reports with source pins, configuration, commands, seed, data identity, topology, activation proof, step series, memory, noise, conclusions, and open gates. Never omit a failed or skipped check.

## Task and document hygiene

Use the Markdown task files directly. Do not add tracker configuration or install a task tool for this scaffold. Keep each task's scope, acceptance, dependencies, and next action in its record. Keep technical conclusions in `docs/`, reusable process in `planning/`, source provenance in `additional/`, and campaign evidence in `runs/`.

## Bounded work and review

Before acting, define a finite acceptance checklist from the task. Do not add requirements without an explicit request, affected contract, or concrete failure. Before an additional experiment, test, or review, name the unresolved question, possible outcomes, and decision each outcome would change; skip work that cannot change the next action or verdict.

For an authorized training change, review runtime cost before coding and against the final implementation. Cover the affected inherited path, computation, memory, host/device synchronization, transfers, communication, compilation, lifecycle frequency, and scale. Record expected costs, evidence, and unresolved risks. Documentation-only work may state why it has no production-runtime impact. Review the scoped result once, fix material findings, and verify the affected behavior; do not expand into an unrelated audit or omit required gates.

After a documentation change, check local links and whitespace. For a training change, inspect its full affected path and perform the scoped correctness and runtime-cost review before handoff.
