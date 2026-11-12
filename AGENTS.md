# Repository operating rules

## Read and maintain

Read [current state](docs/status.md), [baseline](docs/baseline.md) and the active
item in [planning](planning/README.md). The baseline is selected and its basic
smoke passed; performance and candidate acceptance are separate gates.

Each fact has one owner: defaults in `docs/baseline.md`, integrated results and
unknowns in `docs/status.md`, mechanisms/source pointers in `docs/candidates.md`,
graph conclusions in `docs/cuda-graphs.md`, future work in `planning/README.md`,
methods in `planning/validation.md`, run evidence in `runs/`. Update the owner
and link to it instead of creating duplicate status documents or handoff chains.
Historical full records are recoverable through [references](references/README.md).

## Source and authorization

- `gift/` is a flattened source snapshot, not a nested repository or submodule.
  Run Git at this root; do not initialize repositories or refresh submodules
  inside it. `references/` contains inert upstream source, not active imports.
- Training-source writes require authorization for the named tree and scope.
  Preserve the agreed config; do not disable FP8, compile, SAC, monitoring,
  MTP or change dimensions to make a check pass without an explicit decision.
- Package installation, data/model downloads, GPU allocation/training and shared
  environment changes require authorization. Never kill unrelated processes or
  modify shared packages/startup configuration as routine cleanup.
- Commit, push and other publication require authorization; reuse existing scope.
  Preserve contributor edits and use repository-local user identity. No assistant
  attribution. Do not rewrite published history as routine cleanup.
- Keep credentials, keys and `.env` contents out of outputs and tracked files.
  Record reproducible runtime paths without exposing access secrets.
- Before GIFT source changes, read [GIFT rules](gift/AGENTS.md),
  [contributing](gift/CONTRIBUTING.md), [review](gift/REVIEW.md) and applicable
  [scoped rules](gift/.claude/rules/). Preserve precision, safe integer widths,
  autograd, communication, checkpoint and evaluation contracts.

Workspace-only skills are outside a standalone clone: `../.agents/skills/`
contains `code-rules`, `perf-experiments`, `git-gitlab`, `titan-train-job`,
`mlspace-gpu-job` and trace-analysis tools. Read the relevant skill before that
work where available. Their examples do not grant authorization or establish a
working remote environment. Verify current paths, all-node ownership and launch
cleanup behavior before GPU work.

## Correctness and evidence

Follow [validation](planning/validation.md). Preserve data ordering, shifting,
masks and packed boundaries, attention/scan state, loss weights/denominators,
routing, precision, optimizer, schedule and accumulation. Preserve loader/RNG,
validation and persisted state; distinguish exact continuation from weights-only
loading. Any deliberate computation change needs an agreed quality criterion.

Compare elementwise loss/gradients and relevant state for numerical claims;
rounded norms are insufficient. Use matched inputs and stated tolerances. Verify
actual resolved configuration and activation, not only requested settings.
CPU checks do not establish GPU/NCCL behavior; a smoke does not establish speed,
convergence, long-run stability or continuation. Retain failed/skipped checks.

Review affected correctness and runtime costs before a training change and once
against its final implementation: computation, memory, synchronization, copies,
communication, compilation, lifetime and scale, including inherited active code.
Run relevant tests and integration checks. Documentation-only changes have no
runtime effect when source/config/runner bytes remain unchanged.

For performance use A/A noise and alternating unprofiled A/B after warmup; pair
memory by lifecycle phase. Use traces for mechanism, not speed magnitude. Keep
host/device/communication scopes explicit; overlapping sums are not elapsed time.
Include recurring input, metrics, evaluation and checkpoint costs at real cadence.
Never promote an incomplete or failed control comparison.

## Bounded work and artifacts

Define finite acceptance before execution. Reuse completed evidence. Repeat a
check only after a relevant change, inconclusive result or contradictory evidence.
Before extra work identify the unresolved question and decision it changes.
Keep one report/result per experiment and a clear next action. No benchmark or
optimization campaign is authorized by a planning entry alone.

Store large raw traces, logs, checkpoints and caches outside tracked files;
retain precise locations/hashes and compact supporting evidence. Source snapshots
must retain provenance and license; do not silently edit imported reference bytes.
Check maintained local links, JSON and checksums after document/source moves.
Do not repair historical upstream documentation links by modifying preserved code.
