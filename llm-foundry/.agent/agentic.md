# Agent Operating Manual for LLM Foundry

This document defines the mandatory operating rules for an AI agent working in this repository. Treat these rules as hard constraints unless a human explicitly overrides them.

## Environment Boundaries

The agent operates across two environments:

1. `Local Environment`
   Used for code editing, file inspection, Git operations, and merge-request workflow.
2. `Remote GPU Environment`
   Used only for execution, validation, profiling, and log inspection when needed.

The agent must keep these roles separate. The remote environment is not the place to perform source-control work.

## The NFS Advantage

For work inside this repository, the local machine and the remote GPU job see the same files through a shared Network File System (NFS).

This is a critical constraint:

- A file saved locally is immediately visible from the remote GPU environment.
- This repository path is shared across local and remote execution.
- Git is not a deployment or sync mechanism for testing.

This guarantee should be assumed for the current repository root and other verified shared paths. Do not assume that arbitrary paths outside the workspace are on NFS unless that has been confirmed.

### DO

- Edit files locally and assume the remote cluster will see those edits immediately.
- Test the current local workspace remotely without pushing code.
- Prefer writing logs and temporary outputs under this repository or another known NFS-backed path when you want both local and remote visibility.
- Use Git for version control, collaboration, and review, not for remote file synchronization.

### DON'T

- Do not `git push`, `git pull`, or create temporary branches just to make code visible on the cluster.
- Do not copy the repository to a second remote location for testing.
- Do not assume `/tmp`, home-directory scratch locations, or arbitrary absolute paths are shared unless verified.
- Do not treat remote execution as a separate codebase.

## Remote Execution Rules

The agent does not have direct SSH access as part of its normal workflow.

To run code on the remote Cloud GPU cluster, the agent must use the local wrapper script `remote_run.py`.

### Mandatory Rule

Use `remote_run.py` for remote command execution. Do not invoke raw `ssh` commands directly.

`remote_run.py` starts the remote command from this repository root. If a task requires another directory, change into it inside the same quoted command.

### Exact Usage Examples

```bash
python remote_run.py "ls -la /workspace"
python remote_run.py "pwd && ls -la"
python remote_run.py "cd sber_tests && ls -la && pwd"
python remote_run.py "tail -n 200 train.log"
```

### Stateless Shell Constraint

Every call to `remote_run.py` launches a completely isolated, stateless shell.

This means:

- Directory changes do not persist across calls.
- Environment variable exports do not persist across calls.
- Shell state from one invocation is unavailable to the next invocation.

If a command depends on a directory change or environment setup, it must be chained in a single quoted command with `&&`.

### DO

- Chain dependent actions in one command, for example:

```bash
python remote_run.py "cd sber_scripts && ls -la && pwd"
```

- Assume each new invocation starts fresh.
- Keep remote commands explicit and self-contained.

### DON'T

- Do not expect `cd`, `export`, `source`, or shell variables from a previous call to still exist.
- Do not split a stateful workflow across multiple `remote_run.py` invocations unless each call re-establishes its own state.
- Do not bypass `remote_run.py` with ad hoc SSH usage.

## Long-Running and Pretraining Jobs

Never run multi-hour pretraining jobs in the foreground.

The required pattern is:

1. Redirect stdout and stderr to a log file.
2. Send the process to the background.
3. Monitor progress from whichever side can see the log path.

### Required Pattern

```bash
python remote_run.py "bash sber_scripts/infinite_jobs/run.sh <MODALITY> <DEFAULT_CFG> <EDITABLE_CFG> > train.log 2>&1 &"
python remote_run.py "tail -f train.log"
```

If `train.log` is written under this repository or another shared NFS path, it may also be inspected locally. If the log is written to a path that exists only inside the remote job environment, inspect it through `remote_run.py`.

### DO

- Always use log redirection such as `> train.log 2>&1`.
- Always background long-running jobs with `&`.
- Prefer log paths on shared NFS when you want to inspect them from either environment.
- Use `remote_run.py` to inspect logs when the chosen log path is only visible from the remote job.

### DON'T

- Do not leave long-running training attached to the foreground shell.
- Do not rely on an interactive remote session staying open.
- Do not start a training run without an explicit log file.

## Short Debug Launches

When the goal is only to validate startup, wiring, or cluster health, prefer the config-level training debug section instead of manually editing many unrelated fields.

Use `debug.use_debug: true` in the train YAML to apply a short-run profile that can disable benchmarks and evaluators, reduce the train dataset to one stream, skip checkpoint loading, and merge a small `debug.overrides` block for fast startup validation.

Supported shape:

```yaml
debug:
  use_debug: true
  startup_speedups:
    use_one_train_dataloader: true      # keep only one stream from train_loader.dataset.streams
    train_stream_name: null             # optional explicit stream name, otherwise first stream is used
    disable_runtime_benchmark_callbacks: true
    disable_eval_loader: true
    icl_tasks_metrics: null             # null disables all ICL tasks; list filters by task label
    disable_all_evaluators: false       # force-disable eval_loader, eval_loaders, icl_tasks, eval_gauntlet
    load_checkpoints: false             # drops load_path / teacher_load_path and disables autoresume
    fast_model_init: true               # switches non-pretrained sharded models to init_device=meta
  overrides:
    max_duration: 2ba
    save_interval: 0
    console_log_interval: 1ba
    train_loader:
      num_workers: 0
      dataset:
        use_dummy: true
        epoch_size: 32
```

Behavior notes:

- `debug.overrides` is merged after debug speedups.
- Top-level `overrides` is still applied after `debug.overrides`, so normal explicit overrides remain the final authority.
- If debug speedups remove every evaluator, training is forced to `eval_interval: 0` and `eval_before_train: skip`.
- For base configs that already define `train_loader.dataset.epoch_size`, prefer overriding `epoch_size` for dummy smoke runs instead of adding `dataset_size`.

## Multi-GPU and Multi-Node Launching

For real remote multi-GPU or multi-node execution, start from the dedicated launcher scripts under `sber_scripts/infinite_jobs`.

The canonical distributed launcher is:

- `sber_scripts/infinite_jobs/run.sh`

Operational support scripts include:

- `sber_scripts/infinite_jobs/clean.sh`
- `sber_scripts/infinite_jobs/local_run.sh`

### Mandatory Rule

`remote_run.py` gives the agent a shell on the active `mpimaster` node. It does not by itself coordinate worker nodes, hostfiles, or cross-node launch state.

The repository's distributed orchestration already lives in `sber_scripts/infinite_jobs/run.sh`. That script is the reference example for how this environment prepares the host list and launches work across the allocated nodes.

The agent should read `sber_scripts/infinite_jobs` as the canonical pattern for distributed launch in this repository. If the standard script is close to the required workflow, reuse it directly. If a custom launch is needed, modify it or create a similar repo-managed launcher script based on that pattern.

Do not hide real distributed orchestration inside a giant one-off `remote_run.py` command when it should live in a reusable script.

### DO

- Use `sber_scripts/infinite_jobs/run.sh` for distributed training launches.
- Use `remote_run.py` to invoke `sber_scripts/infinite_jobs/run.sh` or another repo-managed launcher script from the master node.
- Treat `sber_scripts/infinite_jobs/run.sh` as the base example for MPI, hostfile, rank, and environment setup in this cluster.
- When special launch behavior is required, modify the existing launcher or create a similar script in the repository instead of packing everything into one long shell string.
- Use `sber_scripts/infinite_jobs/clean.sh` after crashes or aborted runs to clear stale processes and shared memory.
- Keep distributed launch behavior aligned with the repository's existing launcher scripts.

### DON'T

- Do not invent a completely new distributed-launch workflow when `sber_scripts/infinite_jobs` already provides the needed pattern.
- Do not bury complex MPI, rank, or hostfile logic inside an ad hoc inline command if that logic should be reusable.
- Do not debug launcher issues by permanently replacing the standard launcher with one-off commands.

## Strict Git Workflow: Local vs Remote

The agent must separate version-control work from execution work.

### Local Environment: Code and Version Control

All of the following must happen locally:

- File editing
- File creation and deletion
- `git checkout -b`
- `git status`
- `git diff`
- `git add`
- `git commit`
- `git push`
- GitLab Merge Request creation

### Remote Environment: Execution and Testing Only

The remote environment is for:

- Running scripts
- Launching training
- Running tests
- Collecting logs that are not already visible through shared NFS
- Checking runtime behavior
- Measuring performance

### Mandatory Loop

The expected workflow is:

`Edit locally -> Test remotely -> If test fails, fix locally -> If test passes, commit locally -> Push locally -> Create MR locally`

### DO

- Treat the local working tree as the source of truth.
- Make all code changes locally, even when a failure is discovered from a remote run.
- Use remote runs to validate code, not to author code.

### DON'T

- Do not commit from the remote environment.
- Do not push from the remote environment.
- Do not create branches or Merge Requests from the remote environment.
- Do not perform "quick fixes" directly in a remote shell.

## Practical Operating Model

When working on a change, the agent should behave as follows:

1. Inspect the codebase locally.
2. Create or switch to the correct local branch.
3. Edit files locally.
4. Save changes locally.
5. Run remote validation with `remote_run.py`.
6. If the run fails, inspect logs from the side that can see them, then return to local editing.
7. Repeat until the remote test passes.
8. Commit locally.
9. Push locally.
10. Create the GitLab Merge Request locally.

## Decision Rules

When deciding where an action belongs, use this test:

- If the action changes source code or Git history, it belongs locally.
- If the action executes code on the cluster, it belongs remotely through `remote_run.py`.
- If the action reads runtime logs, it belongs wherever the log path is visible. Logs under this repository or another shared NFS path may be read locally or remotely.

If a task seems to require remote file sync, that assumption is wrong unless a human says otherwise. The shared NFS already provides synchronization.

## Library Explanation

This repository is organized into a few major zones. Use this map to find the right place to inspect or modify behavior.

### Core Library

- `llmfoundry/`
  The main Python package. This is where model code, data pipelines, callbacks, optimization logic, schedulers, and utility code live.
- `llmfoundry/models/`
  Model implementations and model-specific components.
- `llmfoundry/data/`
  Dataset builders, modality-specific loaders, and training/evaluation data plumbing.
- `llmfoundry/callbacks/`
  Training callbacks and runtime hooks.
- `llmfoundry/optim/`
  Optimizers and optimizer-related logic.
- `llmfoundry/scheduler/`
  Learning-rate scheduling and related training schedule code.
- `llmfoundry/utils/`
  Shared builders, config helpers, artifact helpers, and general utilities.

### Training and Execution Entrypoints

- `scripts/train/train.py`
  The main training entrypoint used by the repository.
- `scripts/data_prep/`
  Data preparation and dataset conversion scripts.
- `scripts/eval/`
  Evaluation scripts and evaluation helpers.
- `scripts/inference/`
  Inference-time scripts and utilities.

### Configurations

- `sber_configs/`
  Training and experiment configuration files, including modality-specific configurations.

### Cluster and Launch Tooling

- `remote_run.py`
  The required local wrapper for executing commands on the remote cluster.
- `sber_scripts/`
  Sber-specific operational scripts for job submission, launch, release, and support workflows.
- `sber_scripts/infinite_jobs/`
  The reference distributed-launch area for multi-GPU and multi-node jobs.

### Tests

- `sber_tests/`
  Sber-specific integration, utility, and regression tests.
- `tests/`
  General repository tests.

### Dependencies and Supporting Code

- `contrib/`
  In-repo dependency checkouts and companion libraries used by this framework.
- `c_lib/`
  Native or low-level components that support performance-critical behavior.

### Additional Project Areas

- `sber_benchmarks/`
  Benchmark-related code and assets.
- `sber_docker/`
  Docker and environment build assets.
- `docs/`
  Project documentation.
- `sber_notebooks/`
  Notebooks for analysis, exploration, and experiments.

## Final Non-Negotiable Rules

- Local edits in this repository are instantly visible remotely because the repository is on shared NFS.
- Remote execution must go through `remote_run.py`.
- Every `remote_run.py` call is stateless and must be self-contained.
- `remote_run.py` connects to the active master node; distributed worker orchestration should follow the `sber_scripts/infinite_jobs` pattern.
- Long-running training must be backgrounded and logged.
- Multi-GPU and multi-node launches should reuse, modify, or be derived from the `sber_scripts/infinite_jobs` launcher pattern.
- Git work happens locally. Remote work is for execution only.
