"""Shared submission logic for training experiments."""

from __future__ import annotations

import json
import shlex
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_REGISTRY = (REPO_ROOT / "sber_configs" / "baseline_registry").resolve()

JobFactory = Callable[..., Any]


@dataclass(frozen=True)
class Setup:
    """Overrides specific to one training job."""

    run_name: str
    config_overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.run_name, str) or not self.run_name.strip():
            raise ValueError("Setup.run_name must be a non-empty string")
        if "run_name" in self.config_overrides:
            raise ValueError("Setup.config_overrides must not contain run_name")


@dataclass(frozen=True)
class Experiment:
    """Configuration and infrastructure for all jobs in an experiment."""

    exp_name: str
    user_login: str
    baseline_config: Path
    baseline_patch: Path
    setups: list[Setup]
    n_workers: int = 1
    n_gpus: int = 8
    processes_per_worker: int = 8
    enable_auto_track: bool = True
    max_retry: int | None = None
    job_logdir: str | Path | None = None
    slurm_logdir: str | Path | None = None
    extra_env: Mapping[str, str] = field(default_factory=dict)
    queue_name: str | None = None
    priority_class: str | None = None
    region: str | None = None
    submission_type: str | None = None
    dry_run: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.exp_name, str) or not self.exp_name.strip():
            raise ValueError("Experiment.exp_name must be a non-empty string")
        if not isinstance(self.user_login, str) or not self.user_login.strip():
            raise ValueError("Experiment.user_login must be a non-empty string")
        if not self.setups:
            raise ValueError("Experiment must contain at least one setup")
        run_names = [setup.run_name for setup in self.setups]
        if len(run_names) != len(set(run_names)):
            raise ValueError("Experiment setup run_name values must be unique")

    def submit(self, *, job_factory: JobFactory | None = None) -> list[Any]:
        return submit_experiment(self, job_factory=job_factory)


def load_baseline_patch(path: str | Path) -> dict[str, Any]:
    """Load an optional YAML mapping; an empty patch file means no overrides."""
    from omegaconf import OmegaConf

    patch = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    return cast(dict[str, Any], patch or {})


def flatten_overrides(
    overrides: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    """Flatten nested config overrides into OmegaConf dotted paths."""
    flattened: dict[str, Any] = {}
    for raw_key, value in overrides.items():
        if not isinstance(raw_key, str) or not raw_key:
            raise ValueError(
                f"Override keys must be non-empty strings, got {raw_key!r}"
            )
        path = f"{prefix}.{raw_key}" if prefix else raw_key
        if isinstance(value, Mapping) and value:
            nested = flatten_overrides(value, prefix=path)
            duplicate_paths = flattened.keys() & nested.keys()
            if duplicate_paths:
                duplicate = sorted(duplicate_paths)[0]
                raise ValueError(f"Duplicate override path: {duplicate}")
            flattened.update(nested)
        else:
            if path in flattened:
                raise ValueError(f"Duplicate override path: {path}")
            flattened[path] = value
    return flattened


def build_script_args(*override_layers: Mapping[str, Any]) -> str:
    """Merge override layers and render shell-safe OmegaConf CLI arguments."""
    merged: dict[str, Any] = {}
    for layer in override_layers:
        merged.update(flatten_overrides(layer))

    arguments = []
    for path, value in merged.items():
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        arguments.append(shlex.quote(f"{path}={encoded}"))
    return " ".join(arguments)


def submit_experiment(
    experiment: Experiment,
    *,
    job_factory: JobFactory | None = None,
) -> list[Any]:
    """Submit one ``TrainJobLLMF`` per setup without materializing new YAMLs."""
    baseline = Path(experiment.baseline_config).resolve()
    patch = load_baseline_patch(experiment.baseline_patch)
    job_args = {
        "user_login": experiment.user_login,
        "exp_name": experiment.exp_name,
        "n_workers": experiment.n_workers,
        "n_gpus": experiment.n_gpus,
        "processes_per_worker": experiment.processes_per_worker,
        "enable_auto_track": experiment.enable_auto_track,
        "max_retry": experiment.max_retry,
        "job_logdir": (
            None if experiment.job_logdir is None else str(experiment.job_logdir)
        ),
        "slurm_logdir": (
            None if experiment.slurm_logdir is None else str(experiment.slurm_logdir)
        ),
        "extra_env": dict(experiment.extra_env),
        "queue_name": experiment.queue_name,
        "priority_class": experiment.priority_class,
        "region": experiment.region,
        "submission_type": experiment.submission_type,
        "dry_run": experiment.dry_run,
    }

    if job_factory is None:
        from contrib.job_submitter.job_submitter.jobs.llmf.train import TrainJobLLMF

        job_factory = TrainJobLLMF

    dry_run = experiment.dry_run
    action = "planning" if dry_run else "submitting"
    outcome = "planned" if dry_run else "submitted"
    total = len(experiment.setups)
    print(f"[experiment] {action} {total} job(s) from baseline {baseline}")

    results: list[Any] = []
    common_overrides = {
        "user": experiment.user_login,
        "exp_name": experiment.exp_name,
    }
    for index, setup in enumerate(experiment.setups, start=1):
        run_name = setup.run_name
        script_args = build_script_args(
            patch,
            {"run_name": setup.run_name, **setup.config_overrides},
            common_overrides,
        )
        print(f"[experiment] {action} {index}/{total}: {run_name}")
        try:
            job = job_factory(
                path_to_config=baseline,
                script_args=script_args,
                # job-level exp_name only feeds the slurm job name, so name the run there
                **{**job_args, "exp_name": f"{experiment.exp_name} | {run_name}"},
            )
            results.append(job.submit())
        except Exception as exc:
            print(
                f"[experiment] failed {index}/{total}: {run_name} "
                f"({type(exc).__name__})",
                file=sys.stderr,
            )
            raise
        print(f"[experiment] {outcome} {index}/{total}: {run_name}")
        if index < total:
            print()
    return results
