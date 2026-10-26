import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
import re
import shlex
import sys
import time

from omegaconf import OmegaConf

from scripts.train.src.config_loader import (
    prepare_train_config,
    register_relative_config_loader,
)
from sber_scripts.analytics.experimentation.submit import (
    build_script_args,
    load_baseline_patch,
)


def effective_configs(experiment):
    register_relative_config_loader(experiment.baseline_config.parent)
    patch = load_baseline_patch(experiment.baseline_patch)
    configs = []
    for setup in experiment.setups:
        arguments = build_script_args(
            patch,
            {"run_name": setup.run_name, **setup.config_overrides},
            {"user": experiment.user_login, "exp_name": experiment.exp_name},
        )
        config = OmegaConf.merge(
            OmegaConf.load(experiment.baseline_config),
            OmegaConf.from_cli(shlex.split(arguments)),
        )
        configs.append(
            OmegaConf.to_container(
                prepare_train_config(config),
                resolve=True,
                throw_on_missing=True,
            )
        )
    return configs


@contextmanager
def quiet_native_output():
    with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
        yield


def tracking_job(kwargs, config):
    config = deepcopy(config)
    with quiet_native_output():
        from contrib.job_submitter.job_submitter.jobs.llmf.train import TrainJobLLMF

        if not callable(getattr(TrainJobLLMF, "_tracking_merged_config", None)):
            raise ValueError(
                "Native submitter lacks the effective-config tracking hook"
            )

        class TrackingJob(TrainJobLLMF):
            def _tracking_merged_config(self, cfg):
                return deepcopy(config)

            def _resolve_tb_log_dir(self):
                from job_submitter.core.tracking import extract_logs_path

                return extract_logs_path(config)

            def submit(self):
                with quiet_native_output():
                    return super().submit()

        return TrackingJob(**kwargs)


def resume_checkpoint_paths(config):
    filename = config.get("save_latest_filename", "latest-rank{rank}.pt")
    if not filename:
        raise ValueError("--resume requires save_latest_filename")
    paths = [
        Path(config["save_folder"])
        / filename.format(rank=rank, run_name=config["run_name"])
        for rank in range(max(config.get("tp_size", 1), config.get("ep_size", 1)))
    ]
    resolved = [path.resolve() for path in paths]
    folders = {
        target.parent
        for path, target in zip(paths, resolved)
        if path.exists() or path.is_symlink()
    }
    folder = (
        Path(os.path.commonpath(folders))
        if folders
        else Path(config["save_folder"]).resolve()
    )
    print(f"[experiment] resume checkpoint for {config['run_name']}: {folder}")
    failure = f"Checkpoint checks failed for {config['run_name']}"
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{failure}: missing {', '.join(missing)}")
    if len(folders) != 1:
        raise ValueError(
            f"{failure}: EP/TP parts point to different checkpoint folders: "
            + ", ".join(str(folder) for folder in sorted(folders))
        )
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{failure}: multiple EP/TP parts point to the same path")
    print(
        f"[experiment] all checkpoint checks passed for {config['run_name']} "
        f"({len(resolved)}/{len(paths)} EP/TP parts found in the same checkpoint folder)"
    )
    return resolved


def submit_on_cluster(experiment, configs, *, resume=False):
    if sys.platform == "darwin" or not (
        Path("/mnt/nfs").is_dir() or Path("/home/jovyan").is_dir()
    ):
        raise ValueError("Submit from the authorized target cluster checkout")
    if [c["run_name"] for c in configs] != [s.run_name for s in experiment.setups]:
        raise ValueError("Prepared configs do not match the experiment setups")
    save_paths = [c["save_folder"] for c in configs]
    if len(set(save_paths)) != len(save_paths):
        raise ValueError("Run output paths are not unique")
    if resume and any(not c.get("autoresume", False) for c in configs):
        raise ValueError("--resume requires autoresume=true")
    if not resume and any(
        c.get("load_path") or Path(c["save_folder"]).exists() for c in configs
    ):
        raise ValueError("Existing checkpoint/output path; reconcile before submission")
    cache = experiment.extra_env.get("PROPAGATE_DG_CACHE_DIR")
    if cache and not Path(cache).is_dir():
        raise ValueError("Required kernel cache directory is unavailable")

    if resume:
        for config in configs:
            resume_checkpoint_paths(config)
    submitted = 0

    def job_factory(**kwargs):
        nonlocal submitted
        if submitted:
            time.sleep(1)
        config = configs[submitted]
        submitted += 1
        return tracking_job(kwargs, config)

    experiment.submit(job_factory=job_factory)


def _identity(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise argparse.ArgumentTypeError(
            "Use 1-80 ASCII letters, digits, dots, underscores or hyphens"
        )
    return value


def main(build_experiment):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--submit", action="store_true")
    parser.add_argument("--exp-name", type=_identity)
    parser.add_argument("--run-suffix", required=True, type=_identity)
    parser.add_argument("--user-login", type=_identity)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        original = build_experiment()
        experiment = replace(
            original,
            exp_name=args.exp_name or original.exp_name,
            user_login=args.user_login or original.user_login,
            setups=[
                replace(s, run_name=f"{s.run_name}-{args.run_suffix}")
                for s in original.setups
            ],
        )
        configs = effective_configs(experiment)
        print(f"Prepared {len(configs)} configurations")
        if args.submit:
            submit_on_cluster(experiment, configs, resume=args.resume)
    except Exception as exc:
        raise SystemExit(f"{type(exc).__name__}: {exc}") from None
